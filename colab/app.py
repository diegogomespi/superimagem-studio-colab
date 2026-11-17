"""Backend CUDA do SuperImagem para Google Colab.

Mantém o contrato HTTP usado pela interface React. Os arquivos ficam somente
no diretório temporário da sessão do Colab.
"""
from __future__ import annotations

import asyncio
import gc
import hashlib
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from colab.operations import black_white, clean_image, remove_background, vectorize

# BasicSR antigo importa um módulo renomeado nas versões novas do torchvision.
from spandrel import ModelLoader

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(tempfile.mkdtemp(prefix='superimagem-t4-'))
ARTIFACTS_DIR = DATA / 'artifacts'
MODELS_DIR = ROOT / 'colab' / 'models'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    'general': {
        'name': 'RealESRGAN_x4plus',
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'file': 'RealESRGAN_x4plus.pth',
        'sha256': '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1',
        'blocks': 23,
    },
    'illustration': {
        'name': 'RealESRGAN_x4plus_anime_6B',
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        'file': 'RealESRGAN_x4plus_anime_6B.pth',
        'sha256': 'f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da',
        'blocks': 6,
    },
    'vector-refine': {
        'name': 'SwinIR-M_classical_DF2K_x4',
        'url': 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth',
        'file': '001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth',
        'sha256': '4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215',
        # Carregado somente quando escolhido para não atrasar a abertura normal.
        'preload': False,
    },
}

BUILD_VERSION = os.environ.get('SUPERIMAGEM_BUILD_VERSION', 'desenvolvimento')
app = FastAPI(title='SuperImagem GPU T4', version=BUILD_VERSION)
artifacts: dict[str, dict[str, Any]] = {}
jobs: dict[str, dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='superimagem-gpu')
active_lock = threading.Lock()
models: dict[str, Any] = {}


@app.middleware('http')
async def prevent_stale_interface(request, call_next):
    response = await call_next(request)
    if request.url.path in ('/', '/index.html') or request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
    return response


class JobRequest(BaseModel):
    artifactId: str
    operation: str = 'upscale'
    options: dict[str, Any] = {}


def new_id(prefix: str) -> str:
    return f'{prefix}_{uuid.uuid4().hex}'


def artifact_public(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key not in ('path', 'previewPath', 'inspectPath', 'webpPath')}


def version_identity(source: dict[str, Any]) -> tuple[str, int]:
    root_id = source.get('rootId', source['id'])
    number = max((int(item.get('versionNumber', 1)) for item in artifacts.values() if item.get('rootId', item['id']) == root_id), default=1) + 1
    return root_id, number


def processed_filename(original: str, scale: int, model_kind: str, detail_mode: str, extension: str = 'png') -> str:
    stem = Path(original).stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', stem).strip(' .')[:140] or 'imagem'
    model_tag = {
        'general': 'RealESRGAN-x4plus',
        'illustration': 'RealESRGAN-anime-6B',
        'vector-refine': 'SwinIR-classical-x4',
    }.get(model_kind, 'IA-x4')
    mode_tag = f"_{'detalhado' if detail_mode == 'detailed' else 'fiel'}" if scale == 8 else ''
    return f'{stem}_upscale-{scale}x_{model_tag}{mode_tag}.{extension}'


def operation_filename(original: str, operation: str, model_id: str, extension: str = 'png') -> str:
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', Path(original).stem).strip(' .')[:120] or 'imagem'
    tags = {
        'remove-background': 'sem-fundo',
        'clean': 'limpa',
        'black-white': 'preto-e-branco',
        'vectorize': 'vetorizada',
    }
    safe_model = re.sub(r'[^A-Za-z0-9._-]+', '-', model_id).strip('-')[:48]
    return f"{stem}_{tags.get(operation, operation)}_{safe_model}.{extension}"


def create_preview(image: np.ndarray, target: Path, max_side: int = 1024) -> Path:
    height, width = image.shape[:2]
    ratio = min(1.0, max_side / max(width, height))
    preview = image if ratio == 1.0 else cv2.resize(image, (max(1, round(width * ratio)), max(1, round(height * ratio))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode('.png', preview, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError('Falha ao gerar preview leve.')
    encoded.tofile(target)
    return target


def create_inspection_preview(image: np.ndarray, target: Path, max_side: int = 4096) -> Path:
    height, width = image.shape[:2]
    ratio = min(1.0, max_side / max(width, height))
    preview = image if ratio == 1.0 else cv2.resize(image, (max(1, round(width * ratio)), max(1, round(height * ratio))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode('.webp', preview, [cv2.IMWRITE_WEBP_QUALITY, 92])
    if not ok:
        raise RuntimeError('Falha ao gerar preview de inspeção.')
    encoded.tofile(target)
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected_sha256: str) -> None:
    if target.exists() and target.stat().st_size > 1_000_000:
        if sha256_file(target) == expected_sha256:
            return
        target.unlink(missing_ok=True)
    import urllib.request
    partial = target.with_suffix(target.suffix + '.part')
    partial.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'SuperImagem-Colab/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output)
            if partial.stat().st_size <= 1_000_000:
                raise RuntimeError('O arquivo recebido está incompleto.')
            actual_sha256 = sha256_file(partial)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f'Assinatura SHA-256 inválida: {actual_sha256}')
            partial.replace(target)
            return
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(attempt * 2)
    raise RuntimeError(f'Não foi possível baixar o modelo oficial após 3 tentativas: {last_error}')


def get_model(kind: str, job: dict[str, Any]) -> Any:
    if kind not in MODEL_SPECS:
        raise ValueError('Tipo de imagem inválido.')
    spec = MODEL_SPECS[kind]
    model_path = MODELS_DIR / spec['file']
    job['phase'] = 'loading_model'
    if kind not in models:
        download(spec['url'], model_path, spec['sha256'])
        descriptor = ModelLoader().load_from_file(str(model_path)).eval().cuda().half()
        if int(descriptor.scale) != 4:
            raise RuntimeError(f'Escala inesperada do modelo: {descriptor.scale}.')
        models[kind] = descriptor
    return models[kind]


def tile_count(width: int, height: int, tile: int = 400) -> int:
    return math.ceil(width / tile) * math.ceil(height / tile)


def extend_transparent_colors(bgra: np.ndarray) -> np.ndarray:
    """Estende as cores da borda para o vazio e evita halo preto/branco."""
    bgr = bgra[:, :, :3].copy()
    known = (bgra[:, :, 3] > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(10):
        expanded = cv2.dilate(known, kernel, iterations=1)
        frontier = (expanded > 0) & (known == 0)
        if not frontier.any():
            break
        for channel in range(3):
            spread = cv2.dilate(bgr[:, :, channel], kernel, iterations=1)
            bgr[:, :, channel][frontier] = spread[frontier]
        known = expanded
    return bgr


def upscale_rgb_tiled(bgr: np.ndarray, model: Any, job: dict[str, Any], completed: list[int], total: int, tile: int = 400) -> np.ndarray:
    height, width = bgr.shape[:2]
    native = 4
    output = np.empty((height * native, width * native, 3), dtype=np.uint8)
    pad = 24
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if job.get('cancelRequested'):
                raise InterruptedError('Processamento cancelado.')
            core_w, core_h = min(tile, width - x), min(tile, height - y)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(width, x + core_w + pad), min(height, y + core_h + pad)
            patch = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).unsqueeze(0).cuda().half().div_(255.0)
            with torch.inference_mode():
                prediction = model(tensor).clamp_(0, 1)
            result = prediction[0].float().cpu().numpy().transpose(1, 2, 0)
            result = cv2.cvtColor(np.rint(result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            crop_x, crop_y = (x - x0) * native, (y - y0) * native
            output[y * native:(y + core_h) * native, x * native:(x + core_w) * native] = result[crop_y:crop_y + core_h * native, crop_x:crop_x + core_w * native]
            completed[0] += 1
            job['progress']['completedTiles'] = min(completed[0], total)
            del tensor, prediction
    return output


def upscale_with_fallback(bgr: np.ndarray, model: Any, job: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Reduz automaticamente o bloco se a GPU não tiver memória suficiente."""
    height, width = bgr.shape[:2]
    last_error: Exception | None = None
    for tile in (400, 320, 256, 192):
        completed = [0]
        total = tile_count(width, height, tile)
        job['progress'] = {'completedTiles': 0, 'totalTiles': total, 'tileSize': tile}
        try:
            return upscale_rgb_tiled(bgr, model, job, completed, total, tile), total
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            job['phase'] = 'reducing_tile'
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(f'A GPU ficou sem memória mesmo usando blocos de 192 px: {last_error}')


def process_job(job_id: str) -> None:
    job = jobs[job_id]
    started = time.perf_counter()
    peak = psutil.Process().memory_info().rss
    try:
        with active_lock:
            if job.get('cancelRequested'):
                raise InterruptedError('Processamento cancelado.')
            source = artifacts[job['artifactId']]
            options = job['options']
            job.update(status='preparing', phase='preparing')
            operation = job['operation']
            total = 1
            if operation == 'vectorize':
                result_id = new_id('svg')
                output_path = ARTIFACTS_DIR / f'{result_id}.svg'
                model_id = vectorize(Path(source['path']), output_path, options, job)
                filename = operation_filename(source['filename'], operation, model_id, 'svg')
                root_id, version_number = version_identity(source)
                result = {'id': result_id, 'filename': filename, 'mime': 'image/svg+xml', 'width': source['width'], 'height': source['height'], 'hasAlpha': True, 'path': str(output_path), 'previewPath': str(output_path), 'inspectPath': str(output_path), 'webpPath': None, 'parentId': source['id'], 'rootId': root_id, 'versionNumber': version_number, 'operation': operation, 'engine': model_id, 'options': options}
                artifacts[result_id] = result
                job.update(status='done', phase='done', resultArtifactId=result_id, progress={'completedTiles': 1, 'totalTiles': 1}, metrics={'durationMs': round((time.perf_counter() - started) * 1000), 'peakRssBytes': max(peak, psutil.Process().memory_info().rss), 'modelId': model_id})
                return

            raw = np.fromfile(source['path'], dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError('Não foi possível abrir a imagem raster. Para continuar após um SVG, volte para uma versão PNG.')
            has_alpha = image.ndim == 3 and image.shape[2] == 4
            height, width = image.shape[:2]
            if operation == 'upscale':
                scale = int(options.get('scale', 4))
                if scale not in (2, 4, 8):
                    raise ValueError('Use ampliação 2×, 4× ou 8×.')
                detailed = scale == 8 and options.get('detailMode') == 'detailed'
                model = get_model(options.get('upscaleModel', 'general'), job)
                job.update(status='processing', phase='inference')
                source_bgr = extend_transparent_colors(image) if has_alpha else image[:, :, :3]
                first4, total = upscale_with_fallback(source_bgr, model, job)
                if detailed:
                    job['phase'] = 'second_pass'
                    second4, total = upscale_with_fallback(first4, model, job)
                    target_size = (width * 8, height * 8)
                    rgb = cv2.resize(second4, target_size, interpolation=cv2.INTER_LANCZOS4)
                    del second4
                else:
                    target_size = (width * scale, height * scale)
                    rgb = first4 if scale == 4 else cv2.resize(first4, target_size, interpolation=cv2.INTER_LANCZOS4)
                del first4
                if has_alpha:
                    job['phase'] = 'alpha'
                    alpha = cv2.resize(image[:, :, 3], target_size, interpolation=cv2.INTER_LANCZOS4)
                    output = np.dstack((rgb, alpha))
                else:
                    output = rgb
                model_id = MODEL_SPECS[options.get('upscaleModel', 'general')]['name']
                filename = processed_filename(source['filename'], scale, options.get('upscaleModel', 'general'), options.get('detailMode', 'faithful'))
            elif operation == 'remove-background':
                output, model_id = remove_background(image, options, job)
                filename = operation_filename(source['filename'], operation, model_id)
            elif operation == 'clean':
                output, model_id = clean_image(image, options, job)
                filename = operation_filename(source['filename'], operation, model_id)
            elif operation == 'black-white':
                output, model_id = black_white(image, options, job)
                filename = operation_filename(source['filename'], operation, model_id)
            else:
                raise ValueError('Operação inválida.')

            job['phase'] = 'encoding'
            result_id = new_id('img')
            output_path = ARTIFACTS_DIR / f'{result_id}.png'
            ok, encoded = cv2.imencode('.png', output, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                raise RuntimeError('Falha ao gerar PNG.')
            encoded.tofile(output_path)
            preview_path = ARTIFACTS_DIR / f'{result_id}-preview.png'
            create_preview(output, preview_path)
            inspect_path = ARTIFACTS_DIR / f'{result_id}-inspect.webp'
            create_inspection_preview(output, inspect_path)
            out_height, out_width = output.shape[:2]
            root_id, version_number = version_identity(source)
            result = {'id': result_id, 'filename': filename, 'mime': 'image/png', 'width': out_width, 'height': out_height, 'hasAlpha': output.ndim == 3 and output.shape[2] == 4, 'path': str(output_path), 'previewPath': str(preview_path), 'inspectPath': str(inspect_path), 'webpPath': None, 'parentId': source['id'], 'rootId': root_id, 'versionNumber': version_number, 'operation': operation, 'engine': model_id, 'options': options}
            artifacts[result_id] = result
            job.update(status='done', phase='done', resultArtifactId=result_id, progress={'completedTiles': total, 'totalTiles': total}, metrics={'durationMs': round((time.perf_counter() - started) * 1000), 'peakRssBytes': max(peak, psutil.Process().memory_info().rss), 'modelId': model_id})
    except InterruptedError as exc:
        job.update(status='cancelled', phase='cancelled', error={'code': 'CANCELLED', 'message': str(exc)})
    except Exception as exc:
        if job.get('cancelRequested'):
            job.update(status='cancelled', phase='cancelled', error={'code': 'CANCELLED', 'message': 'Processamento cancelado.'})
        else:
            job.update(status='error', phase='error', error={'code': 'PROCESSING_ERROR', 'message': str(exc)})


@app.get('/api/health')
def health() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {'name': props.name, 'memoryBytes': props.total_memory, 'cuda': torch.version.cuda}
    return {'name': 'SuperImagem', 'version': BUILD_VERSION, 'engineAvailable': torch.cuda.is_available(), 'engine': 'cuda' if torch.cuda.is_available() else 'cpu', 'gpu': gpu, 'modelsReady': sorted(models.keys()), 'workerBusy': any(j['status'] in ('queued', 'preparing', 'processing') for j in jobs.values())}


@app.post('/api/models/prepare')
def prepare_models() -> dict[str, Any]:
    """Baixa, valida e carrega todos os modelos antes de liberar a interface."""
    if not torch.cuda.is_available():
        raise HTTPException(503, 'GPU CUDA não encontrada.')
    ready = []
    for kind, spec in MODEL_SPECS.items():
        if spec.get('preload') is False:
            continue
        state: dict[str, Any] = {}
        descriptor = get_model(kind, state)
        model_path = MODELS_DIR / spec['file']
        ready.append({
            'kind': kind,
            'name': spec['name'],
            'bytes': model_path.stat().st_size,
            'sha256': spec['sha256'],
            'scale': int(descriptor.scale),
            'status': 'validated',
        })
    return {'ok': True, 'models': ready}


@app.post('/api/artifacts')
async def upload_artifact(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type not in ('image/png', 'image/jpeg', 'image/webp'):
        raise HTTPException(415, 'Use PNG, JPEG ou WebP.')
    data = await file.read()
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(413, 'O arquivo excede 100 MB.')
    item_id = new_id('img')
    suffix = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp'}[file.content_type]
    path = ARTIFACTS_DIR / f'{item_id}{suffix}'
    path.write_bytes(data)
    try:
        with Image.open(path) as image:
            width, height = image.size
            has_alpha = image.mode in ('RGBA', 'LA') or 'transparency' in image.info
    except Exception:
        path.unlink(missing_ok=True)
        raise HTTPException(400, 'Imagem inválida.')
    item = {'id': item_id, 'filename': file.filename or f'imagem{suffix}', 'mime': file.content_type, 'width': width, 'height': height, 'hasAlpha': has_alpha, 'path': str(path), 'rootId': item_id, 'versionNumber': 1}
    artifacts[item_id] = item
    return artifact_public(item)


@app.get('/api/artifacts/{artifact_id}')
def get_artifact(artifact_id: str) -> dict[str, Any]:
    item = artifacts.get(artifact_id)
    if not item:
        raise HTTPException(404, 'Imagem não encontrada.')
    return artifact_public(item)


@app.delete('/api/artifacts/{artifact_id}')
def delete_artifact(artifact_id: str) -> dict[str, Any]:
    item = artifacts.get(artifact_id)
    if not item:
        raise HTTPException(404, 'Versão não encontrada.')
    if not item.get('parentId'):
        raise HTTPException(400, 'A imagem original não pode ser excluída.')
    children = [child for child in artifacts.values() if child.get('parentId') == artifact_id]
    if children:
        raise HTTPException(409, 'Esta versão possui resultados derivados. Exclua primeiro as versões que nasceram dela.')
    paths = {item.get(key) for key in ('path', 'previewPath', 'inspectPath', 'webpPath') if item.get(key)}
    paths.update(str(path) for path in ARTIFACTS_DIR.glob(f'{artifact_id}-*'))
    for value in paths:
        path = Path(value)
        if path.is_file() and path.parent.resolve() == ARTIFACTS_DIR.resolve():
            path.unlink(missing_ok=True)
    del artifacts[artifact_id]
    return {'ok': True, 'deletedId': artifact_id, 'activateId': item['parentId']}


@app.post('/api/jobs', status_code=202)
async def create_job(request: JobRequest) -> dict[str, Any]:
    gpu_operations = ('upscale', 'remove-background')
    if request.operation in gpu_operations and not torch.cuda.is_available():
        raise HTTPException(503, 'GPU CUDA não encontrada. No Colab, selecione um ambiente T4.')
    allowed = ('upscale', 'remove-background', 'clean', 'black-white', 'vectorize')
    if request.operation not in allowed or request.artifactId not in artifacts:
        raise HTTPException(400, 'Imagem ou operação inválida.')
    if any(j['status'] in ('queued', 'preparing', 'processing') for j in jobs.values()):
        raise HTTPException(409, 'Aguarde o trabalho atual terminar.')
    job_id = new_id('job')
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    job = {'id': job_id, 'artifactId': request.artifactId, 'operation': request.operation, 'options': request.options, 'status': 'queued', 'phase': 'queued', 'progress': None, 'resultArtifactId': None, 'error': None, 'createdAt': now, 'updatedAt': now}
    jobs[job_id] = job
    asyncio.get_running_loop().run_in_executor(executor, process_job, job_id)
    return job


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in jobs:
        raise HTTPException(404, 'Trabalho não encontrado.')
    return jobs[job_id]


@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id: str) -> dict[str, bool]:
    if job_id not in jobs:
        raise HTTPException(404, 'Trabalho não encontrado.')
    jobs[job_id]['cancelRequested'] = True
    jobs[job_id]['phase'] = 'cancelling'
    return {'ok': True}


@app.get('/api/artifacts/{artifact_id}/preview')
def preview(artifact_id: str, full: bool = False, inspection: bool = False) -> FileResponse:
    item = artifacts.get(artifact_id)
    if not item:
        raise HTTPException(404, 'Imagem não encontrada.')
    if full:
        return FileResponse(item['path'], media_type=item['mime'])
    if inspection:
        inspect_path = item.get('inspectPath')
        if not inspect_path:
            source = np.fromfile(item['path'], dtype=np.uint8)
            image = cv2.imdecode(source, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise HTTPException(500, 'Não foi possível gerar o preview de inspeção.')
            inspect_file = ARTIFACTS_DIR / f"{artifact_id}-inspect.webp"
            inspect_path = str(create_inspection_preview(image, inspect_file))
            item['inspectPath'] = inspect_path
        return FileResponse(inspect_path, media_type='image/svg+xml' if item['mime'] == 'image/svg+xml' else 'image/webp')
    preview_path = item.get('previewPath')
    if not preview_path:
        source = np.fromfile(item['path'], dtype=np.uint8)
        image = cv2.imdecode(source, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise HTTPException(500, 'Não foi possível gerar o preview.')
        preview_file = ARTIFACTS_DIR / f"{artifact_id}-preview.png"
        preview_path = str(create_preview(image, preview_file))
        item['previewPath'] = preview_path
    return FileResponse(preview_path, media_type='image/svg+xml' if item['mime'] == 'image/svg+xml' else 'image/png')


@app.get('/api/artifacts/{artifact_id}/download')
def download_artifact(artifact_id: str, format: str = 'png', background: str = '#ffffff') -> FileResponse:
    item = artifacts.get(artifact_id)
    if not item:
        raise HTTPException(404, 'Imagem não encontrada.')
    if item['mime'] == 'image/svg+xml':
        if format != 'svg':
            raise HTTPException(400, 'Esta versão é vetorial. Baixe no formato SVG.')
        return FileResponse(item['path'], media_type='image/svg+xml', filename=item['filename'])
    if format == 'png':
        if item['mime'] == 'image/png':
            return FileResponse(item['path'], media_type='image/png', filename=item['filename'])
        target_png = ARTIFACTS_DIR / f'{artifact_id}-download.png'
        if not target_png.exists():
            with Image.open(item['path']) as source:
                source.save(target_png, format='PNG', compress_level=3)
        png_name = str(Path(item['filename']).with_suffix('.png'))
        return FileResponse(target_png, media_type='image/png', filename=png_name)
    if format == 'webp':
        if max(item['width'], item['height']) > 16383:
            raise HTTPException(400, 'WebP suporta no máximo 16383 px por lado. Baixe o PNG sem perdas.')
        if not item.get('webpPath'):
            source = np.fromfile(item['path'], dtype=np.uint8)
            image = cv2.imdecode(source, cv2.IMREAD_UNCHANGED)
            ok, encoded = cv2.imencode('.webp', image, [cv2.IMWRITE_WEBP_QUALITY, 95])
            if not ok:
                raise HTTPException(500, 'Não foi possível gerar o WebP.')
            target_webp = ARTIFACTS_DIR / f'{artifact_id}-q95.webp'
            encoded.tofile(target_webp)
            item['webpPath'] = str(target_webp)
        webp_name = str(Path(item['filename']).with_suffix('.webp'))
        return FileResponse(item['webpPath'], media_type='image/webp', filename=webp_name)
    image = Image.open(item['path']).convert('RGBA')
    flat = Image.new('RGB', image.size, (255, 255, 255)); flat.paste(image, mask=image.getchannel('A'))
    target = ARTIFACTS_DIR / f'{artifact_id}-white.jpg'; flat.save(target, quality=95, subsampling=0)
    jpeg_name = str(Path(item['filename']).with_suffix('.jpg'))
    return FileResponse(target, media_type='image/jpeg', filename=jpeg_name)


@app.on_event('shutdown')
def cleanup() -> None:
    shutil.rmtree(DATA, ignore_errors=True)


ui_root = ROOT / 'dist' / 'ui'
if ui_root.exists():
    app.mount('/', StaticFiles(directory=ui_root, html=True), name='ui')
