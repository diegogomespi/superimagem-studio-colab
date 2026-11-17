"""Operações independentes do SuperImagem Studio.

Os modelos grandes são carregados somente quando utilizados. As funções deste
módulo não conhecem FastAPI; isso permite testá-las separadamente.
"""
from __future__ import annotations

import gc
import subprocess
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageOps


Progress = Callable[[str, int, int], None]
_MODELS: dict[str, Any] = {}


def _cancelled(job: dict[str, Any]) -> None:
    if job.get("cancelRequested"):
        raise InterruptedError("Processamento cancelado.")


def _pil_rgba_from_bgra(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image, "L").convert("RGBA")
    if image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA), "RGBA")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "RGB").convert("RGBA")


def _bgra_from_pil(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGBA")), cv2.COLOR_RGBA2BGRA)


def _combine_alpha(source: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Mantém transparência já existente em vez de ressuscitar pixels vazios."""
    if source.ndim == 3 and source.shape[2] == 4:
        original = source[:, :, 3].astype(np.float32) / 255.0
        predicted = np.rint(predicted.astype(np.float32) * original).astype(np.uint8)
    return predicted


def remove_background(image: np.ndarray, options: dict[str, Any], job: dict[str, Any]) -> tuple[np.ndarray, str]:
    import torch
    engine = options.get("backgroundEngine", "birefnet-hr")
    refine = bool(options.get("refineEdges", True))
    job.update(status="loading_model", phase="loading_background_model")
    _cancelled(job)
    source_pil = _pil_rgba_from_bgra(image)

    if engine == "ben2":
        if "ben2" not in _MODELS:
            from ben2 import AutoModel
            _MODELS["ben2"] = AutoModel.from_pretrained("PramaLLC/BEN2").to("cuda").eval()
        model = _MODELS["ben2"]
        job.update(status="processing", phase="background_inference", progress={"completedTiles": 0, "totalTiles": 1})
        with torch.inference_mode():
            foreground = model.inference(source_pil.convert("RGB"), refine_foreground=refine)
        output = _bgra_from_pil(foreground)
        output[:, :, 3] = _combine_alpha(image, output[:, :, 3])
        job["progress"] = {"completedTiles": 1, "totalTiles": 1}
        return output, "BEN2-base-refined" if refine else "BEN2-base"

    if engine != "birefnet-hr":
        raise ValueError("Motor de remoção de fundo inválido.")
    if "birefnet-hr" not in _MODELS:
        from transformers import AutoModelForImageSegmentation
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_HR-matting", trust_remote_code=True
        ).eval().cuda().half()
        _MODELS["birefnet-hr"] = model
    model = _MODELS["birefnet-hr"]
    job.update(status="processing", phase="background_inference", progress={"completedTiles": 0, "totalTiles": 1})
    rgb = source_pil.convert("RGB")
    tensor = torch.from_numpy(np.asarray(rgb.resize((2048, 2048), Image.Resampling.LANCZOS)).copy())
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).cuda().half().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406), device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
    with torch.inference_mode():
        prediction = model((tensor - mean) / std)[-1].sigmoid()
        prediction = torch.nn.functional.interpolate(prediction, size=(image.shape[0], image.shape[1]), mode="bilinear", align_corners=False)
    alpha = np.rint(prediction[0, 0].float().cpu().numpy() * 255).astype(np.uint8)
    alpha = _combine_alpha(image, alpha)
    if refine:
        alpha = cv2.medianBlur(alpha, 3)
    output = image.copy() if image.ndim == 3 and image.shape[2] == 4 else np.dstack((image[:, :, :3], alpha))
    output[:, :, 3] = alpha
    del tensor, prediction
    torch.cuda.empty_cache()
    job["progress"] = {"completedTiles": 1, "totalTiles": 1}
    return output, "BiRefNet-HR-matting"


def clean_image(image: np.ndarray, options: dict[str, Any], job: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Limpeza conservadora para foto, estampa e transparência."""
    job.update(status="processing", phase="cleaning", progress={"completedTiles": 0, "totalTiles": 4})
    _cancelled(job)
    strength = max(0, min(100, int(options.get("cleanStrength", 35))))
    style = options.get("cleanStyle", "illustration")
    alpha = image[:, :, 3].copy() if image.ndim == 3 and image.shape[2] == 4 else None
    bgr = image[:, :, :3].copy()
    if strength:
        if style == "photo":
            h = 3 + round(strength * 0.07)
            bgr = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
        else:
            sigma = 12 + strength * 0.8
            bgr = cv2.bilateralFilter(bgr, 7, sigma, sigma)
    job["progress"]["completedTiles"] = 1
    _cancelled(job)
    if bool(options.get("autoContrast", True)):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.4 + strength / 100, tileGridSize=(8, 8)).apply(l)
        bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    job["progress"]["completedTiles"] = 2
    if alpha is not None and bool(options.get("cleanAlpha", True)):
        alpha[alpha <= int(options.get("alphaFloor", 4))] = 0
        alpha[alpha >= int(options.get("alphaCeiling", 251))] = 255
        output = np.dstack((bgr, alpha))
    else:
        output = np.dstack((bgr, alpha)) if alpha is not None else bgr
    job["progress"]["completedTiles"] = 4
    return output, f"OpenCV-clean-{style}"


def black_white(image: np.ndarray, options: dict[str, Any], job: dict[str, Any]) -> tuple[np.ndarray, str]:
    job.update(status="processing", phase="black_white", progress={"completedTiles": 0, "totalTiles": 3})
    bgr = image[:, :, :3]
    if image.ndim == 3 and image.shape[2] == 4:
        coverage = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = np.rint(bgr * coverage + 255 * (1 - coverage)).astype(np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    method = options.get("thresholdMode", "adaptive")
    if method == "manual":
        _, mask = cv2.threshold(gray, int(options.get("threshold", 128)), 255, cv2.THRESH_BINARY)
    elif method == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        block = max(3, int(options.get("adaptiveBlock", 31)) | 1)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, int(options.get("adaptiveC", 7)))
    if options.get("invert", False):
        mask = 255 - mask
    job["progress"]["completedTiles"] = 1
    morphology = int(options.get("lineWeight", 0))
    if morphology:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=min(3, morphology)) if morphology > 0 else cv2.erode(mask, kernel, iterations=min(3, -morphology))
    # Pontos, acentos e traços de um pixel podem ser legítimos: não aplicar
    # mediana global ao bitmap destinado a letras e serigrafia.
    job["progress"]["completedTiles"] = 2
    if image.ndim == 3 and image.shape[2] == 4:
        output = np.dstack((mask, mask, mask, image[:, :, 3]))
    else:
        output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    job["progress"]["completedTiles"] = 3
    return output, f"threshold-{method}"


def prepare_potrace_bitmap(image: np.ndarray) -> np.ndarray:
    """Converte raster para PBM respeitando a transparência.

    Potrace não interpreta alpha. Cores RGB existentes sob pixels transparentes
    (frequentemente preto) não podem participar do limiar, portanto o raster é
    composto em branco antes da binarização.
    """
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        composited = np.rint(bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        gray = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def vectorize(source_path: Path, output_path: Path, options: dict[str, Any], job: dict[str, Any]) -> str:
    job.update(status="processing", phase="vectorizing", progress={"completedTiles": 0, "totalTiles": 1})
    mode = options.get("vectorMode", "color")
    engine = options.get("vectorEngine", "vtracer")
    if engine == "potrace" and mode == "bw":
        temporary = output_path.with_suffix(".pbm")
        encoded = np.fromfile(source_path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Não foi possível abrir a imagem para vetorizar.")
        binary = prepare_potrace_bitmap(image)
        cv2.imwrite(str(temporary), binary)
        try:
            subprocess.run(["potrace", str(temporary), "--svg", "--output", str(output_path), "--turdsize", str(int(options.get("speckle", 2))), "--alphamax", "1.0", "--opttolerance", str(float(options.get("curveTolerance", 0.2)))], check=True, capture_output=True, text=True)
        finally:
            temporary.unlink(missing_ok=True)
        model_id = "Potrace"
    else:
        import vtracer
        if mode == "bw":
            cfg = vtracer.Config(clustering="bw", adaptive=bool(options.get("adaptive", True)), mode="spline")
        else:
            cfg = vtracer.Config.poster()
            cfg.mode = options.get("curveMode", "spline")
            cfg.hierarchical = "cutout"
            colors = int(options.get("maxColors", 16))
            cfg.max_colors = max(2, min(64, colors))
        cfg.convert_file(str(source_path), str(output_path))
        model_id = "VTracer-1.0.0a4"
    if not output_path.exists() or output_path.stat().st_size < 100 or "<svg" not in output_path.read_text(encoding="utf-8", errors="ignore")[:2000]:
        raise RuntimeError("O vetor gerado é inválido.")
    job["progress"] = {"completedTiles": 1, "totalTiles": 1}
    return model_id


def release_optional_models() -> None:
    _MODELS.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
