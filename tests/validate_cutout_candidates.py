"""Teste de orquestração A/B com motores substitutos; não avalia os pesos CUDA."""
import ast
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'colab/app.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'process_job')
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        source = np.full((24, 32, 3), 140, np.uint8)
        source_path = directory / 'original.png'
        cv2.imwrite(str(source_path), source)
        seen = []
        def engine(image, options, job):
            assert np.array_equal(image, source), 'O segundo motor recebeu um recorte em vez da entrada original'
            seen.append(options['backgroundEngine'])
            alpha = np.full(image.shape[:2], 128 if len(seen) == 1 else 200, np.uint8)
            return np.dstack((image, alpha)), options['backgroundEngine']
        def preview(image, path):
            cv2.imwrite(str(path.with_suffix('.png')), image)
            return path.with_suffix('.png')
        artifacts = {'original': dict(id='original', path=str(source_path), filename='original.png')}
        job = dict(artifactId='original', operation='remove-background', options={'backgroundEngine': 'auto'})
        scope = dict(time=time, psutil=SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1))),
                     active_lock=threading.Lock(), jobs={'test': job}, artifacts=artifacts, np=np, cv2=cv2, Path=Path,
                     models={}, release_optional_models=lambda: None, remove_background=engine, ARTIFACTS_DIR=directory,
                     new_id=lambda prefix: prefix + uuid.uuid4().hex,
                     operation_filename=lambda *args: 'candidate.png', version_identity=lambda source: ('original', 2),
                     create_preview=preview, create_inspection_preview=preview)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<app-process>', 'exec'), scope)
        scope['process_job']('test')
        assert job['status'] == 'done', job
        assert seen == ['ben2', 'birefnet-hr']
        assert len(job['candidateArtifactIds']) == 2
        for candidate_id in job['candidateArtifactIds']:
            candidate = artifacts[candidate_id]
            assert candidate['candidate'] and candidate['parentId'] == 'original'
            assert Path(candidate['path']).exists()
        print('OK: candidatos independentes, mesma entrada, PNGs reais e aprovação pendente. Pesos CUDA não testados.')


if __name__ == '__main__':
    main()
