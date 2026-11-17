"""Validação rápida das operações que não precisam da T4.

Execute em um ambiente que tenha numpy, opencv-python-headless, Pillow e
vtracer==1.0.0a4 instalados.
"""
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab.operations import black_white, clean_image, prepare_potrace_bitmap, vectorize


def main() -> None:
    image = np.zeros((96, 128, 4), dtype=np.uint8)
    image[16:80, 24:104, :3] = (20, 120, 240)
    image[16:80, 24:104, 3] = 255

    for operation, options in (
        (clean_image, {"cleanStyle": "illustration", "cleanStrength": 30, "cleanAlpha": True}),
        (black_white, {"thresholdMode": "adaptive", "adaptiveBlock": 31, "adaptiveC": 7}),
    ):
        job: dict = {}
        output, _ = operation(image.copy(), options, job)
        assert output.shape == image.shape and output.dtype == np.uint8
        assert job["progress"]["completedTiles"] == job["progress"]["totalTiles"]

    transparent = np.zeros((40, 60, 4), dtype=np.uint8)
    transparent[10:30, 20:40, :3] = 0
    transparent[10:30, 20:40, 3] = 255
    bitmap = prepare_potrace_bitmap(transparent)
    assert np.all(bitmap[:8] == 255), "O fundo transparente precisa virar branco para o Potrace."
    assert np.all(bitmap[12:28, 22:38] == 0), "A arte preta opaca precisa continuar preta."

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "input.png"
        target = Path(directory) / "output.svg"
        assert cv2.imwrite(str(source), image)
        job = {}
        vectorize(source, target, {"vectorMode": "color", "maxColors": 8}, job)
        assert target.stat().st_size > 100
        assert "<svg" in target.read_text(encoding="utf-8")

    print("OK: limpeza, P&B, alpha para Potrace e SVG VTracer validados.")


if __name__ == "__main__":
    main()
