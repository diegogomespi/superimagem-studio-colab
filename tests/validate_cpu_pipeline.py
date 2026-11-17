"""Validação rápida das operações que não precisam da T4.

Execute em um ambiente que tenha numpy, opencv-python-headless, Pillow e
vtracer==1.0.0a4 instalados.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from colab.operations import black_white, clean_image, vectorize


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

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "input.png"
        target = Path(directory) / "output.svg"
        assert cv2.imwrite(str(source), image)
        job = {}
        vectorize(source, target, {"vectorMode": "color", "maxColors": 8}, job)
        assert target.stat().st_size > 100
        assert "<svg" in target.read_text(encoding="utf-8")

    print("OK: limpeza, preto e branco e SVG VTracer validados.")


if __name__ == "__main__":
    main()
