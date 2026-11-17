"""Tratamentos conservadores independentes de CUDA."""
import cv2
import numpy as np


def improve_raster(image: np.ndarray, denoise: int = 0, sharpness: int = 25) -> np.ndarray:
    denoise = max(0, min(100, int(denoise)))
    sharpness = max(0, min(100, int(sharpness)))
    bgr = image[:, :, :3].copy()
    if denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, denoise * 0.07, denoise * 0.07, 7, 21)
    if sharpness:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        light = lab[:, :, 0].astype(np.float32)
        detail = light - cv2.GaussianBlur(light, (0, 0), 0.8)
        detail[np.abs(detail) < 1.5] = 0
        light += np.clip(detail, -12, 12) * (sharpness / 100)
        lab[:, :, 0] = np.clip(np.rint(light), 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.dstack((bgr, image[:, :, 3])) if image.shape[2] == 4 else bgr
