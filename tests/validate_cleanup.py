"""Regressões de qualidade mensurável, sem modelos CUDA."""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab.operations import clean_image


def run(image, **options):
    return clean_image(image, {'cleanStrength': 100, **options}, {})[0]


rng = np.random.default_rng(42)
flat = np.full((96, 128, 3), 130, np.uint8)
noisy = np.clip(flat.astype(float) + rng.normal(0, 7, flat.shape), 0, 255).astype(np.uint8)
for style in ('photo', 'illustration'):
    out = run(noisy, cleanStyle=style, cleanDenoise=100, cleanContrast=0, cleanSharpness=0)
    assert out.astype(float).std() < noisy.astype(float).std() * .7, 'Denoise não reduziu ruído'
    assert np.array_equal(run(noisy, cleanStyle=style, cleanStrength=0), noisy)
    assert np.array_equal(run(noisy, cleanStyle=style, cleanDenoise=0, cleanContrast=0, cleanSharpness=0, cleanEdges=0), noisy)

logo = np.zeros((80, 110, 4), np.uint8)
logo[:, :, :3] = 24
logo[15:65, 15:90, 3] = 255
logo[30:48, 40:58, 3] = 0  # furo
logo[7, 24, 3] = 255  # acento de um pixel
logo[:, :, 3] = cv2.GaussianBlur(logo[:, :, 3], (0, 0), .65)
logo[7, 24, 3] = 255
snapshot = logo.copy()
out = run(logo, cleanStyle='illustration', cleanDenoise=0, cleanContrast=100, cleanSharpness=0, cleanEdges=100)
assert np.all(out[35:42, 20:30, :3] < logo[35:42, 20:30, :3]), 'Preto não ficou mais firme'
assert np.array_equal(out[:, :, 3] >= 128, logo[:, :, 3] >= 128), 'Contorno/acentos/furos deslocados'
assert np.any(out[:, :, 3] != logo[:, :, 3]), 'Ajuste de borda não teve efeito'
assert np.array_equal(run(logo, cleanStyle='photo')[:, :, 3], logo[:, :, 3]), 'Foto alterou cabelo/alpha'
assert np.array_equal(run(logo, cleanEdges=0)[:, :, 3], logo[:, :, 3]), 'Desligar bordas falhou'
assert np.array_equal(logo, snapshot), 'Original modificada'

step = np.full((64, 96, 3), 210, np.uint8)
step[:, :48] = 45
blurred = cv2.GaussianBlur(step, (0, 0), 1.2)
sharp = run(blurred, cleanDenoise=0, cleanContrast=0, cleanSharpness=100)
assert int(sharp[32, 48, 0]) - int(sharp[32, 47, 0]) > int(blurred[32, 48, 0]) - int(blurred[32, 47, 0]), 'Nitidez não reforçou transição'
assert sharp.shape == blurred.shape
print('OK: denoise foto/arte, preto firme, nitidez, controle de alfa, acento, furo, intensidade zero e original intacta.')
