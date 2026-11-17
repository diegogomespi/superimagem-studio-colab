"""Limpeza raster: foto e arte plana têm orçamentos diferentes de alteração."""
import cv2
import numpy as np


def clean_raster(image, options, job, cancelled):
    style = options.get('cleanStyle', 'illustration')
    if style not in ('photo', 'illustration'):
        raise ValueError('Tipo de limpeza inválido.')
    def amount(key, default):
        return np.clip(float(options.get(key, default)), 0, 100) / 100
    strength = amount('cleanStrength', 35)
    noise = amount('cleanDenoise', 40) * strength
    contrast = amount('cleanContrast', 30) * strength if options.get('autoContrast', True) else 0
    sharp = amount('cleanSharpness', 25) * strength
    edges = amount('cleanEdges', 25) * strength if options.get('cleanAlpha', True) and style == 'illustration' else 0
    print(f'[LIMPEZA v2] {style} | {image.shape[1]}x{image.shape[0]} | intensidade {strength:.0%} | ruído {noise:.0%} | contraste {contrast:.0%} | nitidez {sharp:.0%} | alfa {edges:.0%} | OpenCV/CPU, sem download de modelo', flush=True)
    if not any((noise, contrast, sharp, edges)):
        job['progress']['completedTiles'] = 4
        return image.copy(), f'OpenCV-clean-{style}-v2'
    alpha = image[:, :, 3].copy() if image.shape[2] == 4 else None
    bgr = image[:, :, :3].copy()
    cancelled(job)
    if noise:
        if style == 'photo':
            filtered = cv2.fastNlMeansDenoisingColored(bgr, None, 7 * noise, 10 * noise, 7, 21)
        else:
            filtered = cv2.bilateralFilter(bgr, 5, 8 + 70 * noise, 2 + 5 * noise)
        if alpha is not None:
            # Não misturar RGB arbitrário escondido no fundo nas bordas da arte.
            interior = cv2.erode((alpha == 255).astype(np.uint8), np.ones((7, 7), np.uint8))
            bgr = np.where(interior[:, :, None] > 0, filtered, bgr)
        else:
            bgr = filtered
    job['progress']['completedTiles'] = 1
    cancelled(job)
    if contrast or sharp:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        light = lab[:, :, 0].astype(np.float32)
        if contrast:
            # Curva tonal global: não confundir blocos de cor com iluminação local.
            gain = 1 + contrast * (.18 if style == 'photo' else .65)
            light = np.clip((light - 127.5) * gain + 127.5, 0, 255)
        job['progress']['completedTiles'] = 2
        cancelled(job)
        if sharp:
            if alpha is None:
                smooth = cv2.GaussianBlur(light, (0, 0), .8)
            else:
                coverage = alpha.astype(np.float32) / 255
                weight = cv2.GaussianBlur(coverage, (0, 0), .8)
                smooth = cv2.GaussianBlur(light * coverage, (0, 0), .8) / np.maximum(weight, 1e-6)
            detail = light - smooth
            detail[np.abs(detail) < (2 if style == 'photo' else 1)] = 0
            # Limitar overshoot evita que nitidez vire halo forte.
            light += np.clip(detail, -16, 16) * sharp * (.7 if style == 'photo' else 1.8)
        lab[:, :, 0] = np.clip(np.rint(light), 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Um original monocromático não pode ganhar cores pela conversão LAB.
        mono = np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 1] == image[:, :, 2])
        if mono:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    job['progress']['completedTiles'] = 3
    cancelled(job)
    if alpha is not None and edges:
        # Curva contínua, sem mediana, erosão ou remoção de componentes pequenos.
        # Endpoints e lado do limiar 128 permanecem: furos e acentos não desaparecem.
        coverage = alpha.astype(np.float32) / 255
        smoothstep = coverage * coverage * (3 - 2 * coverage)
        alpha = np.clip(np.rint(255 * (coverage + edges * (smoothstep - coverage))), 0, 255).astype(np.uint8)
    output = np.dstack((bgr, alpha)) if alpha is not None else bgr
    job['progress']['completedTiles'] = 4
    return output, f'OpenCV-clean-{style}-v2'
