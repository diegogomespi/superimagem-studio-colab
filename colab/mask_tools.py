"""Correções de máscara com coordenadas normalizadas e composição contínua."""
import cv2
import numpy as np


def correct_mask(current, source, strokes):
    if len(strokes) > 2000:
        raise ValueError('Muitas pinceladas. Aplique a correção em etapas.')
    height, width = current.shape[:2]
    if current.ndim != 3 or current.shape[2] != 4:
        raise ValueError('Selecione uma versão com canal alpha para corrigir.')
    alpha = current[:, :, 3].astype(np.float32) / 255
    rgb = current[:, :, :3].astype(np.float32)
    source_rgb = source[:, :, :3].astype(np.float32)
    if source.shape[:2] != current.shape[:2]:
        raise ValueError('Entrada original e máscara têm dimensões diferentes.')
    source_alpha = source[:, :, 3] / 255 if source.shape[2] == 4 else np.ones((height, width), np.float32)
    for stroke in strokes:
        points = stroke.get('points', [])
        if len(points) > 10000:
            raise ValueError('Pincelada muito longa.')
        radius = max(1, min(max(width, height), float(stroke.get('size', .03)) * max(width, height) / 2))
        opacity = max(0, min(1, float(stroke.get('opacity', 1))))
        hardness = max(0, min(1, float(stroke.get('hardness', .7))))
        coverage = np.zeros((height, width), np.float32)
        previous = None
        for point in points:
            x, y = float(point[0]), float(point[1])
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError('Coordenada de pincel fora da imagem.')
            target = np.array([x * (width-1), y * (height-1)])
            steps = 1 if previous is None else max(1, int(np.linalg.norm(target-previous) / max(1, radius / 3)))
            start = target if previous is None else previous
            for center in np.linspace(start, target, steps + 1):
                cx, cy = center
                left, top = max(0, int(cx-radius-1)), max(0, int(cy-radius-1))
                right, bottom = min(width, int(cx+radius+2)), min(height, int(cy+radius+2))
                yy, xx = np.ogrid[top:bottom, left:right]
                distance = np.sqrt((xx-cx)**2+(yy-cy)**2)
                dab = np.clip((radius-distance)/max(1, radius*(1-hardness)), 0, 1)
                coverage[top:bottom, left:right] = np.maximum(coverage[top:bottom, left:right], dab)
            previous = target
        coverage *= opacity
        if stroke.get('mode') == 'keep':
            alpha = alpha*(1-coverage) + source_alpha*coverage
            rgb = rgb*(1-coverage[:,:,None]) + source_rgb*coverage[:,:,None]
        else:
            alpha *= 1-coverage
    return np.dstack((np.clip(np.rint(rgb),0,255).astype(np.uint8), np.rint(alpha*255).astype(np.uint8)))
