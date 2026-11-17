"""Fusão local de alfas independentes. Não é reconstrução generativa."""
import cv2
import numpy as np


def make_trimap(mask, radius):
    """Faixa configurável; não interpreta objeto nem reconstrói partes perdidas."""
    radius=max(1,int(radius))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    fg=cv2.erode((mask>245).astype(np.uint8),kernel)>0
    bg=cv2.dilate((mask>5).astype(np.uint8),kernel)==0
    result=np.full(mask.shape,128,np.uint8)
    result[fg]=255;result[bg]=0
    return result


def guided_alpha(base, detail, radius=None):
    if base.shape != detail.shape or base.ndim != 2:
        raise ValueError('As máscaras precisam ter as mesmas dimensões.')
    radius = radius or max(3,min(32,round(min(base.shape)*.008)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    support=cv2.dilate((base>8).astype(np.uint8),kernel)>0
    interior=cv2.erode((base>240).astype(np.uint8),kernel)>0
    unknown=support & ~interior
    result=base.copy()
    result[unknown]=detail[unknown]
    result[~support]=0
    # Não criar buracos novos no núcleo que o guia considerou opaco.
    result[interior]=base[interior]
    return result
