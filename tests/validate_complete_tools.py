"""Regressões locais de pincel, fidelidade e SVG. Não substitui T4/CorelDRAW."""
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from colab.mask_tools import correct_mask
from colab.quality import fidelity,svg_report


def main():
    source=np.full((100,120,3),170,np.uint8)
    cutout=np.dstack((source,np.zeros((100,120),np.uint8)))
    strokes=[dict(mode='keep',size=.15,opacity=1,hardness=.8,points=[[.3,.3],[.7,.3]])]
    kept=correct_mask(cutout,source,strokes)
    assert kept[30,60,3]==255 and kept[90,60,3]==0
    assert np.array_equal(kept[:,:,:3],source)
    removed=correct_mask(kept,source,[dict(mode='remove',size=.1,opacity=1,hardness=1,points=[[.5,.3]])])
    assert removed[30,60,3]==0
    assert np.array_equal(cutout[:,:,3],np.zeros((100,120),np.uint8))
    try:correct_mask(cutout,source,[dict(points=[[-1,0]])])
    except ValueError:pass
    else:raise AssertionError('Coordenadas inválidas aceitas')
    logo=np.full((100,120,3),255,np.uint8);cv2.putText(logo,'AB',(5,65),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2)
    assert fidelity(logo,logo)['edgeAgreement']==1
    assert fidelity(logo,np.full_like(logo,255))['review']=='divergence'
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'valid.svg'
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0 L20 0 L20 20 Z"/></svg>')
        report=svg_report(path);assert report['paths']==1 and report['commands']==4
    print('OK: Manter recupera pixels, Remover altera alpha, entrada intacta, coordenadas, divergência e análise SVG.')


if __name__=='__main__':main()
