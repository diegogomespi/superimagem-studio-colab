"""Executa o CLI real; baixa o binário oficial, sem modelo de IA."""
import sys,tempfile
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from colab.inkvec_engine import convert
from colab.quality import svg_report
with tempfile.TemporaryDirectory() as directory:
    directory=Path(directory);source=directory/'input.png';target=directory/'output.svg'
    image=np.full((128,256,3),255,np.uint8)
    cv2.putText(image,'AB',(10,95),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),5)
    cv2.circle(image,(220,64),22,(20,50,210),-1)
    cv2.imwrite(str(source),image)
    convert(source,target,{}, {})
    report=svg_report(target)
    assert report['paths']>0
    print('OK: Inkvec real produziu SVG:',report)
