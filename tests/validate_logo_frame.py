import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'colab'))
from logo_frame import prepare_reference,restore_frame
for size in ((327,186),(186,327),(1000,70),(70,1000),(512,512)):
    image=Image.new('RGB',size,'white')
    pixels=np.asarray(image).copy();pixels[0:3,:]=(200,0,0);pixels[-3:,:]=(0,0,200);image=Image.fromarray(pixels)
    framed,box=prepare_reference(image)
    assert framed.width%16==0 and framed.height%16==0
    assert box[0]>0 and box[1]>0 and box[2]<framed.width and box[3]<framed.height
    restored=restore_frame(framed,box,size)
    assert restored.size==size
    assert np.asarray(restored)[0,:,0].mean()>150
print('OK: quadros horizontal, vertical, panorâmico e quadrado; margens, proporção e dimensões restauradas. Não é teste de fidelidade do modelo.')
