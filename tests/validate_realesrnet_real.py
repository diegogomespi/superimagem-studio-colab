"""Inferência real em CPU com pesos oficiais. Não mede desempenho da T4.

python tests/validate_realesrnet_real.py CAMINHO_PESO DIRETORIO_RESULTADOS
Depende de torch, spandrel e das dependências normais do backend.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from spandrel import ModelLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from colab.app import MODEL_SPECS, processed_filename, upscale_rgb_tiled

weight=Path(sys.argv[1]);directory=Path(sys.argv[2]);directory.mkdir(parents=True,exist_ok=True)
spec=MODEL_SPECS['logos']
assert hashlib.sha256(weight.read_bytes()).hexdigest()==spec['sha256']
model=ModelLoader().load_from_file(str(weight)).eval().to('cpu')
assert model.scale==4
assert model.supports_half, 'O modelo deve aceitar FP16 no Colab'
torch.set_num_threads(4)
font_path=Path('C:/Windows/Fonts/arial.ttf')
if not font_path.exists():font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
font=ImageFont.truetype(str(font_path),32)
canvas=Image.new('RGB',(384,192),'#ecdb33');draw=ImageDraw.Draw(canvas)
draw.text((18,12),'C.A.C',font=ImageFont.truetype(str(font_path),65),fill='#123d69')
draw.text((10,105),'Atlético · São João',font=font,fill='#172630')
draw.rectangle((12,156,372,161),fill='#123d69')
reference=cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)
small=cv2.resize(reference,(96,48),interpolation=cv2.INTER_AREA)
blur=cv2.GaussianBlur(small,(0,0),.45)
_,jpeg=cv2.imencode('.jpg',blur,[cv2.IMWRITE_JPEG_QUALITY,60])
cases={'logo-small':small,'logo-jpeg':cv2.imdecode(jpeg,cv2.IMREAD_COLOR)}
report={'engine':spec['name'],'device':'cpu','torch':torch.__version__,'sha256':spec['sha256'],'samples':[]}
for name,image in cases.items():
    job={'progress':{'completedTiles':0,'totalTiles':1}}
    start=time.perf_counter()
    out=upscale_rgb_tiled(image,model,job,[0],1,use_half=False,device='cpu')
    assert out.shape==reference.shape and out.dtype==np.uint8
    assert job['progress']['completedTiles']==1
    # Compare o mesmo raster com interpolação; não declarar superioridade universal.
    baseline=cv2.resize(image,(384,192),interpolation=cv2.INTER_LANCZOS4)
    metrics={'name':name,'durationSeconds':round(time.perf_counter()-start,3)}
    for label,pixels in [('model',out),('lanczos',baseline)]:
        metrics[label+'PSNR']=round(float(cv2.PSNR(reference,pixels)),3)
        truth=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)<120
        prediction=cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)<120
        metrics[label+'InkIoU']=round(float(np.count_nonzero(truth & prediction)/max(1,np.count_nonzero(truth | prediction))),4)
    report['samples'].append(metrics)
    cv2.imwrite(str(directory/(name+'.png')),np.concatenate((reference,baseline,out),axis=1))
    print(metrics,flush=True)
# Alpha segue o mesmo caminho da aplicação, sem inferência generativa.
alpha=np.zeros((48,96),np.uint8);alpha[5:40,10:80]=255;alpha[15:25,30:40]=0
alpha4=cv2.resize(alpha,(384,192),interpolation=cv2.INTER_LANCZOS4)
rgba=np.dstack((out,alpha4))
assert rgba.shape==(192,384,4) and rgba[80,140,3]==0
assert 'RealESRNet' in processed_filename('original.png',4,'logos','faithful')
(directory/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('OK: pesos oficiais, Spandrel, inferência real CPU, dimensões 4x, letras/acentos, JPEG e composição RGBA. Avaliação T4 e logo do usuário pendentes.')
