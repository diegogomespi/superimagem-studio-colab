"""Adaptadores de inferência; não depende de ComfyUI."""
import argparse
import json
import numpy as np
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser()
    for name in ('input','output','engine','options'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--mask')
    args = parser.parse_args()
    import torch
    if args.engine != 'foreground' and not torch.cuda.is_available():
        raise RuntimeError('CUDA indisponível para motor opcional.')
    if args.engine != 'foreground':
        free,total=torch.cuda.mem_get_info()
        print(f'[GPU] {torch.cuda.get_device_name(0)} | VRAM livre {free/1024**3:.2f} / {total/1024**3:.2f} GiB',flush=True)
        torch.cuda.reset_peak_memory_stats()
    options = json.load(open(args.options, encoding='utf-8'))
    image = ImageOps.exif_transpose(Image.open(args.input)).convert('RGB')
    width,height = image.size
    if args.engine == 'finegrain':
        from refiners.solutions import BoxSegmenter
        boxes = options.get('boxes', [])
        if not boxes or len(boxes)>20:
            raise ValueError('Desenhe de 1 a 20 caixas para selecionar o objeto.')
        print('[DOWNLOAD] finegrain/finegrain-box-segmenter revision v0.1', flush=True)
        model = BoxSegmenter(device='cuda', margin=.05)
        alpha = np.zeros((height,width),np.uint8)
        for box in boxes:
            if len(box)!=4 or not all(0<=float(x)<=1 for x in box) or box[0]>=box[2] or box[1]>=box[3]:
                raise ValueError('Caixa inválida.')
            mask = model(image,box_prompt=(round(box[0]*width),round(box[1]*height),round(box[2]*width),round(box[3]*height)))
            alpha = np.maximum(alpha,np.asarray(mask.convert('L')))
    elif args.engine == 'fey':
        from nobg.birefnet.modeling_birefnet import BiRefNet, BiRefNetConfig
        from nobg import BiRefNetImageProcessor
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        print('[DOWNLOAD] 1038lab/FeyNobg — alpha original sem pós-limiar', flush=True)
        directory = snapshot_download('1038lab/FeyNobg', revision='59dd27c65909b7dcc2de8c3239712b0b997c3e41', allow_patterns=['config.json','FeyNobg.safetensors','preprocessor_config.json'])
        from pathlib import Path
        directory=Path(directory)
        config=BiRefNetConfig(**json.loads((directory/'config.json').read_text()))
        model=BiRefNet(config)
        model.load_state_dict(load_file(str(directory/'FeyNobg.safetensors')))
        model.eval().cuda().half()
        processor=BiRefNetImageProcessor.from_pretrained(str(directory))
        inputs=processor(image,return_tensors='pt')
        with torch.inference_mode():
            prediction=model(pixel_values=inputs['pixel_values'].cuda().half())
            matte=processor.post_process_alpha_matting(prediction,target_sizes=[(height,width)])[0]
        alpha=np.rint(matte.detach().float().cpu().numpy().squeeze().clip(0,1)*255).astype(np.uint8)
    elif args.engine == 'lucida':
        from transformers import AutoModelForImageSegmentation
        print('[DOWNLOAD] egeorcun/lucida — model.safetensors (v7), normalização ImageNet', flush=True)
        model=AutoModelForImageSegmentation.from_pretrained('egeorcun/lucida',revision='6cbedc9722652dc9a3df91dd871f0c4f3334e922',trust_remote_code=True).eval().cuda()
        tensor=torch.from_numpy(np.asarray(image.resize((1024,1024))).copy()).permute(2,0,1).unsqueeze(0).cuda().float()/255
        mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
        std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
        with torch.inference_mode():
            prediction=model((tensor-mean)/std)[-1].sigmoid()
            matte=torch.nn.functional.interpolate(prediction,size=(height,width),mode='bilinear',align_corners=False)
        alpha=np.rint(matte[0,0].float().cpu().numpy().clip(0,1)*255).astype(np.uint8)
    elif args.engine == 'foreground':
        from pymatting import estimate_foreground_ml
        import cv2
        if not args.mask:raise ValueError('Correção de cor exige alpha aprovado.')
        mask=np.asarray(Image.open(args.mask).convert('L'))
        uncertain=(mask>0)&(mask<255)
        rgb=np.asarray(image).copy()
        if np.any(uncertain):
            yy,xx=np.where(mask>0)
            pad=32
            x0,x1=max(0,int(xx.min())-pad),min(width,int(xx.max())+pad+1)
            y0,y1=max(0,int(yy.min())-pad),min(height,int(yy.max())+pad+1)
            # Contexto completo do objeto, não apenas um recorte fino dos fios.
            roi=rgb[y0:y1,x0:x1].astype(np.float32)/255
            alpha_roi=mask[y0:y1,x0:x1].astype(np.float32)/255
            if roi.shape[0]*roi.shape[1]>18000000:
                raise ValueError('Correção de halos limitada a 18 MP por teste; use uma entrada menor.')
            foreground=estimate_foreground_ml(roi,alpha_roi)
            corrected=np.rint(np.clip(foreground,0,1)*255).astype(np.uint8)
            band=cv2.dilate(uncertain[y0:y1,x0:x1].astype(np.uint8),np.ones((3,3),np.uint8))>0
            rgb[y0:y1,x0:x1][band]=corrected[band]
        Image.fromarray(rgb,'RGB').save(args.output)
        print('[FOREGROUND] RGB corrigido na borda; alpha será preservado pelo chamador.',flush=True)
        return
    else:
        from transformers import VitMatteImageProcessor, VitMatteForImageMatting
        import cv2
        if not args.mask:
            raise ValueError('ViTMatte exige máscara inicial.')
        mask=np.asarray(Image.open(args.mask).convert('L'))
        radius=max(1,round(min(width,height)*float(options.get('trimapRatio',.006))))
        # Worker executado como script: tornar o pacote colab importável.
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from colab.matting_tools import make_trimap
        trimap=make_trimap(mask,radius)
        Image.fromarray(trimap,'L').save(str(args.output)+'.trimap.png')
        print(f'[VITMATTE] raio trimap {radius}px | unknown {np.count_nonzero(trimap==128)} | modo {options.get("mattingMode","tiles")}',flush=True)
        print('[DOWNLOAD] hustvl/vitmatte-small-composition-1k — RGB e trimap',flush=True)
        processor=VitMatteImageProcessor.from_pretrained('hustvl/vitmatte-small-composition-1k',revision='6a58ad7646403c1df626fbd746900aec7361ea1d')
        model=VitMatteForImageMatting.from_pretrained('hustvl/vitmatte-small-composition-1k',revision='6a58ad7646403c1df626fbd746900aec7361ea1d').eval().cuda()
        # Regiões limitadas a 768px com margem; só a faixa incerta é substituída.
        alpha=mask.copy()
        step=max(width,height) if options.get('mattingMode')=='whole' else 512
        if options.get('mattingMode')=='whole' and width*height>2200000:
            raise ValueError('Teste ViTMatte inteiro limitado a 2,2 MP. Não houve redução silenciosa; use entrada menor ou perfil por blocos.')
        for top in range(0,height,step):
            for left in range(0,width,step):
                bottom,right=min(height,top+step),min(width,left+step)
                if not np.any(trimap[top:bottom,left:right]==128):continue
                x0,y0=max(0,left-64),max(0,top-64);x1,y1=min(width,right+64),min(height,bottom+64)
                inputs=processor(images=image.crop((x0,y0,x1,y1)),trimaps=Image.fromarray(trimap[y0:y1,x0:x1]),return_tensors='pt').to('cuda')
                with torch.inference_mode():prediction=model(**inputs).alphas[0,0].float().cpu().numpy()
                local=np.rint(prediction[:y1-y0,:x1-x0].clip(0,1)*255).astype(np.uint8)
                core=local[top-y0:bottom-y0,left-x0:right-x0]
                unknown=trimap[top:bottom,left:right]==128
                alpha[top:bottom,left:right][unknown]=core[unknown]
        changed=np.count_nonzero(alpha != mask)
        print(f'[VITMATTE] pixels alterados {changed}/{mask.size} | diferença média alpha {np.abs(alpha.astype(float)-mask).mean():.3f}/255 | compare bordas antes/depois; não é uma nota de qualidade',flush=True)
    Image.fromarray(alpha,'L').save(args.output)
    print(f'[GPU] Pico alocado {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB',flush=True)
    print(f'[MODELO] {args.engine}: máscara salva na resolução {width}x{height}',flush=True)


if __name__=='__main__':main()
