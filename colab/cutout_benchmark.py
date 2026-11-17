"""Bateria real de recorte; cada preset recebe a mesma imagem intacta."""
import json
import time
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PRESETS = {
    'ben2': ('BEN2 · refinamento interno', 'ben2', {'refineEdges': True}),
    'birefnet': ('BiRefNet HR-Matting · 2048', 'birefnet-hr', {}),
    'fey': ('FeyNoBg · alpha original', 'fey', {}),
    'ben2-matte': ('BEN2 + ViTMatte', 'ben2', {'refineEdges': True, 'advancedMatting': True}),
    'birefnet-matte': ('BiRefNet HR + ViTMatte', 'birefnet-hr', {'advancedMatting': True}),
    'fey-matte': ('FeyNoBg + ViTMatte', 'fey', {'advancedMatting': True}),
    'hybrid': ('Híbrido BEN2 + BiRefNet', 'hybrid', {}),
    'birefnet-contrast': ('BiRefNet HR + contraste de análise', 'birefnet-hr', {'analysisContrast': True}),
    'lucida': ('Lucida · experimental', 'lucida', {}),
    'fey-biref': ('Fey guia + BiRefNet contraste nas bordas', 'fey', {'detailEngine':'birefnet-hr'}),
    'fey-biref-matte': ('Fey guia + BiRefNet bordas + ViTMatte', 'fey', {'detailEngine':'birefnet-hr','advancedMatting':True}),
    'birefnet-contrast-matte': ('BiRefNet contraste + ViTMatte', 'birefnet-hr', {'analysisContrast':True,'advancedMatting':True}),
    'finegrain': ('Finegrain · objeto selecionado', 'finegrain', {}),
    'finegrain-matte': ('Finegrain + ViTMatte', 'finegrain', {'advancedMatting':True}),
    'finegrain-biref-matte': ('Finegrain guia + BiRefNet bordas + ViTMatte', 'finegrain', {'detailEngine':'birefnet-hr','advancedMatting':True}),
    'research-fey': ('P1 Fey original - controle', 'fey', {}),
    'research-fey-color': ('P2 Fey + cor de borda PyMatting', 'fey', {'foregroundCorrection':True}),
    'research-narrow': ('P3 Fey + ViTMatte faixa estreita', 'fey', {'advancedMatting':True,'trimapRatio':.003}),
    'research-wide': ('P4 Fey + ViTMatte faixa larga', 'fey', {'advancedMatting':True,'trimapRatio':.012}),
    'research-wide-color': ('P5 Fey + faixa larga + cor', 'fey', {'advancedMatting':True,'trimapRatio':.012,'foregroundCorrection':True}),
    'research-guide': ('P6 Fey guia + BiRef contraste local', 'fey', {'detailEngine':'birefnet-hr'}),
    'research-guide-color': ('P7 Guia + ViTMatte + cor', 'fey', {'detailEngine':'birefnet-hr','advancedMatting':True,'trimapRatio':.006,'foregroundCorrection':True}),
    'research-whole': ('P8 Fey + ViTMatte inteiro ate 2.2MP', 'fey', {'advancedMatting':True,'trimapRatio':.006,'mattingMode':'whole'}),
}


def validate_presets(keys):
    if not isinstance(keys, list) or not 1 <= len(keys) <= 9 or len(set(keys)) != len(keys) or any(key not in PRESETS for key in keys):
        raise ValueError('Escolha de 1 a 9 testes distintos do catálogo.')


def infer_variant(image, source, folder, engine, options, job, root):
    from colab.operations import remove_background, release_optional_models
    from colab.optional_models import segment
    release_optional_models()
    if engine in ('fey', 'lucida', 'finegrain'):
        target = folder / 'mask.png'
        segment(source, target, engine, options, job, root)
        mask = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2]:
            raise ValueError('Dimensão de máscara inválida.')
        if image.shape[2] == 4: mask = np.minimum(mask, image[:,:,3])
        output = np.dstack((image[:,:,:3], mask))
    else:
        output, _ = remove_background(image.copy(), options, job)
    if options.get('detailEngine'):
        release_optional_models()
        detail,_=remove_background(image.copy(),{**options,'backgroundEngine':'birefnet-hr','analysisContrast':True},job)
        from colab.matting_tools import guided_alpha
        output[:,:,3]=guided_alpha(output[:,:,3],detail[:,:,3])
        del detail
    if options.get('advancedMatting'):
        release_optional_models()
        before, after = folder / 'initial-alpha.png', folder / 'matte.png'
        cv2.imwrite(str(before), output[:,:,3])
        segment(source, after, 'vitmatte', options, job, root, mask=before)
        mask = cv2.imread(str(after), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2]: raise ValueError('Matting inválido.')
        output[:,:,3] = np.minimum(mask,image[:,:,3]) if image.shape[2] == 4 else mask
    if options.get('foregroundCorrection'):
        release_optional_models()
        before,after=folder/'approved-alpha.png',folder/'foreground.png'
        cv2.imwrite(str(before),output[:,:,3])
        segment(source,after,'foreground',options,job,root,mask=before)
        foreground=cv2.imread(str(after),cv2.IMREAD_COLOR)
        if foreground is None or foreground.shape!=output[:,:,:3].shape:raise ValueError('Correção RGB inválida.')
        output[:,:,:3]=foreground
    return output


def run_benchmark(source, destination, options, job, root, infer=None):
    keys = options.get('testPresets', ['ben2','birefnet','fey'])
    validate_presets(keys)
    image = cv2.imdecode(np.fromfile(source,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    if image is None: raise ValueError('Não foi possível ler a entrada da bateria.')
    if image.ndim == 2: image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    infer = infer or infer_variant
    folder = destination.with_suffix('');folder.mkdir(parents=True,exist_ok=True)
    crop = options.get('testCrop')
    if crop:
        from colab.cutout_options import validate_boxes
        validate_boxes([crop])
    h,w = image.shape[:2]
    bounds = (round(crop[0]*w),round(crop[1]*h),round(crop[2]*w),round(crop[3]*h)) if crop else (0,0,w,h)
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]: raise ValueError('Área de inspeção muito pequena.')
    report={'sourceDimensions':[w,h],'cropPixels':list(bounds),'referenceUsedForInference':False,'tests':[]}
    # Guardar caminhos, não todos os rasters de resolução completa em RAM.
    panels=[('00 · Entrada original',Path(source),None)]
    for index,key in enumerate(keys,1):
        if job.get('cancelRequested'):raise InterruptedError('Bateria cancelada.')
        label,engine,preset=PRESETS[key]
        settings={'quality':'max',**preset}
        if engine=='finegrain':settings['boxes']=options.get('boxes',[])
        target_dir=folder/key;target_dir.mkdir(exist_ok=True)
        began=time.perf_counter()
        print(f'[BATERIA] {index}/{len(keys)} {label} | mesma entrada {w}x{h} | {settings}',flush=True)
        job.update(phase=f'Teste {index}/{len(keys)} · {label}',progress=None,testIndex=index,testCount=len(keys))
        record={'id':key,'label':label,'engine':engine,'options':settings}
        try:
            output=infer(image.copy(),Path(source),target_dir,engine,settings,job,root)
            if output.shape!=(h,w,4):raise ValueError('Saída não é RGBA no tamanho original.')
            path=target_dir/'result.png'
            if not cv2.imwrite(str(path),output):raise RuntimeError('Falha ao salvar resultado.')
            cv2.imwrite(str(target_dir/'final-alpha.png'),output[:,:,3])
            panels.append((f'{index:02d} · {label}',path,None))
            record['status']='ok'
            del output
        except InterruptedError:raise
        except Exception as exc:
            record.update(status='error',error=str(exc)[-500:])
            panels.append((f'{index:02d} · {label}',None,str(exc)[-180:]))
            print(f'[BATERIA] FALHA {key}: {exc}',flush=True)
        finally:
            from colab.operations import release_optional_models
            release_optional_models()
        record['durationSeconds']=round(time.perf_counter()-began,2)
        report['tests'].append(record)
    tile=1280;head=100;columns=2 if len(panels)<=4 else 3
    rows=(len(panels)+columns-1)//columns
    board=Image.new('RGBA',(columns*tile,rows*(tile+head)), '#161922')
    background=options.get('testBackground','#e02030')
    if background not in ('transparent','checker'):
        from PIL import ImageColor
        ImageColor.getrgb(background)
    report['montageBackground']=background
    report['individualPngs']='RGBA original; fundo da montagem não é gravado nos individuais'
    font_path=Path('C:/Windows/Fonts/arial.ttf')
    if not font_path.exists():font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),34) if font_path.exists() else ImageFont.load_default(size=34)
    draw=ImageDraw.Draw(board)
    for index,(label,path,error) in enumerate(panels):
        ox=(index%columns)*tile;oy=(index//columns)*(tile+head)
        import unicodedata
        caption=unicodedata.normalize('NFKD',label).encode('ascii','ignore').decode('ascii')
        draw.text((ox+18,oy+15),caption,font=font,fill='white')
        draw.text((ox+18,oy+53),f'Entrada {w}x{h} | mesma area | sem nova ampliacao IA',font=font,fill='#aab3c5')
        if path is None:
            draw.text((ox+20,oy+head+20),'FALHOU — veja relatório no pacote ZIP',font=font,fill='#ff9a85');continue
        with Image.open(path) as full:
            panel=full.convert('RGBA').crop(bounds)
            factor=min((tile-24)/panel.width,(tile-24)/panel.height)
            panel=panel.resize((max(1,round(panel.width*factor)),max(1,round(panel.height*factor))),Image.Resampling.LANCZOS)
        if background=='checker':
            yy,xx=np.indices((panel.height,panel.width));pattern=np.where(((xx//24+yy//24)%2)[:,:,None],np.array([55,59,68]),np.array([32,35,43])).astype(np.uint8)
            backing=Image.fromarray(pattern,'RGB').convert('RGBA')
        else:
            backing=Image.new('RGBA',panel.size,(0,0,0,0) if background=='transparent' else background)
        backing.alpha_composite(panel)
        board.paste(backing,(ox+(tile-panel.width)//2,oy+head+(tile-panel.height)//2))
    board.save(destination,optimize=False,compress_level=3)
    (folder/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    with zipfile.ZipFile(destination.with_suffix('.zip'),'w',compression=zipfile.ZIP_STORED) as archive:
        archive.write(destination,'montagem.png');archive.write(folder/'report.json','report.json')
        for key in keys:
            path=folder/key/'result.png'
            if path.exists():archive.write(path,f'{key}.png')
            for diagnostic in (folder/key).glob('*.png'):
                if diagnostic.name!='result.png':archive.write(diagnostic,f'diagnosticos/{key}/{diagnostic.name}')
    return report
