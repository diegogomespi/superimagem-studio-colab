"""Medições auxiliares de fidelidade; não certificam identidade tipográfica."""
import re
import io
import shutil
import subprocess
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def fidelity(source, output, with_ocr=False):
    def gray(image):
        if image.ndim==2:
            return cv2.resize(image,(min(768,source.shape[1]),min(768,source.shape[0])))
        rgb = image[:, :, :3].astype(np.float32)
        if image.shape[2] == 4:
            a = image[:, :, 3:4].astype(np.float32) / 255
            rgb = rgb * a + 255 * (1 - a)
        result = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        return cv2.resize(result, (min(768, source.shape[1]), min(768, source.shape[0])))
    a, b = gray(source), gray(output)
    ea, eb = cv2.Canny(a, 60, 160) > 0, cv2.Canny(b, 60, 160) > 0
    kernel = np.ones((3, 3), np.uint8)
    da = cv2.dilate(ea.astype(np.uint8), kernel) > 0
    db = cv2.dilate(eb.astype(np.uint8), kernel) > 0
    precision = float((eb & da).sum() / max(1, eb.sum()))
    recall = float((ea & db).sum() / max(1, ea.sum()))
    edge_score = 2 * precision * recall / max(1e-6, precision + recall)
    report=dict(edgeAgreement=round(edge_score, 3), meanPixelDifference=round(float(np.abs(a.astype(float)-b).mean()/255), 3),
                review='divergence' if edge_score < 0.55 else 'review',
                message='Confira letras, proporções e contornos. Esta medição não garante fidelidade nem substitui revisão visual.')
    if with_ocr:
        if shutil.which('tesseract'):
            def read_text(gray):
                enlarged=cv2.resize(gray,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
                _,encoded=cv2.imencode('.png',enlarged)
                result=subprocess.run(['tesseract','stdin','stdout','--psm','11'],input=encoded.tobytes(),capture_output=True,timeout=30)
                return result.stdout.decode('utf-8',errors='replace').strip()
            try:
                before,after=read_text(a),read_text(b)
                normalize=lambda text:re.sub(r'\W+','',text.casefold())
                changed=bool(normalize(before)) and normalize(before)!=normalize(after)
                report['ocr']=dict(before=before,after=after,changed=changed)
                if changed:
                    report['review']='divergence'
                    report['message']='OCR detectou possível mudança de texto. Confira as letras; fontes decorativas podem gerar falso alerta.'
            except (subprocess.TimeoutExpired,OSError) as exc:
                report['ocrWarning']=str(exc)
        else:report['ocrWarning']='Tesseract não disponível; revisão de contornos mantida.'
    return report


def svg_report(path):
    root = ET.parse(path).getroot()
    paths = [node for node in root.iter() if node.tag.rsplit('}', 1)[-1] == 'path']
    if any(node.tag.rsplit('}', 1)[-1] in ('script', 'image', 'foreignObject') for node in root.iter()):
        raise ValueError('SVG contém elemento incompatível com vetor puro.')
    return dict(paths=len(paths), commands=sum(len(re.findall(r'[MmLlHhVvCcSsQqTtAaZz]', node.get('d', ''))) for node in paths),
                bytes=path.stat().st_size)


def compare_svg(path, source):
    import cairosvg
    from PIL import Image
    height,width=source.shape[:2]
    factor=min(1,1024/max(width,height))
    png=cairosvg.svg2png(url=str(path),output_width=max(1,round(width*factor)),output_height=max(1,round(height*factor)))
    rendered=np.asarray(Image.open(io.BytesIO(png)).convert('RGBA'))
    rendered=cv2.cvtColor(rendered,cv2.COLOR_RGBA2BGRA)
    return fidelity(source,rendered)
