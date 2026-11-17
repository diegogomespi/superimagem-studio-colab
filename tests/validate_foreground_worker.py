"""Executa PyMatting real em CPU com composição sintética conhecida."""
import tempfile,runpy,sys
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as d:
    p=Path(d);yy,xx=np.indices((64,64))
    alpha=np.clip((24-np.sqrt((xx-32)**2+(yy-32)**2))/5,0,1).astype(np.float32)
    color=np.array([.18,.12,.08])
    rgb=np.rint((alpha[:,:,None]*color+(1-alpha[:,:,None]))*255).astype(np.uint8)
    mask=np.rint(alpha*255).astype(np.uint8)
    Image.fromarray(rgb).save(p/'input.png');Image.fromarray(mask).save(p/'mask.png')
    (p/'options.json').write_text('{}')
    sys.argv=['worker','--input',str(p/'input.png'),'--output',str(p/'output.png'),'--mask',str(p/'mask.png'),'--engine','foreground','--options',str(p/'options.json')]
    runpy.run_path(str(root/'colab/segmentation_worker.py'),run_name='__main__')
    out=np.asarray(Image.open(p/'output.png'))
    band=(alpha>.1)&(alpha<.9)
    target=color*255
    before=np.abs(rgb[band]-target).mean();after=np.abs(out[band]-target).mean()
    assert out.shape==rgb.shape and after<before,(before,after)
    assert np.array_equal(out[:5,:5],rgb[:5,:5])
    assert np.array_equal(np.asarray(Image.open(p/'mask.png')),mask)
    print(f'OK: PyMatting CPU real; erro RGB sintetico {before:.2f} -> {after:.2f}; fundo distante e arquivo alpha intactos. Nao comprova qualidade em fotos.')
