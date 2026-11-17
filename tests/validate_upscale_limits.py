import ast
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from colab.upscale_limits import validate_upscale

ram=12*1024**3
validate_upscale({'width':820,'height':803},{'scale':4},ram)
for source,options in [
    ({'width':10048,'height':10064},{'scale':4}),
    ({'width':9261,'height':5166},{'scale':4}),
    ({'width':100,'height':100,'operation':'upscale'},{'scale':4}),
    # Saída 2x pequena, mas intermediário nativo 4x não cabe.
    ({'width':2000,'height':2000},{'scale':2}),
]:
    try:
        validate_upscale(source,options,100_000_000 if source['width']==2000 else ram)
        raise AssertionError('Entrada insegura foi aceita')
    except ValueError:pass
validate_upscale({'width':100,'height':100,'operation':'upscale'},{'scale':4,'allowRepeatedUpscale':True},ram)

tree=ast.parse((Path(__file__).resolve().parents[1]/'colab/app.py').read_text(encoding='utf-8'))
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='upscale_with_fallback')
class OOM(Exception):pass
for multiple,expected in ((8,[128,96]),(1,[400,320])):
    calls=[]
    def infer(*args,**kwargs):
        tile=args[5];calls.append(tile)
        if len(calls)==1:raise OOM('test')
        return np.zeros((8,8,3),np.uint8)
    scope={'np':np,'Any':object,'torch':SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None)),'gc':SimpleNamespace(collect=lambda:None),'tile_count':lambda w,h,t:1,'upscale_rgb_tiled':infer}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<fallback>','exec'),scope)
    scope['upscale_with_fallback'](np.zeros((2,2,3),np.uint8),None,{},input_multiple=multiple)
    assert calls==expected,calls
print('OK: orçamento antes da GPU, intermediário 4x, ampliação repetida explícita e fallback de blocos SwinIR/RRDB. Inferência substituída no teste.')
