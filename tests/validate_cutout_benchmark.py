"""Testa montagem e isolamento de entradas; não substitui inferência na T4."""
import json
import sys
import tempfile
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from colab.cutout_benchmark import run_benchmark
from colab.cutout_options import validate_boxes,candidate_engines

for value in (None,[],[[0,0,0,1]],[[0,0,float('nan'),1]],[[0,0,2,1]]):
    try:validate_boxes(value);raise AssertionError('Seleção inválida aceita')
    except ValueError:pass
validate_boxes([[.1,.1,.9,.9]])
assert candidate_engines({'cutoutSubject':'logo'})==('fey','ben2')
assert candidate_engines({'cutoutSubject':'person'})==('fey','birefnet-hr')
with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory)
    source=folder/'source.png';target=folder/'board.png'
    original=np.full((80,100,3),140,np.uint8);cv2.imwrite(str(source),original)
    calls=[]
    def fake(image,path,folder,engine,options,job,root):
        assert image.shape==(80,100,3) and np.array_equal(image,original)
        calls.append(engine);image[:]=0
        if engine=='fey':raise RuntimeError('Falha controlada de teste')
        return np.dstack((original,np.full((80,100),190,np.uint8)))
    report=run_benchmark(source,target,{'testPresets':['ben2','birefnet','fey'],'testCrop':[.1,.1,.7,.7]}, {},folder,infer=fake)
    assert calls==['ben2','birefnet-hr','fey']
    assert [r['status'] for r in report['tests']]==['ok','ok','error']
    assert report['cropPixels']==[10,8,70,56]
    assert np.array_equal(cv2.imread(str(source)),original)
    with Image.open(target) as board:assert board.size==(2560,2760)
    if len(sys.argv)>1:
        import shutil
        shutil.copyfile(target,sys.argv[1])
    with zipfile.ZipFile(target.with_suffix('.zip')) as archive:
        assert set(archive.namelist())=={'montagem.png','report.json','ben2.png','birefnet.png'}
        assert json.loads(archive.read('report.json'))['tests'][2]['status']=='error'
print('OK: entradas independentes, configurações, região comum, nomes, quadros de falha e pacote PNG/JSON. Motores substituídos neste teste.')
