import sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from colab.generative import run_checked
with tempfile.TemporaryDirectory() as directory:
    job={}
    script='import time; print(\'SUPERIMAGEM_PROGRESS {"phase":"reconstructing_logo","progress":{"completedTiles":2,"totalTiles":4}}\',flush=True); time.sleep(.7)'
    run_checked([sys.executable,'-c',script],job,Path(directory)/'worker.log')
    assert job['phase']=='reconstructing_logo'
    assert job['progress']['completedTiles']==2
    assert job['heartbeatEpoch']>0
    print('OK: subprocesso real envia etapa/progresso e heartbeat ao trabalho.')
