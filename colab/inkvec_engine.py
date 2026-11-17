"""CLI oficial Inkvec 0.2.6, sob demanda e com hash fixado."""
import hashlib
import io
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from colab.generative import run_checked


def convert(source, target, options, job):
    windows=os.name=='nt'
    name='inkvec.exe' if windows else 'inkvec'
    suffix='x86_64-pc-windows-msvc.zip' if windows else 'x86_64-unknown-linux-gnu.tar.gz'
    expected='f41d12526748576e035d9f339981762a882269990921928e7dc1732a2430f325' if windows else '541c0250d5735a164efd2619672dad51f09399db0f454800d20bf9370081b962'
    directory=Path(__file__).parent/'vendor'/'inkvec-0.2.6'
    binary=directory/name
    if not binary.exists():
        directory.mkdir(parents=True,exist_ok=True)
        url=f'https://github.com/logolabs/inkvec/releases/download/v0.2.6/inkvec-0.2.6-{suffix}'
        print(f'[DOWNLOAD] Inkvec: {url}',flush=True)
        with urllib.request.urlopen(url,timeout=120) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('SHA-256 do Inkvec inválido.')
        if windows:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                item=next(value for value in archive.namelist() if Path(value).name==name)
                binary.write_bytes(archive.read(item))
        else:
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
                item=next(value for value in archive.getmembers() if Path(value.name).name==name and value.isfile())
                stream=archive.extractfile(item)
                if stream is None:raise RuntimeError('Executável não encontrado no pacote.')
                binary.write_bytes(stream.read())
            binary.chmod(0o755)
        print('[MODELO] Inkvec CLI validado SHA-256; não usa GPU nem modelo generativo.',flush=True)
    command=[str(binary),str(source),'-o',str(target),'--mode','quality','--min-area','0','--max-dim','2048']
    run_checked(command,job,target.with_suffix('.log'))
    return 'Inkvec-0.2.6'
