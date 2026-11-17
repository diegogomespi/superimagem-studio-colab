"""Ambiente opcional separado: dependências generativas não alteram o servidor."""
import os
import subprocess
import sys
import time
import venv
from pathlib import Path


def run_checked(command, job, log, cwd=None, env=None):
    offset = log.stat().st_size if log.exists() else 0
    with log.open('a', encoding='utf-8') as output:
        print(f'[PROCESSO] Iniciando: {Path(command[0]).name} | log {log.name}',flush=True)
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, cwd=cwd, env=env)
        try:
            while process.poll() is None:
                with log.open('r', encoding='utf-8', errors='replace') as reader:
                    reader.seek(offset)
                    chunk = reader.read()
                    offset = reader.tell()
                if chunk:
                    print(chunk, end='', flush=True)
                if job.get('cancelRequested'):
                    raise InterruptedError('Processamento cancelado.')
                time.sleep(0.25)
            with log.open('r', encoding='utf-8', errors='replace') as reader:
                reader.seek(offset)
                tail = reader.read()
            if tail:
                print(tail, end='', flush=True)
            if process.returncode:
                detail = log.read_text(encoding='utf-8', errors='replace')[-2500:]
                raise RuntimeError(f'Falha no processo opcional. O processamento convencional continua disponível.\n{detail}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def reconstruct(source: Path, target: Path, job: dict, root: Path):
    environment = root / 'colab' / '.flux-env'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    marker = environment / '.ready-v1'
    log = target.with_suffix('.log')
    job.update(status='loading_model', phase='installing_flux')
    print('[FLUX] Origem: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B | pesos sob demanda | NF4 / cálculo FP16', flush=True)
    print('[FLUX] Dependências: diffusers 0.37.1, transformers 4.57.6, accelerate 1.12.0, bitsandbytes 0.49.1', flush=True)
    # Colab pode não fornecer ensurepip/python3-venv. Criação sem bootstrap
    # e pip do processo pai direcionado explicitamente ao interpretador isolado.
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
    print(f'[FLUX] Ambiente isolado criado sem ensurepip: {python}', flush=True)
    if not marker.exists():
        run_checked([sys.executable, '-m', 'pip', '--python', str(python), 'install', '--progress-bar', 'on', 'diffusers==0.37.1',
                     'transformers==4.57.6', 'accelerate==1.12.0', 'bitsandbytes==0.49.1',
                     'sentencepiece==0.2.1', 'protobuf>=4,<7'], job, log)
        marker.touch()
    job.update(status='processing', phase='reconstructing_logo')
    run_checked([str(python), str(root / 'colab' / 'flux_worker.py'),
                 '--input', str(source), '--output', str(target)], job, log)
    if not target.exists():
        raise RuntimeError('FLUX não produziu a imagem esperada.')
