"""Instalação isolada e execução cancelável dos motores opcionais."""
import json
import os
import sys
import venv
from pathlib import Path
from colab.generative import run_checked

REQUIREMENTS = {
    'finegrain': ['git+https://github.com/finegrain-ai/refiners.git@bf5ac09c5855bc5167a5b1477370999c19d02bb8'],
    'fey': ['nobg==0.3.2'],
    'lucida': [],
    'vitmatte': [],
}


def segment(source, target, engine, options, job, root, mask=None):
    if engine not in REQUIREMENTS:
        raise ValueError('Motor opcional inválido.')
    environment = root / 'colab' / f'.optional-{engine}'
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    log = target.with_suffix('.log')
    marker = environment / '.ready-v1'
    print(f'[OPCIONAL] {engine}: preparando ambiente e pesos sob demanda', flush=True)
    job.update(status='loading_model', phase='loading_background_model')
    if not marker.exists() and REQUIREMENTS[engine]:
        run_checked([sys.executable, '-m', 'pip', '--python', str(python), 'install', *REQUIREMENTS[engine]], job, log)
        marker.touch()
    configuration = target.with_suffix('.json')
    configuration.write_text(json.dumps(options), encoding='utf-8')
    command = [str(python), str(root / 'colab/segmentation_worker.py'), '--input', str(source), '--output', str(target),
               '--engine', engine, '--options', str(configuration)]
    if mask:
        command += ['--mask', str(mask)]
    try:
        run_checked(command, job, log)
    finally:
        configuration.unlink(missing_ok=True)
    if not target.exists():
        raise RuntimeError(f'{engine} não produziu uma máscara.')
