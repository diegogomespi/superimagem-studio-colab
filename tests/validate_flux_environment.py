"""Regressão do erro ensurepip, sem baixar pesos ou bibliotecas de IA."""
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab import generative


def main():
    with tempfile.TemporaryDirectory(prefix='superimagem-env-test-') as directory:
        root = Path(directory)
        commands = []
        def fake_run(command, job, log):
            commands.append(command)
            if '--output' in command:
                Path(command[command.index('--output') + 1]).touch()
        # Reproduz Colab sem ensurepip. Falharia se voltássemos a with_pip=True.
        with patch.object(venv.EnvBuilder, '_setup_pip', side_effect=RuntimeError('ensurepip indisponível')):
            with patch.object(generative, 'run_checked', fake_run):
                generative.reconstruct(root / 'input.png', root / 'output.png', {}, root)
                generative.reconstruct(root / 'input.png', root / 'output.png', {}, root)
        installs = [command for command in commands if 'install' in command]
        assert len(installs) == 1, 'Dependências não devem reinstalar a cada execução'
        assert installs[0][:4] == [sys.executable, '-m', 'pip', '--python']
        python = root / 'colab/.flux-env' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        subprocess.run([str(python), '-c', 'import sys; assert sys.prefix != sys.base_prefix'], check=True)
        print('OK: ambiente sem ensurepip, recuperação/reutilização e pip direcionado ao ambiente isolado.')


if __name__ == '__main__':
    main()
