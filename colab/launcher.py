from __future__ import annotations

import html
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from IPython.display import HTML, display


LOGS: list[str] = []
DISPLAY_HANDLE = None
SERVER = None


def _append(message: object) -> None:
    for line in str(message).replace("\r", "").split("\n"):
        if line.strip():
            LOGS.append(line.rstrip())
    del LOGS[:-300]


def _render(stage: str, detail: str, progress: int, *, error: str = "", page_url: str = "") -> str:
    color = "#ef4444" if error else "#ff684a"
    safe_stage = html.escape(stage)
    safe_detail = html.escape(error or detail)
    log_text = "\n".join(LOGS)
    safe_log = html.escape(log_text)
    safe_value = html.escape(log_text, quote=True)
    progress = max(0, min(100, int(progress)))
    action = ""
    if page_url:
        safe_url = html.escape(page_url, quote=True)
        action = f"""
        <a href='{safe_url}' target='_blank' rel='noopener'
           style='display:inline-block;background:linear-gradient(90deg,#ff684a,#f23d68);color:#fff;padding:16px 34px;border-radius:10px;text-decoration:none;font-weight:800;font-size:16px;margin:8px 0 22px'>
          ABRIR SUPERIMAGEM EM NOVA ABA →
        </a>
        <p style='color:#817c86;font-size:12px;margin:0 0 18px'>Mantenha a sessão T4 do Colab aberta enquanto estiver usando.</p>
        """
    return f"""
    <div style='background:radial-gradient(circle at top,#26120d 0%,#0d0d12 58%);padding:32px;border-radius:20px;font-family:Inter,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #4d2118;box-shadow:0 24px 60px rgba(0,0,0,.4)'>
      <div style='text-align:center'>
        <div style='display:inline-block;width:15px;height:15px;border-radius:50%;background:{color};box-shadow:0 0 20px {color};margin-right:10px'></div>
        <h2 style='display:inline;color:#fff;font-size:2em'>SuperImagem GPU T4</h2>
        <p style='color:#ffb4a4;margin:10px 0 18px'>{safe_stage}</p>
        <div style='height:16px;background:#251410;border:1px solid #51261c;border-radius:999px;overflow:hidden;margin:0 auto 12px;max-width:720px'>
          <div style='height:100%;width:{progress}%;background:linear-gradient(90deg,#ff684a,#f23d68)'></div>
        </div>
        <strong style='color:#fff;font-size:20px'>{progress}%</strong>
        <p style='color:{"#ff8e8e" if error else "#b7a8a3"};min-height:20px'>{safe_detail}</p>
        {action}
      </div>
      <details {'open' if error else ''} style='background:#08080c;border:1px solid #33201c;border-radius:14px;padding:14px;color:#ddd'>
        <summary style='cursor:pointer;font-weight:700;color:#fff'>Log completo da instalação e inicialização</summary>
        <div style='text-align:right;margin:10px 0'>
          <button onclick="navigator.clipboard.writeText(document.getElementById('superimagem-log').value)" style='background:#2b1712;color:#fff;border:1px solid #6d3022;border-radius:8px;padding:8px 13px;cursor:pointer'>Copiar log</button>
        </div>
        <pre style='white-space:pre-wrap;word-break:break-word;max-height:340px;overflow:auto;color:#d9cbc7;font:12px/1.55 Consolas,monospace'>{safe_log or 'Aguardando...'}</pre>
        <textarea id='superimagem-log' style='position:absolute;left:-9999px'>{safe_value}</textarea>
      </details>
    </div>
    """


def _update(stage: str, detail: str, progress: int, *, error: str = "", page_url: str = "") -> None:
    global DISPLAY_HANDLE
    content = HTML(_render(stage, detail, progress, error=error, page_url=page_url))
    if DISPLAY_HANDLE is None:
        DISPLAY_HANDLE = display(content, display_id=True)
    else:
        DISPLAY_HANDLE.update(content)


def _run(command: list[str], stage: str, detail: str, progress: int, cwd: str | None = None) -> None:
    _append(f"> {detail}")
    _update(stage, detail, progress)
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        _append(line)
        _update(stage, detail, progress)
    if process.wait() != 0:
        raise RuntimeError(f"Falha em: {detail}")
    _append(f"OK: {detail}")


def _consume_server_output(process: subprocess.Popen) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        _append(f"SERVIDOR: {line.rstrip()}")


def run_from_colab(
    repo_url: str = "https://github.com/diegogomespi/superimagem-studio-colab.git",
    branch: str = "main",
    project_dir: str = "/content/SuperImagem",
    port: int = 7860,
) -> None:
    global SERVER
    try:
        LOGS.clear()
        _update("Preparando ambiente", "Validando a GPU T4...", 5)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GPU não encontrada. Selecione GPU T4 no ambiente de execução e tente novamente.")
        gpu = torch.cuda.get_device_properties(0)
        _append(f"GPU: {gpu.name} | VRAM: {gpu.total_memory / 1024**3:.1f} GB | CUDA: {torch.version.cuda}")

        repo = Path(project_dir)
        if (repo / ".git").is_dir():
            _run(["git", "fetch", "--depth", "1", "origin", branch], "Sincronizando GitHub", "Buscando a versão mais recente...", 15, project_dir)
            _run(["git", "reset", "--hard", f"origin/{branch}"], "Sincronizando GitHub", "Aplicando a atualização...", 25, project_dir)
        else:
            shutil.rmtree(repo, ignore_errors=True)
            _run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, project_dir], "Sincronizando GitHub", "Baixando o SuperImagem...", 20)

        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=project_dir, text=True).strip()
        _append(f"VERSÃO GITHUB CARREGADA: {commit}")

        _run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "colab" / "requirements-colab.txt")],
            "Instalando dependências",
            "Preparando ampliação, recorte e vetorização...",
            55,
            project_dir,
        )
        if shutil.which("potrace") is None:
            _run(["apt-get", "update", "-qq"], "Instalando vetorizador", "Atualizando a lista de pacotes...", 60)
            _run(["apt-get", "install", "-y", "-qq", "potrace"], "Instalando vetorizador", "Instalando Potrace para preto e branco...", 65)

        _update("Carregando aplicação", "Iniciando o servidor da página externa...", 78)
        subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        server_env = os.environ.copy()
        server_env['SUPERIMAGEM_BUILD_VERSION'] = commit
        SERVER = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "colab.app:app", "--host", "0.0.0.0", "--port", str(port)],
            cwd=project_dir,
            env=server_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=_consume_server_output, args=(SERVER,), daemon=True).start()

        import requests

        health = None
        for _ in range(90):
            if SERVER.poll() is not None:
                raise RuntimeError("O servidor encerrou durante a inicialização. Abra o log completo abaixo.")
            try:
                response = requests.get(f"http://127.0.0.1:{port}/api/health", timeout=2)
                response.raise_for_status()
                health = response.json()
                break
            except Exception:
                time.sleep(1)
        if health is None:
            raise RuntimeError("O servidor não respondeu dentro de 90 segundos.")

        _append(f"Aplicação pronta: {health['gpu']['name']}")
        _update("Baixando e validando modelos", "Conferindo os dois arquivos oficiais e carregando na GPU...", 86)
        models_response = requests.post(f"http://127.0.0.1:{port}/api/models/prepare", timeout=900)
        models_response.raise_for_status()
        prepared = models_response.json()
        if not prepared.get("ok") or len(prepared.get("models", [])) != 2:
            raise RuntimeError("A validação não confirmou os dois modelos.")
        for model in prepared["models"]:
            size_mb = model["bytes"] / 1024**2
            short_hash = model["sha256"][:12]
            _append(f"MODELO VALIDADO: {model['name']} | {size_mb:.1f} MB | SHA-256 OK ({short_hash}...) | escala {model['scale']}x | GPU pronta")

        _update("Publicando página externa", "Gerando o endereço seguro da sessão...", 95)
        from google.colab import output

        page_url = output.eval_js(f"google.colab.kernel.proxyPort({port})")
        _update("Sistema online!", f"Versão {health['version']} | GPU: {health['gpu']['name']} | 2 modelos baixados, validados e carregados", 100, page_url=page_url)
    except Exception as exc:
        _append(f"ERRO: {type(exc).__name__}: {exc}")
        _update("Falha ao iniciar", "Abra o log completo para identificar a etapa.", 100, error=str(exc))
        raise
