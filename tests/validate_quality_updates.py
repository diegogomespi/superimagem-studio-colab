"""Regressões executáveis sem CUDA nem download de modelos."""
import ast
import html
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from colab.enhancement import improve_raster
from colab.operations import clean_image, black_white


def main():
    rgba = np.full((80, 100, 4), 255, np.uint8)
    rgba[15:65, 35:37, :3] = 0  # traço fino
    rgba[8:10, 35:37, :3] = 0  # acento separado
    rgba[:, :, 3] = np.arange(100, dtype=np.uint8)[None, :] * 2
    snapshot = rgba.copy()
    for noise, sharp in ((0, 0), (0, 25), (20, 25)):
        out = improve_raster(rgba, noise, sharp)
        assert np.array_equal(out[:, :, 3], rgba[:, :, 3])
        assert out.shape == rgba.shape and out.dtype == np.uint8
    assert np.array_equal(snapshot, rgba)
    out, _ = clean_image(rgba, {'cleanAlpha': False, 'cleanStrength': 0, 'autoContrast': False}, {})
    assert np.array_equal(out, rgba)
    opaque = snapshot.copy()
    # Logo colorida com furos, acento isolado e cobertura parcial nas bordas.
    logo = np.zeros((40, 60, 4), np.uint8)
    logo[10:35, 8:50] = (170, 170, 170, 255)
    logo[15:25, 20:30, 3] = 0
    logo[5, 15] = (30, 60, 150, 255)
    logo[10, 8:50, 3] = 96
    original = logo.copy()
    for mode in ('solid', 'auto'):
        out, engine = black_white(logo, {'thresholdMode': mode}, {})
        assert engine == 'threshold-solid'
        assert np.all(out[:, :, :3] == 0), 'Preenchimento ganhou buracos brancos'
        assert np.array_equal(out[:, :, 3], logo[:, :, 3]), 'Cobertura, furos ou acentos alterados'
        assert np.array_equal(logo, original), 'Original alterada'
    try:
        black_white(np.full((10, 10, 3), 255, np.uint8), {'thresholdMode': 'solid'}, {})
        raise AssertionError('Fundo opaco foi aceito como silhueta')
    except ValueError:
        pass
    opaque[:, :, 3] = 255
    for method in ('otsu', 'manual', 'auto'):
        out, _ = black_white(opaque, {'thresholdMode': method, 'threshold': 128}, {})
        assert set(np.unique(out[:, :, :3])).issubset({0, 255})
        assert np.all(out[8:10, 35:37, 0] == 0), 'Acento removido'
        assert np.all(out[15:65, 35:37, 0] == 0), 'Traço fino removido'
    # Testa o HTML efetivamente usado no Colab, sem depender do IPython.
    tree = ast.parse((ROOT / 'colab/launcher.py').read_text(encoding='utf-8'))
    render = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_render')
    scope = {'html': html, 'LOGS': ['[DOWNLOAD] modelo 50%', '<script>não executar</script>']}
    exec(compile(ast.Module(body=[render], type_ignores=[]), '<launcher-render>', 'exec'), scope)
    page = scope['_render']('Sistema online', 'Modelos sob demanda', 100, page_url='https://example.test/')
    assert '[DOWNLOAD] modelo 50%' in page
    assert '&lt;script&gt;' in page and '<script>' not in page
    assert 'ABRIR SUPERIMAGEM' in page
    print('OK: alpha, entrada intacta, desligar filtros, acentos, linhas, P&B puro e log Colab escapado.')


if __name__ == '__main__':
    main()
