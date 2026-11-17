"""Reproduz a anotação inválida sem instalar torch ou baixar checkpoints."""
import ast
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from colab.diffbir_engine import ensure_compatibility

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='diffbir-compat-') as directory:
    repository=Path(directory)
    target=repository/'diffbir/sampler/edm_sampler.py'
    target.parent.mkdir(parents=True)
    if len(sys.argv)>1:
        shutil.copyfile(sys.argv[1],target)
    else:
        target.write_text('''class EDMSampler:
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: torch.Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
    ): pass
''')
    def annotation():
        tree=ast.parse(target.read_text(encoding='utf-8'))
        method=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='sample')
        argument=next(item for item in method.args.args if item.arg=='x_size')
        return eval(compile(ast.Expression(argument.annotation),'<annotation>','eval'),{'torch':SimpleNamespace()})
    try:annotation()
    except AttributeError:pass
    else:raise AssertionError('O teste não reproduziu o erro torch.Tuple')
    ensure_compatibility(repository,root,{},repository/'test.log')
    assert annotation()==tuple[int,...]
    ensure_compatibility(repository,root,{},repository/'test.log')
    assert annotation()==tuple[int,...]
print('OK: erro reproduzido, patch aplicado e repetição segura. Inferência CUDA não testada.')
