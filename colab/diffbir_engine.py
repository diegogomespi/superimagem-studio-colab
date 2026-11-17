"""Restauração opcional via CLI oficial DiffBIR, isolada do servidor."""
import os
import shutil
import sys
import tempfile
import venv
import subprocess
from pathlib import Path
from PIL import Image
from colab.generative import run_checked

REVISION='5c2d6c177c3f2965e6d33a32931248d242ef7b3d'


def ensure_compatibility(repository, root, job, log):
    patch=root/'colab/diffbir_torch_compat.patch'
    command=['git','apply','--check',str(patch)]
    check=subprocess.run(command,cwd=str(repository),capture_output=True,text=True)
    if check.returncode==0:
        run_checked(['git','apply',str(patch)],job,log,cwd=str(repository))
        print('[DIFFBIR] Compatibilidade aplicada: torch.Tuple substituído por tuple[int, ...].',flush=True)
    else:
        reverse=subprocess.run(['git','apply','--reverse','--check',str(patch)],cwd=str(repository),capture_output=True,text=True)
        if reverse.returncode:
            raise RuntimeError('A correção de compatibilidade não corresponde ao código DiffBIR instalado. '+check.stderr[-800:])
        print('[DIFFBIR] Correção torch.Tuple já aplicada.',flush=True)


def restore(source,target,options,job,root):
    environment=root/'colab/.diffbir-env'
    repository=root/'colab/vendor/DiffBIR'
    log=target.with_suffix('.log')
    job.update(status='loading_model',phase='installing_diffbir')
    print(f'[REPOSITÓRIO] https://github.com/XPixelGroup/DiffBIR | revisão {REVISION}',flush=True)
    if not (repository/'.git').exists():
        repository.parent.mkdir(parents=True,exist_ok=True)
        run_checked(['git','clone','https://github.com/XPixelGroup/DiffBIR.git',str(repository)],job,log)
        run_checked(['git','checkout','--detach',REVISION],job,log,cwd=str(repository))
    ensure_compatibility(repository,root,job,log)
    venv.EnvBuilder(with_pip=False,system_site_packages=True).create(environment)
    python=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    marker=environment/'.ready-v2'
    if not marker.exists():
        # Reutiliza torch/CUDA do Colab. Não instala o torch 2.2/cu118 antigo do upstream.
        run_checked([sys.executable,'-m','pip','--python',str(python),'install',
                     'omegaconf==2.3.0','ftfy==6.2.0','pytorch-lightning==2.5.6',
                     'lpips==0.1.4','facexlib==0.3.0','torchsde==0.2.6','pandas>=2,<3',
                     'wcwidth>=0.3.5,<1'],job,log)
        marker.touch()
    cache=os.environ.get('HF_HOME')
    if cache and not (repository/'weights').exists():
        directory=Path(cache).parent/'diffbir-weights';directory.mkdir(parents=True,exist_ok=True)
        (repository/'weights').symlink_to(directory,target_is_directory=True)
    with tempfile.TemporaryDirectory(prefix='diffbir-job-') as directory:
        directory=Path(directory);input_dir=directory/'input';input_dir.mkdir();output_dir=directory/'output'
        rgba=Image.open(source).convert('RGBA');flat=Image.new('RGB',rgba.size,'white');flat.paste(rgba,mask=rgba.getchannel('A'));flat.save(input_dir/'image.png')
        photo=options.get('enhanceMode')=='photo'
        prompt='clear natural photograph, faithful colors, clean detail' if photo else 'flat clean logo, exact original letters and geometry, solid original colors'
        command=[str(python),str(repository/'inference.py'),'--input',str(input_dir),'--output',str(output_dir),
                 '--version','v2.1','--task','sr','--upscale','1','--captioner','none','--precision','fp16',
                 '--steps','10','--cfg_scale','2','--strength','1','--noise_aug','0',
                 '--cleaner_tiled','--cleaner_tile_size','512','--cleaner_tile_stride','256',
                 '--vae_encoder_tiled','--vae_encoder_tile_size','256','--vae_decoder_tiled','--vae_decoder_tile_size','256',
                 '--cldm_tiled','--cldm_tile_size','512','--cldm_tile_stride','256',
                 '--pos_prompt',prompt,'--neg_prompt','changed letters, changed geometry, texture, artifacts, watermark']
        env=os.environ.copy();env['ATTN_MODE']='sdp';env['PYTHONUNBUFFERED']='1'
        job.update(status='processing',phase='restoring_diffbir')
        run_checked(command,job,log,cwd=str(repository),env=env)
        result=output_dir/'image.png'
        if not result.exists():raise RuntimeError('DiffBIR não produziu o arquivo esperado.')
        shutil.copyfile(result,target)
