"""Worker isolado para reconstrução raster de logos com FLUX Klein."""
import argparse
from pathlib import Path
import json
from logo_frame import prepare_reference, restore_frame


def progress(phase, completed=None, total=4):
    value={'phase':phase,'progress':None if completed is None else {'completedTiles':completed,'totalTiles':total}}
    print('SUPERIMAGEM_PROGRESS '+json.dumps(value),flush=True)

PROMPT = (
    'Clean the reference logo into flat 2D artwork suitable for professional tracing. '
    'Preserve the exact visible wording, letter shapes, spacing, composition, symbols, '
    'proportions and original color placement. Remove rendering effects, bevels, '
    'shadows, reflections, textures and noise. Solid flat colors, crisp smooth closed '
    'contours. Do not redesign, add details, replace lettering or invent unreadable text. '
    'Keep the complete logo inside the canvas, preserving framing and safe white margins. '
    'Do not crop or enlarge any element. Pure white background. No mockup, watermark, gradients or 3D effects.'
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--text', default='')
    args = parser.parse_args()
    import torch
    from PIL import Image, ImageOps
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, BitsAndBytesConfig
    from transformers import Qwen3ForCausalLM, BitsAndBytesConfig as TextQuantization
    if not torch.cuda.is_available():
        raise RuntimeError('FLUX exige GPU CUDA neste modo.')
    free,total=torch.cuda.mem_get_info()
    print(f'[GPU] {torch.cuda.get_device_name(0)} | VRAM livre {free/1024**3:.2f} / {total/1024**3:.2f} GiB',flush=True)
    torch.cuda.reset_peak_memory_stats()
    repo = 'black-forest-labs/FLUX.2-klein-4B'
    progress('loading_flux_transformer')
    print(f'[DOWNLOAD] {repo}/transformer — conferindo cache e baixando arquivos ausentes', flush=True)
    transformer = Flux2Transformer2DModel.from_pretrained(
        repo, subfolder='transformer', torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16),
    )
    print(f'[DOWNLOAD] {repo}/text_encoder — conferindo cache e baixando arquivos ausentes', flush=True)
    progress('loading_flux_encoder')
    encoder = Qwen3ForCausalLM.from_pretrained(
        repo, subfolder='text_encoder', torch_dtype=torch.float16,
        quantization_config=TextQuantization(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16),
    )
    print(f'[DOWNLOAD] {repo}/vae, tokenizer e scheduler — preparando pipeline', flush=True)
    progress('loading_flux_pipeline')
    pipe = Flux2KleinPipeline.from_pretrained(repo, transformer=transformer, text_encoder=encoder, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    source = ImageOps.exif_transpose(Image.open(args.input)).convert('RGBA')
    background = Image.new('RGBA', source.size, 'white')
    background.alpha_composite(source)
    reference = background.convert('RGB')
    print('[FLUX] Componentes carregados. Iniciando reconstrução em 4 passos.', flush=True)
    original_size=reference.size
    reference,content_box=prepare_reference(reference)
    width,height=reference.size
    progress('encoding_flux_reference')
    def on_step(pipeline,index,timestep,kwargs):
        progress('decoding_flux_result' if index==3 else 'reconstructing_logo',None if index==3 else index+1)
        return kwargs
    prompt=PROMPT+(f' The exact visible wording is: {args.text[:300]}. Preserve spelling and accents exactly.' if args.text else '')
    result = pipe(image=reference, prompt=prompt, width=width, height=height,
                  num_inference_steps=4, guidance_scale=1.0,
                  callback_on_step_end=on_step,
                  generator=torch.Generator('cpu').manual_seed(231)).images[0]
    progress('decoding_flux_result')
    result=restore_frame(result,content_box,original_size)
    result.save(Path(args.output))
    print(f'[GPU] Pico alocado {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB',flush=True)
    print('[FLUX] Reconstrução salva; aguardando aprovação do usuário.', flush=True)


if __name__ == '__main__':
    main()
