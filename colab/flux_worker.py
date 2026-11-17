"""Worker isolado para reconstrução raster de logos com FLUX Klein."""
import argparse
from pathlib import Path

PROMPT = (
    'Clean the reference logo into flat 2D artwork suitable for professional tracing. '
    'Preserve the exact visible wording, letter shapes, spacing, composition, symbols, '
    'proportions and original color placement. Remove rendering effects, bevels, '
    'shadows, reflections, textures and noise. Solid flat colors, crisp smooth closed '
    'contours. Do not redesign, add details, replace lettering or invent unreadable text. '
    'Pure white background. No mockup, watermark, gradients or 3D effects.'
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    from PIL import Image, ImageOps
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, BitsAndBytesConfig
    from transformers import Qwen3ForCausalLM, BitsAndBytesConfig as TextQuantization
    if not torch.cuda.is_available():
        raise RuntimeError('FLUX exige GPU CUDA neste modo.')
    repo = 'black-forest-labs/FLUX.2-klein-4B'
    print(f'[DOWNLOAD] {repo}/transformer — conferindo cache e baixando arquivos ausentes', flush=True)
    transformer = Flux2Transformer2DModel.from_pretrained(
        repo, subfolder='transformer', torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16),
    )
    print(f'[DOWNLOAD] {repo}/text_encoder — conferindo cache e baixando arquivos ausentes', flush=True)
    encoder = Qwen3ForCausalLM.from_pretrained(
        repo, subfolder='text_encoder', torch_dtype=torch.float16,
        quantization_config=TextQuantization(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16),
    )
    print(f'[DOWNLOAD] {repo}/vae, tokenizer e scheduler — preparando pipeline', flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(repo, transformer=transformer, text_encoder=encoder, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    source = ImageOps.exif_transpose(Image.open(args.input)).convert('RGBA')
    background = Image.new('RGBA', source.size, 'white')
    background.alpha_composite(source)
    reference = background.convert('RGB')
    print('[FLUX] Componentes carregados. Iniciando reconstrução em 4 passos.', flush=True)
    width, height = reference.size
    ratio = min(1024 / max(width, height), 2.0)
    width = max(64, round(width * ratio / 16) * 16)
    height = max(64, round(height * ratio / 16) * 16)
    result = pipe(image=reference, prompt=PROMPT, width=width, height=height,
                  num_inference_steps=4, guidance_scale=1.0,
                  generator=torch.Generator('cpu').manual_seed(231)).images[0]
    result.save(Path(args.output))
    print('[FLUX] Reconstrução salva; aguardando aprovação do usuário.', flush=True)


if __name__ == '__main__':
    main()
