"""Validar a saída antes de reservar GPU ou baixar pesos."""
def validate_upscale(source, options, available):
    scale = int(options.get('scale', 4))
    if scale not in (2, 4, 8):
        raise ValueError('Use ampliação 2×, 4× ou 8×.')
    if source.get('operation') == 'upscale' and not options.get('allowRepeatedUpscale'):
        raise ValueError('Esta versão já foi ampliada. Para comparar motores, volte à entrada anterior. Para ampliar novamente, marque essa opção explicitamente.')
    width, height = int(source['width']), int(source['height'])
    pixels = width * height * scale * scale
    intermediate_scale = 16 if scale == 8 and options.get('detailMode') == 'detailed' else max(4, scale)
    intermediate = width * height * intermediate_scale * intermediate_scale
    estimated = max(pixels * 16, intermediate * 12)
    if pixels > 180_000_000 or estimated > available * .7:
        raise ValueError(f'Saída {width*scale}×{height*scale} não cabe com segurança nesta sessão. Entrada atual: {width}×{height}. Use uma versão menor ou reduza a escala; nenhuma tarefa foi iniciada.')
