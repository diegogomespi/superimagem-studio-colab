# Auditoria dos motores e referências

## Evidências locais — VectorBox

Inspeção somente de leitura da instalação em AppData/Local/Programs/VectorBox/resources.
Não foi copiado código proprietário para o projeto.

- ai/models.js: upscale local usa realesr-general-x4v3.onnx; execução ONNX CPU.
- ai/upscale.js: entrada limitada a 1600 px, blocos 256, margem 10, alpha separado e Lanczos; nitidez opcional por Sharp (sigma 1, m1 0.5, m2 2).
- web/assets/main-BaGZzSiA.js: interface envia detalhes, ruido e nitidez. A função local upscale lê somente scale e nitidez. Os dois primeiros controles não alteram essa função local.
- ai/rembg.js: BiRefNet-general ou ISNet conforme RAM; inferência 1024; saída limitada a 3000; opções de suavização/nitidez da máscara. Isso não equivale a matting com trimap.
- Não foi executado benchmark comparativo nem inspecionado o backend remoto do fornecedor.

## Referências públicas consultadas

- https://github.com/PramaLLC/BEN2 — integração oficial e refinamento de foreground. Separar estimativa de máscara de correção da cor do foreground.
- https://github.com/ZhengPeng7/BiRefNet — segmentação de alta resolução; a presença de objetos extras é erro de seleção semântica, não apenas de contorno.
- https://github.com/1038lab/ComfyUI-RMBG — coleção de motores e segmentação guiada; GPL-3.0. Não incorporar seu código sem avaliar compatibilidade da licença.
- https://github.com/hustvl/ViTMatte/blob/main/run_one_image.py — recebe RGB e trimap, retorna alpha. O exemplo original depende de Detectron2. Avaliar implementação Transformers para evitar esse peso de instalação.
- https://github.com/XPixelGroup/DiffBIR/blob/main/inference.py — v2.1; cleaner_tiled, vae_encoder_tiled, vae_decoder_tiled, cldm_tiled; captioner none evita modelos auxiliares. strength é força de ControlNet: menor permite MAIS criatividade, não mapear diretamente intensidade como se fosse denoise.
- https://github.com/logolabs/inkvec — alternativa para arte plana, CLI Rust e binários. Binário simples tem requisitos Linux inferiores ao binário com restorer ONNX. Comparar antes de mudar padrão.
- https://vectorization.eu/ — rodapé declara VTracer e UniConvertor; expõe filtragem de manchas, precisão de cor e ajustes de curvas. Não prova versão nem presets internos.
- https://upscale.tools/pt/result — leitura automatizada indisponível; capturas fornecidas mostram comportamento visual, não revelam modelo ou backend.

## Problemas comprovados no nosso código

1. BiRefNet refinado aplicava somente mediana 3x3; não havia trimap/matting adicional.
2. Limpar aplicava abertura e fechamento global do alpha, destruindo estruturas estreitas. Removido.
3. Limpar com cleanAlpha false descartava o alpha. Corrigido.
4. P&B aplicava mediana global, capaz de apagar pontos e acentos. Removido.
5. P&B considerava RGB invisível em pixels transparentes. Corrigido com composição branca para análise.
6. SwinIR recebia half incompatível; corrigido anteriormente para FP32.

## Decisões de implementação e critérios

- Recorte híbrido: máscaras independentes da mesma entrada; BEN2 orienta seleção e BiRefNet fornece detalhes próximos ao objeto. Não união/interseção global. Preservar cor original; medir caixas excluídas e cabelo preservado. Precisa teste T4 com originais.
- Matting: trimap com faixa incerta e inferência localizada; nunca prometer recuperar braço inteiro perdido pelos dois motores. Correção manual Manter/Remover necessária.
- Melhorias de imagem: redução de ruído antes de ampliar e nitidez ajustável após ampliar; manter alpha e evitar halos. Medir resultado em vez de copiar apenas nomes dos controles.
- Vetor: manter Potrace; expor presets do VTracer e comparar Inkvec em logos reais. Comparar SVG rasterizado, buracos, formas, nós e tamanho.
- DiffBIR: opcional, sob demanda, sem captioner grande, com blocos; resultado candidato para aceitar/descartar. Não garantir letras idênticas por OCR/SSIM. Não usar prompts fotográficos padrão em logos.
- Histórico múltiplo: Ctrl/Shift e exclusão em lote respeitando dependências, original protegida.

## Validação pendente

Nenhum novo modelo pesado foi executado na T4 nesta auditoria. As capturas não substituem imagens de entrada para medir máscaras e contornos. Instalação, inferência, qualidade e tempo do DiffBIR/ViTMatte/Inkvec continuam pendentes. O estudo e as correções convencionais não representam conclusão de toda a implementação solicitada.
