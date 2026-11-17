# Evolução completa — implementação e validação separadas

## Estado do código

| Item solicitado | Implementação | Verificação disponível |
|---|---|---|
| 1 A/B | BEN2 e BiRefNet sequenciais, candidatos, aceitar/descartar | Orquestração e PNGs com motores substitutos; pesos T4 pendentes |
| 2 Híbrido | Alternativa explícita experimental: BiRefNet na vizinhança da seleção BEN2 | Sintaxe; avaliação visual pendente; não é padrão |
| 3 Finegrain | Worker, revisão refiners fixada, caixas normalizadas, editor com múltiplas caixas | API oficial revisada; T4 pendente |
| 4 FeyNoBg/Lucida | Workers sob demanda; alpha original do Fey; Lucida oficial v7 com normalização | Inferência real pendente |
| 5 ViTMatte/SDMatte | ViTMatte por blocos com trimap, preservando regiões certas; SDMatte avaliado abaixo | Inferência real pendente |
| 6 Cabelo | Mediana removida; matting ViTMatte opcional para candidatos ou recorte individual | Matemática da máscara e testes CPU; cabelo real pendente |
| 7 Pincéis | Manter/Remover, tamanho/dureza/opacidade, desfazer/refazer local e aplicar/cancelar | Teste CPU e API: recupera pixels da entrada e modifica alpha |
| 8 Contraste | CLAHE somente na entrada analisada por BEN2/BiRefNet; RGB final original | Código revisado; comparação visual pendente |
| 9 DiffBIR | CLI upstream fixada, ambiente isolado, captioner none, blocos, FP16, SDP | Sintaxe/argumentos oficiais; instalação e T4 pendentes |
| 10 Melhorar | Análise heurística de ruído, limpeza quando indicada, upscale, nitidez, restauração opcional e P&B | Pipeline convencional/testes CPU; generativo T4 pendente |
| 11 Fidelidade | Acordo de bordas, diferença de luminância e OCR auxiliar Tesseract quando disponível, alerta e aceite manual | Casos sintéticos de identidade/divergência. Sem garantia tipográfica; OCR real depende da instalação do binário |
| 12 Motores SVG | Potrace, VTracer, Inkvec selecionáveis; resultados de motores diferentes como versões | Inkvec real Windows executado; VTracer real executado; Potrace Colab pendente |
| 13 SVG | Tolerância de curvas Potrace, manchas desativadas por padrão, relatório caminhos/comandos/tamanho, comparação rasterizada CairoSVG quando disponível | SVG real Inkvec e VTracer; CorelDRAW pendente. Comandos não equivalem exatamente a nós de edição |
| 14 Histórico múltiplo | Ctrl/Shift, todas, seleção indicada por etiquetas, lote validado antes de apagar, original protegida | API real com dependências e exclusão de ramo |
| 15 Cache Drive | Ativação explícita em Drive já montado; pesos upscale/Hugging Face/DiffBIR | Rotas implementadas; montagem real pendente |
| 16 T4/CorelDRAW | Procedimento de validação abaixo | NÃO executado; não declarar aprovado antes dos testes |

## SDMatte: resultado da avaliação

A referência carrega arquitetura externa SDMatte, componentes diffusion/transformers e checkpoint específico, recebendo imagem e máscara. Não é substituível por um filtro de borda e não deve usar máscara inteiramente branca como seleção automática. A integração de produção permanece condicionada a benchmark na T4, conforme o pedido original de avaliar antes de incorporar. Não há botão fictício de SDMatte. ViTMatte é a primeira implementação de refinamento por trimap, com fallback para a máscara inicial.

## Instalação sob demanda

- Nenhuma instalação completa de ComfyUI.
- Finegrain: refiners `bf5ac09c5855bc5167a5b1477370999c19d02bb8`, pesos `finegrain/finegrain-box-segmenter` revisão `v0.1`.
- Fey: `nobg==0.3.2`, arquivos de `1038lab/FeyNobg`. O espelho não tem model card/licença declarada; validar termos do peso antes de distribuição comercial.
- Lucida: `egeorcun/lucida`, checkpoint padrão v7, MIT; não confundir com m35 que tem normalização embutida.
- ViTMatte: `hustvl/vitmatte-small-composition-1k`, Transformers.
- DiffBIR: `5c2d6c177c3f2965e6d33a32931248d242ef7b3d`; utiliza torch do ambiente, evitando instalar torch/cu118 antigo. Compatibilidade dessa adaptação ainda precisa teste real.
- Inkvec: CLI oficial 0.2.6 com SHA-256 fixado; não baixa restorer nem exige GPU.
- FLUX: ambiente separado; NF4/FP16; ainda sem execução T4 confirmada.

## Limitações explícitas

- Logo fiel impede reconstrução generativa, inclusive com Usar IA marcado. Reconstruir/Fotografia exigem revisão.
- Um recorte perdido em ambos os modelos não é recuperado pelo matting sozinho; usar Manter sobre a entrada.
- Pincel tem preview aproximado; dureza/opacidade finais são calculadas na resolução completa.
- A análise automática de ruído/iluminação é heurística, não uma classificação semântica nem prova de qualidade.
- Não há garantia de letras idênticas com FLUX ou DiffBIR. Comparação de bordas não substitui leitura humana.
- Fontes e licenças dos checkpoints devem acompanhar eventual redistribuição; não foi copiado código GPL do ComfyUI-RMBG.
- CPU/RAM também limitam a sessão. Saídas muito grandes são bloqueadas antes de alocar o resultado.

## Teste real da sessão T4

1. Carregar a atualização no Colab; conferir commit, GPU e logs.
2. Enviar originais: pessoa com caixas, cabelo fino, braço, logo pequena com acentos/furos, JPEG degradado e PNG transparente.
3. Rodar A/B na mesma entrada, escolher candidato e confirmar que só ele entra no histórico.
4. Finegrain com caixa; Fey e Lucida individualmente; ViTMatte ligado/desligado na mesma entrada.
5. Restaurar braço com Manter; verificar com zoom e desfazer/refazer.
6. Melhorar com Usar IA desligado; depois testar FLUX e DiffBIR em Reconstruir. Não ativar pesos todos juntos.
7. P&B: conferir acentos e contraformas; exportar e comparar Potrace/VTracer/Inkvec partindo da mesma versão.
8. Abrir SVG no CorelDRAW, verificar curvas editáveis, furos, cores e ausência de bitmap incorporado.
9. Cancelar tarefas, provocar falta de memória controlada e confirmar que o servidor continua operacional.
10. Ativar cache apenas com Drive montado; encerrar sessão e confirmar reutilização dos pesos em uma sessão nova.

Registrar aprovação/falha por imagem e motor, tempo, RAM/VRAM e commit. Até isso acontecer, os motores novos continuam experimentais.
