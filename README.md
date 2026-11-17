# SuperImagem Studio Colab

Estúdio modular de processamento de imagens usando uma GPU NVIDIA T4 no Google Colab.

Este repositório é independente do `superimagem-colab`, que permanece como a versão simples e estável dedicada somente ao upscale. Aqui serão integradas ferramentas livres, sem ordem obrigatória: ampliar, remover fundo, limpar/preparar e vetorizar. Cada operação usa a versão ativa selecionada e cria uma nova versão sem alterar a imagem original.

## Recursos

- RealESRGAN x4plus oficial para fotos e uso geral.
- RealESRGAN x4plus Anime 6B para ilustrações e estampas.
- BiRefNet HR-Matting como removedor principal e BEN2 como alternativa refinada, ambos carregados sob demanda.
- Limpeza independente para foto/ilustração e limpeza do canal alpha.
- Conversão P&B independente com limiar adaptativo, Otsu ou manual.
- Vetorização colorida VTracer 1.0 (mosaico sem frestas) e Potrace para arte monocromática.
- Processamento CUDA FP16 em blocos de 400 px com sobreposição.
- PNG transparente preservado sem converter o fundo para branco.
- Comparação HD progressiva: abre com preview leve e carrega uma inspeção WebP de até 4096 px para revelar a diferença real do upscale sem travar a página.
- Zoom ancorado no ponteiro do mouse, arraste para navegar e divisor antes/depois alinhado.
- Progresso visual por blocos.
- Inicialização organizada com porcentagem, log completo, cópia do log e diagnóstico de erros.
- Página externa compatível com o endereço protegido do proxy do Google Colab.
- Downloads preservam o nome original e informam escala, modelo e modo utilizado.
- Preview inicial leve de até 1024 px; o download mantém a resolução final completa.
- Escolha explícita entre PNG sem perdas, WebP Q95 com transparência e JPEG Q95 sempre com fundo branco.
- Histórico de versões: cada resultado vira automaticamente a entrada ativa da próxima operação, sem alterar a imagem original.
- Voltar uma etapa, restaurar a original, selecionar qualquer versão anterior e baixar a versão ativa a qualquer momento.
- Ramificações reais: uma nova operação usa exatamente a versão escolhida como pai, mesmo que ela não seja a última da lista.
- Exclusão segura de versões: a original é protegida e uma versão com resultados derivados só pode ser apagada depois das versões-filhas.
- Layout responsivo: cartões lado a lado em tela larga e empilhados em tela estreita.

## Executar no Colab

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diegogomespi/superimagem-studio-colab/blob/main/SuperImagem_GPU_T4_Colab.ipynb)

1. Abra o notebook pelo botão acima.
2. Selecione **GPU T4** no tipo de ambiente de execução.
3. Execute a única célula **INICIAR SUPERIMAGEM GPU T4**.
4. Ao terminar, clique em **ABRIR SUPERIMAGEM EM NOVA ABA**. A interface não ocupa o notebook.

Na inicialização são validados os dois pesos pequenos do upscale. BiRefNet HR e BEN2 são maiores e baixam somente na primeira vez em que o motor é escolhido. Arquivos enviados, pesos baixados e resultados ficam no armazenamento temporário da sessão do Colab.

## Validação

- TypeScript estrito e build Vite.
- Teste HTTP de upload, trabalhos, histórico e downloads PNG/SVG.
- Teste raster sintético de limpeza e preto e branco.
- Geração e validação estrutural de SVG real pelo VTracer.
- A remoção de fundo e o upscale continuam exigindo a validação final numa T4 nova, pois não há CUDA no teste local.

Mais detalhes: [colab/LEIA-ME-COLAB.md](colab/LEIA-ME-COLAB.md).

Planejamento e andamento: [CHECKLIST_IMPLEMENTACAO.md](CHECKLIST_IMPLEMENTACAO.md).
