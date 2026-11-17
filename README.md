# SuperImagem Studio Colab

Estúdio modular de processamento de imagens usando uma GPU NVIDIA T4 no Google Colab.

Este repositório é independente do `superimagem-colab`, que permanece como a versão simples e estável dedicada somente ao upscale. Aqui serão integradas ferramentas livres, sem ordem obrigatória: ampliar, remover fundo, limpar/preparar e vetorizar. Cada operação usa a versão ativa selecionada e cria uma nova versão sem alterar a imagem original.

## Recursos

- RealESRGAN x4plus oficial para fotos e uso geral.
- RealESRGAN x4plus Anime 6B para ilustrações e estampas.
- SwinIR Classical x4 oficial em `Refinar logo / letras`, para recuperação mais fiel de bordas e texto antes da vetorização.
- BiRefNet HR-Matting como removedor principal e BEN2 como alternativa refinada, ambos carregados sob demanda.
- Limpeza independente para foto/ilustração e limpeza do canal alpha.
- Conversão P&B independente com limiar adaptativo, Otsu ou manual.
- Vetorização colorida VTracer 1.0 (mosaico sem frestas) e Potrace para arte monocromática.
- Vetorização P&B respeita o alpha: pixels transparentes são ignorados no traçado e o SVG permanece sem fundo.
- Processamento CUDA em blocos de 400 px com sobreposição: FP16 nos RealESRGAN e FP32 no SwinIR, conforme a compatibilidade de cada arquitetura.
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
- Setas pequenas de voltar/avançar em cada ferramenta, sem ocupar o visualizador nem apagar ramificações.
- Ramificações reais: uma nova operação usa exatamente a versão escolhida como pai, mesmo que ela não seja a última da lista.
- Exclusão segura de versões: a original é protegida e uma versão com resultados derivados só pode ser apagada depois das versões-filhas.
- Aba `Histórico` inspirada no painel do Photoshop, com histórico não linear sempre ativo, descrições das operações e gráfico compacto das ramificações.
- Layout responsivo: cartões lado a lado em tela larga e empilhados em tela estreita.

## Executar no Colab

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diegogomespi/superimagem-studio-colab/blob/main/SuperImagem_GPU_T4_Colab.ipynb)

1. Abra o notebook pelo botão acima.
2. Selecione **GPU T4** no tipo de ambiente de execução.
3. Execute a única célula **INICIAR SUPERIMAGEM GPU T4**.
4. Ao terminar, clique em **ABRIR SUPERIMAGEM EM NOVA ABA**. A interface não ocupa o notebook.

Na inicialização são validados os dois pesos pequenos do upscale comum. SwinIR Classical, BiRefNet HR e BEN2 são baixados sob demanda somente na primeira vez em que cada motor é escolhido. O peso oficial do SwinIR tem aproximadamente 65 MiB. Arquivos enviados, pesos baixados e resultados ficam no armazenamento temporário da sessão do Colab.

## Validação

### Atualização: Usar IA e logs sob demanda

- Ampliar e Preto e branco têm a caixa `Usar IA`, desligada por padrão.
- Ligada, solicita reconstrução raster com FLUX.2 Klein 4B, em ambiente Python separado. Instalação e pesos são obtidos somente no primeiro uso; a primeira execução pode demorar bastante.
- A integração FLUX é experimental: ainda não executada numa T4 deste projeto. Usa NF4 com cálculo FP16; erros são registrados e não desativam os motores convencionais.
- O resultado generativo exige Aceitar ou Descartar antes de continuar. O original permanece disponível. FLUX retorna RGB com fundo branco, não SVG nem alpha inferido. Confira letras e geometria manualmente.
- A inicialização lista o catálogo e o estado do cache. Downloads e carregamentos durante o uso aparecem no log vivo do Colab, com origem, arquivo, bytes/progresso quando disponíveis e confirmação de hash para os pesos oficiais de upscale.
- Download concluído, checksum validado e modelo carregado são eventos distintos. As bibliotecas Hugging Face exibem seus próprios progressos; não se declara SHA verificado para arquivos sem verificação explícita do aplicativo.
- Sauvola e seleção automática Otsu/Sauvola disponíveis no P&B. A escolha automática é uma heurística de iluminação e exige revisão visual.
- Ainda pendentes: DiffBIR, comparação de recortadores A/B, Finegrain, FeyNoBg, Lucida, ViTMatte/SDMatte, pincéis, Inkvec e histórico múltiplo. Esta atualização não declara o conjunto completo pronto.

- TypeScript estrito e build Vite.
- Teste HTTP de upload, trabalhos, histórico e downloads PNG/SVG.
- Teste raster sintético de limpeza e preto e branco.
- Geração e validação estrutural de SVG real pelo VTracer.
- A remoção de fundo e o upscale continuam exigindo a validação final numa T4 nova, pois não há CUDA no teste local.

Mais detalhes: [colab/LEIA-ME-COLAB.md](colab/LEIA-ME-COLAB.md).

Planejamento e andamento: [CHECKLIST_IMPLEMENTACAO.md](CHECKLIST_IMPLEMENTACAO.md).
