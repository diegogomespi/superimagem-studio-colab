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

- A reconstrução generativa foi separada na aba `IA`, com a caixa `Usar IA` desligada por padrão. Ampliar/P&B não executam geradores.
- Ligada, solicita reconstrução raster com FLUX.2 Klein 4B, em ambiente Python separado. Instalação e pesos são obtidos somente no primeiro uso; a primeira execução pode demorar bastante.
- A integração FLUX é experimental: ainda não executada numa T4 deste projeto. Usa NF4 com cálculo FP16; erros são registrados e não desativam os motores convencionais.
- O resultado generativo exige Aceitar ou Descartar antes de continuar. O original permanece disponível. FLUX retorna RGB com fundo branco, não SVG nem alpha inferido. Confira letras e geometria manualmente.
- A inicialização lista o catálogo e o estado do cache. Downloads e carregamentos durante o uso aparecem no log vivo do Colab, com origem, arquivo, bytes/progresso quando disponíveis e confirmação de hash para os pesos oficiais de upscale.
- Download concluído, checksum validado e modelo carregado são eventos distintos. As bibliotecas Hugging Face exibem seus próprios progressos; não se declara SHA verificado para arquivos sem verificação explícita do aplicativo.
- P&B: Automático usa Logo sólida quando existe transparência real e Otsu para fundos opacos. Logo sólida torna RGB preto e mantém o alfa exato, incluindo furos, acentos e cobertura suave. Não restaura detalhes ausentes nem remove resíduos do alfa. Sauvola/Adaptativo continuam disponíveis para documentos com iluminação irregular, sem serem escolhidos por variação de cores da logo.
- Recorte Automático implementa candidatos BEN2/BiRefNet sequenciais, escolha, comparação A/B, aprovação e descarte. Teste de orquestração usa motores substitutos; qualidade dos pesos ainda precisa de T4.
- Instalação FLUX cria venv sem ensurepip, usando pip --python direcionado ao ambiente separado. Teste reproduz indisponibilidade de ensurepip sem baixar modelos.
- Log do Colab abre expandido, preserva até 3000 linhas recentes e mostra saída não silenciosa do pip; logs dos workers incluem também as últimas linhas antes do encerramento.
- Evolução: adaptadores opcionais de DiffBIR/Finegrain/FeyNoBg/Lucida/ViTMatte, pincéis, Inkvec, métricas e histórico múltiplo implementados. Motores CUDA novos ainda experimentais, aguardando inferência real. SDMatte foi avaliado e continua condicionado a benchmark antes de incorporar.
- Estado detalhado dos 16 itens, versões e procedimento de teste: [ENTREGA_16_ITENS.md](ENTREGA_16_ITENS.md). Código integrado não equivale a aprovação de qualidade na T4/CorelDRAW.
- Aba IA gera candidato no tamanho da entrada sem executar upscale em seguida. FLUX recebe referência proporcional com margem; o quadro é restaurado ao tamanho original. Isso corrige o encaixe, mas não impede que o modelo altere letras/desenhos.
- Tarefas ativas são recuperadas por `/api/jobs/active`; 409 inclui a tarefa que bloqueou a solicitação. Uma tarefa só é liberada como interrompida quando sua execução não existe ou já terminou, nunca apenas por estar lenta.
- Indicador central discreto mostra spinner e progresso mensurável quando disponível. Cancelamento permanece no painel lateral; detalhes técnicos permanecem no log Colab. Não há porcentagem fictícia.
- DiffBIR aplica patch versionado para a anotação upstream `torch.Tuple`, incompatível com o PyTorch atual. A preparação do ambiente v2 também exige `wcwidth>=0.3.5`. O teste `validate_diffbir_compat.py` reproduz o erro e valida a aplicação repetida da correção sem baixar pesos.

- TypeScript estrito e build Vite.
- Teste HTTP de upload, trabalhos, histórico e downloads PNG/SVG.
- Teste raster sintético de limpeza e preto e branco.
- Geração e validação estrutural de SVG real pelo VTracer.
- A remoção de fundo e o upscale continuam exigindo a validação final numa T4 nova, pois não há CUDA no teste local.

Mais detalhes: [colab/LEIA-ME-COLAB.md](colab/LEIA-ME-COLAB.md).

Planejamento e andamento: [CHECKLIST_IMPLEMENTACAO.md](CHECKLIST_IMPLEMENTACAO.md).
