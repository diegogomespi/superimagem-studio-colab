# Plano do SuperImagem Studio

## Regra principal

Não existe uma ordem obrigatória de ferramentas. A imagem selecionada é a versão ativa. Qualquer operação usa essa versão como entrada, cria uma nova versão e preserva todas as anteriores, inclusive a original.

Exemplos válidos:

- ampliar → baixar;
- ampliar → vetorizar;
- remover fundo → baixar;
- remover fundo → ampliar → vetorizar;
- limpar → remover fundo → ampliar → converter para preto e branco → vetorizar.

## Ferramentas independentes

### Ampliar

- 2×, 4× e 8×;
- Foto/geral e Ilustração/estampa;
- modo Fiel e Detalhado;
- RGB e alpha processados separadamente;
- CUDA FP16 e processamento em blocos.

### Remover fundo

- comparação controlada entre BiRefNet HR-Matting e BEN2;
- refinamento de cabelos, pelos e transparências parciais;
- descontaminação de cores e remoção de halos;
- fundo uniforme como modo especializado;
- correção guiada posteriormente.

### Limpar e preparar

- remoção de granulado e artefatos JPEG;
- limpeza da transparência;
- remoção de pontos e preenchimento de pequenos buracos;
- contraste e suavização controlados;
- preparação específica para vetor.

### Vetorizar

- colorido com agrupamento de paleta e VTracer;
- preto e branco automático ou ajustável;
- comparação entre VTracer e Potrace no modo binário;
- exportação SVG e, quando validado, PDF/EPS.

## Histórico

- original imutável;
- resultado vira automaticamente a versão ativa;
- selecionar qualquer versão anterior;
- voltar uma etapa;
- restaurar a original;
- baixar a versão ativa em qualquer momento;
- comparação sempre entre a entrada da operação e seu resultado.

## Validação antes de liberar cada motor

- registrar modelo, parâmetros, tempo e GPU;
- testar fotografias, ilustrações, textos, cabelos, pelos e PNG transparente;
- conferir resultado em zoom real;
- manter preview leve e download na resolução completa;
- não declarar uma ferramenta pronta somente porque a API respondeu sem erro.
