# Dossiê para pesquisa independente — SuperImagem Studio

Data: 17/11/2026. Idioma da resposta desejada: português.

## Missão do agente pesquisador

Investigue como construir um pipeline de remoção de fundo e refinamento de cabelos, logos e estampas com qualidade real, usando modelos abertos e uma NVIDIA T4 de 16 GB no Google Colab. Pesquise primeiro; proponha poucos experimentos com justificativa. Não se limite a listar modelos ou recomendar simplesmente o maior checkpoint.

Projeto: https://github.com/diegogomespi/superimagem-studio-colab

Baseline que produziu as montagens enviadas: commit `a5ee63c`.

Arquitetura: Python/FastAPI, PyTorch, frontend React/TypeScript. Sessão temporária do Colab. Modelos opcionais em processos separados, carregamento sob demanda e execução sequencial. Não instalar o ComfyUI inteiro nem reescrever o aplicativo.

O objetivo é preparar PNGs realmente transparentes para uso gráfico, CorelDRAW e estampas. Uma imagem PNG de uma logomarca continua sendo raster; chamá-la de “vetor” na interface não significa que os modelos estejam processando caminhos SVG.

## Problema observado

- Vários motores encontram a pessoa, mas mantêm caixas/produtos próximos ou retiram fios finos.
- Uma máscara mais detalhada pode incluir objetos indesejados. Uma seleção mais correta pode ter cabelo mais duro.
- Há halos claros em cabelos escuros fotografados sobre fundo branco. Acertar o alpha não necessariamente corrige a cor RGB já misturada com o fundo.
- Aplicar ViTMatte não trouxe melhora inequívoca nas montagens. Em algumas logos, conservou ou agravou falhas.
- Os resultados de serviços comerciais e do Photoshop são referências visuais melhores para o usuário. Não conhecemos seus pipelines internos; não presumir que usam os mesmos pesos abertos.
- O usuário não quer testes aleatórios repetidos. Quer entender qual etapa resolve qual defeito e ter presets simples por tipo de imagem.

## Motores efetivamente integrados

### 1. BEN2

Fonte: https://github.com/PramaLLC/BEN2

Checkpoint: https://huggingface.co/PramaLLC/BEN2

Integração: `ben2.AutoModel.from_pretrained(...).to('cuda').eval()` e `inference(image, refine_foreground=True/False)`.

Papel: candidato automático de seleção/remoção de fundo. O autor descreve `refine_foreground` como pós-processamento adicional que pode melhorar bordas, aumentando o tempo. Pesquisar exatamente o que essa opção altera: RGB, alpha, resolução ou múltiplas inferências.

Resultados observados: preservou o corpo em alguns retratos, mas incluiu caixas próximas na foto com prateleiras. Falhou em preservar partes internas do escudo e da logo tipográfica em alguns testes.

Não concluir que é universalmente ruim. Verificar inferência, normalização, checkpoint e uso sobre a mesma entrada.

### 2. BiRefNet HR-Matting

Fonte: https://github.com/ZhengPeng7/BiRefNet

Checkpoint: https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting

Papel: segmentação/matting em alta resolução. O model card informa treinamento em 2048×2048 para matting com transparência.

Integração: Transformers `AutoModelForImageSegmentation`, RGB redimensionado para 1024/1536/2048, normalização ImageNet, FP16, última predição com sigmoid e máscara redimensionada para a dimensão de entrada. O filtro mediano global antigo foi retirado.

Variação “contraste de análise”: CLAHE moderado no canal L da imagem usada para inferência; o RGB final vem da entrada sem essa alteração.

Resultados observados: o usuário aprovou detalhes do cabelo na variação 08 com contraste, mas ela manteve caixas. Nos escudos e na logo preta, várias regiões ficaram transparentes indevidamente.

Pergunta principal: como aproveitar seu detalhe sem herdar objetos de fundo selecionados?

### 3. FeyNoBg

Checkpoint: https://huggingface.co/1038lab/FeyNobg

Referência de integração: https://github.com/1038lab/ComfyUI-RMBG/blob/main/py/AILab_FeyNoBg.py

Pacote usado: `nobg==0.3.2`.

Revision do checkpoint: `59dd27c65909b7dcc2de8c3239712b0b997c3e41`.

Arquivos: `config.json`, `FeyNobg.safetensors`, `preprocessor_config.json`.

Integração: classes BiRefNet do nobg e `BiRefNetImageProcessor`, inferência FP16 e `post_process_alpha_matting` na resolução original. Preserva o alpha contínuo, sem copiar a limiarização do pós-processamento de exemplo do ComfyUI.

Resultados observados: foi o candidato mais convincente para excluir as caixas na foto específica da mulher e preservou melhor o escudo completo. Também preservou a escrita da logo preta melhor que vários outros candidatos. Isso não comprova superioridade em todos os casos.

Pesquisar a diferença entre a predição de alfa, o pós-processamento do processor e o refinamento opcional do exemplo de integração. Não multiplicar RGB por alpha e depois tratar esse resultado como RGB não premultiplicado.

### 4. Lucida

Checkpoint/model card: https://huggingface.co/egeorcun/lucida

Revision utilizada: `6cbedc9722652dc9a3df91dd871f0c4f3334e922`.

É uma adaptação baseada em BiRefNet. O autor a apresenta para camuflagem, transparências, texto/logos, ilustração e efeitos. As comparações do autor são evidência do autor, não validação independente no nosso conjunto.

Integração atual: `AutoModelForImageSegmentation`, entrada 1024, normalização ImageNet, sigmoid na última predição, saída alfa interpolada.

Resultados observados: não resolveu o escudo, incluiu caixas na pessoa e deixou franjas claras no cabelo cacheado. A logo tipográfica teve um resultado relativamente melhor, mas não há PNGs individuais disponíveis para análise precisa desses testes.

Investigar se o checkpoint/revision atual corresponde ao recomendado hoje, se existe processor próprio, requisitos de resolução e diferenças entre versões. Não declarar a integração errada apenas porque uma amostra ficou ruim; verificar código e pesos.

### 5. Finegrain Box Segmenter

Código: https://github.com/finegrain-ai/refiners/blob/main/src/refiners/solutions/box_segmenter.py

Pesos: https://huggingface.co/finegrain/finegrain-box-segmenter

Integração: `refiners.solutions.BoxSegmenter`, `margin=0.05`, caixas normalizadas convertidas para pixels. Revision dos pesos `v0.1` pelo wrapper. Pacote fixado em `bf5ac09c5855bc5167a5b1477370999c19d02bb8` do repositório refiners.

Papel: selecionar um objeto indicado por uma caixa. O wrapper recorta a região com margem, executa MVANet e devolve a máscara ao sistema de coordenadas da imagem. Não é um refinador dedicado de fios.

Erro confirmado: foi executado sem caixa, gerando `ValueError: Desenhe de 1 a 20 caixas...`. Isso é falha de fluxo/validação; não prova falha do modelo. As capturas recentes também mostravam zero regiões. Ainda não temos resultado bem-sucedido documentado desse motor nesse caso.

O repositório refiners informa descontinuação no início de 2025. Verificar manutenção, compatibilidade e alternativas antes de torná-lo dependência central. Verificar separadamente a licença do código e dos pesos.

### 6. ViTMatte

Fonte: https://github.com/hustvl/ViTMatte

Checkpoint atual: https://huggingface.co/hustvl/vitmatte-small-composition-1k

Revision: `6a58ad7646403c1df626fbd746900aec7361ea1d`.

Papel: estimar alpha usando RGB original + trimap com fundo certo, primeiro plano certo e região incerta. Não é um seletor autônomo de pessoa ou logo.

Integração atual: foreground certo a partir de máscara >245 com erosão; fundo certo onde dilatação de máscara >5 permanece vazia; resto recebe 128. Raio proporcional de aproximadamente 0,6% da menor dimensão, mínimo 3. Inferência em núcleos de 512 px com margem de 64; substituir apenas a faixa incerta. Alfa existente da entrada limita a saída.

Resultados observados: não houve melhora clara em toda a foto. Alguns testes de logo ficaram com buracos. O trimap pode congelar regiões incorretas como certas. Há risco de perder contexto/emendas no processamento em blocos.

Pesquisar tamanho da região incerta, conectividade, contexto, critérios de região certa, modelo small/base e Composition-1k versus Distinctions-646. Não presumir que trocar checkpoint resolva um trimap errado.

### 7. Híbrido existente BEN2 + BiRefNet

Não é um checkpoint novo. Executa os dois sobre a mesma entrada e usa o alpha do BiRefNet apenas na vizinhança da máscara do BEN2. A vizinhança é obtida com dilatação proporcional, cerca de 1,5% da menor dimensão.

É uma heurística experimental. Pode excluir detalhes que o BEN2 não encontrou ou permitir objetos próximos. Nos testes enviados, não foi uma solução consistente.

Não confundir com executar um segundo modelo sobre um PNG já recortado: esse não é o procedimento desejado.

## Nove configurações que já foram testadas

1. BEN2 com refinamento interno.
2. BiRefNet HR-Matting 2048.
3. FeyNoBg com alpha original.
4. BEN2 + ViTMatte.
5. BiRefNet HR + ViTMatte.
6. FeyNoBg + ViTMatte.
7. Híbrido BEN2/BiRefNet descrito acima.
8. BiRefNet HR + contraste somente de análise.
9. Lucida.

Finegrain não estava nesse grupo porque exige uma caixa. A nova pesquisa deve incluir uma seleção válida e compará-la com as alternativas, não repetir os mesmos nove sem mudanças justificadas.

## Leitura preliminar das montagens

- Pessoa com prateleiras/caixas: 03 e 06 excluem melhor objetos próximos. O 08 chama atenção pelo cabelo, mas mantém objetos do fundo. Não é aceitável escolher o cabelo e ignorar o erro semântico.
- Escudo azul/amarelo: 03 e 06 preservam a maior parte do escudo; várias máscaras dos outros testes retiram o preenchimento do próprio escudo.
- Logo preta sobre branco: requer preservar letras, acentos, miolos e antialiasing. Um método específico de remoção de fundo uniforme pode ser mais adequado que segmentação semântica genérica.
- Cabelo cacheado sobre branco: vários métodos deixam contorno claro. Investigar cor residual do fundo, não somente quantidade de fios.
- As montagens têm redução para caber nos quadros e fundo quadriculado gravado no RGB. Não são os alfas originais. Não calcular métricas de alpha a partir delas nem prometer ranking fino de fios com base apenas na imagem inteira.

## Hipóteses novas que precisam de investigação

### A. Guia semântico FeyNoBg + detalhe BiRefNet com contraste

Gerar duas máscaras independentes da mesma imagem original. Preservar o núcleo confiável do guia, rejeitar regiões distantes que ele considerou fundo e aproveitar o alpha detalhado apenas em uma faixa local incerta. Avaliar uma variante com ViTMatte depois.

Não usar união completa, interseção completa, mínimo global nem multiplicação indiscriminada: podem respectivamente recuperar caixas ou apagar fios. Definir a faixa e testar casos onde o guia erra. A proposta ainda é hipótese, não solução comprovada.

### B. Seleção guiada + matting

Finegrain com caixa válida → trimap → ViTMatte. Comparar com SAM/segmentação guiada mais atual caso a investigação de compatibilidade favoreça outra ferramenta. A seleção identifica o objeto; o matting calcula transições. Verificar se múltiplas caixas e margens realmente ajudam neste caso.

### C. Estimativa da cor do primeiro plano

Após alpha confiável, investigar descontaminação de bordas/estimativa de foreground. Fonte técnica: https://pymatting.github.io/foreground.html e implementação do rembg em https://github.com/danielgatis/rembg/blob/main/rembg/bg.py .

Avaliar `estimate_foreground_ml` ou alternativa fundamentada. Não aplicar uma correção RGB global arbitrária. Investigar preservação de mechas claras, pele, brilhos, transparência, premultiplicação e intensidade de correção. Medir RAM/tempo em 3K–5K e compatibilidade com Python 3.13 do Colab. Não copiar o pipeline inteiro do rembg se apenas uma etapa for necessária.

### D. Logo com fundo uniforme

Para preto sobre branco, investigar solução determinística que preserve cobertura das bordas, elementos desconectados e furos internos. Explicar quando remover todos os pixels brancos seria errado: o próprio desenho pode conter branco legítimo. Comparar remoção por cor, conectividade com a borda e seleção guiada, usando escolhas explícitas do operador.

### E. Correção manual eficiente

Pincéis Manter/Remover e marcação de região incerta podem corrigir trimap/seleção sem reconstruir pixels que ainda existem na original. Investigar como operadores de Photoshop usam Select and Mask, Refine Hair, Object/Color Aware e Decontaminate Colors. Não afirmar acesso ao código interno do Photoshop.

## Outros motores a considerar, somente com justificativa

- Outros checkpoints de matting ViTMatte e modelos especializados de retrato.
- Segmentação guiada da família SAM combinada a matting, sem tratar máscara binária como cabelo pronto.
- BRIA RMBG ou outros modelos comerciais/abertos: conferir licença e permissão de uso antes da recomendação.
- MODNet como referência leve para retratos, caso tecnicamente adequado.
- SDMatte e refinadores recentes: conferir projeto real, entrada exigida, pesos disponíveis e consumo medido antes de sugerir para T4.
- Pipelines como Matte-Anything: https://github.com/hustvl/Matte-Anything . Investigar o papel de segmentação, trimap e matting; não copiar toda a aplicação.

Não instalar seis modelos novos só para compor um catálogo. Selecionar os candidatos que resolvam defeitos demonstrados nos arquivos enviados.

## Como pesquisar e o que verificar

1. Ler model cards, código de inferência, processors, issues e discussões dos autores. Buscar relatos de cabelo, halo branco, logos, fundo parecido, falso foreground e tiling.
2. Distinguir segmentação, alpha matting e foreground color estimation na resposta.
3. Verificar entrada/saída, normalização, precisão, resolução, redimensionamento, canal alfa e licenças do código e pesos.
4. Informar data/commit/checkpoint. Não recomendar bibliotecas com “pip install -U tudo” que substituam indiscriminadamente Torch/CUDA do Colab.
5. Usar fóruns como pistas e confrontar as conclusões com código/documentação primária. Alegação de marketing não equivale a benchmark independente.
6. Se Photoroom/Canva não publicarem seu método, dizer isso. Não inventar que usam BEN2, BiRefNet ou qualquer outro modelo.

## Plano de validação solicitado

Poucos testes dirigidos, não dezenas de combinações aleatórias. Para cada teste informar: defeito que tenta resolver, entradas, etapas, parâmetros, saída esperada, custo e critério de rejeição.

Conjunto: cabelo cacheado sobre branco; fios claros/escuros sobre fundo semelhante; retrato com objetos encostados; mãos/dedos; logo preta sobre branco; escudo com branco interno; arte com transparência prévia.

Buscar amostras públicas difíceis de benchmarks de matting com alpha de referência. Verificar acesso, licença e se é permitido redistribuir. Um PNG comercial aprovado é referência do operador, não necessariamente ground truth científico.

Quando houver alpha verdadeiro: SAD/MSE, gradient/connectivity e avaliação localizada na região incerta. Sem ground truth: inspeção da mesma região em vermelho, branco e preto; avaliação de objetos indevidos, fios removidos, halo e fidelidade RGB, sem inventar uma nota objetiva de qualidade.

Os PNGs individuais devem manter RGBA real. Fundo vermelho ou quadriculado pertence apenas à montagem de inspeção. Registrar configurações, tempo, pico de VRAM/RAM quando medido e falhas. Não estimar consumo como se tivesse sido medido.

## Arquivos para auditoria do projeto

- `colab/operations.py`: BEN2, BiRefNet, híbrido e preservação de alpha existente.
- `colab/segmentation_worker.py`: Finegrain, FeyNoBg, Lucida e ViTMatte.
- `colab/optional_models.py`: ambientes e dependências dos motores opcionais.
- `colab/cutout_benchmark.py`: presets e montagens.
- `colab/app.py`: tarefas, candidatos, aprovação, arquivos e memória.
- `apps/ui/src/main.tsx`, `BoxSelector.tsx`, `MaskEditor.tsx`: fluxo operacional.
- `SuperImagem_GPU_T4_Colab.ipynb`: instalação e início da sessão.

Problemas de exportação também estão em investigação: ZIP falhou no navegador com `TypeError: Failed to fetch`, enquanto a montagem PNG foi baixada. Não confundir falha de transferência com falha de recorte. A causa exata no proxy Colab ainda não foi comprovada.

## Entregáveis esperados da pesquisa

1. Resumo executivo: diagnóstico e pipeline recomendado para Pessoa/cabelo, Vetor/logo/estampa e Geral.
2. Tabela por motor: função, melhor caso segundo o autor, limitações, inferência correta, licença, manutenção e compatibilidade com a T4.
3. Análise do código atual: erros demonstráveis separados de hipóteses.
4. Três a cinco combinações justificadas, explicando matematicamente como máscaras e cores serão tratadas.
5. Parâmetros iniciais com fonte ou justificativa, sem chamar heurísticas de valores universais.
6. Plano de teste curto e reproduzível, com amostras e critérios de aprovação/rejeição.
7. Recomendação do que manter, corrigir, deixar experimental ou retirar do fluxo padrão.
8. Links primários para cada conclusão decisiva e indicação explícita do que não foi possível verificar.

Não entregar somente “use modelo X”. Precisamos entender por que a combinação deve funcionar e qual defeito específico ela corrige.
