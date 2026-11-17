# Checklist oficial — SuperImagem Studio Colab

Este documento controla o desenvolvimento do sistema completo. Marcar um item como concluído exige implementação, teste e resultado visual aprovado. Uma resposta HTTP sem erro não basta.

## 1. Regras confirmadas do produto

- [x] Manter o projeto completo separado do `superimagem-colab` simples.
- [x] Não impor ordem obrigatória entre as ferramentas.
- [x] Trabalhar sempre sobre a versão ativa selecionada.
- [x] Preservar a imagem original sem alterações.
- [x] Criar uma nova versão para cada operação.
- [x] Permitir baixar a versão ativa em qualquer etapa.
- [x] Permitir ramificações: escolher uma versão antiga e criar um novo caminho a partir dela.
- [x] Exibir claramente qual versão será usada como entrada antes de iniciar um trabalho.
- [ ] Confirmar operações que descartem versões ou reiniciem o trabalho.

Fluxos igualmente válidos:

- ampliar → baixar;
- ampliar → vetorizar;
- remover fundo → baixar;
- remover fundo → ampliar → vetorizar;
- limpar → remover fundo → ampliar → preto e branco → vetorizar.

## 2. Repositório, Colab e inicialização

- [x] Criar o repositório independente `diegogomespi/superimagem-studio-colab`.
- [x] Publicar notebook com uma única célula de inicialização.
- [x] Abrir a interface em página externa ao notebook.
- [x] Usar GPU NVIDIA T4 com CUDA e FP16.
- [x] Mostrar versão/commit carregado no inicializador.
- [x] Manter log externo completo de instalação e inicialização.
- [x] Validar download por tamanho e SHA-256 para os modelos atuais de upscale.
- [ ] Exibir uma confirmação individual para cada novo modelo instalado.
- [ ] Informar nome, tamanho, hash e estado de carregamento de todos os motores.
- [ ] Detectar ausência da T4 e explicar como ativá-la.
- [ ] Detectar sessão antiga do Colab e impedir interface desatualizada em cache.
- [ ] Criar diagnóstico automático de CUDA, VRAM, RAM e espaço disponível.
- [ ] Liberar memória CUDA entre motores pesados quando necessário.
- [ ] Documentar quais arquivos desaparecem ao encerrar a sessão do Colab.

## 3. Arquitetura de versões

- [x] Backend armazenar `parentId` em resultados de upscale.
- [x] Resultado concluído tornar-se automaticamente a versão ativa.
- [x] Selecionar uma versão anterior.
- [x] Voltar uma etapa.
- [x] Restaurar a original sem apagar o histórico.
- [x] Download da versão ativa sem depender do último trabalho.
- [ ] Validar todo o histórico numa sessão real do Colab.
- [ ] Guardar operação, motor, parâmetros, dimensões e horário em cada versão.
- [ ] Mostrar nomes curtos: Original, V2, V3 e assim por diante.
- [x] Mostrar descrição: `Upscale 4× · Foto`, `Fundo removido · BEN2`, etc.
- [ ] Permitir renomear versões.
- [x] Excluir somente uma versão sem apagar outras por acidente.
- [x] Botão `Desfazer` que apenas seleciona o pai da versão ativa.
- [ ] Botão `Refazer` enquanto o caminho ainda existir.
- [x] Árvore visual compacta quando houver ramificações.
- [ ] Manter os artefatos somente na sessão do Colab nesta primeira fase.

## 4. Interface modular

- [x] Criar barra de ferramentas compacta no topo ou na lateral.
- [x] Aba `Ampliar`.
- [x] Aba `Remover fundo`.
- [x] Aba `Limpar`.
- [x] Aba `Vetorizar`.
- [x] Aba `Histórico`.
- [x] Trocar apenas o painel de controles; manter a imagem no mesmo visualizador.
- [x] Destacar a ferramenta aberta sem transformar o sistema em uma sequência.
- [x] Manter todos os controles essenciais dentro da altura da tela em desktop.
- [ ] Layout responsivo de uma coluna em telas estreitas.
- [x] Estado vazio simples para upload por clique ou arrastar.
- [ ] Desabilitar somente os controles incompatíveis, explicando o motivo.
- [x] Mostrar progresso, etapa atual, porcentagem e botão Cancelar.
- [ ] Mostrar erros em português com causa provável e ação recomendada.

## 5. Visualizador e comparador

- [x] Preview inicial leve para não travar a página.
- [x] Preview de inspeção em WebP de até 4096 px.
- [x] Download separado em resolução completa.
- [x] Zoom pela roda do mouse ancorado no ponteiro.
- [x] Arrastar imagem ampliada.
- [x] Comparador com divisor vertical arrastável pela linha inteira.
- [x] `Ctrl` ativa a mão no modo Comparar.
- [x] Comparar sempre a entrada usada pela operação com seu resultado direto.
- [ ] Preservar zoom e posição ao alternar Antes, Resultado e Comparar.
- [ ] Sincronizar exatamente enquadramento e coordenadas dos dois lados.
- [x] Fundos de visualização quadriculado, branco e preto sem alterar o arquivo.
- [ ] Identificar no comparador o motor e a operação de cada lado.
- [ ] Permitir selecionar manualmente quaisquer duas versões para comparar.
- [ ] Testar imagens verticais, horizontais, quadradas e muito grandes.

## 6. Upscale 2×, 4× e 8×

- [x] Real-ESRGAN x4plus para Foto/geral.
- [x] Real-ESRGAN x4plus Anime 6B para Ilustração/estampa.
- [x] Processamento CUDA FP16 em blocos com sobreposição.
- [x] Redução automática do bloco quando faltar VRAM.
- [x] Preservar transparência ampliando RGB e alpha separadamente.
- [x] Opções 2×, 4× e 8×.
- [x] Modos 8× Fiel e Detalhado.
- [ ] Reproduzir e diagnosticar a falta de diferença do modelo Foto/geral.
- [ ] Confirmar no log qual peso foi realmente carregado.
- [ ] Confirmar que Foto/geral e Ilustração/estampa geram resultados diferentes.
- [ ] Confirmar que Fiel e Detalhado executam pipelines diferentes.
- [ ] Validar 8× Foto/geral em recortes reais a 100%, 400% e 800%.
- [ ] Registrar dimensões, hash, tempo, pico de RAM e VRAM.
- [ ] Testar texto pequeno, pele, olhos, cabelos, pelos, tecido e degradês.
- [ ] Evitar duas passagens de IA em letras quando o modo Fiel estiver selecionado.
- [ ] Impedir dimensões que excedam limites do formato selecionado.

## 7. Remoção automática de fundo

- [x] Integrar BiRefNet HR-Matting oficial com CUDA.
- [x] Integrar BEN2 disponível para teste pessoal.
- [ ] Baixar, validar e registrar os pesos de cada motor.
- [x] Carregar modelos pesados somente quando a ferramenta for usada.
- [ ] Executar comparação A/B com as mesmas imagens.
- [ ] Escolher o motor padrão por resultado real, não por propaganda.
- [ ] Criar opção `Automático HQ`.
- [ ] Criar opção especializada `Fundo uniforme`.
- [ ] Preservar cabelos, pelos, dedos, objetos finos e furos internos.
- [ ] Preservar transparência parcial em véu, renda, fumaça e vidro quando possível.
- [ ] Não apagar partes legítimas de cor semelhante ao fundo.
- [x] Gerar PNG RGBA sem criar fundo branco.
- [ ] Mostrar a máscara/alpha para inspeção opcional.
- [ ] Testar retrato, corpo inteiro, duas pessoas, produtos, estampas e ilustrações.

## 8. Refinamento de máscara e bordas

- [ ] Criar faixa de incerteza/trimap a partir da máscara inicial.
- [ ] Avaliar ViTMatte somente na região incerta.
- [ ] Comparar refinamento interno do BEN2 com BiRefNet + ViTMatte.
- [ ] Remover halo branco, preto e contaminação da cor do fundo.
- [ ] Estender cores internas para pixels transparentes sem criar franja.
- [ ] Preencher pequenos buracos incorretos.
- [ ] Remover componentes/pontos soltos do fundo.
- [ ] Controles de borda `Suave`, `Equilibrada` e `Firme`.
- [ ] Controle fino para expandir ou contrair a máscara.
- [ ] Não endurecer cabelos e transparências sem necessidade.
- [ ] Avaliar SAM somente como correção guiada, não como removedor automático principal.
- [ ] Adicionar pincel Manter/Remover numa fase posterior.

## 9. Limpar e preparar imagem

- [x] Ferramenta funcionar de forma independente, sem exigir vetorização.
- [x] Remover ruído e granulação preservando bordas.
- [ ] Reduzir artefatos de compressão JPEG.
- [ ] Limpar pontos coloridos isolados.
- [ ] Suavizar degradês sem criar faixas fortes.
- [x] Limpar canal alpha e pixels quase transparentes indesejados.
- [ ] Remover halos na transparência.
- [ ] Opções `Foto`, `Ilustração` e `Preparar para vetor`.
- [ ] Intensidades `Leve`, `Equilibrada` e `Forte`.
- [ ] Preview antes/depois sem aplicar automaticamente.
- [x] Permitir baixar o raster limpo em PNG ou WebP.
- [x] Preservar cores e transparência quando a preparação P&B não estiver ativada.

## 10. Preparação preto e branco

- [x] Botão `Transformar em preto e branco` independente da vetorização.
- [x] Conversão correta para tons de cinza antes do limiar.
- [ ] Ajuste automático de níveis e contraste.
- [x] Limiar automático por Otsu.
- [x] Limiar adaptativo para iluminação irregular.
- [x] Controle manual do limite preto/branco.
- [ ] Fundo branco no modo de preparação binária.
- [ ] Opção Inverter.
- [ ] Engrossar linhas.
- [ ] Afinar linhas.
- [x] Remover pontos pequenos.
- [ ] Preencher falhas pequenas.
- [ ] Preservar cantos e letras.
- [x] Mostrar resultado P&B antes de vetorizar.
- [x] Permitir baixar somente o PNG P&B preparado.

## 11. Vetorização preto e branco

- [x] Integrar VTracer no modo binário/adaptativo.
- [x] Integrar Potrace para comparação clássica de curvas.
- [x] Usar como entrada a versão ativa ou a versão P&B recém-preparada.
- [ ] Comparar fidelidade, quantidade de nós e tamanho do arquivo.
- [ ] Selecionar automaticamente o melhor preset inicial.
- [ ] Controles de suavidade, cantos e remoção de manchas.
- [x] Preview SVG real, sem incorporar o bitmap dentro do arquivo.
- [ ] Verificar abertura e edição no CorelDRAW.
- [x] Exportar SVG.
- [ ] Avaliar exportação PDF, EPS e DXF com UniConvertor.

## 12. Vetorização colorida

- [x] Integrar VTracer colorido em versão fixada e reproduzível.
- [x] Reduzir paleta antes do traçado.
- [ ] Opções de 2, 4, 8, 16 e 32 cores.
- [x] Paleta automática.
- [ ] Avaliar paleta personalizada posteriormente.
- [ ] Remover formas e manchas menores que o limite escolhido.
- [ ] Preservar transparência e ignorar o fundo transparente.
- [ ] Modos para curvas suaves e cantos firmes.
- [x] Usar mosaico/cutout sem frestas quando comprovadamente melhor.
- [ ] Mostrar número aproximado de formas e tamanho do SVG.
- [ ] Evitar SVG enorme e impraticável em fotografias.
- [ ] Avisar quando a imagem não for adequada à vetorização colorida.
- [ ] Verificar abertura, agrupamento e edição no CorelDRAW.

## 13. Upscale antes da vetorização

- [ ] Oferecer preparação 2× ou 4× Fiel para imagens pequenas e serrilhadas.
- [ ] Não usar 8× Detalhado automaticamente para vetor.
- [ ] Comparar original, 2× Fiel e 4× Fiel antes do VTracer.
- [ ] Medir quantidade de nós, tamanho do SVG e fidelidade renderizada.
- [ ] Evitar detalhes inventados que prejudiquem letras e contornos.
- [ ] Recomendar automaticamente o caminho mais simples quando o original já for suficiente.

## 14. Exportação e nomes de arquivo

- [x] PNG sem perdas com transparência.
- [x] WebP Q95 com transparência dentro do limite de dimensões.
- [x] JPEG Q95 sempre com fundo branco.
- [x] SVG verdadeiro para vetores.
- [ ] PDF/EPS/DXF após validação.
- [x] Manter o nome original como prefixo.
- [x] Acrescentar operação, escala, motor e modo ao nome.
- [ ] Exemplos: `_upscale-4x_foto-fiel`, `_sem-fundo_birefnet-hr`, `_vetor-pb_potrace`.
- [ ] Evitar repetir sufixos desnecessários em cadeias longas.
- [ ] Guardar metadados da cadeia de operações separadamente.

## 15. Desempenho e estabilidade

- [x] Um trabalho pesado por vez.
- [x] Cancelamento solicitado pelo usuário.
- [ ] Cancelamento liberar efetivamente RAM e VRAM.
- [x] Não iniciar trabalho duplicado ao clicar duas vezes.
- [x] Não gerar WebP pesado antes de o usuário solicitar.
- [x] Preview nunca carregar o PNG final gigante no navegador.
- [ ] Limitar memória de CPU durante codificação de arquivos grandes.
- [ ] Usar tiles também nos novos motores quando necessário.
- [ ] Recuperar-se de CUDA Out Of Memory reduzindo bloco ou resolução de inferência.
- [ ] Informar quando uma saída 8× for grande demais para a sessão.
- [ ] Remover artefatos temporários órfãos depois de cancelamentos.

## 16. Logs e confirmação de funcionamento

- [ ] Logar operação, versão de entrada e versão criada.
- [ ] Logar modelo, arquivo do peso e hash.
- [ ] Logar dispositivo CUDA e precisão utilizada.
- [ ] Logar resolução de entrada e saída.
- [ ] Logar tempo total e por etapa.
- [ ] Logar pico aproximado de RAM e VRAM.
- [x] Mostrar confirmação visual `Concluído` somente depois de validar o arquivo gerado.
- [x] Verificar se PNG/SVG abre antes de liberar o download.
- [ ] Disponibilizar botão para copiar o log de erro.
- [ ] Evitar mensagens genéricas como somente `HTTP 404` ou `Falha`.

## 17. Conjunto de testes visuais

- [ ] Fotografias com cabelo claro e escuro.
- [ ] Animais com pelos.
- [ ] Corpo inteiro, mãos e dedos.
- [ ] Duas ou mais pessoas.
- [ ] Produtos com reflexo, vidro e furos.
- [ ] Véu, renda e tecido transparente.
- [ ] Logotipos, letras pequenas e linhas finas.
- [ ] Estampas coloridas e cores chapadas.
- [ ] PNG já transparente.
- [ ] Fundo branco, preto, uniforme e complexo.
- [ ] Imagens pequenas, grandes, verticais e horizontais.
- [ ] Comparação visual em branco, preto e quadriculado.
- [ ] Abertura dos vetores no navegador, Inkscape e CorelDRAW.

## 18. Critério para considerar uma fase pronta

- [x] Código compilado e sem erro de sintaxe.
- [x] Testes automatizados relevantes aprovados.
- [ ] Teste real concluído numa sessão T4 nova.
- [ ] Arquivos dos modelos baixados e validados.
- [ ] Resultado visual comparado com a entrada.
- [ ] Download aberto e conferido na resolução completa.
- [ ] Interface testada em tela larga e estreita.
- [ ] Documentação e checklist atualizados.
- [ ] Commit publicado no GitHub.

## Ordem de desenvolvimento — não é ordem de uso

Esta lista organiza somente a construção do software. O usuário continuará livre para usar as ferramentas em qualquer ordem.

1. Validar histórico e versões ativas no Colab.
2. Corrigir e comprovar o upscale Foto/geral.
3. Criar a navegação modular das ferramentas.
4. Integrar e comparar BiRefNet HR-Matting e BEN2.
5. Implementar limpeza de alpha, bordas e ruído.
6. Implementar preparação preto e branco.
7. Integrar vetorização P&B.
8. Integrar vetorização colorida.
9. Finalizar exportações, testes visuais e documentação.
