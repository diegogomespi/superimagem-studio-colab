# SuperImagem GPU T4

Esta primeira versão executa somente ampliação de imagem em GPU CUDA.

## Como abrir

1. Acesse `https://colab.research.google.com/`.
2. Envie e abra `SuperImagem_GPU_T4_Colab.ipynb`.
3. Em **Ambiente de execução > Alterar o tipo de ambiente**, selecione **GPU T4**.
4. Execute a única célula **INICIAR SUPERIMAGEM GPU T4**.
5. Clique em **ABRIR SUPERIMAGEM EM NOVA ABA**. A interface funciona em uma página separada.
5. O sistema atualizado é baixado automaticamente do GitHub.
6. A interface aparece dentro do próprio notebook.

## Modos

- **2×**: modelo 4× com redução final de alta qualidade.
- **4×**: escala nativa dos modelos.
- **8× fiel**: uma inferência 4× e redimensionamento final conservador.
- **8× detalhado**: duas inferências; é mais lento e pode alterar detalhes pequenos.
- **Foto / geral**: `RealESRGAN_x4plus`, rede completa de 23 blocos.
- **Ilustração / estampa**: `RealESRGAN_x4plus_anime_6B`, rede de 6 blocos.

## Transparência

O RGB passa pela IA, enquanto o canal alpha é ampliado separadamente com Lanczos para preservar a transparência sem criar fundo branco. As cores das bordas transparentes são estendidas antes da IA para reduzir halos. Os botões quadriculado, branco e preto alteram apenas a visualização. JPEG não suporta transparência e é sempre exportado com fundo branco.

## Limites

A sessão do Colab é temporária. Imagens e resultados são apagados quando o ambiente é encerrado. O primeiro processamento baixa o modelo oficial escolhido e por isso demora um pouco mais.
