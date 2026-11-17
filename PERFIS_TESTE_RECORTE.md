# Bateria de pesquisa: 8 perfis

Na aba Remover fundo, abra Bateria de testes e selecione Pesquisa: 8 perfis.
Todos recebem a mesma imagem, na resolução original, sem ampliar ou trocar a versão ativa.

| Perfil | Processamento | Pergunta do teste |
|---|---|---|
| P1 | FeyNoBg original | Qual é o controle sem pós-processamento? |
| P2 | FeyNoBg + PyMatting | Corrigir RGB na borda reduz o halo sem alterar alpha? |
| P3 | FeyNoBg + ViTMatte, raio 0,3% | Uma faixa estreita preserva melhor o recorte? |
| P4 | FeyNoBg + ViTMatte, raio 1,2% | Mais contexto incerto recupera fios ou introduz fundo? |
| P5 | P4 + PyMatting | A borda melhora ao separar alpha e cor? |
| P6 | Guia Fey + detalhe BiRefNet com contraste em banda local | É possível aproveitar o detalhe sem recuperar caixas distantes? |
| P7 | P6 + ViTMatte 0,6% + PyMatting | A combinação ajuda ou acumula defeitos? |
| P8 | Fey + ViTMatte inteiro, 0,6% | Os blocos estão prejudicando o refinamento? |

Parâmetros experimentais, sem alegação de superioridade. P8 aceita até 2,2 MP; acima disso registra erro explícito, sem reduzir a entrada ou cancelar os demais perfis. Para comparação com P8 use uma cópia da entrada com aproximadamente 1500 px no maior lado, mantendo o original intacto. Correção RGB limitada a ROI de 18 MP; falhas opcionais aparecem no relatório.

PyMatting 1.1.16 é instalado sob demanda em ambiente opcional. Estima o foreground com contexto do objeto; apenas pixels da faixa de transição e vizinhança imediata recebem RGB corrigido. O alpha permanece igual. Não recupera fios nem partes excluídas pela máscara. Na combinação, núcleos e fundo distante do guia são protegidos; objetos muito próximos ainda podem reaparecer.

Escolha fundo vermelho, preto ou branco para a montagem. Os PNGs individuais mantêm transparência real. Use Selecionar detalhe para mostrar a mesma área do cabelo em todos os quadros; a inferência continua na imagem inteira.

O ZIP contém montagem, PNGs e report.json, além das máscaras/trimaps produzidos nos subdiretórios diagnósticos. Perfis que falham são registrados com mensagem e duração; não são apresentados como recortes válidos. Download direto e links individuais evitam carregar o ZIP inteiro como blob no navegador; o proxy Colab ainda precisa de validação real.

Para fotos com caixas, compare cabelo, mãos, ausência das caixas e halos. Para escudos, compare preenchimentos internos, letras e contorno completo. Para logo preta sobre branco, estes oito perfis não substituem a futura remoção por cor/conectividade. SAM2 também não foi instalado nesta atualização.

Validação local: testes de trimap/fusão, isolamento da entrada, montagem, falhas parciais, ZIP, API e build. Qualidade dos modelos e consumo real de T4 precisam ser medidos no Colab.

Referências: https://huggingface.co/docs/transformers/main/en/model_doc/vitmatte e https://pymatting.github.io/foreground.html .
