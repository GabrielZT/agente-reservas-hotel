# Limpeza dos dados

- Valores faltantes: o dataset não tem valores faltantes, então não foi preciso imputar nada.
- Duplicatas: 10.275 reservas (28,3% da base original) tinham todos os atributos iguais a outra reserva, mudando só o identificador. Provavelmente são reservas de grupo feitas por agências. Mantivemos só uma cópia de cada, porque cópias idênticas no treino e no teste inflariam as métricas, principalmente as do KNN.
- Efeito da remoção de duplicatas: a taxa de cancelamento caiu de cerca de 33% para cerca de 29% na base toda, e no segmento Offline caiu de cerca de 30% para cerca de 12%.
- Datas inexistentes: 35 reservas (depois da remoção de duplicatas) tinham chegada em 29/02/2018, data que não existe porque 2018 não é bissexto. Foram removidas.
- Reservas sem noites: 75 reservas tinham zero noites (nem de semana nem de fim de semana). Uma hospedagem sem diária é inconsistente, então foram removidas.
- Diária igual a zero: aparece principalmente em reservas de Cortesia (gratuitas), então é um valor legítimo e foi mantido.
- Valores extremos, como antecedência de 443 dias ou 10 crianças, são raros mas plausíveis, e foram mantidos. A padronização (StandardScaler) controla o efeito de escala.

# Principais achados da análise exploratória

- Balanceamento: a base é moderadamente desbalanceada. Cerca de 71% das reservas não são canceladas e 29% são. Um modelo que sempre dissesse "não cancela" já acertaria cerca de 71%, por isso a acurácia sozinha engana.
- Antecedência é a variável que mais separa as classes: quanto mais cedo a reserva é feita, maior a chance de cancelamento. Reservas feitas com muita antecedência dão mais tempo para os planos mudarem. A mediana da antecedência das canceladas é bem maior que a das não canceladas.
- Pedidos especiais reduzem o cancelamento: quem faz pedidos especiais está mais comprometido com a viagem. Reservas com 3 ou mais pedidos especiais praticamente nunca foram canceladas.
- Preço: as reservas canceladas têm diária média um pouco maior que as não canceladas.
- Segmento de mercado é a variável categórica mais importante: Online é o segmento mais comum e o que mais cancela; Offline e Corporativo cancelam bem menos; Cortesia nunca cancelou.
- Vaga de estacionamento: quem pede vaga cancela bem menos, provavelmente porque já planejou a viagem de carro.
- Hóspede recorrente: quem já se hospedou no hotel quase nunca cancela.
- Sazonalidade: os meses de meio de ano (julho e agosto) têm as maiores taxas de cancelamento, e janeiro e dezembro, as menores. Esse efeito não é linear, por isso a correlação do mês com o cancelamento é perto de zero, mas o modelo consegue usá-lo.
- Correlações: nenhuma variável sozinha tem correlação alta com o cancelamento (todas abaixo de 0,4 em módulo). A maior é a da antecedência (positiva) e a segunda é a dos pedidos especiais (negativa). Por isso é preciso combinar atributos, e um modelo não linear funciona melhor.
- Plano de refeição e tipo de quarto: Meia pensão e o quarto Tipo 6 têm as maiores taxas de cancelamento, mas as diferenças são menores que as de segmento.
