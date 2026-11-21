# Sobre o modelo

O modelo é uma rede neural MLP (MLPClassifier, do scikit-learn) dentro de um Pipeline com pré-processamento: StandardScaler nas variáveis numéricas e OneHotEncoder nas categóricas (segmento_mercado e vaga_estacionamento). O Pipeline completo foi salvo com joblib, então recebe os dados legíveis e faz as transformações sozinho, iguais às do treino.

Como o modelo foi treinado: 80% das reservas foram para o treino e 20% para o teste, com divisão estratificada (mesma proporção de cancelamentos nos dois). Os hiperparâmetros (tamanho das camadas ocultas e regularização alpha) foram escolhidos com GridSearchCV e validação cruzada estratificada de 5 dobras, usando o F1 da classe cancelada como critério. O teste foi usado uma única vez, no final.

Por que o MLP? No trabalho anterior comparamos Regressão Logística, KNN e MLP. A Regressão Logística ficou com F1 de teste 0,631, o KNN com 0,683 e o MLP com 0,693, usando os 17 atributos. O MLP foi o melhor porque captura relações não lineares, como a sazonalidade do mês e a combinação entre segmento e antecedência. O KNN decorava o treino (F1 de treino 0,99).

O que mudou em relação ao trabalho anterior: retreinamos o MLP com 8 atributos e ampliamos a grade de regularização (alpha até 0,1), como sugerido nas melhorias do trabalho anterior. As métricas atuais do modelo estão nos fatos de desempenho do modelo.

Como o modelo decide: ele calcula a probabilidade de cancelamento. Se for 50% ou mais, a previsão é "cancelada"; abaixo de 50%, "não cancelada". Quanto mais perto de 50%, menos segura é a previsão.

Atributos mais importantes no trabalho anterior (importância por permutação, queda no F1 quando o atributo é embaralhado): antecedência (0,19), pedidos especiais (0,17), preço da diária (0,09) e segmento de mercado (0,07). Mês de chegada, tipo de quarto, plano de refeição e vaga de estacionamento vêm em seguida, com efeito menor.

# Como entender as métricas

- Acurácia: de cada 100 reservas, quantas o modelo classifica certo (canceladas e não canceladas).
- Precisão: quando o modelo diz que uma reserva vai ser cancelada, quantas vezes ele acerta.
- Recall (sensibilidade): de cada 100 reservas que realmente foram canceladas, quantas o modelo percebe. É a métrica em que o modelo é mais fraco: ele deixa passar parte dos cancelamentos.
- Especificidade: das reservas que foram mantidas, quantas o modelo reconhece como mantidas.
- F1: combina precisão e recall num único número; é a métrica principal porque as classes são desbalanceadas.
- AUC: mede se o modelo dá probabilidades maiores para as reservas canceladas do que para as mantidas. 0,5 é sorteio e 1 é perfeito.
- Explicação para um leigo: o modelo funciona como um gerente experiente que olhou milhares de reservas antigas e aprendeu padrões, por exemplo "reserva feita com muita antecedência pela internet e sem pedidos especiais costuma ser cancelada". Ele dá um palpite com uma porcentagem de chance, mas pode errar, como qualquer previsão.

# Limitações

- O modelo erra: ele perde parte dos cancelamentos reais (recall abaixo de 0,7) e às vezes dá falso alarme em reservas que seriam mantidas. Use a previsão como apoio, não como decisão final.
- Os dados são de um único hotel e só de 2017 e 2018. O comportamento dos hóspedes pode ser diferente em outros hotéis, outros países ou depois da pandemia.
- O modelo não conhece fatores externos, como clima, eventos na cidade, crises, política de reembolso ou preço dos concorrentes.
- Valores fora da faixa vista no treino (por exemplo, antecedência acima de 443 dias ou diária acima de 540 euros) geram previsões menos confiáveis (extrapolação). O assistente avisa quando isso acontece.
- Os segmentos Aviação (100 reservas) e Cortesia (345 reservas) têm poucos exemplos, então as previsões para eles são menos confiáveis.
- Os casos semelhantes que o assistente mostra vêm do mesmo dataset usado no treino. Por isso a comparação com casos reais é indicativa, não uma prova independente.
- O limiar de decisão é 50%. Um hotel poderia usar um limiar menor (por exemplo 40%) para detectar mais cancelamentos, aceitando mais falsos alarmes.
- O assistente não responde sobre outros hotéis nem dá conselhos de viagem, e não usa dados pessoais de hóspedes.
