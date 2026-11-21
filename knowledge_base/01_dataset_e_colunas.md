# Sobre o dataset

O dataset é o Hotel Reservations Dataset, publicado no Kaggle (autor ahsan81). Cada linha é uma reserva de um único hotel, com chegadas em 2017 e 2018. O objetivo do trabalho é prever se uma reserva será cancelada (coluna cancelada = sim) ou não (cancelada = nao).

O dataset original tem 36.275 reservas e 19 colunas. Depois da limpeza (remoção de duplicatas, datas inexistentes e reservas sem noites) ficaram 25.890 reservas. Todas as estatísticas deste assistente usam a base limpa.

O trabalho foi feito por Gabriel Turri e Luis Ferronato Kohn, na disciplina de Inteligência Artificial da UNOESC. O agente é a continuação do Trabalho Prático anterior (classificação com Regressão Logística, KNN e MLP).

# Descrição das colunas

- cancelada: alvo da previsão. "sim" se a reserva foi cancelada, "nao" se foi mantida (no original, booking_status = Canceled ou Not_Canceled).
- antecedencia_dias: antecedência da reserva, ou seja, quantos dias se passaram entre o dia em que a reserva foi feita e o dia da chegada (no original, lead_time).
- preco_medio_diaria: preço médio da diária do quarto, em euros. Os preços são dinâmicos (mudam conforme a data). Diária 0 aparece em reservas de cortesia (no original, avg_price_per_room).
- pedidos_especiais: número de pedidos especiais do hóspede, como andar alto, vista, berço ou cama extra (no original, no_of_special_requests).
- segmento_mercado: canal de origem da reserva. Online = site ou aplicativo; Offline = agência de viagens ou telefone; Corporativo = empresa; Aviação = tripulação de companhia aérea; Cortesia = reserva gratuita (no original, market_segment_type: Online, Offline, Corporate, Aviation, Complementary).
- vaga_estacionamento: "sim" se o hóspede pediu vaga de estacionamento, "nao" caso contrário (no original, required_car_parking_space 1/0).
- mes_chegada: mês da chegada, de 1 (janeiro) a 12 (dezembro) (no original, arrival_month).
- noites_fim_de_semana: noites de sábado ou domingo da estadia (no original, no_of_weekend_nights).
- noites_semana: noites de segunda a sexta da estadia (no original, no_of_week_nights).
- total_noites: soma das noites de fim de semana e de semana (coluna criada para a análise).
- faixa_antecedencia: antecedência agrupada em faixas (0-7 dias, 8-30 dias, 31-90 dias, 91-180 dias, mais de 180 dias), criada para a análise.
- adultos e criancas: número de adultos e de crianças na reserva (no original, no_of_adults e no_of_children).
- plano_refeicao: plano de refeição. Café da manhã (Meal Plan 1), Meia pensão (Meal Plan 2: café e mais uma refeição), Pensão completa (Meal Plan 3) ou Nenhum (Not Selected).
- tipo_quarto: tipo de quarto reservado, de Tipo 1 a Tipo 7. O hotel não divulga o que cada tipo significa (no original, room_type_reserved).
- ano_chegada e dia_chegada: ano (2017 ou 2018) e dia do mês da chegada.
- hospede_recorrente: "sim" se o hóspede já tinha se hospedado no hotel antes (no original, repeated_guest 1/0).
- cancelamentos_anteriores e reservas_anteriores_nao_canceladas: histórico de reservas do hóspede no hotel.

# Atributos usados pelo modelo

O modelo usa 8 atributos: antecedencia_dias, preco_medio_diaria, pedidos_especiais, mes_chegada, noites_fim_de_semana, noites_semana, segmento_mercado e vaga_estacionamento. Para fazer uma previsão, o assistente precisa dos 8.

Por que só 8 atributos? O enunciado permite retreinar com até 8. Escolhemos os mais importantes da análise do trabalho anterior (importância por permutação e correlação) e comparamos subconjuntos na validação cruzada: com esses 8, o F1 do MLP foi praticamente o mesmo que com os 17 atributos originais (0,7020 contra 0,7028). Assim o modelo fica mais simples de usar numa conversa sem perder desempenho.

Por que o ano de chegada ficou de fora? O dataset só tem 2017 e 2018. Usar o ano faria o modelo depender de um período que não se repete, e não faria sentido para reservas futuras.
