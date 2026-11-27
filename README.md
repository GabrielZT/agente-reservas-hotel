# 🏨 Assistente de Reservas — agente com RAG + modelo de ML

Trabalho Final de Inteligência Artificial (UNOESC 2026/2) — **Gabriel Turri e Luis Ferronato Kohn**.

Agente conversacional, escrito em Python, que conversa sobre o dataset **Hotel Reservations** (Kaggle, ahsan81) do Trabalho Prático anterior e **prevê se uma reserva será cancelada** com o nosso modelo (MLP). O Gemini (`gemini-3.5-flash-lite`, plano gratuito) só interpreta a pergunta e redige a resposta; buscar os dados, rodar o modelo e conferir os números é código nosso.

- App: https://agente-reservas-hotel.onrender.com (rota `/saude` para acordar o servidor)
- Repositório: https://github.com/GabrielZT/agente-reservas-hotel
- Base: kit inicial da disciplina (o README original está em `docs/README_KIT_ORIGINAL.md`)

## Pipeline

```
mensagem → 1 INTERPRETAR (Gemini → JSON: intenção, atributos, consulta + filtros, busca)
         → 2 RECUPERAR   (RAG TF-IDF palavras + letras + sinônimos; estatísticas exatas com pandas e filtros)
         → 3 PREVER      (falta dado? pergunta · valida: erro x alerta · para_o_modelo() · MLP .joblib
                          com probabilidade · 15 reservas reais semelhantes)
         → 4 GERAR       (Gemini, só com o CONTEXTO das etapas 2 e 3)
         → 5 VERIFICAR   (todo número da resposta está no CONTEXTO? senão refaz 1 vez ou avisa)
```

Sem function calling automático e sem API de agentes: o `src/pipeline.py` decide cada etapa e chama `generate_content` diretamente.

## Estrutura

```
agente-reservas-hotel/
├── src/                código principal do agente
│   ├── servidor.py         servidor Flask + rota /api/chat (Render: gunicorn src.servidor:app --bind 0.0.0.0:10000)
│   ├── pipeline.py         pipeline das 5 etapas (interpretar, recuperar, prever, gerar, verificar)
│   ├── dados_modelo.py     domínio do hotel, ponte para_o_modelo(), validação, previsão, casos semelhantes, estatísticas, fatos do RAG
│   └── busca_rag.py        busca na base de conhecimento (TF-IDF + sinônimos)
├── scripts/
│   ├── treinar.py          limpeza + treino do modelo (gera data/dataset_legivel.csv e models/)
│   └── conferir.py         gabarito: estatísticas e previsões a partir do CSV e do modelo
├── tests/
│   ├── teste_offline.py    testes sem gastar cota (ponte, validação, erros de domínio, linguagem natural, filtros, RAG, verificação)
│   ├── avaliacao_rag.py    qualidade da busca com 20 perguntas de teste
│   ├── bateria.py          as 12 categorias da bateria da apresentação, com gabarito do nosso código
│   └── comparacao_rag.py   mesmas perguntas com e sem RAG (relatório e demonstração)
├── docs/
│   ├── links.txt           arquivo de entrega (app, GitHub, hash do commit)
│   ├── registro_testes.md  registro avaliado da bateria de testes
│   └── README_KIT_ORIGINAL.md
├── prompts/            instruções das etapas 1 (interpretar.md) e 4 (responder.md)
├── knowledge_base/     base de conhecimento do RAG: análise do trabalho anterior
├── data/               dataset original do Kaggle e dataset limpo e legível
├── models/             modelo.joblib, metricas.json, amostra_conferencia.json, resultados_gridsearch.csv, metricas_teste.csv
├── static/             página de chat (index.html)
└── streamlit_app/      versão alternativa da interface em Streamlit (não publicada)
```

## Modelo

- Classificação binária: `cancelada` = `sim` / `nao` (classe positiva: `sim`).
- 8 atributos (o enunciado permite até 8): `antecedencia_dias`, `preco_medio_diaria`, `pedidos_especiais`, `mes_chegada`, `noites_fim_de_semana`, `noites_semana`, `segmento_mercado`, `vaga_estacionamento`. Na validação cruzada, o MLP com esses 8 teve F1 0,7020 contra 0,7028 com os 17 originais.
- Hiperparâmetros por GridSearchCV estratificado de 5 dobras (21 combinações: 3 arquiteturas × 7 valores de alpha, de 0,001 a 3,0); melhor: alpha 0,1 com camadas ocultas (100, 50).
- Limiar de decisão escolhido por validação cruzada nos dados de treino (`TunedThresholdClassifierCV`, maximizando o F1): 35,4% em vez dos 50% padrão. A previsão usa esse limiar, e as métricas de teste em `models/metricas.json` já o consideram.
- Métricas de teste reais em `models/metricas.json` (acurácia, F1, precisão, recall, especificidade, AUC).

## Como rodar

```bash
python -m venv .venv            # Python 3.13
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/treinar.py       # opcional: o modelo já está em models/
python tests/teste_offline.py
python tests/avaliacao_rag.py
python tests/bateria.py local    # gasta ~30 chamadas do Gemini
python scripts/conferir.py
copy .env.example .env          # e colem a GEMINI_API_KEY
python src/servidor.py          # http://localhost:5000
```

Deploy no Render: build `pip install -r requirements.txt`, start `gunicorn src.servidor:app --bind 0.0.0.0:10000 --workers 1 --threads 4 --timeout 120` (o `--bind` garante a porta que o Render procura), variáveis `GEMINI_API_KEY` e `PYTHON_VERSION=3.13.15`. A chave **nunca** vai para o código nem para o GitHub (`.env` está no `.gitignore`).

> Chave do Gemini: criada no Google AI Studio com a conta UNOESC, no plano gratuito, para o modelo `gemini-3.5-flash-lite`.
> Auto-Deploy desligado no Render: depois de cada push é preciso publicar manualmente (**Manual Deploy → Deploy latest commit**).

## O que mudamos em relação ao kit

1. **Domínio e modelo próprios:** dataset Hotel Reservations com a mesma limpeza do trabalho anterior; colunas e categorias traduzidas para o português no dataset legível; modelo retreinado com 8 atributos escolhidos por validação cruzada; grade do MLP ampliada (alpha de 0,001 até 3,0; o melhor continuou sendo 0,1), como sugerido nas melhorias do trabalho anterior e no feedback do professor. O F1 de teste subiu de 0,693 para 0,715 (limiar de 50%) e para 0,723 com o limiar escolhido por validação cruzada.
2. **Consultas com filtros:** a etapa 1 devolve também `filtros` (`coluna`, `operador`, `valor`), e `estatisticas()` aplica as condições antes de calcular. Ex.: "quantas reservas Online com mais de 100 dias de antecedência foram canceladas?". As consultas usam **todas** as colunas do dataset, não só as do modelo, e trazem maior e menor taxa já calculadas (o Gemini não precisa comparar números).
3. **Entrada em linguagem natural:** "2 semanas" → 14 dias, "3 meses" → 90, "julho" → 7, "duas" → 2, "nenhum", "sem" ou "não" → 0 (só quando não há outro número na frase), "1 milhão" → 1.000.000, "não precisa de vaga" → `nao`, "empresa" → Corporativo.
4. **Regras de domínio do hotel:** antecedência negativa, mês fora de 1–12, estadia sem noites, diária negativa ou acima de 5.000 € viram **erro**; valores fora da faixa do treino viram **alerta**; diária 0 fora do segmento Cortesia gera alerta.
5. **Previsão mais explicada:** probabilidade de cancelamento em %, grau de confiança (perto do limiar de decisão = baixa), recall e precisão de teste junto da previsão; casos semelhantes com 15 reservas reais do mesmo segmento (e mesma vaga), percentual cancelado entre elas e taxa geral do dataset para comparação.
6. **RAG melhor, e medido:** base dividida em 3 arquivos com trechos curtos; mais fatos gerados do CSV (taxa por segmento, plano, quarto, vaga, recorrência, faixa de antecedência, pedidos especiais, mês com nome, erros do modelo, comparação com o palpite ingênuo); busca combinando TF-IDF de palavras e de pedaços de letras, com expansão por sinônimos do domínio. Com `tests/avaliacao_rag.py` (20 perguntas de teste), o acerto nos 5 primeiros trechos subiu de 75% (busca do kit) para 95%, e as 3 perguntas fora do escopo não trazem nenhum trecho.
7. **Página:** exemplos clicáveis, aviso de "servidor acordando" quando a resposta demora, layout para celular.
8. **Testes próprios:** `tests/teste_offline.py` cobre casos impossíveis do domínio, linguagem natural e filtros, com `assert`.
9. **Bateria automatizada:** `tests/bateria.py` roda as 12 categorias da apresentação (no Render ou local), calcula o gabarito com o nosso código e salva um registro datado em `docs/` (o avaliado para o relatório é `docs/registro_testes.md`).
10. **Verificação mais transparente e mais barata:** quando a resposta é refeita, a linha 🔧 mostra quais números não foram conferidos. Com isso descobrimos que o Gemini fazia contas ("deixa passar 32 de cada 100" = 100 − 68). Passamos a entregar essas contas prontas no contexto (`metricas_explicadas()`) e incluímos o prompt da persona entre os textos confiáveis (ele cita "2017 e 2018"). Na bateria, as respostas refeitas caíram de 5 em 10 respostas geradas para 0 em 12, economizando chamadas da cota.
11. **"Essa previsão faz sentido?":** a etapa 1 passou a repassar os dados da reserva em qualquer intenção, então o código roda de novo o modelo e os casos semelhantes nessa pergunta (antes ele só repetia os números da resposta anterior).
12. **Grade de alpha ampliada (feedback):** no trabalho anterior o melhor alpha (0,1) ficava na borda da grade; agora ela vai de 0,001 a 3,0 (21 combinações) e o melhor continuou sendo 0,1, o que confirma que não estava limitado pela grade.
13. **Tabelas exportadas em CSV (feedback):** `scripts/treinar.py` salva `models/resultados_gridsearch.csv` (todas as combinações da grade) e `models/metricas_teste.csv` (métricas de teste com 50% e com o limiar escolhido) na mesma execução que gera o modelo, e as tabelas do relatório saem desses arquivos, para os números baterem com a execução final.
14. **Limiar escolhido por validação cruzada (feedback):** o limiar de decisão deixou de ser 50% fixo e passou a ser escolhido com `TunedThresholdClassifierCV` só com os dados de treino (35,4%). O agente (`prever()`), a base de conhecimento e as métricas usam esse limiar; `metricas.json` guarda também todas as métricas com 50% para comparação, e `models/metricas_teste.csv` põe os dois limiares lado a lado (F1 0,715 e recall 0,681 contra F1 0,723 e recall 0,769). O custo é a precisão cair de 0,753 para 0,683, com mais falsos alarmes (de 332 para 532); em troca, o modelo detecta 1.144 dos 1.487 cancelamentos do teste, contra 1.013 antes.
