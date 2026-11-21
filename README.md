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

Sem function calling automático e sem API de agentes: o `agente.py` decide cada etapa e chama `generate_content` diretamente.

## Estrutura

| Arquivo | O que faz |
|---|---|
| `treinar.py` | limpeza (igual ao trabalho anterior), gera `dados/dataset_legivel.csv` e treina o Pipeline (StandardScaler + OneHotEncoder + MLP) com GridSearchCV; salva `modelo/` |
| `ferramentas.py` | constantes do domínio, ponte `para_o_modelo()`, validação, previsão, casos semelhantes, estatísticas com filtros, fatos para o RAG |
| `agente.py` | pipeline das 5 etapas |
| `recuperacao.py` | busca na base de conhecimento |
| `conhecimento/*.md` | análise do trabalho anterior: colunas, limpeza, EDA, modelo, métricas, limitações |
| `prompts/` | instruções das etapas 1 (interpretar) e 4 (responder) |
| `app.py` + `static/index.html` | servidor Flask e página de chat (Render) |
| `testes/testar_sem_api.py` | testes sem gastar cota (ponte, validação, erros de domínio, linguagem natural, filtros, RAG, verificação) |
| `testes/avaliar_rag.py` | mede a qualidade da busca com 20 perguntas de teste |
| `conferir.py` | gabarito (estatísticas e previsões) a partir do CSV e do modelo |
| `testes/comparar_sem_rag.py` | mesmas perguntas com e sem RAG (relatório e demonstração) |
| `testes/bateria.py` | as 12 categorias da bateria da apresentação, com gabarito do nosso código |
| `docs/registro_testes.md` | registro da bateria (pergunta, etapas, tempo, resposta, avaliação) |

## Modelo

- Classificação binária: `cancelada` = `sim` / `nao` (classe positiva: `sim`).
- 8 atributos (o enunciado permite até 8): `antecedencia_dias`, `preco_medio_diaria`, `pedidos_especiais`, `mes_chegada`, `noites_fim_de_semana`, `noites_semana`, `segmento_mercado`, `vaga_estacionamento`. Na validação cruzada, o MLP com esses 8 teve F1 0,7020 contra 0,7028 com os 17 originais.
- Métricas de teste reais em `modelo/metricas.json` (acurácia, F1, precisão, recall, especificidade, AUC).

## Como rodar

```bash
python -m venv .venv            # Python 3.13
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python treinar.py               # opcional: o modelo já está em modelo/
python testes/testar_sem_api.py
python testes/avaliar_rag.py
python testes/bateria.py local    # gasta ~30 chamadas do Gemini
python conferir.py
copy .env.example .env          # e colem a GEMINI_API_KEY
python app.py                   # http://localhost:5000
```

Deploy no Render: build `pip install -r requirements.txt`, start `gunicorn app:app --workers 1 --threads 4 --timeout 120`, variáveis `GEMINI_API_KEY` e `PYTHON_VERSION=3.13.15`. A chave **nunca** vai para o código nem para o GitHub (`.env` está no `.gitignore`).

> Chave do Gemini: a conta UNOESC não conseguiu criar a chave no AI Studio (erro "Project creation is disabled"). Conforme o enunciado, usamos uma conta Google pessoal, no plano gratuito, com o modelo `gemini-3.5-flash-lite`.

## O que mudamos em relação ao kit

1. **Domínio e modelo próprios:** dataset Hotel Reservations com a mesma limpeza do trabalho anterior; colunas e categorias traduzidas para o português no dataset legível; modelo retreinado com 8 atributos escolhidos por validação cruzada; grade do MLP ampliada (alpha até 0,1), como sugerido nas melhorias do trabalho anterior. O F1 de teste subiu de 0,693 para 0,715.
2. **Consultas com filtros:** a etapa 1 devolve também `filtros` (`coluna`, `operador`, `valor`), e `estatisticas()` aplica as condições antes de calcular. Ex.: "quantas reservas Online com mais de 100 dias de antecedência foram canceladas?". As consultas usam **todas** as colunas do dataset, não só as do modelo, e trazem maior e menor taxa já calculadas (o Gemini não precisa comparar números).
3. **Entrada em linguagem natural:** "2 semanas" → 14 dias, "3 meses" → 90, "julho" → 7, "duas" → 2, "nenhum", "sem" ou "não" → 0 (só quando não há outro número na frase), "1 milhão" → 1.000.000, "não precisa de vaga" → `nao`, "empresa" → Corporativo.
4. **Regras de domínio do hotel:** antecedência negativa, mês fora de 1–12, estadia sem noites, diária negativa ou acima de 5.000 € viram **erro**; valores fora da faixa do treino viram **alerta**; diária 0 fora do segmento Cortesia gera alerta.
5. **Previsão mais explicada:** probabilidade de cancelamento em %, grau de confiança (perto do limiar de 50% = baixa), recall e precisão de teste junto da previsão; casos semelhantes com 15 reservas reais do mesmo segmento (e mesma vaga), percentual cancelado entre elas e taxa geral do dataset para comparação.
6. **RAG melhor, e medido:** base dividida em 3 arquivos com trechos curtos; mais fatos gerados do CSV (taxa por segmento, plano, quarto, vaga, recorrência, faixa de antecedência, pedidos especiais, mês com nome, erros do modelo, comparação com o palpite ingênuo); busca combinando TF-IDF de palavras e de pedaços de letras, com expansão por sinônimos do domínio. Com `testes/avaliar_rag.py` (20 perguntas de teste), o acerto nos 5 primeiros trechos subiu de 70% (busca do kit) para 95%, e as 3 perguntas fora do escopo não trazem nenhum trecho.
7. **Página:** exemplos clicáveis, aviso de "servidor acordando" quando a resposta demora, layout para celular.
8. **Testes próprios:** `testes/testar_sem_api.py` cobre casos impossíveis do domínio, linguagem natural e filtros, com `assert`.
9. **Bateria automatizada:** `testes/bateria.py` roda as 12 categorias da apresentação (no Render ou local), calcula o gabarito com o nosso código e salva um registro datado em `docs/` (o avaliado para o relatório é `docs/registro_testes.md`).
10. **Verificação mais transparente e mais barata:** quando a resposta é refeita, a linha 🔧 mostra quais números não foram conferidos. Com isso descobrimos que o Gemini fazia contas ("deixa passar 32 de cada 100" = 100 − 68). Passamos a entregar essas contas prontas no contexto (`metricas_explicadas()`) e incluímos o prompt da persona entre os textos confiáveis (ele cita "2017 e 2018"). Na bateria, as respostas refeitas caíram de 5 em 10 respostas geradas para 0 em 12, economizando chamadas da cota.
11. **"Essa previsão faz sentido?":** a etapa 1 passou a repassar os dados da reserva em qualquer intenção, então o código roda de novo o modelo e os casos semelhantes nessa pergunta (antes ele só repetia os números da resposta anterior).
