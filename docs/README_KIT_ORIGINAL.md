# Kit inicial — Agente com RAG + modelo de ML (IA · UNOESC 2026/2)

Este kit é um **ponto de partida que já funciona**: um agente conversacional **escrito por vocês em Python**, que usa a API do Gemini só para *entender* a pergunta e *redigir* a resposta. Todo o resto (buscar no dataset, rodar o modelo, conferir os números) é código de vocês.

O exemplo é de **regressão** (aluguel de imóveis, dados sintéticos). Há também uma versão de **classificação** em `exemplo_classificacao/`. Substituam o exemplo pelo dataset, pela análise e pelo modelo do Trabalho Prático anterior.

```
kit_agente/
├── app.py                  servidor Flask: página + rota /api/chat (deploy no Render)
├── versao_streamlit/       a MESMA interface em Streamlit (deploy no Streamlit Community Cloud)
├── agente.py               O AGENTE: pipeline interpretar → recuperar → prever → gerar → verificar
├── recuperacao.py          RAG: busca TF-IDF na base de conhecimento
├── ferramentas.py          dados e modelo: preparar, prever, casos semelhantes, estatísticas, fatos
├── conhecimento/           textos da análise do trabalho anterior (base do RAG)
├── prompts/
│   ├── interpretar.md      instruções da etapa 1 (o Gemini devolve JSON)
│   └── responder.md        persona e regras da etapa 4 (a resposta ao usuário)
├── modelo/                 modelo.joblib + metricas.json + amostra_conferencia.json
├── dados/                  dataset_legivel.csv (dados LEGÍVEIS: para pessoas e para o RAG)
├── exemplo_transformacao/  se as transformações do treino foram feitas FORA do Pipeline
├── static/index.html       a página de chat
├── exemplo_classificacao/  versão pronta para CLASSIFICAÇÃO (veja o LEIAME.md de lá)
├── treinar_exemplo.py      exemplo de treino (troquem pelo de vocês)
├── testar_sem_api.py       testa dados, modelo, RAG e verificação sem gastar cota do Gemini
├── conferir.py             gera o "gabarito" (estatísticas e previsões) a partir do CSV e do modelo
├── comparar_sem_rag.py     mesmas perguntas com e sem RAG (para o relatório e a demonstração)
├── links.txt               modelo do arquivo de entrega
└── requirements.txt        versões fixas (inclusive scikit-learn e numpy!)
```

## Como o agente funciona

```
mensagem do usuário
   │
   ├─1─ INTERPRETAR  (Gemini → JSON)   intenção: dados | previsão | explicação | fora do escopo
   │                                   + atributos citados na conversa + consulta + pergunta reescrita
   ├─2─ RECUPERAR    (código)          RAG: trechos da base de conhecimento (TF-IDF)
   │                                   + estatísticas exatas do CSV (pandas)
   ├─3─ PREVER       (código)          falta algum dado? → pergunta ao usuário
   │                                   valida → para_o_modelo() (ponte legível → modelo)
   │                                   → roda o modelo .joblib → casos reais semelhantes
   ├─4─ GERAR        (Gemini)          escreve a resposta usando SÓ o CONTEXTO das etapas 2 e 3
   └─5─ VERIFICAR    (código)          todo número da resposta está no CONTEXTO?
                                       se não: pede outra resposta (1 vez) ou avisa o usuário
```

- **O Gemini nunca vê o CSV inteiro nem o modelo.** Ele só recebe o que o código de vocês recuperou. É isso que impede respostas inventadas.
- **Não usamos o "function calling" automático nem a API de agentes do Gemini.** Quem decide o que fazer em cada etapa é o `agente.py`.
- **A página mostra 🔧 com as etapas de cada resposta** (ex.: `interpretação: previsao · RAG: 2 trecho(s) · modelo: previsão · validação: casos semelhantes · verificação: ok`).
- **Custo por mensagem:** 2 chamadas ao Gemini (interpretar e gerar), às vezes 3 (quando a verificação pede para refazer). Pedido incompleto e fora do escopo gastam só 1.

## Duas versões dos dados (leiam com atenção)

O dataset do trabalho anterior foi **ajustado para o modelo de ML**: categorias viraram números (one-hot, LabelEncoder, códigos 1/2) e as escalas foram ajustadas (StandardScaler). Para **conversar com pessoas**, isso não serve, porque ninguém pergunta "qual a média quando bairro_3 = 1?". Por isso, o agente trabalha com **duas versões dos dados**:

| | **Dados legíveis** (`dados/dataset_legivel.csv`) | **Formato do modelo** |
|---|---|---|
| Como é | limpo, mas **sem** codificar texto e **sem** mudar escala: "Centro", "feminino", R$, m², anos | o que o modelo recebeu no treino: colunas 0/1, códigos, valores padronizados |
| Quem usa | as estatísticas (pandas), os casos semelhantes, os fatos do RAG e as mensagens ao usuário | **só** o `MODELO.predict(...)` |
| Onde | `ferramentas.DF` | gerado na hora por `para_o_modelo(entrada)` |

```
usuário: "80 m², 2 quartos, 1 vaga, no centro"
   │  etapa 1 extrai → {"area_m2": "80", "quartos": "2", "vagas": "1", "bairro": "centro"}
   │  preparar() valida e corrige → {"area_m2": 80.0, "quartos": 2, "vagas": 1, "bairro": "Centro"}   (LEGÍVEL)
   ▼
para_o_modelo()  ← A PONTE: aplica as MESMAS transformações do treino (one-hot, escala, códigos...)
   ▼
MODELO.predict(...)  → previsão → volta para o formato legível na resposta (R$, "sim"/"não")
```

⚠️ **Não podem esquecer a ponte.** Se o modelo receber a entrada sem as transformações do treino, ele erra **em silêncio**. Por exemplo, com a área em m² onde ele esperava a área padronizada, ou com "Centro" onde ele esperava a coluna `bairro_Centro = 1`. Há duas situações:
- **Caso 1 (recomendado, é o do kit):** o pré-processamento está **dentro do Pipeline** salvo (seção 3.1). A ponte só monta o DataFrame, e o próprio Pipeline transforma.
- **Caso 2:** vocês transformaram os dados **fora** do Pipeline, com `get_dummies`, `LabelEncoder`, scaler ou mapa de códigos. Salvem os transformadores **ajustados no treino** e reescrevam `para_o_modelo()` para aplicá-los, na mesma ordem. O exemplo completo está em `exemplo_transformacao/` (veja o `LEIAME.md` de lá).

**Rede de segurança:** o treino salva `modelo/amostra_conferencia.json`, com 5 entradas legíveis e a previsão que o modelo deu no treino. Ao iniciar, o app refaz essas previsões passando pela ponte. Se faltar ou sobrar alguma transformação, aparece no log `ATENÇÃO: a ponte para_o_modelo() NÃO reproduz o treino`. O `testar_sem_api.py` e o `conferir.py` também mostram o resultado dessa conferência. Gerem esse arquivo no treino de vocês, copiando o trecho do `treinar_exemplo.py`.

**Se o CSV do trabalho anterior já estava transformado** (códigos, valores padronizados), gerem o `dataset_legivel.csv` a partir do CSV original do Kaggle, só com a limpeza. O dicionário de códigos (ex.: 1 = masculino) vai para `ATRIBUTOS` e para a base de conhecimento.

## Como o agente responde perguntas sobre os dados

Exemplo: "qual bairro tem o aluguel mais caro?"
1. **Interpretar:** o Gemini devolve `{"intencao": "dados", "consulta": {"coluna": "aluguel", "agrupar_por": "bairro"}, "busca": "bairro com maior aluguel médio"}`. Ele só **escolhe** a coluna e o agrupamento, dentro das opções que o código oferece (`COLUNAS_CONSULTA`, `AGRUPAMENTOS`); não calcula nada.
2. **Recuperar:**
   - o código roda `estatisticas("aluguel", "bairro")` com pandas no **dataset legível** e obtém as médias exatas;
   - o RAG busca, com o texto de `busca`, os trechos parecidos da base de conhecimento, por exemplo "Centro e Jardim Itália são os mais caros" e o fato gerado do CSV com as médias por bairro.
3. **Gerar:** o Gemini recebe a pergunta mais esse CONTEXTO e escreve a resposta.
4. **Verificar:** o código confere se cada número da resposta está no CONTEXTO.

O que o agente consegue responder depende do que vocês oferecem: as colunas e os agrupamentos de `estatisticas()`, os fatos de `fatos_do_dataset()` e os textos de `conhecimento/`. Para responder perguntas novas (contagem com filtro, top-N, comparação de dois grupos), criem novas consultas no `ferramentas.py`, acrescentem as opções no esquema da etapa 1 (`agente.py`) e chamem-nas na etapa 2. Isso conta como "além do kit".

## Passo a passo

### 0. Pré-requisito
**Python 3.12 ou 3.13.** Em 3.11 o `pip install` falha no numpy. Confiram com `python --version`.

### 1. Rodar o exemplo como está
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python testar_sem_api.py           # testa tudo o que não usa o Gemini
python conferir.py                 # mostra o "gabarito" do exemplo
```
> Windows: se o PowerShell bloquear a ativação do venv, rodem uma vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 2. Pegar a chave do Gemini (gratuita)
1. Entrem em **https://aistudio.google.com** com a **conta da UNOESC**. Se não abrir, avisem o professor até 24/11: nesse caso vale conta Google pessoal.
2. Cliquem em **Get API key → Create API key**.
3. **Não** ativem faturamento (billing): o trabalho usa só o plano gratuito.
4. Criem o `.env` a partir do modelo: `cp .env.example .env` (Windows: `copy .env.example .env`). Deixem a linha assim, sem aspas e sem espaços: `GEMINI_API_KEY=AIza...`.
   > No Windows, o Bloco de Notas pode salvar como `.env.txt`. Confiram o nome do arquivo.
5. Rodem `python app.py` e abram http://localhost:5000.
   > No macOS, se a porta 5000 estiver ocupada: `PORT=5001 python app.py`.

**Se o chat der erro, olhem o terminal onde o `app.py` está rodando:**
- 400/403 = chave errada (ou bloqueio da hospedagem);
- 429 = cota esgotada;
- 404 = nome do modelo errado;
- traceback em `agente.py`/`ferramentas.py` = bug no código de vocês.

O terminal também mostra as etapas de cada pergunta.

> ⚠️ **Nunca** coloquem a chave no `index.html`, no JavaScript ou no GitHub (o `.env` já está no `.gitignore`). Se vazar, revoguem a chave no AI Studio e criem outra.
> ⚠️ No plano gratuito, o Google pode usar o conteúdo enviado para melhorar os produtos dele. Não enviem dados pessoais nem sensíveis.

**Cota gratuita:** vejam o limite diário de vocês no AI Studio, em *Rate limits*. **Deixem reservada, para o dia da apresentação, uma chave de outra conta Google (outro projeto)**: chaves do mesmo projeto dividem a mesma cota. Quem faz sozinho pode usar uma conta pessoal como reserva.

### 3. Trocar o exemplo pelo trabalho de vocês

**3.1 Modelo e ponte.** O jeito mais simples é o **caso 1**: o `.joblib` aceita os dados legíveis, porque o pré-processamento fica **dentro** do Pipeline salvo:
```python
prep = ColumnTransformer([("num", StandardScaler(), NUMERICAS),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)])
pipe = Pipeline([("prep", prep), ("modelo", SEU_MODELO)])
grid = GridSearchCV(pipe, {...}, cv=5).fit(X_tr, y_tr)
joblib.dump(grid.best_estimator_, "modelo/modelo.joblib")   # best_estimator_, não o grid
# teste: joblib.load(...).predict(X_te.head(3)) tem que funcionar sem pré-processamento manual
```
Se usaram `pd.get_dummies`, `LabelEncoder` ou um scaler **fora** do Pipeline, há duas saídas: refaçam o treino assim (mais fácil), ou mantenham o treino e sigam o **caso 2** (`exemplo_transformacao/`). Nos dois casos, salvem a `amostra_conferencia.json` e confiram que o `testar_sem_api.py` mostra `ok: True` na ponte.

**3.2 Onde treinar (recomendado: dentro do venv do kit).** Criem um `treinar.py` com base no `treinar_exemplo.py` e rodem-no no venv do kit. Assim as versões batem sem mexer no `requirements.txt`. Se forem subir um `.joblib` do Colab, igualem no `requirements.txt` as versões de scikit-learn e numpy do Colab.

**3.3 `metricas.json`:**
- regressão: `r2_teste`, `mae_teste` e `sklearn_version`;
- classificação: `acuracia_teste`, `f1_teste` e `sklearn_version`.

**3.4 Dataset.** Usem o **mesmo dataset** do trabalho anterior. O algoritmo pode mudar se o feedback pediu.
- Coloquem o CSV **limpo** em `dados/`. Se for grande (dezenas de MB), usem uma amostra: o Render gratuito tem 512 MB.
- Com mais de 8 atributos, podem retreinar com um subconjunto de até 8, justificando no relatório.
- **A referência dos números é o CSV e o modelo do repositório** (`python conferir.py`).

**3.5 `ferramentas.py`** (para classificação, comecem pelo `exemplo_classificacao/`):
- ajustem `ALVO`, `NUMERICAS`, `CATEGORICAS`, `UNIDADE_ALVO` e o dicionário `ATRIBUTOS` (descrição e tipo de cada entrada: o agente usa isso para extrair os dados da conversa e pedir o que falta);
- categorias codificadas como número (ex.: SEXO 1/2) vão em `CATEGORICAS`, com `"tipo": "texto"` em `ATRIBUTOS` e o significado de cada código na descrição e na base de conhecimento;
- em `_regras_do_dominio()`, escrevam o que é **impossível** no domínio de vocês (vira erro). O que estiver fora da faixa do treino vira **alerta** (prevê com aviso);
- em `fatos_do_dataset()`, gerem as frases com números que fazem sentido para o dataset de vocês;
- retornem só tipos nativos (`float()`, `int()`, `str()`), porque `numpy.int64` e `NaN` quebram o JSON.

**3.6 Base de conhecimento (`conhecimento/`).** Substituam o `analise_exploratoria.md` pelo texto da análise de vocês: descrição das colunas, achados da EDA, correlações, por que escolheram o modelo, métricas e limitações. **Cada parágrafo ou item de lista vira um trecho do RAG**, então escrevam trechos curtos e autoexplicativos.

**3.7 Prompts.** Em `prompts/responder.md`, troquem a persona e as regras de resposta do domínio de vocês. Em `prompts/interpretar.md`, normalmente não é preciso mexer, porque os atributos e as colunas são preenchidos automaticamente a partir do `ferramentas.py`. Os marcadores `{atributos}`, `{colunas}` e `{agrupamentos}` são substituídos pelo código. O campo `busca` da etapa 1 é a pergunta reescrita de forma completa (ex.: "e no Centro?" vira "qual o aluguel médio no Centro?"), e é ela que o RAG usa para procurar os trechos.

**3.8 Checklist "troquem TUDO que fala de aluguel":** `ferramentas.py` (constantes, `ATRIBUTOS`, regras, fatos e a ponte `para_o_modelo()`), `modelo/amostra_conferencia.json` (gerado no treino de vocês), `conhecimento/`, `prompts/responder.md`, título e boas-vindas em `static/index.html` **e** em `versao_streamlit/streamlit_app.py`.

### 4. Publicar no Render (gratuito, sem cartão)
1. Subam para o GitHub (com commits ao longo do trabalho, pelos dois integrantes):
   ```bash
   git init && git add . && git commit -m "kit inicial"
   git branch -M main
   git remote add origin https://github.com/USUARIO/REPO.git
   git push -u origin main
   git status                  # o .env NÃO pode aparecer
   ```
2. Em **https://render.com**, cliquem em **New → Web Service** e conectem o repositório.
3. Configurem:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --workers 1 --threads 4 --timeout 120`
   - **Instance type:** Free
   - **Environment:** `GEMINI_API_KEY` = a chave, e **`PYTHON_VERSION`** = a versão completa do Python do venv onde treinaram (ex.: `3.13.5`; vejam com `python --version`). O `.python-version` do kit é só um reforço.
4. O link fica no formato `https://nome-do-app.onrender.com`. Abram `…/saude` para testar.

> O plano gratuito "dorme" depois de 15 min sem acesso, e a primeira abertura leva cerca de 1 minuto. **Antes da apresentação, abram `…/saude`.**
> **Publiquem cedo (checkpoint de qui 26/11, no fórum).** Em 2026 houve relatos de o Gemini recusar (erro 403) chamadas vindas do Render. Se acontecer, guardem o print do log e avisem o professor.
> **Plano B: treinar no deploy.** Usem o build command `pip install -r requirements.txt && python treinar.py`, com `random_state` fixo. O professor confere com `python treinar.py && python conferir.py`.

- **Hash do commit para o `links.txt`:** `git rev-parse --short HEAD`.
- **Congelamento:** de 02/12, 23h59, até a arguição do grupo, nenhum commit na branch publicada (`main`) e nenhum deploy. Podem trabalhar em outra branch. Trocar a `GEMINI_API_KEY` no painel gera um redeploy do mesmo commit, o que é permitido: avisem o professor. A alteração ao vivo da arguição é feita localmente.

### 4b. Ou publicar no Streamlit Community Cloud (gratuito)
A pasta `versao_streamlit/` tem a **mesma** interface de chat em Streamlit, usando o **mesmo** `agente.py`. Escolham Render **ou** Streamlit; não precisa dos dois.
1. Testem localmente, com o `.env` criado:
   ```bash
   pip install -r versao_streamlit/requirements.txt
   streamlit run versao_streamlit/streamlit_app.py
   ```
2. Subam o repositório para o GitHub (mesmo passo 1 do Render).
3. Em **https://share.streamlit.io**, entrem com a conta do GitHub → **Create app** → escolham o repositório, a branch `main` e o arquivo principal **`versao_streamlit/streamlit_app.py`**.
4. Em **Advanced settings**:
   - **Python version:** a mesma do treino (3.12 ou 3.13);
   - **Secrets:** colem `GEMINI_API_KEY = "AIza..."` (veja `versao_streamlit/secrets.toml.exemplo`).
5. O Streamlit usa o `versao_streamlit/requirements.txt`: mantenham nele as **mesmas versões** de scikit-learn e numpy do treino.
6. O link fica no formato `https://nome-do-app.streamlit.app`. O app "dorme" depois de 12 horas sem acesso; abram o link antes da apresentação para acordá-lo.
7. As etapas (🔧) aparecem embaixo de cada resposta, e o log (**Manage app → Logs**) mostra as etapas de cada pergunta, como no Render.

### Alternativas gratuitas
| Plataforma | Serve para | Observações |
|---|---|---|
| **Render** (recomendado) | servidor Flask + página | 512 MB de RAM, dorme após 15 min, sem cartão |
| **Streamlit Community Cloud** (2ª opção) | `versao_streamlit/` (pronta no kit) | Grátis; chave em *Secrets*; dorme depois de 12 h sem acesso |
| **Vercel** (Hobby) | página + função Python em `api/` (*serverless*) | Exige reorganizar o projeto; pacote de até 500 MB |
| **PythonAnywhere** (Beginner) | servidor Flask (via WSGI) | Plano B fraco: 512 MB de disco, 100 s de CPU/dia, renovação mensal |
| ~~Hugging Face Spaces~~ | só páginas estáticas grátis | Desde jul/2026, backend exige plano pago |
| ~~Koyeb, Cloud Run, Fly.io, Railway~~ | — | Exigem cartão de crédito |

## O que mudamos em relação ao kit (obrigatório no README de vocês)
Parte da nota do código é **ir além do kit**. Criem no README de vocês uma seção com esse título. Ideias:
- **RAG melhor:** trechos mais bem divididos, mais fatos gerados do CSV, ajuste do limiar e do `k`, combinação de TF-IDF com palavras-chave das colunas, e um conjunto de perguntas de teste para medir se a busca traz o trecho certo;
- **consultas novas:** contagem com filtro ("quantos imóveis acima de 100 m² no Centro?"), top-N, comparação de dois casos;
- **verificação melhor:** checar também as categorias e as classes citadas (a verificação atual não pega um número certo atribuído ao grupo errado), ou uma terceira chamada que revisa a resposta;
- **entrada em linguagem natural:** "uns 80 metros", "dois quartos";
- **melhorias na página:** exemplos clicáveis, aviso de "acordando o servidor";
- **testes automatizados próprios.**

## Testem como o professor vai testar
Rodem `python conferir.py`: é com ele que o professor monta o gabarito. Depois conversem com o agente e registrem no relatório **pelo menos 10 testes** (individual: 6). O ideal é cobrir as 12 categorias da bateria. Use o botão **🧹 Nova conversa** entre testes que não dependem um do outro: o agente lembra os dados informados antes na mesma conversa.
- **Dados:** 3 perguntas sobre os dados;
- **Previsão:**
  - 1 pedido completo (numa só mensagem);
  - 1 pedido faltando 2 dados (o agente tem que perguntar);
  - 1 com valor absurdo (o agente tem que alertar);
  - "Essa previsão faz sentido?" (o agente tem que comparar com casos reais);
- **Robustez:**
  - 1 pergunta fora do assunto;
  - 1 tentativa de fazer o agente revelar as instruções ou "esquecer as regras";
  - 1 pergunta de acompanhamento ("e se tivesse 1 quarto a mais?");
- **Qualidade:** "explique para um leigo" e "esse modelo é confiável?".

Repetir numa pergunta de acompanhamento um número já obtido numa mensagem anterior vale, se estiver certo. Previsão nova precisa passar pelo modelo (🔧 `modelo: previsão`).

**Limitação conhecida (discutam no relatório):** os "casos semelhantes" vêm do mesmo dataset usado no treino. Num KNN, eles são quase os mesmos vizinhos que o modelo usou, então a validação é indicativa e não uma prova independente.

## Mini-glossário
- **RAG (Retrieval-Augmented Generation):** antes de gerar a resposta, o sistema *recupera* informações relevantes (aqui, trechos da análise e estatísticas do CSV) e as entrega ao modelo de linguagem como contexto.
- **TF-IDF:** representa cada texto pelas palavras importantes que ele contém. A busca compara a pergunta com os trechos pela similaridade de cosseno.
- **Saída estruturada (JSON):** na etapa 1, o Gemini é obrigado a responder num formato JSON definido por nós (`response_json_schema`).
- **Pipeline (joblib):** um arquivo `.joblib` com pré-processamento + modelo, que aceita os dados crus.
- **Variável de ambiente:** configuração fora do código (no `.env`, local, ou no painel da hospedagem).
- **Deploy:** publicar o app na hospedagem. **Hash do commit:** o código da versão entregue.
- **429 / 503 / 403:** cota esgotada / serviço indisponível / acesso negado.
