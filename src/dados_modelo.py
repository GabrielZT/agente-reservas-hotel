"""
Dados e modelo do grupo: tudo o que é DETERMINÍSTICO (não usa o Gemini).
Domínio: cancelamento de reservas de hotel (Hotel Reservations, Kaggle) — CLASSIFICAÇÃO.

O agente (pipeline.py) decide QUANDO usar cada função; aqui fica O QUE cada uma faz:
  - preparar(entrada)                       -> converte texto em valores, valida (erro x alerta)
  - prever(entrada)                         -> roda o modelo .joblib (classe + probabilidade)
  - casos_semelhantes(entrada)              -> reservas reais parecidas no dataset (validação)
  - estatisticas(coluna, grupo, filtros)    -> números exatos do dataset (pandas)
  - fatos_do_dataset()                      -> frases com números reais para o RAG

Sempre devolvem tipos nativos (float, int, str, list, dict): numpy.int64 e NaN quebram o JSON.
"""
import json
import logging
import math
import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
import sklearn

log = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parent.parent           # raiz do projeto (este arquivo está em src/)
MODELO = joblib.load(BASE / "models" / "modelo.joblib")
METRICAS = json.loads((BASE / "models" / "metricas.json").read_text(encoding="utf-8"))
LIMIAR = float(METRICAS.get("limiar_de_decisao", 0.5))   # limiar escolhido por validação cruzada no treino
LIMIAR_TEXTO = f"{100 * LIMIAR:.1f}%".replace(".", ",")  # "35,4%" (mesmo formato do resto do projeto)
DF = pd.read_csv(BASE / "data" / "dataset_legivel.csv")

TAREFA = "classificacao"
ALVO = "cancelada"                                     # classes: "sim" (cancelada) / "nao"
CLASSE_POSITIVA = "sim"                                # a classe de interesse (cancelamento)
NUMERICAS = ["antecedencia_dias", "preco_medio_diaria", "pedidos_especiais", "mes_chegada",
             "noites_fim_de_semana", "noites_semana"]
CATEGORICAS = ["segmento_mercado", "vaga_estacionamento"]
UNIDADE_ALVO = "classe: a reserva foi cancelada (sim/nao)"

# Atributos de ENTRADA do modelo: o agente usa isto para extrair os valores da conversa e para
# perguntar o que falta. tipo: "numero", "inteiro" ou "texto".
ATRIBUTOS = {
    "antecedencia_dias": {"tipo": "inteiro", "descricao": "antecedência da reserva: quantos dias antes da chegada ela foi feita"},
    "preco_medio_diaria": {"tipo": "numero", "descricao": "preço médio da diária do quarto, em euros"},
    "pedidos_especiais": {"tipo": "inteiro", "descricao": "número de pedidos especiais feitos pelo hóspede (ex.: andar alto, berço, vista)"},
    "mes_chegada": {"tipo": "inteiro", "descricao": "mês da chegada ao hotel (1 a 12 ou o nome do mês)"},
    "noites_fim_de_semana": {"tipo": "inteiro", "descricao": "número de noites de fim de semana (sábado ou domingo) da estadia"},
    "noites_semana": {"tipo": "inteiro", "descricao": "número de noites de segunda a sexta da estadia"},
    "segmento_mercado": {"tipo": "texto", "descricao": "segmento de mercado da reserva (Online = site/aplicativo; Offline = agência ou telefone; Corporativo = empresa; Aviação = companhia aérea; Cortesia = gratuita)"},
    "vaga_estacionamento": {"tipo": "texto", "descricao": "se o hóspede pediu vaga de estacionamento (sim ou nao)"},
}

if METRICAS.get("sklearn_version") != sklearn.__version__:
    log.error("ATENÇÃO: modelo salvo com scikit-learn %s, mas o servidor usa %s. Ajustem o requirements.txt.",
              METRICAS.get("sklearn_version"), sklearn.__version__)


def _norm(texto) -> str:
    """'  Aviação ' e 'aviacao' viram a mesma chave."""
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c)).casefold().strip()


def _nativo(v):
    """numpy.int64 -> int, numpy.float64 -> float (para virar JSON)."""
    return v.item() if hasattr(v, "item") else v


CATEGORIAS = {col: {_norm(v): _nativo(v) for v in DF[col].dropna().unique()} for col in CATEGORICAS}
# sinônimos em linguagem natural (além do kit): "empresa" -> Corporativo, "não precisa" -> nao...
SINONIMOS = {
    "segmento_mercado": {"corporate": "Corporativo", "corporativa": "Corporativo", "empresa": "Corporativo",
                         "complementary": "Cortesia", "gratuita": "Cortesia", "gratis": "Cortesia",
                         "aviation": "Aviação", "companhia aerea": "Aviação", "aerea": "Aviação",
                         "internet": "Online", "site": "Online", "app": "Online", "aplicativo": "Online",
                         "agencia": "Offline", "telefone": "Offline", "balcao": "Offline"},
    "vaga_estacionamento": {"s": "sim", "yes": "sim", "1": "sim", "true": "sim", "precisa": "sim", "com vaga": "sim",
                            "quer": "sim", "pediu": "sim", "n": "nao", "no": "nao", "0": "nao", "false": "nao",
                            "nao precisa": "nao", "sem vaga": "nao", "nenhuma": "nao", "nao pediu": "nao"},
}
for col in CATEGORICAS:                                 # o agente mostra as opções válidas
    ATRIBUTOS[col]["valores"] = sorted(str(v) for v in CATEGORIAS[col].values())
FAIXAS = {col: (float(DF[col].min()), float(DF[col].max())) for col in NUMERICAS}
DESVIOS = {col: float(DF[col].std()) or 1.0 for col in NUMERICAS}

# Consultas sobre os dados (etapa 2): podem usar TODAS as colunas do dataset legível, não só as do modelo
COLUNAS_NUMERICAS_DF = [c for c in DF.columns if pd.api.types.is_numeric_dtype(DF[c])]
COLUNAS_TEXTO_DF = [c for c in DF.columns if c not in COLUNAS_NUMERICAS_DF]
COLUNAS_CONSULTA = COLUNAS_NUMERICAS_DF + [ALVO]       # ALVO -> contagem/proporção das classes
AGRUPAMENTOS = [c for c in COLUNAS_TEXTO_DF if c != ALVO] + [
    "mes_chegada", "ano_chegada", "pedidos_especiais", "adultos", "criancas", "total_noites", ALVO]
OPERADORES = ["==", "!=", ">", ">=", "<", "<="]
ORDEM_FAIXAS = ["0-7 dias", "8-30 dias", "31-90 dias", "91-180 dias", "mais de 180 dias"]
NOMES_MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
               "setembro", "outubro", "novembro", "dezembro"]
MESES = [_norm(m) for m in NOMES_MESES]                 # sem acento, para comparar com _norm(texto)


# ---------------------------------------------------------------- PONTE: dataset legível -> modelo
# Caso 1 do kit: o pré-processamento (StandardScaler + OneHotEncoder) está DENTRO do Pipeline salvo
# por scripts/treinar.py. Por isso a ponte só monta o DataFrame com as colunas na ordem do treino.
# conferir_transformacao() compara, ao iniciar o app, as previsões desta ponte com as do treino.
def para_o_modelo(entrada: dict) -> pd.DataFrame:
    """Entrada legível (dict) -> DataFrame no formato que o modelo espera."""
    return pd.DataFrame([entrada])[NUMERICAS + CATEGORICAS]


def conferir_transformacao() -> dict:
    """Refaz as previsões da amostra salva no treino (models/amostra_conferencia.json) passando
    pela ponte para_o_modelo(). Se não baterem, a transformação está diferente da do treino."""
    arquivo = BASE / "models" / "amostra_conferencia.json"
    if not arquivo.exists():
        return {"ok": False, "detalhes": ["models/amostra_conferencia.json não existe: rodem scripts/treinar.py"]}
    problemas = []
    for item in json.loads(arquivo.read_text(encoding="utf-8")):
        try:
            obtido = _nativo(MODELO.predict(para_o_modelo(item["entrada"]))[0])
        except Exception as e:                     # ex.: colunas diferentes das do treino
            problemas.append(f"o modelo recusou a entrada transformada ({type(e).__name__}: {str(e)[:200]})")
            continue
        esperado = item["previsao_no_treino"]
        if str(obtido) != str(esperado):
            problemas.append(f"entrada {item['entrada']}: esperado {esperado}, obtido {obtido}")
    return {"ok": not problemas, "detalhes": problemas}


def faltando(entrada: dict) -> list[str]:
    """Atributos que o usuário ainda não informou."""
    return [a for a in ATRIBUTOS if entrada.get(a) in (None, "")]


# ---------------------------------------------------------------- entrada em linguagem natural
_EXTENSO = {"zero": 0, "nenhum": 0, "nenhuma": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3,
            "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10, "onze": 11,
            "doze": 12, "quinze": 15, "vinte": 20, "trinta": 30, "quarenta": 40, "cinquenta": 50,
            "sessenta": 60, "noventa": 90, "cem": 100}
# "sem pedidos", "não fez", "grátis" = 0, mas só quando não há outro número ("não sei, uns dois" = 2)
_NEGACAO = {"sem", "nao", "nem", "gratis", "gratuita", "gratuito"}
_MILHAR = r"-?[1-9]\d{0,2}(?:\.\d{3})+(?:,\d+)?"


def _numero(valor) -> float:
    """'80', '5.000', '€ 120,50', '2,5', '5 mil', 'duas', 'nenhum', 'sem' -> float (formato brasileiro)."""
    s = str(valor)
    m = re.search(_MILHAR + r"(?![\d.])|-?\d+(?:[.,]\d+)?", s)
    if not m:
        palavras = re.findall(r"[a-z]+", _norm(s))
        achadas = [_EXTENSO[p] for p in palavras if p in _EXTENSO]
        if not achadas and _NEGACAO & set(palavras):
            achadas = [0]
        if not achadas:
            raise ValueError(valor)
        x = float(achadas[0])
    else:
        t = m.group()
        t = t.replace(".", "").replace(",", ".") if re.fullmatch(_MILHAR, t) else t.replace(",", ".")
        x = float(t)
    if not math.isfinite(x):
        raise ValueError(valor)
    if re.search(r"\bmilh(ao|oes)\b", _norm(s)):
        return x * 1_000_000
    return x * 1000 if re.search(r"\bmil\b", s, re.I) else x


def _dias(valor) -> float:
    """Antecedência: '30', '2 semanas', '3 meses', 'um ano' -> dias."""
    x, s = _numero(valor), _norm(valor)
    if re.search(r"\bsemanas?\b", s):
        return x * 7
    if re.search(r"\b(mes|meses)\b", s):
        return x * 30
    if re.search(r"\banos?\b", s):
        return x * 365
    return x


def _mes(valor) -> float:
    """'julho', 'jul', '7', 'mês 7' -> 7."""
    s = _norm(valor)
    for i, nome in enumerate(MESES, start=1):
        if re.search(rf"\b{nome[:3]}", s):
            return float(i)
    return _numero(valor)


def _categoria(nome: str, valor):
    chave = _norm(valor)
    achado = CATEGORIAS[nome].get(chave) or SINONIMOS.get(nome, {}).get(chave)
    if achado is None and nome == "vaga_estacionamento":       # "não", "sem estacionamento"...
        achado = "nao" if re.search(r"\b(nao|sem)\b", chave) else ("sim" if re.search(r"\b(sim|com)\b", chave) else None)
    return achado


def _regras_do_dominio(e: dict) -> list[str]:
    """Valores IMPOSSÍVEIS (ou absurdos) no mundo real -> erro (não prevê)."""
    erros = []
    if e["antecedencia_dias"] < 0:
        erros.append("a antecedência não pode ser negativa (a reserva tem que ser feita antes da chegada)")
    elif e["antecedencia_dias"] > 3650:
        erros.append("uma antecedência de mais de 10 anos não é plausível para uma reserva de hotel")
    if e["preco_medio_diaria"] < 0:
        erros.append("o preço da diária não pode ser negativo")
    elif e["preco_medio_diaria"] > 5000:
        erros.append("uma diária acima de 5.000 euros não é plausível para este hotel")
    if not 0 <= e["pedidos_especiais"] <= 20:
        erros.append("o número de pedidos especiais precisa estar entre 0 e 20")
    if not 1 <= e["mes_chegada"] <= 12:
        erros.append("o mês de chegada precisa estar entre 1 (janeiro) e 12 (dezembro)")
    if e["noites_fim_de_semana"] < 0 or e["noites_semana"] < 0:
        erros.append("o número de noites não pode ser negativo")
    elif e["noites_fim_de_semana"] + e["noites_semana"] == 0:
        erros.append("a estadia precisa ter pelo menos 1 noite")
    elif e["noites_fim_de_semana"] + e["noites_semana"] > 365:
        erros.append("uma estadia de mais de 1 ano não é uma reserva de hotel comum")
    return erros


def preparar(entrada: dict) -> tuple[dict, list[str], list[str]]:
    """Converte tipos (inclusive linguagem natural), corrige caixa/acento e separa
    ERROS (impossível) de ALERTAS (fora da faixa vista no treino)."""
    pronto, erros, alertas = {}, [], []
    for nome, info in ATRIBUTOS.items():
        valor = entrada.get(nome)
        if info["tipo"] == "texto":
            achado = _categoria(nome, valor)
            if achado is None:
                erros.append(f"{nome} '{str(valor)[:40]}' não existe no dataset. Opções: {info['valores']}")
            pronto[nome] = achado
            continue
        try:
            conversor = {"antecedencia_dias": _dias, "mes_chegada": _mes}.get(nome, _numero)
            x = conversor(valor)
        except (TypeError, ValueError):
            erros.append(f"{nome} precisa ser um número (recebido: {str(valor)[:40]!r})")
            continue
        if info["tipo"] == "inteiro" and not x.is_integer():
            erros.append(f"{nome} precisa ser um número inteiro (recebido: {str(valor)[:40]!r})")
            continue
        pronto[nome] = int(x) if info["tipo"] == "inteiro" else round(x, 2)
    if erros:
        return pronto, erros, alertas
    erros += _regras_do_dominio(pronto)
    if erros:
        return pronto, erros, alertas
    for col in NUMERICAS:
        mn, mx = FAIXAS[col]
        if not (mn <= pronto[col] <= mx):
            alertas.append(f"{col}={pronto[col]:g} está fora da faixa do dataset ({mn:g} a {mx:g}); a previsão é uma extrapolação")
    if pronto["preco_medio_diaria"] == 0 and pronto["segmento_mercado"] != "Cortesia":
        alertas.append("diária igual a 0 só é comum em reservas de Cortesia; confira o preço")
    return pronto, erros, alertas


# ---------------------------------------------------------------- modelo
def metricas_explicadas() -> dict:
    """Métricas de teste + as mesmas em linguagem de leigo, JÁ CALCULADAS (o Gemini não faz contas;
    se fizer, a verificação recusa o número)."""
    m = {k: v for k, v in METRICAS.items() if k not in ("sklearn_version", "atributos_do_modelo")}
    percebe = round(100 * METRICAS["recall_teste"])
    m.update({
        "acuracia_percentual": round(100 * METRICAS["acuracia_teste"], 1),
        "de_cada_100_cancelamentos_o_modelo_percebe": percebe,
        "de_cada_100_cancelamentos_o_modelo_deixa_passar": 100 - percebe,
        "de_cada_100_alertas_de_cancelamento_acertam": round(100 * METRICAS["precisao_teste"]),
        "de_cada_100_alertas_de_cancelamento_sao_falsos": 100 - round(100 * METRICAS["precisao_teste"]),
        "de_cada_100_reservas_o_modelo_acerta": round(100 * METRICAS["acuracia_teste"]),
    })
    return m


def prever(entrada: dict) -> dict:
    """Roda o modelo treinado com uma entrada já preparada (use preparar() antes)."""
    X = para_o_modelo(entrada)                          # ponte: legível -> formato do modelo
    probs = {str(_nativo(c)): float(p) for c, p in zip(MODELO.classes_, MODELO.predict_proba(X)[0])}
    p_canc = probs[CLASSE_POSITIVA]
    classe = CLASSE_POSITIVA if p_canc >= LIMIAR else "nao"
    margem = (p_canc - LIMIAR) / (1 - LIMIAR) if p_canc >= LIMIAR else (LIMIAR - p_canc) / LIMIAR
    return {
        "previsao": classe,
        "previsao_por_extenso": "a reserva SERÁ cancelada" if classe == CLASSE_POSITIVA else "a reserva NÃO será cancelada",
        "probabilidade_cancelamento_percentual": round(100 * p_canc, 1),
        "probabilidade_nao_cancelamento_percentual": round(100 * (1 - p_canc), 1),
        "limiar_de_decisao_percentual": round(100 * LIMIAR, 1),
        "confianca": (f"baixa: probabilidade perto do limiar de {LIMIAR_TEXTO}" if margem < 0.3
                    else "moderada" if margem < 0.7 else "alta"),
        "acuracia_teste": float(METRICAS["acuracia_teste"]),
        "f1_teste": float(METRICAS["f1_teste"]),
        "recall_teste": float(METRICAS["recall_teste"]),
        "precisao_teste": float(METRICAS["precisao_teste"]),
    }


def casos_semelhantes(entrada: dict, previsao: str | None = None, quantidade: int = 15) -> dict:
    """Reservas reais mais parecidas: mesmo segmento (e mesma vaga, se houver casos suficientes),
    distância padronizada nas numéricas."""
    base = DF[DF["segmento_mercado"] == entrada["segmento_mercado"]]
    mesma_vaga = base[base["vaga_estacionamento"] == entrada["vaga_estacionamento"]]
    if len(mesma_vaga) >= 3 * quantidade:
        base = mesma_vaga
    if len(base) < quantidade:
        base = DF
    dist = sum(((base[c] - entrada[c]) / DESVIOS[c]) ** 2 for c in NUMERICAS) ** 0.5
    proximos = base.assign(_d=dist).nsmallest(quantidade, "_d").drop(columns="_d")
    reais = proximos[ALVO].astype(str)
    n_canc = int((reais == CLASSE_POSITIVA).sum())
    pct = round(100 * n_canc / len(proximos), 1)
    resultado = {
        "quantidade_de_casos": int(len(proximos)),
        "canceladas_entre_os_casos": n_canc,
        "nao_canceladas_entre_os_casos": int(len(proximos) - n_canc),
        "percentual_cancelado_nos_casos": pct,
        "classe_mais_comum_nos_casos": CLASSE_POSITIVA if n_canc * 2 > len(proximos) else "nao",
        "faixa_antecedencia_dos_casos": [int(proximos["antecedencia_dias"].min()), int(proximos["antecedencia_dias"].max())],
        "faixa_preco_dos_casos": [float(proximos["preco_medio_diaria"].min()), float(proximos["preco_medio_diaria"].max())],
        "exemplos": json.loads(proximos[NUMERICAS + CATEGORICAS + [ALVO]].head(5).to_json(orient="records", force_ascii=False)),
        "taxa_geral_de_cancelamento_percentual": round(100 * float((DF[ALVO] == CLASSE_POSITIVA).mean()), 1),
        "observacao": "os casos vêm do mesmo dataset usado no treino: a comparação é indicativa, não uma prova",
    }
    if previsao is not None:
        resultado["previsao_concorda_com_a_maioria"] = str(previsao) == resultado["classe_mais_comum_nos_casos"]
    return resultado


# ---------------------------------------------------------------- consultas aos dados
def _aplicar_filtros(filtros: list | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Filtros [{coluna, operador, valor}] -> (DataFrame filtrado, filtros aplicados, erros)."""
    df, aplicados, erros = DF, [], []
    for f in filtros or []:
        if not isinstance(f, dict):
            continue
        col, op, valor = str(f.get("coluna", "")), str(f.get("operador", "==")), f.get("valor", "")
        if col not in DF.columns or op not in OPERADORES or str(valor).strip() == "":
            erros.append(f"filtro ignorado: {f}")
            continue
        if col in COLUNAS_NUMERICAS_DF:
            try:
                v = _mes(valor) if col == "mes_chegada" else _numero(valor)
            except ValueError:
                erros.append(f"filtro ignorado (valor não numérico): {f}")
                continue
        else:
            opcoes = {_norm(x): x for x in DF[col].dropna().unique()}
            v = opcoes.get(_norm(valor)) or (_categoria(col, valor) if col in CATEGORIAS else None)
            if v is None or op not in ("==", "!="):
                erros.append(f"filtro ignorado: {col} {op} {valor!r} (opções: {sorted(map(str, opcoes.values()))})")
                continue
        serie = df[col]
        mascara = {"==": serie == v, "!=": serie != v, ">": serie > v, ">=": serie >= v,
                   "<": serie < v, "<=": serie <= v}[op]
        df = df[mascara]
        aplicados.append(f"{col} {op} {_nativo(v)}")
    return df, aplicados, erros


def _ordenar_grupos(indice) -> list:
    return [g for g in ORDEM_FAIXAS if g in indice] if set(indice) <= set(ORDEM_FAIXAS) else list(indice)


def estatisticas(coluna: str, agrupar_por: str = "", filtros: list | None = None) -> dict:
    """Estatísticas REAIS de uma coluna (opcionalmente por grupo e com filtros).
    coluna == ALVO -> contagem e taxa de cancelamento; numérica -> contagem, média, mediana, mín., máx."""
    if coluna not in COLUNAS_CONSULTA:
        return {"erro": f"coluna inválida; opções: {COLUNAS_CONSULTA}"}
    if agrupar_por and agrupar_por not in AGRUPAMENTOS:
        return {"erro": f"agrupamento inválido; opções: {AGRUPAMENTOS}"}
    df, aplicados, erros = _aplicar_filtros(filtros)
    saida = {"coluna": coluna, "reservas_consideradas": int(len(df))}
    if aplicados:
        saida["filtros_aplicados"] = aplicados
    if erros:
        saida["avisos"] = erros
    if df.empty:
        saida["erro"] = "nenhuma reserva atende aos filtros"
        return saida
    if coluna == ALVO:                                   # alvo é classe: conta em vez de tirar média
        classes = df[ALVO].astype(str)
        if agrupar_por and agrupar_por != ALVO:
            cont = pd.crosstab(df[agrupar_por], classes)
            grupos = []
            for g in _ordenar_grupos(cont.index):
                total = int(cont.loc[g].sum())
                canc = int(cont.loc[g].get(CLASSE_POSITIVA, 0))
                grupos.append({agrupar_por: _nativo(g), "total_reservas": total, "canceladas": canc,
                               "nao_canceladas": total - canc,
                               "taxa_cancelamento_percentual": round(100 * canc / total, 2)})
            saida["por_grupo"] = grupos
            if len(grupos) > 1:
                maior = max(grupos, key=lambda x: x["taxa_cancelamento_percentual"])
                menor = min(grupos, key=lambda x: x["taxa_cancelamento_percentual"])
                saida["maior_taxa"] = {agrupar_por: maior[agrupar_por], "taxa_percentual": maior["taxa_cancelamento_percentual"]}
                saida["menor_taxa"] = {agrupar_por: menor[agrupar_por], "taxa_percentual": menor["taxa_cancelamento_percentual"]}
            return saida
        total, canc = int(len(classes)), int((classes == CLASSE_POSITIVA).sum())
        saida.update({"total_reservas": total, "canceladas": canc, "nao_canceladas": total - canc,
                      "taxa_cancelamento_percentual": round(100 * canc / total, 2)})
        return saida
    resumo = ["count", "mean", "median", "min", "max"]
    if agrupar_por:
        tabela = df.groupby(agrupar_por)[coluna].agg(resumo).round(2)
        tabela = tabela.loc[_ordenar_grupos(tabela.index)].reset_index()
        saida["por_grupo"] = json.loads(tabela.to_json(orient="records", force_ascii=False))
        return saida
    serie = df[coluna].agg(resumo).round(2)
    saida["resumo"] = {k: (None if pd.isna(v) else float(v)) for k, v in serie.items()}
    saida["soma"] = float(round(df[coluna].sum(), 2))
    return saida


# ---------------------------------------------------------------- fatos para o RAG
def _pct(x: float) -> str:
    return f"{100 * x:.1f}%".replace(".", ",")


def fatos_do_dataset() -> list[str]:
    """Frases com números reais, geradas do CSV, para a base de conhecimento do RAG.
    Cada frase é um trecho: começa com palavras que as pessoas usariam na pergunta."""
    canc = (DF[ALVO] == CLASSE_POSITIVA)
    m = METRICAS
    fatos = [
        f"Tamanho do dataset: depois da limpeza, o dataset tem {len(DF)} reservas e {DF.shape[1]} colunas. "
        f"Foram canceladas {int(canc.sum())} reservas ({_pct(canc.mean())}) e não canceladas {int((~canc).sum())} ({_pct(1 - canc.mean())}).",
        f"Taxa geral de cancelamento: {_pct(canc.mean())} das reservas foram canceladas.",
        f"Desempenho do modelo no conjunto de teste ({m['reservas_teste']} reservas, com o limiar de decisão escolhido): "
        f"acurácia {m['acuracia_teste']}, F1 {m['f1_teste']}, precisão {m['precisao_teste']}, "
        f"recall (sensibilidade) {m['recall_teste']}, especificidade {m['especificidade_teste']}, "
        f"AUC {m['auc_teste']}. Modelo: {m['modelo']}.",
        f"Erros do modelo no teste: das {m['canceladas_no_teste']} reservas canceladas, o modelo detectou "
        f"{m['cancelamentos_detectados_no_teste']} e perdeu {m['cancelamentos_perdidos_no_teste']}; "
        f"deu {m['falsos_alarmes_no_teste']} falsos alarmes (previu cancelamento em reservas que foram mantidas).",
        f"Limiar de decisão: o modelo prevê cancelamento quando a probabilidade de cancelamento é de "
        f"{LIMIAR_TEXTO} ou mais. O limiar foi escolhido por validação cruzada nos dados de treino "
        f"(maximizando o F1), em vez do 50% padrão. Com 50% o F1 de teste seria {m.get('f1_teste_limiar_0_5')} "
        f"e o recall {m.get('recall_teste_limiar_0_5')}; com o limiar escolhido, F1 {m['f1_teste']} e recall {m['recall_teste']}.",
        f"Comparação com um palpite ingênuo: quem dissesse sempre 'não cancela' acertaria {m['acuracia_de_quem_sempre_diz_nao']} "
        f"das reservas do teste; o modelo acerta {m['acuracia_teste']}. Por isso a acurácia sozinha engana e o F1 é a métrica principal.",
        f"Período dos dados: chegadas de {int(DF['ano_chegada'].min())} e {int(DF['ano_chegada'].max())} "
        f"({int((DF['ano_chegada'] == 2017).sum())} reservas em 2017 e {int((DF['ano_chegada'] == 2018).sum())} em 2018).",
    ]
    if "precisao_teste_limiar_0_5" in m:                  # gerado pelo scripts/treinar.py atual
        fatos.append(
            f"O que mudou com o limiar de decisão: com 50%, o modelo detectava {m['cancelamentos_detectados_no_teste_limiar_0_5']} "
            f"dos {m['canceladas_no_teste']} cancelamentos do teste, com precisão {m['precisao_teste_limiar_0_5']} e "
            f"{m['falsos_alarmes_no_teste_limiar_0_5']} falsos alarmes; com {LIMIAR_TEXTO}, detecta "
            f"{m['cancelamentos_detectados_no_teste']}, com precisão {m['precisao_teste']} e {m['falsos_alarmes_no_teste']} "
            f"falsos alarmes. Ou seja, percebe mais cancelamentos em troca de mais alarmes falsos.")
    for col, nome in [("antecedencia_dias", "antecedência (dias)"), ("preco_medio_diaria", "preço médio da diária (euros)"),
                      ("pedidos_especiais", "pedidos especiais"), ("total_noites", "total de noites")]:
        s = DF[col]
        med = DF.groupby(DF[ALVO])[col].agg(["mean", "median"])
        fatos.append(f"{nome.capitalize()}: varia de {s.min():g} a {s.max():g}, média {s.mean():.2f}, mediana {s.median():g}. "
                     f"Nas canceladas a média é {med.loc['sim', 'mean']:.2f} (mediana {med.loc['sim', 'median']:g}); "
                     f"nas não canceladas, {med.loc['nao', 'mean']:.2f} (mediana {med.loc['nao', 'median']:g}).")
    for col, nome in [("segmento_mercado", "segmento de mercado"), ("plano_refeicao", "plano de refeição"),
                      ("tipo_quarto", "tipo de quarto"), ("vaga_estacionamento", "vaga de estacionamento"),
                      ("hospede_recorrente", "hóspede recorrente"), ("faixa_antecedencia", "faixa de antecedência"),
                      ("pedidos_especiais", "número de pedidos especiais"), ("mes_chegada", "mês de chegada")]:
        g = canc.groupby(DF[col]).agg(["mean", "size"])
        ordem = _ordenar_grupos(g.index) if col == "faixa_antecedencia" else g.sort_values("mean", ascending=False).index
        fatos.append(f"Taxa de cancelamento por {nome}, da maior para a menor: " if col != "faixa_antecedencia"
                     else f"Taxa de cancelamento por {nome}: ")
        rotulo = (lambda k: f"{NOMES_MESES[int(k) - 1]} ({k})") if col == "mes_chegada" else str
        fatos[-1] += "; ".join(f"{rotulo(k)}: {_pct(g.loc[k, 'mean'])} de {int(g.loc[k, 'size'])} reservas" for k in ordem) + "."
    corr = DF[NUMERICAS + ["adultos", "criancas"]].corrwith(canc.astype(int)).sort_values(key=abs, ascending=False)
    fatos.append("Qual variável mais influencia o cancelamento? Correlação (r) de cada variável numérica com o cancelamento: "
                 + "; ".join(f"{k} r = {v:.2f}".replace(".", ",") for k, v in corr.items()) + ".")
    return fatos


_conferencia = conferir_transformacao()
if not _conferencia["ok"]:
    log.error("ATENÇÃO: a ponte para_o_modelo() NÃO reproduz o treino: %s", _conferencia["detalhes"][:3])
