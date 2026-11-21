"""
Treino do modelo do grupo: previsão de CANCELAMENTO de reservas de hotel (classificação).
Dataset: Hotel Reservations (Kaggle, ahsan81) — o MESMO do Trabalho Prático anterior.

    python scripts/treinar.py

Gera (rodar SEMPRE dentro do venv do kit, para as versões do scikit-learn/numpy baterem):
  data/dataset_legivel.csv         base LIMPA e LEGÍVEL (colunas e categorias em português) -> perguntas, RAG, casos
  models/modelo.joblib             Pipeline completo (ColumnTransformer + MLPClassifier) -> aceita os dados legíveis
  models/metricas.json             métricas REAIS no conjunto de teste
  models/amostra_conferencia.json  5 entradas legíveis + previsão do treino (confere a ponte para_o_modelo)

Mudanças em relação ao notebook do trabalho anterior:
  - mesmas regras de limpeza (duplicatas, 29/02/2018, reservas com 0 noites);
  - colunas e categorias traduzidas para o português (o agente conversa com pessoas);
  - subconjunto de 8 atributos (o enunciado permite até 8): na validação cruzada o MLP com esses 8
    teve F1 0,7020 contra 0,7028 com os 17 originais, ou seja, praticamente o mesmo desempenho;
  - arrival_year ficou de fora: só tem 2017 e 2018 e não faz sentido para reservas futuras;
  - grade do MLP ampliada em alpha (0,05 e 0,1), como sugerido nas melhorias do trabalho anterior.
"""
import datetime
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
os.chdir(Path(__file__).resolve().parent.parent)   # trabalha na raiz do projeto
RANDOM_STATE = 42

# ------------------------------------------------------------------ 1. carga e limpeza (igual ao notebook)
bruto = pd.read_csv("data/hotel_reservations_original.csv")
df = bruto.drop(columns="Booking_ID").drop_duplicates()


def _data_valida(linha) -> bool:
    try:
        datetime.date(int(linha.arrival_year), int(linha.arrival_month), int(linha.arrival_date))
        return True
    except ValueError:
        return False


df = df[df.apply(_data_valida, axis=1)]
df = df[(df.no_of_weekend_nights + df.no_of_week_nights) > 0].reset_index(drop=True)

# ------------------------------------------------------------------ 2. versão LEGÍVEL (português)
NOMES = {
    "no_of_adults": "adultos",
    "no_of_children": "criancas",
    "no_of_weekend_nights": "noites_fim_de_semana",
    "no_of_week_nights": "noites_semana",
    "type_of_meal_plan": "plano_refeicao",
    "required_car_parking_space": "vaga_estacionamento",
    "room_type_reserved": "tipo_quarto",
    "lead_time": "antecedencia_dias",
    "arrival_year": "ano_chegada",
    "arrival_month": "mes_chegada",
    "arrival_date": "dia_chegada",
    "market_segment_type": "segmento_mercado",
    "repeated_guest": "hospede_recorrente",
    "no_of_previous_cancellations": "cancelamentos_anteriores",
    "no_of_previous_bookings_not_canceled": "reservas_anteriores_nao_canceladas",
    "avg_price_per_room": "preco_medio_diaria",
    "no_of_special_requests": "pedidos_especiais",
    "booking_status": "cancelada",
}
leg = df.rename(columns=NOMES)
leg["segmento_mercado"] = leg["segmento_mercado"].map({
    "Online": "Online", "Offline": "Offline", "Corporate": "Corporativo",
    "Complementary": "Cortesia", "Aviation": "Aviação"})
leg["plano_refeicao"] = leg["plano_refeicao"].map({
    "Meal Plan 1": "Café da manhã", "Meal Plan 2": "Meia pensão",
    "Meal Plan 3": "Pensão completa", "Not Selected": "Nenhum"})
leg["tipo_quarto"] = leg["tipo_quarto"].str.replace("Room_Type ", "Tipo ", regex=False)
leg["vaga_estacionamento"] = leg["vaga_estacionamento"].map({1: "sim", 0: "nao"})
leg["hospede_recorrente"] = leg["hospede_recorrente"].map({1: "sim", 0: "nao"})
leg["cancelada"] = leg["cancelada"].map({"Canceled": "sim", "Not_Canceled": "nao"})
leg["total_noites"] = leg["noites_fim_de_semana"] + leg["noites_semana"]
leg["faixa_antecedencia"] = pd.cut(leg["antecedencia_dias"], [-1, 7, 30, 90, 180, 10_000],
                                   labels=["0-7 dias", "8-30 dias", "31-90 dias", "91-180 dias", "mais de 180 dias"]).astype(str)
assert leg.isna().sum().sum() == 0, "algum mapa de tradução deixou valores vazios"
leg.to_csv("data/dataset_legivel.csv", index=False)

# ------------------------------------------------------------------ 3. modelo (Pipeline completo)
NUMERICAS = ["antecedencia_dias", "preco_medio_diaria", "pedidos_especiais", "mes_chegada",
             "noites_fim_de_semana", "noites_semana"]
CATEGORICAS = ["segmento_mercado", "vaga_estacionamento"]
ALVO = "cancelada"

X, y = leg[NUMERICAS + CATEGORICAS], leg[ALVO]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

prep = ColumnTransformer([("num", StandardScaler(), NUMERICAS),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)])
pipe = Pipeline([("prep", prep), ("clf", MLPClassifier(max_iter=1000, random_state=RANDOM_STATE))])
grade = {"clf__hidden_layer_sizes": [(50,), (100,), (100, 50)],
         "clf__alpha": [0.001, 0.01, 0.05, 0.1]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(pipe, grade, cv=cv, n_jobs=-1,
                    scoring={"f1": "f1", "auc": "roc_auc"}, refit="f1")
inicio = time.time()
# pos_label: o scorer "f1" usa a classe positiva = "sim" (maior em ordem alfabética), que é a de interesse
grid.fit(X_tr, (y_tr == "sim").astype(int))
print(f"GridSearchCV: {len(grid.cv_results_['params'])} combinações em {time.time() - inicio:.0f} s; melhor {grid.best_params_}")

# re-treina o melhor Pipeline com o alvo em texto ("sim"/"nao"), para o modelo já devolver a classe legível
melhor = grid.best_estimator_
melhor.fit(X_tr, y_tr)
prev = melhor.predict(X_te)
proba_sim = melhor.predict_proba(X_te)[:, list(melhor.classes_).index("sim")]
vn, fp, fn, vp = confusion_matrix(y_te, prev, labels=["nao", "sim"]).ravel()
parametros = {k.replace("clf__", ""): (list(v) if isinstance(v, tuple) else v) for k, v in grid.best_params_.items()}
metricas = {
    "modelo": f"MLPClassifier (rede neural) {parametros}",
    "acuracia_teste": round(float(accuracy_score(y_te, prev)), 3),
    "f1_teste": round(float(f1_score(y_te, prev, pos_label="sim")), 3),
    "precisao_teste": round(float(precision_score(y_te, prev, pos_label="sim")), 3),
    "recall_teste": round(float(recall_score(y_te, prev, pos_label="sim")), 3),
    "especificidade_teste": round(float(vn / (vn + fp)), 3),
    "auc_teste": round(float(roc_auc_score((y_te == "sim").astype(int), proba_sim)), 3),
    "f1_validacao_cruzada": round(float(grid.best_score_), 3),
    "acuracia_de_quem_sempre_diz_nao": round(float((y_te == "nao").mean()), 3),
    "reservas_teste": int(len(y_te)),
    "canceladas_no_teste": int((y_te == "sim").sum()),
    "cancelamentos_detectados_no_teste": int(vp),
    "cancelamentos_perdidos_no_teste": int(fn),
    "falsos_alarmes_no_teste": int(fp),
    "atributos_do_modelo": NUMERICAS + CATEGORICAS,
    "sklearn_version": sklearn.__version__,
}
joblib.dump(melhor, "models/modelo.joblib")

# Amostra para conferir a PONTE legível -> modelo (dados_modelo.conferir_transformacao)
amostra = [{"entrada": {k: (v.item() if hasattr(v, "item") else v) for k, v in linha.items()},
            "previsao_no_treino": (p.item() if hasattr(p, "item") else p)}
           for (_, linha), p in zip(X_te.head(5).iterrows(), melhor.predict(X_te.head(5)))]
json.dump(amostra, open("models/amostra_conferencia.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(metricas, open("models/metricas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(metricas, ensure_ascii=False, indent=1))
print("Classes:", leg[ALVO].value_counts().to_dict(), "| linhas:", len(leg), "| numpy", np.__version__)
