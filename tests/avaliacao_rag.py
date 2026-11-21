"""
Avaliação da BUSCA do RAG (além do kit), sem gastar cota do Gemini:   python tests/avaliacao_rag.py

Para cada pergunta de teste há um pedaço de texto que o trecho CERTO contém. Medimos:
  - acerto@k: o trecho certo veio entre os k primeiros?
  - MRR: média de 1/posição do trecho certo (1 = sempre em 1º lugar)
e comparamos a busca do kit (só TF-IDF de palavras) com a nossa (palavras + letras + sinônimos).
Perguntas fora do escopo devem trazer NENHUM trecho acima do limiar.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # importa os módulos da raiz do projeto

import os

os.environ.setdefault("GEMINI_API_KEY", "teste-offline")
from src.busca_rag import Recuperador  # noqa: E402

TESTES = [
    ("qual segmento cancela mais?", "Taxa de cancelamento por segmento de mercado"),
    ("o modelo é confiável?", "Desempenho do modelo no conjunto de teste"),
    ("quais as limitações do modelo?", "Limitações: O modelo erra"),
    ("qual variável mais influencia o cancelamento?", "Qual variável mais influencia o cancelamento?"),
    ("o que é recall?", "Recall (sensibilidade)"),
    ("explique para um leigo como o modelo funciona", "Explicação para um leigo"),
    ("quantas reservas tem o dataset?", "Tamanho do dataset"),
    ("por que vocês removeram duplicatas?", "Duplicatas: 10.275"),
    ("por que escolheram o MLP?", "Por que o MLP?"),
    ("em que mês as pessoas cancelam mais?", "Taxa de cancelamento por mês de chegada"),
    ("pedidos especiais diminuem o cancelamento?", "Taxa de cancelamento por número de pedidos especiais"),
    ("reservar com muita antecedência aumenta o cancelamento?", "Taxa de cancelamento por faixa de antecedência"),
    ("o que significa o segmento offline?", "segmento_mercado: canal de origem"),
    ("qual o preço médio da diária?", "Preço médio da diária (euros): varia"),
    ("quem pede vaga de estacionamento cancela menos?", "Taxa de cancelamento por vaga de estacionamento"),
    ("por que usaram só 8 atributos?", "Por que só 8 atributos?"),
    ("hóspedes que já vieram antes cancelam?", "Taxa de cancelamento por hóspede recorrente"),
    ("quantos cancelamentos o modelo deixa passar?", "Erros do modelo no teste"),
    ("a acurácia não engana?", "Comparação com um palpite ingênuo"),
    ("de que anos são os dados?", "Período dos dados"),
]
FORA = ["receita de bolo de chocolate", "quem ganhou a copa do mundo?", "me conte uma piada"]


def avaliar(rec: Recuperador, k: int = 5) -> dict:
    acertos1 = acertosk = rr = 0
    falhas = []
    for pergunta, esperado in TESTES:
        notas = rec.notas(pergunta)
        ordem = list(notas.argsort()[::-1])
        posicao = next((i + 1 for i, j in enumerate(ordem) if esperado.casefold() in rec.trechos[j].casefold()), None)
        acertos1 += posicao == 1
        acertosk += posicao is not None and posicao <= k
        rr += 1 / posicao if posicao else 0
        if not posicao or posicao > k:
            falhas.append(f"{pergunta!r} -> posição {posicao}")
    fora_ok = sum(not rec.buscar(p) for p in FORA)
    n = len(TESTES)
    return {"acerto@1": round(acertos1 / n, 2), f"acerto@{k}": round(acertosk / n, 2), "MRR": round(rr / n, 3),
            "fora_do_escopo_sem_trecho": f"{fora_ok}/{len(FORA)}", "falhas": falhas}


if __name__ == "__main__":
    for nome, rec in [("kit (só palavras)", Recuperador(combinar=False, expandir_pergunta=False)),
                      ("palavras + letras", Recuperador(combinar=True, expandir_pergunta=False)),
                      ("palavras + sinônimos", Recuperador(combinar=False, expandir_pergunta=True)),
                      ("nossa (palavras + letras + sinônimos)", Recuperador())]:
        print(f"{nome:40s}", avaliar(rec))
