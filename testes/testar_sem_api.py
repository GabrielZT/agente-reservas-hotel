"""Testa dados, modelo, RAG e verificação SEM gastar cota do Gemini:  python testes/testar_sem_api.py
Monta os exemplos a partir da 1ª linha do CSV e testa também: erros de domínio do hotel, linguagem natural
("2 semanas", "julho", "sem" = 0), filtros nas consultas e o conversor de números (com assert)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # importa os módulos da raiz do projeto

import os
os.environ.setdefault("GEMINI_API_KEY", "teste-offline")       # o Gemini não é chamado aqui

import ferramentas as F
from agente import verificar
from recuperacao import RECUPERADOR

print("Conferência da ponte legível -> modelo:", F.conferir_transformacao())
linha = F.DF.iloc[0]
exemplo = {a: str(F._nativo(linha[a])) for a in F.ATRIBUTOS}
print("0. Exemplo (1ª linha do CSV):", exemplo)
print("1. Faltando:", F.faltando({a: v for a, v in list(exemplo.items())[:-2]}))
entrada, erros, alertas = F.preparar(exemplo)
print("2. Preparar:", entrada, erros, alertas)
assert not erros, f"a 1ª linha do CSV deveria ser válida: {erros}"
p = F.prever(entrada)
print("3. Previsão:", p)
print("4. Casos semelhantes:", {k: v for k, v in F.casos_semelhantes(entrada, p["previsao"]).items() if k != "casos"})
num = F.NUMERICAS[0]
print("5. Fora da faixa (alerta):", F.preparar({**exemplo, num: "600"})[2])
if F.CATEGORICAS:
    print("6. Erro (categoria):", F.preparar({**exemplo, F.CATEGORICAS[0]: "Categoria-Que-Nao-Existe"})[1])
print("7. Estatísticas:", F.estatisticas(F.ALVO, F.AGRUPAMENTOS[0] if F.AGRUPAMENTOS else ""))
print("8. RAG:")
for pergunta in ["qual variável mais influencia o cancelamento?", "o modelo é confiável?", "qual segmento cancela mais?", "receita de bolo"]:
    print("  ", pergunta, "->", [(t["similaridade"], t["trecho"][:60]) for t in RECUPERADOR.buscar(pergunta)])
print("9. Verificação OK  :", verificar(f"Métricas: {list(F.METRICAS.values())[1]}".replace(".", ","), {"m": F.METRICAS}, []))
print("10. Verificação FALHA:", verificar("O valor é 9.999,00", {"previsao_do_modelo": p}, []))
impossiveis = {"antecedência negativa": {"antecedencia_dias": "-5"}, "mês 13": {"mes_chegada": "13"},
               "zero noites": {"noites_fim_de_semana": "0", "noites_semana": "0"}, "diária negativa": {"preco_medio_diaria": "-10"},
               "diária absurda": {"preco_medio_diaria": "1 milhão"}}
for nome, troca in impossiveis.items():
    erros = F.preparar({**exemplo, **troca})[1]
    print(f"11. Erro de domínio ({nome}):", erros)
    assert erros, f"{nome} deveria virar erro"
# "sem"/"não" = 0, mas um número dito junto vale mais ("não sei, uns dois" = 2); "semana(s)" não é "sem"
for texto, esperado in [("sem", 0), ("sem pedidos especiais", 0), ("não fez nenhum", 0), ("não", 0),
                        ("não sei, uns dois", 2), ("grátis", 0), ("3", 3)]:
    obtido = F._numero(texto)
    assert obtido == esperado, f"_numero({texto!r}) = {obtido}, esperado {esperado}"
assert F._dias("uma semana") == 7 and F._dias("duas semanas") == 14
print("11b. 'sem'/'não' = 0: ok")
print("12. Linguagem natural:", F.preparar({**exemplo, "antecedencia_dias": "3 semanas", "mes_chegada": "agosto",
                                             "pedidos_especiais": "dois", "vaga_estacionamento": "não precisa"})[0])
print("13. Filtros:", {k: v for k, v in F.estatisticas("cancelada", "", [{"coluna": "segmento_mercado", "operador": "==", "valor": "online"},
                                                                       {"coluna": "antecedencia_dias", "operador": ">", "valor": "100"}]).items()})
