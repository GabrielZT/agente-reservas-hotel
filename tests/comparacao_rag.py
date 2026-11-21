"""
Comparação para o relatório: as MESMAS perguntas respondidas
  (a) pelo Gemini "puro", só com a persona (sem RAG, sem modelo, sem verificação), e
  (b) pelo nosso agente completo.

    python tests/comparacao_rag.py            (usa a chave do .env; gasta ~3 chamadas por pergunta)

Perguntas do nosso dataset (hotel); as diferenças vão para a seção "com RAG x sem RAG" do relatório.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # importa os módulos da raiz do projeto

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from google.genai import types  # noqa: E402

from src import pipeline  # noqa: E402

PERGUNTAS = [
    "Qual segmento de mercado tem a maior taxa de cancelamento?",
    "Qual a taxa de cancelamento geral do dataset?",
    "Uma reserva online feita 200 dias antes, diária de 150 euros, sem pedidos especiais, chegada em agosto, "
    "2 noites de fim de semana e 3 de semana, sem vaga de estacionamento, vai ser cancelada?",
    "Qual variável mais influencia o cancelamento?",
    "Esse modelo é confiável?",
]

for p in PERGUNTAS:
    puro = pipeline._gemini(pipeline.INSTRUCOES, [types.Content(role="user", parts=[types.Part(text=p)])])
    completo = pipeline.responder(p, [])
    print("=" * 80, "\nPERGUNTA:", p)
    print("\n[SEM RAG]\n", puro)
    print("\n[AGENTE COMPLETO]  etapas:", " · ".join(completo["etapas"]), "\n", completo["resposta"])
