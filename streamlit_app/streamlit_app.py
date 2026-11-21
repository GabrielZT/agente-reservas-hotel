"""
Versão STREAMLIT da página de chat: usa o MESMO agente (src/pipeline.py), só troca a interface.

Rodar localmente (na raiz do projeto, com o .env criado):
    streamlit run streamlit_app/streamlit_app.py
Publicar no Streamlit Community Cloud: veja o README (seção "Publicar no Streamlit").
"""
import os
import sys
import time
from collections import deque
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))                        # para importar o pacote src/ da raiz do projeto

try:                                                 # no Streamlit Cloud a chave vem de st.secrets
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:                                    # localmente, sem secrets.toml: usa o .env
    pass
from dotenv import load_dotenv                       # noqa: E402

load_dotenv(RAIZ / ".env")
if not os.environ.get("GEMINI_API_KEY", "").strip() or os.environ["GEMINI_API_KEY"].startswith("cole-aqui"):
    st.error("Configure a GEMINI_API_KEY (arquivo .env local ou Secrets do Streamlit Cloud).")
    st.stop()

from google.genai import errors                      # noqa: E402
from src import pipeline                             # noqa: E402

MAX_CARACTERES = 1000
LIMITE_SESSAO = (40, 600)                            # 40 mensagens a cada 10 min por sessão
LIMITE_GLOBAL = (150, 86400)                         # 150 mensagens/dia no app todo (protege a cota)


@st.cache_resource
def _contador_global() -> deque:
    return deque()                                   # compartilhado entre todas as sessões do app


def _excedeu(fila: deque, limite: tuple[int, int]) -> bool:
    maximo, janela = limite
    agora = time.time()
    while fila and agora - fila[0] > janela:
        fila.popleft()
    if len(fila) >= maximo:
        return True
    fila.append(agora)
    return False


st.set_page_config(page_title="Assistente de Reservas", page_icon="🏨")
st.title("🏨 Assistente de Reservas")
st.caption("Pergunte sobre as reservas do hotel ou peça a previsão de cancelamento de uma reserva.")

if "historico" not in st.session_state:
    st.session_state.historico = []                  # [{"papel": "user"|"model", "texto": ..., "etapas": [...]}]
    st.session_state.acessos = deque()
if st.button("🧹 Nova conversa"):
    st.session_state.historico = []

with st.chat_message("assistant"):
    st.write("Olá! Posso responder perguntas sobre o dataset de reservas de hotel ou estimar se uma reserva vai ser cancelada.")
for item in st.session_state.historico:
    with st.chat_message("user" if item["papel"] == "user" else "assistant"):
        st.text(item["texto"])                       # st.text: mostra como texto puro (sem Markdown/HTML)
        if item.get("etapas"):
            st.caption("🔧 " + " · ".join(item["etapas"]))

pergunta = st.chat_input("Ex.: Qual segmento de mercado cancela mais?", max_chars=MAX_CARACTERES)
if pergunta:
    with st.chat_message("user"):
        st.text(pergunta)
    if _excedeu(_contador_global(), LIMITE_GLOBAL) or _excedeu(st.session_state.acessos, LIMITE_SESSAO):
        st.warning("Limite de mensagens atingido. Aguarde alguns minutos.")
        st.stop()
    historico = [{"papel": h["papel"], "texto": h["texto"]} for h in st.session_state.historico]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Consultando os dados..."):
                resultado = pipeline.responder(pergunta, historico)
        except errors.APIError as e:
            st.error("Limite gratuito do Gemini atingido. Tente de novo em instantes." if getattr(e, "code", None) == 429
                     else f"O agente está indisponível no momento (Gemini {getattr(e, 'code', '?')}).")
            st.stop()
        except Exception as e:
            print("Falha inesperada no agente:", repr(e))   # aparece no log do Streamlit Cloud
            st.error("Erro interno no agente. Veja o log do app.")
            st.stop()
        st.text(resultado["resposta"])
        st.caption("🔧 " + " · ".join(resultado["etapas"]))
        print("pergunta=", repr(pergunta[:80]), "etapas=", resultado["etapas"])   # log, como no Flask
    st.session_state.historico += [{"papel": "user", "texto": pergunta},
                                   {"papel": "model", "texto": resultado["resposta"], "etapas": resultado["etapas"]}]
