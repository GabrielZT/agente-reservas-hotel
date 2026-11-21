"""
Recuperação (o "R" do RAG): encontra na base de conhecimento os trechos mais relacionados
com a pergunta, usando TF-IDF + similaridade de cosseno (scikit-learn, roda local e de graça).

Base de conhecimento = arquivos .md da pasta knowledge_base/ (descrição das colunas, limpeza, EDA,
modelo, métricas, limitações do trabalho anterior) + frases com números reais geradas do CSV
(dados_modelo.fatos_do_dataset()).

Além do kit (medido com tests/avaliacao_rag.py):
  - duas representações combinadas: TF-IDF de PALAVRAS (1 e 2 palavras) e TF-IDF de pedaços de
    CARACTERES (4 e 5 letras), que casa "cancela" com "cancelamento" e "confiável" com "confiança";
  - expansão da pergunta com sinônimos do domínio ("confiável" -> métricas, limitações, erros...).
"""
import re
import unicodedata
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .dados_modelo import fatos_do_dataset

BASE = Path(__file__).resolve().parent.parent           # raiz do projeto (este arquivo está em src/)
STOPWORDS_PT = """a o as os um uma uns umas de do da dos das no na nos nas em por para com sem sobre
e ou que se qual quais quanto quanta quantos quantas como onde quando é são foi ser ter tem há mais menos
muito pouco me te lhe isso isto esse essa este esta ao aos à às pelo pela pelos pelas meu minha seu sua
vai vou você voce pode posso quem""".split()
# sem acento, igual ao texto depois do strip_accents do TfidfVectorizer
STOPWORDS_PT = sorted({"".join(c for c in unicodedata.normalize("NFKD", w) if not unicodedata.combining(c))
                       for w in STOPWORDS_PT})

# pergunta (sem acento) -> palavras acrescentadas à busca
EXPANSOES = {
    r"confia|acert|err[oa]|precis|bom o modelo|funciona bem": "métricas acurácia F1 recall erros limitações",
    r"leigo|simples|crianca|nao entendi|explica": "explicação para um leigo como entender as métricas",
    r"influenc|importan|fator|pesa|impacta|motivo": "correlação importância por permutação variável mais influencia",
    r"canal|segmento|online|offline|corporativ|cortesia|aviac": "segmento de mercado",
    r"limita|fraco|problema|cuidado|nao sabe": "limitações",
    r"quantas reservas|tamanho|linhas|registros": "tamanho do dataset reservas",
    r"duplicad|limpez|removid|tratamento": "limpeza dos dados",
    r"porque mlp|por que mlp|algoritmo|rede neural|qual modelo": "sobre o modelo por que o MLP",
    r"mes|julho|agosto|dezembro|janeiro|sazonal|estacao|verao|ferias": "taxa de cancelamento por mês de chegada sazonalidade",
    r"deixa passar|perde|nao detecta|falso alarme|falsos": "erros do modelo no teste detectou perdeu falsos alarmes",
    r"ja vier|voltam|recorrente|fiel|ja se hosped": "hóspede recorrente",
    r"anteceden|lead|dias antes|cedo": "antecedência faixa de antecedência",
    r"pedido|especia": "pedidos especiais",
    r"preco|diaria|caro|barato|euro": "preço médio da diária",
}


def _sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)).casefold()


def expandir(pergunta: str) -> str:
    """Acrescenta à pergunta os termos usados na base de conhecimento para o mesmo assunto."""
    base = _sem_acento(pergunta)
    extras = [termos for padrao, termos in EXPANSOES.items() if re.search(padrao, base)]
    return " ".join([pergunta] + extras)


def _trechos_dos_arquivos() -> list[str]:
    """Quebra cada .md em trechos: parágrafos e itens de lista viram trechos separados,
    e cada trecho leva o título da sua seção (ex.: "Limitações: o modelo não conhece...")."""
    trechos = []
    for arquivo in sorted((BASE / "knowledge_base").glob("*.md")):
        secao = ""
        for linha_ou_bloco in re.split(r"\n\s*\n|\n(?=\s*[-*] )|\n(?=#)", arquivo.read_text(encoding="utf-8")):
            bloco = linha_ou_bloco.strip()
            if bloco.startswith("#"):                                     # título de seção
                titulo, _, resto = bloco.partition("\n")
                secao = titulo.strip("# ").strip()
                bloco = resto.strip()
            bloco = bloco.strip(" \n-*")
            if len(bloco) > 25:
                trechos.append(f"{secao}: {bloco}" if secao else bloco)
    return trechos


class Recuperador:
    def __init__(self, combinar: bool = True, expandir_pergunta: bool = True):
        self.combinar, self.expandir_pergunta = combinar, expandir_pergunta
        self.trechos = _trechos_dos_arquivos() + fatos_do_dataset()
        self.palavras = TfidfVectorizer(strip_accents="unicode", lowercase=True,
                                        stop_words=STOPWORDS_PT, ngram_range=(1, 2))
        self.m_palavras = self.palavras.fit_transform(self.trechos)
        self.letras = TfidfVectorizer(strip_accents="unicode", lowercase=True, analyzer="char_wb",
                                      ngram_range=(4, 5), sublinear_tf=True)
        self.m_letras = self.letras.fit_transform(self.trechos)

    def notas(self, pergunta: str):
        consulta = expandir(pergunta) if self.expandir_pergunta else pergunta
        notas = cosine_similarity(self.palavras.transform([consulta]), self.m_palavras)[0]
        if self.combinar:
            notas = 0.5 * notas + 0.5 * cosine_similarity(self.letras.transform([consulta]), self.m_letras)[0]
        return notas

    def buscar(self, pergunta: str, k: int = 5, limiar: float = 0.08) -> list[dict]:
        """Devolve até k trechos com similaridade acima do limiar, do mais para o menos parecido."""
        if not pergunta.strip():
            return []
        notas = self.notas(pergunta)
        ordem = notas.argsort()[::-1][:k]
        return [{"trecho": self.trechos[i], "similaridade": round(float(notas[i]), 3)}
                for i in ordem if notas[i] >= limiar]


RECUPERADOR = Recuperador()
