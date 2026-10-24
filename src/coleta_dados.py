"""Coleta e controle de qualidade dos dados do projeto.

Baixa quatro fontes públicas e gratuitas, de janeiro de 2000 até a última data disponível:

    - Ibovespa (^BVSP): abertura, máxima, mínima, fechamento -> Yahoo Finance (yfinance)
    - Dólar/real (PTAX venda)  -> Banco Central do Brasil, SGS série 1
    - Meta Selic               -> Banco Central do Brasil, SGS série 432
    - VIX (VIXCLS)             -> FRED (Federal Reserve Bank of St. Louis)

Seguindo o plano de dados:
    - o calendário da B3 (dias em que o Ibovespa negocia) é a base da amostra;
    - nos dias em que só o outro mercado funcionou, repete-se a última observação
      disponível (usa só informação do passado, então não há viés de antecipação);
    - NADA é interpolado e nenhuma linha é apagada: linhas com preços inválidos
      são apenas sinalizadas na coluna `ohlc_invalido`;
    - a contagem de dados faltantes de cada série é registrada em data/coleta_log.txt.

Uso:
    python src/coleta_dados.py
"""

from __future__ import annotations

import datetime as dt
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
INICIO = "2000-01-01"
RAIZ = Path(__file__).resolve().parent.parent
PASTA_BRUTO = RAIZ / "data" / "raw"
PASTA_FINAL = RAIZ / "data" / "processed"
ARQ_LOG = RAIZ / "data" / "coleta_log.txt"

# A API do BCB limita consultas diárias a janelas de 10 anos; usamos blocos de 9.
ANOS_POR_BLOCO = 9
CABECALHO = {"User-Agent": "Mozilla/5.0 (projeto academico UFJF)"}


def _get(
    url: str,
    params: dict | None = None,
    tentativas: int = 3,
    timeout: int = 60,
    headers: dict | None = None,
) -> requests.Response:
    """GET com timeout e novas tentativas simples."""
    ultimo_erro: Exception | None = None
    cab = CABECALHO if headers is None else headers
    for i in range(tentativas):
        try:
            resp = requests.get(url, params=params, headers=cab, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as erro:  # noqa: PERF203
            ultimo_erro = erro
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"Falha ao acessar {url}: {ultimo_erro}")


def _get_json(url: str, params: dict | None = None, tentativas: int = 4) -> list:
    """GET que exige resposta em JSON.

    A API do BCB às vezes responde com uma página vazia ou de erro (sobretudo
    depois de vários pedidos seguidos). Nesse caso esperamos e tentamos de novo;
    se continuar falhando, mostramos o começo da resposta para diagnóstico.
    """
    resp: requests.Response | None = None
    for i in range(tentativas):
        resp = _get(url, params)
        try:
            return resp.json()
        except ValueError:
            time.sleep(5 * (i + 1))
    trecho = (resp.text[:300] if resp is not None else "").strip() or "(resposta vazia)"
    raise RuntimeError(
        f"O BCB não devolveu JSON para {url} {params} após {tentativas} tentativas.\n"
        f"Status HTTP: {resp.status_code if resp is not None else '?'}\n"
        f"Início da resposta: {trecho}"
    )


# ---------------------------------------------------------------------------
# Download de cada fonte
# ---------------------------------------------------------------------------
def baixar_ibovespa(inicio: str = INICIO) -> pd.DataFrame:
    """Ibovespa diário (open, high, low, close) pelo Yahoo Finance."""
    import yfinance as yf

    df = yf.download("^BVSP", start=inicio, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("O Yahoo Finance não retornou dados para ^BVSP.")
    # Versões novas do yfinance devolvem colunas em dois níveis; ficamos com o primeiro.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].rename(
        columns={"Open": "abertura", "High": "maxima", "Low": "minima", "Close": "fechamento"}
    )
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "data"
    return df.sort_index()


def baixar_sgs(codigo: int, nome: str, inicio: str = INICIO) -> pd.Series:
    """Série do SGS/Banco Central, em blocos para respeitar o limite de 10 anos."""
    hoje = dt.date.today()
    inicio_dt = dt.date.fromisoformat(inicio)
    partes: list[pd.DataFrame] = []
    ini = inicio_dt
    while ini <= hoje:
        fim = min(dt.date(ini.year + ANOS_POR_BLOCO, ini.month, 1) - dt.timedelta(days=1), hoje)
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
        params = {
            "formato": "json",
            "dataInicial": ini.strftime("%d/%m/%Y"),
            "dataFinal": fim.strftime("%d/%m/%Y"),
        }
        dados = _get_json(url, params)
        if dados:
            partes.append(pd.DataFrame(dados))
        ini = fim + dt.timedelta(days=1)
        time.sleep(1)  # pausa curta entre pedidos para não sobrecarregar a API
    if not partes:
        raise RuntimeError(f"O SGS não retornou dados para a série {codigo}.")
    df = pd.concat(partes, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    serie = df.drop_duplicates("data").set_index("data")["valor"].sort_index()
    serie.name = nome
    return serie


def baixar_vix(inicio: str = INICIO) -> pd.Series:
    """VIX (VIXCLS) pelo FRED, em CSV, sem necessidade de chave de API."""
    # O FRED às vezes demora ou corta conexões que parecem automáticas. Tentamos
    # identificações diferentes: a padrão da biblioteca requests e a de um navegador
    # comum. Pedimos só o período da amostra (cosd = data inicial).
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {"id": "VIXCLS", "cosd": inicio}
    opcoes_cabecalho = [
        {},  # identificação padrão do requests
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/csv,text/plain,*/*",
        },
    ]
    erros: list[str] = []
    resp = None
    for cab in opcoes_cabecalho:
        try:
            resp = _get(url, params, tentativas=2, timeout=120, headers=cab)
            break
        except RuntimeError as erro:
            erros.append(str(erro))
    if resp is None:
        raise RuntimeError("Não foi possível baixar o VIX do FRED:\n" + "\n".join(erros))
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["data", "vix"]  # o nome da 1a coluna mudou entre versões do FRED
    df["data"] = pd.to_datetime(df["data"])
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")  # "." vira NaN
    serie = df.set_index("data")["vix"].sort_index()
    return serie.loc[inicio:]


# ---------------------------------------------------------------------------
# Controle de qualidade e montagem da base
# ---------------------------------------------------------------------------
def sinalizar_ohlc_invalido(df: pd.DataFrame) -> pd.Series:
    """True quando os preços violam a lógica básica (sem apagar a linha).

    Regras: preços positivos; máxima >= abertura, fechamento e mínima;
    mínima <= abertura, fechamento e máxima.
    """
    o, h, l, c = df["abertura"], df["maxima"], df["minima"], df["fechamento"]
    positivos = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    logica = (h >= o) & (h >= c) & (h >= l) & (l <= o) & (l <= c)
    return ~(positivos & logica) | df[["abertura", "maxima", "minima", "fechamento"]].isna().any(axis=1)


def _log_retorno(preco: pd.Series) -> pd.Series:
    """Log-retorno diário; fica NaN se algum dos dois preços não for positivo."""
    razao = preco / preco.shift(1)
    return np.log(razao.where(razao > 0))


def montar_base(
    ibov: pd.DataFrame, ptax: pd.Series, selic: pd.Series, vix: pd.Series
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Junta tudo no calendário da B3 e conta o que faltava antes de repetir a última observação."""
    base = ibov.copy()
    base["ohlc_invalido"] = sinalizar_ohlc_invalido(base)

    faltantes: dict[str, int] = {}
    for serie in (ptax, selic, vix):
        # Valor da série em cada dia da B3 (NaN onde o outro mercado não funcionou).
        alinhada = serie.reindex(base.index)
        faltantes[serie.name] = int(alinhada.isna().sum())
        # Última observação disponível até a data (só passado): sem interpolação.
        # Primeiro juntamos o histórico completo da série para o ffill enxergar
        # observações de dias que a B3 não negociou.
        completa = serie.reindex(serie.index.union(base.index)).ffill()
        base[serie.name] = completa.reindex(base.index)

    base["retorno_log"] = _log_retorno(base["fechamento"])
    base["selic_var"] = base["selic_meta"].diff()
    base["dolar_var_log"] = _log_retorno(base["ptax_venda"])
    return base, faltantes


def escrever_log(base: pd.DataFrame, faltantes: dict[str, int], hoje: dt.date) -> str:
    linhas = [
        f"Data da coleta (acesso): {hoje.strftime('%d/%m/%Y')}",
        f"Período da base: {base.index.min().date()} a {base.index.max().date()}",
        f"Observações (dias de negociação da B3): {len(base)}",
        f"Linhas com preços inconsistentes (sinalizadas, não removidas): {int(base['ohlc_invalido'].sum())}",
        "Dias da B3 sem observação própria na série (preenchidos com a última disponível):",
    ]
    linhas += [f"  - {nome}: {n}" for nome, n in faltantes.items()]
    sobra = int(base[["ptax_venda", "selic_meta", "vix"]].isna().any(axis=1).sum())
    linhas.append(f"Linhas ainda com algum valor ausente (início da amostra): {sobra}")
    return "\n".join(linhas) + "\n"


def main() -> None:
    hoje = dt.date.today()
    PASTA_BRUTO.mkdir(parents=True, exist_ok=True)
    PASTA_FINAL.mkdir(parents=True, exist_ok=True)

    print("Baixando Ibovespa (Yahoo Finance)...")
    ibov = baixar_ibovespa()
    print("Baixando PTAX venda (BCB SGS 1)...")
    ptax = baixar_sgs(1, "ptax_venda")
    print("Baixando meta Selic (BCB SGS 432)...")
    selic = baixar_sgs(432, "selic_meta")
    print("Baixando VIX (FRED)...")
    vix = baixar_vix()

    ibov.to_csv(PASTA_BRUTO / "ibovespa_yahoo.csv")
    ptax.to_csv(PASTA_BRUTO / "ptax_venda_sgs1.csv")
    selic.to_csv(PASTA_BRUTO / "selic_meta_sgs432.csv")
    vix.to_csv(PASTA_BRUTO / "vix_fred.csv")

    base, faltantes = montar_base(ibov, ptax, selic, vix)
    base.to_csv(PASTA_FINAL / "dados_mercado.csv")

    log = escrever_log(base, faltantes, hoje)
    ARQ_LOG.write_text(log, encoding="utf-8")
    print("\n" + log)
    print(f"Base final salva em {PASTA_FINAL / 'dados_mercado.csv'}")


if __name__ == "__main__":
    main()
