"""Construção das variáveis do projeto (Etapa 2, Passo 1).

Lê a base gerada por `coleta_dados.py` e cria:

    - a variância diária de Garman-Klass (1980) e a volatilidade anualizada em %;
    - os alvos de previsão para os horizontes de 1 e 22 dias úteis;
    - os componentes do HAR (Corsi, 2009): volatilidade média em 1, 5 e 22 dias;
    - os preditores exógenos (retorno, VIX, dólar, Selic);
    - sinalizações de qualidade (amplitude zero e picos suspeitos), sem apagar nada.

Regra de ouro (sem viés de antecipação): na linha do dia t, todo PREDITOR usa só
informação até t, e todo ALVO usa só os dias t+1 em diante. O script confere isso
automaticamente antes de salvar.

Uso:
    python src/features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
ARQ_BASE = RAIZ / "data" / "processed" / "dados_mercado.csv"
ARQ_SAIDA = RAIZ / "data" / "processed" / "features.csv"

DIAS_UTEIS_ANO = 252          # convenção para anualizar a volatilidade diária
HORIZONTES = (1, 22)          # dias úteis à frente
JANELAS_HAR = {"d": 1, "s": 5, "m": 22}  # diária, semanal e mensal

# Regra de pico suspeito (possível erro de dado). Serve só para separar dias a
# inspecionar; não altera nenhum valor e não entra em nenhum modelo (olha dias futuros).
#
# Histórico (registrado no relatório da Etapa 2):
#   - Regra original, definida antes de olhar os casos: amplitude do dia,
#     ln(máxima/mínima), maior que 5x a mediana da amplitude dos 20 vizinhos.
#     Falhou: marcou dois choques reais (18/05/2017 e 05/10/2026) e não marcou
#     os dois dias suspeitos conhecidos (24/11/2000 e 20/03/2001).
#   - Regra revisada (24/10/2026), desenhada DEPOIS de ver os casos: amplitude
#     maior que 3x a dos vizinhos E corpo do dia, |ln(fechamento/abertura)|,
#     menor que 10% da amplitude. Em choques reais o preço anda e fica lá (corpo
#     de 75% a 96% da amplitude); nos suspeitos aparece uma máxima ou mínima
#     extrema e o dia fecha quase onde abriu (corpo de 1% a 7%).
LIMITE_PICO = 3.0
LIMITE_CORPO = 0.10
VIZINHOS_PICO = 10


# ---------------------------------------------------------------------------
# Volatilidade
# ---------------------------------------------------------------------------
def variancia_garman_klass(df: pd.DataFrame) -> pd.Series:
    """Variância diária de Garman-Klass.

    sigma2 = 0,5 * [ln(H/L)]^2 - (2*ln2 - 1) * [ln(C/O)]^2

    Supõe média zero e ignora o salto entre o fechamento de ontem e a abertura de hoje.
    Nunca é negativa, porque |ln(C/O)| <= ln(H/L) em preços válidos.
    """
    log_hl = np.log(df["maxima"] / df["minima"])
    log_co = np.log(df["fechamento"] / df["abertura"])
    var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return var.rename("var_gk")


def para_vol_anual(variancia: pd.Series) -> pd.Series:
    """Converte variância diária em volatilidade anualizada em %."""
    return 100 * np.sqrt(DIAS_UTEIS_ANO * variancia)


def media_passada(var: pd.Series, k: int) -> pd.Series:
    """Média da variância nos dias t-k+1 a t (só passado e presente).

    Diferença declarada em relação a Corsi (2009): lá, os componentes semanal e
    mensal são médias das VOLATILIDADES diárias; aqui, são médias das VARIÂNCIAS,
    convertidas depois em volatilidade, para seguir a mesma construção do alvo.
    """
    return var.rolling(k, min_periods=k).mean()


def media_futura(var: pd.Series, h: int) -> pd.Series:
    """Média da variância nos dias t+1 a t+h (só futuro).

    rolling(h) no dia t+h cobre t+1..t+h; shift(-h) traz esse valor para a linha t.
    """
    return var.rolling(h, min_periods=h).mean().shift(-h)


# ---------------------------------------------------------------------------
# Sinalizações de qualidade
# ---------------------------------------------------------------------------
def sinalizar_picos(df: pd.DataFrame) -> pd.DataFrame:
    """Marca dias com amplitude muito acima da dos vizinhos (para inspeção)."""
    amplitude = np.log(df["maxima"] / df["minima"])
    # Uma coluna para cada vizinho: os 10 dias anteriores e os 10 posteriores.
    vizinhos = pd.concat(
        [amplitude.shift(i) for i in range(1, VIZINHOS_PICO + 1)]
        + [amplitude.shift(-i) for i in range(1, VIZINHOS_PICO + 1)],
        axis=1,
    )
    mediana_viz = vizinhos.median(axis=1)  # nas pontas da amostra, usa os vizinhos que existem
    razao = amplitude / mediana_viz
    corpo = np.log(df["fechamento"] / df["abertura"]).abs() / amplitude
    return pd.DataFrame(
        {
            "amplitude_zero": df["maxima"] == df["minima"],
            "razao_amplitude": razao,
            "corpo_amplitude": corpo,
            "pico_suspeito": (razao > LIMITE_PICO) & (corpo < LIMITE_CORPO),
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------
def construir_features(base: pd.DataFrame) -> pd.DataFrame:
    """Cria todas as variáveis a partir da base de mercado."""
    out = pd.DataFrame(index=base.index)

    var = variancia_garman_klass(base)
    out["var_gk"] = var
    out["vol_gk"] = para_vol_anual(var)

    # Alvos: volatilidade média nos próximos h dias.
    for h in HORIZONTES:
        out[f"alvo_h{h}"] = para_vol_anual(media_futura(var, h))

    # Componentes do HAR: volatilidade média nos últimos 1, 5 e 22 dias.
    for nome, k in JANELAS_HAR.items():
        out[f"har_{nome}"] = para_vol_anual(media_passada(var, k))

    # Preditores exógenos (todos conhecidos no fechamento de t).
    out["retorno_log"] = base["retorno_log"]
    out["log_vix"] = np.log(base["vix"])
    out["dolar_var_log"] = base["dolar_var_log"]
    out["selic_meta"] = base["selic_meta"]
    out["selic_var"] = base["selic_var"]

    # Sinalizações (não alteram nenhum valor).
    out["ohlc_invalido"] = base["ohlc_invalido"].astype(bool)
    out = out.join(sinalizar_picos(base))
    return out


# ---------------------------------------------------------------------------
# Verificações automáticas
# ---------------------------------------------------------------------------
PREDITORES = ["har_d", "har_s", "har_m", "retorno_log", "log_vix", "dolar_var_log", "selic_meta", "selic_var"]


def verificar(base: pd.DataFrame, feats: pd.DataFrame) -> list[str]:
    """Confere a construção e devolve a lista de checagens que passaram."""
    ok: list[str] = []

    # 1) Variância de Garman-Klass nunca negativa.
    assert (feats["var_gk"].dropna() >= 0).all(), "Variância de Garman-Klass negativa."
    ok.append("variância de Garman-Klass nunca negativa")

    # 2) Preditores não usam o futuro: recalcular com a base cortada num dia
    #    qualquer tem de dar exatamente os mesmos valores até esse dia.
    for corte in ["2005-06-30", "2012-03-15", "2020-03-16"]:
        parcial = construir_features(base.loc[:corte])
        a = parcial[PREDITORES]
        b = feats.loc[:corte, PREDITORES]
        pd.testing.assert_frame_equal(a, b, check_freq=False)
    ok.append("preditores idênticos quando a base é cortada (não usam dados futuros)")

    # 3) Alvos usam exatamente os dias t+1..t+h, conferido à mão em datas sorteadas.
    rng = np.random.default_rng(42)
    posicoes = rng.integers(30, len(feats) - 30, size=50)
    var = feats["var_gk"].to_numpy()
    for h in HORIZONTES:
        for p in posicoes:
            esperado = 100 * np.sqrt(DIAS_UTEIS_ANO * var[p + 1 : p + 1 + h].mean())
            assert np.isclose(feats[f"alvo_h{h}"].iloc[p], esperado), f"Alvo h={h} errado na posição {p}."
        # As últimas h linhas não podem ter alvo.
        assert feats[f"alvo_h{h}"].iloc[-h:].isna().all()
        assert feats[f"alvo_h{h}"].iloc[:-h].notna().all()
    ok.append("alvos = média de t+1 a t+h (50 datas sorteadas) e vazios nas últimas h linhas")

    # 4) O componente diário do HAR é a volatilidade do próprio dia.
    assert np.allclose(feats["har_d"], feats["vol_gk"], equal_nan=True)
    ok.append("componente diário do HAR = volatilidade do dia")
    return ok


def resumo(feats: pd.DataFrame) -> str:
    linhas = [
        f"Período: {feats.index.min().date()} a {feats.index.max().date()} ({len(feats)} dias)",
        f"Dias com amplitude zero (máxima = mínima): {int(feats['amplitude_zero'].sum())}",
        f"Dias com pico suspeito (amplitude > {LIMITE_PICO:g}x a dos {2 * VIZINHOS_PICO} vizinhos "
        f"e corpo < {LIMITE_CORPO:.0%} da amplitude): {int(feats['pico_suspeito'].sum())} -> "
        + ", ".join(d.strftime("%d/%m/%Y") for d in feats.index[feats["pico_suspeito"]]),
        "  (sinalizados, não alterados: valores mantidos na análise principal)",
    ]
    for h in HORIZONTES:
        linhas.append(f"Linhas com alvo h={h}: {int(feats[f'alvo_h{h}'].notna().sum())}")
    linhas.append(f"Linhas com todos os preditores: {int(feats[PREDITORES].notna().all(axis=1).sum())}")
    return "\n".join(linhas)


def main() -> None:
    base = pd.read_csv(ARQ_BASE, index_col="data", parse_dates=["data"])
    feats = construir_features(base)

    print("Verificações:")
    for item in verificar(base, feats):
        print(f"  OK - {item}")

    feats.to_csv(ARQ_SAIDA)
    print("\n" + resumo(feats))
    print(f"\nVariáveis salvas em {ARQ_SAIDA}")


if __name__ == "__main__":
    main()
