"""Previsão com janela expansiva (Etapa 3, Passos 1 e 2).

Como funciona a janela expansiva:
    - o teste começa em 02/01/2007 e é dividido em blocos de 22 dias úteis;
    - no início de cada bloco, TODOS os modelos são reestimados com todos os
      dados já conhecidos (de 2000 até a véspera do bloco), inclusive o fator de
      escala do GARCH e a correção c das versões em log;
    - com esses parâmetros, o modelo prevê cada dia do bloco usando os
      preditores do próprio dia (conhecidos no fechamento de t).
    A primeira janela (2000-2006) é o treino da divisão fixa da Etapa 2, então o
    primeiro bloco reproduz as previsões da Etapa 2 (conferido em `conferir`).

Escalas de estimação (decisão 2 do plano da Etapa 3):
    - "variância": o modelo é estimado com o alvo em variância e avaliado pelo
      MSE na variância. O critério de estimação (erro quadrático) é a própria perda.
    - "log": o modelo é estimado com o alvo em log da variância e avaliado pela
      QLIKE. A QLIKE mede erro proporcional, e a regressão em log também.
      Ao voltar do log, exp(previsão) fica sistematicamente baixa (estima a
      mediana, não a média). A correção multiplica por
          c = média( alvo / exp(previsão) )  no treino,
      que é exatamente a constante que minimiza a QLIKE no treino:
      d/dc média[ y/(c·f) + ln(c·f) ] = 0  =>  c = média(y/f).
    - GARCH: estimado por máxima verossimilhança normal, que equivale a
      minimizar uma QLIKE dos retornos ao quadrado. Uma só versão.

Todas as previsões saem na escala da VARIÂNCIA anualizada, em (% a.a.)²
(a volatilidade anualizada em % ao quadrado), a mesma do alvo.

Uso:
    python src/janela_expansiva.py              # rodada principal (~30 min)
    python src/janela_expansiva.py --robustez   # sem os 3 dias suspeitos + divisão fixa
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor

import features as F
import modelos as M
from features import DIAS_UTEIS_ANO
from modelos import HAR

RAIZ = Path(__file__).resolve().parent.parent
PASTA_RES = RAIZ / "data" / "resultados"

PASSO = 22  # dias úteis entre reestimações
HORIZONTES = (1, 22)

# Decisão 7 do plano da Etapa 3: a Selic em NÍVEL sai, porque no teste ela fica
# fora da faixa do treino em 77% dos dias e uma árvore não extrapola. Fica só a
# variação. Decisão tomada DEPOIS de ver o teste da Etapa 2 (declarada no texto).
EXOGENAS = ["retorno_log", "log_vix", "dolar_var_log", "selic_var"]
CONJUNTOS_RF = {"só HAR": HAR, "HAR + exógenas": HAR + EXOGENAS}
ESCALAS = ("variância", "log")
PREDITORES = HAR + EXOGENAS

# Hiperparâmetros do RF escolhidos por validação cruzada temporal SÓ no treino
# de 2000-2006 (src/ajuste_rf.py; tabela em data/resultados/ajuste_rf.csv).
# Chave: (horizonte, conjunto de variáveis, escala). Fixos em todas as janelas.
# Em todas as versões venceu max_features = 1/3; o tamanho mínimo da folha
# ficou em 25 ou 50 dias. Na Etapa 2 eram 10 dias e todas as variáveis.
PARAMS_RF = {
    (1, "só HAR", "variância"): {"min_samples_leaf": 50, "max_features": 1 / 3},
    (1, "só HAR", "log"): {"min_samples_leaf": 50, "max_features": 1 / 3},
    (1, "HAR + exógenas", "variância"): {"min_samples_leaf": 25, "max_features": 1 / 3},
    (1, "HAR + exógenas", "log"): {"min_samples_leaf": 25, "max_features": 1 / 3},
    (22, "só HAR", "variância"): {"min_samples_leaf": 50, "max_features": 1 / 3},
    (22, "só HAR", "log"): {"min_samples_leaf": 50, "max_features": 1 / 3},
    (22, "HAR + exógenas", "variância"): {"min_samples_leaf": 50, "max_features": 1 / 3},
    (22, "HAR + exógenas", "log"): {"min_samples_leaf": 50, "max_features": 1 / 3},
}
FIXOS_RF = {"n_estimators": 500, "random_state": 42, "n_jobs": -1}

# Os 3 dias com pico suspeito (Etapa 2), excluídos no teste de robustez.
DIAS_SUSPEITOS = ["2000-11-24", "2000-12-18", "2001-03-20"]


def alvo_variancia(df: pd.DataFrame, h: int) -> pd.Series:
    """Alvo na escala da variância anualizada, em (% a.a.)²."""
    return df[f"alvo_h{h}"] ** 2


def nome_rf(conjunto: str, escala: str) -> str:
    return f"RF {conjunto} ({escala})"


# ---------------------------------------------------------------------------
# HAR nas duas escalas
# ---------------------------------------------------------------------------
def _x_har(df: pd.DataFrame, escala: str) -> pd.DataFrame:
    """Componentes do HAR na escala do modelo: variância ou log da variância."""
    v = df[HAR] ** 2
    return sm.add_constant(v if escala == "variância" else np.log(v), has_constant="add")


def ajustar_har(treino: pd.DataFrame, h: int, escala: str):
    """HAR por MQO. Devolve (modelo, c); c = 1 na escala da variância."""
    y = alvo_variancia(treino, h)
    X = _x_har(treino, escala)
    if escala == "variância":
        return sm.OLS(y, X).fit(), 1.0
    mod = sm.OLS(np.log(y), X).fit()
    c = float(np.mean(y / np.exp(mod.fittedvalues)))
    return mod, c


def prever_har(mod, c: float, df: pd.DataFrame, escala: str) -> pd.Series:
    p = mod.predict(_x_har(df, escala))
    return p if escala == "variância" else c * np.exp(p)


# ---------------------------------------------------------------------------
# Random forest nas duas escalas
# ---------------------------------------------------------------------------
def ajustar_rf(X: pd.DataFrame, y_var: pd.Series, escala: str, params: dict):
    """Estima o RF e devolve (modelo, c).

    Na escala "variância", c = 1. Na escala "log", c é calculado com as
    previsões OOB ("out of bag"): cada dia é previsto só pelas árvores que não
    o usaram no sorteio. Com as previsões comuns do treino, o RF quase decora
    os dados e c sairia perto de 1, sem corrigir nada.
    """
    if escala == "variância":
        rf = RandomForestRegressor(**params)
        rf.fit(X, y_var)
        return rf, 1.0
    if escala == "log":
        rf = RandomForestRegressor(**params, oob_score=True)
        rf.fit(X, np.log(y_var))
        c = float(np.mean(y_var.to_numpy() / np.exp(rf.oob_prediction_)))
        return rf, c
    raise ValueError(f"escala desconhecida: {escala}")


def prever_rf(rf, c: float, X: pd.DataFrame, escala: str) -> np.ndarray:
    """Previsão do RF na escala da variância."""
    p = rf.predict(X)
    return p if escala == "variância" else c * np.exp(p)


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------
def garch_bloco(retornos_pct: pd.Series, feats: pd.DataFrame, data_ini: pd.Timestamp, datas_bloco):
    """Estima o GARCH com os retornos ANTES de data_ini e prevê os dias do bloco.

    Devolve (previsões por horizonte na escala da variância anualizada, resultado, fator).
    A previsão feita no dia t usa os parâmetros estimados e os retornos até t.
    """
    am = arch_model(retornos_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(last_obs=data_ini, disp="off")  # last_obs é exclusivo
    fator = M.fator_escala_garch(res, feats)
    prev = res.forecast(horizon=max(HORIZONTES), start=data_ini, reindex=False).variance
    prev = prev.loc[prev.index.isin(datas_bloco)]
    # variância diária em %² -> variância anualizada em (% a.a.)²
    saida = {h: DIAS_UTEIS_ANO * fator * prev.iloc[:, :h].mean(axis=1) for h in HORIZONTES}
    return saida, res, fator


# ---------------------------------------------------------------------------
# Janela expansiva
# ---------------------------------------------------------------------------
def inicios_dos_blocos(indice: pd.DatetimeIndex, passo: int | None) -> list[int]:
    """Posições (no índice completo) do primeiro dia de cada bloco do teste.

    passo=None dá um único bloco: a divisão fixa da Etapa 2, sem reestimação.
    """
    _, inicio_teste = M.datas_divisao(indice)
    p0 = indice.get_loc(inicio_teste)
    if passo is None:
        return [p0]
    return list(range(p0, len(indice), passo))


def rodar(base: pd.DataFrame, feats: pd.DataFrame, passo: int | None = PASSO, rotulo: str = "") -> dict:
    """Roda a janela expansiva para os dois horizontes.

    Devolve, para cada h, um DataFrame de previsões (uma linha por dia de teste)
    e um DataFrame de parâmetros (uma linha por bloco).
    """
    indice = feats.index
    inicios = inicios_dos_blocos(indice, passo)
    fins = inicios[1:] + [len(indice)]
    retornos_pct = 100 * base["retorno_log"].dropna()

    prevs = {h: [] for h in HORIZONTES}
    params = {h: [] for h in HORIZONTES}
    t0 = time.time()
    for b, (p_ini, p_fim) in enumerate(zip(inicios, fins)):
        data_ini = indice[p_ini]
        datas_bloco = indice[p_ini:p_fim]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # avisos de convergência ficam registrados em "garch_convergiu"
            garch, res_g, fator = garch_bloco(retornos_pct, feats, data_ini, datas_bloco)

        for h in HORIZONTES:
            alvo = f"alvo_h{h}"
            # Treino: só linhas cujo alvo (t+1..t+h) termina ANTES do primeiro
            # dia do bloco (mesma regra da divisão fixa da Etapa 2).
            treino = feats.iloc[: p_ini - h].dropna(subset=PREDITORES + [alvo])
            bloco = feats.iloc[p_ini:p_fim].dropna(subset=PREDITORES + [alvo])
            if bloco.empty:
                continue
            ultima = indice.get_loc(treino.index[-1])
            assert ultima + h < p_ini, "vazamento: alvo do treino entra no bloco"

            y_tr = alvo_variancia(treino, h)
            out = pd.DataFrame(index=bloco.index)
            out["realizado"] = alvo_variancia(bloco, h)
            out["bloco"] = b
            out["Ingênuo"] = bloco["har_m"] ** 2
            reg = {"bloco": b, "inicio": data_ini, "n_treino": len(treino),
                   "garch_omega": res_g.params["omega"], "garch_alpha": res_g.params["alpha[1]"],
                   "garch_beta": res_g.params["beta[1]"], "garch_fator": fator,
                   "garch_convergiu": res_g.convergence_flag == 0}
            for escala in ESCALAS:
                mod, c = ajustar_har(treino, h, escala)
                out[f"HAR ({escala})"] = prever_har(mod, c, bloco, escala)
                reg[f"HAR ({escala}) c"] = c
                for k, v in mod.params.items():
                    reg[f"HAR ({escala}) {k}"] = v
            out["GARCH"] = garch[h].reindex(bloco.index)
            for conj, cols in CONJUNTOS_RF.items():
                for escala in ESCALAS:
                    rf, c = ajustar_rf(treino[cols], y_tr, escala, {**PARAMS_RF[(h, conj, escala)], **FIXOS_RF})
                    out[nome_rf(conj, escala)] = prever_rf(rf, c, bloco[cols], escala)
                    reg[f"{nome_rf(conj, escala)} c"] = c
            prevs[h].append(out)
            params[h].append(reg)
        if b % 20 == 0 or b == len(inicios) - 1:
            print(f"{rotulo} bloco {b + 1}/{len(inicios)} ({data_ini.date()}) - {time.time() - t0:.0f} s", flush=True)

    return {h: {"previsoes": pd.concat(prevs[h]), "parametros": pd.DataFrame(params[h]).set_index("bloco")}
            for h in HORIZONTES}


# ---------------------------------------------------------------------------
# Conferências
# ---------------------------------------------------------------------------
def conferir(base: pd.DataFrame, feats: pd.DataFrame, R: dict) -> list[str]:
    """Checagens automáticas da rodada principal."""
    ok = []
    for h in HORIZONTES:
        p = R[h]["previsoes"]
        # 1) Todas as previsões existem e são positivas (exceto, possivelmente, o
        #    HAR em variância, que é uma reta e pode dar valor negativo).
        modelos = [c for c in p.columns if c not in ("realizado", "bloco")]
        assert p[modelos].notna().all().all(), f"previsão faltando (h={h})"
        positivas = [m for m in modelos if m != "HAR (variância)"]
        assert (p[positivas] > 0).all().all(), f"previsão não positiva (h={h})"
        ok.append(f"h={h}: {len(p)} dias de teste, {p['bloco'].nunique()} blocos, todas as previsões existem; "
                  f"HAR (variância) negativo em {int((p['HAR (variância)'] <= 0).sum())} dias")

    # 2) O primeiro bloco reproduz a divisão fixa da Etapa 2 (HAR em variância e GARCH).
    for h in HORIZONTES:
        treino, teste = M.dividir(feats, h, M.HAR + M.EXOGENAS)
        har_e2 = sm.OLS(treino[f"alvo_h{h}"] ** 2, sm.add_constant(treino[HAR] ** 2)).fit()
        p = R[h]["previsoes"]
        b0 = p[p["bloco"] == 0]
        esperado_har = har_e2.predict(sm.add_constant(teste.loc[b0.index, HAR] ** 2, has_constant="add"))
        g = M.ajustar_garch(base)
        _, inicio_teste = M.datas_divisao(feats.index)
        esperado_garch = M.prever_garch(g, h, inicio_teste, M.fator_escala_garch(g, feats)) ** 2
        assert np.allclose(b0["HAR (variância)"], esperado_har), f"primeiro bloco do HAR difere da Etapa 2 (h={h})"
        assert np.allclose(b0["GARCH"], esperado_garch.loc[b0.index]), f"primeiro bloco do GARCH difere da Etapa 2 (h={h})"
    ok.append("primeiro bloco = previsões da divisão fixa da Etapa 2 (HAR em variância e GARCH com fator)")

    # 3) O GARCH de cada bloco foi estimado só com dias anteriores ao bloco.
    r = 100 * base["retorno_log"].dropna()
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    for data_ini in R[1]["parametros"]["inicio"].iloc[[0, 50, -1]]:
        res = am.fit(last_obs=data_ini, disp="off")
        assert res.conditional_volatility.dropna().index.max() < data_ini
    ok.append("GARCH estimado só com dias anteriores ao bloco (conferido em 3 blocos)")
    return ok


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def salvar(R: dict, sufixo: str = "") -> None:
    PASTA_RES.mkdir(parents=True, exist_ok=True)
    for h in HORIZONTES:
        R[h]["previsoes"].to_csv(PASTA_RES / f"previsoes_h{h}{sufixo}.csv", index_label="data")
        R[h]["parametros"].to_csv(PASTA_RES / f"parametros_h{h}{sufixo}.csv")


def carregar(sufixo: str = "") -> dict:
    """Lê as previsões salvas (usado pelo notebook 03)."""
    return {h: pd.read_csv(PASTA_RES / f"previsoes_h{h}{sufixo}.csv", index_col="data", parse_dates=["data"])
            for h in HORIZONTES}


def main() -> None:
    base = pd.read_csv(RAIZ / "data" / "processed" / "dados_mercado.csv", index_col="data", parse_dates=["data"])
    t0 = time.time()
    if "--robustez" in sys.argv:
        feats_sem = F.construir_features(base, dias_excluidos=DIAS_SUSPEITOS)
        salvar(rodar(base, feats_sem, rotulo="[sem 3 dias]"), "_sem3dias")
        salvar(rodar(base, F.construir_features(base), passo=None, rotulo="[divisão fixa]"), "_divisao_fixa")
    else:
        feats = F.construir_features(base)
        R = rodar(base, feats, rotulo="[principal]")
        print("\nConferências:")
        for item in conferir(base, feats, R):
            print("  OK -", item)
        salvar(R)
    print(f"\nTempo total: {(time.time() - t0) / 60:.1f} min. Arquivos em {PASTA_RES}")


if __name__ == "__main__":
    main()
