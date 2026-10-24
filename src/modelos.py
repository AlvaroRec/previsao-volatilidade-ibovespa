"""Modelos de previsão da volatilidade (Etapa 2, Passo 3).

Primeira versão, com DIVISÃO FIXA:
    - treino: todos os dias de negociação de 2000 a 2006;
    - teste: de 2007 em diante;
    - cada modelo é estimado UMA vez no treino e prevê o teste com os parâmetros
      fixos (sem reestimar). A janela expansiva fica para a Etapa 3.

Unidade de tudo: volatilidade anualizada em %, a mesma do alvo (ver features.py).

Modelos:
    - HAR (Corsi, 2009): MQO do alvo sobre as médias de 1, 5 e 22 dias.
    - GARCH(1,1) (Bollerslev, 1986): estimado nos retornos diários; a previsão
      de h dias é a média das variâncias previstas para t+1..t+h, multiplicada
      por um fator de escala calculado no treino (ver decisão b no plano).
    - Random forest: duas versões, só com os componentes HAR e com HAR +
      variáveis exógenas, para separar o efeito do método do efeito das
      variáveis.
    - Referência ingênua: a volatilidade média dos últimos 22 dias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor

from features import DIAS_UTEIS_ANO

ANO_FIM_TREINO = 2006
HAR = ["har_d", "har_s", "har_m"]
EXOGENAS = ["retorno_log", "log_vix", "dolar_var_log", "selic_meta", "selic_var"]

# Hiperparâmetros do random forest: fixos, escolhidos antes de ver o teste e
# NÃO ajustados (o ajuste com validação cruzada temporal é da Etapa 3).
# São os padrões do scikit-learn, exceto:
#   n_estimators=500   -> mais árvores deixam a média mais estável;
#   min_samples_leaf=10 -> cada folha resume pelo menos 10 dias, para a árvore
#                          não "decorar" dias isolados de uma série ruidosa.
PARAMS_RF = {"n_estimators": 500, "min_samples_leaf": 10, "random_state": 42, "n_jobs": -1}


# ---------------------------------------------------------------------------
# Divisão treino/teste
# ---------------------------------------------------------------------------
def datas_divisao(indice: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Último dia de negociação do treino e primeiro dia do teste (tirados dos dados)."""
    fim_treino = indice[indice.year <= ANO_FIM_TREINO].max()
    inicio_teste = indice[indice.year > ANO_FIM_TREINO].min()
    return fim_treino, inicio_teste


def dividir(feats: pd.DataFrame, h: int, colunas: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino e teste para o horizonte h.

    No treino só entram linhas cujo alvo (dias t+1..t+h) termina até o último
    dia de 2006. Sem esse corte, as últimas linhas do treino usariam dias de
    2007, que pertencem ao teste.
    """
    alvo = f"alvo_h{h}"
    fim_treino, inicio_teste = datas_divisao(feats.index)
    pos_fim = feats.index.get_loc(fim_treino)
    treino = feats.iloc[: pos_fim - h + 1]
    teste = feats.loc[inicio_teste:]
    treino = treino.dropna(subset=colunas + [alvo])
    teste = teste.dropna(subset=colunas + [alvo])
    return treino, teste


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def ajustar_har(treino: pd.DataFrame, h: int):
    """HAR por MQO.

    Os erros-padrão usam a correção de Newey-West (HAC) com 22 defasagens:
    com h = 22 os alvos de dias vizinhos compartilham 21 dias, então os resíduos
    são autocorrelacionados, e os erros-padrão comuns ficariam pequenos demais.
    """
    X = sm.add_constant(treino[HAR])
    return sm.OLS(treino[f"alvo_h{h}"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 22})


def prever_har(modelo, teste: pd.DataFrame) -> pd.Series:
    return modelo.predict(sm.add_constant(teste[HAR], has_constant="add")).rename("HAR")


def ajustar_garch(base: pd.DataFrame):
    """GARCH(1,1) com média constante e erros normais, estimado só no treino.

    Retornos em % (recomendação da biblioteca `arch` para a otimização numérica).
    """
    _, inicio_teste = datas_divisao(base.index)
    r = 100 * base["retorno_log"].dropna()
    modelo = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    # last_obs é exclusivo: a estimação usa os dias ANTES do primeiro dia do teste.
    return modelo.fit(last_obs=inicio_teste, disp="off")


def fator_escala_garch(res_garch, feats: pd.DataFrame) -> float:
    """Variância média de Garman-Klass ÷ variância média do GARCH, ambas no treino."""
    var_garch_treino = (res_garch.conditional_volatility.dropna() ** 2) / 1e4  # de %² para decimal
    var_gk_treino = feats["var_gk"].loc[var_garch_treino.index]
    return float(var_gk_treino.mean() / var_garch_treino.mean())


def prever_garch(res_garch, h: int, inicio_teste: pd.Timestamp, fator: float = 1.0) -> pd.Series:
    """Volatilidade anualizada prevista para os dias t+1..t+h, para cada t do teste.

    A previsão feita no dia t usa os parâmetros do treino e os retornos até t.
    """
    prev = res_garch.forecast(horizon=h, start=inicio_teste, reindex=False)
    var_media = prev.variance.iloc[:, :h].mean(axis=1) / 1e4  # média de t+1..t+h, em decimal
    return (100 * np.sqrt(DIAS_UTEIS_ANO * fator * var_media)).rename("GARCH")


def ajustar_rf(treino: pd.DataFrame, h: int, colunas: list[str]) -> RandomForestRegressor:
    rf = RandomForestRegressor(**PARAMS_RF)
    rf.fit(treino[colunas], treino[f"alvo_h{h}"])
    return rf


def prever_rf(rf: RandomForestRegressor, teste: pd.DataFrame, colunas: list[str], nome: str) -> pd.Series:
    return pd.Series(rf.predict(teste[colunas]), index=teste.index, name=nome)


# ---------------------------------------------------------------------------
# Rodada completa e avaliação
# ---------------------------------------------------------------------------
def rodar(base: pd.DataFrame, feats: pd.DataFrame, h: int) -> dict:
    """Estima todos os modelos no treino e devolve as previsões no teste."""
    todas = HAR + EXOGENAS
    treino, teste = dividir(feats, h, todas)

    har = ajustar_har(treino, h)
    garch = ajustar_garch(base)
    fator = fator_escala_garch(garch, feats)
    rf_har = ajustar_rf(treino, h, HAR)
    rf_exo = ajustar_rf(treino, h, todas)

    _, inicio_teste = datas_divisao(feats.index)
    prev = pd.DataFrame({
        "Ingênuo (média 22 dias)": teste["har_m"],
        "HAR": prever_har(har, teste),
        "GARCH sem ajuste": prever_garch(garch, h, inicio_teste).reindex(teste.index),
        "GARCH com fator de escala": prever_garch(garch, h, inicio_teste, fator).reindex(teste.index),
        "RF (só HAR)": prever_rf(rf_har, teste, HAR, "RF (só HAR)"),
        "RF (HAR + exógenas)": prever_rf(rf_exo, teste, todas, "RF (HAR + exógenas)"),
    })
    return {
        "h": h, "treino": treino, "teste": teste, "previsoes": prev,
        "realizado": teste[f"alvo_h{h}"], "har": har, "garch": garch,
        "fator": fator, "rf_har": rf_har, "rf_exo": rf_exo,
    }


def perdas_robustas(previsoes: pd.DataFrame, realizado: pd.Series, referencia: str = "HAR") -> pd.DataFrame:
    """Métricas de erro robustas a um alvo medido com ruído (Patton, 2011).

    Patton (2011) mostra que, quando o alvo é uma estimativa ruidosa da
    volatilidade verdadeira (como o Garman-Klass), só algumas funções de perda
    preservam a ordem correta entre modelos. Entre elas estão o MSE calculado
    sobre a VARIÂNCIA e a QLIKE. O MSE sobre a volatilidade, usado em `avaliar`,
    não está entre elas. Esta função serve de verificação de robustez.

    QLIKE = média de [ alvo/previsão - ln(alvo/previsão) - 1 ], com variâncias.
    """
    var_prev = previsoes**2          # volatilidade anualizada ao quadrado = variância anualizada
    var_real = realizado**2
    mse_var = (var_prev.sub(var_real, axis=0) ** 2).mean()
    razao = var_prev.rdiv(var_real, axis=0)  # alvo / previsão
    qlike = (razao - np.log(razao) - 1).mean()
    mse_vol = (previsoes.sub(realizado, axis=0) ** 2).mean()
    return pd.DataFrame({
        "MSE na volatilidade / HAR": mse_vol / mse_vol[referencia],
        "MSE na variância / HAR": mse_var / mse_var[referencia],
        "QLIKE / HAR": qlike / qlike[referencia],
        "posição (MSE vol.)": mse_vol.rank().astype(int),
        "posição (MSE var.)": mse_var.rank().astype(int),
        "posição (QLIKE)": qlike.rank().astype(int),
    })


def avaliar(previsoes: pd.DataFrame, realizado: pd.Series, referencia: str = "HAR") -> pd.DataFrame:
    """MSE, RMSE e MSE relativo ao modelo de referência (< 1 = melhor que ele)."""
    erros = previsoes.sub(realizado, axis=0)
    mse = (erros**2).mean()
    return pd.DataFrame({
        "MSE": mse,
        "RMSE (p.p. de vol)": np.sqrt(mse),
        f"MSE / MSE do {referencia}": mse / mse[referencia],
        "Erro médio (viés)": erros.mean(),
    })
