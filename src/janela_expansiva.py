"""Previsão com janela expansiva (Etapa 3).

Passo 1 (este arquivo, por enquanto): definições comuns aos modelos da Etapa 3.
O laço da janela expansiva entra no Passo 2.

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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from modelos import HAR

# Decisão 7 do plano da Etapa 3: a Selic em NÍVEL sai, porque no teste ela fica
# fora da faixa do treino em 77% dos dias e uma árvore não extrapola. Fica só a
# variação. Decisão tomada DEPOIS de ver o teste da Etapa 2 (declarada no texto).
EXOGENAS = ["retorno_log", "log_vix", "dolar_var_log", "selic_var"]
CONJUNTOS_RF = {"só HAR": HAR, "HAR + exógenas": HAR + EXOGENAS}
ESCALAS = ("variância", "log")


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


def alvo_variancia(df: pd.DataFrame, h: int) -> pd.Series:
    """Alvo na escala da variância anualizada, em (% a.a.)²."""
    return df[f"alvo_h{h}"] ** 2


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
