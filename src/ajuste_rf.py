"""Escolha dos hiperparâmetros do random forest (Etapa 3, Passo 1).

Usa SÓ a primeira janela de treino (linhas cujo alvo termina até o fim de 2006).
Nenhum dado do teste entra aqui. A escolha é feita uma única vez e fica fixa em
todas as reestimações da janela expansiva (decisão 8 do plano da Etapa 3).

Como funciona a validação cruzada temporal (TimeSeriesSplit, 5 dobras):
o treino é dividido em 6 blocos em ordem de data. Na dobra 1, estima com o
bloco 1 e avalia no bloco 2; na dobra 2, estima com os blocos 1-2 e avalia no 3;
e assim por diante. O modelo nunca é avaliado num período anterior ao que usou
para estimar. Entre a parte de estimação e a de avaliação fica um intervalo de
h dias, para que os alvos (médias de t+1 a t+h) das duas partes não se sobreponham.

Critério de cada versão (coerente com a perda em que ela será avaliada):
    - RF em variância: MSE na variância;
    - RF em log: QLIKE, com a correção c calculada só na parte de estimação.

Uso:
    python src/ajuste_rf.py             # grade de hiperparâmetros (~5 min)
    python src/ajuste_rf.py --arvores   # checagem do número de árvores
"""

from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import features as F
import janela_expansiva as J
import modelos as M
from avaliacao import perda_mse, perda_qlike

RAIZ = Path(__file__).resolve().parent.parent
PASTA_RES = RAIZ / "data" / "resultados"

# Grade pequena, definida antes de rodar:
#   min_samples_leaf: quantos dias, no mínimo, cada folha resume. Folhas maiores
#                     suavizam a previsão (menos "decoreba" de dias isolados).
#   max_features:     fração das variáveis sorteadas em cada divisão. 1/3 é o
#                     padrão clássico de Breiman para regressão; 1.0 usa todas.
GRADE = {"min_samples_leaf": [5, 10, 25, 50, 100], "max_features": [1 / 3, 1.0]}
N_DOBRAS = 5
BASE_PARAMS = {"n_estimators": 500, "random_state": 42, "n_jobs": -1}


def treino_primeira_janela(feats: pd.DataFrame, h: int) -> pd.DataFrame:
    """Mesmo treino da divisão fixa da Etapa 2, com as variáveis da Etapa 3."""
    treino, _ = M.dividir(feats, h, J.HAR + J.EXOGENAS)
    return treino


def nota_cv(treino: pd.DataFrame, h: int, cols: list[str], escala: str, params: dict) -> float:
    """Perda média nas dobras de avaliação (quanto menor, melhor)."""
    X, y = treino[cols], J.alvo_variancia(treino, h)
    perda = perda_mse if escala == "variância" else perda_qlike
    notas = []
    for i_est, i_av in TimeSeriesSplit(n_splits=N_DOBRAS, gap=h).split(X):
        rf, c = J.ajustar_rf(X.iloc[i_est], y.iloc[i_est], escala, params)
        prev = J.prever_rf(rf, c, X.iloc[i_av], escala)
        notas.append(float(np.mean(perda(y.iloc[i_av].to_numpy(), prev))))
    return float(np.mean(notas))


def rodar_grade(feats: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for h, (nome_conj, cols), escala in product((1, 22), J.CONJUNTOS_RF.items(), J.ESCALAS):
        treino = treino_primeira_janela(feats, h)
        for folha, mf in product(GRADE["min_samples_leaf"], GRADE["max_features"]):
            params = {**BASE_PARAMS, "min_samples_leaf": folha, "max_features": mf}
            linhas.append({
                "h": h, "variáveis": nome_conj, "escala": escala,
                "min_samples_leaf": folha, "max_features": round(mf, 3),
                "perda CV": nota_cv(treino, h, cols, escala, params),
            })
    tab = pd.DataFrame(linhas)
    grupo = tab.groupby(["h", "variáveis", "escala"])["perda CV"]
    tab["relativa ao melhor"] = tab["perda CV"] / grupo.transform("min")
    return tab


def checar_arvores(feats: pd.DataFrame, n_arvores=(100, 200, 500)) -> pd.DataFrame:
    """Quantas árvores bastam? (também só com o treino de 2000-2006)

    O RF tem um componente aleatório: dois RFs iguais com sementes diferentes
    dão previsões um pouco diferentes. Essa diferença cai com o número de
    árvores. Aqui, para cada versão (com os hiperparâmetros escolhidos), medimos
    a diferença entre duas sementes, relativa ao desvio-padrão do alvo na escala
    do modelo, e o tempo de uma estimação.
    """
    linhas = []
    for h, (nome_conj, cols), escala in product((1, 22), J.CONJUNTOS_RF.items(), J.ESCALAS):
        treino = treino_primeira_janela(feats, h)
        X, y = treino[cols], J.alvo_variancia(treino, h)
        y_mod = y if escala == "variância" else np.log(y)
        escolhidos = J.PARAMS_RF[(h, nome_conj, escala)]
        for n in n_arvores:
            prevs, tempos = [], []
            for semente in (1, 2):
                params = {**escolhidos, "n_estimators": n, "random_state": semente, "n_jobs": -1}
                t0 = time.time()
                rf, c = J.ajustar_rf(X, y, escala, params)
                tempos.append(time.time() - t0)
                p = rf.predict(X)
                prevs.append(p)
            dif = np.sqrt(np.mean((prevs[0] - prevs[1]) ** 2)) / y_mod.std()
            linhas.append({"h": h, "variáveis": nome_conj, "escala": escala, "árvores": n,
                           "diferença entre sementes / dp do alvo": dif, "segundos por estimação": np.mean(tempos)})
    return pd.DataFrame(linhas)


def main() -> None:
    base = pd.read_csv(RAIZ / "data" / "processed" / "dados_mercado.csv", index_col="data", parse_dates=["data"])
    feats = F.construir_features(base)
    if "--arvores" in sys.argv:
        tab = checar_arvores(feats)
        tab.to_csv(PASTA_RES / "ajuste_rf_arvores.csv", index=False)
        print(tab.round(4).to_string(index=False))
        return
    t0 = time.time()
    tab = rodar_grade(feats)
    PASTA_RES.mkdir(parents=True, exist_ok=True)
    tab.to_csv(PASTA_RES / "ajuste_rf.csv", index=False)
    melhores = tab.loc[tab.groupby(["h", "variáveis", "escala"])["perda CV"].idxmin()]
    pd.set_option("display.width", 200)
    print(tab.round(4).to_string(index=False))
    print("\nMelhor combinação de cada versão:")
    print(melhores.round(4).to_string(index=False))
    print(f"\nTempo: {time.time() - t0:.0f} s. Tabela salva em {PASTA_RES / 'ajuste_rf.csv'}")


if __name__ == "__main__":
    main()
