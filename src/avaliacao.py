"""Funções de perda e testes de comparação (Etapa 3).

Todas as previsões e alvos aqui estão na escala da VARIÂNCIA anualizada, em
(% a.a.)², isto é, a volatilidade anualizada em % elevada ao quadrado.

Perdas principais (Patton, 2011): ambas ordenam os modelos corretamente mesmo
quando o alvo é uma estimativa ruidosa da volatilidade verdadeira.
    - MSE na variância: (previsão - alvo)².
    - QLIKE: alvo/previsão - ln(alvo/previsão) - 1. Mede o erro PROPORCIONAL
      e pune mais a previsão baixa demais do que a alta demais.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def perda_mse(real, prev):
    """Erro quadrático na variância, dia a dia."""
    return (prev - real) ** 2


def perda_qlike(real, prev):
    """QLIKE dia a dia. Vale zero quando a previsão acerta em cheio."""
    razao = real / prev
    return razao - np.log(razao) - 1


PERDAS = {"MSE na variância": perda_mse, "QLIKE": perda_qlike}
