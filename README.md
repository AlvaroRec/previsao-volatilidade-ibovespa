# Previsão da volatilidade do Ibovespa

Projeto aplicado da disciplina *Introdução à Modelagem Econômica e Ciência de Dados* (UFJF, 2026), Faculdade de Economia.

## Pergunta de pesquisa

Modelos de machine learning preveem a volatilidade do Ibovespa fora da amostra melhor do que os modelos GARCH(1,1) e HAR?

## Hipóteses

- **H1:** o HAR supera o GARCH(1,1).
- **H2:** o ML com variáveis exógenas supera os dois modelos de referência, principalmente no horizonte de 22 dias.
- **H3:** o ganho do ML se concentra em períodos de estresse (2008, 2015–2016 e 2020).

## Modelos

- GARCH(1,1) (Bollerslev, 1986)
- HAR (Corsi, 2009)
- Random forest

Horizontes de previsão: 1 e 22 dias úteis. Janela expansiva, com reestimação a cada 22 dias úteis. Teste fora da amostra a partir de 2007. Erro medido pelo MSE e comparação pelo teste de Diebold-Mariano.

## Dados

Todos públicos e gratuitos, em frequência diária, a partir de janeiro de 2000.

| Variável | Fonte |
|---|---|
| Ibovespa (abertura, máxima, mínima, fechamento) | Yahoo Finance (`^BVSP`, via `yfinance`) |
| Dólar/real (PTAX venda) | Banco Central do Brasil, SGS série 1 |
| Meta Selic | Banco Central do Brasil, SGS série 432 |
| VIX | FRED, série `VIXCLS` (Cboe) |

A volatilidade diária é estimada com o estimador de Garman e Klass (1980), que usa abertura, máxima, mínima e fechamento.

**Os dados não são versionados neste repositório.** Os dados do Yahoo Finance não são redistribuídos; o repositório contém apenas os scripts de coleta.

## Como reproduzir

Requer Python 3.10 ou superior.

```bash
# 1. criar e ativar um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate

# 2. instalar as dependências
pip install -r requirements.txt

# 3. baixar e tratar os dados
python src/coleta_dados.py

# 4. construir as variáveis (volatilidade, alvos e preditores)
python src/features.py

# 5. rodar a análise exploratória (gera as figuras em figuras/)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

O script de coleta salva os arquivos brutos em `data/raw/`, a base final em `data/processed/dados_mercado.csv` e um registro da coleta (data de acesso e contagem de dados faltantes) em `data/coleta_log.txt`. O script de variáveis confere automaticamente a ausência de viés de antecipação e salva `data/processed/features.csv`.

## Estrutura

```
src/coleta_dados.py   coleta e controle de qualidade dos dados
src/features.py       volatilidade de Garman-Klass, alvos, componentes HAR e preditores
notebooks/01_eda.ipynb  análise exploratória dos dados
figuras/              figuras geradas pelos notebooks
data/                 dados baixados (não versionados)
```

## Status

Etapa 1 (proposta e plano de dados): concluída. Etapa 2 (análise exploratória e modelagem): em andamento.

## Referências principais

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Garman, M. B.; Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *The Journal of Business*, 53(1), 67–78.
- Diebold, F. X.; Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
