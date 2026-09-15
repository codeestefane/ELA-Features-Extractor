# pflacco_ela_features

Projeto para extração de atributos ELA (Exploratory Landscape Analysis) a partir de arquivos CSV de benchmark, usando a biblioteca `pflacco`.

## O que o projeto faz

Este repositório combina dados de desempenho de algoritmos com análise de paisagens de otimização para gerar uma matriz de features em formato tabular, pronta para uso em modelos de Machine Learning e estudos comparativos.

A lógica principal está em [extract_features.py](extract_features.py). Ele:

- percorre os arquivos CSV presentes em [classif.svm](classif.svm)
- amostra subconjuntos de dados com tamanhos variados e sementes diferentes
- calcula métricas ELA para cada instância usando `pflacco`
- concatena as features em um único conjunto de dados para posterior análise

O repositório também inclui um exemplo de saída já gerado em [ela_features_pflacco.csv](ela_features_pflacco.csv), com colunas como `ela_meta.*`, `ela_distr.*`, `cm_angle.*`, `limo.*`, `pca.*` e `disp.*`.

## Por que este projeto é útil

Esse tipo de pipeline é útil quando você deseja:

- caracterizar a estrutura de um problema de otimização/benchmarking
- transformar dados de desempenho em descritores numéricos
- comparar diferentes amostras e configurações de forma reproducível
- alimentar modelos preditivos com informação sobre a dificuldade e a geometria do problema

Além disso, o script automatiza o processo de extração para vários arquivos ao mesmo tempo, evitando a repetição manual de cálculos e garantindo uma base consistente para experimentos.

## Estrutura do projeto

- [extract_features.py](extract_features.py): script principal que gera as features ELA
- [requirements.txt](requirements.txt): dependências do projeto
- [classif.svm](classif.svm): conjunto de bancos de dados em CSV usados como entrada
- [ela_features_pflacco.csv](ela_features_pflacco.csv): exemplo de saída produzida no formato final

## Como começar

### 1) Pré-requisitos

- Python 3
- ambiente virtual recomendado

### 2) Instalação

No diretório do projeto, execute:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3) Execução do pipeline

Para gerar as features, rode:

```bash
python extract_features.py
```

O script lê os arquivos em [classif.svm](classif.svm), processa amostras em diferentes tamanhos e sementes, e salva a tabela final em CSV na raiz do projeto.

### 4) Exemplo de uso

Abaixo está um exemplo do fluxo principal implementado no script:

```python
import pandas as pd

# leitura de um arquivo de benchmark
arquivo = "classif.svm/1037_ada_prior.csv"
dados = pd.read_csv(arquivo)

X = dados[["cost", "gamma"]]
y = dados["ber.test.mean"]

# a extração das features ELA ocorre internamente usando a biblioteca pflacco
# e gera uma linha com métricas dos diversos grupos de ELA
```

Os dados de entrada esperados incluem as colunas principais:

- `cost`
- `gamma`
- `ber.test.mean`

Essas colunas são usadas para calcular as features para cada amostra extraída.

## Observações importantes

- O script é orientado a experimentação em lote: ele percorre vários arquivos e combina resultados em uma única tabela.
- A quantidade de amostras e as sementes são definidas no próprio código, permitindo variações reproduzíveis.
- A parte mais importante do processamento é baseada em funções da biblioteca `pflacco`, como `calculate_ela_meta`, `calculate_ela_distribution`, `calculate_cm_angle`, `calculate_limo`, `calculate_pca` e outras métricas ELA.

## Dependências principais

As dependências mais relevantes deste projeto incluem:

- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `pflacco`
- `tqdm`

A lista completa está em [requirements.txt](requirements.txt).

## Contribuição

Qualquer ajuste no processamento, na extração de features ou na organização do repositório pode ser feito diretamente no script principal e em novos arquivos de apoio. Para manter o projeto consistente, recomenda-se:

1. usar um ambiente virtual
2. instalar as dependências do arquivo [requirements.txt](requirements.txt)
3. testar alterações comparando os resultados gerados com a saída de referência
4. manter os nomes das colunas e o formato CSV compatíveis com o fluxo atual

## Resumo

Este projeto oferece uma base prática para transformar dados de benchmark em descritores ELA usando `pflacco`, com foco em reprodutibilidade, automação e preparação de dados para análise e modelagem preditiva.
