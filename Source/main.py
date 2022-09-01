import pandas as pd

import numpy as np

# Dados totais / brutos
dados = pd.read_csv('Data/data.csv')

# Dados separados por genêros musicais
dados_generos = pd.read_csv('Data/data_by_genres.csv')

# Dados separados por ano de lançamento da música
dados_anos = pd.read_csv('Data/data_by_year.csv')

print(dados_generos.head())
Import de dados