#Imports necessários:

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Definição de uma SEED padrão:

SEED = 1224
np.random.seed(1224)

#Leitura dos CSV utilizando Pandas:

# Dados totais / brutos
dados = pd.read_csv('Data/data.csv')

# Dados separados por genêros musicais
dados_generos = pd.read_csv('Data/data_by_genres.csv')

# Dados separados por ano de lançamento da música
dados_anos = pd.read_csv('Data/data_by_year.csv')

#Tratamentos das variáveis:

# Exclusão de colunas não significativas
dados = dados.drop(['explicit', 'key', 'mode'], axis=1)

# Exclusão de colunas não significativas
dados_generos = dados_generos.drop(['key', 'mode'], axis=1)

# Filtragem de dados para remoção dos gêneros
dados_generos_sem_genres = dados_generos.drop('genres', axis=1)

# Exclusão de dados com anos que não constam nos dados
dados_anos = dados_anos[dados_anos['year']>=2000]
# Exclusão de colunas não significativas
dados_anos = dados_anos.drop(['key', 'mode'], axis=1)

# Reset de index devido ao corte de dados
dados_anos.reset_index()

# Análise da característica Loudness x Anos com PLOTLY EXPRESS:
fig = px.line(dados_anos, x='year', y='loudness', markers=True, title='Variação do Loudness x Anos:')
# fig.show() 

# Análise das Variáveis x Anos com PLOTLY GRAPH OBJECTS:
fig = go.Figure()

#Adição das Variáveis que serão analisadas
fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['acousticness'], name='Acousticness'))

fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['valence'], name='Valence'))

fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['danceability'], name='Danceability'))

fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['energy'], name='Energy'))

fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['instrumentalness'], name='Instrumentalness'))

fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['liveness'], name='Liveness'))

fig.add_trace(go.Scatter(x=dados_anos['year'], y=dados_anos['speechiness'], name='Speechiness'))

fig = px.imshow(dados.corr(), text_auto=True)

# fig.show()


# Pipeline
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2, random_state=SEED))])

# Treino de pipeline
genre_embedding_pca = pca_pipeline.fit_transform(dados_generos_sem_genres)

# Salvar os eixos X - Y
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding_pca)

# Clusterização dos gêneros:
kmeans_pca = KMeans(n_clusters=10, verbose=False, random_state=SEED)
kmeans_pca.fit(projection)

dados_generos['cluster_pca'] = kmeans_pca.predict(projection)
projection['cluster_pca'] = kmeans_pca.predict(projection)

projection['genres'] = dados_generos['genres']

fig = px.scatter(projection, x='x', y='y', color='cluster_pca', hover_data=['x', 'y', 'genres'])
fig.show()
