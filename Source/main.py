#Imports necessários:

from http import client
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

from pandas.core.dtypes.cast import maybe_upcast
from sklearn.metrics.pairwise import euclidean_distances

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import  SpotifyClientCredentials

import matplotlib.pyplot as plt
from skimage import io

# Definição de uma SEED padrão:

SEED = 1224
np.random.seed(1224)

# Declaração de variável:

nome_musica = 'Juice WRLD - Bandit ft. NBA Youngboy'

#Leitura dos CSV utilizando Pandas:

# Dados totais / brutos
dados = pd.read_csv('https://raw.githubusercontent.com/sthemonica/music-clustering/main/Dados/Dados_totais.csv')

# Dados separados por genêros musicais
dados_generos = pd.read_csv('https://raw.githubusercontent.com/sthemonica/music-clustering/main/Dados/data_by_genres.csv')

# Dados separados por ano de lançamento da música
dados_anos = pd.read_csv('https://raw.githubusercontent.com/sthemonica/music-clustering/main/Dados/data_by_year.csv')

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

# Pipeline
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2, random_state=SEED))])

# Treino de pipeline GENÊRO
genre_embedding_pca = pca_pipeline.fit_transform(dados_generos_sem_genres)

# Salvar os eixos X - Y
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding_pca)

# Clusterização dos gêneros:
kmeans_pca = KMeans(n_clusters=5, verbose=True, random_state=SEED)
kmeans_pca.fit(projection)

dados_generos['cluster_pca'] = kmeans_pca.predict(projection)
projection['cluster_pca'] = kmeans_pca.predict(projection)

projection['generos'] = dados_generos['genres']

# fig = px.scatter_3d(projection, x=0, y=1, z=2, color='cluster_pca',hover_data=['song'])
# fig.update_traces(marker_size = 2)
# fig.show()

# Dummie dos dados
ohe = OneHotEncoder(dtype=int)
colunas_ohe = ohe.fit_transform(dados[['artists']]).toarray()
dados2 = dados.drop('artists', axis=1)

dados_musicas_dummies = pd.concat([dados2, pd.DataFrame(colunas_ohe, columns=ohe.get_feature_names_out(['artists']))], axis=1)

# dados.shape
# dados_musicas_dummies.shape

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=0.7, random_state=SEED))])

music_embedding_pca = pca_pipeline.fit_transform(dados_musicas_dummies.drop(['id','name','artists_song'], axis=1))
projection_m = pd.DataFrame(data=music_embedding_pca)

kmeans_pca_pipeline = KMeans(n_clusters=50, verbose=False, random_state=SEED)

kmeans_pca_pipeline.fit(projection_m)

dados['cluster_pca'] = kmeans_pca_pipeline.predict(projection_m)
projection_m['cluster_pca'] = kmeans_pca_pipeline.predict(projection_m)
projection_m['artist'] = dados['artists']
projection_m['song'] = dados['artists_song']

fig = px.scatter(projection_m, x=0, y=1, color='cluster_pca', hover_data=[0, 1, 'song'])
# fig.show()

cluster = list(projection_m[projection_m['song']== nome_musica]['cluster_pca'])[0]
musicas_recomendadas = projection_m[projection_m['cluster_pca']== cluster][[0, 1, 'song']]
x_musica = list(projection_m[projection_m['song']== nome_musica][0])[0]
y_musica = list(projection_m[projection_m['song']== nome_musica][1])[0]

#distâncias euclidianas
distancias = euclidean_distances(musicas_recomendadas[[0, 1]], [[x_musica, y_musica]])
musicas_recomendadas['id'] = dados['id']
musicas_recomendadas['distancias']= distancias
recomendada = musicas_recomendadas.sort_values('distancias').head(10)
# print(recomendada)

scope = 'user-library-read playlist-modify-private'
OAuth = SpotifyOAuth(
    scope=scope,
    redirect_uri='http://localhost:5000/callback',
    client_id='64bfebcef50c4786929c82637f1c89ff',
    client_secret='b3ecdf49d72a484b8e8053ffed3d3160'
)

client_credentials_manager = SpotifyClientCredentials(client_id = '64bfebcef50c4786929c82637f1c89ff', client_secret = 'b3ecdf49d72a484b8e8053ffed3d3160')
sp = spotipy.Spotify(client_credentials_manager= client_credentials_manager)


