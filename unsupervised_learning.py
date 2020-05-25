import pandas as pd
import numpy as np
import seaborn as sns
from create_dataframe import Data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

seed = 5
np.random.seed(seed)

uri_filmes = 'https://raw.githubusercontent.com/oyurimatheus/clusterirng/master/movies/movies.csv'
data = Data()
data.dataframe(uri_filmes)

modelo = KMeans(n_clusters=17)
modelo.fit(data.generos_escalados)

# print(f'Grupos {modelo.labels_}')
# print(generos.columns)
# print(modelo.cluster_centers_)

grupos = pd.DataFrame(modelo.cluster_centers_, columns=data.generos.columns)
# print(grupos)
# print(modelo.cluster_centers_)

grupos.transpose().plot.bar(subplots=True, sharex=False, figsize=(25, 50), rot=0)
# grupo = 0
# filtro = modelo.labels_ == grupo
# data_dos_filmes[filtro].sample(10)

tsne = TSNE()
visualizacao = tsne.fit_transform(data.generos_escalados)
# print(generos_escalados)
# print(visualizacao)

f, axes = plt.subplots()
sns.set(rc={'figure.figsize': (13, 13)})
sns.scatterplot(x=visualizacao[:, 0], y=visualizacao[:, 1], hue=modelo.labels_, palette=sns.color_palette('Set1', 17))