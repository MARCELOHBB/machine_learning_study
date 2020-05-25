import pandas as pd
from sklearn.cluster import KMeans
from create_dataframe import dataframe

generos_escalados = dataframe()[-1]

def kmeans(numero_de_clusters, generos):
  modelo = KMeans(n_clusters=numero_de_clusters)
  modelo.fit(generos)
  return [numero_de_clusters, modelo.inertia_]

resultado = [kmeans(numero_de_grupos, generos_escalados) for numero_de_grupos in range(1, 41)]

resultado = pd.DataFrame(resultado, columns=['grupos', 'inertia'])

resultado.inertia.plot(xticks=resultado.grupos)