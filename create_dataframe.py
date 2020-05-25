import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Data:
  def __init__(self):
    self.filmes= None
    self.generos= None
    self.dados_dos_filmes= None
    self.generos_escalados= None

  def dataframe(self, uri_filmes):
    seed = 5
    np.random.seed(seed)

    self.filmes = pd.read_csv(uri_filmes)
    self.filmes.columns = ['filme_id', 'titulo', 'generos']

    # print(self.filmes.head())

    self.generos = self.filmes.generos.str.get_dummies()
    
    # print(generos)

    self.dados_dos_filmes = pd.concat([self.filmes, self.generos], axis=1)

    # print(self.dados_dos_filmes.head())

    scaler = StandardScaler()
    self.generos_escalados = scaler.fit_transform(self.generos)

    return self.filmes, self.generos, self.dados_dos_filmes, self.generos_escalados