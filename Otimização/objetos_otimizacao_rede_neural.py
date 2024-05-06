###############################################################################
#                       Importano Bibliotecas Necessárias                     #
###############################################################################

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


###############################################################################
#                                  DataModule                                 #
###############################################################################

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        tamanho_teste,
        semente_aleatoria,
        pedaco,
        tamanho_lote=256,
        num_trabalhadores=6,
   
         
    ):
        super().__init__()

        self.tamanho_lote = tamanho_lote
        self.num_trabalhadores = num_trabalhadores
        self.tamanho_teste = tamanho_teste
        self.semente_aleatoria = semente_aleatoria
        self.pedaco = pedaco

    def setup(self, stage):
        """Ocorre após o `prepare_data`. Aqui devemos alterar o estado da classe
        para adicionar as informações referentes aos conjuntos de treino, teste
        e validação. O argumento `stage` deve existir e ele indica em qual
        estágio o processo de treino está (pode ser `fit` para
        treinamento/validação e `test` para teste).

        É nesta etapa onde aplicamos transformações aos dados caso necessário.

        """
        features = ['Su', 'E', 'G', 'mu', 'Ro']
        target = ['Sy']
        df = pd.read_pickle("../Conjuntos de dados/Dataset.pickle")

        
        df = df.reindex(features + target, axis=1)
        df = df.dropna()
        
        df_embaralhado = df.sample(frac=1,random_state = 0)

        y_geral = np.array(df_embaralhado[target[0]])
        x_geral = np.array(df_embaralhado.drop(columns=[target[0]]))
        n_divisoes = int(1/self.tamanho_teste)
        ys= np.array_split(y_geral, n_divisoes)
        xs = np.array_split(x_geral, n_divisoes)
        
        y_teste_premium = ys.pop(0)
        x_teste_premium = xs.pop(0)
        self.y_teste_premium = y_teste_premium
        self.x_teste_premium = x_teste_premium
        

        # Daqui pra baixo que 
        
        y_local =  np.concatenate(ys)
        x_local =  np.concatenate(xs)
        ys_local= np.array_split(y_local, n_divisoes)
        xs_local = np.array_split(x_local, n_divisoes)
        
        y_teste = ys_local.pop(self.pedaco)
        x_teste = xs_local.pop(self.pedaco)
        
        y_treino =  np.concatenate(ys_local)
        x_treino =  np.concatenate(xs_local)
        
        X_treino = x_treino#.values
        y_treino = y_treino.reshape(-1,1)

        self.x_scaler = MaxAbsScaler()
        self.x_scaler.fit(X_treino)

        self.y_scaler = MaxAbsScaler()
        self.y_scaler.fit(y_treino)

        if stage == "fit":

            X_treino = self.x_scaler.transform(X_treino)
            y_treino = self.y_scaler.transform(y_treino)

            self.X_treino = torch.tensor(X_treino, dtype=torch.float32)
            self.y_treino = torch.tensor(y_treino, dtype=torch.float32)

        if stage == "test":
            X_teste = x_teste
            y_teste = y_teste.reshape(-1,1)

            X_teste = self.x_scaler.transform(X_teste)
            y_teste = self.y_scaler.transform(y_teste)

            self.X_teste = torch.tensor(X_teste, dtype=torch.float32)
            self.y_teste = torch.tensor(y_teste, dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_treino, self.y_treino),
            batch_size=self.tamanho_lote,
            num_workers=self.num_trabalhadores,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=self.tamanho_lote,
            num_workers=self.num_trabalhadores,
        )

    def test_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_teste, self.y_teste),
            batch_size=self.tamanho_lote,
            num_workers=self.num_trabalhadores,
        )


###############################################################################
#                                     MLP                                     #
###############################################################################
        
class MLP(L.LightningModule):
    def __init__(
        self, num_dados_de_entrada, neuronios_camadas, vieses, num_targets = 1
    ):
        super().__init__()
        self.camadas = nn.Sequential()
        
        for i in range(len(neuronios_camadas) - 1):
            if i == 0:
                self.camadas.add_module(f'linear{i}', nn.Linear(num_dados_de_entrada, neuronios_camadas[i]))
                self.camadas.add_module(f'relu{i}', nn.ReLU())
            else:
                self.camadas.add_module(f'linear{i}', nn.Linear(neuronios_camadas[i - 1], neuronios_camadas[i]))
                self.camadas.add_module(f'relu{i}', nn.ReLU())
        
        # Adicionando a última camada Linear
        self.camadas.add_module(f'linear{len(neuronios_camadas) - 1}', nn.Linear(neuronios_camadas[-2], num_targets, bias=vieses[-1]))
            
            
        self.fun_perda = F.mse_loss

        self.perdas_treino = []
        self.perdas_val = []

        self.curva_aprendizado_treino = []
        self.curva_aprendizado_val = []

    def forward(self, x):
        x = self.camadas(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.fun_perda(y, y_pred)

        self.log("loss", loss, prog_bar=False)
        self.perdas_treino.append(loss)

        return loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.fun_perda(y, y_pred)

        self.log("test_loss", loss)

        return loss

    def on_train_epoch_end(self):
        # Atualiza curva de aprendizado
        perda_media = torch.stack(self.perdas_treino).mean()
        self.curva_aprendizado_treino.append(float(perda_media))
        self.perdas_treino.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer