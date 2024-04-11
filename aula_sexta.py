from google.colab import drive
drive.mount('/content/drive', force_remount=True)
import pandas as pd
dados = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/iris.csv', sep=',')
dados.head(5)

dados = dados.drop(columns=["class"])
dados.head(5)

from sklearn.cluster import KMeans #Clusterizador (grupos)
import matplotlib.pyplot as plt # graficos
import math # matematica
from scipy.spatial.distance import cdist # para calcular distancia e distorções
import numpy as np # para procedimentos numericos
distortions = []
K = range (1,101)
#Treinar iterativamente conforme n_clusters = K[1]
for i in K:
  #fit -> treina
  dados_kmeans_model = KMeans(n_clusters=i).fit(dados)
  distortions.append(
      sum(np.min(
          cdist(dados,dados_kmeans_model.cluster_centers_,'euclidean'),axis =1)/dados.shape[0])
  )
print(distortions)

#Exibir grafico

fig, ax = plt.subplots()
ax.plot(K,distortions)
ax.set(xlabel = 'n_clusters', ylabel = 'Distorção', title ='Elbow de distorção')
ax.grid()
fig.savefig("elbow_distorcao.png")
plt.show()

#Calcula o número ótimo de clusters
#Distancia sempre é absoluta, sempre converter para modulo por conta disso
x0 = K[0]
y0 = distortions[0]
xn = K[len(K)-1]
yn = distortions[len(distortions)-1]
#itera nos pontos gerados durante os treinamentos preliminares
distancias = []
for i in range(len(distortions)):
  x = K[i]
  y = distortions[i]
  numerador = abs((yn-y0)*x -(xn-x0)*y + xn*y0 - yn*x0)
  denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
  distancias.append(numerador/denominador)

#Maior distancia
n_cluster_otimo = K[distancias.index(np.max(distancias))]
#Resposta é 8

#Treinar o modelo definitivo com a quantidade de clusters otimos
dados_kmeans_model = KMeans(n_clusters = n_cluster_otimo, random_state = 42).fit(dados)
print(dados_kmeans_model.cluster_centers_)
#Salva modelo definitivo
from pickle import dump
dump(dados_kmeans_model, open('/content/drive/MyDrive/Colab Notebooks/iris_clusters_2024.pkl', 'wb'))

