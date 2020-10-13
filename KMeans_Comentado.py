# Importando as bibliotecas
from sklearn.datasets import load_wine # Dataset de vinhos
from sklearn.cluster import KMeans # Algoritmo K-Means
import pandas as pd # Biblioteca do pandas (para tratar o dataset)
import numpy as np # Biblioteca do numpy (para operar com matrizes)
import matplotlib.pyplot as plt # Biblioteca do matplotlib.pyplot (para plotar gráficos)

# Carregando o dataset
wine = load_wine() # A variável "wine" recebe o dataset de vinhos
wine_df = pd.DataFrame(data= np.c_[wine['data'], wine['target']], columns= wine['feature_names'] + ['target']) # Transformando o dataset em um DataFrame do pandas

# Analisando o dataset
"""
wine_df.describe() # A função describe() abre uma tabela com informações de todas as colunas do dataset
wine_df_0 = wine_df.iloc[:59] # Parte do dataset contendo somente vinhos do tipo 0
wine_df_1 = wine_df.iloc[59:130] # Parte do dataset contendo somente vinhos do tipo 1
wine_df_2 = wine_df.iloc[130:] # Parte do dataset contendo somente vinhos do tipo 2
wine_df_0.head(10) # A função head() abre uma tabela de visualização do dataset
wine_df_1.head(10) # O número 10 significa que 10 linhas serão mostradas na tabela
wine_df_2.head(10) # As três linhas de código com o comando "head()" abrem as tabelas de cada tipo de vinho
wine_df.corr() # A função corr() mostra a correlação das "features"(Colunas) com o "target"(Resposta)
"""
# Tratando o dataset
wine_shuffle = wine_df.sample(frac=1).reset_index(drop=True) # A função sample() embaralha o dataset, frac = 1 garante que nenhum dado se perca, e a função reset_index() reorganiza os valores dos indexadores das linhas
wine_shuffle.head(20) # Printando o dataset para conferir se o código acima funcionou

# Dividindo o dataset
Features = ['malic_acid', 'total_phenols', 'flavanoids', 'od280/od315_of_diluted_wines'] # Armazenando as colunas que serão usadas no treino na variável "Features"
x_train, x_test = wine_shuffle[Features][:130], wine_shuffle[Features][130:] # Separando 130 linhas de dados do dataset para o dataset de treino (x_train) e o restante para teste (x_test)
y_train, y_test = wine_shuffle['target'][:130], wine_shuffle['target'][130:] # Fazendo o mesmo que no código acima, porém para a coluna "target", das respostas
x_test.head() # Conferindo o código acima

# Treinando o K-Means
model = KMeans(n_clusters = 3, random_state= 42) # Atribuindo a classe KMeans() ao objeto "model"
model.fit(x_train, y_train) # Treinando o algoritmo

# Testando o algoritmo
prediction = model.predict(x_test) # Prevendo os resultados do dataset de teste com o modelo e salvando-os na variável "prediction"
prediction = pd.Series(prediction, name = 'target') # Convertendo a matriz de resultados para uma série do pandas, para poder calcular a precisão

# Calculando a precisão do modelo
acertos = 0 # Variável para contar a quantidade de respostas corretas
for i in range(len(y_test)):
    if prediction[i] == y_test[i+130]: # Comparando os resultados previstos com os resultados reais
        acertos += 1 # Somando os resultados corretos
accuracy = (acertos/len(y_test)) # Calculando a precisão do modelo
print('Precisão do modelo: {}'.format(accuracy)) # Imprimindo a precisão do modelo

# Plotando os resultados
pontos = [] # Lista vazia para armazenar os pontos 
for i in range(len(y_test)):
    pontos.append(i) # Adicionando os pontos na lista
plt.figure() # Criando a figura
plt.title('Resultados Obtidos vs Resultados Reais (K-Means)') # Título do gráfico
plt.scatter(pontos, prediction, label='Resultados Obtidos') # Plotando os resultados previstos
plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.25)) # Configurando a legenda
plt.scatter(pontos, y_test.reset_index(drop=True), label='Resultados Reais') # Plotando os resultados reais
plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.25)) # Configurando a legenda
plt.xlabel('Amostragem') # Nome da abscissa
plt.ylabel('Respostas') # Nome da ordenada

