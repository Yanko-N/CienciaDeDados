# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:43:22 2023

@author: Vitor Novo && David Afonso && João Vaz
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#FUNÇÕES


#Exercicio 4
def tarefa4_Funcao(nome):
    
    # Filtrar apenas os dados do país escolhido
    paisData = asia_df[asia_df["Country"] == nome]
    
    # Encontrar o máximo da coluna "Life_expectancy"
    maximo = paisData["Life_expectancy"].max()
    
    # Recuperar o ano correspondente ao valor máximo
    anoMax= paisData.loc[paisData["Life_expectancy"] == maximo, "Year"].iloc[0]
    print(" O ano que o pais " + nome + " obteve uma maior estimativa de vida foi em :" + str(anoMax))
   
    
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
# Exercicio 1

# Caminho do Ficheiro a ler
filepath = "C:\\Users\Vitor Novo\\Downloads\\Life-Expectancy-Data-Updated.csv"

# df é um data frame
df = pd.read_csv(filepath)

# Criar um novo DataFrame com apenas a informação da Região "Asia"
asia_df = df[df["Region"] == "Asia"]

# Gravar este DataFrame num novo ficheiro .csv
asia_df.to_csv("asiaDataFrame.csv", index=False)

#---------------------------------------------------------------------------------------------------------------
#Exercicio 2
paises2 = ["China", "India", "Japan", "Thailand"]
# Filtrar o DataFrame para conter apenas os dados desses países
tarefa2_df = asia_df[asia_df["Country"].isin(paises2)]

# Criar o gráfico de linhas
for pais in paises2:
    pais_df = tarefa2_df[tarefa2_df["Country"] == pais]
    plt.plot(pais_df["Year"], pais_df["Under_five_deaths"], label=pais)
    
# Personalizar o gráfico
plt.title("Mortes de crianças com menos de cinco anos por 1000 habitantes na Ásia")
plt.xlabel("Ano")
plt.ylabel("Mortes por 1000 habitantes")
plt.legend()
plt.show()

#---------------------------------------------------------------------------------------------------------------
#Exercicio 3


paises3 = ["China", "India", "Japan", "Thailand"]

#criar dataframe com a informação filtrada do paises
aux_df =asia_df[asia_df["Country"].isin(paises3)]

anos=[]

i = 2000
while i < 2015 :
    i += 1
    anos.append(i)
    
    
#crar dataframe com a informação filtrada dos anos
tarefa3_df = aux_df[aux_df["Year"].isin(anos)]

populacoes = []

for pais in paises3:
    pais_df = tarefa3_df[tarefa3_df["Country"] == pais]
    media =  pais_df["Population_mln"].mean()
    populacoes.append(media)


plt.pie(populacoes, labels = paises3)
plt.title("Média da população total em milhões nos anos 2000 a 2015")
plt.show()


#---------------------------------------------------------------------------------------------------------------
#Exercicio 4
tarefa4_Funcao("China")

#---------------------------------------------------------------------------------------------------------------
#Exercicio 5


# Criando o gráfico de dispersão com a regressão linear
sns.lmplot(x='Incidents_HIV', y='Life_expectancy', data = asia_df, height=6, aspect=1.5)

# Definindo o título e os rótulos dos eixos
plt.title("Relação entre HIV e Expectativa de Vida na Ásia")
plt.xlabel("Incidentes de HIV por 1000 habitantes de 15 a 49 anos")
plt.ylabel("Expectativa média de vida")

# Exibindo o gráfico
plt.show()


#---------------------------------------------------------------------------------------------------------------
#Exercicio 6

#---------------------------------------------------------------------------------|
#                               EXPLICAÇÃO                                        |
#---------------------------------------------------------------------------------|
#                                                                                 |                                        |
#   Escolhemos colunas como features para ser selecionar aleatoriamente um        |
#   subconjunto dos dados de treinamento para determinar a melhor característica  |
#   (feature).                                                                    |
#                                                                                 |
#   Usando uma instância do modelo de regressão Random Forest.                    |
#                  --> (RandomForestRegressor) <--                                |
#   Estamos definindo o número de árvores de decisão (n_estimators) como 100      |
#   e definindo um seed Value aleatório.                                          |
#                                                                                 |
#                                                                                 |
#   A função train_test_split() divide os dados aleatoriamente em conjuntos de    |
#   treinamento e teste. Aqui, definimos o tamanho do conjunto de teste como 20%  |
#   dos dados. Definimos um seed Value aleatório.                                 |
#                                                                                 |
#   Treinamos o modelo de regressão Random Forest usando conjunto de treinamento. |
#                                                                                 |
#   Depois colocamos as previsões que nos são calculadas num dataframe e          |
#   fazemos a media para ter um valor medio concreto.                             |
#                                                                                 |   
#   Após isso calculamos o desempenho usando R².                                  | 
#   O coeficiente R² varia de 0 a 1.                                              |
#   Sendo 1 o valor ideal que indica um ajuste perfeito                           |
#                                                                                 |
#---------------------------------------------------------------------------------|



# Selecionar as colunas que serão usadas como características (features)
features = ["Year", "Adult_mortality", "Infant_deaths", "Alcohol_consumption", "Hepatitis_B", "Measles", "BMI", "Under_five_deaths", "Polio", "Diphtheria", "Incidents_HIV", "Schooling"]

# Selecionar a coluna que será usada como rótulo (label)
label = "Life_expectancy"

# Separar as características e o rótulo em conjuntos distintos
X = df[features]
y = df[label]

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciar o modelo de regressão
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Treinar o modelo usando o conjunto de treinamento
rf.fit(X_train, y_train)

# Fazer uma previsão usando o conjunto de teste
previsões = rf.predict(X_test)

# Avaliar o desempenho do modelo usando a métrica R²
r2 = r2_score(y_test, previsões)

previsãoFinal= previsões.mean();
    


print("A nossa previsão é de: ", previsãoFinal, " e temos " , r2 , " de certeza a partir do algoritmo machine learning RandomForest");



    

