from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# *** TESTANDO O MODELO PEDINDO INPUT DO USUÁRIO *** # 

uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
dados = pd.read_csv(uri)

a_renomear = {
    'mileage_per_year': 'milhas_por_ano',
    'model_year': 'ano_do_modelo',
    'price': 'preco',
    'sold': 'vendido'
}

dados = dados.rename(columns = a_renomear)

a_trocar = {
    'yes': 1,
    'no': 0
}

dados['vendido'] = dados.vendido.map(a_trocar)


ano_atual = datetime.today().year
dados['idade_do_modelo'] = ano_atual - dados.ano_do_modelo
dados['km_por_ano'] = dados.milhas_por_ano * 1.60934

# Dropando as colunas que não serão usadas
dados = dados.drop(columns = ['Unnamed: 0', 'milhas_por_ano', 'ano_do_modelo'])

x = dados[['preco', 'idade_do_modelo', 'km_por_ano']]
y = dados['vendido']

SEED = 5
np.random.seed(SEED)

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y,
                                                        test_size = 0.25,
                                                        stratify = y)


modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(raw_treino_x, treino_y)
previsoes = modelo.predict(raw_teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"A acúracia foi de {acuracia}%")

from joblib import dump, load
dump(modelo, 'modelo.joblib')

modelo_carregado = load('modelo.joblib')

print("Preco: ")
preco = input()
print("Ano: ")
ano = input()
print("KM: ")
km = input()

entrada = [[preco, ano, km]]
resultado = modelo_carregado.predict(entrada)[0]
print("Vai vender ?")
if resultado == 1:
  print('Sim')
else:
  print('Não')