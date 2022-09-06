import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# scikit learn utilites
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# mljar-supervised package
from supervised.automl import AutoML

import warnings
from warnings import simplefilter
warnings.filterwarnings("ignore", category=RuntimeWarning)
simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', None)
random_state = 42
np.random.seed(42)
error = False


# ================ Funções ================

def autoML(X_train, X_test, y_train, y_test):
    # train models with AutoML
    automl = AutoML(mode="Perform")
    automl.fit(X_train, y_train)

    # compute the accuracy on test data
    predictions = automl.predict_all(X_test)
    print(predictions.head())
    print("Test accuracy:", accuracy_score(
        y_test, predictions["label"].astype(int)))

if not os.path.isfile('train.csv'):
    error = True
    print("Arquivo de Treino não existe!!")

if not os.path.isfile('test.csv'):
    error = True
    print("Arquivo de Test não existe!!")

if not os.path.isfile('infoColumns.csv'):
    error = True
    print("Arquivo de info das colunas não existe!!")

if not error:
    train = pd.read_csv("train.csv")
    test = pd.read_csv('test.csv')
    infoColumns = pd.read_csv('infoColumns.csv')

    train.drop(['HS_CPF', 'ORIENTACAO_SEXUAL',
               'RELIGIAO'], axis=1, inplace=True)
    test.drop(['HS_CPF', 'ORIENTACAO_SEXUAL',
              'RELIGIAO'], axis=1, inplace=True)

    # drop colunas onde a correlação com o TARGET é menor que 1
    for index, row in infoColumns.iterrows():
        try:
            if row['correlation'] < 1:
                    train.drop([row['names']], axis=1, inplace=True)
        except: 
            continue
        
    # Removendo colunas com mais de 50% de valores Nulos
        for index, row in infoColumns.iterrows():
            try:
                if row['%null'] > 0.50:
                        train.drop([row['names']], axis=1, inplace=True)
            except: 
                continue
            
    # Transforma os valores -inf para NaN
    train[train < 0] = np.nan
    test[test < 0] = np.nan

    # Separa atributos de entrada em X e as classes em y
    train = train.to_numpy()
    X = train[:, 0: (train.shape[1]-1)]
    y = train[:, (train.shape[1]-1)]

    # Normalizando os dados
    scaler = preprocessing.MinMaxScaler().fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=random_state, stratify=y)

    autoML(X_train, X_test, y_train, y_test)