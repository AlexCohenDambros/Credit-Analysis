import pandas as pd
import numpy as np
import os

# scikit learn utilites
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
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
    
    train.drop(['HS_CPF', 'ORIENTACAO_SEXUAL', 'RELIGIAO'], axis = 1, inplace = True)
    test.drop(['HS_CPF', 'ORIENTACAO_SEXUAL', 'RELIGIAO'], axis = 1, inplace = True)
    
    # Separa atributos de entrada em X e as classes em y
    train = train.to_numpy()
    X = train[:,0:66]
    y = train[:,66]
    
    # Normalizando os dados
    scaler = preprocessing.MinMaxScaler().fit(X)
    X = scaler.transform(X)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=random_state)

    
    # train models with AutoML
    automl = AutoML(mode="Perform")
    automl.fit(X_train, y_train)

    # compute the accuracy on test data
    predictions = automl.predict_all(X_test)
    print(predictions.head())
    print("Test accuracy:", accuracy_score(y_test, predictions["label"].astype(int)))
    
  
