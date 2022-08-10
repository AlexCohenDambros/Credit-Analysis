import pandas as pd
import numpy as np
import os

pd.set_option('display.max_rows', None)

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
  
