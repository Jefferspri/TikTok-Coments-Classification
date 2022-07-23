# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 15:37:49 2022

@author: Pc
"""

# Train Code - TikTok Sentiment Classification
#################################################

from sklearn.decomposition import TruncatedSVD 
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import pandas as pd



# Cargar la tabla transformada
def read_file_excel(filename):
    df = pd.read_excel(os.path.join('../data/processed', filename))
    X_train = df.drop(['sent'],axis=1)
    y_train = df[['sent']]
    print(filename, ' cargado correctamente')
    
    # Aplicamos la descomposición SVD para poder trabajar con menos variables (30)
    truncatedSVD=TruncatedSVD(30)
    X_train = truncatedSVD.fit_transform(X_train)
    
    # Entrenamos el modelo con toda la muestra
    rf = RandomForestClassifier()
    rf_model = rf.fit(X_train, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(rf_model, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')



# Entrenamiento completo
def main():
    read_file_excel('tiktok_train.xlsx')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()