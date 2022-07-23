# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 16:01:43 2022

@author: Pc
"""

# Código de Scoring - TikTok Sentiment Classification
#####################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import pickle
import os


# Cargar la tabla transformada
def score_model(filename, scores):
    df = pd.read_excel(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')
    
    # Aplicamos la descomposición SVD para poder trabajar con menos variables (30)
    truncatedSVD=TruncatedSVD(30)
    X_score = truncatedSVD.fit_transform(df)
    
    
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    
    # Predecimos sobre el set de datos de Scoring    
    res = model.predict(X_score).reshape(-1,1)
    pred = pd.DataFrame(res, columns=['PREDICT'])
    pred.to_excel(os.path.join('../data/scores/', scores))
    print(scores, 'exportado correctamente en la carpeta scores')


# Scoring desde el inicio
def main():
    df = score_model('tiktok_score.xlsx','final_score.xlsx')
    print('Finalizó el Scoring del Modelo')


if __name__ == "__main__":
    main()