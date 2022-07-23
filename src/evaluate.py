
# Evaluation Code - TikTikTok Sentiment Classification
############################################################################

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.decomposition import TruncatedSVD


# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_excel(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    
    # Predecimos sobre el set de datos de validación 
    X_test = df.drop(['sent'],axis=1)
    y_test = df[['sent']]
    
    # Aplicamos la descomposición SVD para poder trabajar con menos variables (30)
    truncatedSVD=TruncatedSVD(30)
    X_test = truncatedSVD.fit_transform(X_test)
    
    # Predict
    y_pred_test = model.predict(X_test)
    
    # Generamos métricas de diagnóstico
    cm_test = confusion_matrix(y_test,y_pred_test)
    print("Matriz de confusion: ")
    print(cm_test)
    print('Accuracy: %.2f%%' %(accuracy_score(y_test, y_pred_test)*100))  
    print('Precision: %.2f%%' % (precision_score(y_test, y_pred_test, average= 'weighted')*100))
    print('Recall: %.2f%%' % (recall_score(y_test, y_pred_test, average= 'weighted')*100))


# Validación desde el inicio
def main():
    df = eval_model('tiktok_val.xlsx')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()