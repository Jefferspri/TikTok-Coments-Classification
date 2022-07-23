# Script data preparation
###################################

import pandas as pd
import os


import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')
stop_words_sp = set(stopwords.words('spanish'))
from nltk.stem import SnowballStemmer
stm = SnowballStemmer('spanish') 

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_excel(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stop_words_sp]
    stemi = [stm.stem(token) for token in text]
    return stemi


# Realizamos la transformación de datos
def data_preparation(df):
    
    # TF-IDF
    tfidf_vect = TfidfVectorizer(analyzer=clean_text)
    X_tfidf = tfidf_vect.fit_transform(df['detail_comment'])
    # To DF
    X_features = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vect.get_feature_names())
    # Sum columns values
    column = pd.DataFrame(X_features.sum())
    # Drop no valid columns
    X_features.drop(columns=[''], inplace=True)
    # Stop words 
    stop = list(column[column[0]<2].index)
    X_features.drop(columns=stop, inplace=True)
    X_features['sent'] = df['sent']
    
    print('Transformación de datos completa')
    return X_features


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, flag, filename):
    
    if flag != 'yes':
        df.drop(columns=['sent'], inplace=True)
        dfp = df
    else:
        dfp = df
    
    dfp.to_excel(os.path.join('../data/processed/', filename), index=False)
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación
def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('tiktok-default.xlsx')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, 'yes','tiktok_train.xlsx')
    # Matriz de Validación
    df2 = read_file_csv('tiktok-new.xlsx')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, 'yes','tiktok_val.xlsx')
    # Matriz de Scoring
    df3 = read_file_csv('tiktok-score.xlsx')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, 'no','tiktok_score.xlsx')
    

if __name__ == "__main__":
    main()
    
    
