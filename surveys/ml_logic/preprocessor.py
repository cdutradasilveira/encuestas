import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def transform_genre_values(X: pd.DataFrame) -> pd.DataFrame:

    X.loc[X['Genero'] == 2, 'Genero'] = 0

    return X

def unificar_nans(X: pd.DataFrame) -> pd.DataFrame:

    X.replace(-1, np.nan, inplace=True)

    return X

def categorias_voto_presidente(X: pd.DataFrame) -> pd.DataFrame:

    X.loc[X['Voto_Presidente'] > 3, 'Voto_Presidente'] = 0

    return X

def categorias_voto_pm(X: pd.DataFrame) -> pd.DataFrame:

    mask = X['Encuesta'] == 2

    X.loc[mask, 'Voto_PM'] = 0

    X.loc[mask & (X['Voto_PM_Partido'].between(1, 3)), 'Voto_PM'] = 1
    X.loc[mask & (X['Voto_PM_Partido'] == 6), 'Voto_PM'] = 2
    X.loc[mask & ((X['Voto_PM_Partido'].between(4, 5)) | (X['Voto_PM_Partido'].between(7, 9))), 'Voto_PM'] = 3

    X = X.drop(columns=['Voto_PM_Partido'])

    X.loc[X['Voto_PM'] > 3, 'Voto_PM'] = 0

    return X

def categorias_voto_gobernador(X: pd.DataFrame) -> pd.DataFrame:

    X.loc[X['Voto_Gobernador'] > 3, 'Voto_Gobernador'] = 0

    return X

def categorias_rrss(X: pd.DataFrame) -> pd.DataFrame:

    mask = X['Encuesta'] == 6

    X.loc[mask & (X['RRSS'] == 9), 'RRSS'] = 8

    return X

def preprocess_features(X: pd.DataFrame):

    columns_ohe = ['Exp_Ganador_PM', 'Distrito', 'Problema_2', 'Eval_AMLO', 'Eval_EA', 'Eval_JJF', 'Nunca', 'Voto_Presidente', 'Voto_Gobernador', 'Conocimiento_OBOR', 'Conocimiento_JJF', 'Conocimiento_PK', 'Simpatia_MC', 'Simpatia_Mor', 'Simpatia_FUT', 'Genero', 'Escolaridad', 'RRSS', 'Programa_Social']
    columns_no_ohe = X.drop(columns=columns_ohe).columns

    knn_imputer = KNNImputer(n_neighbors=5)

    X_imputed = pd.DataFrame(knn_imputer.fit_transform(X), columns=X.columns)

    for col in columns_ohe:
        X_imputed[col] = X_imputed[col].round().astype(int)

    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

    X_categorical = X_imputed[columns_ohe]
    X_encoded = pd.DataFrame(ohe.fit_transform(X_categorical), columns=ohe.get_feature_names_out(columns_ohe))

    X_proc = pd.concat([X_imputed.drop(columns=columns_ohe), X_encoded], axis=1)

    scaler = MinMaxScaler()

    X_proc['Edad'] = scaler.fit_transform(X_proc[['Edad']].values)
    X_proc['Encuesta'] = scaler.fit_transform(X_proc[['Encuesta']].values)

    return X_proc
