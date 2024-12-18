import statsmodels.api as sm
import pickle
import os
import pandas as pd
from surveys.ml_logic.data import merge_data
from surveys.ml_logic.preprocessor import transform_genre_values, unificar_nans, categorias_rrss, categorias_voto_gobernador, categorias_voto_pm, categorias_voto_presidente, preprocess_features

def fit_model(i, y, X):
    y_sm = y.map(lambda x: 1 if x == i else 0)
    X_sm = sm.add_constant(X)
    model = sm.Logit(y_sm, X_sm).fit(maxiter=500, method='bfgs')
    return y_sm, model

def export_model(model, number):

    pickles_folder = os.path.join(os.path.dirname(__file__), '..', 'artifacts')

    with open(f"model_{number}.pkl", "wb") as file:
        pickle.dump(model, file)

def load_model(number):

    pickles_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', f"model_{number}.pkl")
    my_model = pickle.load(open(pickles_path,"rb"))
    return my_model

def train_model():
    data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

    csvs = ['noviembre_23.csv', 'abril_24.csv', 'octubre_23.csv', 'mayo_24.csv', 'enero_24.csv', 'julio_23.csv']

    dataframes = [pd.read_csv(os.path.join(data_folder, csv)) for csv in csvs]

    data = merge_data(dataframes)
    data = transform_genre_values(data)
    data = unificar_nans(data)
    data = categorias_rrss(data)
    data = categorias_voto_gobernador(data)
    data = categorias_voto_pm(data)
    data = categorias_voto_presidente(data)

    X = data.drop(columns='Voto_PM')
    y = data['Voto_PM']

    X_proc = preprocess_features(X)

    y_0, model_0 = fit_model(0, y, X_proc)
    y_1, model_1 = fit_model(1, y, X_proc)
    y_2, model_2 = fit_model(2, y, X_proc)
    y_3, model_3 = fit_model(3, y, X_proc)

    return y_0, y_1, y_2, y_3, model_0, model_1, model_2, model_3

def predict(model, tol=0.05):

    mandatory_features = ['const', 'Edad', 'Encuesta', 'Genero_1', 'Escolaridad_3', 'Escolaridad_5', 'Escolaridad_7', 'Escolaridad_9', 'Escolaridad_11']
    condition = (model.pvalues < tol) | (model.params.index.isin(mandatory_features))
    filtered_params = pd.DataFrame({
        'Feature': model.params[condition].index,
        'Value': model.params[condition].values
    })
    return filtered_params
