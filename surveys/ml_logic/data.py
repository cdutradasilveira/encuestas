import pandas as pd

def merge_data(dataframes) -> pd.DataFrame:

    columnas = set()
    for df in dataframes:
        columnas.update(df.columns)

    data = pd.DataFrame(columns=columnas)

    for df in dataframes:
        print(df.shape)
        data = pd.merge(data, df, on=list(columnas & set(df.columns)), how='outer')

    data = data.drop(columns=['Participa', 'Aborto', 'Coalicion', 'Positivo_OBOR', 'Positivo_PK', 'Positivo_JJF', 'Imagen_EA', 'Imagen_CS', 'Imagen_AMLO', 'Imagen_CD', 'Voto_Gobernador_Partido', 'Voto_PM_2'])

    print("✅ data merged and cleaned")

    return data
