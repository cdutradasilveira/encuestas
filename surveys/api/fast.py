import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from surveys.ml_logic.model import load_model, predict

from surveys.ml_logic.mesa import ejecutar_mesa

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict_candidate(number_candidate:int=3):

    model = load_model(int(number_candidate))
    params_mesa = predict(model, tol=0.05)
    print(params_mesa.shape)
    print(params_mesa)
    print(type(model))
    votos_a, votos_b, porcentaje_a, porcentaje_b, porcentaje_participacion, analisis = ejecutar_mesa(params_mesa)
    return {"votos_candidato_a":votos_a,
            "votos_candidato_b":votos_b,
            "porcentaje_candidato_a":porcentaje_a,
            "porcentaje_candidato_b":porcentaje_b,
            "porcentaje_participacion":porcentaje_participacion,
            "analisis_gpt":analisis}

@app.get("/")
def root():
    return dict(greeting="Hello")
