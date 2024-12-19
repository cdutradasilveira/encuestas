import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Simulación de Votación", layout="wide")

#Pr

# Título y descripción
st.title("Simulación de Votación")
st.markdown("""
Esta aplicación permite visualizar los resultados de una simulación electoral para diferentes candidatos
utilizando el modelo de Mesa.
""")

# Input del usuario
st.sidebar.header("Configuración")
number_candidate = st.sidebar.selectbox(
    "Selecciona un candidato para generar la simulación:",
    options=["1", "2", "3"],
    format_func=lambda x: f"Candidato {x}"
)

# Botón para procesar
if st.sidebar.button("Generar Resultados"):
    # Solicitud al endpoint
    surveys_api_url = 'https://surveys-v4-314221417170.us-west1.run.app/predict'
    try:
        with st.spinner("Obteniendo datos del servidor..."):
            response = requests.get(surveys_api_url, params={"number_candidate": number_candidate})
            response.raise_for_status()
            dict_result = response.json()

        # Mostrar resultados en la interfaz
        st.success("Datos obtenidos exitosamente")

        # 1) Gráfico de torta
        st.subheader("Distribución de Votos")
        votos = [dict_result["votos_candidato_a"], dict_result["votos_candidato_b"]]
        labels = ["Candidato A", "Candidato B"]

        # Redondear valores absolutos y crear etiquetas personalizadas
        total_votos = sum(votos)
        labels_custom = [
            f"{label}\n{int(round(voto))} votos\n({voto/total_votos:.1%})"
            for label, voto in zip(labels, votos)
        ]

        fig, ax = plt.subplots(figsize=(5, 5))  # Reducir tamaño de la figura
        ax.pie(
            votos, labels=labels_custom, autopct=None, startangle=90, colors=["#1f77b4", "#ff7f0e"]
        )
        ax.axis("equal")
        st.pyplot(fig)

        # 2) Porcentaje de participación
        st.subheader("Participación Electoral")
        st.metric("Porcentaje de Participación", f"{dict_result['porcentaje_participacion']:.2f}%")

        # 3) Análisis GPT
        st.subheader("Análisis GPT del Resultado")
        st.write(dict_result["analisis_gpt"])

        # 4) Tabla de Participación por Género y Distrito
        st.subheader("Participación por Género y Distrito")
        tabla_participacion = pd.DataFrame(dict_result["tabla_part"])
        tabla_percent = tabla_participacion.div(tabla_participacion["Total"], axis=0) * 100
        st.dataframe(tabla_percent.style.format("{:.2f}%"))

        # 5) Histograma de Probabilidad de Voto
        st.subheader("Distribución de Probabilidad de Voto")
        prob_voto = dict_result["probabilidad_voto"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(prob_voto, bins=10, color="#2ca02c", edgecolor="black")
        ax.set_xlabel("Probabilidad de Voto")
        ax.set_ylabel("Cantidad de Agentes")
        st.pyplot(fig)

        # 6) Mapa del municipio de Jalisco
        st.subheader("Mapa del Municipio de Jalisco")
        mapa_data = pd.DataFrame({
            "lat": [20.676722],
            "lon": [-103.347115],
            "info": ["Municipio de Jalisco"]
        })
        st.map(mapa_data)

    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectarse con la API: {e}")

else:
    st.info("Selecciona un candidato y presiona el botón para generar los resultados.")
