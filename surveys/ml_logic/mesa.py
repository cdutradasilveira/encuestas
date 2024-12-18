import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.cluster import KMeans

import networkx as nx
import re
import os

from dotenv import load_dotenv

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

import openai


class VoterAgent(Agent):
    def __init__(self, unique_id, model, Edad, Genero_1, Distrito, Factor_Expansion, coef_df, **kwargs):
        super().__init__(model)
        self.unique_id = unique_id
        self.model = model
        self.Genero_1 = Genero_1
        self.Edad = Edad
        self.Distrito = Distrito
        self.Factor_Expansion = Factor_Expansion

        for key, value in kwargs.items():
            setattr(self, key, value)

# Coeficientes del modelo de regresión logística
        for _, row in coef_df.iterrows():
            setattr(self, f"coef_{row['Feature']}", row['Value'])

        self.const = getattr(self, 'const', 0)

        self.voto = self.asignar_voto_inicial()

        self.participa = False

        self.voto_emitido = None


    def calcular_probabilidad_voto(self):
    # Inicializar z con el intercepto (const)
        z = getattr(self, 'const', 0)

    # Iterar sobre los coeficientes que empiezan con 'coef_'
        for attr, value in self.__dict__.items():
            if attr.startswith('coef_'):
                feature_name = attr[len('coef_'):]  # Extraer el nombre de la característica
                feature_value = getattr(self, feature_name, 0)  # Obtener el valor de la característica
                z += value * feature_value  # Sumar el producto del coeficiente por el valor de la característica

    # Calcular la probabilidad usando la función sigmoide
        prob_voto_A = 1 / (1 + np.exp(-z))
        return prob_voto_A

    def asignar_voto_inicial(self):
        return "Candidato A" if self.calcular_probabilidad_voto() >= 0.5 else "Candidato B"

#Influencia de vecinos
    def step(self):
        # Revisar los vecinos
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)

        if neighbors:
            # Crear una lista de tuplas (edad, voto) de los vecinos
            vecinos_info = [(neighbor.Edad, neighbor.voto) for neighbor in neighbors]

            # Calcular el voto ponderado por edad
            votos_ponderados = {"Candidato A": 0, "Candidato B": 0}
            for Edad, voto in vecinos_info:
                votos_ponderados[voto] += Edad  # Sumar la edad como peso al voto

            # Determinar el voto de la mayoría ponderado por la edad
            voto_mayoria = max(votos_ponderados, key=votos_ponderados.get)

                # Ajustar el intercepto en función del voto mayoritario ponderado por edad
            if voto_mayoria != self.voto:
                # La influencia disminuye con la edad del agente
                factor_influencia = max(0.1, 1 - (self.Edad / 100))  # Entre 0.1 y 1.0
                influencia = sum([Edad for Edad, voto in vecinos_info if voto == voto_mayoria]) / len(vecinos_info)
                ajuste = 1 * (influencia / 100) * factor_influencia  # Ajuste proporcional a la edad (normalizado)

                # Modificar la probabilidad en función del ajuste calculado
                self.coef_const += ajuste if voto_mayoria == "Candidato A" else -ajuste

        # Recalcular el voto después de la influencia de los vecinos
        self.voto = "Candidato A" if self.calcular_probabilidad_voto() >= 0.5 else "Candidato B"

#Función de participación
def participar_en_votacion(Genero_1, agente):

    factor_genero = 0.6 if Genero_1 == 0 else 0.8

    factor_escolaridad = 0.5

    if agente.Escolaridad_11:
        factor_escolaridad = 0.9
    elif agente.Escolaridad_9:
        factor_escolaridad = 0.85
    elif agente.Escolaridad_7:
        factor_escolaridad = 0.75
    elif agente.Escolaridad_5:
        factor_escolaridad = 0.65
    elif agente.Escolaridad_3:
        factor_escolaridad = 0.55

    # Calcular la probabilidad final combinando los factores de género y escolaridad
    probabilidad_participacion = factor_genero * factor_escolaridad

    # Decidir participación individual
    return random.random() <= probabilidad_participacion


def normalizar_participacion(agentes, objetivo=0.7, tolerancia=0.05):

    # Determinar quiénes participan inicialmente
    participantes = [agente for agente in agentes if agente.participa]
    total_participantes = len(participantes)
    total_agentes = len(agentes)

    # Calcular el porcentaje actual de participación
    participacion_actual = total_participantes / total_agentes

    # Ajustar la participación si está fuera del rango deseado
    if participacion_actual < objetivo - tolerancia:
        # Incrementar participación seleccionando más agentes aleatoriamente
        no_participantes = [agente for agente in agentes if not agente.participa]
        extra_participantes = int((objetivo - participacion_actual) * total_agentes)
        seleccionados = random.sample(no_participantes, min(extra_participantes, len(no_participantes)))
        for agente in seleccionados:
            agente.participa = True

    elif participacion_actual > objetivo + tolerancia:
        # Reducir participación seleccionando menos agentes aleatoriamente
        seleccionados = random.sample(participantes, int((participacion_actual - objetivo) * total_agentes))
        for agente in seleccionados:
            agente.participa = False

#Creación del modelo: número de agentes, función de activación,
#número de nodos y pr de interacción con agentes fuera de,
#tipo de red, y steps

class VotingModel(Model):
    def __init__(self, coef_df, csv_path= os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'demograficos.csv'), k=20, p=0.20):
        self.random = random.Random(42)
        self.schedule = RandomActivation(self)
        self.steps = 0

        # Leer la tabla de proporciones desde el archivo CSV
        df = pd.read_csv(csv_path)
        self.num_agents = int(df['NUMERO_AGENTES'].sum())

        # Crear una red de pequeño mundo
        self.graph = nx.watts_strogatz_graph(n=self.num_agents, k=k, p=p)
        self.grid = NetworkGrid(self.graph)

        variable_names = coef_df['Feature'].tolist()

         # Crear agentes según las combinaciones de género, edad y distrito
        for index, row in df.iterrows():
            num_agents = int(row['NUMERO_AGENTES'])
            distrito = int(row['DISTRITO'])
            genero = 0 if 'HOMBRES' in row['COMBINACION'] else 1
            edad_match = re.search(r'(\d{2,})', row['COMBINACION'])
            edad_range = edad_match.group(1) if edad_match else None
            factor_expansion = row['FACTOR']

            for _ in range(num_agents):
                edad = self.obtener_edad(edad_range)
                edad_normalizada = (edad - 18) / (85 - 18)  # Normalizar edad entre 0 y 1

                # Generar variables aleatorias excluyendo 'Edad', 'Genero_1' y 'Distrito'
                variables = {
                    name: self.random.choice([0, 1])
                    for name in variable_names if name not in ['Edad', 'Genero_1', 'Distrito' "FACTOR"]}


                # Evaluar las variables específicas
                eval_jjf_4 = variables.get("Eval_JJF_4", 0)
                eval_jjf_8 = variables.get("Eval_JJF_8", 0)
                eval_jjf_9 = variables.get("Eval_JJF_9", 0)
                eval_jjf_10 = variables.get("Eval_JJF_10", 0)
                simpatia_fut_2 = variables.get("Simpatia_FUT_2", 0)
                simpatia_fut_3 = variables.get("Simpatia_FUT_3", 0)
                simpatia_fut_4 = variables.get("Simpatia_FUT_4", 0)

                # Regla para determinar el voto
                 # Regla para determinar el voto
                if any([eval_jjf_4, eval_jjf_8, eval_jjf_9, eval_jjf_10,
                        simpatia_fut_2, simpatia_fut_3, simpatia_fut_4]):
                    voto = "Candidato B"
                else:
                    voto = "Candidato A"



                # Crear y agregar el agente con todas las variables
                agent = VoterAgent(
                    unique_id=self.schedule.get_agent_count(),
                    model=self,
                    Edad=edad_normalizada,
                    Genero_1=genero,
                    Distrito=distrito,
                    Factor_Expansion=factor_expansion,
                    voto=voto,
                    coef_df=coef_df,
                    **variables  # Pasar todas las variables como argumentos al agente
                )
                self.schedule.add(agent)
                node = self.random.choice(list(self.graph.nodes()))
                self.grid.place_agent(agent, node)


        # **Asignar participación inicial basada en el género y escolaridad**
        for agent in self.schedule.agents:
            agent.participa = participar_en_votacion(agent.Genero_1, agent)

        # **Normalizar para asegurar el 70% de participación**

        num_participan = sum(1 for agent in self.schedule.agents if agent.participa)
        normalizar_participacion(self.schedule.agents)
        num_participan = sum(1 for agent in self.schedule.agents if agent.participa)

        # **Asignar el voto emitido según participación**
        for agent in self.schedule.agents:
            agent.voto_emitido = "Sí" if agent.participa else "No"
            num_votan = sum(1 for agent in self.schedule.agents if agent.voto_emitido == "Sí")

        # Recopilar datos del modelo
        self.datacollector = DataCollector(
            agent_reporters={"Voto": "voto", "Edad": "Edad", "Genero_1": "Genero_1", "Distrito": "Distrito"})


    def obtener_edad(self, edad_str):
        try:
            if edad_str.isdigit():
                return int(edad_str)
            elif '_' in edad_str:
                edad_start, edad_end = map(int, edad_str.split('_'))
                return random.randint(edad_start, edad_end)
            elif edad_str == "65_Y_MAS":
                return random.randint(65, 80)
            else:
                raise ValueError(f"Formato de edad no reconocido: {edad_str}")
        except Exception as e:
            raise ValueError(f"Error al procesar la edad: {edad_str}. Detalle: {e}")

    def contar_votos_con_expansion(self):
        votos_a = sum(agent.Factor_Expansion for agent in self.schedule.agents if agent.voto == "Candidato A" and agent.participa)
        votos_b = sum(agent.Factor_Expansion for agent in self.schedule.agents if agent.voto == "Candidato B" and agent.participa)
        total_votos = votos_a + votos_b

        if total_votos == 0:
            porcentaje_a = 0
            porcentaje_b = 0
        else:
            porcentaje_a = (votos_a / total_votos) * 100
            porcentaje_b = (votos_b / total_votos) * 100

        return votos_a, votos_b, porcentaje_a, porcentaje_b



    def step(self):
        self.datacollector.collect(self)
        # Ejecutar el paso de cada agente
        self.schedule.step()
        self.steps += 1
        self.contar_votos_con_expansion()

        # Contar agentes participantes
        num_participantes = sum(1 for agent in self.schedule.agents if agent.participa)
        print(f"Total de agentes participantes: {num_participantes}")

        # Calcular total de votos ponderados
        total_votos_ponderados = sum(agent.Factor_Expansion for agent in self.schedule.agents if agent.participa)
        print(f"Total de votos ponderados: {total_votos_ponderados}")


def calcular_porcentaje_votos(model, porcentaje_anterior_A=0, porcentaje_anterior_B=0):

    # Contar los votos ponderados para cada candidato
    votos_a = sum(agent.Factor_Expansion for agent in model.schedule.agents if agent.voto == "Candidato A" and agent.participa)
    votos_b = sum(agent.Factor_Expansion for agent in model.schedule.agents if agent.voto == "Candidato B" and agent.participa)
    total_votos = votos_a + votos_b

    # Calcular porcentajes
    if total_votos == 0:
        porcentaje_a = 0
        porcentaje_b = 0
    else:
        porcentaje_a = (votos_a / total_votos) * 100
        porcentaje_b = (votos_b / total_votos) * 100

    return porcentaje_a, porcentaje_b

def clusterizar_agentes(model, num_clusters):
    # Crear una lista con los atributos de cada agente
    data = []
    for agent in model.schedule.agents:
        voto_numerico = 0 if agent.voto == "Candidato A" else 1
        data.append([agent.Genero_1, agent.Edad, voto_numerico])

    # Convertir a un array de NumPy
    data_array = np.array(data)

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data_array)

    # Asignar el cluster a cada agente
    for i, agent in enumerate(model.schedule.agents):
        agent.cluster = kmeans.labels_[i]

#Gráfica de clusters


def graficar_votos_por_cluster(model):
    # Crear una lista con los datos de los agentes
    data = []
    for agent in model.schedule.agents:
        if agent.participa:
            data.append({
                "Cluster": agent.cluster,
                "Voto": agent.voto,
                "Factor_Expansion": agent.Factor_Expansion
            })

    # Convertir a DataFrame
    df = pd.DataFrame(data)

    # Contar votos ponderados por cluster y por candidato
    votos_por_cluster = df.groupby(['Cluster', 'Voto'])['Factor_Expansion'].sum().unstack(fill_value=0)

    # Graficar barras apiladas con los votos ponderados
    votos_por_cluster.plot(kind='bar', stacked=True, colormap='viridis')
    plt.xlabel("Cluster")
    plt.ylabel("Número de Votos Ponderados")
    plt.title("Distribución de Votos Ponderados por Cluster")
    plt.legend(title="Candidato")
    plt.xticks(rotation=0)  # Mantener las etiquetas del eje X horizontales
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Función para calcular participación
def calcular_participacion(model):
    # Crear una lista con la información relevante de cada agente
    data = []
    for agent in model.schedule.agents:
        data.append({
            "Genero": "Femenino" if agent.Genero_1 == 1 else "Masculino",
            "Distrito": agent.Distrito,
            "Participa": agent.participa
        })

    # Convertir la lista en un DataFrame
    df2 = pd.DataFrame(data)

    # Calcular la participación total
    total_participantes = df2['Participa'].sum()
    total_agentes = len(df2)
    porcentaje_participacion = (total_participantes / total_agentes) * 100

    # Crear una tabla pivot con la participación por género y distrito
    tabla_participacion = pd.pivot_table(
        df2,
        values='Participa',
        index='Distrito',
        columns='Genero',
        aggfunc='sum',
        fill_value=0,
        margins=True,
        margins_name="Total"
    )

    return total_agentes, total_participantes, porcentaje_participacion, tabla_participacion

# Probabilidades de voto

def visualizar_histograma_probabilidades(model):
    """
    Genera un histograma de las probabilidades de voto para entender la seguridad del voto.
    """
    # Recolectar las probabilidades de voto de todos los agentes
    probabilidades = [agent.calcular_probabilidad_voto() for agent in model.schedule.agents]

    # Crear el histograma
    plt.figure(figsize=(10, 6))
    plt.hist(probabilidades, bins=20, color='skyblue', edgecolor='black')

    # Añadir etiquetas y título
    plt.xlabel('Probabilidad de Votar por Candidato A')
    plt.ylabel('Número de Agentes')
    plt.title('Histograma de Probabilidades de Voto (Seguridad del Voto)')

    # Mostrar la gráfica
    plt.grid(True)
    plt.show()

def recolectar_resultados(model, coef_df):
    # Contadores para los votos ponderados de cada candidato
    votos_candidato_A = sum(agent.Factor_Expansion for agent in model.schedule.agents if agent.voto == "Candidato A" and agent.participa)
    votos_candidato_B = sum(agent.Factor_Expansion for agent in model.schedule.agents if agent.voto == "Candidato B" and agent.participa)

    # Calcular el total de votos ponderados
    total_votos = votos_candidato_A + votos_candidato_B

    # Calcular porcentajes ponderados
    porcentaje_A = (votos_candidato_A / total_votos) * 100 if total_votos > 0 else 0
    porcentaje_B = (votos_candidato_B / total_votos) * 100 if total_votos > 0 else 0

    # Agregar datos agregados al resultado
    resumen_electoral = {
        "total_votos_ponderados": total_votos,
        "votos_candidato_A_ponderados": votos_candidato_A,
        "votos_candidato_B_ponderados": votos_candidato_B,
        "porcentaje_candidato_A": f"{porcentaje_A:.2f}%",
        "porcentaje_candidato_B": f"{porcentaje_B:.2f}%"
    }

    # Recopilar los coeficientes del modelo
    coeficientes_modelo = coef_df.to_dict(orient='records')

    return {
        "resumen_electoral": resumen_electoral,
        "coeficientes_modelo": coeficientes_modelo
    }

def analizar_chatgpt(resultados):
    resumen_electoral = resultados['resumen_electoral']
    coeficientes_modelo = resultados['coeficientes_modelo']

    prompt = f"""
    Eres un analista político experto. Te proporciono los resultados de una simulación de votantes, incluyendo votos ponderados y los coeficientes asociados al modelo de predicción.

    **Resumen Electoral**:
    {resumen_electoral}

    **Coeficientes del Modelo**:
    {coeficientes_modelo}

    Por favor, realiza el siguiente análisis:
    1. **Resumen General de los Resultados**.
    2. **Análisis de Coeficientes**: Explica cómo los coeficientes han influido en las decisiones de voto.
    3. **Estrategias para los Candidatos**: Recomienda estrategias basadas en los datos para mejorar el desempeño de cada candidato.
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un analista político experto."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content



# Ejecutar el modelo
def ejecutar_mesa(coef_df):
    # Crear el modelo
    model = VotingModel(coef_df=coef_df)

    for step_num in range(5):
        print(f"\n--- Step {step_num + 1} ---")
        model.step()

    # Síntesis de votos
    votos_a, votos_b, porcentaje_a, porcentaje_b = model.contar_votos_con_expansion()
    print("\n--- Resultados Finales ---")
    print(f"Votos por Candidato A (ponderados): {votos_a:.2f}")
    print(f"Votos por Candidato B (ponderados): {votos_b:.2f}")
    print(f"Porcentaje de votos por Candidato A: {porcentaje_a:.2f}%")
    print(f"Porcentaje de votos por Candidato B: {porcentaje_b:.2f}%")

    # Realizar clustering de los agentes y visualizar los resultados con el nuevo gráfico
    print("\n--- Distribución de Votos por Cluster ---")
    clusterizar_agentes(model, num_clusters=4)  # Asegúrate de llamar a esta función para asignar clusters
    #graficar_votos_por_cluster(model)

    # Calcular y mostrar la participación
    total_agentes, total_participantes, porcentaje_participacion, tabla_participacion = calcular_participacion(model)
    print(f"\n--- Participación Electoral Total ---")
    print(f"Total de agentes: {total_agentes}")
    print(f"Total de participantes: {total_participantes}")
    print(f"Porcentaje de participación: {porcentaje_participacion:.2f}%\n")

    print("--- Participación por Género y Distrito ---")
    print(tabla_participacion)

    # Visualizar el histograma de probabilidades de voto
    print("\n--- Histograma de Probabilidades de Voto ---")
    #visualizar_histograma_probabilidades(model)

   # Recolectar los resultados
    resultados = recolectar_resultados(model, coef_df)

    # Analizar los resultados con la API de ChatGPT
    analisis = analizar_chatgpt(resultados)

    # Imprimir el análisis final
    print("\n--- Análisis de la API de ChatGPT ---")
    print(analisis)

    return votos_a, votos_b, porcentaje_a, porcentaje_b, porcentaje_participacion, analisis
