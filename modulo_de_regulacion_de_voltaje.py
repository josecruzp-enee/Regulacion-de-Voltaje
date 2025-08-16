
"""
Created on Wed May 21 20:27:41 2025

@author: José Nikol Cruz
"""


"""
Resumen de Secciones y Funciones del Script

# ====================== Sección #1: Diccionarios y Datos Básicos ======================
  - Subsección #1.1: Diccionario de conductores
    - Función #1.1.1: bibloteca_conductores() - Devuelve diccionario con parámetros eléctricos de conductores estándar

  - Subsección #1.2: Lectura y preparación de datos desde Excel
    - Función #1.2.1: leer_hojas_excel(archivo) - Lee las hojas principales del archivo Excel
    - Función #1.2.2: extraer_valores_clave(df_parametros, df_info) - Extrae parámetros clave desde DataFrames
    - Función #1.2.3: preparar_conexiones(df_conexiones) - Prepara columnas y listas necesarias para cálculo
    - Función #1.2.4: cargar_datos_circuito(archivo) - Función principal que carga y retorna todos los datos necesarios



# ====================== Sección #2: Cálculos Eléctricos Básicos ======================
  - Subsección #2.1: Constantes y propiedades de línea
    - Función #2.1.1: definir_constantes() - Devuelve diccionario con constantes físicas y eléctricas
    - Función #2.1.2: obtener_resistencia_por_vano() - Calcula resistencia por tramo o vano
    - Función #2.1.3: obtener_reactancia_por_vano_geometrica() - Calcula reactancia inductiva por vano

  - Subsección #2.2: Cálculo de impedancia y admitancia de línea
    - Función #2.2.1: resistencia_por_vano - Calcula la resistencia del conductor por vano (tramo de línea) según el calibre y la resistividad.
    - Función #2.2.2: reactancia_por_vano_geometrica - Calcula la reactancia de un vano de línea usando la disposición geométrica de los conductores.
    - Función #2.2.3: calcular_impedancia - Combina resistencia y reactancia para obtener la impedancia total del vano.
    - Función #2.2.4: calcular_admitancia - Calcula la admitancia (inversa de la impedancia) de un vano para modelar el flujo de corriente.

   - Subsección #2.3: Cálculo de potencias y cargas por usuario
    - Función #2.3.1: calcular_kva_usuario - Determina la demanda de potencia en kVA para cada usuario según su consumo y factor de potencia.
    - Función #2.3.2: calcular_potencias - Calcula potencias activa, reactiva y aparente totales a partir de tensiones y corrientes medidas.
    - Función #2.3.3: calcular_impedancia_admitancia - Calcula y devuelve tanto la impedancia como la admitancia para un tramo específico.
    - Función #2.3.4: calcular_carga_por_usuario - Distribuye la carga total entre los usuarios conectados según su consumo estimado.
    - Función #2.3.5: imprimir_resumen_carga - Muestra en consola un resumen de cargas por usuario y totales para verificación rápida.
    - Función #2.3.6: calcular_potencia_carga - Calcula la potencia consumida por cada carga o grupo de cargas específicas.



# ====================== Sección #3: Matrices y Nodos ======================
  -  Subsección #3.1: Manejo de nodos y construcción de matrices 
    - Función #3.1.1: obtener_nodos_e_indices - Obtiene la lista ordenada de nodos y un diccionario que mapea cada nodo a su índice en la matriz.
    - Función #3.1.2: construir_matriz_admitancia - Construye la matriz nodal de admitancia Y a partir del DataFrame con admitancias de vanos y cargas.
    - Función #3.1.3: extraer_submatrices - Extrae la submatriz Yrr (sin nodo slack), vector Y_r0, define nodo slack y su índice.
    - Función #3.1.4: calcular_matriz_admitancia - Función principal para obtener nodos, construir matriz Y y extraer submatrices necesarias para cálculo de voltajes.




# ====================== Sección #4: Cálculo de Voltajes y Regulación ======================
  - # Subsección #4.1: Cálculo de voltajes nodales
    - Función #4.1.1: calcular_voltajes_nodales - Calcula los voltajes nodales resolviendo el sistema lineal de admitancias
 
  - # Subsección #4.2: Cálculo de regulación de voltaje
    - Función #4.2.1: calcular_regulacion_voltaje - Calcula la regulación de voltaje (%) en cada nodo en relación al nodo slack



# ====================== Sección #5: Cálculo de Corrientes, Pérdidas y Proyección ======================

  # Subsección #5.1: Cálculo de corrientes en los vanos
    - Función #5.1.1: calcular_corrientes(df, V) - Calcula corriente en cada vano usando Z y voltajes nodales.
    - Función #5.1.2: calcular_perdidas_y_proyeccion(df) - Calcula pérdidas por vano y las proyecta a 15 años.
    - Función #5.1.3: proyectar_demanda(...) - Proyecta demanda y % de carga sobre transformador.


# ====================== Sección #6: Cálculo Simbólico con SymPy para Validación ======================
  - Subsección #6.1: Variables y matrices simbólicas
    - Función #6.1.1: crear_matrices_y_variables_simbólicas(Yrr, nodos, nodo_slack) - Crea matrices y variables simbólicas
    - Función #6.1.2: resolver_sistema_simbólico(Yrr_sym, Y_r0_sym, V0) - Resuelve sistema simbólico de voltajes nodales
    - Función #6.1.3: safe_print(...) - Impresión segura para evitar errores de codificación
    - Función #6.1.4: imprimir_resultados_simbólicos(Yrr_simb_con_cargas, vector_voltajes) - Imprime matrices y variables simbólicas
    - Función #6.1.5: construir_y_resolver_simbólico(Yrr, Y_r0, V0, nodos, nodo_slack) - Función orquestadora que llama las anteriores y devuelve resultados simbólicos


# ====================== Sección #7: Visualización y Gráficos ======================
- Subsección #7.1: Gráficos básicos y preparación visual
  - Función #7.1.1: graficar_voltajes(df_voltajes) - Genera gráfico de voltajes nodales
  - Función #7.1.2: graficar_corrientes(df_corrientes) - Genera gráfico de corrientes en vanos
  - Función #7.1.3: graficar_perdidas(df_perdidas) - Grafica pérdidas eléctricas por tramo

# ====================== Sección #8: Manejo y Dibujo del Grafo de la Red ======================
- Subsección #8.1: Creación y visualización de grafo
  - Función #8.1.1: crear_grafo(nodos_inicio, nodos_final, usuarios, distancias) - Crea grafo NetworkX con atributos
  - Función #8.1.2: calcular_posiciones_red(G, nodo_raiz, escala, dy) - Calcula posiciones 2D para visualización
  - Función #8.1.3: dibujar_nodos_transformador(ax, G, posiciones, capacidad_transformador) - Dibuja nodo transformador destacado
  - Función #8.1.4: dibujar_nodos_generales(ax, G, posiciones) - Dibuja otros nodos de la red
  - Función #8.1.5: dibujar_aristas(ax, G, posiciones) - Dibuja aristas y bucles (autoenlaces)
  - Función #8.1.6: dibujar_etiquetas_nodos(ax, G, posiciones) - Añade etiquetas con números de nodo
  - Función #8.1.7: dibujar_acometidas(ax, posiciones, nodos_final, usuarios) - Muestra cargas en nodos finales
  - Función #8.1.8: dibujar_distancias_tramos(ax, G, posiciones) - Muestra distancia en metros en cada tramo
  - Función #8.1.9: crear_grafico_nodos(nodos_inicio, nodos_final, usuarios, distancias, capacidad_transformador) - Orquesta todo el dibujo y devuelve buffer imagen


# ====================== Sección #9: Contenido de Informe PDF ======================
- Subsección #9.1: Preparación de información para reporte
  - Función #9.1.1: preparar_info_proyecto - Ajusta nombres y agrega factor de coincidencia a los DataFrames de info y parámetros del proyecto.
  - Función #9.1.2: seccion_info_proyecto - Crea los elementos de contenido para la sección de información y parámetros del proyecto (para reporte PDF).
  - Función #9.1.3: crear_tabla_usuarios_conectados - Prepara tabla con nodos, distancia y usuarios conectados.
  - Función #9.1.4: crear_tabla_voltajes - Crea sección de análisis de voltajes nodales con tabla formateada.
  - Función #9.1.5: crear_comentario_regulacion - Genera comentario con color según si la regulación de voltaje está dentro del rango ±5%.
  - Función #9.1.6: crear_grafico_voltajes_pdf - Genera gráfico de voltajes nodales como imagen para insertar en reporte.
  - Función #9.1.7: seccion_usuarios_y_voltajes - Genera sección que incluye tabla de usuarios, tabla de voltajes, comentario y gráfico.
  - Función #9.1.8: crear_grafico_demanda - Genera gráfico de proyección de demanda en un periodo de años.
  - Función #9.1.9: preparar_tabla_proyeccion - Prepara DataFrame para mostrar proyección de demanda y pérdidas con formato adecuado.
  - Función #9.1.10: crear_elementos_tabla - Crea elementos para mostrar tabla de proyección con subtitulo y espacio.
  - Función #9.1.11: crear_comentario_cargabilidad - Crea comentario sobre cargabilidad del equipo basado en % carga y voltajes.
  - Función #9.1.12: seccion_proyeccion_perdidas_y_demanda - Genera sección completa de proyección de demanda, pérdidas, gráfico y comentario de cargabilidad.



# ====================== Sección #10: Visualización de la Red Eléctrica (Gráficos de Nodos) ======================
- Subsección #10.1: Funciones para graficar nodos y componentes de la red
  - Función #10.1.1: crear_grafo(nodos_inicio, nodos_final, usuarios, distancias) - Crea un grafo NetworkX con nodos y atributos
  - Función #10.1.2: calcular_posiciones_red(G, nodo_raiz=1, escala=0.05, dy=1.5) - Calcula posiciones 2D para cada nodo desde nodo raíz
  - Función #10.1.3: dibujar_nodos_transformador(ax, G, posiciones, capacidad_transformador) - Dibuja el nodo transformador con símbolo especial y etiqueta
  - Función #10.1.4: dibujar_nodos_generales(ax, G, posiciones) - Dibuja los nodos generales con forma y color específicos
  - Función #10.1.5: dibujar_aristas(ax, G, posiciones) - Dibuja aristas y bucles con distinción visual
  - Función #10.1.6: dibujar_etiquetas_nodos(ax, G, posiciones) - Añade etiquetas numéricas a cada nodo
  - Función #10.1.7: dibujar_acometidas(ax, posiciones, nodos_final, usuarios) - Dibuja acometidas con corriente de usuarios en nodos finales
  - Función #10.1.8: dibujar_distancias_tramos(ax, G, posiciones) - Muestra distancias en metros entre nodos
  - Función #10.1.9: crear_grafico_nodos(nodos_inicio, nodos_final, usuarios, distancias, capacidad_transformador) - Orquesta el gráfico completo y devuelve buffer de imagen



# ====================== Sección #11: Generación de Informe PDF ======================
- Subsección #11.1: Creación y armado del informe en PDF con reportlab
  - Función #11.1.1: crear_pdf - Genera PDF con todas las secciones y gráficos incluidos
  - Función #11.1.2: cargar_y_preparar_datos - Carga datos, calcula parámetros y prepara DataFrames y listas
  - Función #11.1.3: calcular_flujo_carga_y_perdidas - Ejecuta flujo de carga, voltajes, corrientes, pérdidas y proyecciones
  - Función #11.1.4: calcular_regulacion_y_proyeccion - Calcula proyección de demanda y regulación de voltajes
  - Función #11.1.5: preparar_proyeccion_y_regulacion - Prepara DataFrames de proyección y regulación para informe
  - Función #11.1.6: generar_informe_pdf - Orquesta la creación final del PDF con datos, gráficos y secciones


"""
 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
import tempfile
import os
import math
import sympy as sp
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus import Image
import webbrowser
import seaborn as sns
from io import BytesIO
from reportlab.lib.enums import TA_CENTER
import platform
from reportlab.lib.units import inch
from reportlab.lib.units import cm, inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Frame
from reportlab.platypus import PageTemplate
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import red
import io
import networkx as nx
import matplotlib.pyplot as plt
import io
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rcParams
import streamlit as st
import sys
from colorama import init, Fore, Style
init(autoreset=True)






#Para que pueda imprimir en la web
def safe_print(text):
    if sys.stdout.encoding and sys.stdout.encoding.lower() == 'utf-8':
        print(text)  # si consola soporta UTF-8, imprime todo
    else:
        # Opcional: imprimir sin emojis para evitar errores
        import re
        text_sin_emojis = re.sub(r'[^\x00-\x7F]+','', text)
        print(text_sin_emojis)




# ====================== Sección #1: Diccionarios y Datos Básicos ======================

# Subsección #1.1: Diccionario de conductores y parámetros básicos

# Función #1.1.1: bibloteca_conductores
# Esta función devuelve un diccionario con los tipos de conductores
# y sus propiedades eléctricas: resistencia (R), reactancia (X) y radio en metros.
# No recibe parámetros. Devuelve el diccionario conductores.
def bibloteca_conductores():
    conductores = {
        '2'  : {'R': 0.1563, 'X': 0.08,  'radio_m': 0.00735},
        '1/0': {'R': 0.1239, 'X': 0.078, 'radio_m': 0.00825},
        '2/0': {'R': 0.0983, 'X': 0.075, 'radio_m': 0.00927},
        '3/0': {'R': 0.3740, 'X': 0.29277, 'radio_m': 0.01040},
        '4/0': {'R': 0.0618, 'X': 0.071, 'radio_m': 0.01168},
        '266': {'R': 0.0590, 'X': 0.070, 'radio_m': 0.01290}
    }
    return conductores
 


# Subsección #1.2: Cálculo de demanda según área y factores de coincidencia, según Manual de Obras de ENEE. 

# Función #1.2.1: calcular_kva_por_area
# Calcula el kVA asignado según el área del lote en m², basado en reglas de bloques.
# Recibe: area_m2 (float)
# Devuelve: kVA asignado (float)
def calcular_kva_por_area(area_m2):
    if area_m2 < 75:
        return 1.0
    elif 75 <= area_m2 <= 149:
        return 1.5
    elif 150 <= area_m2 <= 189:
        return 2.0
    elif 190 <= area_m2 <= 439:
        return 3.0
    elif 440 <= area_m2 <= 750:
        bloques = (area_m2 - 440) / 100
        bloques_int = int(bloques) if bloques.is_integer() else int(bloques) + 1
        kva_adicional = 1.2 * bloques_int
        kva_total = 5.2 + kva_adicional
        return min(kva_total, 9.0)
    else:
        raise ValueError("Área mayor a 750 m2: servicio exclusivo")

# Función #1.2.2: factor_coincidencia
# Asigna el factor de coincidencia según la cantidad de usuarios (lotes).
# Recibe: usuarios (int)
# Devuelve: factor de coincidencia (float)
def factor_coincidencia(usuarios):
    tabla = [
        (1, 1.00), (2, 0.93), (3, 0.85), (4, 0.77), (5, 0.69), (6, 0.67),
        (7, 0.66), (8, 0.65), (9, 0.64), (10, 0.63), (12, 0.62), (16, 0.61),
        (22, 0.60), (28, 0.59), (34, 0.58), (41, 0.57), (51, 0.56), (63, 0.55)
    ]
    for umbral, valor in tabla:
        if usuarios <= umbral:
            return valor
    return 0.54


# Subsección #1.3: Funciones para leer y preparar datos desde Excel

# Función #1.3.1: leer_hojas_excel
# Lee las hojas relevantes del archivo Excel con pandas.
# Recibe: archivo (string ruta)
# Devuelve: 3 DataFrames: conexiones, parametros, info_proyecto
def leer_hojas_excel(archivo):
    df_conexiones = pd.read_excel(archivo, sheet_name='conexiones')
    df_parametros = pd.read_excel(archivo, sheet_name='parametros', usecols=[0, 1])
    df_info = pd.read_excel(archivo, sheet_name='info_proyecto', usecols=[0, 1])

    df_parametros.columns = ['Parámetro', 'Valor']
    df_info.columns = ['info', 'Valor']

    return df_conexiones, df_parametros, df_info

# Función #1.3.2: extraer_valores_clave
# Extrae parámetros clave desde los DataFrames de parámetros e info.
# Recibe: df_parametros, df_info (DataFrames)
# Devuelve: valores separados (tipo_conductor, area_lote, capacidad_transformador, etc.)
def extraer_valores_clave(df_parametros, df_info):
    tipo_conductor = df_parametros.loc[df_parametros['Parámetro'] == 'TipoConductor', 'Valor'].values[0]
    area_lote = float(df_parametros.loc[df_parametros['Parámetro'] == 'area_lote', 'Valor'].values[0])
    capacidad_transformador = float(df_parametros.loc[df_parametros['Parámetro'] == 'capacidad_transformador', 'Valor'].values[0])

    proyecto_numero = df_info.loc[df_info['info'] == 'NumeroProyecto', 'Valor'].values[0]
    proyecto_nombre = df_info.loc[df_info['info'] == 'NombreProyecto', 'Valor'].values[0]
    transformador_numero = df_info.loc[df_info['info'] == 'NumeroTransformador', 'Valor'].values[0]

    return tipo_conductor, area_lote, capacidad_transformador, proyecto_numero, proyecto_nombre, transformador_numero

# Función #1.3.3: preparar_conexiones
# Convierte a numérico columnas relevantes del DataFrame conexiones y crea listas.
# Recibe: df_conexiones (DataFrame)
# Devuelve: df_conexiones modificado, y listas de usuarios, distancias, nodos inicio y fin.
def preparar_conexiones(df_conexiones):
    for col in ['distancia', 'usuarios', 'nodo_inicial', 'nodo_final']:
        df_conexiones[col] = pd.to_numeric(df_conexiones[col], errors='coerce')


    usuarios = df_conexiones['usuarios'].tolist()
    distancias = df_conexiones['distancia'].tolist()
    nodos_inicio = df_conexiones['nodo_inicial'].tolist()
    nodos_final = df_conexiones['nodo_final'].tolist()

    return df_conexiones, usuarios, distancias, nodos_inicio, nodos_final

# Función #1.3.4: cargar_datos_circuito
# Función principal para cargar y preparar todos los datos necesarios desde el Excel.
# Recibe: archivo (string)
# Devuelve: múltiples variables necesarias para el análisis
def cargar_datos_circuito(archivo='datos_circuito.xlsx'):
    df_conexiones, df_parametros, df_info = leer_hojas_excel(archivo)
    tipo_conductor, area_lote, capacidad_transformador, proyecto_numero, proyecto_nombre, transformador_numero = extraer_valores_clave(df_parametros, df_info)
    df_conexiones, usuarios, distancias, nodos_inicio, nodos_final = preparar_conexiones(df_conexiones)

    return (df_conexiones, df_parametros, df_info,
            tipo_conductor, area_lote, capacidad_transformador,
            proyecto_numero, proyecto_nombre, transformador_numero,
            usuarios, distancias, nodos_inicio, nodos_final)

# ====================== Sección #2: Cálculos Eléctricos ======================

# Subsección #2.1: Constantes y parámetros físicos

# Función #2.1.1: definir_constantes
# Define las constantes eléctricas y físicas usadas en los cálculos, como frecuencia, separación entre fases, tensión nominal y factor de potencia.
# No recibe parámetros. Devuelve un diccionario con las constantes.
def definir_constantes():
    constantes = {
        'frecuencia': 60,                 # Hz
        'separacion_fases_metros': 0.2032,  # 8 pulgadas en metros
        'V_nom': 240,                    # voltios
        'fp': 0.9                       # factor de potencia
    }
    return constantes


# Subsección #2.2: Cálculo de impedancia y admitancia de línea


# Función #2.2.1: resistencia_por_vano
# Calcula la resistencia total del vano (doble tramo) en ohmios.
# Recibe: conductores (diccionario), tipo_conductor (str), distancia_metros (float)
# Devuelve: resistencia total (float)
def resistencia_por_vano(conductores, tipo_conductor, distancia_metros):
    if tipo_conductor not in conductores:
        raise ValueError(f"Conductor '{tipo_conductor}' no está en el diccionario.")
    distancia_km = distancia_metros / 1000
    return 2 * conductores[tipo_conductor]['R'] * distancia_km


# Función #2.2.2: reactancia_por_vano_geometrica
# Calcula la reactancia inductiva del vano considerando la geometría y frecuencia.
# Recibe: distancia_metros (float), separacion_fases_metros (float), radio_conductor_metros (float), frecuencia (float, default 60 Hz)
# Devuelve: reactancia inductiva total (float)
def reactancia_por_vano_geometrica(distancia_metros, separacion_fases_metros, radio_conductor_metros, frecuencia=60):
    mu_0 = 4 * np.pi * 1e-7
    L_prim = (mu_0 / (2 * np.pi)) * np.log(separacion_fases_metros / radio_conductor_metros)
    X_prim = 2 * np.pi * frecuencia * L_prim
    X_vano = 2 * X_prim * distancia_metros
    return X_vano


# Función #2.2.3: calcular_impedancia
# Calcula la impedancia compleja de un vano usando resistencia y reactancia.
# Recibe: row (pandas Series con columnas 'resistencia_vano' y 'reactancia_vano')
# Devuelve: impedancia compleja (complex)
def calcular_impedancia(row):
    d = row['distancia']
    if d == 0:
        Z_vano = 0 + 0j
    else:
        Z_vano = row['resistencia_vano'] + 1j * row['reactancia_vano']
    return Z_vano


# Función #2.2.4: calcular_admitancia
# Calcula la admitancia (inversa de impedancia), evitando división por cero.
# Recibe: z (complex)
# Devuelve: admitancia (complex)
def calcular_admitancia(z):
    if abs(z) < 1e-12:
        Y_vano = 0 + 0j
    else:
        Y_vano = 1 / z
    print(f"Impedancia: {z}, Admitancia calculada: {Y_vano}")
    return Y_vano


# Subsección #2.3: Cálculo de potencias y cargas por usuario

# Función #2.3.1: calcular_kva_usuario
# Calcula el kVA asignado por usuario en función del área del lote.
# Recibe: area_lote (float)
# Devuelve: kVA usuario (float)



# Función #2.3.2: calcular_potencias
# Calcula las potencias activas, reactivas y complejas para cada nodo según usuarios, kVA y factor de coincidencia.
# Recibe: df (DataFrame con columna 'usuarios'), kva_usuario (float), factor_coinc (float), fp (float, default 0.9)
# Devuelve: DataFrame con columnas nuevas: 'kva', 'P', 'Q', 'S_compleja', 'S_VA'
def calcular_potencias(df, kva_usuario, factor_coinc, fp=0.9):
    df['kva'] = df['usuarios'] * kva_usuario * factor_coinc
    df['P'] = df['kva'] * fp
    df['Q'] = df['kva'] * np.sin(np.arccos(fp))
    df['S_compleja'] = df['P'] + 1j * df['Q']
    df['S_VA'] = df['S_compleja'] * 1000
    return df


# Función #2.3.3: calcular_impedancia_admitancia
# Calcula la impedancia y admitancia de carga para cada nodo, junto con conductancia y susceptancia.
# Recibe: df (DataFrame con columna 'S_VA'), V_nom (float, default 240)
# Devuelve: DataFrame con columnas 'Z_carga', 'Y_carga', 'G_carga', 'B_carga'
def calcular_impedancia_admitancia(df, V_nom=240):
    df['Z_carga'] = (V_nom ** 2) / np.conj(df['S_VA'])
    df['Y_carga'] = 1 / df['Z_carga']
    df['G_carga'] = df['Y_carga'].apply(lambda y: y.real)
    df['B_carga'] = df['Y_carga'].apply(lambda y: y.imag)
    return df


# Función #2.3.4: calcular_carga_por_usuario
# Función general que combina cálculo de potencias e impedancias para cada nodo.
# Recibe: df (DataFrame), kva_usuario (float), factor_coinc (float), fp (float), V_nom (float)
# Devuelve: DataFrame con cálculos completos
def calcular_carga_por_usuario(df, kva_usuario, factor_coinc, fp=0.9, V_nom=240):
    df = calcular_potencias(df, kva_usuario, factor_coinc, fp)
    df = calcular_impedancia_admitancia(df, V_nom)
    return df


# Función #2.3.5: imprimir_resumen_carga
# Imprime en consola resumen de carga total y tabla de admitancias y cargas por nodo (para depuración).
# Recibe: df (DataFrame), total_usuarios (int), kva_usuario (float), factor_coinc (float), potencia_total_kva (float)
# Devuelve: nada
def imprimir_resumen_carga(df, total_usuarios, kva_usuario, factor_coinc, potencia_total_kva):
    print(f"\nTotal usuarios: {total_usuarios}")
    print(f"kVA por usuario: {kva_usuario:.2f}")
    print(f"Factor de coincidencia: {factor_coinc:.2f}")
    print(f"Potencia total demandada (kVA): {potencia_total_kva:.2f}")
    print("="*40, "\n")

    '''if 'Y_vano' in df.columns:
        print("\nTabla de Admitancias:")
        print(df[['Y_vano', 'Y_carga']])
        print("="*40, "\n")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("\nTabla de Cargas por Nodo:")
    print(df[['nodo_final', 'kva', 'P', 'Q', 'S_compleja', 'Z_carga', 'Y_carga']])
    print("="*40, "\n")
    '''

# Función #2.3.6: calcular_potencia_carga
# Función principal para calcular potencias totales, carga por usuario y factor de coincidencia.
# Recibe: df (DataFrame), area_lote (float), fp (float), V_nom (float)
# Devuelve: df (DataFrame con cargas), potencia_total_kva (float), factor_coinc (float)
def calcular_potencia_carga(df, area_m2, fp=0.9, V_nom=240):
    total_usuarios = df['usuarios'].sum()
    kva_usuario = calcular_kva_por_area(area_m2)
    factor_coinc = factor_coincidencia(total_usuarios)
    potencia_total_kva = total_usuarios * kva_usuario * factor_coinc

    df = calcular_carga_por_usuario(df, kva_usuario, factor_coinc, fp=fp, V_nom=V_nom)
    imprimir_resumen_carga(df, total_usuarios, kva_usuario, factor_coinc, potencia_total_kva)

    return df, potencia_total_kva, factor_coinc



# ====================== Sección #3: Matrices y Nodos ======================

# Subsección #3.1: Manejo de nodos y construcción de matrices

# Función #3.1.1: obtener_nodos_e_indices
# Obtiene la lista ordenada de nodos y un diccionario que mapea cada nodo a su índice en la matriz.
# Recibe: df (DataFrame con columnas 'nodo_inicial' y 'nodo_final')
# Devuelve: lista nodos (ordenada), diccionario {nodo: índice}
def obtener_nodos_e_indices(df):
    nodos = sorted(set(df['nodo_inicial']).union(set(df['nodo_final'])))
    indice_nodos = {nodo: i for i, nodo in enumerate(nodos)}
    return nodos, indice_nodos


# Función #3.1.2: construir_matriz_admitancia
# Construye la matriz nodal de admitancia Y a partir del DataFrame con admitancias de vanos y cargas.
# Recibe: df (DataFrame con columnas 'nodo_inicial', 'nodo_final', 'Y_vano', 'Y_carga'), nodos (lista), indice_nodos (dicc)
# Devuelve: matriz Y (numpy ndarray)
def construir_matriz_admitancia(df, nodos, indice_nodos):
    n = len(nodos)
    Y = np.zeros((n, n), dtype=complex)

    for _, row in df.iterrows():
        i = indice_nodos[row['nodo_inicial']]
        j = indice_nodos[row['nodo_final']]
        Y_vano = row['Y_vano']

        # Añadir admitancias de vano
        Y[i, i] += Y_vano
        Y[j, j] += Y_vano
        Y[i, j] -= Y_vano
        Y[j, i] -= Y_vano

        # Añadir admitancia de carga en nodo j si existe
        if 'Y_carga' in df.columns and not pd.isna(row['Y_carga']):
            Y[j, j] += row['Y_carga']

    return Y


# Función #3.1.3: extraer_submatrices
# Extrae la submatriz Yrr (sin nodo slack), vector Y_r0, define nodo slack y su índice.
# Recibe: matriz Y (numpy ndarray), lista nodos
# Devuelve: Yrr (numpy ndarray), Y_r0 (numpy ndarray columna), nodo_slack (valor), slack_index (int)
def extraer_submatrices(Y, nodos):
    nodo_slack = nodos[0]  # Primer nodo es nodo slack
    slack_index = 0

    indices_no_slack = [i for i in range(len(nodos)) if i != slack_index]

    Yrr = Y[np.ix_(indices_no_slack, indices_no_slack)]
    Y_r0 = Y[indices_no_slack, slack_index].reshape(-1, 1)

    return Yrr, Y_r0, nodo_slack, slack_index


# Función #3.1.4: calcular_matriz_admitancia
# Función principal para obtener nodos, construir matriz Y y extraer submatrices necesarias para cálculo de voltajes.
# Recibe: df (DataFrame con datos de vanos y cargas)
# Devuelve: matriz Y completa, Yrr, Y_r0, lista nodos, índice slack
def calcular_matriz_admitancia(df):
    nodos, indice_nodos = obtener_nodos_e_indices(df)
    Y = construir_matriz_admitancia(df, nodos, indice_nodos)
    Yrr, Y_r0, nodo_slack, slack_index = extraer_submatrices(Y, nodos)

    '''# Imprimir para depuración
    print("Matriz admitancia nodal (Y):")
    print(Y)
    print("="*40, "\n")
    print(f"Nodo slack definido: {nodo_slack}")
    print(f"Tamaño matriz Yrr: {Yrr.shape}")
    print("Matriz Yrr:")
    print(Yrr)
    print("="*40, "\n")
    print("Vector Y_r0:")
    print(Y_r0)
    print("="*40, "\n")'''

    return Y, Yrr, Y_r0, nodos, slack_index




# ====================== Sección #4: Cálculo de Voltajes y Regulación ======================

# Subsección #4.1: Cálculo de voltajes nodales

# Función #4.1.1: calcular_voltajes_nodales
# Calcula los voltajes nodales resolviendo el sistema lineal de admitancias (excluyendo nodo slack).
# Recibe:
# - Yrr: matriz admitancia submatriz sin nodo slack
# - Y_r0: vector admitancia con nodo slack
# - slack_index: índice del nodo slack
# - nodos: lista de nodos
# - V0: voltaje nodo slack (por defecto 240 V con ángulo 0)
# Devuelve:
# - Vector complejo de voltajes V con nodo slack incluido
# - DataFrame con voltajes nodales y ángulos
def calcular_voltajes_nodales(Yrr, Y_r0, slack_index, nodos, V0=240+0j):
    n = len(nodos)
    indices_no_slack = [i for i in range(n) if i != slack_index]

    # Resolver V_r = inv(Yrr) * (-Y_r0 * V0)
    lado_derecho = -Y_r0 * V0
    Y_rr_inv = np.linalg.inv(Yrr)
    V_r = np.dot(Y_rr_inv, lado_derecho)

    # Construcción vector completo con nodo slack incluido
    V = np.zeros(n, dtype=complex)
    V[slack_index] = V0
    for i, idx in enumerate(indices_no_slack):
        V[idx] = V_r[i]

    print("Voltajes nodales calculados:")
    for i, nodo in enumerate(nodos):
        magnitud = abs(V[i])
        angulo = np.angle(V[i], deg=True)
        print(f"Nodo {nodo}: |V| = {magnitud:.2f} V, ángulo = {angulo:.2f}°")
    print("="*40, "\n")

    df_voltajes = pd.DataFrame({
        'Nodo': nodos,
        'Magnitud (V)': [abs(v) for v in V],
        'Ángulo (°)': [np.angle(v, deg=True) for v in V]
    })

    return V, df_voltajes


# Subsección #4.2: Cálculo de regulación de voltaje

# Función #4.2.1: calcular_regulacion_voltaje
# Calcula la regulación de voltaje (%) en cada nodo en relación al nodo slack.
# Recibe:
# - V: vector complejo de voltajes nodales
# - nodos: lista nodos
# - nodo_slack: nodo slack
# Devuelve:
# - DataFrame con nodo, voltaje en p.u., voltaje absoluto y regulación %
def calcular_regulacion_voltaje(V, nodos, nodo_slack):
    V0 = V[nodos.index(nodo_slack)]
    regulacion = []

    for i, nodo in enumerate(nodos):
        regul = (abs(V0) - abs(V[i])) / abs(V0) * 100
        regulacion.append({
            'Nodo': nodo,
            'Voltaje (p.u.)': V[i],
            'Voltaje Absoluto (V)': abs(V[i]) * 240,  # Ajusta base si necesario
            'Regulación (%)': regul
        })

    df_regulacion = pd.DataFrame(regulacion)
    
    return df_regulacion




# ====================== Sección #5: Cálculo de Corrientes, Pérdidas y Proyección ======================

# Subsección #5.1: Cálculo de corrientes por vano

# Función #5.1.1: calcular_corrientes
# Calcula las corrientes en cada vano basándose en los voltajes nodales y las impedancias de vano.
# Recibe:
# - df: DataFrame con columnas 'nodo_inicial', 'nodo_final', 'Z_vano'
# - V: array o lista con voltajes nodales complejos
# Devuelve:
# - df con columna nueva 'I_vano' que contiene la corriente compleja por vano
def calcular_corrientes(df, V):
    I_vanos = []

    for _, row in df.iterrows():
        ni = int(row['nodo_inicial'])
        nf = int(row['nodo_final'])
        Z = row['Z_vano']

        if Z == 0:
            I = 0 + 0j
        else:
            I = (V[ni - 1] - V[nf - 1]) / Z
        
        I_vanos.append(I)
    
    df = df.copy()
    df['I_vano'] = I_vanos
    return df




# Función #5.1.2: calcular_perdidas_y_proyeccion
# Calcula las pérdidas en cada vano y proyecta las pérdidas anuales a 15 años con crecimiento.
# Recibe:
# - df: DataFrame con columnas 'I_vano' y 'resistencia_vano'
# Devuelve:
# - df con columna adicional 'P_perdida' (pérdidas en Watts)
# - perdida_total: pérdidas totales anuales en kWh
# - proyeccion: lista con proyección de pérdidas para 15 años (crecimiento anual 3%)
def calcular_perdidas_y_proyeccion(df):
    P_perdidas = []
    
    for _, row in df.iterrows():
        I = row['I_vano']
        R = row['resistencia_vano'].real
        P_perdida = (abs(I) ** 2) * R
        P_perdidas.append(float(P_perdida))
    
    df = df.copy()
    df['P_perdida'] = P_perdidas

    perdida_total = sum(P_perdidas) * 24 * 365 / 1000  # De Watts a kWh anuales

    proyeccion = [perdida_total * ((1 + 0.03) ** i) for i in range(1, 16)]

    return df, perdida_total, proyeccion




# Función #5.1.3: proyectar_demanda
# Calcula la proyección de demanda para 15 años con tasa de crecimiento especificada.
# Recibe:
# - potencia_total_kva: demanda actual total en kVA
# - df_parametros: DataFrame con parámetros (para obtener capacidad_transformador)
# - df_proyeccion: DataFrame con años (puede estar vacío al inicio)
# - crecimiento: tasa anual de crecimiento (default 2%)
# - años: cantidad de años a proyectar (default 15)
# Devuelve:
# - df_proyeccion actualizado con columnas 'Demanda_kva' y '% Carga (%)', además de formatos string para reporte
def proyectar_demanda(potencia_total_kva, df_parametros, df_proyeccion, crecimiento=0.02, años=15):
    proyeccion_demanda = [
        potencia_total_kva * ((1 + crecimiento) ** i) for i in range(1, años + 1)
    ]
    capacidad_transformador = float(df_parametros.loc['capacidad_transformador', 'Valor'])

    df_proyeccion['Demanda_kva'] = proyeccion_demanda
    df_proyeccion['% Carga (%)'] = (df_proyeccion['Demanda_kva'] / capacidad_transformador) * 100

    # Formateo para reporte
    df_proyeccion['Demanda (kVA)'] = df_proyeccion['Demanda_kva'].map(lambda x: f"{x:,.2f}")
    df_proyeccion['% Carga (%)'] = df_proyeccion['% Carga (%)'].map(lambda x: f"{x:.2f}%")

    return df_proyeccion



"""
# ====================== Sección #6: Cálculo Simbólico con SymPy para Validación ======================

# Subsección #6.1: Creación de variables y matrices simbólicas de admitancia

# Función #6.1.1: crear_matrices_y_variables_simbólicas
# Recibe:
# - Yrr: matriz numpy de admitancia entre nodos no slack
# - nodos: lista completa de nodos
# - nodo_slack: nodo slack (referencia)
# Devuelve:
# - Yrr_simb_con_cargas: matriz simbólica de admitancia con cargas y ramas
# - vector_voltajes: vector simbólico de variables voltajes nodales no slack
def crear_matrices_y_variables_simbólicas(Yrr, nodos, nodo_slack):
    import sympy as sp

    # Nodos no slack
    nodos_no_slack = [n for n in nodos if n != nodo_slack]
    n = len(nodos_no_slack)
    
    # Variables simbólicas para voltajes nodales no slack
    variables = [sp.symbols(f'V{n}') for n in nodos_no_slack]
    vector_voltajes = sp.Matrix(variables)

    # Admitancias simbólicas de carga por nodo no slack
    Y_cargas = {n: sp.symbols(f'Yc_{n}') for n in nodos_no_slack}

    # Admitancias simbólicas de ramas entre nodos (solo i<j)
    Y_ramas = {}
    for i in nodos_no_slack:
        for j in nodos_no_slack:
            if i < j:
                Y_ramas[(i, j)] = sp.symbols(f'Yr_{i}{j}')

    # Inicializar matriz simbólica Yrr con ceros
    Yrr_simb_con_cargas = sp.zeros(n)

    # Construcción matriz Yrr simbólica con cargas en diagonal y ramas fuera de diagonal
    for idx_i, i in enumerate(nodos_no_slack):
        for idx_j, j in enumerate(nodos_no_slack):
            if i == j:
                suma_ramas = 0
                for k in nodos_no_slack:
                    if k != i:
                        par = (min(i,k), max(i,k))
                        if par in Y_ramas:
                            suma_ramas += Y_ramas[par]
                Yrr_simb_con_cargas[idx_i, idx_j] = suma_ramas + Y_cargas[i]
            else:
                par = (min(i,j), max(i,j))
                if par in Y_ramas:
                    Yrr_simb_con_cargas[idx_i, idx_j] = -Y_ramas[par]
                else:
                    Yrr_simb_con_cargas[idx_i, idx_j] = 0

    return Yrr_simb_con_cargas, vector_voltajes


# Función #6.1.2: resolver_sistema_simbólico
# Recibe:
# - Yrr_sym: matriz simbólica de admitancia entre nodos no slack
# - Y_r0_sym: vector simbólico de admitancia entre nodos no slack y nodo slack
# - V0: voltaje complejo del nodo slack (constante)
# Devuelve:
# - V_r_simbolico: expresión simbólica de voltajes nodales no slack
def resolver_sistema_simbólico(Yrr_sym, Y_r0_sym, V0):
    V_r_simbolico = Yrr_sym.inv() * (-Y_r0_sym * V0)
    return V_r_simbolico


# Función #6.1.3: safe_print
# Función para impresión segura en consola con posibles problemas de codificación
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'encoding'}
        print(*args, **filtered_kwargs)


# Función #6.1.4: imprimir_resultados_simbólicos
# Recibe:
# - Yrr_simb_con_cargas: matriz simbólica de admitancia con cargas y ramas
# - vector_voltajes: vector simbólico de variables voltajes nodales no slack
def imprimir_resultados_simbólicos(Yrr_simb_con_cargas, vector_voltajes):
    import sympy as sp
    import sys

    safe_print("🔷 Matriz simbólica de admitancias Yrr con cargas en diagonal:")
    if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('utf'):
        sp.pprint(Yrr_simb_con_cargas, use_unicode=True)
    else:
        sp.pprint(Yrr_simb_con_cargas, use_unicode=False)

    safe_print("\n🔷 Vector simbólico de voltajes incógnitos V_r:")
    if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('utf'):
        sp.pprint(vector_voltajes, use_unicode=True)
    else:
        sp.pprint(vector_voltajes, use_unicode=False)


# Función principal #6.1.5: construir_y_resolver_simbólico
# Orquesta la creación de matrices, solución y impresión
# Recibe:
# - Yrr: matriz numpy de admitancia entre nodos no slack
# - Y_r0: vector numpy de admitancia entre nodos no slack y nodo slack
# - V0: voltaje complejo del nodo slack (constante)
# - nodos: lista completa de nodos
# - nodo_slack: nodo slack (referencia)
# Devuelve:
# - V_r_simbolico: expresión simbólica de voltajes nodales no slack
# - Yrr_simb_con_cargas: matriz simbólica de admitancia con cargas y ramas
# - vector_voltajes: vector simbólico de variables voltajes nodales no slack
def construir_y_resolver_simbólico(Yrr, Y_r0, V0, nodos, nodo_slack):
    

    Yrr_simb_con_cargas, vector_voltajes = crear_matrices_y_variables_simbólicas(Yrr, nodos, nodo_slack)

    Yrr_sym = sp.Matrix(Yrr_simb_con_cargas)
    Y_r0_sym = sp.Matrix(Y_r0)

    V_r_simbolico = resolver_sistema_simbólico(Yrr_sym, Y_r0_sym, V0)

    imprimir_resultados_simbólicos(Yrr_simb_con_cargas, vector_voltajes)

    return V_r_simbolico, Yrr_simb_con_cargas, vector_voltajes

"""








# ====================== Sección #7: Funciones para PDF ======================

# ---------- Subsección #7.1: Encabezado y Fondo del PDF ----------

# Función #7.1.1: encabezado
# Dibuja el encabezado con texto fijo en el canvas del PDF.
# Recibe: canvas, doc (documento PDF)
# Devuelve: nada (dibuja sobre canvas)
def encabezado(canvas, doc):
    canvas.saveState()
    ancho_pagina = doc.pagesize[0]
    y = doc.height + doc.topMargin - 20

    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawCentredString(ancho_pagina / 2, y, "Empresa Nacional de Energía Eléctrica, ENEE")

    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawCentredString(ancho_pagina / 2, y - 18, "Análisis de Red Secundaria de Distribución")

    canvas.restoreState()

# Función #7.1.2: aplicar_fondo
# Aplica una imagen de fondo al PDF.
# Recibe: canvas, doc, ruta_imagen_fondo (string)
# Devuelve: nada (dibuja sobre canvas)
def aplicar_fondo(cnv, doc, ruta_imagen_fondo):
    try:
        ancho, alto = doc.pagesize
        imagen_fondo = ImageReader(ruta_imagen_fondo)
        cnv.drawImage(imagen_fondo, 0, 0, width=ancho, height=alto)
    except Exception as e:
        print(f"Error al aplicar fondo: {e}")

# Función #7.1.3: fondo_y_encabezado
# Combina la función de aplicar fondo y encabezado para el PDF.
# Recibe: canvas, doc
# Devuelve: nada (dibuja sobre canvas)
def fondo_y_encabezado(cnv, doc):
    ruta_imagen_fondo = r"C:\Users\José Nikol Cruz\Desktop\José Nikol Cruz\Python Programas\Imagen Encabezado.jpg"
    aplicar_fondo(cnv, doc, ruta_imagen_fondo)
    encabezado(cnv, doc)


# ---------- Subsección #7.2: Estilos y Elementos Básicos ----------

# Función #7.2.1: obtener_estilos
# Crea y devuelve una hoja de estilos con estilos personalizados para el PDF.
# Recibe: ninguno
# Devuelve: styles (stylesheet)
def obtener_estilos():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ComentarioAlerta',
        fontSize=16,
        leading=20,
        textColor=colors.red,
        alignment=TA_CENTER,
        underline=True,
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    ))
    return styles




# ---------- Subsección #7.3: Creación de textos ----------

# Función #7.3.1: crear_titulo
# Crea un objeto Paragraph para un título centrado con tamaño y espacios personalizables.
# Recibe: texto (string), tamano (int, default=16), espacio_arriba (int, default=24), espacio_abajo (int, default=24)
# Devuelve: Paragraph (elemento para PDF)
def crear_titulo(texto, tamano=16, espacio_arriba=24, espacio_abajo=24):
    estilo = ParagraphStyle(
        name='Titulo', fontSize=tamano, leading=tamano+4, alignment=TA_CENTER,
        spaceBefore=espacio_arriba, spaceAfter=espacio_abajo, fontName='Helvetica-Bold')
    return Paragraph(texto, estilo)

# Función #7.3.2: crear_subtitulo
# Crea un objeto Paragraph para un subtítulo alineado a la izquierda con tamaño y espacios personalizables.
# Recibe: texto (string), tamano (int, default=16), espacio_arriba (int, default=12), espacio_abajo (int, default=12)
# Devuelve: Paragraph (elemento para PDF)
def crear_subtitulo(texto, tamano=16, espacio_arriba=12, espacio_abajo=12):
    estilo = ParagraphStyle(
        name='Subtitulo', fontSize=tamano, leading=tamano+4, alignment=TA_LEFT,
        spaceBefore=espacio_arriba, spaceAfter=espacio_abajo, fontName='Helvetica-Bold')
    return Paragraph(texto, estilo)


# ---------- Subsección #7.4: Tablas y gráficos ----------

# Función #7.4.1: crear_tabla
# Crea una tabla para PDF a partir de un DataFrame, con opciones de encabezados y alineación.
# Recibe: df (DataFrame), encabezados (lista, opcional), alineacion (string, default 'LEFT')
# Devuelve: objeto Table para PDF
def crear_tabla(df, encabezados=None, alineacion='LEFT'):
    if encabezados is None:
        data = [df.columns.to_list()] + df.values.tolist()
    else:
        data = [encabezados] + df.values.tolist()
    tabla = Table(data, hAlign=alineacion)
    estilo = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#d3d3d3")),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), alineacion),
    ]
    tabla.setStyle(estilo)
    return tabla

# Función #7.4.2: generador_de_graficos
# Genera un gráfico de línea y devuelve el buffer de imagen PNG para usar en PDF.
# Recibe: x (lista o array), y (lista o array), titulo (string), xlabel (string), ylabel (string)
# Devuelve: BytesIO con imagen PNG
def generador_de_graficos(x, y, titulo, xlabel, ylabel):
    plt.figure(figsize=(6, 3))
    plt.plot(x, y, marker='o')
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(ticks=x, labels=[str(int(val)) for val in x])
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf




# ====================== Sección #8: Funciones para análisis y preparación de datos ======================

# ---------- Subsección #8.1: Crear sección de pérdidas y corrientes en PDF ----------

# Función #8.1.1: crear_seccion_perdidas_y_corrientes
# Genera una lista de elementos PDF con tabla y resumen de pérdidas y corrientes por vano.
# Recibe: 
#   df_resultado (DataFrame con columnas: nodo_inicial, nodo_final, I_vano, P_perdida),
#   perdida_total (float),
#   proyeccion (valor opcional para incluir proyección, no usado aquí)
# Devuelve: lista de elementos para PDF (Paragraphs, Tables, Spacer)
def crear_seccion_perdidas_y_corrientes(df_resultado, perdida_total, proyeccion, voltaje_slack=240):
    elementos = []
    elementos.append(crear_subtitulo("Corrientes Vano"))
    
    df_tabla = df_resultado[['nodo_inicial', 'nodo_final', 'I_vano']].copy()
    df_tabla['Tramo'] = df_tabla['nodo_inicial'].astype(str) + '-' + df_tabla['nodo_final'].astype(str)
    df_tabla['|I| (A)'] = df_tabla['I_vano'].apply(lambda x: round(abs(x), 1))
    df_tabla = df_tabla.rename(columns={'nodo_inicial': 'Nodo Inicial', 'nodo_final': 'Nodo Final'})
    df_tabla = df_tabla[['Nodo Inicial', 'Nodo Final', 'Tramo', '|I| (A)']]
    
    tabla = crear_tabla(df_tabla, encabezados=df_tabla.columns.tolist())
    elementos.append(tabla)
    elementos.append(Spacer(1, 12))
    
    if 'kva' in df_resultado.columns:
        kva_total = df_resultado['kva'].sum()
        corriente_transfo = (kva_total * 1000) / voltaje_slack
        comentario = f"La corriente del transformador es de {corriente_transfo:.2f} A"
        elementos.append(Paragraph(comentario))  # estilo por defecto
    
    return elementos






# ---------- Subsección #8.2: Preparar DataFrame de voltajes para reporte ----------

# Función #8.2.1: preparar_df_voltajes
# Combina información de voltajes y regulación, formatea columnas para reporte.
# Recibe:
#   df_voltajes (DataFrame con columnas: Nodo, Magnitud (V), Ángulo (°))
#   df_regulacion (DataFrame con columnas: Nodo, Regulación (%))
# Devuelve:
#   df_final con columnas: Nodo, Voltaje (p.u.), Voltaje Absoluto (V), Regulación (%)
def preparar_df_voltajes(df_voltajes, df_regulacion):
    
    print(df_regulacion.columns)
    print(df_regulacion.head())
    
    # Unir los dos DataFrames por la columna 'Nodo'
    df_temp = pd.merge(df_voltajes, df_regulacion[['Nodo', 'Regulación (%)']], on='Nodo', how='left')
    
    # Calcular columna de voltaje en p.u. en forma polar
    df_temp['Voltaje (p.u.)'] = df_temp.apply(
        lambda r: f"{r['Magnitud (V)']/240:.3f} ∠ {r['Ángulo (°)']:.1f}°", axis=1
    )

    # Formato de voltaje absoluto
    df_temp['Voltaje Absoluto (V)'] = df_temp['Magnitud (V)'].map(lambda x: f"{x:.2f}")

    # Formato de regulación
    df_temp['Regulación (%)'] = df_temp['Regulación (%)'].map(lambda x: f"{x:.2f}")

    # Seleccionar columnas deseadas
    df_final = df_temp[['Nodo', 'Voltaje (p.u.)', 'Voltaje Absoluto (V)', 'Regulación (%)']]

    return df_final


# ====================== Sección #9: Contenido de Informe de PDF ======================

# Subsección #9.1: Preparación de información para reporte

# Función #9.1.1: preparar_info_proyecto
# Ajusta nombres y agrega factor de coincidencia a los DataFrames de info y parámetros del proyecto.
# Recibe:
# - df_info: DataFrame con información del proyecto
# - df_parametros: DataFrame con parámetros generales del sistema
# - factor_coinc: float con factor de coincidencia
# Devuelve:
# - df_info_copy: DataFrame modificado con nombres legibles
# - df_parametros_copy: DataFrame modificado con nombres legibles y factor agregado
def preparar_info_proyecto(df_info, df_parametros, factor_coinc, potencia_total_kva):
    df_info_copy = df_info.copy()
    df_info_copy.loc[df_info_copy['info'] == 'NumeroProyecto', 'info'] = 'Código de Proyecto'
    df_info_copy.loc[df_info_copy['info'] == 'NombreProyecto', 'info'] = 'Nombre del Proyecto'
    df_info_copy.loc[df_info_copy['info'] == 'NumeroTransformador', 'info'] = 'Número del Transformador del Proyecto'

    df_parametros_copy = df_parametros.copy()
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'TipoConductor', 'Parámetro'] = 'Calibre del Conductor'
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'area_lote', 'Parámetro'] = 'Tamaño del Lote Típico m2'
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'Regulacion_Primaria', 'Parámetro'] = 'Regulación en la Red Primaria, %'
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'Regulacion_Transformador', 'Parámetro'] = 'Regulación Nominal en el Transformador, %'
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'capacidad_transformador', 'Parámetro'] = 'Capacidad Nominal del Transformador en KVA'
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'Voltaje_Nominal_Primario', 'Parámetro'] = 'Voltaje Nominal de Red Primaria en V'
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'Voltaje_Nominal_Secundario', 'Parámetro'] = 'Voltaje Nominal de Red Secundaria en V'
    
    nuevas_filas = pd.DataFrame([
        {'Parámetro': 'Factor de Coincidencia', 'Valor': f"{factor_coinc:.2f}"},
        {'Parámetro': 'Carga Inicial KVA', 'Valor': f"{potencia_total_kva/0.9:.2f}"}
    ])
    df_parametros_copy = pd.concat([df_parametros_copy, nuevas_filas], ignore_index=True)

    return df_info_copy, df_parametros_copy


# Función #9.1.2: seccion_info_proyecto
# Crea los elementos de contenido para la sección de información y parámetros del proyecto (para reporte PDF).
# Recibe:
# - df_info_copy: DataFrame con información del proyecto
# - df_parametros_copy: DataFrame con parámetros del proyecto
# Devuelve:
# - lista de elementos (Paragraphs, Tables, Spacers) para reporte
def seccion_info_proyecto(df_info_copy, df_parametros_copy, potencia_total_kva):
    elementos = []
    elementos.append(Spacer(1, 6)) 
    elementos.append(crear_titulo("Gerencia de Distribución, Unidad de Proyectos"))
    elementos.append(Spacer(1, 12))

    elementos.append(crear_subtitulo("1. Información del Proyecto"))
    tabla_info = crear_tabla(df_info_copy, encabezados=['Parámetro', 'Valor'])
    elementos.append(tabla_info)
    elementos.append(Spacer(1, 12))

    elementos.append(crear_subtitulo("2. Parámetros Generales del Sistema"))
    tabla_param = crear_tabla(df_parametros_copy, encabezados=['Parámetro', 'Valor'])
    elementos.append(tabla_param)
    elementos.append(Spacer(1, 12))

    return elementos


# Función #9.1.3: crear_tabla_usuarios_conectados
# Prepara tabla con nodos, distancia y usuarios conectados.
# Recibe:
# - df_conexiones: DataFrame con información de conexiones y usuarios
# Devuelve:
# - tabla con columnas renombradas para reporte
def crear_tabla_usuarios_conectados(df_conexiones):
    df_clean = df_conexiones.copy()
    for col in ['usuarios', 'distancia', 'nodo_inicial', 'nodo_final']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        else:
            df_clean[col] = float('nan')

    # Aquí asumo que las columnas P y kva ya están calculadas antes
    columnas_tabla = ["nodo_final", "distancia", "usuarios", "P", "kva"]
    df_tabla = df_clean[columnas_tabla].copy()

    # Renombrar columnas
    df_tabla.columns = ["Nodo", "Distancia (m)", "Usuarios", "P (kW)", "S (kVA)"]

    # Convertir Nodo y Usuarios a enteros
    df_tabla['Nodo'] = df_tabla['Nodo'].astype('Int64')
    df_tabla['Usuarios'] = df_tabla['Usuarios'].astype('Int64')

    # Redondear las demás columnas a 1 decimal
    for col in ["Distancia (m)", "P (kW)", "S (kVA)"]:
        df_tabla[col] = df_tabla[col].round(1)
        

    return crear_tabla(df_tabla, encabezados=df_tabla.columns.tolist(), alineacion='CENTER')


# Función #9.1.4: crear_tabla_voltajes
# Crea sección de análisis de voltajes nodales con tabla formateada.
# Recibe:
# - df_voltajes: DataFrame con voltajes nodales
# - df_regulacion: DataFrame con regulaciones porcentuales
# Devuelve:
# - lista de elementos para reporte (subtitulo, tabla, espacio)
def crear_tabla_voltajes(df_voltajes, df_regulacion):
    elementos = []
    elementos.append(crear_subtitulo("5. Análisis de Voltajes Nodales"))
    
    df_tabla = preparar_df_voltajes(df_voltajes, df_regulacion)
    tabla = crear_tabla(df_tabla, alineacion='CENTER')
    elementos.append(tabla)
    elementos.append(Spacer(1, 12))
    
    return elementos


# Función #9.1.5: crear_comentario_regulacion
# Genera comentario con color según si la regulación de voltaje está dentro del rango ±5%.
# Recibe:
# - df_voltajes: DataFrame con magnitud de voltajes
# Devuelve:
# - Paragraph con comentario formateado para reporte
def crear_comentario_regulacion(df_voltajes):
    voltajes = df_voltajes['Magnitud (V)'].astype(float) / 240
    min_v = voltajes.min()
    max_v = voltajes.max()

    if min_v >= 0.95 and max_v <= 1.05:
        comentario = f"✅ La regulación de voltaje es aceptable (±5%). Voltajes entre {min_v:.3f} y {max_v:.3f} p.u."
        color_comentario = colors.green
    else:
        comentario = f"⚠️ La regulación de voltaje está fuera del rango aceptable (±5%). Voltajes entre {min_v:.3f} y {max_v:.3f} p.u."
        color_comentario = colors.red

    parrafo = Paragraph(comentario, ParagraphStyle(
        name='ComentarioRegulacion',
        fontSize=12,
        leading=14,
        alignment=1,
        textColor=color_comentario,
        spaceBefore=12,
        spaceAfter=12,
        fontName='Helvetica-Bold'
    ))
    
    return parrafo


# Función #9.1.6: crear_grafico_voltajes_pdf
# Genera gráfico de voltajes nodales como imagen para insertar en reporte.
# Recibe:
# - df_voltajes: DataFrame con nodos y magnitudes
# Devuelve:
# - lista con Image y Spacer para reporte
def crear_grafico_voltajes_pdf(df_voltajes):
    buf_grafico = generador_de_graficos(
        df_voltajes['Nodo'], df_voltajes['Magnitud (V)'],
        "Análisis Nodal de Voltajes", "Nodo", "Magnitud (V)"
    )
    imagen = Image(buf_grafico, width=5 * inch, height=3 * inch)
    imagen.hAlign = 'CENTER'
    return [imagen, Spacer(1, 12)]


# Función #9.1.7: seccion_usuarios_y_voltajes
# Genera sección que incluye tabla de usuarios, tabla de voltajes, comentario y gráfico.
# Recibe:
# - df_conexiones: DataFrame de conexiones/usuarios
# - df_voltajes: DataFrame de voltajes
# - df_regulacion: DataFrame de regulaciones
# Devuelve:
# - lista de elementos para reporte
def seccion_usuarios_y_voltajes(df_conexiones, df_voltajes, df_regulacion):
    elementos = []
    
    elementos.append(crear_tabla_usuarios_conectados(df_conexiones))
    elementos.append(Spacer(1, 12))

    elementos.extend(crear_tabla_voltajes(df_voltajes, df_regulacion))
    elementos.append(crear_comentario_regulacion(df_voltajes))
    elementos.append(Spacer(1, 12))

    elementos.extend(crear_grafico_voltajes_pdf(df_voltajes))

    return elementos


# Función #9.1.8: crear_grafico_demanda
# Genera gráfico de proyección de demanda en un periodo de años.
# Recibe:
# - proyeccion_demanda: lista o array de demandas anuales
# Devuelve:
# - buffer de imagen PNG del gráfico
def crear_grafico_demanda(proyeccion_demanda):
    años = list(range(1, len(proyeccion_demanda) + 1))
    
    plt.figure(figsize=(6,4))
    plt.plot(años, proyeccion_demanda, marker='o', linestyle='-', color='b')
    plt.title('Proyección de Demanda a 15 años')
    plt.xlabel('Años')
    plt.ylabel('Demanda (kVA)')
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG')
    plt.close()
    buf.seek(0)
    return buf


# Función #9.1.9: preparar_tabla_proyeccion
# Prepara DataFrame para mostrar proyección de demanda y pérdidas con formato adecuado.
# Recibe:
# - df_proyeccion: DataFrame con datos de demanda
# - proyeccion_perdidas: lista o array de pérdidas anuales
# Devuelve:
# - DataFrame formateado con columnas: Año, Demanda, Pérdidas, % Carga
def preparar_tabla_proyeccion(df_proyeccion, proyeccion_perdidas):
    if 'Pérdidas (kWh)' not in df_proyeccion.columns:
        df_proyeccion = df_proyeccion.copy()
        df_proyeccion['Pérdidas (kWh)'] = proyeccion_perdidas

    df_mostrar = df_proyeccion.copy()

    df_mostrar['% Carga (%)'] = df_mostrar['% Carga (%)'].astype(str).str.strip().str.rstrip('%')
    df_mostrar['% Carga (%)'] = pd.to_numeric(df_mostrar['% Carga (%)'], errors='coerce').map(lambda x: f"{x:.2f}%")

    df_mostrar['Demanda (kVA)'] = df_mostrar['Demanda_kva'].astype(float).map(lambda x: f"{x:,.2f}")
    df_mostrar['Pérdidas (kWh)'] = df_mostrar['Pérdidas (kWh)'].astype(float).map(lambda x: f"{x:,.2f}")

    columnas_a_mostrar = ['Año', 'Demanda (kVA)', 'Pérdidas (kWh)', '% Carga (%)']
    return df_mostrar[columnas_a_mostrar]


# Función #9.1.10: crear_elementos_tabla
# Crea elementos para mostrar tabla de proyección con subtitulo y espacio.
# Recibe:
# - df_tabla: DataFrame formateado para mostrar
# Devuelve:
# - lista de elementos para reporte
def crear_elementos_tabla(df_tabla):
    elementos = []
    elementos.append(crear_subtitulo("6. Proyección de Demanda y Pérdidas Eléctricas Anuales"))
    tabla = crear_tabla(df_tabla, encabezados=df_tabla.columns.tolist())
    elementos.append(tabla)
    elementos.append(Spacer(1, 12))
    return elementos


# Función #9.1.11: crear_comentario_cargabilidad
# Crea comentario sobre cargabilidad del equipo basado en % carga y voltajes.
# Recibe:
# - df_proyeccion: DataFrame con % carga
# - df_voltajes: DataFrame con voltajes nodales
# Devuelve:
# - Paragraph con comentario formateado para reporte
def crear_comentario_cargabilidad(df_proyeccion, df_voltajes):
    ultimo_pct_carga = df_proyeccion['% Carga (%)'].str.strip().str.rstrip('%').astype(float).iloc[-1]

    if ultimo_pct_carga < 100:
        comentario = "✅ La cargabilidad del equipo está dentro del valor nominal (menor al 100%)."
        color_comentario = colors.green
    elif 100 <= ultimo_pct_carga <= 110:
        comentario = "⚠️ La cargabilidad del equipo supera el valor nominal pero está dentro del límite térmico aceptado (hasta 110%)."
        color_comentario = colors.yellow
    else:
        comentario = "❌ La cargabilidad del equipo sobrepasa el límite térmico aceptado (mayor al 110%)."
        color_comentario = colors.red

    comentario_parrafo = Paragraph(comentario, ParagraphStyle(
        name='ComentarioCarga',
        fontSize=12,
        leading=14,
        alignment=1,
        textColor=color_comentario,
        spaceBefore=12,
        spaceAfter=12,
        fontName='Helvetica-Bold'
    ))
    return comentario_parrafo


# Función #9.1.12: seccion_proyeccion_perdidas_y_demanda
# Genera sección completa de proyección de demanda, pérdidas, gráfico y comentario de cargabilidad.
# Recibe:
# - capacidad_transformador_kva: float (no usado directamente, pero para futuras referencias)
# - df_proyeccion: DataFrame con datos de proyección de demanda
# - proyeccion_perdidas: lista o array con pérdidas anuales
# - df_voltajes: DataFrame con voltajes nodales
# Devuelve:
# - lista de elementos para reporte
def seccion_proyeccion_perdidas_y_demanda(capacidad_transformador_kva, df_proyeccion, proyeccion_perdidas, df_voltajes):
    df_tabla = preparar_tabla_proyeccion(df_proyeccion, proyeccion_perdidas)
    elementos = crear_elementos_tabla(df_tabla)

    buf_grafico_demanda = crear_grafico_demanda(df_proyeccion['Demanda_kva'].astype(float))
    imagen_demanda = Image(buf_grafico_demanda, width=5*inch, height=3*inch)
    imagen_demanda.hAlign = 'CENTER'
    elementos.append(imagen_demanda)
    elementos.append(Spacer(1, 12))

    comentario = crear_comentario_cargabilidad(df_proyeccion, df_voltajes)
    elementos.append(comentario)
    elementos.append(Spacer(1, 12))

    return elementos




# ====================== Sección #10: Visualización de la Red y Diagramas ======================



# Función #10.1: crear_grafo
# Crea un grafo NetworkX con nodos y aristas, donde cada arista contiene atributos de usuarios y distancia.
# Recibe:
# - nodos_inicio: lista o iterable con nodos iniciales de cada arista
# - nodos_final: lista o iterable con nodos finales de cada arista
# - usuarios: lista o iterable con cantidad de usuarios por arista
# - distancias: lista o iterable con distancias por arista
# Devuelve:
# - grafo G de NetworkX con atributos añadidos
def crear_grafo(nodos_inicio, nodos_final, usuarios, distancias):
    G = nx.Graph()
    for ni, nf, u, d in zip(nodos_inicio, nodos_final, usuarios, distancias):
        G.add_edge(ni, nf, usuarios=u, distancia=d)
    return G


# Función #10.2: calcular_posiciones_red
# Calcula posiciones para los nodos en el grafo usando método recursivo basado en distancias y nivel vertical (y).
# Recibe:
# - G: grafo NetworkX
# - nodo_raiz: nodo desde donde se inicia la posición (default=1)
# - escala: factor para escala horizontal de distancias
# - dy: desplazamiento vertical entre nodos hermanos
# Devuelve:
# - diccionario posiciones {nodo: (x, y)}
def calcular_posiciones_red(G, nodo_raiz=1, escala=0.05, dy=1.5):
    posiciones = {}
    usados = set()

    def asignar_posiciones(nodo, x, y):
        if nodo in usados:
            return
        usados.add(nodo)
        posiciones[nodo] = (x, y)

        vecinos = [v for v in G.neighbors(nodo) if v not in usados]
        vecinos.sort()

        for i, vecino in enumerate(vecinos):
            distancia = G[nodo][vecino].get('distancia', 0)
            dx = distancia * escala
            nuevo_x = x + dx
            nuevo_y = y - dy * (i - (len(vecinos) - 1) / 2)

            asignar_posiciones(vecino, nuevo_x, nuevo_y)

    asignar_posiciones(nodo_raiz, 0, 0)
    return posiciones


# Función #10.3: dibujar_nodos_transformador
# Dibuja el nodo transformador con forma triangular y etiqueta con capacidad en kVA.
# Recibe:
# - ax: objeto Axes de matplotlib
# - G: grafo NetworkX
# - posiciones: diccionario de posiciones nodales
# - capacidad_transformador: valor numérico en kVA para etiqueta
def dibujar_nodos_transformador(ax, G, posiciones, capacidad_transformador):
    tamaño_transformador = 400
    nx.draw_networkx_nodes(G, posiciones,
                           nodelist=[1],
                           node_shape='^',
                           node_color='orange',
                           node_size=tamaño_transformador,
                           label='Transformador (Nodo 1)')

    x, y = posiciones[1]
    etiqueta = f'Transformador\n{capacidad_transformador} kVA'
    ax.text(x - 1, y, etiqueta, fontsize=9, ha='center', color='black')


# Función #10.4: dibujar_nodos_generales
# Dibuja los demás nodos con forma circular y color celeste.
# Recibe:
# - ax: objeto Axes de matplotlib
# - G: grafo NetworkX
# - posiciones: diccionario de posiciones nodales
def dibujar_nodos_generales(ax, G, posiciones):
    tamaño_nodos = 200
    otros_nodos = [n for n in G.nodes if n != 1]
    nx.draw_networkx_nodes(G, posiciones,
                           nodelist=otros_nodos,
                           node_shape='o',
                           node_color='lightblue',
                           node_size=tamaño_nodos)


# Función #10.5: dibujar_aristas
# Dibuja las aristas normales y marca bucles con círculos y etiquetas especiales.
# Recibe:
# - ax: objeto Axes de matplotlib
# - G: grafo NetworkX
# - posiciones: diccionario de posiciones nodales
def dibujar_aristas(ax, G, posiciones):
    aristas_normales = [(u, v) for u, v in G.edges() if u != v]
    nx.draw_networkx_edges(G, posiciones, edgelist=aristas_normales, width=2)

    # Bucles (autoenlaces), excepto en nodo 1
    bucles = [(u, v) for u, v in G.edges() if u == v and u != 1]
    for nodo, _ in bucles:
        x, y = posiciones[nodo]
        circle = plt.Circle((x, y), 0.1, color='red', fill=False, linestyle='--', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y + 0.12, 'Bucle', fontsize=8, color='red', ha='center')


# Función #10.6: dibujar_etiquetas_nodos
# Dibuja las etiquetas con número de nodo en las posiciones correspondientes.
# Recibe:
# - ax: objeto Axes de matplotlib
# - G: grafo NetworkX
# - posiciones: diccionario de posiciones nodales
def dibujar_etiquetas_nodos(ax, G, posiciones):
    etiquetas_nodos = {n: str(n) for n in G.nodes}
    nx.draw_networkx_labels(G, posiciones, etiquetas_nodos, font_size=12, font_weight='bold')


# Función #10.7: dibujar_acometidas
# Dibuja líneas punteadas hacia abajo y etiquetas con cantidad de usuarios conectados a cada nodo final.
# Recibe:
# - ax: objeto Axes de matplotlib
# - posiciones: diccionario de posiciones nodales
# - nodos_final: lista nodos finales de cada tramo
# - usuarios: lista número de usuarios conectados por tramo
def dibujar_acometidas(ax, posiciones, nodos_final, usuarios):
    usuarios_por_nodo = {}
    for nf, u in zip(nodos_final, usuarios):
        usuarios_por_nodo[nf] = usuarios_por_nodo.get(nf, 0) + u

    for n, usuarios_nodo in usuarios_por_nodo.items():
        if n in posiciones:
            x, y = posiciones[n]
            x_u, y_u = x, y - 0.2
            ax.plot([x, x_u], [y, y_u], color='gray', linestyle='--', linewidth=1)
            ax.text(x_u, y_u - 0.03, f"{usuarios_nodo} A", fontsize=9, color='blue',
                    ha='center', va='top')


# Función #10.8: dibujar_distancias_tramos
# Dibuja etiquetas con la distancia en metros entre nodos en el centro de cada tramo.
# Recibe:
# - ax: objeto Axes de matplotlib
# - G: grafo NetworkX
# - posiciones: diccionario de posiciones nodales
def dibujar_distancias_tramos(ax, G, posiciones):
    for (u, v, d) in G.edges(data=True):
        if u != v:
            x1, y1 = posiciones[u]
            x2, y2 = posiciones[v]
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2

            dx = x2 - x1
            dy = y2 - y1
            dist = (dx**2 + dy**2)**0.5
            offset_x = -dy / dist * 0.1
            offset_y = dx / dist * 0.1

            ax.text(xm + offset_x, ym + offset_y, f"{d['distancia']} m",
                    color='red', fontsize=9, ha='center', va='center')


# Función #10.9: crear_grafico_nodos
# Genera el gráfico completo de la red con nodos, aristas, etiquetas, acometidas y distancias.
# Recibe:
# - nodos_inicio: lista nodos iniciales
# - nodos_final: lista nodos finales
# - usuarios: lista número de usuarios por tramo
# - distancias: lista distancias por tramo
# - capacidad_transformador: valor numérico en kVA para nodo transformador
# Devuelve:
# - buffer de imagen PNG con gráfico generado
def crear_grafico_nodos(nodos_inicio, nodos_final, usuarios, distancias, capacidad_transformador):
    G = crear_grafo(nodos_inicio, nodos_final, usuarios, distancias)
    posiciones = calcular_posiciones_red(G, nodo_raiz=1, escala=0.05)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    dibujar_nodos_transformador(ax, G, posiciones, capacidad_transformador)
    dibujar_nodos_generales(ax, G, posiciones)
    dibujar_aristas(ax, G, posiciones)
    dibujar_etiquetas_nodos(ax, G, posiciones)
    dibujar_acometidas(ax, posiciones, nodos_final, usuarios)
    dibujar_distancias_tramos(ax, G, posiciones)

    plt.title("Diagrama de Nodos del Transformador")
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf



# ---------- Generar PDF ----------


# ====================== Sección #11: Generación de Informe PDF ======================


# Función #11.1: crear_pdf
# Genera un archivo PDF con las secciones proporcionadas y un gráfico de la red.
# Recibe:
# - nombre_archivo: nombre del archivo PDF a generar
# - secciones: lista de listas con elementos para agregar al documento (Paragraphs, Tables, etc.)
# - nodos_inicio, nodos_final, usuarios, distancias: datos para generar el gráfico de la red
# - capacidad_transformador: valor numérico para la etiqueta del transformador en el gráfico
def crear_pdf(nombre_archivo, secciones, nodos_inicio, nodos_final, usuarios, distancias, capacidad_transformador):
    doc = SimpleDocTemplate(nombre_archivo, pagesize=letter)

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
    template = PageTemplate(id='fondo_total', frames=[frame], onPage=fondo_y_encabezado)
    doc.addPageTemplates([template])

    elementos = []

    for seccion in secciones:
        elementos.extend(seccion)

    # Generar imagen del gráfico de nodos y convertirla a objeto Image para PDF
    buf = crear_grafico_nodos(nodos_inicio, nodos_final, usuarios, distancias, capacidad_transformador)
    imagen_pdf = Image(buf, width=6*inch, height=4.5*inch)
    imagen_pdf.hAlign = 'CENTER'

    elementos.append(imagen_pdf)
    elementos.append(Spacer(1, 12))

    doc.build(elementos)


# Función #11.2: cargar_y_preparar_datos
# Carga datos desde archivo, calcula parámetros eléctricos y prepara DataFrame para análisis.
# Recibe:
# - archivo: ruta o nombre del archivo de datos
# Devuelve:
# - dataframes y parámetros necesarios para cálculos y gráficos posteriores
def cargar_y_preparar_datos(archivo):
    df_conexiones, df_parametros, df_info, tipo_conductor, area_lote, capacidad_transformador, proyecto_numero, proyecto_nombre, transformador_numero, usuarios, distancias, nodos_inicio, nodos_final = cargar_datos_circuito(archivo)
    
    conductores = bibloteca_conductores()
    separacion_fases_m = 0.2032
    radio_conductor_m = 0.00735

    df_conexiones['resistencia_vano'] = df_conexiones.apply(
        lambda row: resistencia_por_vano(conductores, tipo_conductor, row['distancia']), axis=1
    )
    df_conexiones['reactancia_vano'] = df_conexiones.apply(
        lambda row: reactancia_por_vano_geometrica(row['distancia'], separacion_fases_m, radio_conductor_m), axis=1
    )
    df_conexiones['Z_vano'] = df_conexiones.apply(calcular_impedancia, axis=1)
    df_conexiones['Y_vano'] = df_conexiones['Z_vano'].apply(calcular_admitancia)

    return df_conexiones, df_parametros, df_info, tipo_conductor, area_lote, capacidad_transformador, usuarios, distancias, nodos_inicio, nodos_final


# Función #11.3: calcular_flujo_carga_y_perdidas
# Realiza cálculos eléctricos para flujo de carga, voltajes nodales y pérdidas.
# Recibe:
# - df_conexiones: DataFrame con datos de conexiones y parámetros eléctricos
# - df_parametros: DataFrame con parámetros adicionales
# - area_lote: área del lote para cálculo de cargas
# Devuelve:
# - df_conexiones actualizado, pérdidas totales, proyección de pérdidas, potencia total, matrices admitancia y datos nodales
def calcular_flujo_carga_y_perdidas(df_conexiones, df_parametros, area_lote):
    total_usuarios = df_conexiones['usuarios'].sum()
    factor_coinc = factor_coincidencia(total_usuarios)
    
    df_conexiones, potencia_total_kva, factor_coinc = calcular_potencia_carga(df_conexiones, area_lote)
    
    Y, Yrr, Y_r0, nodos, slack_index = calcular_matriz_admitancia(df_conexiones)
    V, _ = calcular_voltajes_nodales(Yrr, Y_r0, slack_index, nodos)
    nodo_slack = nodos[slack_index]

    """V_r_sim, Yrr_simb, vec_V = construir_y_resolver_simbólico(Yrr, Y_r0, V[slack_index], nodos, nodo_slack)"""

    df_conexiones = calcular_corrientes(df_conexiones, V)
    df_conexiones, perdida_total, proyeccion_perdidas = calcular_perdidas_y_proyeccion(df_conexiones)

    return df_conexiones, perdida_total, proyeccion_perdidas, potencia_total_kva, Yrr, Y_r0, slack_index, nodos, nodo_slack, factor_coinc


# Función #11.4: calcular_regulacion_y_proyeccion
# Calcula proyección de demanda, voltajes y regulación para horizonte temporal definido.
# Recibe:
# - potencia_total_kva: potencia base para proyección
# - df_parametros: DataFrame con parámetros de proyecto
# - matrices Yrr, Y_r0, índice slack, nodos, nodo slack
# Devuelve:
# - DataFrames de proyección, voltajes y regulación
def calcular_regulacion_y_proyeccion(potencia_total_kva, df_parametros, Yrr, Y_r0, slack_index, nodos, nodo_slack):
    df_proyeccion = pd.DataFrame({'Año': range(1, 16)})
    df_proyeccion = proyectar_demanda(potencia_total_kva, df_parametros.set_index('Parámetro'), df_proyeccion, crecimiento=0.02, años=15)

    V, df_voltajes = calcular_voltajes_nodales(Yrr, Y_r0, slack_index, nodos, V0=240+0j)
    df_regulacion = calcular_regulacion_voltaje(V, nodo_slack=nodo_slack, nodos=nodos)

    return df_proyeccion, df_voltajes, df_regulacion


# Función #11.5: preparar_proyeccion_y_regulacion
# Alias que combina funciones para preparar los DataFrames de proyección y regulación.
def preparar_proyeccion_y_regulacion(potencia_total_kva, df_parametros, Yrr, Y_r0, slack_index, nodos, nodo_slack):
    return calcular_regulacion_y_proyeccion(potencia_total_kva, df_parametros, Yrr, Y_r0, slack_index, nodos, nodo_slack)


# Función #11.6: generar_informe_pdf
# Orquesta la generación completa del informe PDF con todas las secciones y gráficos.
# Recibe:
# - df_info, df_parametros, factor_coinc, df_conexiones, df_voltajes, df_regulacion,
#   capacidad_transformador, df_proyeccion, proyeccion_perdidas, perdida_total,
#   nodos_inicio, nodos_final, usuarios, distancias
def generar_informe_pdf(df_info, df_parametros, factor_coinc, potencia_total_kva,
                        df_conexiones, df_voltajes, df_regulacion,
                        capacidad_transformador, df_proyeccion, proyeccion_perdidas,
                        perdida_total, nodos_inicio, nodos_final, usuarios, distancias):

    df_info_copy, df_parametros_copy = preparar_info_proyecto(df_info, df_parametros, factor_coinc, potencia_total_kva)

    secciones = [
        seccion_info_proyecto(df_info_copy, df_parametros_copy, potencia_total_kva),
        seccion_usuarios_y_voltajes(df_conexiones, df_voltajes, df_regulacion),
        seccion_proyeccion_perdidas_y_demanda(capacidad_transformador, df_proyeccion, proyeccion_perdidas, df_voltajes),
        crear_seccion_perdidas_y_corrientes(df_conexiones, perdida_total, proyeccion_perdidas)
    ]

    crear_pdf("informe_red_electrica.pdf", secciones, nodos_inicio, nodos_final, usuarios, distancias, capacidad_transformador)
def obtener_datos_para_pdf_corto(ruta_excel):
    import os
    from tus_funciones import (  # Ajusta estos imports si están en otros archivos
        cargar_y_preparar_datos,
        calcular_flujo_carga_y_perdidas,
        calcular_regulacion_y_proyeccion
    )

    carpeta_excel = os.path.dirname(ruta_excel)
    os.chdir(carpeta_excel)
    archivo = os.path.basename(ruta_excel)
    
    # 1. Cargar datos y preparar variables
    (df_conexiones, df_parametros, df_info, tipo_conductor, area_lote, capacidad_transformador,
     usuarios, distancias, nodos_inicio, nodos_final) = cargar_y_preparar_datos(archivo)   
    
    # 2. Cálculo de flujo de carga y pérdidas
    df_conexiones, perdida_total, proyeccion_perdidas, potencia_total_kva, \
    Yrr, Y_r0, slack_index, nodos, nodo_slack, factor_coinc = calcular_flujo_carga_y_perdidas(df_conexiones, df_parametros, area_lote)
    
    # 3. Voltajes y regulación
    df_proyeccion, df_voltajes, df_regulacion = calcular_regulacion_y_proyeccion(
        potencia_total_kva, df_parametros, Yrr, Y_r0, slack_index, nodos, nodo_slack
    )

    # 4. Retornar datos que necesita el PDF corto
    return (
        df_info, 
        potencia_total_kva, 
        perdida_total, 
        capacidad_transformador,
        nodos_inicio, 
        nodos_final, 
        usuarios, 
        distancias, 
        df_voltajes, 
        df_regulacion
    )


def main(ruta_archivo='datos_circuito.xlsx'):
    archivo = ruta_archivo
    
    # Cargar datos y preparar variables iniciales
    (df_conexiones, df_parametros, df_info, tipo_conductor, area_lote, capacidad_transformador,
     usuarios, distancias, nodos_inicio, nodos_final) = cargar_y_preparar_datos(archivo)   
    
    # Cálculo flujo de carga, corrientes y pérdidas
    df_conexiones, perdida_total, proyeccion_perdidas, potencia_total_kva, Yrr, Y_r0, slack_index, nodos, nodo_slack, factor_coinc = calcular_flujo_carga_y_perdidas(df_conexiones, df_parametros, area_lote)
    
    # Calcular regulación y proyección de demanda
    df_proyeccion, df_voltajes, df_regulacion = calcular_regulacion_y_proyeccion(potencia_total_kva, df_parametros, Yrr, Y_r0, slack_index, nodos, nodo_slack)
    
    # Generar informe PDF con todos los datos calculados
    potencia_total_kva = df_conexiones['P'].sum()  # O tu cálculo específico

    generar_informe_pdf(df_info, df_parametros, factor_coinc, potencia_total_kva,
                    df_conexiones, df_voltajes, df_regulacion,
                    capacidad_transformador, df_proyeccion, proyeccion_perdidas,
                    perdida_total, nodos_inicio, nodos_final, usuarios, distancias)


def main_con_ruta_archivo(ruta_excel):
    carpeta_excel = os.path.dirname(ruta_excel)
    os.chdir(carpeta_excel)
    archivo = os.path.basename(ruta_excel)
    
    # Cargar datos y preparar variables iniciales
    (df_conexiones, df_parametros, df_info, tipo_conductor, area_lote, capacidad_transformador,
     usuarios, distancias, nodos_inicio, nodos_final) = cargar_y_preparar_datos(archivo)   
    
    # Cálculo flujo de carga, corrientes y pérdidas
    df_conexiones, perdida_total, proyeccion_perdidas, potencia_total_kva, Yrr, Y_r0, slack_index, nodos, nodo_slack, factor_coinc = calcular_flujo_carga_y_perdidas(df_conexiones, df_parametros, area_lote)
    
    # Calcular regulación y proyección de demanda
    df_proyeccion, df_voltajes, df_regulacion = calcular_regulacion_y_proyeccion(potencia_total_kva, df_parametros, Yrr, Y_r0, slack_index, nodos, nodo_slack)
    
    # Generar informe PDF con todos los datos calculados
    potencia_total_kva = df_conexiones['P'].sum()  # O tu cálculo específico

    generar_informe_pdf(df_info, df_parametros, factor_coinc, potencia_total_kva,
                    df_conexiones, df_voltajes, df_regulacion,
                    capacidad_transformador, df_proyeccion, proyeccion_perdidas,
                    perdida_total, nodos_inicio, nodos_final, usuarios, distancias)


if __name__ == "__main__":
    main()
    

   


