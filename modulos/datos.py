# -*- coding: utf-8 -*-
"""
datos.py
Gestión de datos básicos para el análisis de red secundaria.

Incluye:
1. Biblioteca de conductores
2. Cálculo de kVA por área y factor de coincidencia
3. Lectura y preparación de datos desde Excel
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


# =====================================================
# Sección #1: Diccionario de conductores y parámetros
# =====================================================

def biblioteca_conductores() -> Dict[str, Dict[str, float]]:
    """
    Devuelve un diccionario con propiedades eléctricas de conductores.
    Valores típicos según tablas de la ENEE.
    """
    return {
        '2':   {'R': 0.1563, 'X': 0.0800,  'radio_m': 0.00735},
        '1/0': {'R': 0.1239, 'X': 0.0780,  'radio_m': 0.00825},
        '2/0': {'R': 0.0983, 'X': 0.0750,  'radio_m': 0.00927},
        '3/0': {'R': 0.374, 'X': 0.0730,  'radio_m': 0.01040},
        '4/0': {'R': 0.0618, 'X': 0.0710,  'radio_m': 0.01168},
        '266': {'R': 0.0590, 'X': 0.0700,  'radio_m': 0.01290},
    }


# =====================================================
# Sección #2: Cálculos básicos de demanda
# =====================================================

def calcular_kva_por_area(area_m2: float) -> float:
    """
    Calcula el kVA asignado según el área del lote en m².
    Reglas basadas en el Manual de Obras de la ENEE.
    """
    area_m2 = float(np.real(area_m2))
    if area_m2 <= 0:
        raise ValueError("Área inválida: debe ser mayor a 0 m²")

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
        return min(5.2 + kva_adicional, 9.0)
    else:
        raise ValueError("Área mayor a 750 m²: requiere servicio exclusivo")


def factor_coincidencia(usuarios: int) -> float:
    """
    Asigna el factor de coincidencia según la cantidad de usuarios (lotes).
    Basado en tabla de la ENEE.
    """
    tabla = [
        (1, 1.00), (2, 0.93), (3, 0.85), (4, 0.77), (5, 0.69), (6, 0.67),
        (7, 0.66), (8, 0.65), (9, 0.64), (10, 0.63), (12, 0.62), (16, 0.61),
        (22, 0.60), (28, 0.59), (34, 0.58), (41, 0.57), (51, 0.56), (63, 0.55)
    ]
    for umbral, fc in tabla:
        if usuarios <= umbral:
            return fc
    return 0.54


# =====================================================
# Sección #3: Lectura de datos desde Excel
# =====================================================

def leer_hojas_excel(archivo: str):
    """
    Lee las hojas relevantes del archivo Excel.
    Devuelve DataFrames: conexiones, parametros, info_proyecto.
    """
    try:
        df_conexiones = pd.read_excel(archivo, sheet_name="conexiones")
        df_parametros = pd.read_excel(archivo, sheet_name="parametros", usecols=[0, 1])
        df_info = pd.read_excel(archivo, sheet_name="info_proyecto", usecols=[0, 1])
    except Exception as e:
        raise FileNotFoundError(f"Error al leer hojas de Excel: {e}")

    # Limpieza de nombres
    df_parametros.columns = df_parametros.columns.str.strip()
    df_info.columns = df_info.columns.str.strip()

    df_parametros.columns = ["Parámetro", "Valor"]
    df_info.columns = ["info", "Valor"]

    return df_conexiones, df_parametros, df_info


def extraer_valores_clave(df_parametros: pd.DataFrame, df_info: pd.DataFrame) -> Dict[str, Any]:
    """
    Extrae parámetros clave desde los DataFrames.
    Devuelve diccionario con valores principales.
    """
    valores = {}

    try:
        valores["tipo_conductor"] = df_parametros.loc[df_parametros["Parámetro"] == "TipoConductor", "Valor"].iloc[0]
        valores["area_lote"] = float(df_parametros.loc[df_parametros["Parámetro"] == "area_lote", "Valor"].iloc[0])
        valores["capacidad_transformador"] = float(df_parametros.loc[df_parametros["Parámetro"] == "capacidad_transformador", "Valor"].iloc[0])
    except Exception:
        raise KeyError("Faltan parámetros esenciales en la hoja 'parametros'")

    try:
        valores["proyecto_numero"] = df_info.loc[df_info["info"] == "NumeroProyecto", "Valor"].iloc[0]
        valores["proyecto_nombre"] = df_info.loc[df_info["info"] == "NombreProyecto", "Valor"].iloc[0]
        valores["registro_transformador"] = df_info.loc[df_info["info"] == "registrotransformador", "Valor"].iloc[0]
        # Renombramos para reportes
        df_info.loc[df_info["info"] == "registrotransformador", "info"] = "Registro del Transformador"
    except Exception:
        raise KeyError("Faltan datos esenciales en la hoja 'info_proyecto'")

    return valores


def preparar_conexiones(df_conexiones: pd.DataFrame):
    """
    Convierte columnas relevantes a numéricas y extrae listas clave.
    """
    columnas = ["distancia", "usuarios", "usuarios_especiales",
                "area_lote", "area_especial", "nodo_inicial", "nodo_final"]

    for col in columnas:
        if col in df_conexiones.columns:
            df_conexiones[col] = pd.to_numeric(df_conexiones[col], errors="coerce")

    # Quitar parte imaginaria si hubiera
    for col in ["area_lote", "area_especial"]:
        if col in df_conexiones.columns:
            df_conexiones[col] = df_conexiones[col].apply(lambda x: float(np.real(x)) if pd.notna(x) else 0.0)

    return {
        "df_conexiones": df_conexiones,
        "usuarios": df_conexiones["usuarios"].tolist(),
        "distancias": df_conexiones["distancia"].tolist(),
        "nodos_inicio": df_conexiones["nodo_inicial"].tolist(),
        "nodos_final": df_conexiones["nodo_final"].tolist(),
        "usuarios_especiales": df_conexiones["usuarios_especiales"].tolist(),
        "areas_especiales": df_conexiones["area_especial"].tolist(),
    }


def cargar_datos_circuito(archivo: str = "datos_red_secundaria.xlsx") -> Dict[str, Any]:
    """
    Función principal para cargar y preparar todos los datos del circuito.
    Devuelve un diccionario con DataFrames y valores clave.
    """
    df_conexiones, df_parametros, df_info = leer_hojas_excel(archivo)
    valores = extraer_valores_clave(df_parametros, df_info)
    conexiones = preparar_conexiones(df_conexiones)

    return {
        "df_conexiones": conexiones["df_conexiones"],
        "df_parametros": df_parametros,
        "df_info": df_info,
        **valores,
        "usuarios": conexiones["usuarios"],
        "distancias": conexiones["distancias"],
        "nodos_inicio": conexiones["nodos_inicio"],
        "nodos_final": conexiones["nodos_final"],
        "usuarios_especiales": conexiones["usuarios_especiales"],
        "areas_especiales": conexiones["areas_especiales"],
    }

