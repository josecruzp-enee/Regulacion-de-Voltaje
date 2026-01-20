# -*- coding: utf-8 -*-
"""
lineas.py
Cálculos de parámetros de línea: resistencias, reactancias, impedancias

MODO RV-2002 (TABULADO):
- Se ignora geometría (separación, radio, frecuencia)
- R y X se toman desde biblioteca_conductores() (Ω/km por conductor)
- Para el vano (ida + retorno): R_vano = 2*R_km*L_km, X_vano = 2*X_km*L_km
"""

import numpy as np


def resistencia_por_vano(conductores, tipo_conductor, distancia_metros):
    """
    R_vano = 2 * R(Ω/km) * L(km)
    """
    if tipo_conductor not in conductores:
        raise ValueError(f"Conductor '{tipo_conductor}' no está en el diccionario.")
    distancia_km = distancia_metros / 1000.0
    return 2.0 * float(conductores[tipo_conductor]["R"]) * distancia_km


def reactancia_por_vano_geometrica(conductores, tipo_conductor, distancia_metros, separacion_fases_metros=None, radio_conductor_metros=None, frecuencia=60):
    """
    COMPATIBILIDAD (MODO RV):
    Se mantiene el nombre, pero se ignora geometría y se usa X tabulada.

    X_vano = 2 * X(Ω/km) * L(km)
    """
    if tipo_conductor not in conductores:
        raise ValueError(f"Conductor '{tipo_conductor}' no está en el diccionario.")
    if "X" not in conductores[tipo_conductor]:
        raise ValueError(f"Conductor '{tipo_conductor}' no tiene 'X' en el diccionario.")
    distancia_km = distancia_metros / 1000.0
    return 2.0 * float(conductores[tipo_conductor]["X"]) * distancia_km


def calcular_impedancia(row):
    if row["distancia"] == 0:
        return 0 + 0j
    return row["resistencia_vano"] + 1j * row["reactancia_vano"]


def calcular_admitancia(z):
    if abs(z) < 1e-12:
        return 0 + 0j
    return 1 / z
