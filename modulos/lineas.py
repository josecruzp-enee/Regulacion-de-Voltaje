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
from modulos.conductores import biblioteca_conductores
CONDUCTORES = biblioteca_conductores()


def resistencia_por_vano(conductores, tipo_conductor, distancia_m):
    if tipo_conductor not in conductores:
        raise ValueError(f"Conductor '{tipo_conductor}' no está en el diccionario.")
    L_km = float(distancia_m) / 1000.0
    return 2.0 * float(conductores[tipo_conductor]["R"]) * L_km


def reactancia_por_vano(conductores, tipo_conductor, distancia_m):
    if tipo_conductor not in conductores:
        raise ValueError(f"Conductor '{tipo_conductor}' no está en el diccionario.")
    L_km = float(distancia_m) / 1000.0
    return 2.0 * float(conductores[tipo_conductor]["X"]) * L_km



def calcular_impedancia(row):
    if row["distancia"] == 0:
        return 0 + 0j
    return row["resistencia_vano"] + 1j * row["reactancia_vano"]


def calcular_admitancia(z):
    if abs(z) < 1e-12:
        return 0 + 0j
    return 1 / z

