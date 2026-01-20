# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:13:40 2025

@author: José Nikol Cruz
"""

# -*- coding: utf-8 -*-
"""
lineas.py
Cálculos de parámetros de línea: resistencias, reactancias, impedancias
"""

import numpy as np

def resistencia_por_vano(conductores, tipo_conductor, distancia_metros):
    if tipo_conductor not in conductores:
        raise ValueError(f"Conductor '{tipo_conductor}' no está en el diccionario.")
    distancia_km = distancia_metros / 1000
    return 2 * conductores[tipo_conductor]['R'] * distancia_km

def reactancia_por_vano_geometrica(distancia_metros, separacion_fases_metros, radio_conductor_metros, frecuencia=60):
    """
    MODO RV-2002:
    Ignora geometría y usa X tabulada equivalente a RV para 3/0:
    X_km = 0.29277 Ω/km por conductor (separación 8")
    X_vano = 2 * X_km * L(km)
    """
    X_km = 0.29277  # Ω/km por conductor (RV)
    distancia_km = distancia_metros / 1000
    return 2 * X_km * distancia_km

def calcular_impedancia(row):
    if row['distancia'] == 0:
        return 0 + 0j
    return row['resistencia_vano'] + 1j * row['reactancia_vano']

def calcular_admitancia(z):
    if abs(z) < 1e-12:
        return 0 + 0j
    return 1 / z

