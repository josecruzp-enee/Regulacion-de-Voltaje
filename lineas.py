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
    mu_0 = 4 * np.pi * 1e-7
    L_prim = (mu_0 / (2 * np.pi)) * np.log(separacion_fases_metros / radio_conductor_metros)
    X_prim = 2 * np.pi * frecuencia * L_prim
    return 2 * X_prim * distancia_metros

def calcular_impedancia(row):
    if row['distancia'] == 0:
        return 0 + 0j
    return row['resistencia_vano'] + 1j * row['reactancia_vano']

def calcular_admitancia(z):
    if abs(z) < 1e-12:
        return 0 + 0j
    return 1 / z
