# -*- coding: utf-8 -*-
"""
corrientes.py
Sección #5: Cálculo de Corrientes, Pérdidas y Proyección
"""

import pandas as pd
import numpy as np


# ====================== Sección #5: Cálculo de Corrientes, Pérdidas y Proyección ======================

# ---------------------------------------------------------------------------
# Subsección #5.1: Cálculo de corrientes por vano
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Subsección #5.1: Pérdidas y proyección
# ---------------------------------------------------------------------------

# Función #5.1.2: calcular_perdidas_y_proyeccion
# Calcula las pérdidas en cada vano y proyecta las pérdidas anuales a 15 años con crecimiento.
# Recibe:
# - df: DataFrame con columnas 'I_vano' y 'resistencia_vano'
# Devuelve:
# - df con columna adicional 'P_perdida' (pérdidas en Watts)
# - perdida_total: pérdidas totales anuales en kWh
# - proyeccion: lista con proyección de pérdidas para 15 años (crecimiento anual 2%)
def calcular_perdidas_y_proyeccion(df, LF=0.4, crecimiento=0.02):
    """
    Pérdidas de línea (I^2*R) con R de lazo ya incluida en 'resistencia_vano'.
    Annualiza con k = 0.2*LF + 0.8*LF^2 (LF por norma = 0.4 -> k=0.208).
    """
    P_perdidas = []
    for _, row in df.iterrows():
        I = row['I_vano']                      # A (compleja o real)
        R_loop = float(row['resistencia_vano'].real)  # Ω (YA es de lazo)
        P_perdida = (abs(I) ** 2) * R_loop     # W por tramo
        P_perdidas.append(float(P_perdida))

    df = df.copy()
    df['P_perdida'] = P_perdidas

    # Potencia pérdidas a condición de cálculo (W)
    P_perdida_total = sum(P_perdidas)

    # Loss factor k desde LF fijo de la norma
    k = 0.2 * LF + 0.8 * (LF ** 2)  # con LF=0.4 -> k=0.208

    # Energía anual (kWh/año)
    perdida_total = P_perdida_total * 8760 * k / 1000.0

    # Proyección (crecimiento anual)
    proyeccion = [perdida_total * ((1 + crecimiento) ** i) for i in range(0, 15)]

    return df, perdida_total, proyeccion




    