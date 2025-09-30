# -*- coding: utf-8 -*-
"""
demanda.py
Módulo para proyección de demanda y pérdidas en redes secundarias
"""

import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.platypus import Image


# =====================================================
# Función: proyección de demanda y pérdidas
# =====================================================
def proyectar_demanda(potencia_total_kva, df_parametros, df_proyeccion,
                      crecimiento=0.02, años=15, proyeccion_perdidas=None):
    """
    Calcula la proyección de demanda y pérdidas para un número de años,
    comenzando desde el año 0 hasta el año `años`.

    Devuelve un DataFrame con columnas:
    - Año
    - Demanda_kva (numérico)
    - % Carga (%) (numérico)
    - Pérdidas (kWh) (numérico)
    - Demanda (kVA) (texto para reporte)
    - % Carga (texto para reporte)
    - Pérdidas (kWh) (fmt) (texto para reporte)
    """

    # Definir rango de años (0 a años)
    df_proyeccion['Año'] = range(0, años + 1)

    # Proyección de demanda (incluye año 0 como base)
    proyeccion_demanda = [
        potencia_total_kva * ((1 + crecimiento) ** i) for i in range(0, años + 1)
    ]

    # Capacidad del transformador
    capacidad_transformador = float(df_parametros.loc['capacidad_transformador', 'Valor'])

    # Columnas numéricas
    df_proyeccion['Demanda_kva'] = proyeccion_demanda
    df_proyeccion['% Carga (%)'] = (df_proyeccion['Demanda_kva'] / capacidad_transformador) * 100

    # Pérdidas: si no existen, igualamos con NaN
    if proyeccion_perdidas is not None and len(proyeccion_perdidas) > 0:
        perdidas_ajustadas = list(proyeccion_perdidas) + [proyeccion_perdidas[-1]]
        df_proyeccion['Pérdidas (kWh)'] = perdidas_ajustadas[:len(df_proyeccion)]
    else:
        df_proyeccion['Pérdidas (kWh)'] = np.nan

    # Columnas formateadas para reporte
    df_proyeccion['Demanda (kVA)'] = df_proyeccion['Demanda_kva'].map(lambda x: f"{x:,.2f}")
    df_proyeccion['% Carga'] = df_proyeccion['% Carga (%)'].map(lambda x: f"{x:.2f}%")
    df_proyeccion['Pérdidas (kWh) (fmt)'] = df_proyeccion['Pérdidas (kWh)'].map(
        lambda x: f"{x:,.2f}" if pd.notna(x) else "-"
    )

    return df_proyeccion


# =====================================================
# Función: gráfico de demanda proyectada
# =====================================================
def crear_grafico_demanda(df_proyeccion):
    """
    Genera un gráfico SOLO de la demanda proyectada
    y devuelve un objeto Image para insertar en el PDF.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(
        df_proyeccion['Año'],
        df_proyeccion['Demanda_kva'],
        marker='o',
        color="tab:blue",
        label="Demanda (kVA)"
    )

    ax.set_xlabel("Años")
    ax.set_ylabel("Demanda (kVA)")
    ax.set_title("Proyección de Demanda")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Image(buf)
