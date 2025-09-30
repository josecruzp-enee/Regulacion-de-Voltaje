# -*- coding: utf-8 -*-
"""
voltajes.py
Sección #4: Cálculo de Voltajes y Regulación
"""

import numpy as np
import pandas as pd
from reportlab.platypus import Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import matplotlib.pyplot as plt
import io


# ====================== Sección #4: Cálculo de Voltajes y Regulación ======================

# ---------------------------------------------------------------------------
# Subsección #4.1: Cálculo de voltajes nodales
# ---------------------------------------------------------------------------

def calcular_voltajes_nodales(Yrr, Y_r0, slack_index, nodos, V0=240+0j):
    """
    Calcula los voltajes nodales resolviendo el sistema lineal de admitancias (excluyendo nodo slack).
    """
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


# ---------------------------------------------------------------------------
# Subsección #4.2: Cálculo de regulación de voltaje
# ---------------------------------------------------------------------------

def calcular_regulacion_voltaje(V, nodos, nodo_slack, Vbase=240):
    """
    Calcula la regulación de voltaje (%) en cada nodo en relación al nodo slack.
    """
    V0 = V[nodos.index(nodo_slack)]
    regulacion = []

    for i, nodo in enumerate(nodos):
        regul = (abs(V0) - abs(V[i])) / abs(V0) * 100
        regulacion.append({
            'Nodo': nodo,
            'Voltaje (p.u.)': abs(V[i]) / Vbase,
            'Voltaje Absoluto (V)': abs(V[i]),  
            'Regulación (%)': regul
        })

    df_regulacion = pd.DataFrame(regulacion)
    return df_regulacion


# ---------------------------------------------------------------------------
# Subsección #4.3: Preparación de DataFrame de voltajes para reporte
# ---------------------------------------------------------------------------

def preparar_df_voltajes(df_voltajes, df_regulacion, Vbase=240):
    """
    Une voltajes y regulación para reporte.
    """
    df_temp = pd.merge(df_voltajes, df_regulacion[['Nodo', 'Regulación (%)']], on='Nodo', how='left')

    df_temp['Voltaje (p.u.)'] = df_temp['Magnitud (V)'] / Vbase
    df_temp['Voltaje (p.u.)'] = df_temp['Voltaje (p.u.)'].map(lambda x: f"{x:.3f}")
    df_temp['Voltaje Absoluto (V)'] = df_temp['Magnitud (V)'].map(lambda x: f"{x:.2f}")
    df_temp['Regulación (%)'] = df_temp['Regulación (%)'].map(lambda x: f"{x:.2f}")

    return df_temp[['Nodo', 'Voltaje (p.u.)', 'Voltaje Absoluto (V)', 'Regulación (%)']]



# ---------------------------------------------------------------------------
# Subsección #4.4: Secciones de voltajes para PDF
# ---------------------------------------------------------------------------




def crear_comentario_regulacion(df_voltajes):
    """
    Genera comentario sobre regulación de voltaje (aceptable o no).
    """
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
        alignment=TA_CENTER,
        textColor=color_comentario,
        spaceBefore=12,
        spaceAfter=12,
        fontName='Helvetica-Bold'
    ))
    
    return parrafo





