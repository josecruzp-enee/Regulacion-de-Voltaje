
# utilidades_red.py
import pandas as pd
import numpy as np
import sympy as sp
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from io import BytesIO
import matplotlib.pyplot as plt
import os

# ========== PARTE 1: FUNCIONES BÁSICAS ==========

def cargar_datos_excel(nombre_archivo):
    df_conexiones = pd.read_excel(nombre_archivo, sheet_name='conexiones')
    df_parametros = pd.read_excel(nombre_archivo, sheet_name='parametros', index_col=0)
    df_info = pd.read_excel(nombre_archivo, sheet_name='info_proyecto', index_col=0)
    return df_conexiones, df_parametros, df_info

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

# ========== PARTE 2: EJECUCIÓN COMPLETA ==========

def ejecutar_analisis(nombre_archivo):
    df_conexiones, df_parametros, df_info = cargar_datos_excel(nombre_archivo)
    tipo_conductor = df_parametros.loc['TipoConductor', 'Valor']
    area_lote = df_parametros.loc['area_lote', 'Valor']
    capacidad_transformador = float(df_parametros.loc['capacidad_transformador', 'Valor'])

    # Simulación parcial para demostración:
    df_voltajes = pd.DataFrame({
        'Nodo': [1, 2, 3],
        'Magnitud': [240.0, 236.5, 233.2],
        'Ángulo (°)': [0.0, -1.5, -3.2],
        'Regulación (%)': [0.0, 1.46, 2.83]
    })
    df_regulacion = pd.DataFrame({
        'Nodo': [2, 3],
        'Regulación %': [1.46, 2.83]
    })

    # Generación de PDF simple
    nombre_pdf = "informe_red_electrica.pdf"
    doc = SimpleDocTemplate(nombre_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    elementos = []
    elementos.append(Paragraph("Informe de Red Secundaria", styles['Title']))
    elementos.append(Spacer(1, 12))
    elementos.append(Paragraph("Voltajes nodales:", styles['Heading2']))
    data = [df_voltajes.columns.tolist()] + df_voltajes.values.tolist()
    tabla = Table(data)
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    elementos.append(tabla)
    doc.build(elementos)

    return {"voltajes": df_voltajes, "regulacion": df_regulacion}, nombre_pdf
