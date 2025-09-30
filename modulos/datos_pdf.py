# -*- coding: utf-8 -*-
"""
datos_pdf.py
Preparación de datos y gráficos para reportes PDF.
(Sin estilos ni dependencias de pdf_largo/pdf_corto)
"""

import pandas as pd
from reportlab.platypus import Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle

import matplotlib.pyplot as plt
import io


# ---------------------------------------------------------------------------
# Corrientes y pérdidas
# ---------------------------------------------------------------------------
def tabla_corrientes_vano(df_resultado, perdida_total, voltaje_slack=240):
    """
    Prepara tabla de corrientes por vano y calcula comentario de corriente del transformador.
    Devuelve (df_tabla, comentario_texto).
    """
    df_tabla = df_resultado[['nodo_inicial', 'nodo_final', 'I_vano']].copy()
    df_tabla['Tramo'] = df_tabla['nodo_inicial'].astype(str) + '-' + df_tabla['nodo_final'].astype(str)
    df_tabla['|I| (A)'] = df_tabla['I_vano'].apply(lambda x: round(abs(x), 1))
    df_tabla = df_tabla.rename(columns={'nodo_inicial': 'Nodo Inicial', 'nodo_final': 'Nodo Final'})
    df_tabla = df_tabla[['Nodo Inicial', 'Nodo Final', 'Tramo', '|I| (A)']]

    comentario = None
    if 'kva' in df_resultado.columns:
        kva_total = df_resultado['kva'].sum()
        corriente_transfo = (kva_total * 1000) / voltaje_slack
        I_abs = abs(corriente_transfo)
        comentario = f"La corriente del transformador es de {I_abs:.2f} A"
        

    return df_tabla, comentario


# ---------------------------------------------------------------------------
# Información del proyecto
# ---------------------------------------------------------------------------
def preparar_info_proyecto(df_info, df_parametros, factor_coinc, potencia_total_kva):
    """
    Devuelve copias de df_info y df_parametros con etiquetas amigables
    y valores adicionales (factor de coincidencia, carga inicial).
    """
    df_info_copy = df_info.copy()
    df_info_copy.loc[df_info_copy['info'] == 'NumeroProyecto', 'info'] = 'Código de Proyecto'
    df_info_copy.loc[df_info_copy['info'] == 'NombreProyecto', 'info'] = 'Nombre del Proyecto'
    df_info_copy.loc[df_info_copy['info'] == 'registrotransformador', 'info'] = 'Registro del Transformador'

    df_parametros_copy = df_parametros.copy()
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'TipoConductor', 'Parámetro'] = 'Calibre del Conductor'
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'area_lote', 'Parámetro'] = 'Tamaño del Lote Típico m²'
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'Regulacion_Primaria', 'Parámetro'] = 'Regulación en la Red Primaria, %'
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'Regulacion_Transformador', 'Parámetro'] = 'Regulación Nominal en el Transformador, %'
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'capacidad_transformador', 'Parámetro'] = 'Capacidad Nominal del Transformador (kVA)'
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'Voltaje_Nominal_Primario', 'Parámetro'] = 'Voltaje Nominal de Red Primaria (V)'
    df_parametros_copy.loc[df_parametros_copy['Parámetro'] == 'Voltaje_Nominal_Secundario', 'Parámetro'] = 'Voltaje Nominal de Red Secundaria (V)'

    nuevas_filas = pd.DataFrame([
        {'Parámetro': 'Factor de Coincidencia', 'Valor': f"{factor_coinc:.2f}"},
        {'Parámetro': 'Carga Inicial (KVA)', 'Valor': f"{potencia_total_kva:.2f}"}
    ])
    df_parametros_copy = pd.concat([df_parametros_copy, nuevas_filas], ignore_index=True)

    return df_info_copy, df_parametros_copy


# ---------------------------------------------------------------------------
# Usuarios conectados
# ---------------------------------------------------------------------------
def preparar_tabla_usuarios_conectados(df_conexiones):
    df_clean = df_conexiones.copy()

    # Si existen usuarios especiales, los sumamos a los normales
    if "usuarios_especiales" in df_clean.columns:
        df_clean["usuarios_totales"] = df_clean["usuarios"].fillna(0) + df_clean["usuarios_especiales"].fillna(0)
    else:
        df_clean["usuarios_totales"] = df_clean["usuarios"].fillna(0)

    # Columnas a mostrar
    columnas_tabla = ["nodo_final", "distancia", "usuarios_totales", "P", "S_compleja"]

    # Renombrar para presentación
    df_tabla = df_clean[columnas_tabla].copy()
    df_tabla.rename(columns={
        "usuarios_totales": "Usuarios",
        "nodo_final": "Nodo",
        "distancia": "Distancia (m)",
        "P": "P (kW)",
        "S_compleja": "S (kVA)"
    }, inplace=True)

    # Redondear valores
    df_tabla["P (kW)"] = df_tabla["P (kW)"].apply(lambda x: round(x.real, 2))
    df_tabla["S (kVA)"] = df_tabla["S (kVA)"].apply(lambda x: round(abs(x), 2))

    return df_tabla


# ---------------------------------------------------------------------------
# Proyección de demanda y pérdidas
# ---------------------------------------------------------------------------


def preparar_tabla_proyeccion(df_proyeccion, proyeccion_perdidas):
    """
    Prepara DataFrame con demanda, pérdidas y % carga.
    """
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


def generar_comentario_cargabilidad(df_proyeccion):
    """
    Devuelve comentario (texto + color) sobre la cargabilidad del equipo.
    Acepta que la columna '% Carga (%)' sea string con '%' o float.
    """
    col = df_proyeccion['% Carga (%)']

    # Normalizar a float (quita % si es string)
    if col.dtype == "object":
        ultimo_pct_carga = col.str.strip().str.rstrip('%').astype(float).iloc[-1]
    else:
        ultimo_pct_carga = col.astype(float).iloc[-1]

    if ultimo_pct_carga < 100:
        return "✅ La cargabilidad está dentro del valor nominal (<100%).", colors.green
    elif 100 <= ultimo_pct_carga <= 110:
        return "⚠️ La cargabilidad supera el valor nominal pero está dentro del límite aceptado (≤110%).", colors.yellow
    else:
        return "❌ La cargabilidad sobrepasa el límite térmico aceptado (>110%).", colors.red
