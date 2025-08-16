# informe_corto.py

import streamlit as st
import io
import pandas as pd
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepInFrame
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors

from módulo_de_regulacion_de_voltaje import (
    cargar_datos_circuito, resistencia_por_vano, reactancia_por_vano_geometrica,
    calcular_impedancia, calcular_admitancia, calcular_potencia_carga,
    calcular_matriz_admitancia, calcular_voltajes_nodales,
    calcular_corrientes, calcular_perdidas_y_proyeccion, bibloteca_conductores,
    factor_coincidencia, calcular_regulacion_y_proyeccion,
    crear_grafico_nodos, crear_grafico_voltajes, crear_grafico_proyeccion
)

# ==========================
# Función para generar PDF
# ==========================
def generar_pdf_dashboard_bytes(potencia_total_kva, perdida_total, capacidad_transformador,
                               nodos_inicio, nodos_final, usuarios, distancias,
                               df_regulacion, df_voltajes, df_proyeccion,
                               df_corrientes):

    buffer_pdf = io.BytesIO()
    doc = SimpleDocTemplate(buffer_pdf, pagesize=landscape(letter),
                            rightMargin=15, leftMargin=15, topMargin=15, bottomMargin=15)
    elementos = []

    # Título
    estilo_titulo = ParagraphStyle('titulo', fontSize=16, alignment=TA_CENTER, spaceAfter=10, fontName='Helvetica-Bold')
    elementos.append(Paragraph("Informe de Red Eléctrica - Dashboard", estilo_titulo))
    elementos.append(Spacer(1,6))

    # --- Tablas ---
    # Tabla Nodos
    tabla_nodos_data = [['Nodo Inicio','Nodo Final','Usuarios','Distancia (m)']]
    for i in range(len(nodos_inicio)):
        tabla_nodos_data.append([str(nodos_inicio[i]), str(nodos_final[i]),
                                 str(usuarios[i]), f"{distancias[i]:.2f}"])
    tabla_nodos = Table(tabla_nodos_data, colWidths=[40]*4)
    tabla_nodos.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0), colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),6),
        ('GRID',(0,0),(-1,-1),0.25,colors.black)
    ]))

    # Tabla Regulación
    tabla_volt_data = [list(df_regulacion.columns)]
    for row in df_regulacion.itertuples(index=False):
        tabla_volt_data.append([str(cell) for cell in row])
    tabla_volt = Table(tabla_volt_data, colWidths=[50]*len(df_regulacion.columns))
    tabla_volt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.darkblue),
        ('TEXTCOLOR',(0,0),(-1,0), colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),6),
        ('GRID',(0,0),(-1,-1),0.25,colors.black)
    ]))

    # Tabla Corrientes
    tabla_corr_data = [['Nodo Ini','Nodo Fin','Tramo','|I| (A)']]
    # Convertimos a DataFrame si df_corrientes es lista
    if isinstance(df_corrientes, list):
        df_corrientes = pd.DataFrame(df_corrientes)
    for row in df_corrientes.itertuples(index=False):
        tabla_corr_data.append([row.nodo_inicial,row.nodo_final,row.tramo,f"{row.I:.1f}"])
    tabla_corr = Table(tabla_corr_data, colWidths=[40]*4)
    tabla_corr.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.green),
        ('TEXTCOLOR',(0,0),(-1,0), colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),6),
        ('GRID',(0,0),(-1,-1),0.25,colors.black)
    ]))

    # --- Gráficos ---
    buf_nodos = crear_grafico_nodos(nodos_inicio, nodos_final, usuarios, distancias, capacidad_transformador)
    buf_voltajes = crear_grafico_voltajes(df_voltajes)
    buf_proy = crear_grafico_proyeccion(df_proyeccion)

    img_nodos = Image(buf_nodos, width=250, height=150)
    img_volt = Image(buf_voltajes, width=250, height=150)
    img_proy = Image(buf_proy, width=250, height=150)

    # --- Frames horizontales ---
    left_frame_content = [tabla_nodos, Spacer(1,2), tabla_volt, Spacer(1,2), tabla_corr]
    right_frame_content = [img_nodos, Spacer(1,2), img_volt, Spacer(1,2), img_proy]

    k_left = KeepInFrame(250, 600, left_frame_content)
    k_right = KeepInFrame(300, 600, right_frame_content)

    main_table = Table([[k_left, k_right]], colWidths=[270,320])
    main_table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP')]))
    elementos.append(main_table)

    doc.build(elementos)
    buffer_pdf.seek(0)
    return buffer_pdf

# ==========================
# Streamlit App
# ==========================
st.title("Análisis de Regulación de Voltaje a nivel de Red Secundaria")

archivo = 'datos_circuito.xlsx'

# --- Cargar datos y calcular ---
df_conexiones, df_parametros, df_info, tipo_conductor, area_lote, capacidad_transformador, \
proyecto_numero, proyecto_nombre, transformador_numero, usuarios, distancias, nodos_inicio, nodos_final = cargar_datos_circuito(archivo)

conductores = bibloteca_conductores()
sep_fases = 0.2032
radio_cond = 0.00735

df_conexiones['resistencia_vano'] = df_conexiones.apply(lambda row: resistencia_por_vano(conductores, tipo_conductor, row['distancia']), axis=1)
df_conexiones['reactancia_vano'] = df_conexiones.apply(lambda row: reactancia_por_vano_geometrica(row['distancia'], sep_fases, radio_cond), axis=1)
df_conexiones['Z_vano'] = df_conexiones.apply(calcular_impedancia, axis=1)
df_conexiones['Y_vano'] = df_conexiones['Z_vano'].apply(calcular_admitancia)

total_usuarios = df_conexiones['usuarios'].sum()
factor_coinc = factor_coincidencia(total_usuarios)
df_conexiones, potencia_total_kva, _ = calcular_potencia_carga(df_conexiones, area_lote)
Y, Yrr, Y_r0, nodos, slack_index = calcular_matriz_admitancia(df_conexiones)
V, _ = calcular_voltajes_nodales(Yrr, Y_r0, slack_index, nodos)
df_conexiones = calcular_corrientes(df_conexiones, V)
df_conexiones, perdida_total, df_corrientes = calcular_perdidas_y_proyeccion(df_conexiones)

df_proyeccion, df_voltajes, df_regulacion = calcular_regulacion_y_proyeccion(
    potencia_total_kva, df_parametros, Yrr, Y_r0, slack_index, nodos, nodos[slack_index]
)

# --- Generar PDF como bytes ---
pdf_bytes = generar_pdf_dashboard_bytes(
    potencia_total_kva, perdida_total, capacidad_transformador,
    nodos_inicio, nodos_final, usuarios, distancias,
    df_regulacion, df_voltajes, df_proyeccion, df_corrientes
)

# --- Botón de descarga ---
st.download_button(
    label="📄 Descargar PDF Dashboard",
    data=pdf_bytes,
    file_name="informe_corto.pdf",  # <- ahora coincide con tu archivo
    mime="application/pdf"
)
