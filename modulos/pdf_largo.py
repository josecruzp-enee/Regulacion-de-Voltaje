# -*- coding: utf-8 -*-
"""
pdf_largo.py
Generación de informe largo en PDF para análisis de red secundaria.

Refactor:
- Añade columnas r/x por conductor (1c) para reporte sin romper cálculo (lazo).
- Añade sección "Circuito equivalente nodal (Nivel 2)" usando modulos.equivalente_nodal
"""

import os
import pandas as pd
import numpy as np

from reportlab.platypus import (
    SimpleDocTemplate, Frame, PageTemplate, Spacer,
    Paragraph, Table, TableStyle
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.utils import ImageReader

# ===================== IMPORTS DE MÓDULOS =====================
from modulos.datos import cargar_datos_circuito, biblioteca_conductores
from modulos.lineas import resistencia_por_vano, reactancia_por_vano, calcular_impedancia, calcular_admitancia
from modulos.cargas import calcular_potencia_carga
from modulos.constantes import definir_constantes
from modulos.matrices import calcular_matriz_admitancia
from modulos.voltajes import calcular_voltajes_nodales, calcular_regulacion_voltaje
from modulos.corrientes import calcular_corrientes, calcular_perdidas_y_proyeccion
from modulos.datos_pdf import preparar_info_proyecto, preparar_tabla_proyeccion, tabla_corrientes_vano
from modulos.secciones_pdf import crear_grafico_voltajes_pdf, seccion_usuarios_y_voltajes
from modulos.demanda import proyectar_demanda, crear_grafico_demanda
from modulos.graficos_red import crear_grafico_nodos
from modulos.datos_pdf import generar_comentario_cargabilidad
from modulos.datos import factor_coincidencia

# NUEVO: equivalente nodal
from modulos.equivalente_nodal import construir_equivalente_nodal


# ============================================================
# CONFIG (ajusta si tus columnas se llaman diferente)
# ============================================================
COL_NI = "nodo_inicio"
COL_NF = "nodo_final"


# ============================================================
# Estilos y utilidades
# ============================================================
def encabezado(canvas, doc):
    canvas.saveState()
    ancho_pagina = doc.pagesize[0]
    y = doc.height + doc.topMargin - 20
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawCentredString(ancho_pagina / 2, y, "Empresa Nacional de Energía Eléctrica, ENEE")
    canvas.drawCentredString(ancho_pagina / 2, y - 18, "Análisis de Red Secundaria de Distribución")
    canvas.restoreState()


def aplicar_fondo(cnv, doc, ruta_imagen_fondo):
    try:
        ancho, alto = doc.pagesize
        imagen_fondo = ImageReader(ruta_imagen_fondo)
        cnv.drawImage(imagen_fondo, 0, 0, width=ancho, height=alto)
    except Exception as e:
        print(f"Error al aplicar fondo: {e}")


def fondo_y_encabezado(cnv, doc):
    ruta_imagen_fondo = os.path.join(os.path.dirname(__file__), "Imagen Encabezado.jpg")
    aplicar_fondo(cnv, doc, ruta_imagen_fondo)
    encabezado(cnv, doc)


def crear_titulo(texto, tamano=16):
    estilo = ParagraphStyle(
        name='Titulo', fontSize=tamano, leading=tamano+4,
        alignment=TA_CENTER, spaceBefore=24, spaceAfter=24,
        fontName='Helvetica-Bold'
    )
    return Paragraph(texto, estilo)


def crear_subtitulo(texto, tamano=14):
    estilo = ParagraphStyle(
        name='Subtitulo', fontSize=tamano, leading=tamano+4,
        alignment=TA_LEFT, spaceBefore=12, spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    return Paragraph(texto, estilo)


def crear_tabla(df, encabezados=None, alineacion='CENTER'):
    data = [encabezados] + df.values.tolist() if encabezados else [df.columns.to_list()] + df.values.tolist()
    tabla = Table(data, hAlign=alineacion)
    tabla.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#d3d3d3")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), alineacion),
    ])
    return tabla


# ============================================================
# Secciones PDF existentes
# ============================================================
def seccion_info_proyecto_render(df_info):
    return [
        Spacer(1, 6),
        crear_titulo("Gerencia de Distribución, Unidad de Proyectos"),
        Spacer(1, 12),
        crear_subtitulo("1. Información del Proyecto"),
        crear_tabla(df_info, encabezados=['Parámetro', 'Valor']),
        Spacer(1, 12),
    ]


def seccion_parametros_render(df_parametros, factor_coinc, potencia_total_kva):
    df_parametros_ext = df_parametros.copy()
    df_parametros_ext.loc[len(df_parametros_ext)] = ["Potencia Total Estimada (kVA)", f"{potencia_total_kva:.2f}"]
    df_parametros_ext.loc[len(df_parametros_ext)] = ["Factor de Coincidencia", f"{factor_coinc:.3f}"]

    return [
        crear_subtitulo("2. Parámetros Generales del Sistema"),
        crear_tabla(df_parametros_ext, encabezados=['Parámetro', 'Valor']),
        Spacer(1, 12),
    ]


def seccion_proyeccion_render(df_proyeccion):
    elementos = [crear_subtitulo("5. Proyección de Demanda y Pérdidas Eléctricas Anuales")]

    cols = ['Año', 'Demanda (kVA)', '% Carga (%)']

    if 'Pérdidas (kWh) (fmt)' in df_proyeccion.columns:
        cols.append('Pérdidas (kWh) (fmt)')
    elif 'Pérdidas (kWh)' in df_proyeccion.columns:
        cols.append('Pérdidas (kWh)')

    df_reporte = df_proyeccion[cols].copy()

    if pd.api.types.is_numeric_dtype(df_reporte['% Carga (%)']):
        df_reporte['% Carga (%)'] = df_reporte['% Carga (%)'].map(lambda x: f"{x:.2f}%")

    elementos.append(crear_tabla(df_reporte, encabezados=df_reporte.columns.tolist()))
    elementos.append(Spacer(1, 12))

    img_demanda = crear_grafico_demanda(df_proyeccion)
    img_demanda.drawWidth = 5 * inch
    img_demanda.drawHeight = 3 * inch
    elementos.append(img_demanda)
    elementos.append(Spacer(1, 12))

    texto, color = generar_comentario_cargabilidad(df_proyeccion)
    estilo = ParagraphStyle(name="ComentarioCargabilidad", fontSize=13, leading=16, alignment=TA_LEFT, textColor=color)
    elementos.append(Paragraph(texto, estilo))
    elementos.append(Spacer(1, 12))

    return elementos


def seccion_corrientes_render(df_conexiones, perdida_total):
    df_tabla, comentario = tabla_corrientes_vano(df_conexiones, perdida_total)
    tabla = Table([df_tabla.columns.tolist()] + df_tabla.values.tolist())
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))

    elementos = [
        crear_subtitulo("6. Corrientes en tramos de red"),
        tabla,
        Spacer(1, 12),
        Paragraph(
            f"Pérdida total estimada en la red: {perdida_total:,.2f} kWh/año",
            ParagraphStyle(name="ComentarioPerdidas", fontSize=12, leading=14, alignment=TA_LEFT)
        ),
    ]
    if comentario:
        elementos.append(Paragraph(comentario, ParagraphStyle(
            name="ComentarioCorriente", fontSize=12, leading=14, alignment=TA_LEFT,
            textColor=colors.blue
        )))
    elementos.append(Spacer(1, 12))
    return elementos


# ============================================================
# NUEVA sección: Equivalente nodal nivel 2
# ============================================================
def seccion_equivalente_nodal_render(eq):
    elementos = []
    elementos.append(crear_subtitulo("7. Circuito equivalente nodal (Nivel 2)"))

    elementos.append(Paragraph(
        f"Nodo slack: {eq.slack_nodo} (índice {eq.slack_index})",
        ParagraphStyle(name="SlackEq", fontSize=11, leading=13, alignment=TA_LEFT)
    ))
    elementos.append(Spacer(1, 6))

    elementos.append(crear_subtitulo("7.1 Mapa Nodo → Índice"))
    elementos.append(crear_tabla(eq.df_mapa, encabezados=eq.df_mapa.columns.tolist()))
    elementos.append(Spacer(1, 12))

    elementos.append(crear_subtitulo("7.2 Ramas del circuito (R, X, Z, Y)"))
    elementos.append(crear_tabla(eq.df_ramas, encabezados=eq.df_ramas.columns.tolist()))
    elementos.append(Spacer(1, 12))

    elementos.append(crear_subtitulo("7.3 Matriz Y (G + jB)"))
    if eq.nota_Y:
        elementos.append(Paragraph(eq.nota_Y, ParagraphStyle(
            name="NotaRecorteY", fontSize=9, leading=11, alignment=TA_LEFT, textColor=colors.grey
        )))
    elementos.append(crear_tabla(eq.df_Y, encabezados=eq.df_Y.columns.tolist()))
    elementos.append(Spacer(1, 12))

    elementos.append(crear_subtitulo("7.4 Submatriz Yrr"))
    if eq.nota_Yrr:
        elementos.append(Paragraph(eq.nota_Yrr, ParagraphStyle(
            name="NotaRecorteYrr", fontSize=9, leading=11, alignment=TA_LEFT, textColor=colors.grey
        )))
    elementos.append(crear_tabla(eq.df_Yrr, encabezados=eq.df_Yrr.columns.tolist()))
    elementos.append(Spacer(1, 12))

    elementos.append(crear_subtitulo("7.5 Vector Y_r0"))
    elementos.append(crear_tabla(eq.df_Yr0, encabezados=eq.df_Yr0.columns.tolist()))
    elementos.append(Spacer(1, 12))

    elementos.append(Paragraph(
        "Ecuación nodal reducida:  V_r = inv(Yrr) · (I_r − Y_r0·V0).",
        ParagraphStyle(name="EqNodal", fontSize=10, leading=12, alignment=TA_LEFT, textColor=colors.grey)
    ))
    elementos.append(Spacer(1, 12))
    return elementos


# ============================================================
# Función principal: construir PDF
# ============================================================
def generar_informe_pdf(
    df_info, df_parametros, factor_coinc, potencia_total_kva,
    df_conexiones, df_voltajes, df_regulacion,
    capacidad_transformador, df_proyeccion,
    perdida_total, nodos_inicio, nodos_final, usuarios, distancias,
    eq=None,
    ruta_salida=None
):
    try:
        registro_transformador = df_info.loc[df_info["info"] == "Registro del Transformador", "Valor"].values[0]
        registro_transformador = str(registro_transformador).strip()
    except Exception:
        registro_transformador = "Sin Registro"

    if ruta_salida is None:
        ruta_salida = f"Análisis completo de Transformador {registro_transformador}.pdf"

    doc = SimpleDocTemplate(ruta_salida, pagesize=letter)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    template = PageTemplate(id="fondo_total", frames=[frame], onPage=fondo_y_encabezado)
    doc.addPageTemplates([template])

    elementos = []

    # Secciones
    for seccion in [
        seccion_info_proyecto_render(df_info),
        seccion_parametros_render(df_parametros, factor_coinc, potencia_total_kva),
        seccion_usuarios_y_voltajes(df_conexiones, df_voltajes, df_regulacion, crear_tabla, crear_subtitulo),
        seccion_proyeccion_render(df_proyeccion),
        seccion_corrientes_render(df_conexiones, perdida_total),
        seccion_equivalente_nodal_render(eq) if eq is not None else [],
    ]:
        elementos.extend(seccion)

    # Gráfico red (ya existente)
    grafico = crear_grafico_nodos(
        nodos_inicio, nodos_final, usuarios, distancias,
        capacidad_transformador, df_conexiones
    )
    grafico.drawWidth = 6 * inch
    grafico.drawHeight = 4.5 * inch
    elementos.append(grafico)

    doc.build(elementos)
    print(f"✅ Informe largo generado: {ruta_salida}")
    return ruta_salida


# ============================================================
# Wrapper: desde Excel
# ============================================================
def generar_pdf_largo(ruta_excel):
    ruta_excel = os.path.abspath(ruta_excel)

    # Cargar datos como diccionario
    datos = cargar_datos_circuito(ruta_excel)

    df_conexiones = datos["df_conexiones"]
    df_parametros = datos["df_parametros"]
    df_info = datos["df_info"]
    tipo_conductor = datos["tipo_conductor"]
    area_lote = datos["area_lote"]
    capacidad_transformador = datos["capacidad_transformador"]
    usuarios = datos["usuarios"]
    distancias = datos["distancias"]
    nodos_inicio = datos["nodos_inicio"]
    nodos_final = datos["nodos_final"]

    # --- Impedancias de vano (cálculo)
    conductores = biblioteca_conductores()

    df_conexiones["resistencia_vano"] = df_conexiones["distancia"].apply(
        lambda d: resistencia_por_vano(conductores, tipo_conductor, d)
    )
    df_conexiones["reactancia_vano"] = df_conexiones["distancia"].apply(
        lambda d: reactancia_por_vano(conductores, tipo_conductor, d)
    )

    # Para reporte (por conductor, sin romper cálculo)
    df_conexiones["r_vano_1c"] = df_conexiones["resistencia_vano"] / 2.0
    df_conexiones["x_vano_1c"] = df_conexiones["reactancia_vano"] / 2.0

    df_conexiones["Z_vano"] = df_conexiones.apply(calcular_impedancia, axis=1)
    df_conexiones["Y_vano"] = df_conexiones["Z_vano"].apply(calcular_admitancia)

    # --- Cargas
    factor_coinc = factor_coincidencia(df_conexiones["usuarios"].sum())
    df_conexiones, potencia_total_kva, _ = calcular_potencia_carga(
        df_conexiones, area_lote, tipo_conductor
    )

    # --- Matriz Y y voltajes
    Y, Yrr, Y_r0, nodos, slack_index = calcular_matriz_admitancia(df_conexiones)
    V, df_voltajes = calcular_voltajes_nodales(Yrr, Y_r0, slack_index, nodos, V0=240+0j)
    nodo_slack = nodos[slack_index]

    # --- Corrientes y pérdidas
    df_conexiones = calcular_corrientes(df_conexiones, V)
    df_conexiones, perdida_total, proyeccion_perdidas = calcular_perdidas_y_proyeccion(df_conexiones)

    # --- Proyección
    df_proyeccion = pd.DataFrame({'Año': range(0, 16)})
    df_proyeccion = proyectar_demanda(
        potencia_total_kva, df_parametros.set_index('Parámetro'), df_proyeccion,
        crecimiento=0.02, años=15, proyeccion_perdidas=proyeccion_perdidas
    )

    # --- Construir equivalente nodal nivel 2 (módulo)
    eq = construir_equivalente_nodal(
        df_conexiones,
        nodos=nodos,
        slack_index=slack_index,
        Y=Y,
        Yrr=Yrr,
        Y_r0=Y_r0,
        col_ni="nodo_inicial",   # <-- tu columna real
        col_nf="nodo_final",     # <-- tu columna real
        col_dist="distancia",    # <-- tu columna real
        col_r="resistencia_vano",
        col_x="reactancia_vano",
        col_z="Z_vano",
        col_y="Y_vano",
    )


    # --- Regulación
    df_reg = calcular_regulacion_voltaje(V, nodos=nodos, nodo_slack=nodo_slack)

    return generar_informe_pdf(
        df_info, df_parametros, factor_coinc, potencia_total_kva,
        df_conexiones, df_voltajes, df_reg,
        capacidad_transformador, df_proyeccion,
        perdida_total, nodos_inicio, nodos_final, usuarios, distancias,
        eq=eq
    )

