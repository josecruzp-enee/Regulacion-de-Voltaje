
# ===================== IMPORTS DE MÓDULOS =====================
# -*- coding: utf-8 -*-
"""
pdf_largo.py
Generación de informe largo en PDF para análisis de red secundaria
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

# === Estilos y utilidades ===
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
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#d3d3d3")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), alineacion),
    ])
    return tabla


# === Secciones PDF ===
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

    # Columnas básicas siempre presentes
    cols = ['Año', 'Demanda (kVA)', '% Carga (%)']

    # Manejar pérdidas según lo que exista
    if 'Pérdidas (kWh) (fmt)' in df_proyeccion.columns:
        cols.append('Pérdidas (kWh) (fmt)')
    elif 'Pérdidas (kWh)' in df_proyeccion.columns:
        cols.append('Pérdidas (kWh)')

    # Si no existe ninguna columna de pérdidas, no pasa nada
    df_reporte = df_proyeccion[cols].copy()

    # Asegurar formato de % Carga si es numérico
    if pd.api.types.is_numeric_dtype(df_reporte['% Carga (%)']):
        df_reporte['% Carga (%)'] = df_reporte['% Carga (%)'].map(lambda x: f"{x:.2f}%")

    # Insertar tabla
    elementos.append(crear_tabla(df_reporte, encabezados=df_reporte.columns.tolist()))
    elementos.append(Spacer(1, 12))

    # Insertar gráfico
    serie_demanda = df_proyeccion.get("Demanda_kva")
    img_demanda = crear_grafico_demanda(df_proyeccion)
    img_demanda.drawWidth = 5 * inch
    img_demanda.drawHeight = 3 * inch
    elementos.append(img_demanda)
    elementos.append(Spacer(1, 12))

    # Comentario de cargabilidad
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
        Paragraph(f"Pérdida total estimada en la red: {perdida_total:,.2f} kWh/año",
                  ParagraphStyle(name="ComentarioPerdidas", fontSize=12, leading=14, alignment=TA_LEFT)),
    ]
    if comentario:
        elementos.append(Paragraph(comentario, ParagraphStyle(
            name="ComentarioCorriente", fontSize=12, leading=14, alignment=TA_LEFT, textColor=colors.blue)))
    elementos.append(Spacer(1, 12))
    return elementos


# === Funciones principales ===
def generar_informe_pdf(df_info, df_parametros, factor_coinc, potencia_total_kva,
                        df_conexiones, df_voltajes, df_regulacion,
                        capacidad_transformador, df_proyeccion,
                        perdida_total, nodos_inicio, nodos_final, usuarios, distancias,
                        ruta_salida=None):

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
    for seccion in [
        seccion_info_proyecto_render(df_info),
        seccion_parametros_render(df_parametros, factor_coinc, potencia_total_kva),
        seccion_usuarios_y_voltajes(df_conexiones, df_voltajes, df_regulacion, crear_tabla, crear_subtitulo),
        seccion_proyeccion_render(df_proyeccion),
        seccion_corrientes_render(df_conexiones, perdida_total),
    ]:
        elementos.extend(seccion)

    grafico = crear_grafico_nodos(nodos_inicio, nodos_final, usuarios, distancias,
                                  capacidad_transformador, df_conexiones)
    grafico.drawWidth = 6*inch
    grafico.drawHeight = 4.5*inch
    elementos.append(grafico)

    doc.build(elementos)
    print(f"✅ Informe largo generado: {ruta_salida}")
    return ruta_salida


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

    conductores = biblioteca_conductores()
    ddf_conexiones["resistencia_vano"] = df_conexiones["distancia"].apply(
        lambda d: resistencia_por_vano(conductores, tipo_conductor, d)
    )
    df_conexiones["reactancia_vano"] = df_conexiones["distancia"].apply(
        lambda d: reactancia_por_vano(conductores, tipo_conductor, d)
    )
    df_conexiones["Z_vano"] = df_conexiones.apply(calcular_impedancia, axis=1)
    df_conexiones["Y_vano"] = df_conexiones["Z_vano"].apply(calcular_admitancia)

    factor_coinc = factor_coincidencia(df_conexiones["usuarios"].sum())
    df_conexiones, potencia_total_kva, _ = calcular_potencia_carga(
        df_conexiones, area_lote, tipo_conductor
    )

    Y, Yrr, Y_r0, nodos, slack_index = calcular_matriz_admitancia(df_conexiones)
    V, df_voltajes = calcular_voltajes_nodales(Yrr, Y_r0, slack_index, nodos, V0=240+0j)
    nodo_slack = nodos[slack_index]

    df_conexiones = calcular_corrientes(df_conexiones, V)
    df_conexiones, perdida_total, proyeccion_perdidas = calcular_perdidas_y_proyeccion(df_conexiones)

    df_proyeccion = pd.DataFrame({'Año': range(0, 16)})
    df_proyeccion = proyectar_demanda(
        potencia_total_kva, df_parametros.set_index('Parámetro'), df_proyeccion,
        crecimiento=0.02, años=15, proyeccion_perdidas=proyeccion_perdidas
    )

    return generar_informe_pdf(
        df_info, df_parametros, factor_coinc, potencia_total_kva,
        df_conexiones, df_voltajes, calcular_regulacion_voltaje(V, nodos=nodos, nodo_slack=nodo_slack),
        capacidad_transformador, df_proyeccion,
        perdida_total, nodos_inicio, nodos_final, usuarios, distancias
    )


