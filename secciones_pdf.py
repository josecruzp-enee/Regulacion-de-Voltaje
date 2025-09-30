# -*- coding: utf-8 -*-
"""
secciones_pdf.py
Armado de secciones para los reportes PDF.
"""

import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import Spacer, Image, Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors

from modulos.datos_pdf import (
    preparar_tabla_usuarios_conectados,
)
from modulos.voltajes import crear_comentario_regulacion
from modulos.voltajes import preparar_df_voltajes



def crear_tabla_voltajes(df_voltajes, df_regulacion, crear_tabla_fn, crear_subtitulo_fn):
    """
    Crea tabla de voltajes para PDF.
    Recibe funciones externas para tabla y subtítulo (desde pdf_largo).
    """
    elementos = []
    elementos.append(crear_subtitulo_fn("4. Análisis de Voltajes Nodales"))

    df_tabla = preparar_df_voltajes(df_voltajes, df_regulacion)
    tabla = crear_tabla_fn(df_tabla, alineacion='CENTER')
    elementos.append(tabla)
    elementos.append(Spacer(1, 12))

    return elementos


def crear_grafico_voltajes_pdf(df_voltajes):
    """
    Genera gráfico de voltajes nodales para insertar en PDF.
    """
    plt.figure(figsize=(6, 3))
    plt.plot(df_voltajes['Nodo'], df_voltajes['Magnitud (V)'], marker='o')
    plt.title("Análisis Nodal de Voltajes")
    plt.xlabel("Nodo")
    plt.ylabel("Magnitud (V)")
    plt.grid(True)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    imagen = Image(buf, width=5 * 72, height=3 * 72)  # 72 dpi = 1 inch
    imagen.hAlign = 'CENTER'
    return imagen   # 👈 devolver solo la imagen



def seccion_usuarios_y_voltajes(df_conexiones, df_voltajes, df_regulacion, crear_tabla_fn, crear_subtitulo_fn):
    elementos = []

    # ✅ convertir DF a Table antes de agregarlo
    df_usuarios = preparar_tabla_usuarios_conectados(df_conexiones)
    elementos.append(crear_subtitulo_fn("3. Usuarios Conectados y Tramos"))
    elementos.append(
        crear_tabla_fn(
            df_usuarios,
            encabezados=df_usuarios.columns.tolist(),
            alineacion='CENTER'
        )
    )
    elementos.append(Spacer(1, 12))

    # tabla de voltajes (esto ya estaba bien)
    elementos.extend(crear_tabla_voltajes(df_voltajes, df_regulacion, crear_tabla_fn, crear_subtitulo_fn))

    # comentario de regulación + gráfico (ya estaba bien)
    elementos.append(crear_comentario_regulacion(df_voltajes))
    elementos.append(Spacer(1, 12))
    elementos.append(crear_grafico_voltajes_pdf(df_voltajes))

    return elementos

