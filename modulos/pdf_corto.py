# -*- coding: utf-8 -*-
"""
pdf_corto.py
Generación de informe corto en PDF para análisis de red secundaria
"""

import os
import pandas as pd
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Spacer,
    Paragraph, Table, TableStyle
)
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors

# ==== Módulos propios ====
from modulos.datos_pdf import preparar_info_proyecto, tabla_corrientes_vano
from modulos.secciones_pdf import crear_grafico_voltajes_pdf
from modulos.demanda import (
    proyectar_demanda,
    proyectar_regulacion_rv,
    resumen_regulacion_y_demanda,  # se mantiene import por si lo usas después
    crear_grafico_demanda,
)
from modulos.graficos_red import crear_grafico_nodos
from modulos.datos import cargar_datos_circuito, biblioteca_conductores
from modulos.lineas import resistencia_por_vano, reactancia_por_vano, calcular_impedancia, calcular_admitancia
from modulos.cargas import calcular_potencia_carga
from modulos.matrices import calcular_matriz_admitancia
from modulos.voltajes import calcular_voltajes_nodales, calcular_regulacion_voltaje, preparar_df_voltajes
from modulos.corrientes import calcular_corrientes, calcular_perdidas_y_proyeccion


# =====================================================
# Estilos básicos
# =====================================================
estilos = getSampleStyleSheet()
estilo_header = estilos["Normal"]
estilo_header.alignment = TA_CENTER
estilo_header.fontSize = 9


def crear_titulo(texto):
    return Paragraph(texto, ParagraphStyle(
        name="TituloCorto",
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=10,
        textColor=colors.HexColor("#003366")
    ))


def crear_subtitulo(texto):
    return Paragraph(texto, ParagraphStyle(
        name="SubtituloCorto",
        fontSize=10,
        alignment=TA_CENTER,
        spaceAfter=8,
        textColor=colors.HexColor("#006699")
    ))


def crear_tabla(df, encabezados=None, alineacion="CENTER", font_size=8, col_widths=None):
    """Tabla normal (sin formato especial en encabezados)."""
    if encabezados:
        headers = [Paragraph(str(h), estilo_header) for h in encabezados]
        data = [headers] + df.values.tolist()
    else:
        data = df.values.tolist()

    tabla = Table(data, hAlign=alineacion, colWidths=col_widths)
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), alineacion),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, -1), font_size),  # solo filas normales
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    return tabla


def crear_tabla_voltajes(df, encabezados=None, alineacion="CENTER", font_size=8):
    """Tabla de voltajes con anchos de columna fijos y encabezados simplificados."""
    if encabezados:
        headers = [Paragraph(str(h).replace(" ", "<br/>", 1), estilo_header) for h in encabezados]
        data = [headers] + df.values.tolist()
    else:
        data = df.values.tolist()

    # Ajusta manualmente cada columna
    col_widths = [0.6 * inch, 0.8 * inch, 1 * inch, 1 * inch]

    tabla = Table(data, hAlign=alineacion, colWidths=col_widths)
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), alineacion),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    return tabla


# =====================================================
# Plantilla: 3 columnas con fondo
# =====================================================
ancho_pagina, alto_pagina = landscape(letter)
margen = 20

ancho_col1 = (ancho_pagina - 4 * margen) * 0.35
ancho_col2 = (ancho_pagina - 4 * margen) * 0.35
ancho_col3 = (ancho_pagina - 4 * margen) * 0.30

frame1 = Frame(margen, margen, ancho_col1, alto_pagina - 2 * margen, id="col1")
frame2 = Frame(margen + ancho_col1 + margen, margen, ancho_col2, alto_pagina - 2 * margen, id="col2")
frame3 = Frame(margen + ancho_col1 + ancho_col2 + 2 * margen, margen, ancho_col3, alto_pagina - 2 * margen, id="col3")


def aplicar_fondo(cnv, doc, ruta_imagen):
    try:
        cnv.saveState()

        # 🔹 Tamaño CONTROLADO del logo
        logo_ancho = 90
        logo_alto = 45

        # 🔹 Posición (arriba izquierda)
        x = doc.leftMargin
        y = doc.pagesize[1] - logo_alto - 15

        imagen = ImageReader(ruta_imagen)
        cnv.drawImage(
            imagen,
            x, y,
            width=logo_ancho,
            height=logo_alto,
            mask="auto"
        )

        cnv.restoreState()
    except Exception as e:
        print(e)


plantilla = PageTemplate(
    id="tres_columnas_con_fondo",
    frames=[frame1, frame2, frame3],
    onPage=lambda cnv, doc: aplicar_fondo(cnv, doc, os.path.join(os.path.dirname(__file__), "Membrete_SMART_EDS.png")
    ),
)


# =====================================================
# Función principal: Generar PDF corto
# =====================================================
def generar_pdf_corto(ruta_excel, ruta_salida=None):
    datos = obtener_datos_para_pdf_corto(ruta_excel)

    df_info_pdf, df_parametros_pdf = preparar_info_proyecto(
        datos["df_info"], datos["df_parametros"],
        datos["factor_coinc"], datos["potencia_total_kva"]
    )

    try:
        registro_transformador = datos["df_info"].loc[
            datos["df_info"]["info"] == "Registro", "Valor"
        ].values[0]
        registro_transformador = str(registro_transformador).strip()
    except Exception:
        registro_transformador = "Sin Registro"

    if ruta_salida is None:
        carpeta_excel = os.path.dirname(ruta_excel)
        ruta_salida = os.path.join(
            carpeta_excel,
            f"Resumen de Análisis de Transformador {registro_transformador}.pdf"
        )

    # Construcción del documento
    def construir_doc(incluir_corrientes=True):
        doc = BaseDocTemplate(ruta_salida, pagesize=landscape(letter), pageTemplates=[plantilla])
        elementos = []
        elementos.append(Spacer(1, 40))
        elementos.append(crear_titulo("Análisis de Regulación de Voltaje y Cargabilidad del Transformador"))
        elementos.append(Spacer(1, 10))

        elementos.append(crear_subtitulo("Información General"))
        elementos.append(crear_tabla(df_info_pdf, df_info_pdf.columns.tolist(), "CENTER"))
        elementos.append(Spacer(1, 10))

        elementos.append(crear_subtitulo("Parámetros del Proyecto"))
        elementos.append(crear_tabla(df_parametros_pdf, df_parametros_pdf.columns.tolist(), "CENTER"))
        elementos.append(Spacer(1, 10))

        elementos.append(crear_subtitulo("Regulación de Voltaje (%)"))
        df_voltajes_fmt = preparar_df_voltajes(datos["df_voltajes"], datos["df_regulacion"])
        elementos.append(crear_tabla_voltajes(df_voltajes_fmt, df_voltajes_fmt.columns.tolist(), "CENTER"))
        elementos.append(Spacer(1, 10))

        # ✅ QUITADO: Resumen corto de Regulación y Demanda (NO imprimir)
        # (No se agrega nada aquí)

        # ====== Tabla de proyección (incluye % Reg.) ======
        elementos.append(crear_subtitulo("Demanda Proyectada"))
        elementos.append(crear_tabla(
            datos["df_proyeccion"][["Año", "Demanda (kVA)", "Pérdidas (kWh) (fmt)", "% Carga", "% Reg."]],
            ["Año", "Demanda (kVA)", "Pérdidas (kWh)", "% Carga", "% Reg."],
            "CENTER",
            font_size=7
        ))

        if incluir_corrientes:
            elementos.append(crear_subtitulo("Corrientes por Vano"))
            elementos.extend(datos["elementos_corrientes"])
            elementos.append(Spacer(1, 10))

        for img_obj in [datos["buf_grafico_voltajes"], datos["buf_grafico_demanda"], datos["grafico_nodos"]]:
            if img_obj:
                img_obj.drawWidth = 3.2 * inch
                img_obj.drawHeight = 2.3 * inch
                elementos.append(img_obj)
                elementos.append(Spacer(1, 10))

        return doc, elementos

    # 1ª construcción (con corrientes)
    doc, elementos = construir_doc(incluir_corrientes=True)
    doc.build(elementos)
    num_paginas = doc.page

    # Si excede 1 hoja → rehacer sin corrientes
    if num_paginas > 1:
        doc, elementos = construir_doc(incluir_corrientes=False)
        doc.build(elementos)
        print("⚠️ Se eliminó la tabla de corrientes para mantener el informe en 1 sola hoja.")

    print(f"✅ PDF generado: {ruta_salida}")
    return ruta_salida


# =====================================================
# Preparar datos para el PDF corto
# =====================================================
def obtener_datos_para_pdf_corto(ruta_excel):
    datos = cargar_datos_circuito(ruta_excel)

    df_conexiones = datos["df_conexiones"]
    tipo_conductor = datos["tipo_conductor"]

    tipo_conductor = str(tipo_conductor).strip()
    conductores = biblioteca_conductores()
    df_conexiones["resistencia_vano"] = df_conexiones["distancia"].apply(
        lambda d: resistencia_por_vano(conductores, tipo_conductor, d)
    )

    df_conexiones["reactancia_vano"] = df_conexiones["distancia"].apply(
        lambda d: reactancia_por_vano(conductores, tipo_conductor, d)
    )

    df_conexiones["Z_vano"] = df_conexiones.apply(calcular_impedancia, axis=1)
    df_conexiones["Y_vano"] = df_conexiones["Z_vano"].apply(calcular_admitancia)

    df_conexiones, potencia_total_kva, factor_coinc = calcular_potencia_carga(
        df_conexiones, datos["area_lote"], tipo_conductor
    )

    Y, Yrr, Y_r0, nodos, slack_index = calcular_matriz_admitancia(df_conexiones)
    V, df_voltajes = calcular_voltajes_nodales(Yrr, Y_r0, slack_index, nodos, V0=240 + 0j)
    nodo_slack = nodos[slack_index]

    df_conexiones = calcular_corrientes(df_conexiones, V)
    df_conexiones, perdida_total, proyeccion_perdidas = calcular_perdidas_y_proyeccion(df_conexiones)

    df_proyeccion = pd.DataFrame({"Año": range(0, 16)})
    df_proyeccion = proyectar_demanda(
        potencia_total_kva,
        datos["df_parametros"].set_index("Parámetro"),
        df_proyeccion,
        crecimiento=0.02,
        años=15,
        proyeccion_perdidas=proyeccion_perdidas
    )

    # ===== Regulación base (peor nodo) =====
    df_regulacion = calcular_regulacion_voltaje(V, nodos=nodos, nodo_slack=nodo_slack)

    col_reg = next((c for c in df_regulacion.columns if "regul" in c.lower()), None)
    if col_reg is None:
        raise ValueError(
            f"No encontré columna de regulación en df_regulacion. Columnas: {list(df_regulacion.columns)}"
        )

    # ✅ Necesario para poder proyectar % Reg. correctamente
    reg_base_pct = float(pd.to_numeric(df_regulacion[col_reg], errors="coerce").max())

    # ===== Proyectar %Reg estilo RV usando "Demanda (kVA)" =====
    df_proyeccion = proyectar_regulacion_rv(
        df_proyeccion,
        reg_base_pct=reg_base_pct,
        kva_base=None,              # toma kVA_base del año 0 desde "Demanda (kVA)"
        col_kva="Demanda (kVA)",
        col_out="% Reg.",
        col_out_num="Reg_pct"
    )

    # ❌ NO se calcula df_resumen_reg porque no se imprimirá
    # df_resumen_reg = resumen_regulacion_y_demanda(...)

    # ===== Corrientes por vano =====
    df_corrientes, comentario_corrientes = tabla_corrientes_vano(
        df_conexiones, perdida_total, voltaje_slack=240
    )
    elementos_corrientes = [
        crear_tabla(df_corrientes[["Tramo", "|I| (A)"]], ["Tramo", "|I| (A)"], "CENTER"),
        Paragraph(
            comentario_corrientes or "⚠️ No se generó comentario de corrientes.",
            ParagraphStyle(name="ComentarioCorrientes", fontSize=10, alignment=TA_CENTER)
        )
    ]

    return {
        **datos,
        "potencia_total_kva": potencia_total_kva,
        "perdida_total": perdida_total,
        "proyeccion_perdidas": proyeccion_perdidas,
        "df_voltajes": df_voltajes,
        "df_regulacion": df_regulacion,
        "reg_base_pct": reg_base_pct,   # opcional, por si luego lo quieres mostrar
        "df_proyeccion": df_proyeccion,
        "df_conexiones": df_conexiones,
        "elementos_corrientes": elementos_corrientes,
        "factor_coinc": factor_coinc,
        "buf_grafico_voltajes": crear_grafico_voltajes_pdf(df_voltajes),
        "buf_grafico_demanda": crear_grafico_demanda(df_proyeccion),
        "grafico_nodos": crear_grafico_nodos(
            datos["nodos_inicio"], datos["nodos_final"],
            datos["usuarios"], datos["distancias"],
            datos["capacidad_transformador"], df_conexiones
        ),
    }


if __name__ == "__main__":
    ruta_excel = os.path.join(os.path.dirname(__file__), "datos_red_secundaria.xlsx")
    generar_pdf_corto(ruta_excel)


