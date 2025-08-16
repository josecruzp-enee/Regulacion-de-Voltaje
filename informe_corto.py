# informe_corto.py

from modulo_de_regulacion_de_voltaje import (
    cargar_datos_circuito, resistencia_por_vano, reactancia_por_vano_geometrica,
    calcular_impedancia, calcular_admitancia, calcular_potencia_carga,
    calcular_matriz_admitancia, calcular_voltajes_nodales,
    calcular_corrientes, calcular_perdidas_y_proyeccion, bibloteca_conductores,
    factor_coincidencia
)

from módulo_de_regulacion_de_voltaje import calcular_regulacion_y_proyeccion

import pandas as pd
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors


def generar_pdf_resumen(potencia_total_kva, perdida_total, capacidad_transformador,
                        nodos_inicio, nodos_final, usuarios, distancias, df_regulacion):
    doc = SimpleDocTemplate("informe_corto.pdf", pagesize=landscape(letter))
    estilos = getSampleStyleSheet()
    elementos = []

    estilo_titulo = ParagraphStyle('titulo', fontSize=20, alignment=TA_CENTER, spaceAfter=20, fontName='Helvetica-Bold')
    elementos.append(Paragraph("Resumen del Informe de Red Eléctrica", estilo_titulo))

    resumen_texto = f"""
    <b>Potencia total instalada:</b> {potencia_total_kva:.2f} kVA<br/>
    <b>Pérdidas totales:</b> {perdida_total:.2f} kW<br/>
    <b>Capacidad del transformador:</b> {capacidad_transformador} kVA<br/>
    """
    elementos.append(Paragraph(resumen_texto, estilos['Normal']))
    elementos.append(Spacer(1, 12))

    # Tabla 1
    tabla1_data = [['Nodo Inicio', 'Nodo Final', 'Usuarios', 'Distancia (m)']]
    for i in range(len(nodos_inicio)):
        tabla1_data.append([
            str(nodos_inicio[i]), str(nodos_final[i]),
            str(usuarios[i]), f"{distancias[i]:.2f}"
        ])
    tabla1 = Table(tabla1_data, colWidths=[60] * 4)
    tabla1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elementos.append(tabla1)
    elementos.append(Spacer(1, 10))

    # Tabla 2 - Regulación
    if df_regulacion is not None:
        tabla2_data = [list(df_regulacion.columns)]
        for row in df_regulacion.itertuples(index=False):
            tabla2_data.append([str(cell) for cell in row])
        tabla2 = Table(tabla2_data, colWidths=[80] * len(df_regulacion.columns))
        tabla2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        elementos.append(tabla2)

    doc.build(elementos)
    print("✅ PDF corto generado: informe_corto.pdf")


def generar_datos_y_pdf_corto():
    archivo = 'datos_circuito.xlsx'
    df_conexiones, df_parametros, df_info, tipo_conductor, area_lote, capacidad_transformador, proyecto_numero, proyecto_nombre, transformador_numero, usuarios, distancias, nodos_inicio, nodos_final = cargar_datos_circuito(archivo)

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
    df_conexiones, perdida_total, _ = calcular_perdidas_y_proyeccion(df_conexiones)

    _, _, df_regulacion = calcular_regulacion_y_proyeccion(potencia_total_kva, df_parametros, Yrr, Y_r0, slack_index, nodos, nodos[slack_index])

    generar_pdf_resumen(potencia_total_kva, perdida_total, capacidad_transformador,
                        nodos_inicio, nodos_final, usuarios, distancias, df_regulacion)


if __name__ == "__main__":
    generar_datos_y_pdf_corto()

