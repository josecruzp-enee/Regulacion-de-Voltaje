from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import pandas as pd
import numpy as np
from reportlab.pdfbase.pdfmetrics import stringWidth
from modulo_de_regulacion_de_voltaje import obtener_datos_para_pdf_corto, preparar_info_proyecto


# Dimensiones de la página y margen
ancho_pagina, alto_pagina = landscape(letter)
margen = 20

# Proporciones de las columnas (ej: 40%, 40%, 20%)
ancho_col1 = (ancho_pagina - 4*margen) * 0.35
ancho_col2 = (ancho_pagina - 4*margen) * 0.35
ancho_col3 = (ancho_pagina - 4*margen) * 0.30

# Posición horizontal de cada columna
x_col1 = margen
x_col2 = x_col1 + ancho_col1 + margen
x_col3 = x_col2 + ancho_col2 + margen

# Crear frames
frame1 = Frame(x_col1, margen, ancho_col1, alto_pagina - 2*margen, id='col1')
frame2 = Frame(x_col2, margen, ancho_col2, alto_pagina - 2*margen, id='col2')
frame3 = Frame(x_col3, margen, ancho_col3, alto_pagina - 2*margen, id='col3')

# Función para aplicar fondo
def aplicar_fondo(cnv, doc, ruta_imagen_fondo):
    try:
        ancho, alto = doc.pagesize
        imagen_fondo = ImageReader(ruta_imagen_fondo)
        cnv.drawImage(imagen_fondo, 0, 0, width=ancho, height=alto)
    except Exception as e:
        print(f"Error al aplicar fondo: {e}")

ruta_imagen_fondo = r"C:\Users\José Nikol Cruz\Desktop\José Nikol Cruz\Python Programas\Imagen Encabezado.jpg"
# Plantilla con tres columnas y fondo
plantilla = PageTemplate(
    id='tres_columnas_con_fondo',
    frames=[frame1, frame2, frame3],
    onPage=lambda cnv, doc: aplicar_fondo(cnv, doc, "Imagen Encabezado.jpg")
)

# Crear documento
doc = BaseDocTemplate(
    "Informe_Corto.pdf",
    pagesize=landscape(letter),
    pageTemplates=[plantilla]
)

# ---------------------------
# Función para crear tablas ajustadas al contenido
# ---------------------------
def crear_tabla(df, encabezados, ancho_frame):
    """
    Crea una tabla de ReportLab donde el ancho total se ajusta al ancho del frame/columna.
    - df: DataFrame con los datos.
    - encabezados: lista con los nombres de las columnas.
    - ancho_frame: ancho disponible del frame donde irá la tabla.
    """
    estilo_texto = ParagraphStyle(
        name="Pequeno",
        fontSize=8,
        leading=8,
        alignment=TA_CENTER
    )

    # Convertir encabezados y celdas a Paragraph
    data = [[Paragraph(str(h), estilo_texto) for h in encabezados]]
    for _, row in df.iterrows():
        fila = [Paragraph(str(v), estilo_texto) for v in row]
        data.append(fila)

    # Número de columnas
    n_cols = len(encabezados)

    # Dividir ancho del frame entre columnas
    colWidths = [ancho_frame / n_cols] * n_cols

    # Crear tabla
    tabla = Table(data, colWidths=colWidths, hAlign='LEFT')
    tabla.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#d3d3d3")),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    return tabla


def generar_pdf_corto(ruta_excel, nombre_pdf="Informe_Corto.pdf"):
    # ===============================
    # 1. Obtener datos desde Excel
    # ===============================
    (df_info, df_parametros, potencia_total_kva, perdida_total, capacidad_transformador,
     nodos_inicio, nodos_final, usuarios, distancias,
     df_voltajes, df_regulacion, df_proyeccion, proyeccion_perdidas,
     grafico_nodos, buf_grafico_voltajes, buf_grafico_demanda,
     df_tabla_demanda, df_conexiones, elementos_corrientes, factor_coinc) = obtener_datos_para_pdf_corto(ruta_excel)
    print(elementos_corrientes)
    # ===============================
    # 2. Preparar tablas
    # ===============================
    # Encabezados
    encabezados_demanda = ['Año', 'Demanda (kVA)', 'Pérdidas (kWh)', '% Carga (%)']
    encabezados_regulacion = ['Nodo', 'Voltaje (p.u.)', 'Voltaje (V)', 'Regulación (%)']


    # Tabla demanda: asegurar que son float
    df_tabla_demanda['Demanda (kVA)'] = df_tabla_demanda['Demanda (kVA)'].astype(float)
    df_tabla_demanda['Pérdidas (kWh)'] = df_tabla_demanda['Pérdidas (kWh)'].apply(lambda x: float(str(x).replace(',', '')))
    df_tabla_demanda['% Carga (%)'] = df_tabla_demanda['% Carga (%)']

    # Tabla regulación: valores reales y normalizados
    df_regulacion_real = pd.DataFrame({
        'Nodo': df_regulacion['Nodo'].apply(lambda x: int(np.real(x))),
        'Voltaje (p.u.)': df_regulacion['Voltaje (p.u.)'].apply(lambda x: round(abs(x)/240, 3)),
        'Voltaje (V)': df_regulacion['Voltaje Absoluto (V)'].apply(lambda x: round(abs(x)/240, 2)),
        'Regulación (%)': df_regulacion['Regulación (%)'].apply(lambda x: round(np.real(x), 2))
    })

    # Info del proyecto y parámetros preparados
    df_info_pdf, df_parametros_pdf = preparar_info_proyecto(df_info, df_parametros, factor_coinc, potencia_total_kva)

    # ===============================
    # 3. Crear PDF
    # ===============================
    doc = BaseDocTemplate(nombre_pdf, pagesize=landscape(letter), pageTemplates=[plantilla])
    elementos = []

    estilo_pequeno = ParagraphStyle(name="Pequeno", fontSize=8, leading=10)

    # 3.1 Título
    # Estilo para el título
    estilo_titulo = ParagraphStyle(
    name='TituloGrande',
    fontSize=14,       # tamaño de la letra
    leading=16,        # espacio entre líneas (ligeramente mayor que fontSize)
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
    )

    # Estilo para el subtítulo
    estilo_subtitulo = ParagraphStyle(
    name='Subtitulo',
    fontSize=12,       # tamaño de la letra
    leading=14,        # espacio entre líneas (ligeramente mayor que fontSize)
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
    )



    elementos.append(Spacer(1, 40))
    elementos.append(Paragraph("Análisis de Regulación de Voltaje de Red Secundaria y Cargabilidad de Transformador", estilo_titulo))
    elementos.append(Spacer(1, 14))
    
    # 3.2 Información General
    elementos.append(Paragraph("Información General", estilo_subtitulo))
    tabla_info = crear_tabla(
    df_info_pdf,
    df_info_pdf.columns.tolist(),
    ancho_frame=frame1._width
    )
    elementos.append(tabla_info)
    elementos.append(Spacer(1, 14))
    
    
    # Tabla de parámetros del proyecto
    elementos.append(Paragraph("Parámetros del Proyecto", estilo_subtitulo))
    tabla_parametros = crear_tabla(
    df_parametros_pdf,
    df_parametros_pdf.columns.tolist(),
    ancho_frame=frame1._width  # Ajusta al ancho de la columna
    )
    elementos.append(tabla_parametros)
    elementos.append(Spacer(1, 14))

    # 3.3 Regulación de Voltaje
    elementos.append(Paragraph("Regulación de Voltaje (%)", estilo_subtitulo))

    tabla_regulacion = crear_tabla(
    df_regulacion_real,
    encabezados_regulacion,
    ancho_frame=frame1._width  # o frame2._width según la columna donde quieras ponerla
    )
    elementos.append(tabla_regulacion)
    elementos.append(Spacer(1, 60))
    
    

    # 3.4 Demanda proyectada
    elementos.append(Paragraph("Demanda Proyectada", estilo_subtitulo))
    tabla_demanda = crear_tabla(
    df_tabla_demanda,
    encabezados_demanda,
    ancho_frame=frame1._width  # o frame2._width si va en otra columna
    )
    elementos.append(tabla_demanda)
    elementos.append(Spacer(1, 10))


    # Limpiar elementos de corrientes para eliminar títulos duplicados
    elementos_corrientes_limpios = [
    e for e in elementos_corrientes 
    if not (hasattr(e, 'getPlainText') and "Corriente" in e.getPlainText())
    ]

    # 3.5 Corrientes por vano
    elementos.append(Paragraph("Corriente de Vanos", estilo_subtitulo))
    # Agregamos la tabla de corrientes ya limpia
    elementos.extend(elementos_corrientes_limpios)
    elementos.append(Spacer(1, 12))

    # 3.6 Gráficos
    for buf_img in [buf_grafico_voltajes, buf_grafico_demanda, grafico_nodos]:
        img = Image(buf_img)
        img.drawHeight = 2.5*inch
        img.drawWidth = 2.5*inch
        elementos.append(img)
        elementos.append(Spacer(1, 4))

    # ===============================
    # 4. Generar PDF
    # ===============================
    doc.build(elementos)
    print(f"PDF generado: {nombre_pdf}")



# --------------------------- 
# Uso mínimo
# --------------------------- 
if __name__ == "__main__":
    ruta_excel = r"C:\Users\José Nikol Cruz\Desktop\José Nikol Cruz\Python Programas\datos_circuito.xlsx"
    generar_pdf_corto(ruta_excel)


