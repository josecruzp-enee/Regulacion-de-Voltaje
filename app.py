# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 12:38:48 2025

@author: José Nikol Cruz
"""

import streamlit as st
import modulo_de_regulacion_de_voltaje as mod  # tu módulo con funciones
import tempfile
import os

st.title("Generador de Informe de Análisis de Regulación en red secundaria desde Excel")

# Subir archivo Excel
uploaded_file = st.file_uploader("Selecciona el archivo Excel", type=["xls", "xlsx"])

if uploaded_file is not None:
    # Guardar temporalmente el archivo subido
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        ruta_excel = tmp_file.name

    st.write("Archivo recibido. Procesando...")

    try:
        # Aquí llamamos a la función principal de tu módulo para procesar y generar PDF
        # Ajusta esto según cómo tengas tu función para procesar y generar el PDF:
        mod.main_con_ruta_archivo(ruta_excel)  # o la función que hayas definido que tome ruta

        # El PDF se genera en la carpeta actual, asumimos el nombre fijo, por ej:
        nombre_pdf = "informe_red_electrica.pdf"
        
        # Mostrar botón para descargar PDF
        with open(nombre_pdf, "rb") as pdf_file:
            PDFbyte = pdf_file.read()
            st.download_button(label="Descargar Informe PDF",
                               data=PDFbyte,
                               file_name=nombre_pdf,
                               mime='application/pdf')

    except Exception as e:
        st.error(f"Ocurrió un error al generar el PDF: {e}")

    # Borrar archivo temporal después del proceso
    os.remove(ruta_excel)

if __name__ == "__main__":
    try:
        # Todo tu código de Streamlit aquí
        st.title("Generador de Informe de Análisis de Regulación en red secundaria desde Excel")

        uploaded_file = st.file_uploader("Selecciona el archivo Excel", type=["xls", "xlsx"])

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                tmp_file.write(uploaded_file.read())
                ruta_excel = tmp_file.name

            st.write("Archivo recibido. Procesando...")

            try:
                mod.main_con_ruta_archivo(ruta_excel)

                nombre_pdf = "informe_red_electrica.pdf"
                with open(nombre_pdf, "rb") as pdf_file:
                    PDFbyte = pdf_file.read()
                    st.download_button(label="Descargar Informe PDF",
                                       data=PDFbyte,
                                       file_name=nombre_pdf,
                                       mime='application/pdf')
            except Exception as e:
                st.error(f"Ocurrió un error al generar el PDF: {e}")

            os.remove(ruta_excel)

    except FileNotFoundError:
        # Ignoramos el aviso de /tmp/app.py
        pass

