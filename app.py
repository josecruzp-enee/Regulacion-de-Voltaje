# app.py
import streamlit as st
import modulo_de_regulacion_de_voltaje as mod
import tempfile
import os

st.title("Generador de Informes de Red Eléctrica")

# Subir archivo Excel
uploaded_file = st.file_uploader("Selecciona el archivo Excel", type=["xls", "xlsx"])

# Selector de tipo de informe
tipo_informe = st.selectbox("¿Qué tipo de informe deseas generar?", ["Completo", "Corto"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        ruta_excel = tmp_file.name

    try:
        if tipo_informe == "Completo":
            st.write("Generando informe completo...")
            mod.main_con_ruta_archivo(ruta_excel)
            nombre_pdf = "informe_red_electrica.pdf"

        else:
            st.write("Generando informe corto...")
            datos = mod.obtener_datos_para_pdf_corto(ruta_excel)
            mod.generar_pdf_corto(*datos)
            nombre_pdf = "informe_corto.pdf"

        # Descargar
        with open(nombre_pdf, "rb") as pdf_file:
            PDFbyte = pdf_file.read()
            st.download_button(label="Descargar Informe PDF", data=PDFbyte, file_name=nombre_pdf, mime='application/pdf')

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

    os.remove(ruta_excel)
