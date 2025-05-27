
# app.py
import streamlit as st
from utilidades_red import ejecutar_analisis
import os

st.set_page_config(page_title="Análisis Red Eléctrica", layout="centered")
st.title("Análisis de Red Secundaria de Distribución")

st.markdown("""
Esta aplicación permite subir un archivo Excel con la información de un circuito de distribución monofásico,
realizar el análisis de flujo de carga, calcular regulación, pérdidas eléctricas, cargabilidad del transformador y generar un informe en PDF.
""")

archivo = st.file_uploader("Sube tu archivo de datos (datos_circuito.xlsx)", type=["xlsx"])

if archivo:
    with open("datos_circuito.xlsx", "wb") as f:
        f.write(archivo.read())

    st.success("Archivo cargado correctamente. Ejecutando análisis...")

    try:
        resultados, mensaje_pdf = ejecutar_analisis("datos_circuito.xlsx")

        st.subheader("✅ Resultados principales")
        st.dataframe(resultados["voltajes"])
        st.dataframe(resultados["regulacion"])

        st.subheader("📥 Descargar informe PDF")
        with open(mensaje_pdf, "rb") as f:
            st.download_button("Descargar PDF", f, file_name="informe_red_electrica.pdf")

    except Exception as e:
        st.error(f"Ocurrió un error al ejecutar el análisis: {e}")
