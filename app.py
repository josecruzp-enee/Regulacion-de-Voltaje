# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 13:18:21 2025

@author: José Nikol Cruz
"""

# -*- coding: utf-8 -*-
"""
app_streamlit.py
Interfaz Streamlit para análisis de red secundaria
"""

import streamlit as st
import pandas as pd
from modulos import datos, calculos, matrices, voltajes
from modulos.pdf_corto import generar_pdf_corto
from modulos.pdf_largo import generar_pdf_largo

st.set_page_config(page_title="Análisis de Red Secundaria", layout="wide")

st.title("⚡ Análisis de Red Secundaria")

# --- Subir archivo Excel ---
archivo_excel = st.file_uploader("📂 Sube el archivo Excel de la red secundaria", type=["xlsx"])

if archivo_excel is not None:
    # Guardar temporalmente
    ruta_excel = "datos_temp.xlsx"
    with open(ruta_excel, "wb") as f:
        f.write(archivo_excel.getbuffer())

    # Cargar datos
    df_conexiones, df_parametros, df_info, *_ = datos.cargar_datos_circuito(ruta_excel)

    st.subheader("📑 Información del Proyecto")
    st.dataframe(df_info)
    st.dataframe(df_parametros)

    # Ejecutar cálculos
    df_resultados, potencia_total, factor_coinc = calculos.realizar_calculos(
        (df_conexiones, df_parametros, df_info)
    )

    st.subheader("📊 Resultados de Cálculo")
    st.dataframe(df_resultados)
    st.write(f"**Potencia total (kVA):** {potencia_total:.2f}")
    st.write(f"**Factor de coincidencia:** {factor_coinc:.2f}")

    # Matriz de admitancia
    Y, Yrr, Y_r0, nodos, slack_index = matrices.calcular_matriz_admitancia(df_resultados)
    st.subheader("🔢 Matriz de Admitancia Y")
    st.write(pd.DataFrame(Y))

    # Voltajes
    V, df_voltajes = voltajes.calcular_voltajes_nodales(Yrr, Y_r0, slack_index, nodos, V0=240+0j)
    st.subheader("⚡ Voltajes nodales")
    st.dataframe(df_voltajes)

    # Generar PDF
    opcion = st.radio("📄 ¿Qué reporte deseas generar?", ["Corto", "Largo", "Ambos"])
    if st.button("Generar Reporte PDF"):
        if opcion == "Corto":
            ruta_pdf = generar_pdf_corto(ruta_excel)
            with open(ruta_pdf, "rb") as f:
                st.download_button("⬇️ Descargar Informe Corto", f, file_name="Informe_Corto.pdf")
        elif opcion == "Largo":
            ruta_pdf = generar_pdf_largo(ruta_excel)
            with open(ruta_pdf, "rb") as f:
                st.download_button("⬇️ Descargar Informe Largo", f, file_name="Informe_Largo.pdf")
        else:
            ruta_corto = generar_pdf_corto(ruta_excel)
            ruta_largo = generar_pdf_largo(ruta_excel)
            with open(ruta_corto, "rb") as f:
                st.download_button("⬇️ Descargar Informe Corto", f, file_name="Informe_Corto.pdf")
            with open(ruta_largo, "rb") as f:
                st.download_button("⬇️ Descargar Informe Largo", f, file_name="Informe_Largo.pdf")
