# app.py
import streamlit as st
import tempfile
import os
import modulo_de_regulacion_de_voltaje as mod
from utilidades_red import generar_pdf_corto  # Asegúrate que esta función ya existe

st.title("📊 Generador de Informe Corto de Red Eléctrica")

# Subir archivo Excel
archivo_excel = st.file_uploader("Selecciona el archivo Excel", type=["xlsx", "xls"])

if archivo_excel is not None:
    # Guardar temporalmente el archivo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(archivo_excel.read())
        ruta_temp = tmp.name

    st.success("Archivo cargado correctamente. Procesando...")

    try:
        # Obtener los datos necesarios desde el módulo
        (df_info, potencia_total_kva, perdida_total, capacidad_transformador,
         nodos_inicio, nodos_final, usuarios, distancias, df_voltajes, df_regulacion) = mod.obtener_datos_para_pdf_corto(ruta_temp)

        # Nombre del PDF generado
        nombre_pdf = "informe_corto_generado.pdf"

        # Generar PDF
        generar_pdf_corto(df_info, potencia_total_kva, perdida_total,
                          capacidad_transformador, nodos_inicio, nodos_final,
                          usuarios, distancias, df_voltajes, df_regulacion,
                          nombre_pdf)

        # Mostrar botón para descargar el PDF
        with open(nombre_pdf, "rb") as pdf_file:
            st.download_button(
                label="📥 Descargar Informe PDF Corto",
                data=pdf_file,
                file_name=nombre_pdf,
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"❌ Error al generar el informe: {e}")

    # Limpiar archivo temporal
    os.remove(ruta_temp)
