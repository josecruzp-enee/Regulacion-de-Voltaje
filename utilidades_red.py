# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 12:57:23 2025

@author: José Nikol Cruz
"""

import modulo_de_regulacion_de_voltaje as mod

def ejecutar_analisis(ruta_excel):
    # Aquí usas las funciones de tu módulo para procesar el archivo Excel
    # Ejemplo simplificado:
    # Llama a la función principal de tu módulo que hace todos los cálculos
    # y genera el PDF

    # Dependiendo de tu módulo, adapta esta parte:
    mod.main_con_ruta_archivo(ruta_excel)  # O el nombre que le diste a tu función

    # Luego, si tienes resultados que mostrar (DataFrames), devuélvelos:
    # Aquí solo pongo ejemplos, adapta según tu caso
    resultados = {
        "voltajes": mod.df_voltajes,
        "regulacion": mod.df_regulacion,
    }

    # Y la ruta del PDF generado
    ruta_pdf = "informe_red_electrica.pdf"

    return resultados, ruta_pdf