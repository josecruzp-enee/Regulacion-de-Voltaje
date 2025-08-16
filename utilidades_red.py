# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 12:57:23 2025

@author: José Nikol Cruz
"""


import os
import modulo_de_regulacion_de_voltaje as mod
from modulo_de_regulacion_de_voltaje import (   
    cargar_y_preparar_datos,
    calcular_flujo_carga_y_perdidas,
    calcular_regulacion_y_proyeccion
)

def ejecutar_analisis(ruta_excel):
    """
    Ejecuta el análisis completo y genera el PDF.
    Retorna un diccionario con resultados y la ruta del PDF generado.
    """
    # Llamar a la función principal de tu módulo
    mod.main_con_ruta_archivo(ruta_excel)

    # Ejemplo: si guardas df_voltajes y df_regulacion dentro del módulo
    resultados = {
        "voltajes": getattr(mod, "df_voltajes", None),
        "regulacion": getattr(mod, "df_regulacion", None),
    }

    ruta_pdf = "informe_red_electrica.pdf"
    return resultados, ruta_pdf


def obtener_datos_para_pdf_corto(ruta_excel):
    """
    Obtiene datos necesarios para generar un informe corto en PDF
    a partir de un archivo Excel.
    """
    carpeta_excel = os.path.dirname(ruta_excel)
    archivo = os.path.basename(ruta_excel)
    ruta_completa = os.path.join(carpeta_excel, archivo)

    # 1. Cargar datos
    (df_conexiones, df_parametros, df_info, tipo_conductor, area_lote, capacidad_transformador,
     usuarios, distancias, nodos_inicio, nodos_final) = cargar_y_preparar_datos(ruta_completa)

    # 2. Flujo de carga y pérdidas
    df_conexiones, perdida_total, proyeccion_perdidas, potencia_total_kva, \
    Yrr, Y_r0, slack_index, nodos, nodo_slack, factor_coinc = calcular_flujo_carga_y_perdidas(
        df_conexiones, df_parametros, area_lote
    )

    # 3. Voltajes y regulación
    df_proyeccion, df_voltajes, df_regulacion = calcular_regulacion_y_proyeccion(
        potencia_total_kva, df_parametros, Yrr, Y_r0, slack_index, nodos, nodo_slack
    )

    # 4. Retornar datos
    return (
    df_info,
    potencia_total_kva,
    perdida_total,
    capacidad_transformador,
    nodos_inicio,
    nodos_final,
    usuarios,
    distancias,
    df_regulacion
)

