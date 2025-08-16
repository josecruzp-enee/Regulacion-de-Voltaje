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
def obtener_datos_para_pdf_corto(ruta_excel):
    import os
    from .tus_funciones import (  # Ajusta estos imports si están en otros archivos
        cargar_y_preparar_datos,
        calcular_flujo_carga_y_perdidas,
        calcular_regulacion_y_proyeccion
    )

    carpeta_excel = os.path.dirname(ruta_excel)
    os.chdir(carpeta_excel)
    archivo = os.path.basename(ruta_excel)
    
    # 1. Cargar datos y preparar variables
    (df_conexiones, df_parametros, df_info, tipo_conductor, area_lote, capacidad_transformador,
     usuarios, distancias, nodos_inicio, nodos_final) = cargar_y_preparar_datos(archivo)   
    
    # 2. Cálculo de flujo de carga y pérdidas
    df_conexiones, perdida_total, proyeccion_perdidas, potencia_total_kva, \
    Yrr, Y_r0, slack_index, nodos, nodo_slack, factor_coinc = calcular_flujo_carga_y_perdidas(df_conexiones, df_parametros, area_lote)
    
    # 3. Voltajes y regulación
    df_proyeccion, df_voltajes, df_regulacion = calcular_regulacion_y_proyeccion(
        potencia_total_kva, df_parametros, Yrr, Y_r0, slack_index, nodos, nodo_slack
    )

    # 4. Retornar datos que necesita el PDF corto
    return (
        df_info, 
        potencia_total_kva, 
        perdida_total, 
        capacidad_transformador,
        nodos_inicio, 
        nodos_final, 
        usuarios, 
        distancias, 
        df_voltajes, 
        df_regulacion
    )


    return resultados, ruta_pdf
