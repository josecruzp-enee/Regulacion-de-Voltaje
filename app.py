# -*- coding: utf-8 -*-
"""
app_secundaria.py
App principal para análisis de red secundaria
"""

import os
import pandas as pd
from colorama import init
from modulos import datos, calculos, matrices, voltajes, simbolicos
from modulos.pdf_corto import generar_pdf_corto
from modulos.pdf_largo import generar_pdf_largo

# Inicializar colorama
init(autoreset=True)

def main():
    ruta_excel = "datos_red_secundaria.xlsx"

    # Crear carpeta "Reportes" en la misma ruta que el Excel
    carpeta_reportes = os.path.join(os.path.dirname(ruta_excel), "Reportes")
    os.makedirs(carpeta_reportes, exist_ok=True)

    # 1) Cargar datos desde Excel
    datos_cargados = datos.cargar_datos_circuito(ruta_excel)

    # 2) Ejecutar cálculos eléctricos
    df_resultados, potencia_total, factor_coinc = calculos.realizar_calculos(datos_cargados)

    print("\n=== Resultados del análisis ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df_resultados)
    print(f"\nPotencia total (kVA): {potencia_total:.2f}")
    print(f"Factor de coincidencia aplicado: {factor_coinc:.2f}")

    # 3) Construir matrices
    Y, Yrr, Y_r0, nodos, slack_index = matrices.calcular_matriz_admitancia(df_resultados)
    print("\n=== Matriz de Admitancia (Y) ===")
    print(Y)

    # 4) Resolver voltajes con Numpy
    print("\n=== Voltajes numéricos (con Numpy) ===")
    V, df_voltajes = voltajes.calcular_voltajes_nodales(
        Yrr, Y_r0, slack_index, nodos, V0=240 + 0j
    )
    print(df_voltajes)

    # 5) Validación simbólica (opcional)
    '''
    print("\n=== Voltajes simbólicos (con Sympy) ===")
    V_r_simbolico, Yrr_simb, vector_voltajes = simbolicos.construir_y_resolver_simbólico(
        Yrr, Y_r0, 240 + 0j, nodos, nodos[slack_index]
    )
    print(V_r_simbolico)
    '''

    # 6) Preguntar qué reporte generar
    print("\n=== Generación de reportes PDF ===")
    opcion = input("¿Qué reporte deseas generar? (corto / largo / ambos): ").strip().lower()

    if opcion == "corto":
        ruta_pdf = generar_pdf_corto(ruta_excel)
        print(f"✅ Informe corto generado: {ruta_pdf}")
    elif opcion == "largo":
        ruta_pdf = generar_pdf_largo(ruta_excel)
        print(f"✅ Informe largo generado: {ruta_pdf}")
    elif opcion == "ambos":
        ruta_corto = generar_pdf_corto(ruta_excel)
        ruta_largo = generar_pdf_largo(ruta_excel)
        print(f"✅ Se generaron ambos reportes:\n- {ruta_corto}\n- {ruta_largo}")
    else:
        print("⚠️ Opción no válida. No se generó ningún reporte.")


if __name__ == "__main__":
    main()
