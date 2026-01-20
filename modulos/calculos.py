# -*- coding: utf-8 -*-
"""
calculos.py
Orquestador: centraliza cálculos eléctricos
"""

from .cargas import calcular_potencia_carga
from .lineas import (
    resistencia_por_vano,
    reactancia_por_vano,
    calcular_impedancia,
    calcular_admitancia,
)
from .constantes import definir_constantes


# ======================
# Función orquestadora
# ======================
def realizar_calculos(datos_cargados: dict):
    try:
        df_conexiones = datos_cargados["df_conexiones"]
        df_parametros = datos_cargados["df_parametros"]
        df_info = datos_cargados["df_info"]

        area_lote = float(df_parametros.loc[df_parametros['Parámetro'] == 'area_lote', 'Valor'].values[0])
        tipo_conductor = df_parametros.loc[df_parametros['Parámetro'] == 'TipoConductor', 'Valor'].values[0]
    except Exception as e:
        raise ValueError(f"⚠️ Error al leer parámetros desde df_parametros: {e}")

    df_resultados, potencia_total_kva, factor_coinc = calcular_potencia_carga(
        df_conexiones, area_lote, tipo_conductor
    )
    return df_resultados, potencia_total_kva, factor_coinc

