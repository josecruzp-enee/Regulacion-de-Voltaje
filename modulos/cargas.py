# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:14:12 2025

@author: José Nikol Cruz
"""

# -*- coding: utf-8 -*-
"""
cargas.py
Cálculos de cargas, potencias y admitancias
"""

import numpy as np
from .datos import calcular_kva_por_area, factor_coincidencia, biblioteca_conductores
from .lineas import resistencia_por_vano, reactancia_por_vano, calcular_impedancia, calcular_admitancia


def calcular_potencias(df_conexiones, kva_usuario, factor_coinc, fp=0.9):
    df = df_conexiones.copy()
    df['kva'] = df['usuarios'] * kva_usuario * factor_coinc
    df['P'] = df['kva'] * fp
    df['Q'] = df['kva'] * np.sin(np.arccos(fp))
    df['S_compleja'] = df['P'] + 1j * df['Q']
    df['S_VA'] = df['S_compleja'] * 1000
    return df

def calcular_impedancia_admitancia(df, V_nom=240):
    df['Z_carga'] = (V_nom ** 2) / np.conj(df['S_VA'])
    df['Y_carga'] = 1 / df['Z_carga']
    df['G_carga'] = df['Y_carga'].apply(lambda y: y.real)
    df['B_carga'] = df['Y_carga'].apply(lambda y: y.imag)
    return df

def calcular_carga_por_usuario(df, factor_coinc, kva_usuario, fp=0.9, V_nom=240):
    df = calcular_potencias(df, kva_usuario, factor_coinc, fp)
    return calcular_impedancia_admitancia(df, V_nom)

def calcular_potencia_carga(tabla_potencia, area_m2, tipo_conductor, fp=0.9, V_nom=240):
    conductores = biblioteca_conductores()

    # Usuarios normales
    total_usuarios_normales = tabla_potencia['usuarios'].sum()
    kva_usuario_normal = calcular_kva_por_area(area_m2)

    # Usuarios especiales
    if 'usuarios_especiales' in tabla_potencia.columns and 'area_especial' in tabla_potencia.columns:
        tabla_potencia['kva_especial'] = tabla_potencia.apply(
            lambda row: row['usuarios_especiales'] * calcular_kva_por_area(row['area_especial'])
            if row['usuarios_especiales'] > 0 else 0,
            axis=1
        )
    else:
        tabla_potencia['kva_especial'] = 0

    total_usuarios = total_usuarios_normales + tabla_potencia['usuarios_especiales'].sum()
    factor_coinc = factor_coincidencia(total_usuarios)

    # kVA total por tramo
    tabla_potencia['kva_normal'] = tabla_potencia['usuarios'] * kva_usuario_normal
    tabla_potencia['kva_total'] = (tabla_potencia['kva_normal'] + tabla_potencia['kva_especial']) * factor_coinc

    # Potencias
    tabla_potencia['P'] = tabla_potencia['kva_total'] * fp
    tabla_potencia['Q'] = tabla_potencia['kva_total'] * np.sin(np.arccos(fp))
    tabla_potencia['S_compleja'] = tabla_potencia['P'] + 1j * tabla_potencia['Q']
    tabla_potencia['S_VA'] = tabla_potencia['S_compleja'] * 1000

    # Impedancias
    tabla_potencia = calcular_impedancia_admitancia(tabla_potencia, V_nom)
    tabla_potencia['resistencia_vano'] = tabla_potencia.apply(
        lambda row: resistencia_por_vano(conductores, tipo_conductor, row['distancia']), axis=1
    )
    tabla_potencia['reactancia_vano'] = tabla_potencia.apply(
        lambda row: reactancia_por_vano_geometrica(conductores, tipo_conductor, row['distancia']), axis=1    
    )
    
    tabla_potencia['Z_vano'] = tabla_potencia.apply(calcular_impedancia, axis=1)
    tabla_potencia['Y_vano'] = tabla_potencia['Z_vano'].apply(calcular_admitancia)
    tabla_potencia['kva'] = tabla_potencia['kva_total']

    potencia_total_kva = float(tabla_potencia['kva_total'].sum())
    return tabla_potencia, potencia_total_kva, factor_coinc


