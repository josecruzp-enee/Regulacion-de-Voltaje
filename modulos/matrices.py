# -*- coding: utf-8 -*-
"""
matrices.py
Sección #3: Matrices y Nodos
"""

import numpy as np
import pandas as pd

# ====================== Sección #3: Matrices y Nodos ======================

# Subsección #3.1: Manejo de nodos y construcción de matrices

# Función #3.1.1: obtener_nodos_e_indices
# Obtiene la lista ordenada de nodos y un diccionario que mapea cada nodo a su índice en la matriz.
# Recibe: df (DataFrame con columnas 'nodo_inicial' y 'nodo_final')
# Devuelve: lista nodos (ordenada), diccionario {nodo: índice}
def obtener_nodos_e_indices(df):
    nodos = sorted(set(df['nodo_inicial']).union(set(df['nodo_final'])))
    indice_nodos = {nodo: i for i, nodo in enumerate(nodos)}
    return nodos, indice_nodos


# Función #3.1.2: construir_matriz_admitancia
# Construye la matriz nodal de admitancia Y a partir del DataFrame con admitancias de vanos y cargas.
# Recibe: df (DataFrame con columnas 'nodo_inicial', 'nodo_final', 'Y_vano', 'Y_carga'), nodos (lista), indice_nodos (dicc)
# Devuelve: matriz Y (numpy ndarray)
def construir_matriz_admitancia(df, nodos, indice_nodos):
    n = len(nodos)
    Y = np.zeros((n, n), dtype=complex)

    for _, row in df.iterrows():
        i = indice_nodos[row['nodo_inicial']]
        j = indice_nodos[row['nodo_final']]
        Y_vano = row['Y_vano']

        # Añadir admitancias de vano
        Y[i, i] += Y_vano
        Y[j, j] += Y_vano
        Y[i, j] -= Y_vano
        Y[j, i] -= Y_vano

        # Añadir admitancia de carga en nodo j si existe
        if 'Y_carga' in df.columns and not pd.isna(row['Y_carga']):
            Y[j, j] += row['Y_carga']

    return Y


# Función #3.1.3: extraer_submatrices
# Extrae la submatriz Yrr (sin nodo slack), vector Y_r0, define nodo slack y su índice.
# Recibe: matriz Y (numpy ndarray), lista nodos
# Devuelve: Yrr (numpy ndarray), Y_r0 (numpy ndarray columna), nodo_slack (valor), slack_index (int)
def extraer_submatrices(Y, nodos):
    nodo_slack = nodos[0]  # Primer nodo es nodo slack
    slack_index = 0

    indices_no_slack = [i for i in range(len(nodos)) if i != slack_index]

    Yrr = Y[np.ix_(indices_no_slack, indices_no_slack)]
    Y_r0 = Y[indices_no_slack, slack_index]

    return Yrr, Y_r0, nodo_slack, slack_index


# Función #3.1.4: calcular_matriz_admitancia
# Función principal para obtener nodos, construir matriz Y y extraer submatrices necesarias para cálculo de voltajes.
# Recibe: df (DataFrame con datos de vanos y cargas)
# Devuelve: matriz Y completa, Yrr, Y_r0, lista nodos, índice slack
def calcular_matriz_admitancia(df):
    nodos, indice_nodos = obtener_nodos_e_indices(df)
    Y = construir_matriz_admitancia(df, nodos, indice_nodos)
    Yrr, Y_r0, nodo_slack, slack_index = extraer_submatrices(Y, nodos)
    return Y, Yrr, Y_r0, nodos, slack_index

