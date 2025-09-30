## -*- coding: utf-8 -*-
"""
simbolicos.py
Sección #6: Cálculo Simbólico con SymPy para Validación
"""

import sympy as sp
import sys

# ====================== Sección #6: Cálculo Simbólico ======================

# Subsección #6.1: Creación de variables y matrices simbólicas de admitancia

# Función #6.1.1: crear_matrices_y_variables_simbólicas
# Recibe:
# - Yrr: matriz numpy de admitancia entre nodos no slack
# - nodos: lista completa de nodos
# - nodo_slack: nodo slack (referencia)
# Devuelve:
# - Yrr_simb_con_cargas: matriz simbólica de admitancia con cargas y ramas
# - vector_voltajes: vector simbólico de variables voltajes nodales no slack
def crear_matrices_y_variables_simbólicas(Yrr, nodos, nodo_slack):
    # Nodos no slack
    nodos_no_slack = [n for n in nodos if n != nodo_slack]
    n = len(nodos_no_slack)
    
    # Variables simbólicas para voltajes nodales no slack
    variables = [sp.symbols(f'V{n}') for n in nodos_no_slack]
    vector_voltajes = sp.Matrix(variables)

    # Admitancias simbólicas de carga por nodo no slack
    Y_cargas = {n: sp.symbols(f'Yc_{n}') for n in nodos_no_slack}

    # Admitancias simbólicas de ramas entre nodos (solo i<j)
    Y_ramas = {}
    for i in nodos_no_slack:
        for j in nodos_no_slack:
            if i < j:
                Y_ramas[(i, j)] = sp.symbols(f'Yr_{i}{j}')

    # Inicializar matriz simbólica Yrr con ceros
    Yrr_simb_con_cargas = sp.zeros(n)

    # Construcción matriz Yrr simbólica con cargas en diagonal y ramas fuera de diagonal
    for idx_i, i in enumerate(nodos_no_slack):
        for idx_j, j in enumerate(nodos_no_slack):
            if i == j:
                suma_ramas = 0
                for k in nodos_no_slack:
                    if k != i:
                        par = (min(i, k), max(i, k))
                        if par in Y_ramas:
                            suma_ramas += Y_ramas[par]
                Yrr_simb_con_cargas[idx_i, idx_j] = suma_ramas + Y_cargas[i]
            else:
                par = (min(i, j), max(i, j))
                if par in Y_ramas:
                    Yrr_simb_con_cargas[idx_i, idx_j] = -Y_ramas[par]
                else:
                    Yrr_simb_con_cargas[idx_i, idx_j] = 0

    return Yrr_simb_con_cargas, vector_voltajes


# Función #6.1.2: resolver_sistema_simbólico
# Recibe:
# - Yrr_sym: matriz simbólica de admitancia entre nodos no slack
# - Y_r0_sym: vector simbólico de admitancia entre nodos no slack y nodo slack
# - V0: voltaje complejo del nodo slack (constante)
# Devuelve:
# - V_r_simbolico: expresión simbólica de voltajes nodales no slack
def resolver_sistema_simbólico(Yrr_sym, Y_r0_sym, V0):
    V_r_simbolico = Yrr_sym.inv() * (-Y_r0_sym * V0)
    return V_r_simbolico


# Función #6.1.3: safe_print
# Función para impresión segura en consola con posibles problemas de codificación
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'encoding'}
        print(*args, **filtered_kwargs)


# Función #6.1.4: imprimir_resultados_simbólicos
# Recibe:
# - Yrr_simb_con_cargas: matriz simbólica de admitancia con cargas y ramas
# - vector_voltajes: vector simbólico de variables voltajes nodales no slack
def imprimir_resultados_simbólicos(Yrr_simb_con_cargas, vector_voltajes):
    safe_print("🔷 Matriz simbólica de admitancias Yrr con cargas en diagonal:")
    if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('utf'):
        sp.pprint(Yrr_simb_con_cargas, use_unicode=True)
    else:
        sp.pprint(Yrr_simb_con_cargas, use_unicode=False)

    safe_print("\n🔷 Vector simbólico de voltajes incógnitos V_r:")
    if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('utf'):
        sp.pprint(vector_voltajes, use_unicode=True)
    else:
        sp.pprint(vector_voltajes, use_unicode=False)


# Función principal #6.1.5: construir_y_resolver_simbólico
# Orquesta la creación de matrices, solución y impresión
# Recibe:
# - Yrr: matriz numpy de admitancia entre nodos no slack
# - Y_r0: vector numpy de admitancia entre nodos no slack y nodo slack
# - V0: voltaje complejo del nodo slack (constante)
# - nodos: lista completa de nodos
# - nodo_slack: nodo slack (referencia)
# Devuelve:
# - V_r_simbolico: expresión simbólica de voltajes nodales no slack
# - Yrr_simb_con_cargas: matriz simbólica de admitancia con cargas y ramas
# - vector_voltajes: vector simbólico de variables voltajes nodales no slack
def construir_y_resolver_simbólico(Yrr, Y_r0, V0, nodos, nodo_slack):
    Yrr_simb_con_cargas, vector_voltajes = crear_matrices_y_variables_simbólicas(Yrr, nodos, nodo_slack)

    Yrr_sym = sp.Matrix(Yrr_simb_con_cargas)
    Y_r0_sym = sp.Matrix(Y_r0)

    V_r_simbolico = resolver_sistema_simbólico(Yrr_sym, Y_r0_sym, V0)

    imprimir_resultados_simbólicos(Yrr_simb_con_cargas, vector_voltajes)

    return V_r_simbolico, Yrr_simb_con_cargas, vector_voltajes
