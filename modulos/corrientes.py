# -*- coding: utf-8 -*-
"""
corrientes.py
Sección #5: Cálculo de Corrientes, Pérdidas y Proyección
"""

import pandas as pd
import numpy as np


# ====================== Sección #5: Cálculo de Corrientes, Pérdidas y Proyección ======================

# ---------------------------------------------------------------------------
# Subsección #5.1: Cálculo de corrientes por vano
# ---------------------------------------------------------------------------

# Función #5.1.1: calcular_corrientes
# Calcula las corrientes en cada vano basándose en los voltajes nodales y las impedancias de vano.
# Recibe:
# - df: DataFrame con columnas 'nodo_inicial', 'nodo_final', 'Z_vano'
# - V: array o lista con voltajes nodales complejos
# Devuelve:
# - df con columna nueva 'I_vano' que contiene la corriente compleja por vano
import numpy as np
import pandas as pd

def calcular_corrientes(df: pd.DataFrame, V: np.ndarray, V_base: float = 240.0) -> pd.DataFrame:
    """
    - Tramo real (ni!=nf): I_vano = (Vni - Vnf) / Z_vano
    - Carga en nodo (ni==nf): I_vano = S/Vn (usa 'kva_total' o 'kva')

    Nota: para ni==nf, Z_vano no aplica.
    """
    df = df.copy()

    # qué columna usar para kVA
    col_kva = "kva_total" if "kva_total" in df.columns else ("kva" if "kva" in df.columns else None)
    if col_kva is not None:
        df[col_kva] = pd.to_numeric(df[col_kva], errors="coerce").fillna(0.0).astype(float)

    I_list = []

    for _, row in df.iterrows():
        ni = int(row["nodo_inicial"])
        nf = int(row["nodo_final"])

        # Caso carga en nodo: 1-1, 2-2, etc.
        if ni == nf:
            if col_kva is None:
                # si no hay kVA, no se puede calcular corriente de carga
                I_list.append(0.0)
                continue

            S_kva = float(row.get(col_kva, 0.0) or 0.0)
            Vn = abs(V[ni - 1]) if (len(V) >= ni and abs(V[ni - 1]) > 0) else float(V_base)

            I_list.append((S_kva * 1000.0) / Vn if (S_kva > 0 and Vn > 0) else 0.0)
            continue

        # Caso tramo real: corriente por impedancia
        Z = row.get("Z_vano", 0)
        if Z == 0 or Z == 0.0:
            I_list.append(0 + 0j)
        else:
            I_list.append((V[ni - 1] - V[nf - 1]) / Z)

    df["I_vano"] = I_list
    return df


# ---------------------------------------------------------------------------
# Subsección #5.1: Pérdidas y proyección
# ---------------------------------------------------------------------------

# Función #5.1.2: calcular_perdidas_y_proyeccion
# Calcula las pérdidas en cada vano y proyecta las pérdidas anuales a 15 años con crecimiento.
# Recibe:
# - df: DataFrame con columnas 'I_vano' y 'resistencia_vano'
# Devuelve:
# - df con columna adicional 'P_perdida' (pérdidas en Watts)
# - perdida_total: pérdidas totales anuales en kWh
# - proyeccion: lista con proyección de pérdidas para 15 años (crecimiento anual 2%)
def calcular_perdidas_y_proyeccion(df, LF=0.4, crecimiento=0.02):
    P_perdidas = []
    for _, row in df.iterrows():
        I = row["I_vano"]
        R_loop = float(row["resistencia_vano"].real)  # Ω lazo
        P_perdidas.append(float((abs(I) ** 2) * R_loop))  # W

    df = df.copy()
    df["P_perdida"] = P_perdidas

    P_peak_w = sum(P_perdidas)  # asumimos que I corresponde a pico
    k = 0.2 * LF + 0.8 * (LF ** 2)

    perdida_anual_kwh = P_peak_w * 8760 * k / 1000.0

    # ✅ pérdidas crecen ~ (1+g)^(2i)
    proyeccion = [perdida_anual_kwh * ((1 + crecimiento) ** (2 * i)) for i in range(0, 15)]

    return df, perdida_anual_kwh, proyeccion





    

