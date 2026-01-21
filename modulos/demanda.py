# -*- coding: utf-8 -*-
"""
demanda.py
Módulo para proyección de demanda, pérdidas y %Reg (modo RV) en redes secundarias
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.platypus import Image


# =====================================================
# Helpers
# =====================================================
def _to_float_kva_col(df: pd.DataFrame, col_kva: str) -> pd.Series:
    """
    Convierte una columna de kVA que puede venir como:
    - numérica
    - string con comas (ej: "1,234.56")
    a float seguro.
    """
    s = df[col_kva].astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


# =====================================================
# Proyección de demanda y pérdidas
# =====================================================
def proyectar_demanda(
    potencia_total_kva,
    df_parametros: pd.DataFrame,
    df_proyeccion: pd.DataFrame,
    crecimiento: float = 0.02,
    años: int = 15,
    proyeccion_perdidas=None
) -> pd.DataFrame:
    """
    Calcula la proyección de demanda y pérdidas desde año 0 hasta año `años`.

    Devuelve df_proyeccion con columnas:
    - Año
    - Demanda_kva (numérico)
    - % Carga (%) (numérico)
    - Pérdidas (kWh) (numérico)
    - Demanda (kVA) (texto para reporte)
    - % Carga (texto para reporte)
    - Pérdidas (kWh) (fmt) (texto para reporte)
    """

    # Años 0..N
    df_proyeccion["Año"] = range(0, int(años) + 1)

    potencia_total_kva = float(potencia_total_kva)
    crecimiento = float(crecimiento)

    # Demanda proyectada
    df_proyeccion["Demanda_kva"] = [
        potencia_total_kva * ((1.0 + crecimiento) ** i)
        for i in range(0, int(años) + 1)
    ]

    # Capacidad del transformador
    capacidad_transformador = float(df_parametros.loc["capacidad_transformador", "Valor"])

    # % carga
    df_proyeccion["% Carga (%)"] = (df_proyeccion["Demanda_kva"] / capacidad_transformador) * 100.0

    # Pérdidas
    if proyeccion_perdidas is not None and len(proyeccion_perdidas) > 0:
        perdidas_ajustadas = list(proyeccion_perdidas) + [proyeccion_perdidas[-1]]
        df_proyeccion["Pérdidas (kWh)"] = perdidas_ajustadas[: len(df_proyeccion)]
    else:
        df_proyeccion["Pérdidas (kWh)"] = np.nan

    # Formatos para reporte
    df_proyeccion["Demanda (kVA)"] = df_proyeccion["Demanda_kva"].map(lambda x: f"{x:,.2f}")
    df_proyeccion["% Carga"] = df_proyeccion["% Carga (%)"].map(lambda x: f"{x:.2f}%")
    df_proyeccion["Pérdidas (kWh) (fmt)"] = df_proyeccion["Pérdidas (kWh)"].map(
        lambda x: f"{x:,.2f}" if pd.notna(x) else "-"
    )

    return df_proyeccion


# =====================================================
# Proyección de %Reg (modo RV-2002)
# =====================================================
def proyectar_regulacion_rv(
    df_proyeccion: pd.DataFrame,
    *,
    reg_base_pct: float,
    kva_base: float | None = None,
    col_kva: str = "Demanda (kVA)",  # <- USAMOS ESTA, CON COMAS
    col_out: str = "% Reg.",
    col_out_num: str = "Reg_pct"
) -> pd.DataFrame:
    """
    %Reg(t) = %Reg_base * kVA(t)/kVA_base

    - reg_base_pct: regulación base máxima (peor nodo) en año 0 (%)
    - kva_base: demanda base (kVA). Si None, se toma del año 0 de col_kva.
    - col_kva: puede ser string con comas.
    - col_out_num: numérica (útil para buscar máximos/año crítico)
    - col_out: formateada 'x.xx%'
    """

    reg_base_pct = float(reg_base_pct)

    # kVA(t) desde columna con comas
    kva_t = _to_float_kva_col(df_proyeccion, col_kva)

    # kVA_base
    if kva_base is None:
        kva_base = float(kva_t.iloc[0]) if len(kva_t) else 1.0

    kva_base = float(kva_base) if float(kva_base) != 0 else 1.0

    # cálculo
    df_proyeccion[col_out_num] = reg_base_pct * (kva_t / kva_base)

    # formato reporte
    df_proyeccion[col_out] = df_proyeccion[col_out_num].map(lambda x: f"{x:.2f}%")

    return df_proyeccion


# =====================================================
# Resumen "bonito" para PDF corto
# =====================================================
def resumen_regulacion_y_demanda(
    df_proyeccion: pd.DataFrame,
    *,
    años_objetivo: int = 15,
    col_dem_fmt: str = "Demanda (kVA)",
    col_carga_num: str = "% Carga (%)",
    col_reg_num: str = "Reg_pct",
    reg_base_pct: float | None = None
) -> pd.DataFrame:
    """
    Devuelve un DataFrame pequeño para el PDF corto con:
    - Reg base (año 0)
    - Reg año N
    - Reg máxima y año crítico
    - Demanda año 0 y año N
    - % carga año N (num)
    """

    if "Año" not in df_proyeccion.columns:
        raise ValueError("df_proyeccion no tiene columna 'Año'.")

    # asegurar que exista Reg_pct
    if col_reg_num not in df_proyeccion.columns:
        raise ValueError(f"No existe columna '{col_reg_num}'. Llama primero proyectar_regulacion_rv().")

    # filas base y objetivo
    df0 = df_proyeccion.loc[df_proyeccion["Año"] == 0]
    dfn = df_proyeccion.loc[df_proyeccion["Año"] == int(años_objetivo)]

    if df0.empty:
        raise ValueError("No se encontró Año == 0 en df_proyeccion.")
    if dfn.empty:
        # si no existe ese año, usa el último
        dfn = df_proyeccion.tail(1)

    # valores
    dem0 = str(df0.iloc[0][col_dem_fmt]) if col_dem_fmt in df_proyeccion.columns else f"{float(df0.iloc[0]['Demanda_kva']):,.2f}"
    demn = str(dfn.iloc[0][col_dem_fmt]) if col_dem_fmt in df_proyeccion.columns else f"{float(dfn.iloc[0]['Demanda_kva']):,.2f}"

    reg0 = float(df0.iloc[0][col_reg_num])
    regn = float(dfn.iloc[0][col_reg_num])

    # máximo y año crítico
    idx_max = df_proyeccion[col_reg_num].astype(float).idxmax()
    reg_max = float(df_proyeccion.loc[idx_max, col_reg_num])
    anio_crit = int(df_proyeccion.loc[idx_max, "Año"])

    carga_n = float(dfn.iloc[0][col_carga_num]) if col_carga_num in df_proyeccion.columns else np.nan

    # reg base informado (si te lo pasan, lo muestro, si no, uso reg0)
    reg_base_show = float(reg_base_pct) if reg_base_pct is not None else reg0

    resumen = pd.DataFrame(
        [
            ["Demanda", "Año 0", dem0],
            ["Demanda", f"Año {años_objetivo}", demn],
            ["% Carga", f"Año {años_objetivo}", f"{carga_n:.2f}%"] if pd.notna(carga_n) else ["% Carga", f"Año {años_objetivo}", "-"],
            ["% Reg", "Base (año 0, peor nodo)", f"{reg_base_show:.2f}%"],
            ["% Reg", f"Año {años_objetivo}", f"{regn:.2f}%"],
            ["% Reg", f"Máxima (año crítico {anio_crit})", f"{reg_max:.2f}%"],
        ],
        columns=["Variable", "Escenario", "Valor"]
    )

    return resumen


# =====================================================
# Gráfico (demanda)
# =====================================================
def crear_grafico_demanda(df_proyeccion: pd.DataFrame):
    """
    Genera un gráfico SOLO de la demanda proyectada
    y devuelve un objeto Image para insertar en el PDF.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(
        df_proyeccion["Año"],
        df_proyeccion["Demanda_kva"],
        marker="o",
        color="tab:blue",
        label="Demanda (kVA)"
    )

    ax.set_xlabel("Años")
    ax.set_ylabel("Demanda (kVA)")
    ax.set_title("Proyección de Demanda")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Image(buf)
