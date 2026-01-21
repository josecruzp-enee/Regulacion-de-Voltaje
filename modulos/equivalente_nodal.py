# modulos/equivalente_nodal.py
# -*- coding: utf-8 -*-
"""
equivalente_nodal.py
Construye y formatea el circuito equivalente nodal (Nivel 2) para reportes.

Objetivo:
- NO tocar tu solver.
- Devolver estructuras listas para PDF:
  - mapa de nodos
  - matrices Y, Yrr, Y_r0 en DataFrames formateados (G + jB)
  - tabla de ramas (R, X, Z, Y) para el “circuito con R y X”
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================
def fmt_c(z: complex, nd: int = 4) -> str:
    """Formatea complejo como 'a + jb'."""
    a = float(np.real(z))
    b = float(np.imag(z))
    return f"{a:.{nd}f} {b:+.{nd}f}j"


def safe_div(a, b):
    return a / b if (b is not None and b != 0 and not np.isnan(b)) else np.nan


def matriz_a_df(
    Y: np.ndarray,
    nodos: List[Any],
    *,
    nd: int = 4,
    max_n: int = 10,
    titulo: str = "Y"
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Convierte matriz compleja a DataFrame con etiquetas.
    Si hay muchos nodos, recorta para PDF.
    """
    n = len(nodos)
    if n > max_n:
        idx = list(range(max_n))
        nodos_show = [nodos[i] for i in idx]
        Y_show = Y[np.ix_(idx, idx)]
        nota = f"{titulo}: Mostrando {max_n} de {n} nodos (recorte para legibilidad)."
    else:
        nodos_show = nodos
        Y_show = Y
        nota = None

    df = pd.DataFrame(
        [[fmt_c(Y_show[i, j], nd=nd) for j in range(len(nodos_show))] for i in range(len(nodos_show))],
        columns=[str(x) for x in nodos_show],
        index=[str(x) for x in nodos_show],
    ).reset_index().rename(columns={"index": "Nodo"})
    return df, nota


# ============================================================
# Resultado estructurado
# ============================================================
@dataclass
class EquivalenteNodal:
    df_mapa: pd.DataFrame
    df_Y: pd.DataFrame
    nota_Y: Optional[str]
    df_Yrr: pd.DataFrame
    nota_Yrr: Optional[str]
    df_Yr0: pd.DataFrame
    df_ramas: pd.DataFrame  # “circuito con R y X”
    slack_nodo: Any
    slack_index: int


# ============================================================
# Constructor principal
# ============================================================
def construir_equivalente_nodal(
    df_conexiones: pd.DataFrame,
    *,
    nodos: List[Any],
    slack_index: int,
    Y: np.ndarray,
    Yrr: np.ndarray,
    Y_r0: np.ndarray,
    V0: complex = 240 + 0j,
    col_ni: str = "nodo_inicio",
    col_nf: str = "nodo_final",
    col_dist: str = "distancia",
    col_r: str = "resistencia_vano",
    col_x: str = "reactancia_vano",
    col_z: str = "Z_vano",
    col_y: str = "Y_vano",
    nd: int = 4,
    max_n: int = 10,
) -> EquivalenteNodal:
    """
    Devuelve todo lo necesario para reportar el equivalente nodal.
    No calcula Y (eso lo hace tu módulo matrices), solo formatea y arma tablas.
    """

    # ---- mapa de nodos
    df_mapa = pd.DataFrame({
        "Índice": list(range(len(nodos))),
        "Nodo": [str(n) for n in nodos],
        "Slack": ["SI" if i == slack_index else "" for i in range(len(nodos))]
    })

    # ---- matrices
    df_Y, nota_Y = matriz_a_df(Y, nodos, nd=nd, max_n=max_n, titulo="Y")
    nodos_r = [nodos[i] for i in range(len(nodos)) if i != slack_index]
    df_Yrr, nota_Yrr = matriz_a_df(Yrr, nodos_r, nd=nd, max_n=max_n, titulo="Yrr")

    # ---- vector Y_r0
    Yr0 = np.asarray(Y_r0)
    if Yr0.ndim == 1:
    # (n-1,)
      Yr0_col = Yr0.reshape(-1, 1)
    elif Yr0.ndim == 2:
    # (n-1,1) o (1,n-1)
        if Yr0.shape[1] == 1:
        Yr0_col = Yr0
    elif Yr0.shape[0] == 1:
        Yr0_col = Yr0.T
    else:
        # caso raro: matriz no vector
        raise ValueError(f"Y_r0 tiene forma inesperada: {Yr0.shape}")
else:
    raise ValueError(f"Y_r0 tiene ndim inesperado: {Yr0.ndim}")

# Seguridad: tamaño debe coincidir con nodos_r
if Yr0_col.shape[0] != len(nodos_r):
    raise ValueError(
        f"Tamaño de Y_r0 ({Yr0_col.shape[0]}) no coincide con nodos sin slack ({len(nodos_r)}). "
        f"Forma original Y_r0: {np.asarray(Y_r0).shape}"
    )

df_Yr0 = pd.DataFrame({
    "Nodo": [str(n) for n in nodos_r],
    "Y_r0 (S)": [fmt_c(Yr0_col[i, 0], nd=nd) for i in range(Yr0_col.shape[0])]
})

    # ---- tabla de ramas (circuito con R y X)
    # Usamos la info ya existente en df_conexiones: Ni, Nf, Dist, r, x, Z, Y
    df = df_conexiones.copy()

    # Asegurar columnas
    for c in (col_ni, col_nf, col_dist, col_r, col_x):
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida en df_conexiones: '{c}'")

    # Si no existen Z/Y en df, los calculamos aquí con r y x
    if col_z not in df.columns:
        df[col_z] = df.apply(
            lambda row: 0j if float(row[col_dist]) == 0 else complex(float(row[col_r]), float(row[col_x])),
            axis=1
        )
    if col_y not in df.columns:
        df[col_y] = df[col_z].apply(lambda z: 0j if abs(z) < 1e-12 else 1 / z)

    df_ramas = df[[col_ni, col_nf, col_dist, col_r, col_x, col_z, col_y]].copy()
    df_ramas = df_ramas.rename(columns={
        col_ni: "Ni",
        col_nf: "Nf",
        col_dist: "Dist (m)",
        col_r: "r (Ω)",
        col_x: "x (Ω)",
        col_z: "Z (Ω)",
        col_y: "Y (S)"
    })

    # Formato
    df_ramas["Dist (m)"] = df_ramas["Dist (m)"].map(lambda v: f"{float(v):.0f}")
    df_ramas["r (Ω)"] = df_ramas["r (Ω)"].map(lambda v: f"{float(v):.6f}")
    df_ramas["x (Ω)"] = df_ramas["x (Ω)"].map(lambda v: f"{float(v):.6f}")
    df_ramas["Z (Ω)"] = df_ramas["Z (Ω)"].map(lambda z: fmt_c(z, nd=6))
    df_ramas["Y (S)"] = df_ramas["Y (S)"].map(lambda y: fmt_c(y, nd=6))

    slack_nodo = nodos[slack_index]

    return EquivalenteNodal(
        df_mapa=df_mapa,
        df_Y=df_Y, nota_Y=nota_Y,
        df_Yrr=df_Yrr, nota_Yrr=nota_Yrr,
        df_Yr0=df_Yr0,
        df_ramas=df_ramas,
        slack_nodo=slack_nodo,
        slack_index=slack_index
    )
