# -*- coding: utf-8 -*-
"""
lineas.py
Cálculos de parámetros de línea: resistencias, reactancias, impedancias

MODO RV-2002 (TABULADO) – CONSISTENTE CON RV:
- conductores[tipo]["R"] y ["X"] se interpretan como Ω/km POR CONDUCTOR (1c)
- Para caída F–N (lazo ida+retorno): se usa factor_lazo = 2
- Para reportes: se exponen también r/x por conductor (sin el factor 2)
"""

from __future__ import annotations

from .datos import biblioteca_conductores

CONDUCTORES = biblioteca_conductores()


# ============================================================
# Parámetros base (por conductor)
# ============================================================
def R_X_por_km(conductores: dict, tipo_conductor: str) -> tuple[float, float]:
    """Devuelve (R_km_1c, X_km_1c) en Ω/km por conductor (1c)."""
    if tipo_conductor not in conductores:
        raise ValueError(f"Conductor '{tipo_conductor}' no está en el diccionario.")
    R_km = float(conductores[tipo_conductor]["R"])
    X_km = float(conductores[tipo_conductor]["X"])
    return R_km, X_km


# ============================================================
# Cálculo del vano (misma API que ya usas)
# ============================================================
def resistencia_por_vano(conductores, tipo_conductor, distancia_m, factor_lazo: float = 2.0):
    """
    Resistencia del vano en Ω.
    Por defecto calcula LAZO (ida+retorno) para que cuadre con RV.
    """
    L_km = float(distancia_m) / 1000.0
    R_km_1c, _ = R_X_por_km(conductores, tipo_conductor)
    return factor_lazo * R_km_1c * L_km


def reactancia_por_vano(conductores, tipo_conductor, distancia_m, factor_lazo: float = 2.0):
    """
    Reactancia del vano en Ω.
    Por defecto calcula LAZO (ida+retorno) para que cuadre con RV.
    """
    L_km = float(distancia_m) / 1000.0
    _, X_km_1c = R_X_por_km(conductores, tipo_conductor)
    return factor_lazo * X_km_1c * L_km


# ============================================================
# Auxiliares para reporte (sin tocar el resto de tu app)
# ============================================================
def resistencia_por_vano_conductor(conductores, tipo_conductor, distancia_m):
    """Resistencia del vano por conductor (1c) en Ω (sin retorno)."""
    return resistencia_por_vano(conductores, tipo_conductor, distancia_m, factor_lazo=1.0)


def reactancia_por_vano_conductor(conductores, tipo_conductor, distancia_m):
    """Reactancia del vano por conductor (1c) en Ω (sin retorno)."""
    return reactancia_por_vano(conductores, tipo_conductor, distancia_m, factor_lazo=1.0)


def calcular_impedancia(row):
    if row["distancia"] == 0:
        return 0 + 0j
    return row["resistencia_vano"] + 1j * row["reactancia_vano"]


def calcular_admitancia(z):
    if abs(z) < 1e-12:
        return 0 + 0j
    return 1 / z
