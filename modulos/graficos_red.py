# -*- coding: utf-8 -*-
"""
modulos/graficos_red.py

Funciones para visualización de la red eléctrica y generación de diagramas
usando NetworkX + Matplotlib.
- Nodo 1 se dibuja igual que los demás (círculo).
- El transformador se dibuja a la par del nodo 1 (triángulo aparte).
"""

from __future__ import annotations

import io
import matplotlib.pyplot as plt
import networkx as nx
from reportlab.platypus import Image
from reportlab.lib.units import inch

# Importa función para cargar datos (si aplica en tu proyecto)
try:
    from modulos.datos import cargar_datos_circuito
except Exception:
    cargar_datos_circuito = None


# ============================================================
# Entradas -> Validación -> Cálculos -> Salidas
# ============================================================

# -----------------------------
# Entradas / Construcción grafo
# -----------------------------

def crear_grafo(nodos_inicio, nodos_final, usuarios, distancias) -> nx.Graph:
    G = nx.Graph()
    for ni, nf, u, d in zip(nodos_inicio, nodos_final, usuarios, distancias):
        G.add_edge(int(ni), int(nf), usuarios=int(u), distancia=float(d))
    return G


def verificar_grafo(G: nx.Graph, nodo_raiz: int = 1) -> dict:
    """
    Verifica conectividad del grafo desde el nodo raíz.
    """
    if nodo_raiz not in G:
        return {"ok": False, "error": f"Nodo raíz {nodo_raiz} no existe.", "nodos": []}

    alcanzables = set(nx.node_connected_component(G, nodo_raiz))
    todos = set(G.nodes())

    return {
        "ok": alcanzables == todos,
        "nodos": sorted(todos),
        "vecinos_raiz": sorted(list(G.neighbors(nodo_raiz))),
        "desconectados": sorted(list(todos - alcanzables)),
        "aristas": sorted((int(u), int(v)) for u, v in G.edges()),
    }


# -----------------------------
# Cálculo de posiciones
# -----------------------------

def calcular_posiciones_red(G: nx.Graph, nodo_raiz: int = 1, escala=None, dy: float = 1.5) -> dict:
    """
    Disposición horizontal principal con ramificaciones verticales (DFS).
    """
    posiciones = {}
    visitados = set()

    if escala is None:
        total_dist = sum(nx.get_edge_attributes(G, "distancia").values()) or 1.0
        escala = 5.0 / (total_dist + 1.0)

    def dfs(nodo, x, y):
        visitados.add(nodo)
        posiciones[nodo] = (x, y)

        vecinos = [v for v in G.neighbors(nodo) if v not in visitados]
        vecinos.sort()

        for i, v in enumerate(vecinos):
            d = float(G[nodo][v].get("distancia", 0.0))
            dx = d * escala
            ny = y - dy * (i - (len(vecinos) - 1) / 2.0)
            dfs(v, x + dx, ny)

    dfs(nodo_raiz, 0.0, 0.0)
    return posiciones


# -----------------------------
# Dibujo
# -----------------------------

def dibujar_aristas(ax, G, posiciones):
    nx.draw_networkx_edges(
        G,
        posiciones,
        edge_color="black",
        width=2,
        ax=ax,
    )


def dibujar_nodos(ax, G: nx.Graph, posiciones: dict):
    """
    IMPORTANTE: NO pasar zorder aquí (NetworkX puede no soportarlo).
    """
    nx.draw_networkx_nodes(
        G,
        posiciones,
        nodelist=list(G.nodes()),
        node_size=220,
        node_color="lightblue",
        edgecolors="black",
        ax=ax,
    )


def dibujar_etiquetas_nodos(ax, G, posiciones):
    labels = {int(n): str(int(n)) for n in G.nodes()}
    nx.draw_networkx_labels(
        G,
        posiciones,
        labels=labels,
        font_size=10,
        font_color="black",
        ax=ax,
    )


def dibujar_acometidas(ax, posiciones: dict, df_conexiones):
    """
    Dibuja línea punteada y texto 'Usuarios' bajo cada nodo_final.
    df_conexiones debe tener: nodo_final, usuarios (y opcional usuarios_especiales)
    """
    for _, row in df_conexiones.iterrows():
        nf = int(row["nodo_final"])
        normales = int(row["usuarios"])
        especiales = int(row.get("usuarios_especiales", 0))

        if nf not in posiciones:
            continue

        x, y = posiciones[nf]
        y2 = y - 0.25

        ax.plot([x, x], [y, y2], "--", color="gray", linewidth=1)

        ax.text(
            x, y2 - 0.05,
            f"Usuarios: {normales}",
            fontsize=11,
            color="blue",
            ha="center",
            va="top",
        )

        if especiales > 0:
            ax.text(
                x, y2 - 0.20,
                f"Especiales: {especiales}",
                fontsize=11,
                color="red",
                ha="center",
                va="top",
            )


def dibujar_distancias_tramos(ax, G: nx.Graph, posiciones: dict):
    for u, v, d in G.edges(data=True):
        dist = float(d.get("distancia", 0.0))
        if dist <= 0:
            continue
        x1, y1 = posiciones[u]
        x2, y2 = posiciones[v]
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax.text(xm, ym + 0.15, f"{dist} m", fontsize=11, color="red", ha="center")



def dibujar_transformador_a_lado(ax, posiciones, capacidad_transformador, nodo=1, dx=-1.2, dy=0.0):
    if nodo not in posiciones:
        return
    x, y = posiciones[nodo]
    xt, yt = x + dx, y + dy

    ax.scatter([xt], [yt], marker="^", s=260, c="orange", edgecolors="black")
    ax.text(
        xt - 0.15, yt,
        f"Transformador\n{capacidad_transformador} kVA",
        fontsize=9, ha="right", va="center",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
    )



# -----------------------------
# Salida principal (ReportLab Image)
# -----------------------------

def crear_grafico_nodos(nodos_inicio, nodos_final, usuarios, distancias, capacidad_transformador, df_conexiones):
    """
    Devuelve reportlab.platypus.Image listo para meter en PDF.
    """
    G = crear_grafo(nodos_inicio, nodos_final, usuarios, distancias)

    # Debug opcional (no rompe nada)
    info = verificar_grafo(G, nodo_raiz=1)
    if not info.get("ok", True):
        print("⚠️ Grafo no completamente conectado:", info)

    posiciones = calcular_posiciones_red(G, nodo_raiz=1)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Orden de dibujo (equivalente a zorder, pero compatible)
    dibujar_aristas(ax, G, posiciones)
    dibujar_nodos(ax, G, posiciones)
    dibujar_etiquetas_nodos(ax, G, posiciones)
    dibujar_acometidas(ax, posiciones, df_conexiones)
    dibujar_distancias_tramos(ax, G, posiciones)
    dibujar_transformador_a_lado(ax, posiciones, capacidad_transformador, nodo=1)


    plt.title("Diagrama de Nodos del Transformador")
    plt.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    img = Image(buf, width=5 * inch, height=3 * inch)
    img.hAlign = "CENTER"
    return img


def crear_grafico_nodos_desde_archivo(ruta_excel: str):
    """
    Si tu proyecto tiene cargar_datos_circuito, genera el gráfico desde el Excel.
    """
    if cargar_datos_circuito is None:
        raise ImportError("No se encontró cargar_datos_circuito. Revisa modulos/datos.py")

    (
        df_conexiones,
        df_parametros,
        df_info,
        tipo_conductor,
        area_lote,
        capacidad_transformador,
        proyecto_numero,
        proyecto_nombre,
        transformador_numero,
        usuarios,
        distancias,
        nodos_inicio,
        nodos_final,
    ) = cargar_datos_circuito(ruta_excel)

    return crear_grafico_nodos(
        nodos_inicio=df_conexiones["nodo_inicial"].astype(int).tolist(),
        nodos_final=df_conexiones["nodo_final"].astype(int).tolist(),
        usuarios=df_conexiones["usuarios"].astype(int).tolist(),
        distancias=df_conexiones["distancia"].astype(float).tolist(),
        capacidad_transformador=capacidad_transformador,
        df_conexiones=df_conexiones,
    )





