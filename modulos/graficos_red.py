# -*- coding: utf-8 -*-
"""
modulos/graficos_red.py

Diagrama de red estilo plano:
- Nodos como círculos iguales (incluye nodo 1).
- Aristas ortogonales (L) recortadas para que NO atraviesen los nodos.
- Transformador como triángulo A LA PAR del nodo 1, con conexión corta recortada.
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

def calcular_posiciones_red(
    G: nx.Graph,
    nodo_raiz: int = 1,
    escala=None,
    dy: float = 1.5,
) -> dict:
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
# Utilidades geométricas
# -----------------------------

def _dist(a, b) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return (dx * dx + dy * dy) ** 0.5


def _trim_segment(a, b, cut_a: float = 0.0, cut_b: float = 0.0):
    """
    Recorta un segmento a->b:
    - cut_a: cuánto recortar desde 'a' hacia 'b'
    - cut_b: cuánto recortar desde 'b' hacia 'a'
    """
    L = _dist(a, b)
    if L < 1e-9:
        return a, b
    ux = (b[0] - a[0]) / L
    uy = (b[1] - a[1]) / L
    a2 = (a[0] + ux * cut_a, a[1] + uy * cut_a)
    b2 = (b[0] - ux * cut_b, b[1] - uy * cut_b)
    return a2, b2


def _draw_L_path(ax, a, b, cut_start: float, cut_end: float, lw: float = 2.0):
    """
    Dibuja una conexión ortogonal tipo 'L' entre a (x1,y1) y b (x2,y2),
    recortando cerca del nodo de inicio y del nodo final para no atravesar círculos.

    Estrategia:
    - tramo1: a -> (x2, y1)
    - tramo2: (x2, y1) -> b
    - recorte se aplica SOLO cerca de a y cerca de b (no en el codo).
    """
    x1, y1 = a
    x2, y2 = b
    codo = (x2, y1)

    # Si queda degenerado (misma x o misma y), es segmento recto: recortamos simple.
    if abs(x1 - x2) < 1e-9 or abs(y1 - y2) < 1e-9:
        aa, bb = _trim_segment(a, b, cut_a=cut_start, cut_b=cut_end)
        ax.plot([aa[0], bb[0]], [aa[1], bb[1]], color="black", linewidth=lw)
        return

    # Tramo A->CODO (recorta en A)
    aa1, bb1 = _trim_segment(a, codo, cut_a=cut_start, cut_b=0.0)
    ax.plot([aa1[0], bb1[0]], [aa1[1], bb1[1]], color="black", linewidth=lw)

    # Tramo CODO->B (recorta en B)
    aa2, bb2 = _trim_segment(codo, b, cut_a=0.0, cut_b=cut_end)
    ax.plot([aa2[0], bb2[0]], [aa2[1], bb2[1]], color="black", linewidth=lw)


# -----------------------------
# Dibujo
# -----------------------------

def dibujar_aristas(ax, G: nx.Graph, posiciones: dict, recorte_nodo: float = 0.16):
    """
    Aristas ortogonales (L) recortadas para que NO entren al centro del nodo.
    """
    for u, v in G.edges():
        a = posiciones[u]
        b = posiciones[v]
        _draw_L_path(ax, a, b, cut_start=recorte_nodo, cut_end=recorte_nodo, lw=2.0)


def dibujar_nodos(ax, G: nx.Graph, posiciones: dict):
    nx.draw_networkx_nodes(
        G,
        posiciones,
        nodelist=list(G.nodes),
        node_size=220,
        node_color="lightblue",
        edgecolors="black",
        ax=ax,
    )


def dibujar_etiquetas_nodos(ax, G: nx.Graph, posiciones: dict):
    nx.draw_networkx_labels(
        G,
        posiciones,
        labels={n: str(n) for n in G.nodes},
        font_size=12,
        font_weight="bold",
        ax=ax,
    )


def dibujar_acometidas(ax, posiciones: dict, df_conexiones, omitir_nodos: set[int] | None = None):
    """
    Dibuja línea punteada y texto 'Usuarios' bajo cada nodo_final.
    df_conexiones debe tener: nodo_final, usuarios (y opcional usuarios_especiales)
    """
    omitir_nodos = omitir_nodos or set()

    for _, row in df_conexiones.iterrows():
        nf = int(row["nodo_final"])

        if nf in omitir_nodos:
            continue
        if nf not in posiciones:
            continue

        normales = int(row.get("usuarios", 0))
        especiales = int(row.get("usuarios_especiales", 0) or 0)

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
    """
    Etiqueta distancia cerca del punto medio del segmento visual (simple).
    Nota: como la arista es en L, usamos el promedio de extremos para ubicar texto.
    """
    for u, v, d in G.edges(data=True):
        x1, y1 = posiciones[u]
        x2, y2 = posiciones[v]
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist = d.get("distancia", "")
        ax.text(xm, ym + 0.15, f"{dist} m", fontsize=11, color="red", ha="center")


def dibujar_transformador_a_lado(
    ax,
    posiciones: dict,
    capacidad_transformador,
    nodo: int = 1,
    dx: float = -0.9,
    dy: float = 0.0,
    conectar: bool = True,
    recorte: float = 0.16,
):
    """
    Dibuja el símbolo del transformador A LA PAR del nodo (triángulo aparte)
    y lo conecta con una línea corta recortada para que no atraviese el nodo/triángulo.
    """
    if nodo not in posiciones:
        return

    x, y = posiciones[nodo]
    xt, yt = x + dx, y + dy

    if conectar:
        a = (x, y)
        b = (xt, yt)
        a2, b2 = _trim_segment(a, b, cut_a=recorte, cut_b=recorte)
        ax.plot([a2[0], b2[0]], [a2[1], b2[1]], color="black", linewidth=2)

    ax.scatter([xt], [yt], marker="^", s=260, c="orange", edgecolors="black")

    ax.text(
        xt - 0.15,
        yt,
        f"Transformador\n{capacidad_transformador} kVA",
        fontsize=9,
        ha="right",
        va="center",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )


# -----------------------------
# Salida principal (ReportLab Image)
# -----------------------------

def crear_grafico_nodos(
    nodos_inicio,
    nodos_final,
    usuarios,
    distancias,
    capacidad_transformador,
    df_conexiones,
):
    """
    Devuelve reportlab.platypus.Image listo para meter en PDF.
    """
    # Entradas
    G = crear_grafo(nodos_inicio, nodos_final, usuarios, distancias)

    # Validación (no rompe, solo avisa)
    info = verificar_grafo(G, nodo_raiz=1)
    if not info.get("ok", True):
        print("⚠️ Grafo no completamente conectado:", info)

    # Cálculos
    posiciones = calcular_posiciones_red(G, nodo_raiz=1)

    # Salidas (dibujo)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # RADIO VIRTUAL del nodo (en coords del layout)
    # Si querés más “estilo AutoCAD” (punto más grande), subilo a 0.18–0.22
    RECORTE_NODO = 0.16

    # Orden de dibujo
    dibujar_aristas(ax, G, posiciones, recorte_nodo=RECORTE_NODO)
    dibujar_nodos(ax, G, posiciones)
    dibujar_etiquetas_nodos(ax, G, posiciones)

    # Si NO querés acometida en el nodo 1 (por el TS), dejalo omitido:
    dibujar_acometidas(ax, posiciones, df_conexiones, omitir_nodos={1})

    dibujar_distancias_tramos(ax, G, posiciones)

    dibujar_transformador_a_lado(
        ax,
        posiciones,
        capacidad_transformador,
        nodo=1,
        dx=-0.9,
        dy=0.0,
        conectar=True,
        recorte=RECORTE_NODO,
    )

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
