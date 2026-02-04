# -*- coding: utf-8 -*-
"""
graficos_red.py
Funciones para visualización de la red eléctrica y generación de diagramas
usando NetworkX y Matplotlib.
"""

import io
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from reportlab.platypus import Image
from reportlab.lib.units import inch

# ============================================================
# Carga de datos
# ============================================================

try:
    from modulos.datos import cargar_datos_circuito
except ImportError:
    from datos import cargar_datos_circuito


# ============================================================
# Creación y verificación de grafo
# ============================================================

def crear_grafo(nodos_inicio, nodos_final, usuarios, distancias):
    G = nx.Graph()
    for ni, nf, u, d in zip(nodos_inicio, nodos_final, usuarios, distancias):
        G.add_edge(
            int(ni),
            int(nf),
            usuarios=int(u),
            distancia=float(d)
        )
    return G


def verificar_grafo(G, nodo_raiz=1):
    """
    Verifica conectividad real del grafo desde el nodo raíz.
    """
    if nodo_raiz not in G:
        return {
            "ok": False,
            "error": f"Nodo raíz {nodo_raiz} no existe"
        }

    alcanzables = set(nx.node_connected_component(G, nodo_raiz))
    todos = set(G.nodes())

    return {
        "ok": alcanzables == todos,
        "nodos": sorted(todos),
        "vecinos_raiz": sorted(G.neighbors(nodo_raiz)),
        "desconectados": sorted(todos - alcanzables),
        "aristas": sorted((int(u), int(v)) for u, v in G.edges())
    }


# ============================================================
# Posicionamiento del grafo
# ============================================================

def calcular_posiciones_red(G, nodo_raiz=1, escala=None, dy=1.5):
    """
    Disposición horizontal principal con ramificaciones verticales.
    """
    posiciones = {}
    visitados = set()

    if escala is None:
        total_dist = sum(nx.get_edge_attributes(G, "distancia").values())
        escala = 5 / (total_dist + 1)

    def dfs(nodo, x, y):
        visitados.add(nodo)
        posiciones[nodo] = (x, y)

        vecinos = [v for v in G.neighbors(nodo) if v not in visitados]
        vecinos.sort()

        for i, v in enumerate(vecinos):
            d = G[nodo][v]["distancia"]
            dx = d * escala
            ny = y - dy * (i - (len(vecinos) - 1) / 2)
            dfs(v, x + dx, ny)

    dfs(nodo_raiz, 0, 0)
    return posiciones


# ============================================================
# Dibujo de elementos
# ============================================================

def dibujar_nodos(ax, G, posiciones):
    nx.draw_networkx_nodes(
        G,
        posiciones,
        nodelist=list(G.nodes),
        node_size=220,
        node_color="lightblue",
        edgecolors="black",
        ax=ax
    )
)


def dibujar_aristas(ax, G, posiciones):
    """
    Aristas ortogonales (L) SIN stubs especiales.
    """
    for u, v in G.edges():
        x1, y1 = posiciones[u]
        x2, y2 = posiciones[v]

        cx, cy = x2, y1
        ax.plot([x1, cx], [y1, cy], color="black", linewidth=2, zorder=1)
        ax.plot([cx, x2], [cy, y2], color="black", linewidth=2, zorder=1)


def dibujar_etiquetas_nodos(ax, G, posiciones):
    nx.draw_networkx_labels(
        G,
        posiciones,
        labels={n: str(n) for n in G.nodes},
        font_size=12,
        font_weight="bold",
        ax=ax
    )


def dibujar_acometidas(ax, posiciones, df_conexiones):
    for _, row in df_conexiones.iterrows():
        nf = int(row["nodo_final"])
        usuarios = int(row["usuarios"])
        especiales = int(row.get("usuarios_especiales", 0))

        if nf not in posiciones:
            continue

        x, y = posiciones[nf]
        y2 = y - 0.25

        ax.plot([x, x], [y, y2], "--", color="gray", linewidth=1, zorder=2)

        ax.text(
            x, y2 - 0.05,
            f"Usuarios: {usuarios}",
            fontsize=11,
            color="blue",
            ha="center",
            va="top",
            zorder=4
        )

        if especiales > 0:
            ax.text(
                x, y2 - 0.20,
                f"Especiales: {especiales}",
                fontsize=11,
                color="red",
                ha="center",
                va="top",
                zorder=4
            )


def dibujar_distancias_tramos(ax, G, posiciones):
    for u, v, d in G.edges(data=True):
        x1, y1 = posiciones[u]
        x2, y2 = posiciones[v]

        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            xm,
            ym + 0.15,
            f"{d['distancia']} m",
            fontsize=11,
            color="red",
            ha="center",
            va="center",
            zorder=4
        )


def dibujar_transformador_a_lado(
    ax,
    posiciones,
    capacidad_transformador,
    nodo=1,
    dx=-0.9,
    dy=0.0
):
    """
    Dibuja el símbolo del transformador A LA PAR del nodo,
    sin reemplazarlo ni tocar aristas.
    """
    if nodo not in posiciones:
        return

    x, y = posiciones[nodo]
    xt, yt = x + dx, y + dy

    ax.scatter(
        [xt],
        [yt],
        marker="^",
        s=260,
        c="orange",
        edgecolors="black",
        zorder=5
    )

    ax.text(
        xt - 0.15,
        yt,
        f"Transformador\n{capacidad_transformador} kVA",
        fontsize=9,
        ha="right",
        va="center",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        zorder=6
    )


# ============================================================
# Función principal
# ============================================================

def crear_grafico_nodos(
    nodos_inicio,
    nodos_final,
    usuarios,
    distancias,
    capacidad_transformador,
    df_conexiones
):
    G = crear_grafo(nodos_inicio, nodos_final, usuarios, distancias)

    info = verificar_grafo(G, nodo_raiz=1)
    if not info["ok"]:
        print("⚠️ Advertencia: grafo no completamente conectado")
        print(info)

    posiciones = calcular_posiciones_red(G, nodo_raiz=1)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    dibujar_nodos(ax, G, posiciones)
    dibujar_aristas(ax, G, posiciones)
    dibujar_etiquetas_nodos(ax, G, posiciones)
    dibujar_acometidas(ax, posiciones, df_conexiones)
    dibujar_distancias_tramos(ax, G, posiciones)
    dibujar_transformador_a_lado(
        ax,
        posiciones,
        capacidad_transformador,
        nodo=1
    )

    plt.title("Diagrama de Nodos del Transformador")
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", bbox_inches="tight")
    plt.close()

    buf.seek(0)
    img = Image(buf, width=5 * inch, height=3 * inch)
    img.hAlign = "CENTER"
    return img


# ============================================================
# Desde archivo Excel
# ============================================================

def crear_grafico_nodos_desde_archivo(ruta_excel):
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
        nodos_final
    ) = cargar_datos_circuito(ruta_excel)

    return crear_grafico_nodos(
        nodos_inicio=df_conexiones["nodo_inicial"].astype(int).tolist(),
        nodos_final=df_conexiones["nodo_final"].astype(int).tolist(),
        usuarios=df_conexiones["usuarios"].astype(int).tolist(),
        distancias=df_conexiones["distancia"].astype(float).tolist(),
        capacidad_transformador=capacidad_transformador,
        df_conexiones=df_conexiones
    )

