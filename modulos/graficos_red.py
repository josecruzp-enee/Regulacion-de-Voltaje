# -*- coding: utf-8 -*-
"""
modulos/graficos_red.py

Diagrama de nodos práctico (tipo plano):
- Troncal principal HORIZONTAL (y = 0)
- Ramas en carriles (arriba/abajo) sin diagonales
- Aristas siempre ORTOGONALES (horizontal + vertical)
- Todos los nodos iguales (círculos)
- Transformador se dibuja a la par del nodo 1 (triángulo aparte)
- Se eliminan self-loops (ej. 1->1) para evitar el “óvalo” en el nodo

Flujo lineal: Entradas -> Validación -> Cálculos -> Salidas
"""

from __future__ import annotations

import io
from typing import Dict, Tuple, Any, Iterable, Callable

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
# Entradas
# ============================================================

def crear_grafo(
    nodos_inicio: Iterable[int],
    nodos_final: Iterable[int],
    usuarios: Iterable[int],
    distancias: Iterable[float],
    eliminar_self_loops: bool = True,
) -> nx.Graph:
    """
    Construye el grafo a partir de listas paralelas.
    NOTA: si viene un tramo (1->1) (self-loop) se ignora por defecto.
    """
    G = nx.Graph()
    for ni, nf, u, d in zip(nodos_inicio, nodos_final, usuarios, distancias):
        ni = int(ni)
        nf = int(nf)
        if eliminar_self_loops and ni == nf:
            continue
        G.add_edge(ni, nf, usuarios=int(u), distancia=float(d))
    return G


# ============================================================
# Validación
# ============================================================

def verificar_grafo(G: nx.Graph, nodo_raiz: int = 1) -> Dict[str, Any]:
    """
    Verifica conectividad desde el nodo raíz y reporta self-loops (si existieran).
    """
    out: Dict[str, Any] = {
        "ok": True,
        "error": "",
        "nodos": sorted(int(n) for n in G.nodes()),
        "aristas": sorted((int(u), int(v)) for u, v in G.edges()),
        "vecinos_raiz": [],
        "desconectados": [],
        "self_loops": [],
    }

    if nodo_raiz not in G:
        out["ok"] = False
        out["error"] = f"Nodo raíz {nodo_raiz} no existe."
        return out

    out["vecinos_raiz"] = sorted(int(v) for v in G.neighbors(nodo_raiz))

    # self-loops (por seguridad)
    loops = [(int(u), int(v)) for u, v in G.edges() if int(u) == int(v)]
    out["self_loops"] = loops
    if loops:
        out["ok"] = False
        out["error"] = f"Self-loop(s) detectado(s): {loops}"

    # conectividad
    alcanzables = set(nx.node_connected_component(G, nodo_raiz))
    todos = set(G.nodes())
    descon = sorted(int(n) for n in (todos - alcanzables))
    out["desconectados"] = descon
    if descon and out["ok"]:
        out["ok"] = False
        out["error"] = "Grafo no completamente conectado desde el nodo raíz."

    return out


# ============================================================
# Cálculos (layout práctico sin diagonales)
# ============================================================

def _peso_distancia(u: int, v: int, data: dict) -> float:
    return float(data.get("distancia", 0.0))


def _dijkstra_paths(G: nx.Graph, raiz: int) -> Tuple[Dict[int, float], Dict[int, list]]:
    """
    Distancia acumulada (sumando 'distancia') y paths desde raiz.
    """
    dist, paths = nx.single_source_dijkstra(G, raiz, weight=_peso_distancia)
    # normalizar llaves int
    dist2 = {int(k): float(v) for k, v in dist.items()}
    paths2 = {int(k): [int(x) for x in p] for k, p in paths.items()}
    return dist2, paths2


def _expandir_rama_en_carril(
    G: nx.Graph,
    parent: int,
    node: int,
    pos: Dict[int, Tuple[float, float]],
    y: float,
    wfun: Callable[[int, int, dict], float] = _peso_distancia,
):
    """
    Expande una rama manteniendo y fijo (sin diagonales).
    x crece según distancia acumulada desde el punto de conexión.
    """
    # Colocar node si no está
    if parent not in pos:
        return

    x_parent = pos[parent][0]
    w = wfun(parent, node, G[parent][node])
    pos[node] = (x_parent + w, y)

    # DFS en la rama, evitando volver al padre
    vecinos = sorted(int(v) for v in G.neighbors(node) if int(v) != int(parent))
    for v in vecinos:
        if v in pos:
            continue
        _expandir_rama_en_carril(G, parent=node, node=v, pos=pos, y=y, wfun=wfun)


def calcular_posiciones_ortogonales(
    G: nx.Graph,
    raiz: int = 1,
    dy: float = 1.6,
    alternar_arriba_abajo: bool = True,
) -> Dict[int, Tuple[float, float]]:
    """
    Layout tipo plano:
    1) Define troncal como el camino más largo (por distancia acumulada) desde la raíz.
    2) Coloca troncal en y=0, con x = distancia acumulada real (metros).
    3) Cada rama que sale del troncal se pone en un carril fijo (y = ±k*dy).
       - Si alternar_arriba_abajo=True: rama1 abajo, rama2 arriba, rama3 abajo...
       - Si False: todas abajo.
    """
    if raiz not in G:
        return {}

    # (1) Troncal: camino a nodo más lejano
    dist, paths = _dijkstra_paths(G, raiz)
    far = max(dist, key=lambda n: dist[n])  # nodo más lejano
    troncal = paths[far]                    # lista de nodos
    set_troncal = set(troncal)

    # (2) Posiciones del troncal
    pos: Dict[int, Tuple[float, float]] = {}
    pos[troncal[0]] = (0.0, 0.0)

    x_acc = 0.0
    for a, b in zip(troncal, troncal[1:]):
        x_acc += _peso_distancia(a, b, G[a][b])
        pos[b] = (x_acc, 0.0)

    # (3) Ramas desde troncal
    lane = 0
    for t in troncal:
        vecinos_fuera = sorted(int(v) for v in G.neighbors(t) if int(v) not in set_troncal)
        for v in vecinos_fuera:
            lane += 1

            if alternar_arriba_abajo:
                # 1 abajo, 2 arriba, 3 abajo, 4 arriba...
                sign = -1 if (lane % 2 == 1) else +1
            else:
                sign = -1

            y_lane = sign * ( (lane + 1) // 2 ) * dy if alternar_arriba_abajo else (-lane * dy)

            # Colocar el primer nodo de la rama en el carril
            # (su x lo definirá _expandir_rama_en_carril con distancia real)
            _expandir_rama_en_carril(G, parent=int(t), node=int(v), pos=pos, y=y_lane)

    return pos


# ============================================================
# Salidas (dibujo)
# ============================================================

def dibujar_aristas_ortogonales(ax, G: nx.Graph, pos: Dict[int, Tuple[float, float]]):
    """
    Dibuja aristas sin diagonales: horizontal y luego vertical.
    """
    for u, v in G.edges():
        u = int(u)
        v = int(v)
        if u == v:
            continue
        if u not in pos or v not in pos:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Tramo horizontal
        ax.plot([x1, x2], [y1, y1], color="black", linewidth=2)
        # Tramo vertical
        ax.plot([x2, x2], [y1, y2], color="black", linewidth=2)


def dibujar_nodos(ax, G: nx.Graph, pos: Dict[int, Tuple[float, float]]):
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(G.nodes()),
        node_shape="o",
        node_size=220,
        node_color="lightblue",
        edgecolors="black",
        linewidths=1.0,
        ax=ax,
    )


def dibujar_etiquetas_nodos(ax, G: nx.Graph, pos: Dict[int, Tuple[float, float]]):
    labels = {int(n): str(int(n)) for n in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=10,
        font_color="black",
        ax=ax,
    )


def dibujar_acometidas(ax, pos: Dict[int, Tuple[float, float]], df_conexiones):
    """
    Dibuja línea punteada y texto 'Usuarios' bajo cada nodo_final.
    df_conexiones debe tener: nodo_final, usuarios (y opcional usuarios_especiales)
    """
    for _, row in df_conexiones.iterrows():
        nf = int(row["nodo_final"])
        normales = int(row["usuarios"])
        especiales = int(row.get("usuarios_especiales", 0))

        if nf not in pos:
            continue

        x, y = pos[nf]
        y2 = y - 0.28

        ax.plot([x, x], [y, y2], "--", color="gray", linewidth=1)

        ax.text(
            x, y2 - 0.06,
            f"Usuarios: {normales}",
            fontsize=11,
            color="blue",
            ha="center",
            va="top",
        )

        if especiales > 0:
            ax.text(
                x, y2 - 0.22,
                f"Especiales: {especiales}",
                fontsize=11,
                color="red",
                ha="center",
                va="top",
            )


def dibujar_distancias_tramos(ax, G: nx.Graph, pos: Dict[int, Tuple[float, float]], mostrar_ceros: bool = False):
    """
    Etiqueta distancias cerca del tramo horizontal (más legible).
    Si mostrar_ceros=False, no dibuja distancias <= 0 (evita '0.0 m').
    """
    for u, v, data in G.edges(data=True):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        if u not in pos or v not in pos:
            continue

        dist = float(data.get("distancia", 0.0))
        if (not mostrar_ceros) and dist <= 0:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Texto sobre el tramo horizontal (x1->x2) a la altura y1
        xm = (x1 + x2) / 2.0
        ym = y1

        ax.text(
            xm, ym + 0.18,
            f"{dist} m",
            fontsize=11,
            color="red",
            ha="center",
        )


def dibujar_transformador_a_lado(
    ax,
    pos: Dict[int, Tuple[float, float]],
    capacidad_transformador: float,
    nodo: int = 1,
    dx: float = -12.0,   # OJO: como ahora x está en "metros", esto también es "metros".
    dy: float = 0.0,
):
    """
    Dibuja el transformador a la par del nodo 1.
    Como x está en metros reales, dx también es en metros.
    Ajusta dx si lo querés más cerca/lejos.
    """
    if nodo not in pos:
        return

    x, y = pos[nodo]
    xt, yt = x + dx, y + dy

    ax.scatter([xt], [yt], marker="^", s=260, c="orange", edgecolors="black", zorder=5)
    ax.text(
        xt - 1.0, yt,
        f"Transformador\n{capacidad_transformador} kVA",
        fontsize=9, ha="right", va="center",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        zorder=6,
    )


# ============================================================
# Salida principal (ReportLab Image)
# ============================================================

def crear_grafico_nodos(
    nodos_inicio,
    nodos_final,
    usuarios,
    distancias,
    capacidad_transformador,
    df_conexiones,
    alternar_arriba_abajo: bool = True,
):
    """
    Devuelve reportlab.platypus.Image listo para meter en PDF.
    Layout práctico sin diagonales.
    """
    # Entradas
    G = crear_grafo(nodos_inicio, nodos_final, usuarios, distancias, eliminar_self_loops=True)

    # Validación (no rompe, solo avisa)
    info = verificar_grafo(G, nodo_raiz=1)
    if not info.get("ok", True):
        print("⚠️ Problema con el grafo:", info)

    # Cálculos (posiciones en "metros")
    pos = calcular_posiciones_ortogonales(G, raiz=1, dy=1.6, alternar_arriba_abajo=alternar_arriba_abajo)

    # Salidas (dibujo)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Orden correcto
    dibujar_aristas_ortogonales(ax, G, pos)
    dibujar_nodos(ax, G, pos)
    dibujar_etiquetas_nodos(ax, G, pos)
    dibujar_acometidas(ax, pos, df_conexiones)
    dibujar_distancias_tramos(ax, G, pos, mostrar_ceros=False)
    dibujar_transformador_a_lado(ax, pos, capacidad_transformador, nodo=1, dx=-12.0, dy=0.0)

    plt.title("Diagrama de Nodos del Transformador")
    plt.axis("off")

    # Para que no se deforme visualmente
    ax.set_aspect("equal", adjustable="datalim")

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", dpi=150)
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
        alternar_arriba_abajo=True,
    )
