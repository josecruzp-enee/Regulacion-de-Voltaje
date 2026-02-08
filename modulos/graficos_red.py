# -*- coding: utf-8 -*-
"""
modulos/graficos_red.py

Diagrama esquemático ortogonal (tipo AutoCAD), compacto y determinista.

Flujo:
Entradas -> Validación -> Normalización -> Layout -> Dibujo -> Salida

Reglas:
- Tramos: ni != nf (con distancia)
- Usuarios en nodo: ni == nf (sin tramo; ej. 1->1)
- Usuarios por nodo se dibujan UNA vez (no duplicados)
- Troncal (camino raíz->hoja más largo) se pliega en serpiente (U/C)
- Ramas ortogonales colgadas desde la troncal
- Quiebres insertan nodos CODO_* (solo dibujo)
- Transformador cerca del nodo 1, sin línea
"""

from __future__ import annotations

import io
import matplotlib.pyplot as plt
import networkx as nx
from reportlab.platypus import Image
from reportlab.lib.units import inch

try:
    from modulos.datos import cargar_datos_circuito
except Exception:
    cargar_datos_circuito = None


# ============================================================
# Entradas / Normalización
# ============================================================

def separar_tramos_y_usuarios(df_conexiones):
    """
    Devuelve:
      - df_tramos: filas ni!=nf, con distancia
      - usuarios_por_nodo: dict[int,int confirmados] (incluye ni==nf y también suma usuarios de acometidas)
    Nota:
      En tus datos, 'usuarios' representa la acometida asociada a nodo_final.
      Y cuando ni==nf, lo interpretamos como usuarios "en ese nodo".
    """
    import pandas as pd

    df = df_conexiones.copy()

    # Normaliza tipos
    for c in ("nodo_inicial", "nodo_final"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "usuarios" not in df.columns:
        df["usuarios"] = 0
    df["usuarios"] = pd.to_numeric(df["usuarios"], errors="coerce").fillna(0).astype(int)

    if "usuarios_especiales" not in df.columns:
        df["usuarios_especiales"] = 0
    df["usuarios_especiales"] = pd.to_numeric(df["usuarios_especiales"], errors="coerce").fillna(0).astype(int)

    if "distancia" not in df.columns:
        df["distancia"] = 0.0
    df["distancia"] = pd.to_numeric(df["distancia"], errors="coerce").fillna(0.0).astype(float)

    # 1) Usuarios en nodo (ni==nf)
    df_nodo = df[df["nodo_inicial"] == df["nodo_final"]].copy()
    usuarios_nodo = (
        df_nodo.groupby("nodo_final")[["usuarios", "usuarios_especiales"]].sum()
    )

    # 2) Tramos reales
    df_tramos = df[df["nodo_inicial"] != df["nodo_final"]].copy()

    # 3) Acometidas (usuarios por nodo_final en tramos)
    usuarios_acom = (
        df_tramos.groupby("nodo_final")[["usuarios", "usuarios_especiales"]].sum()
    )

    # 4) Merge final usuarios_por_nodo
    usuarios_por_nodo = {}
    all_nodes = set(usuarios_nodo.index.tolist()) | set(usuarios_acom.index.tolist())
    for n in all_nodes:
        u = int(usuarios_nodo.loc[n, "usuarios"]) if n in usuarios_nodo.index else 0
        ue = int(usuarios_nodo.loc[n, "usuarios_especiales"]) if n in usuarios_nodo.index else 0
        ua = int(usuarios_acom.loc[n, "usuarios"]) if n in usuarios_acom.index else 0
        uea = int(usuarios_acom.loc[n, "usuarios_especiales"]) if n in usuarios_acom.index else 0
        usuarios_por_nodo[int(n)] = {
            "usuarios": u + ua,
            "usuarios_especiales": ue + uea,
        }

    return df_tramos, usuarios_por_nodo


def construir_grafo_desde_tramos(df_tramos) -> nx.Graph:
    """
    Grafo solo con tramos reales (ni!=nf).
    """
    G = nx.Graph()
    for _, r in df_tramos.iterrows():
        ni = int(r["nodo_inicial"])
        nf = int(r["nodo_final"])
        d = float(r.get("distancia", 0.0) or 0.0)
        G.add_node(ni)
        G.add_node(nf)
        # Si hay duplicado, nos quedamos con la mayor distancia (o la última)
        G.add_edge(ni, nf, distancia=d)
    return G


def verificar_grafo(G: nx.Graph, nodo_raiz: int = 1) -> None:
    if nodo_raiz not in G and G.number_of_nodes() > 0:
        raise ValueError(f"Nodo raíz {nodo_raiz} no existe en el grafo.")
    if G.number_of_nodes() == 0:
        raise ValueError("Grafo vacío: no hay tramos.")


# ============================================================
# Layout (serpiente U/C + ramas)
# ============================================================

def es_arbol_radial(G: nx.Graph, root: int) -> bool:
    if G.number_of_edges() == 0:
        return True
    if not nx.is_connected(G):
        return False
    return G.number_of_edges() == G.number_of_nodes() - 1


def camino_troncal_mas_largo(G: nx.Graph, root: int) -> list[int]:
    dist, paths = nx.single_source_dijkstra(G, root, weight="distancia")
    hojas = [n for n in G.nodes() if n != root and G.degree(n) == 1]
    if not hojas:
        return [root]
    hoja_max = max(hojas, key=lambda n: dist.get(n, 0.0))
    return paths[hoja_max]


def parent_order(G: nx.Graph, root: int):
    parent = {root: None}
    order = [root]
    stack = [root]
    while stack:
        u = stack.pop()
        for v in G.neighbors(u):
            if v in parent:
                continue
            parent[v] = u
            order.append(v)
            stack.append(v)
    return parent, order


def layout_serpiente(G: nx.Graph, root: int = 1, ancho: float = 3.6, salto: float = 1.2):
    """
    Devuelve:
      GD: grafo dibujo con CODO_*
      pos: posiciones
      nodos_reales: set[int]
    """
    nodos_reales = set(G.nodes())
    if G.number_of_nodes() == 0:
        return nx.Graph(), {}, set()

    # escala auto
    total = sum(nx.get_edge_attributes(G, "distancia").values()) or 1.0
    escala = 6.0 / (total + 1.0)

    # Si no es árbol, layout simple a la derecha
    if not es_arbol_radial(G, root):
        GD = G.copy()
        pos = {root: (0.0, 0.0)}
        parent, order = parent_order(GD, root)
        for n in order:
            if n == root:
                continue
            p = parent[n]
            d = float(GD[p][n].get("distancia", 0.0)) * escala
            x0, y0 = pos[p]
            pos[n] = (x0 + d, y0)
        return GD, pos, nodos_reales

    troncal = camino_troncal_mas_largo(G, root)
    troncal_set = set(troncal)

    GD = nx.Graph()
    GD.add_nodes_from(G.nodes(data=True))

    pos = {root: (0.0, 0.0)}
    x, y = 0.0, 0.0
    direccion = +1
    codo_id = 0

    def new_codo():
        nonlocal codo_id
        codo_id += 1
        return f"CODO_{codo_id:03d}"

    def add_edge(a, b, dist_m: float, es_codo: bool = False):
        GD.add_node(a)
        GD.add_node(b)
        GD.add_edge(a, b, distancia=float(dist_m), es_codo=bool(es_codo))

    # Troncal serpiente
    for i in range(len(troncal) - 1):
        u = troncal[i]
        v = troncal[i + 1]
        dist_m = float(G[u][v].get("distancia", 0.0) or 0.0)
        dx = dist_m * escala

        if direccion == +1:
            if x + dx <= ancho:
                x2 = x + dx
                pos[v] = (x2, y)
                add_edge(u, v, dist_m)
                x = x2
            else:
                # parte en borde derecho y baja
                a_borde_units = ancho - x
                a_borde_m = a_borde_units / escala if escala else dist_m
                rem_m = max(dist_m - a_borde_m, 0.0)

                c1 = new_codo()
                pos[c1] = (ancho, y)
                add_edge(u, c1, a_borde_m)

                c2 = new_codo()
                y -= salto
                pos[c2] = (ancho, y)
                add_edge(c1, c2, 0.0, es_codo=True)

                direccion = -1
                x = ancho

                x2 = x - (rem_m * escala)
                pos[v] = (x2, y)
                add_edge(c2, v, rem_m)
                x = x2
        else:
            if x - dx >= 0.0:
                x2 = x - dx
                pos[v] = (x2, y)
                add_edge(u, v, dist_m)
                x = x2
            else:
                a_borde_units = x
                a_borde_m = a_borde_units / escala if escala else dist_m
                rem_m = max(dist_m - a_borde_m, 0.0)

                c1 = new_codo()
                pos[c1] = (0.0, y)
                add_edge(u, c1, a_borde_m)

                c2 = new_codo()
                y -= salto
                pos[c2] = (0.0, y)
                add_edge(c1, c2, 0.0, es_codo=True)

                direccion = +1
                x = 0.0

                x2 = x + (rem_m * escala)
                pos[v] = (x2, y)
                add_edge(c2, v, rem_m)
                x = x2

    # Ramas ortogonales: primer tramo vertical, resto horizontal
    parent, _ = parent_order(G, root)
    rama_slot = {}

    def place_branch(child: int, attach: int):
        k = rama_slot.get(attach, 0)
        rama_slot[attach] = k + 1

        # alterna arriba/abajo, y separa por nivel
        sign = +1 if (k % 2 == 0) else -1
        extra = 0.35 * k

        xa, ya = pos[attach]
        d0 = float(G[attach][child].get("distancia", 0.0) or 0.0) * escala
        pos[child] = (xa, ya + sign * (d0 + extra))
        add_edge(attach, child, float(G[attach][child].get("distancia", 0.0) or 0.0))

        stack = [(child, attach)]
        while stack:
            u, p = stack.pop()
            xu, yu = pos[u]
            for v in G.neighbors(u):
                if v == p or v in troncal_set or v in pos:
                    continue
                dist_m = float(G[u][v].get("distancia", 0.0) or 0.0)
                pos[v] = (xu + (dist_m * escala), yu)
                add_edge(u, v, dist_m)
                stack.append((v, u))

    for t in troncal:
        for nb in G.neighbors(t):
            if nb in troncal_set:
                continue
            if parent.get(nb) == t and nb not in pos:
                place_branch(nb, t)

    return GD, pos, nodos_reales


# ============================================================
# Dibujo
# ============================================================

def draw_edges(ax, GD: nx.Graph, pos: dict, lw: float = 2.0):
    for u, v, _ in GD.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=lw)


def draw_nodes(ax, GD: nx.Graph, pos: dict, nodos_reales: set[int]):
    reales = [n for n in GD.nodes() if n in nodos_reales]
    codos = [n for n in GD.nodes() if n not in nodos_reales]

    nx.draw_networkx_nodes(GD, pos, nodelist=reales, node_size=220, node_color="lightblue", edgecolors="black", ax=ax)
    if codos:
        nx.draw_networkx_nodes(GD, pos, nodelist=codos, node_size=70, node_color="lightgray", edgecolors="black", ax=ax)


def draw_labels(ax, GD: nx.Graph, pos: dict, nodos_reales: set[int]):
    labels = {n: str(n) for n in GD.nodes() if n in nodos_reales}
    nx.draw_networkx_labels(GD, pos, labels=labels, font_size=12, font_weight="bold", ax=ax)


def draw_distances(ax, GD: nx.Graph, pos: dict):
    for u, v, d in GD.edges(data=True):
        dist = float(d.get("distancia", 0.0) or 0.0)
        if dist <= 0 or u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, f"{dist:.1f} m", fontsize=11, color="red", ha="center")


def draw_users(ax, pos: dict, usuarios_por_nodo: dict[int, dict], nodos_reales: set[int]):
    """
    Determinista + anti-solape:
    - Mantiene el texto debajo del nodo.
    - Si hay nodos muy cerca en X en la misma "banda" de Y, apila hacia abajo.
    """
    y_linea = 0.25
    y_text = 0.08

    # parámetros de anti-solape (en unidades del plano)
    y_band = 0.35      # cuantiza Y para agrupar nodos en la misma "fila"
    min_dx = 0.55      # si dos nodos quedan más cerca que esto en X, se apila
    step_down = 0.22   # cuánto baja cada nivel adicional

    # armar lista de nodos dibujables (solo reales, con usuarios)
    items = []
    for n in sorted(usuarios_por_nodo.keys()):
        if n not in nodos_reales or n not in pos:
            continue
        u = int(usuarios_por_nodo[n].get("usuarios", 0) or 0)
        ue = int(usuarios_por_nodo[n].get("usuarios_especiales", 0) or 0)
        if u <= 0 and ue <= 0:
            continue
        x, y = pos[n]
        band = int(round(y / y_band))
        items.append((band, x, y, n, u, ue))

    # agrupar por banda (fila) y resolver solapes en X apilando hacia abajo
    from collections import defaultdict
    grupos = defaultdict(list)
    for band, x, y, n, u, ue in items:
        grupos[band].append((x, y, n, u, ue))

    for band in sorted(grupos.keys()):
        grupo = sorted(grupos[band], key=lambda t: t[0])  # por x
        prev_x = None
        level = 0

        for x, y, n, u, ue in grupo:
            if prev_x is None:
                level = 0
            else:
                # si está muy cerca del anterior, bajamos un nivel
                if abs(x - prev_x) < min_dx:
                    level += 1
                else:
                    level = 0

            prev_x = x

            y2 = y - y_linea
            extra_down = level * step_down

            ax.plot([x, x], [y, y2], "--", color="gray", linewidth=1)

            ax.text(
                x, (y2 - y_text) - extra_down,
                f"Usuarios: {u}",
                fontsize=11, color="blue",
                ha="center", va="top",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2),
            )

            if ue > 0:
                ax.text(
                    x, (y2 - y_text - 0.15) - extra_down,
                    f"Especiales: {ue}",
                    fontsize=11, color="red",
                    ha="center", va="top",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2),
                )



def draw_transformer(ax, pos: dict, kva, nodo: int = 1, dx: float = -0.9, dy: float = 0.0):
    if nodo not in pos:
        return
    x, y = pos[nodo]
    xt, yt = x + dx, y + dy
    ax.scatter([xt], [yt], marker="^", s=260, c="orange", edgecolors="black")
    ax.text(
        xt - 0.15, yt,
        f"Transformador\n{kva} kVA",
        fontsize=9, ha="right", va="center",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )


# ============================================================
# Salida principal
# ============================================================

def crear_grafico_nodos_df(df_conexiones, capacidad_transformador, nodo_raiz: int = 1):
    """
    Implementación nueva: toma df_conexiones y calcula:
      - tramos (ni!=nf)
      - usuarios por nodo (incluye ni==nf como 'usuarios en nodo')
    """
    df_tramos, usuarios_por_nodo = separar_tramos_y_usuarios(df_conexiones)
    G = construir_grafo_desde_tramos(df_tramos)

    verificar_grafo(G, nodo_raiz=nodo_raiz)

    GD, pos, nodos_reales = layout_serpiente(G, root=nodo_raiz, ancho=3.6, salto=1.2)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    draw_edges(ax, GD, pos, lw=2.0)
    draw_nodes(ax, GD, pos, nodos_reales)
    draw_labels(ax, GD, pos, nodos_reales)

    # ✅ Usuarios deterministas por nodo (incluye el 1->1 correctamente en nodo 1)
    draw_users(ax, pos, usuarios_por_nodo, nodos_reales)

    draw_distances(ax, GD, pos)
    draw_transformer(ax, pos, capacidad_transformador, nodo=nodo_raiz, dx=-0.9, dy=0.0)

    plt.title("Diagrama de Nodos del Transformador")
    plt.axis("off")

    xs = [p[0] for p in pos.values()] if pos else [0.0]
    ys = [p[1] for p in pos.values()] if pos else [0.0]
    pad = 0.9
    extra_left = 1.3  # ajustable
    ax.set_xlim(min(xs) - (pad + extra_left), max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.set_aspect("equal", adjustable="box")

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    img = Image(buf, width=5 * inch, height=3 * inch)
    img.hAlign = "CENTER"
    return img


def crear_grafico_nodos(
    nodos_inicio,
    nodos_final,
    usuarios,
    distancias,
    capacidad_transformador,
    df_conexiones=None,
):
    """
    ✅ Wrapper retrocompatible (firma vieja).
    - Si te pasan df_conexiones, lo usa (preferido).
    - Si no, construye un df mínimo con las listas.
    """
    import pandas as pd

    if df_conexiones is None:
        df_conexiones = pd.DataFrame({
            "nodo_inicial": list(nodos_inicio),
            "nodo_final": list(nodos_final),
            "usuarios": list(usuarios),
            "distancia": list(distancias),
        })
    else:
        # Asegura que al menos existan columnas base
        if "distancia" not in df_conexiones.columns and distancias is not None:
            df_conexiones = df_conexiones.copy()
            df_conexiones["distancia"] = list(distancias)

    return crear_grafico_nodos_df(
        df_conexiones=df_conexiones,
        capacidad_transformador=capacidad_transformador,
        nodo_raiz=1,
    )


def crear_grafico_nodos_desde_archivo(ruta_excel: str):
    if cargar_datos_circuito is None:
        raise ImportError("No se encontró cargar_datos_circuito. Revisa modulos/datos.py")

    (
        df_conexiones,
        _df_parametros,
        _df_info,
        _tipo_conductor,
        _area_lote,
        capacidad_transformador,
        *_resto
    ) = cargar_datos_circuito(ruta_excel)

    return crear_grafico_nodos_df(
        df_conexiones=df_conexiones,
        capacidad_transformador=capacidad_transformador,
        nodo_raiz=1,
    )

