# -*- coding: utf-8 -*-
"""
modulos/graficos_red.py
Diagrama ortogonal compacto, determinista, sin solapes en textos usando "lanes" (carriles).
Flujo: Entradas -> Normalización -> Layout -> Render -> Salida
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


# =========================
# Normalización
# =========================

def calcular_kva_por_nodo(tabla_potencia) -> dict[int, float]:
    import pandas as pd
    df = tabla_potencia.copy()
    if "nodo_final" not in df.columns:
        return {}
    col = "kva_total" if "kva_total" in df.columns else ("kva" if "kva" in df.columns else None)
    if col is None:
        return {}
    df["nodo_final"] = pd.to_numeric(df["nodo_final"], errors="coerce").fillna(0).astype(int)
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    out = df.groupby("nodo_final")[col].sum().to_dict()
    return {int(k): float(v) for k, v in out.items()}


def separar_tramos_y_usuarios(df_conexiones):
    import pandas as pd
    df = df_conexiones.copy()

    for c in ("nodo_inicial", "nodo_final"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in ("usuarios", "usuarios_especiales"):
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "distancia" not in df.columns:
        df["distancia"] = 0.0
    df["distancia"] = pd.to_numeric(df["distancia"], errors="coerce").fillna(0.0).astype(float)

    df_nodo = df[df["nodo_inicial"] == df["nodo_final"]]
    df_tramos = df[df["nodo_inicial"] != df["nodo_final"]]

    usuarios_nodo = df_nodo.groupby("nodo_final")[["usuarios", "usuarios_especiales"]].sum()
    usuarios_acom = df_tramos.groupby("nodo_final")[["usuarios", "usuarios_especiales"]].sum()

    usuarios_por_nodo = {}
    all_nodes = set(usuarios_nodo.index.tolist()) | set(usuarios_acom.index.tolist())
    for n in all_nodes:
        u = int(usuarios_nodo.loc[n, "usuarios"]) if n in usuarios_nodo.index else 0
        ue = int(usuarios_nodo.loc[n, "usuarios_especiales"]) if n in usuarios_nodo.index else 0
        ua = int(usuarios_acom.loc[n, "usuarios"]) if n in usuarios_acom.index else 0
        uea = int(usuarios_acom.loc[n, "usuarios_especiales"]) if n in usuarios_acom.index else 0
        usuarios_por_nodo[int(n)] = {"usuarios": u + ua, "usuarios_especiales": ue + uea}

    return df_tramos, usuarios_por_nodo


def construir_grafo(df_tramos) -> nx.Graph:
    G = nx.Graph()
    for _, r in df_tramos.iterrows():
        ni = int(r["nodo_inicial"])
        nf = int(r["nodo_final"])
        d = float(r.get("distancia", 0.0) or 0.0)
        G.add_edge(ni, nf, distancia=d)
    return G


def verificar_grafo(G: nx.Graph, root: int):
    if G.number_of_nodes() == 0:
        raise ValueError("Grafo vacío: no hay tramos (ni!=nf).")
    if root not in G:
        raise ValueError(f"Nodo raíz {root} no existe en el grafo.")


# =========================
# Layout (serpiente + ramas)
# =========================

def _es_arbol(G: nx.Graph) -> bool:
    return nx.is_connected(G) and (G.number_of_edges() == G.number_of_nodes() - 1)


def _camino_troncal(G: nx.Graph, root: int) -> list[int]:
    dist, paths = nx.single_source_dijkstra(G, root, weight="distancia")
    hojas = [n for n in G.nodes() if n != root and G.degree(n) == 1]
    if not hojas:
        return [root]
    hoja_max = max(hojas, key=lambda n: dist.get(n, 0.0))
    return paths[hoja_max]


def _parent(G: nx.Graph, root: int):
    parent = {root: None}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in G.neighbors(u):
            if v in parent:
                continue
            parent[v] = u
            stack.append(v)
    return parent


def layout_serpiente(G: nx.Graph, root: int = 1, ancho: float = 5.2, salto: float = 1.8):
    nodos_reales = set(G.nodes())
    total = sum(nx.get_edge_attributes(G, "distancia").values()) or 1.0
    escala = 6.0 / (total + 1.0)

    if not _es_arbol(G):
        pos = {root: (0.0, 0.0)}
        parent = _parent(G, root)
        for n in list(parent.keys()):
            if n == root:
                continue
            p = parent[n]
            d = float(G[p][n].get("distancia", 0.0)) * escala
            x0, y0 = pos[p]
            pos[n] = (x0 + d, y0)
        return G.copy(), pos, nodos_reales

    troncal = _camino_troncal(G, root)
    troncal_set = set(troncal)

    GD = nx.Graph()
    pos = {root: (0.0, 0.0)}
    GD.add_nodes_from(G.nodes(data=True))

    x, y = 0.0, 0.0
    dir_ = +1
    codo_id = 0

    def codo():
        nonlocal codo_id
        codo_id += 1
        return f"CODO_{codo_id:03d}"

    def add(a, b, dm, es_codo=False):
        GD.add_node(a); GD.add_node(b)
        GD.add_edge(a, b, distancia=float(dm), es_codo=bool(es_codo))

    # troncal en serpiente con codos
    for i in range(len(troncal) - 1):
        u, v = troncal[i], troncal[i + 1]
        dm = float(G[u][v].get("distancia", 0.0) or 0.0)
        du = dm * escala

        if dir_ == +1:
            if x + du <= ancho:
                x2 = x + du; pos[v] = (x2, y); add(u, v, dm); x = x2
            else:
                a_units = ancho - x
                a_m = a_units / escala if escala else dm
                rem = max(dm - a_m, 0.0)
                c1 = codo(); pos[c1] = (ancho, y); add(u, c1, a_m)
                c2 = codo(); y -= salto; pos[c2] = (ancho, y); add(c1, c2, 0.0, True)
                dir_ = -1; x = ancho
                x2 = x - rem * escala; pos[v] = (x2, y); add(c2, v, rem); x = x2
        else:
            if x - du >= 0.0:
                x2 = x - du; pos[v] = (x2, y); add(u, v, dm); x = x2
            else:
                a_units = x
                a_m = a_units / escala if escala else dm
                rem = max(dm - a_m, 0.0)
                c1 = codo(); pos[c1] = (0.0, y); add(u, c1, a_m)
                c2 = codo(); y -= salto; pos[c2] = (0.0, y); add(c1, c2, 0.0, True)
                dir_ = +1; x = 0.0
                x2 = x + rem * escala; pos[v] = (x2, y); add(c2, v, rem); x = x2

    # ramas: 1er tramo vertical, resto horizontal
    parent = _parent(G, root)
    slots = {}

    def place(child, attach):
        k = slots.get(attach, 0); slots[attach] = k + 1
        sign = +1 if k % 2 == 0 else -1
        extra = 0.35 * k

        xa, ya = pos[attach]
        dm = float(G[attach][child].get("distancia", 0.0) or 0.0)
        pos[child] = (xa, ya + sign * (dm * escala + extra))
        add(attach, child, dm)

        stack = [(child, attach)]
        while stack:
            u, p = stack.pop()
            xu, yu = pos[u]
            for v in G.neighbors(u):
                if v == p or v in troncal_set or v in pos:
                    continue
                dm2 = float(G[u][v].get("distancia", 0.0) or 0.0)
                pos[v] = (xu + dm2 * escala, yu)
                add(u, v, dm2)
                stack.append((v, u))

    for t in troncal:
        for nb in G.neighbors(t):
            if nb in troncal_set:
                continue
            if parent.get(nb) == t and nb not in pos:
                place(nb, t)

    return GD, pos, nodos_reales


# =========================
# Texto sin solape (lanes)
# =========================

def asignar_lanes_por_x(x, lanes, ancho_colision=0.75):
    """
    lanes: list[list[float]] -> cada lane guarda xs usados
    Si x está cerca de algún x existente en un lane, colisiona.
    """
    for i, xs in enumerate(lanes):
        if all(abs(x - x2) > ancho_colision for x2 in xs):
            xs.append(x)
            return i
    lanes.append([x])
    return len(lanes) - 1


# =========================
# Render
# =========================

def render_diagrama(ax, GD, pos, nodos_reales, usuarios_por_nodo, kva_por_nodo, cap_kva, root=1):
    # edges
    for u, v, _ in GD.edges(data=True):
        if u in pos and v in pos:
            x1, y1 = pos[u]; x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.0)

    # nodes
    reales = [n for n in GD.nodes() if n in nodos_reales]
    codos = [n for n in GD.nodes() if n not in nodos_reales]
    nx.draw_networkx_nodes(GD, pos, nodelist=reales, node_size=220, node_color="lightblue", edgecolors="black", ax=ax)
    if codos:
        nx.draw_networkx_nodes(GD, pos, nodelist=codos, node_size=70, node_color="lightgray", edgecolors="black", ax=ax)

    # labels nodos
    nx.draw_networkx_labels(GD, pos, labels={n: str(n) for n in reales}, font_size=12, font_weight="bold", ax=ax)

    # distancias
    for u, v, d in GD.edges(data=True):
        dist = float(d.get("distancia", 0.0) or 0.0)
        if dist <= 0 or u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, f"{dist:.1f} m", fontsize=11, color="red", ha="center")

    # transformador (solo símbolo)
    if root in pos:
        x, y = pos[root]
        xt, yt = x - 0.9, y
        ax.scatter([xt], [yt], marker="^", s=260, c="orange", edgecolors="black")
        ax.text(xt - 0.15, yt, f"Transformador\n{cap_kva} kVA", fontsize=9, ha="right", va="center",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    # usuarios/kVA: carriles en banda inferior (SIN solape)
    xs = [p[0] for p in pos.values()] if pos else [0.0]
    ys = [p[1] for p in pos.values()] if pos else [0.0]
    y_min = min(ys) if ys else 0.0

    base_y = y_min - 0.65      # banda inferior dentro del mismo gráfico
    lane_step = 0.42           # separación vertical entre carriles
    lanes = []                 # xs ocupados por carril

    for n in sorted(usuarios_por_nodo.keys()):
        if n not in nodos_reales or n not in pos:
            continue
        u = int(usuarios_por_nodo[n].get("usuarios", 0) or 0)
        ue = int(usuarios_por_nodo[n].get("usuarios_especiales", 0) or 0)
        if u <= 0 and ue <= 0:
            continue

        x, y = pos[n]
        ax.plot([x, x], [y, y - 0.35], "--", color="gray", linewidth=1)

        kva = float(kva_por_nodo.get(n, 0.0)) if isinstance(kva_por_nodo, dict) else 0.0
        txt = f"Usuarios: {u}" + (f"\nDemanda: {kva:.1f} kVA" if kva > 0 else "")
        if ue > 0:
            txt += f"\nEspeciales: {ue}"

        lane = asignar_lanes_por_x(x, lanes, ancho_colision=0.85)
        yt = base_y - lane * lane_step

        ax.text(
            x, yt, txt,
            fontsize=11, color="blue",
            ha="center", va="top",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2),
        )


def fig_to_rl_image(fig, width=5*inch, height=3*inch):
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    buf.seek(0)
    img = Image(buf, width=width, height=height)
    img.hAlign = "CENTER"
    return img


# =========================
# API
# =========================

def crear_grafico_nodos_df(df_conexiones, capacidad_transformador, nodo_raiz: int = 1, tabla_potencia=None):
    df_tramos, usuarios_por_nodo = separar_tramos_y_usuarios(df_conexiones)
    G = construir_grafo(df_tramos)
    verificar_grafo(G, nodo_raiz)

    GD, pos, nodos_reales = layout_serpiente(G, root=nodo_raiz, ancho=5.2, salto=1.8)
    kva_por_nodo = calcular_kva_por_nodo(tabla_potencia) if tabla_potencia is not None else {}

    fig, ax = plt.subplots(figsize=(10, 6))

    # limites fijos antes de render (estabiliza todo)
    xs = [p[0] for p in pos.values()] if pos else [0.0]
    ys = [p[1] for p in pos.values()] if pos else [0.0]
    pad, extra_left = 0.9, 1.3
    ax.set_xlim(min(xs) - (pad + extra_left), max(xs) + pad)

    # ✅ el piso se deja con margen para banda de texto, SIN cambiar figsize ni PDF
    y_min = min(ys) if ys else 0.0
    y_max = max(ys) if ys else 0.0
    ax.set_ylim(y_min - 2.2, y_max + 0.9)  # reserva 2.2 para lanes
    ax.set_aspect("equal", adjustable="box")

    render_diagrama(ax, GD, pos, nodos_reales, usuarios_por_nodo, kva_por_nodo, capacidad_transformador, root=nodo_raiz)

    ax.set_title("Diagrama de Nodos del Transformador")
    ax.axis("off")

    img = fig_to_rl_image(fig, width=5*inch, height=3*inch)
    plt.close(fig)
    return img


def crear_grafico_nodos(nodos_inicio, nodos_final, usuarios, distancias, capacidad_transformador, df_conexiones=None, tabla_potencia=None):
    import pandas as pd
    if df_conexiones is None:
        df_conexiones = pd.DataFrame({
            "nodo_inicial": list(nodos_inicio),
            "nodo_final": list(nodos_final),
            "usuarios": list(usuarios),
            "distancia": list(distancias),
        })
    else:
        if "distancia" not in df_conexiones.columns and distancias is not None:
            df_conexiones = df_conexiones.copy()
            df_conexiones["distancia"] = list(distancias)

    return crear_grafico_nodos_df(df_conexiones, capacidad_transformador, nodo_raiz=1, tabla_potencia=tabla_potencia)


def crear_grafico_nodos_desde_archivo(ruta_excel: str):
    if cargar_datos_circuito is None:
        raise ImportError("No se encontró cargar_datos_circuito. Revisa modulos/datos.py")

    (df_conexiones, _df_param, _df_info, _tipo, _area, cap_kva, *_resto) = cargar_datos_circuito(ruta_excel)
    return crear_grafico_nodos_df(df_conexiones, cap_kva, nodo_raiz=1, tabla_potencia=df_conexiones)
