# -*- coding: utf-8 -*-
"""
modulos/graficos_red.py

Diagrama ortogonal compacto, determinista.
Salida: imagen (PNG) embebida en ReportLab.

Flujo:
Entradas -> Normalización -> Layout -> Render (diagrama + tabla) -> Salida

Reglas:
- Tramos: ni != nf (con distancia)
- Usuarios en nodo: ni == nf (sin tramo)
- Usuarios por nodo: se consolidan (nodo + acometidas)
- Troncal (camino root->hoja más largo) se pliega en serpiente (U/C)
- Ramas ortogonales colgadas desde la troncal
- Quiebres insertan nodos CODO_* (solo dibujo)
- Transformador cerca del nodo 1, sin línea
- Usuarios/Demanda NO van pegados al nodo: van en tabla inferior dentro de la misma imagen
"""

from __future__ import annotations

import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from reportlab.platypus import Image
from reportlab.lib.units import inch

try:
    from modulos.datos import cargar_datos_circuito
except Exception:
    cargar_datos_circuito = None


# ============================================================
# Normalización
# ============================================================

def calcular_kva_por_nodo(tabla_potencia) -> dict[int, float]:
    """
    Suma kVA por nodo_final desde una tabla de cargas.
    Usa 'kva_total' si existe; si no, usa 'kva'. Si no hay columnas, retorna {}.
    """
    import pandas as pd

    if tabla_potencia is None:
        return {}

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
    """
    Devuelve:
      - df_tramos: filas ni!=nf, con distancia
      - usuarios_por_nodo: dict[n]={"usuarios":..,"usuarios_especiales":..}
    """
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

    # 1) Usuarios en nodo (ni==nf)
    df_nodo = df[df["nodo_inicial"] == df["nodo_final"]]
    usuarios_nodo = df_nodo.groupby("nodo_final")[["usuarios", "usuarios_especiales"]].sum()

    # 2) Tramos reales
    df_tramos = df[df["nodo_inicial"] != df["nodo_final"]]

    # 3) Acometidas (usuarios por nodo_final en tramos)
    usuarios_acom = df_tramos.groupby("nodo_final")[["usuarios", "usuarios_especiales"]].sum()

    # 4) Merge final
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
    """Grafo solo con tramos reales (ni!=nf)."""
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


# ============================================================
# Layout (serpiente + ramas)
# ============================================================

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
    """
    Devuelve:
      GD: grafo dibujo (incluye CODO_*)
      pos: dict posiciones
      nodos_reales: set[int]
    """
    nodos_reales = set(G.nodes())
    total = sum(nx.get_edge_attributes(G, "distancia").values()) or 1.0
    escala = 6.0 / (total + 1.0)

    # Caso no radial: layout simple a la derecha
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
        GD.add_node(a)
        GD.add_node(b)
        GD.add_edge(a, b, distancia=float(dm), es_codo=bool(es_codo))

    # Troncal serpiente con codos
    for i in range(len(troncal) - 1):
        u, v = troncal[i], troncal[i + 1]
        dm = float(G[u][v].get("distancia", 0.0) or 0.0)
        du = dm * escala

        if dir_ == +1:
            if x + du <= ancho:
                x2 = x + du
                pos[v] = (x2, y)
                add(u, v, dm)
                x = x2
            else:
                a_units = ancho - x
                a_m = a_units / escala if escala else dm
                rem = max(dm - a_m, 0.0)

                c1 = codo()
                pos[c1] = (ancho, y)
                add(u, c1, a_m)

                c2 = codo()
                y -= salto
                pos[c2] = (ancho, y)
                add(c1, c2, 0.0, True)

                dir_ = -1
                x = ancho

                x2 = x - rem * escala
                pos[v] = (x2, y)
                add(c2, v, rem)
                x = x2
        else:
            if x - du >= 0.0:
                x2 = x - du
                pos[v] = (x2, y)
                add(u, v, dm)
                x = x2
            else:
                a_units = x
                a_m = a_units / escala if escala else dm
                rem = max(dm - a_m, 0.0)

                c1 = codo()
                pos[c1] = (0.0, y)
                add(u, c1, a_m)

                c2 = codo()
                y -= salto
                pos[c2] = (0.0, y)
                add(c1, c2, 0.0, True)

                dir_ = +1
                x = 0.0

                x2 = x + rem * escala
                pos[v] = (x2, y)
                add(c2, v, rem)
                x = x2

    # Ramas: 1er tramo vertical, resto horizontal
    parent = _parent(G, root)
    slots = {}

    def place(child, attach):
        k = slots.get(attach, 0)
        slots[attach] = k + 1
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


# ============================================================
# Render (diagrama arriba + tabla abajo)
# ============================================================

def render_diagrama(ax, GD, pos, nodos_reales, cap_kva, root=1):
    # Edges
    for u, v, _ in GD.edges(data=True):
        if u in pos and v in pos:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.0)

    # Nodes
    reales = [n for n in GD.nodes() if n in nodos_reales]
    codos = [n for n in GD.nodes() if n not in nodos_reales]

    nx.draw_networkx_nodes(
        GD, pos,
        nodelist=reales,
        node_size=220,
        node_color="lightblue",
        edgecolors="black",
        ax=ax
    )
    if codos:
        nx.draw_networkx_nodes(
            GD, pos,
            nodelist=codos,
            node_size=70,
            node_color="lightgray",
            edgecolors="black",
            ax=ax
        )

    # Labels nodos reales
    nx.draw_networkx_labels(
        GD, pos,
        labels={n: str(n) for n in reales},
        font_size=12,
        font_weight="bold",
        ax=ax
    )

    # Distancias
    for u, v, d in GD.edges(data=True):
        dist = float(d.get("distancia", 0.0) or 0.0)
        if dist <= 0 or u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.15,
            f"{dist:.1f} m",
            fontsize=11,
            color="red",
            ha="center",
        )

    # Transformador
    if root in pos:
        x, y = pos[root]
        xt, yt = x - 0.9, y
        ax.scatter([xt], [yt], marker="^", s=260, c="orange", edgecolors="black")
        ax.text(
            xt - 0.15, yt,
            f"Transformador\n{cap_kva} kVA",
            fontsize=9,
            ha="right",
            va="center",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )


def dibujar_tabla_cargas(ax_tbl, usuarios_por_nodo: dict[int, dict], kva_por_nodo: dict[int, float]):
    """
    Tabla dentro de la figura (matplotlib).
    Incluye solo nodos con usuarios/kVA/especiales > 0.
    """
    ax_tbl.axis("off")

    filas = []
    for n in sorted(set(usuarios_por_nodo.keys()) | set(kva_por_nodo.keys())):
        u = int(usuarios_por_nodo.get(n, {}).get("usuarios", 0) or 0)
        ue = int(usuarios_por_nodo.get(n, {}).get("usuarios_especiales", 0) or 0)
        kva = float(kva_por_nodo.get(n, 0.0) or 0.0)
        if u <= 0 and ue <= 0 and kva <= 0:
            continue
        filas.append([str(n), str(u), f"{kva:.1f}", str(ue)])

    if not filas:
        ax_tbl.text(0.5, 0.5, "Sin cargas por nodo.", ha="center", va="center", fontsize=10)
        return

    col_labels = ["Nodo", "Usuarios", "Demanda (kVA)", "Especiales"]

    tbl = ax_tbl.table(
        cellText=filas,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.05, 0.0, 0.90, 1.0],
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.2)

    ax_tbl.set_title("Detalle de Cargas por Nodo", fontsize=10, pad=4)


def fig_to_rl_image(fig, width=5 * inch, height=3 * inch):
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    buf.seek(0)
    img = Image(buf, width=width, height=height)
    img.hAlign = "CENTER"
    return img


# ============================================================
# API principal
# ============================================================

def crear_grafico_nodos_df(df_conexiones, capacidad_transformador, nodo_raiz: int = 1, tabla_potencia=None):
    df_tramos, usuarios_por_nodo = separar_tramos_y_usuarios(df_conexiones)
    G = construir_grafo(df_tramos)
    verificar_grafo(G, nodo_raiz)

    GD, pos, nodos_reales = layout_serpiente(G, root=nodo_raiz, ancho=5.2, salto=1.8)
    kva_por_nodo = calcular_kva_por_nodo(tabla_potencia) if tabla_potencia is not None else {}

    # Figura con 2 áreas: diagrama + tabla (misma imagen)
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1.2], hspace=0.12)

    ax = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    # Límites del diagrama (arriba)
    xs = [p[0] for p in pos.values()] if pos else [0.0]
    ys = [p[1] for p in pos.values()] if pos else [0.0]
    pad, extra_left = 0.9, 1.3
    ax.set_xlim(min(xs) - (pad + extra_left), max(xs) + pad)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    ax.set_aspect("equal", adjustable="box")

    render_diagrama(ax, GD, pos, nodos_reales, capacidad_transformador, root=nodo_raiz)
    ax.set_title("Diagrama de Nodos del Transformador")
    ax.axis("off")

    # Tabla inferior (misma imagen)
    dibujar_tabla_cargas(ax_tbl, usuarios_por_nodo, kva_por_nodo)

    img = fig_to_rl_image(fig, width=5 * inch, height=3 * inch)
    plt.close(fig)
    return img


def crear_grafico_nodos(
    nodos_inicio,
    nodos_final,
    usuarios,
    distancias,
    capacidad_transformador,
    df_conexiones=None,
    tabla_potencia=None,
):
    """
    Wrapper retrocompatible:
    - Si df_conexiones viene, se usa.
    - Si no, construye df mínimo con listas.
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
        if "distancia" not in df_conexiones.columns and distancias is not None:
            df_conexiones = df_conexiones.copy()
            df_conexiones["distancia"] = list(distancias)

    return crear_grafico_nodos_df(
        df_conexiones=df_conexiones,
        capacidad_transformador=capacidad_transformador,
        nodo_raiz=1,
        tabla_potencia=tabla_potencia,
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
        tabla_potencia=df_conexiones,
    )
