# -*- coding: utf-8 -*-
"""
modulos/graficos_red.py

Diagrama estilo plano (esquemático tipo AutoCAD) - versión robusta:
- Radial (árbol): topología secundaria típica.
- Troncal plegable tipo U/C (serpiente) para caber en el PDF.
- Todo ortogonal (sin diagonales).
- Si hay quiebre (L), se inserta un nodo "CODO_*" (solo para dibujo).
- Distancias proporcionales (escala) y preservadas al partir tramos por pliegue.
- Nodo 1 igual que los demás.
- Transformador a la par del nodo 1 (SIN línea de conexión).
- Acometidas con anti-solape y sin duplicaciones (groupby nodo_final).
"""

from __future__ import annotations

import io
import math
import matplotlib.pyplot as plt
import networkx as nx
from reportlab.platypus import Image
from reportlab.lib.units import inch

try:
    from modulos.datos import cargar_datos_circuito
except Exception:
    cargar_datos_circuito = None


# ============================================================
# Entradas -> Validación -> Cálculos -> Salidas
# ============================================================

def crear_grafo(nodos_inicio, nodos_final, usuarios, distancias) -> nx.Graph:
    """
    Construye grafo.
    Caso especial: fila ni==nf (ej. 1->1, dist=0) se interpreta como usuarios en el nodo (no tramo).
    """
    G = nx.Graph()

    all_nodes = set(int(x) for x in list(nodos_inicio) + list(nodos_final))
    for n in all_nodes:
        G.add_node(int(n), usuarios_nodo=0)

    for ni, nf, u, d in zip(nodos_inicio, nodos_final, usuarios, distancias):
        ni = int(ni)
        nf = int(nf)
        u = int(u)
        d = float(d)

        if ni == nf:
            G.nodes[ni]["usuarios_nodo"] = G.nodes[ni].get("usuarios_nodo", 0) + u
            continue

        G.add_edge(ni, nf, usuarios=u, distancia=d)

    return G


def verificar_grafo(G: nx.Graph, nodo_raiz: int = 1) -> dict:
    if nodo_raiz not in G:
        return {"ok": False, "error": f"Nodo raíz {nodo_raiz} no existe.", "nodos": []}

    if G.number_of_nodes() == 0:
        return {"ok": False, "error": "Grafo vacío.", "nodos": []}

    alcanzables = set(nx.node_connected_component(G, nodo_raiz)) if G.number_of_edges() > 0 else {nodo_raiz}
    todos = set(G.nodes())

    return {
        "ok": alcanzables == todos,
        "nodos": sorted(todos),
        "vecinos_raiz": sorted(list(G.neighbors(nodo_raiz))) if nodo_raiz in G else [],
        "desconectados": sorted(list(todos - alcanzables)),
        "aristas": sorted((int(u), int(v)) for u, v in G.edges()),
    }


# ============================================================
# Layout: Troncal radial + plegado U/C con nodos CODO
# ============================================================

def _es_arbol_radial(G: nx.Graph, nodo_raiz: int = 1) -> bool:
    """
    Radial = conectado y sin ciclos (árbol).
    """
    if nodo_raiz not in G:
        return False
    if G.number_of_edges() == 0:
        return True
    if not nx.is_connected(G):
        return False
    # árbol: |E| = |V|-1
    return G.number_of_edges() == G.number_of_nodes() - 1


def _camino_troncal_mas_largo(G: nx.Graph, nodo_raiz: int = 1) -> list[int]:
    """
    En árbol radial, el troncal se toma como el camino desde la raíz hasta la hoja
    con mayor distancia acumulada (ponderado por 'distancia').
    """
    # Dijkstra desde la raíz
    dist, paths = nx.single_source_dijkstra(G, nodo_raiz, weight="distancia")
    # hoja: grado 1 (excepto raíz)
    hojas = [n for n in G.nodes() if n != nodo_raiz and G.degree(n) == 1]
    if not hojas:
        return [nodo_raiz]

    hoja_max = max(hojas, key=lambda n: dist.get(n, 0.0))
    return paths[hoja_max]


def _build_parent_order(G: nx.Graph, root: int) -> tuple[dict[int, int | None], list[int]]:
    parent: dict[int, int | None] = {root: None}
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


def construir_grafo_dibujo_serpiente(
    G: nx.Graph,
    nodo_raiz: int = 1,
    escala: float | None = None,
    ancho_max_units: float = 3.6,   # ancho "útil" para plegar (en unidades del plano)
    salto_fila_units: float = 1.2,  # separación vertical entre filas de la "U/C"
) -> tuple[nx.Graph, dict, set[int]]:
    """
    Devuelve:
    - GD: grafo para dibujar (incluye nodos CODO_*).
    - pos: posiciones para GD.
    - nodos_reales: set con nodos del grafo original (para etiquetas y acometidas).
    """
    if nodo_raiz not in G:
        return nx.Graph(), {}, set()

    # Escala automática: que el total no explote
    if escala is None:
        total = sum(nx.get_edge_attributes(G, "distancia").values()) or 1.0
        escala = 6.0 / (total + 1.0)

    nodos_reales = set(G.nodes())

    # Si no es radial (árbol), caemos a un dibujo simple sin nodos codo
    if not _es_arbol_radial(G, nodo_raiz=nodo_raiz):
        GD = G.copy()
        pos = {nodo_raiz: (0.0, 0.0)}
        # layout simple tipo BFS a la derecha
        parent, order = _build_parent_order(GD, nodo_raiz)
        for n in order:
            if n == nodo_raiz:
                continue
            p = parent[n]
            if p is None:
                continue
            dp = float(GD[p][n].get("distancia", 0.0) or 0.0) * escala
            x0, y0 = pos.get(p, (0.0, 0.0))
            pos[n] = (x0 + dp, y0)
        return GD, pos, nodos_reales

    troncal = _camino_troncal_mas_largo(G, nodo_raiz=nodo_raiz)
    troncal_set = set(troncal)

    GD = nx.Graph()
    GD.add_nodes_from(G.nodes(data=True))
    # Copiamos aristas originales, pero no las agregamos todavía: iremos insertando CODO si hace falta.
    # GD tendrá aristas "partidas" cuando haya plegado.

    pos: dict[object, tuple[float, float]] = {nodo_raiz: (0.0, 0.0)}

    # Guardamos dirección actual por nodo troncal para orientar ramas
    dir_troncal: dict[int, int] = {nodo_raiz: +1}  # +1 derecha, -1 izquierda
    fila_y = 0.0
    x = 0.0
    direccion = +1
    codo_id = 0

    def _nuevo_codo() -> str:
        nonlocal codo_id
        codo_id += 1
        return f"CODO_{codo_id:03d}"

    def _add_edge(a, b, distancia_m: float, es_codo: bool = False):
        GD.add_node(a)
        GD.add_node(b)
        GD.add_edge(a, b, distancia=float(distancia_m), es_codo=bool(es_codo))

    # ===== 1) Construir troncal serpenteante con nodos CODO en quiebres =====
    for i in range(len(troncal) - 1):
        u = troncal[i]
        v = troncal[i + 1]
        dist_m = float(G[u][v].get("distancia", 0.0) or 0.0)

        # Dirección para este tramo (persistente por fila)
        dir_troncal[u] = direccion

        dx_units = dist_m * escala

        # límites por fila: [0, ancho_max_units]
        if direccion == +1:
            if x + dx_units <= ancho_max_units:
                # cabe directo
                x2 = x + dx_units
                pos[v] = (x2, fila_y)
                _add_edge(u, v, dist_m, es_codo=False)
                x = x2
            else:
                # Partimos: u -> CODO (hasta borde) -> CODO (baja fila) -> v (sigue en sentido inverso)
                # tramo hasta borde
                faltan_units = (x + dx_units) - ancho_max_units
                a_borde_units = dx_units - faltan_units
                a_borde_m = a_borde_units / escala if escala else dist_m
                rem_m = max(dist_m - a_borde_m, 0.0)

                c1 = _nuevo_codo()
                pos[c1] = (ancho_max_units, fila_y)
                _add_edge(u, c1, a_borde_m, es_codo=False)

                c2 = _nuevo_codo()
                fila_y -= salto_fila_units
                pos[c2] = (ancho_max_units, fila_y)
                _add_edge(c1, c2, 0.0, es_codo=True)

                direccion = -1
                dir_troncal[c2] = direccion
                x = ancho_max_units

                # ahora avanzamos a la izquierda remanente
                x2 = x - (rem_m * escala)
                pos[v] = (x2, fila_y)
                _add_edge(c2, v, rem_m, es_codo=False)
                x = x2
        else:
            # direccion == -1
            if x - dx_units >= 0.0:
                x2 = x - dx_units
                pos[v] = (x2, fila_y)
                _add_edge(u, v, dist_m, es_codo=False)
                x = x2
            else:
                faltan_units = (dx_units - x)  # cuánto se sale por la izquierda
                a_borde_units = dx_units - faltan_units
                a_borde_m = a_borde_units / escala if escala else dist_m
                rem_m = max(dist_m - a_borde_m, 0.0)

                c1 = _nuevo_codo()
                pos[c1] = (0.0, fila_y)
                _add_edge(u, c1, a_borde_m, es_codo=False)

                c2 = _nuevo_codo()
                fila_y -= salto_fila_units
                pos[c2] = (0.0, fila_y)
                _add_edge(c1, c2, 0.0, es_codo=True)

                direccion = +1
                dir_troncal[c2] = direccion
                x = 0.0

                x2 = x + (rem_m * escala)
                pos[v] = (x2, fila_y)
                _add_edge(c2, v, rem_m, es_codo=False)
                x = x2

    dir_troncal[troncal[-1]] = direccion

    # ===== 2) Colgar ramas (no troncal) ortogonales desde nodos reales del troncal =====
    # Construimos padre para recorrer árbol desde la raíz
    parent, order = _build_parent_order(G, nodo_raiz)

    # Para evitar solapes entre ramas en el mismo punto, alternamos signo y acumulamos "nivel"
    rama_slot: dict[int, int] = {}  # nodo_troncal -> contador

    def _dir_en_nodo_troncal(n: int) -> int:
        # Si ese nodo está en pos, usamos la dirección de su fila (dir_troncal); si no, +1
        return dir_troncal.get(n, +1)

    def _place_subtree(start: int, attach: int):
        """
        Coloca subárbol que cuelga de 'attach' (nodo en troncal) comenzando por 'start' (hijo no troncal).
        Primer tramo vertical, luego los siguientes tramos siguen horizontal en dirección de fila.
        """
        # slot vertical para separar múltiples ramas
        k = rama_slot.get(attach, 0)
        rama_slot[attach] = k + 1
        sign = +1 if (k % 2 == 0) else -1
        extra_sep = 0.35 * k  # separación adicional (unidades) para no pisarse

        xa, ya = pos.get(attach, (0.0, 0.0))
        d0 = float(G[attach][start].get("distancia", 0.0) or 0.0) * escala
        y_start = ya + sign * (d0 + extra_sep)

        pos[start] = (xa, y_start)
        _add_edge(attach, start, float(G[attach][start].get("distancia", 0.0) or 0.0), es_codo=False)

        # DFS desde start (sin volver a attach)
        stack = [(start, attach)]
        while stack:
            u, p = stack.pop()
            xu, yu = pos.get(u, (xa, y_start))
            for v in G.neighbors(u):
                if v == p:
                    continue
                if v in troncal_set:
                    continue
                if v in pos:
                    continue
                dist_m = float(G[u][v].get("distancia", 0.0) or 0.0)
                d_units = dist_m * escala
                direc = _dir_en_nodo_troncal(attach)
                pos[v] = (xu + direc * d_units, yu)  # horizontal en dirección de la fila
                _add_edge(u, v, dist_m, es_codo=False)
                stack.append((v, u))

    # Recorremos nodos de la troncal (reales) y colgamos hijos no troncal
    for t in troncal:
        for nb in G.neighbors(t):
            if nb in troncal_set:
                continue
            # nb cuelga de t
            if parent.get(nb) == t or parent.get(t) == nb:
                # aseguramos que sea "hijo" (en árbol, basta con evitar duplicados)
                if nb not in pos:
                    _place_subtree(nb, attach=t)

    return GD, pos, nodos_reales


# ============================================================
# Dibujo
# ============================================================

def dibujar_aristas_rectas(ax, GD: nx.Graph, pos: dict, lw: float = 2.0):
    """
    Dibuja todas las aristas como segmentos rectos (ya que los quiebres son nodos CODO).
    """
    for u, v, d in GD.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=lw)


def dibujar_nodos(ax, GD: nx.Graph, pos: dict, nodos_reales: set[int]):
    reales = [n for n in GD.nodes() if n in nodos_reales]
    codos = [n for n in GD.nodes() if n not in nodos_reales]

    nx.draw_networkx_nodes(
        GD, pos,
        nodelist=reales,
        node_size=220,
        node_color="lightblue",
        edgecolors="black",
        ax=ax,
    )
    if codos:
        nx.draw_networkx_nodes(
            GD, pos,
            nodelist=codos,
            node_size=70,
            node_color="lightgray",
            edgecolors="black",
            ax=ax,
        )


def dibujar_etiquetas_nodos(ax, GD: nx.Graph, pos: dict, nodos_reales: set[int]):
    labels = {n: str(n) for n in GD.nodes() if n in nodos_reales}
    nx.draw_networkx_labels(
        GD, pos,
        labels=labels,
        font_size=12,
        font_weight="bold",
        ax=ax,
    )


def dibujar_acometidas(ax, posiciones: dict, df_conexiones, nodos_reales: set[int], omitir_nodos: set[int] | None = None):
    """
    Acometidas anti-solape (robusto):
    - Agrupa por nodo_final (evita duplicados).
    - Apila textos por "columna" (cercanía en X) para que no se monten.
    - Solo dibuja en nodos reales (no CODO).
    """
    import pandas as pd

    omitir_nodos = omitir_nodos or set()

    # Parámetros de layout (ajustables)
    y_linea = 0.25        # largo de la línea punteada hacia abajo
    y_texto_1 = 0.08      # separación inicial texto respecto a la línea
    y_stack_gap = 0.22    # separación vertical entre textos apilados
    x_thresh = 0.35       # umbral para considerar misma "columna" (cercanía en X)
    x_stagger = 0.22      # zig-zag opcional para nodos muy pegados en X

    df = df_conexiones.copy()

    if "usuarios_especiales" not in df.columns:
        df["usuarios_especiales"] = 0

    df["nodo_final"] = pd.to_numeric(df["nodo_final"], errors="coerce").fillna(0).astype(int)
    df["usuarios"] = pd.to_numeric(df.get("usuarios", 0), errors="coerce").fillna(0).astype(int)
    df["usuarios_especiales"] = pd.to_numeric(df.get("usuarios_especiales", 0), errors="coerce").fillna(0).astype(int)

    # ✅ Agrupación real (no duplicados)
    df_agg = (
        df.groupby("nodo_final", as_index=False)[["usuarios", "usuarios_especiales"]]
          .sum()
    )

    items = []
    for _, row in df_agg.iterrows():
        nf = int(row["nodo_final"])
        if nf in omitir_nodos:
            continue
        if nf not in nodos_reales:
            continue
        if nf not in posiciones:
            continue

        x, y = posiciones[nf]
        normales = int(row["usuarios"])
        especiales = int(row["usuarios_especiales"])
        items.append((nf, x, y, normales, especiales))

    # Ordenar por X para formar “columnas”
    items.sort(key=lambda t: t[1])

    # stacks: lista de columnas -> (x_ref, next_y_text)
    stacks: list[tuple[float, float | None]] = []

    def _get_stack_index(x: float) -> int:
        for i, (xr, _) in enumerate(stacks):
            if abs(x - xr) <= x_thresh:
                return i
        stacks.append((x, None))
        return len(stacks) - 1

    for idx, (nf, x, y, normales, especiales) in enumerate(items):
        # pequeño zig-zag si dos nodos vienen pegados
        dx = 0.0
        if idx > 0 and abs(x - items[idx - 1][1]) <= x_thresh:
            dx = x_stagger if (idx % 2 == 0) else -x_stagger

        # línea punteada hacia abajo
        y2 = y - y_linea
        ax.plot([x, x], [y, y2], "--", color="gray", linewidth=1)

        # asignar columna y apilar
        si = _get_stack_index(x)
        xr, next_y = stacks[si]

        y_text = y2 - y_texto_1
        if next_y is None:
            stacks[si] = (xr, y_text - y_stack_gap)
        else:
            # siempre baja para no pisar
            if y_text > next_y:
                y_text = next_y
            stacks[si] = (xr, y_text - y_stack_gap)

        # texto usuarios (con fondo para legibilidad)
        ax.text(
            x + dx, y_text,
            f"Usuarios: {normales}",
            fontsize=11, color="blue",
            ha="center", va="top",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2),
        )

        if especiales > 0:
            ax.text(
                x + dx, y_text - 0.15,
                f"Especiales: {especiales}",
                fontsize=11, color="red",
                ha="center", va="top",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2),
            )



def dibujar_distancias_tramos(ax, GD: nx.Graph, pos: dict):
    """
    Dibuja distancias por tramo (omite tramos 'codo' con distancia=0).
    """
    for u, v, d in GD.edges(data=True):
        dist = float(d.get("distancia", 0.0) or 0.0)
        if dist <= 0:
            continue
        if u not in pos or v not in pos:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        ax.text(xm, ym + 0.15, f"{dist:.1f} m", fontsize=11, color="red", ha="center")


def dibujar_transformador_sin_linea(
    ax,
    pos: dict,
    capacidad_transformador,
    nodo: int = 1,
    dx: float = -0.9,
    dy: float = 0.0
):
    """
    Transformador a la par del nodo (sin línea).
    """
    if nodo not in pos:
        return
    x, y = pos[nodo]
    xt, yt = x + dx, y + dy

    ax.scatter([xt], [yt], marker="^", s=260, c="orange", edgecolors="black")
    ax.text(
        xt - 0.15, yt,
        f"Transformador\n{capacidad_transformador} kVA",
        fontsize=9, ha="right", va="center",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
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
):
    G = crear_grafo(nodos_inicio, nodos_final, usuarios, distancias)

    info = verificar_grafo(G, nodo_raiz=1)
    if not info.get("ok", True):
        print("⚠️ Grafo no completamente conectado:", info)

    GD, posiciones, nodos_reales = construir_grafo_dibujo_serpiente(
        G,
        nodo_raiz=1,
        escala=None,
        ancho_max_units=3.6,
        salto_fila_units=1.2,
    )

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    dibujar_aristas_rectas(ax, GD, posiciones, lw=2.0)
    dibujar_nodos(ax, GD, posiciones, nodos_reales=nodos_reales)
    dibujar_etiquetas_nodos(ax, GD, posiciones, nodos_reales=nodos_reales)

    # Acometidas (incluye nodo 1 si existe en df)
    dibujar_acometidas(ax, posiciones, df_conexiones, nodos_reales=nodos_reales, omitir_nodos=set())

    dibujar_distancias_tramos(ax, GD, posiciones)
    dibujar_transformador_sin_linea(ax, posiciones, capacidad_transformador, nodo=1, dx=-0.9, dy=0.0)

    plt.title("Diagrama de Nodos del Transformador")
    plt.axis("off")

    # ✅ Centrar el nodo 1 en la figura (encuadre simétrico alrededor de (0,0))
    xs = [p[0] for p in posiciones.values()] if posiciones else [0.0]
    ys = [p[1] for p in posiciones.values()] if posiciones else [0.0]
    pad = 0.9
    max_dx = max(abs(min(xs)), abs(max(xs))) + pad
    max_dy = max(abs(min(ys)), abs(max(ys))) + pad
    ax.set_xlim(-max_dx, max_dx)
    ax.set_ylim(-max_dy, max_dy)
    ax.set_aspect("equal", adjustable="box")

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    img = Image(buf, width=5 * inch, height=3 * inch)
    img.hAlign = "CENTER"
    return img


def crear_grafico_nodos_desde_archivo(ruta_excel: str):
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


