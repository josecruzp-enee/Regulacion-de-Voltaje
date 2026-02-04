# -*- coding: utf-8 -*-
"""
modulos/graficos_red.py

Diagrama estilo plano (esquemático tipo AutoCAD):
- Troncal horizontal (vecino del nodo 1 con mayor distancia acumulada hacia una hoja).
- Otras ramas salen verticales desde el nodo 1.
- Todo ortogonal (sin diagonales).
- Distancias proporcionales (escala real).
- Nodo 1 igual que los demás.
- Transformador a la par del nodo 1 (SIN línea de conexión).
- Acometidas con anti-solape.

NOTA DE SESIÓN (para dejar constancia):
- El usuario terminó insatisfecho y con sensación de haber perdido el día por iteraciones fallidas.
- Este archivo busca ser una versión final robusta (sin KeyError y sin duplicaciones), lista para producción.
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


# -----------------------------
# Layout plano real (robusto)
# -----------------------------

def calcular_posiciones_plano_real(G: nx.Graph, nodo_raiz: int = 1, escala=None) -> dict[int, tuple[float, float]]:
    """
    Layout ortogonal a escala REAL:
    - Elige troncal como el vecino del nodo raíz que lleva a la mayor distancia acumulada hacia una hoja.
    - Troncal en horizontal (y=0).
    - Otras ramas salen verticales desde la raíz (x=0) con y=distancia*escala.
    - Dentro de cada rama, se avanza a la derecha con distancia*escala.
    Robusto: no usa estructuras tipo 'hijos[p]' que puedan reventar por llaves faltantes.
    """
    if escala is None:
        total = sum(nx.get_edge_attributes(G, "distancia").values()) or 1.0
        escala = 6.0 / (total + 1.0)

    if nodo_raiz not in G:
        return {}

    pos: dict[int, tuple[float, float]] = {nodo_raiz: (0.0, 0.0)}
    vecinos = sorted(G.neighbors(nodo_raiz))
    if not vecinos:
        return pos

    def max_dist_en_subarbol(vecino: int) -> float:
        """
        DFS desde 'vecino' evitando regresar a la raíz.
        Retorna la mayor distancia acumulada a cualquier nodo alcanzable.
        """
        best = float(G[nodo_raiz][vecino].get("distancia", 0.0) or 0.0)
        stack = [(vecino, nodo_raiz, best)]
        seen = set([nodo_raiz])

        while stack:
            n, parent, dist_acc = stack.pop()
            seen.add(n)
            if dist_acc > best:
                best = dist_acc

            for w in G.neighbors(n):
                if w == parent:
                    continue
                if w in seen:
                    continue
                d = float(G[n][w].get("distancia", 0.0) or 0.0)
                stack.append((w, n, dist_acc + d))

        return best

    # Troncal: vecino con mayor distancia acumulada
    v_troncal = max(vecinos, key=max_dist_en_subarbol)

    # Colocar troncal horizontal desde la raíz
    d_tr = float(G[nodo_raiz][v_troncal].get("distancia", 0.0) or 0.0) * escala
    pos[v_troncal] = (d_tr, 0.0)

    # Otras ramas verticales desde la raíz (a escala real)
    otros = [v for v in vecinos if v != v_troncal]
    sign = +1
    for v in otros:
        dv = float(G[nodo_raiz][v].get("distancia", 0.0) or 0.0) * escala
        pos[v] = (0.0, sign * dv)
        sign *= -1

    # Propagar cada rama hacia la derecha (horizontal) con distancias reales
    def propagar(p: int, parent: int | None):
        x0, y0 = pos[p]
        for h in sorted(G.neighbors(p)):
            if parent is not None and h == parent:
                continue
            if h in pos:
                continue
            dd = float(G[p][h].get("distancia", 0.0) or 0.0) * escala
            pos[h] = (x0 + dd, y0)
            propagar(h, parent=p)

    propagar(v_troncal, parent=nodo_raiz)
    for v in otros:
        propagar(v, parent=nodo_raiz)

    return pos


# -----------------------------
# Dibujo
# -----------------------------

def dibujar_aristas_ortogonales(ax, G: nx.Graph, pos: dict[int, tuple[float, float]], lw: float = 2.0):
    """
    Aristas ortogonales (sin diagonales).
    Si no están alineados, se dibuja:
      (x1,y1)->(x1,y2)->(x2,y2)
    """
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        if abs(x1 - x2) < 1e-9 or abs(y1 - y2) < 1e-9:
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=lw)
        else:
            ax.plot([x1, x1], [y1, y2], color="black", linewidth=lw)
            ax.plot([x1, x2], [y2, y2], color="black", linewidth=lw)


def dibujar_nodos(ax, G: nx.Graph, pos: dict[int, tuple[float, float]]):
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(G.nodes),
        node_size=220,
        node_color="lightblue",
        edgecolors="black",
        ax=ax,
    )


def dibujar_etiquetas_nodos(ax, G: nx.Graph, pos: dict[int, tuple[float, float]]):
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: str(n) for n in G.nodes},
        font_size=12,
        font_weight="bold",
        ax=ax,
    )


def dibujar_acometidas(ax, posiciones: dict, df_conexiones, omitir_nodos: set[int] | None = None):
    """
    Acometidas con anti-solape:
    - Stagger en X cuando nodos están cerca.
    - Apilado en Y por “zona de X”.
    """
    omitir_nodos = omitir_nodos or set()

    y_linea = 0.25
    y_texto_1 = 0.08
    y_stack_gap = 0.22
    x_thresh = 0.35
    x_stagger = 0.22

    items = []
    for _, row in df_conexiones.iterrows():
        nf = int(row["nodo_final"])
        if nf in omitir_nodos:
            continue
        if nf not in posiciones:
            continue
        normales = int(row.get("usuarios", 0) or 0)
        especiales = int(row.get("usuarios_especiales", 0) or 0)
        x, y = posiciones[nf]
        items.append((nf, x, y, normales, especiales))

    items.sort(key=lambda t: t[1])

    stacks: list[tuple[float, float | None]] = []  # (x_ref, next_y_text)

    def _get_stack(x):
        for i, (xr, ny) in enumerate(stacks):
            if abs(x - xr) <= x_thresh:
                return i
        stacks.append((x, None))
        return len(stacks) - 1

    for idx, (nf, x, y, normales, especiales) in enumerate(items):
        dx = 0.0
        if idx > 0:
            x_prev = items[idx - 1][1]
            if abs(x - x_prev) <= x_thresh:
                dx = x_stagger if (idx % 2 == 0) else -x_stagger

        y2 = y - y_linea
        ax.plot([x, x], [y, y2], "--", color="gray", linewidth=1)

        si = _get_stack(x)
        xr, next_y = stacks[si]

        y_text = y2 - y_texto_1
        if next_y is None:
            stacks[si] = (xr, y_text - y_stack_gap)
        else:
            if y_text > next_y:
                y_text = next_y
            stacks[si] = (xr, y_text - y_stack_gap)

        ax.text(
            x + dx, y_text,
            f"Usuarios: {normales}",
            fontsize=11, color="blue",
            ha="center", va="top",
        )

        if especiales > 0:
            ax.text(
                x + dx, y_text - 0.15,
                f"Especiales: {especiales}",
                fontsize=11, color="red",
                ha="center", va="top",
            )


def dibujar_distancias_tramos(ax, G: nx.Graph, pos: dict[int, tuple[float, float]]):
    for u, v, d in G.edges(data=True):
        dist = float(d.get("distancia", 0.0) or 0.0)
        if dist <= 0:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        ax.text(xm, ym + 0.15, f"{dist:.1f} m", fontsize=11, color="red", ha="center")


def dibujar_transformador_sin_linea(
    ax,
    pos: dict[int, tuple[float, float]],
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
    G = crear_grafo(nodos_inicio, nodos_final, usuarios, distancias)

    info = verificar_grafo(G, nodo_raiz=1)
    if not info.get("ok", True):
        print("⚠️ Grafo no completamente conectado:", info)

    posiciones = calcular_posiciones_plano_real(G, nodo_raiz=1)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    dibujar_aristas_ortogonales(ax, G, posiciones, lw=2.0)
    dibujar_nodos(ax, G, posiciones)
    dibujar_etiquetas_nodos(ax, G, posiciones)

    # SI querés acometidas en el nodo 1, NO lo omitimos:
    dibujar_acometidas(ax, posiciones, df_conexiones, omitir_nodos=set())

    dibujar_distancias_tramos(ax, G, posiciones)

    dibujar_transformador_sin_linea(ax, posiciones, capacidad_transformador, nodo=1, dx=-0.9, dy=0.0)

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
