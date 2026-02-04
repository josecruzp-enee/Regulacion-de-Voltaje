# -*- coding: utf-8 -*-
"""
modulos/graficos_red.py

Diagrama estilo plano (tipo AutoCAD esquemático):
- Troncal horizontal (camino más largo desde el nodo 1).
- Ramales verticales desde el nodo de derivación.
- Todo ortogonal (sin diagonales).
- Sin "nodos fantasma": el ramal nace visualmente del nodo correcto.
- Nodo 1 igual que los demás.
- Transformador a la par del nodo 1 (SIN línea de conexión).
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
    Construye grafo radial.
    Caso especial: fila ni==nf (ej. 1->1, dist=0) se interpreta como carga/usuarios en el nodo.
    """
    G = nx.Graph()

    # asegura nodos
    all_nodes = set(int(x) for x in list(nodos_inicio) + list(nodos_final))
    for n in all_nodes:
        G.add_node(int(n), usuarios_nodo=0)

    for ni, nf, u, d in zip(nodos_inicio, nodos_final, usuarios, distancias):
        ni = int(ni)
        nf = int(nf)
        u = int(u)
        d = float(d)

        if ni == nf:
            # usuarios en el nodo, no es tramo
            G.nodes[ni]["usuarios_nodo"] = G.nodes[ni].get("usuarios_nodo", 0) + u
            continue

        G.add_edge(ni, nf, usuarios=u, distancia=d)

    return G


def verificar_grafo(G: nx.Graph, nodo_raiz: int = 1) -> dict:
    if nodo_raiz not in G:
        return {"ok": False, "error": f"Nodo raíz {nodo_raiz} no existe.", "nodos": []}

    if G.number_of_nodes() == 0:
        return {"ok": False, "error": "Grafo vacío.", "nodos": []}

    # si hay varios componentes, avisar
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
# Layout: troncal + ramales
# -----------------------------

def _camino_mas_largo_desde_raiz(G: nx.Graph, raiz: int = 1) -> list[int]:
    """
    Devuelve el camino raíz->hoja que maximiza distancia total (suma de 'distancia').
    Asume red radial (o casi radial); si hay ciclos, toma BFS como referencia.
    """
    if raiz not in G:
        return []

    # BFS para padres
    padres = {raiz: None}
    orden = [raiz]
    for u in orden:
        for v in sorted(G.neighbors(u)):
            if v in padres:
                continue
            padres[v] = u
            orden.append(v)

    hijos = {n: [] for n in padres}
    for n, p in padres.items():
        if p is not None:
            hijos[p].append(n)

    hojas = [n for n in padres if len(hijos.get(n, [])) == 0]
    if not hojas:
        return [raiz]

    def reconstruir_camino(h):
        path = []
        n = h
        while n is not None:
            path.append(n)
            n = padres[n]
        return list(reversed(path))

    def costo_dist(path):
        s = 0.0
        for a, b in zip(path[:-1], path[1:]):
            s += float(G[a][b].get("distancia", 0.0))
        return s

    mejor = [raiz]
    mejor_costo = -1.0
    for h in hojas:
        p = reconstruir_camino(h)
        c = costo_dist(p)
        if c > mejor_costo:
            mejor_costo = c
            mejor = p

    return mejor


def calcular_posiciones_troncal_ramales(
    G: nx.Graph,
    nodo_raiz: int = 1,
    dy: float = 1.8,
    escala: float | None = None,
) -> dict[int, tuple[float, float]]:
    """
    Coloca:
      - Troncal horizontal (y=0) siguiendo distancias reales acumuladas.
      - Ramales salen verticales desde el nodo troncal, y luego continúan horizontal a la derecha.
    """
    if G.number_of_nodes() == 0:
        return {}

    # escala metros->coords
    if escala is None:
        total_dist = sum(nx.get_edge_attributes(G, "distancia").values()) or 1.0
        escala = 6.0 / (total_dist + 1.0)

    troncal = _camino_mas_largo_desde_raiz(G, raiz=nodo_raiz)
    if not troncal:
        troncal = [nodo_raiz]

    set_troncal = set(troncal)

    # BFS padres/hijos para orientar ramas desde la raíz
    padres = {nodo_raiz: None}
    orden = [nodo_raiz]
    for u in orden:
        for v in sorted(G.neighbors(u)):
            if v in padres:
                continue
            padres[v] = u
            orden.append(v)

    hijos = {n: [] for n in padres}
    for n, p in padres.items():
        if p is not None:
            hijos[p].append(n)

    pos: dict[int, tuple[float, float]] = {}

    # 1) troncal horizontal
    x = 0.0
    pos[troncal[0]] = (x, 0.0)
    for a, b in zip(troncal[:-1], troncal[1:]):
        d = float(G[a][b].get("distancia", 0.0))
        x += d * escala
        pos[b] = (x, 0.0)

    # 2) ramales: alternar arriba/abajo por cada nodo troncal para que se lea bien
    sign = +1
    usados_offset: dict[int, int] = {n: 0 for n in set_troncal}

    def colocar_subarbol(padre: int, hijo: int, side_sign: int):
        px, py = pos[padre]
        d = float(G[padre][hijo].get("distancia", 0.0)) * escala

        if padre in set_troncal:
            k = usados_offset[padre]
            usados_offset[padre] += 1
            y_h = py + side_sign * dy * (k + 1)
            # primer tramo del ramal: vertical (misma x)
            pos[hijo] = (px, y_h)
        else:
            # dentro del ramal, avanzar hacia la derecha
            pos[hijo] = (px + d, py)

        for h2 in sorted(hijos.get(hijo, [])):
            colocar_subarbol(hijo, h2, side_sign)

    for n in troncal:
        for h in sorted(hijos.get(n, [])):
            if h in set_troncal:
                continue
            colocar_subarbol(n, h, sign)
            sign *= -1

    # 3) si algo quedó sin posición (por datos raros), ponerlo cerca
    for n in G.nodes():
        if n not in pos:
            pos[n] = (0.0, -dy)

    return pos


# -----------------------------
# Dibujo
# -----------------------------

def dibujar_aristas_ortogonales(ax, G: nx.Graph, pos: dict[int, tuple[float, float]], lw: float = 2.0):
    """
    Dibuja aristas ortogonales SIN diagonales.
    El codo se hace siempre saliendo del nodo padre:
      (x1,y1)->(x1,y2)->(x2,y2)
    """
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # recta pura
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


def dibujar_acometidas(ax, pos: dict[int, tuple[float, float]], df_conexiones, omitir_nodos: set[int] | None = None):
    """
    Dibuja línea punteada y texto 'Usuarios' bajo cada nodo_final.
    """
    omitir_nodos = omitir_nodos or set()

    for _, row in df_conexiones.iterrows():
        nf = int(row["nodo_final"])
        if nf in omitir_nodos:
            continue
        if nf not in pos:
            continue

        normales = int(row.get("usuarios", 0) or 0)
        especiales = int(row.get("usuarios_especiales", 0) or 0)

        x, y = pos[nf]
        y2 = y - 0.25

        ax.plot([x, x], [y, y2], "--", color="gray", linewidth=1)
        ax.text(x, y2 - 0.05, f"Usuarios: {normales}", fontsize=11, color="blue", ha="center", va="top")

        if especiales > 0:
            ax.text(x, y2 - 0.20, f"Especiales: {especiales}", fontsize=11, color="red", ha="center", va="top")


def dibujar_distancias_tramos(ax, G: nx.Graph, pos: dict[int, tuple[float, float]]):
    """
    Etiqueta distancia. Omite distancias 0.0 (ensucia).
    """
    for u, v, d in G.edges(data=True):
        dist = float(d.get("distancia", 0.0) or 0.0)
        if dist <= 0:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        ax.text(xm, ym + 0.15, f"{dist:.1f} m", fontsize=11, color="red", ha="center")


def dibujar_transformador_sin_linea(ax, pos: dict[int, tuple[float, float]], capacidad_transformador, nodo: int = 1,
                                   dx: float = -0.9, dy: float = 0.0):
    """
    Transformador A LA PAR del nodo (sin línea).
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

    # Layout final (troncal + ramales)
    posiciones = calcular_posiciones_troncal_ramales(G, nodo_raiz=1)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Dibujo
    dibujar_aristas_ortogonales(ax, G, posiciones, lw=2.0)
    dibujar_nodos(ax, G, posiciones)
    dibujar_etiquetas_nodos(ax, G, posiciones)

    # Nodo 1 sin acometida (por TS)
    dibujar_acometidas(ax, posiciones, df_conexiones, omitir_nodos={1})

    dibujar_distancias_tramos(ax, G, posiciones)

    # TS sin línea
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
