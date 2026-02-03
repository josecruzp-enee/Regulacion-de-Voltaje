# -*- coding: utf-8 -*-
"""
graficos_red.py

Diagrama formal tipo "unifilar CAD" para red secundaria:
- Layout: troncal horizontal + ramas verticales (ordenado)
- Estilo: sobrio, 2 colores (negro y rojo para distancias)
- Salida: ReportLab Image (PNG embebible en PDF)

Requisitos:
- networkx
- matplotlib
- reportlab

Autor: (tu proyecto)
"""

from __future__ import annotations

import io
from io import BytesIO
from typing import Dict, Tuple, Optional, Iterable, List

import matplotlib.pyplot as plt
import networkx as nx
from reportlab.platypus import Image
from reportlab.lib.units import inch

# Importa función para cargar datos
try:
    from modulos.datos import cargar_datos_circuito
except ImportError:
    from datos import cargar_datos_circuito  # fallback si se ejecuta fuera del paquete


# =============================================================================
# Creación de grafo
# =============================================================================

def crear_grafo(
    nodos_inicio: Iterable[int],
    nodos_final: Iterable[int],
    usuarios: Iterable[int],
    distancias: Iterable[float],
) -> nx.Graph:
    """
    Crea un grafo no dirigido con atributos:
      - usuarios (int) en arista
      - distancia (float) en arista
    """
    G = nx.Graph()
    for ni, nf, u, d in zip(nodos_inicio, nodos_final, usuarios, distancias):
        try:
            ni_i = int(ni)
            nf_i = int(nf)
            u_i = int(u)
            d_f = float(d)
            G.add_edge(ni_i, nf_i, usuarios=u_i, distancia=d_f)
        except Exception as e:
            print(f"⚠️ Error al agregar arista {ni}-{nf}: {e}")
    return G


# =============================================================================
# Layout formal: troncal + ramas
# =============================================================================

def _edge_dist(G: nx.Graph, u: int, v: int) -> float:
    return float(G[u][v].get("distancia", 0.0))


def calcular_posiciones_red_troncal(
    G: nx.Graph,
    nodo_raiz: int = 1,
    escala: Optional[float] = None,
    sep_y: float = 1.0,
    margen_y: float = 0.8,
    factor_estiramiento_x: float = 12.0,
) -> Dict[int, Tuple[float, float]]:
    """
    Layout "CAD-like":
    - Construye árbol BFS desde nodo_raiz para ordenar radial (evita caos si hay ciclos).
    - Identifica troncal como camino raíz->hoja más lejana por distancia acumulada.
    - Troncal en y=0, ramas en niveles y separados (arriba/abajo alternando).
    - X proporcional a distancia acumulada (con estiramiento para apariencia lineal).

    Retorna dict {nodo: (x,y)}.
    """
    if nodo_raiz not in G.nodes:
        return {n: (0.0, 0.0) for n in G.nodes}

    # Escala dinámica si no se da
    if escala is None:
        total_dist = sum(nx.get_edge_attributes(G, "distancia").values()) or 1.0
        escala = 10.0 / (total_dist + 1.0)  # base

    # Árbol BFS (rompe ciclos para layout)
    T = nx.bfs_tree(G, source=nodo_raiz)

    # Distancia acumulada en el árbol (con pesos de G)
    dist_acum: Dict[int, float] = {nodo_raiz: 0.0}
    for u in nx.topological_sort(T):
        for v in T.successors(u):
            dist_acum[v] = dist_acum[u] + _edge_dist(G, u, v)

    # hoja más lejana
    hojas = [n for n in T.nodes if T.out_degree(n) == 0]
    if not hojas:
        return {nodo_raiz: (0.0, 0.0)}

    hoja_lejana = max(hojas, key=lambda n: dist_acum.get(n, 0.0))

    # Troncal = camino raíz -> hoja_lejana
    troncal = nx.shortest_path(T, source=nodo_raiz, target=hoja_lejana)
    troncal_set = set(troncal)

    # Posiciones
    pos: Dict[int, Tuple[float, float]] = {}

    # Colocar troncal en horizontal
    for n in troncal:
        x = dist_acum.get(n, 0.0) * escala * factor_estiramiento_x
        pos[n] = (x, 0.0)

    # Control de "slots" y para ramas
    ocupados_y: List[float] = []

    def y_libre(signo: int) -> float:
        k = 1
        while True:
            y = signo * (margen_y + (k - 1) * sep_y)
            if all(abs(y - yo) >= sep_y * 0.95 for yo in ocupados_y):
                ocupados_y.append(y)
                return y
            k += 1

    # Colocar subárbol con y fijo
    def colocar_subarbol(raiz_rama: int, y_base: float):
        stack = [raiz_rama]
        while stack:
            n = stack.pop()
            x = dist_acum.get(n, 0.0) * escala * factor_estiramiento_x
            pos[n] = (x, y_base)
            hijos = list(T.successors(n))
            for h in reversed(hijos):
                stack.append(h)

    # Recorrer troncal y colgar ramas
    signo = 1
    for n in troncal:
        hijos = list(T.successors(n))
        ramas = [h for h in hijos if h not in troncal_set]
        for r in ramas:
            yb = y_libre(signo)
            signo *= -1
            colocar_subarbol(r, yb)

    # Si hay nodos no alcanzados (grafo no conectado), colócalos aparte
    for n in G.nodes:
        if n not in pos:
            yb = y_libre(signo)
            signo *= -1
            pos[n] = (0.0, yb)

    return pos


# =============================================================================
# Dibujo formal
# =============================================================================

def _configurar_estilo_matplotlib():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10


def dibujar_transformador(ax, posiciones, nodo_raiz: int, capacidad_transformador: float, etiqueta_ts: str = "TS"):
    x, y = posiciones[nodo_raiz]
    # Triángulo discreto
    ax.scatter([x], [y], marker="^", s=120, c="black", zorder=5)

    # Etiqueta formal
    texto = f"{etiqueta_ts}\n{capacidad_transformador:.0f} kVA"
    ax.text(
        x - 0.6, y + 0.10,
        texto,
        ha="right", va="bottom",
        fontsize=9, color="black"
    )


def dibujar_nodos(ax, posiciones, nodo_raiz: int):
    # Puntos negros para nodos generales
    xs = []
    ys = []
    for n, (x, y) in posiciones.items():
        if n == nodo_raiz:
            continue
        xs.append(x)
        ys.append(y)
    ax.scatter(xs, ys, s=18, c="black", zorder=4)


def dibujar_aristas_formales(ax, G: nx.Graph, posiciones, aristas_extra_punteadas: bool = True):
    """
    Dibuja aristas del grafo completo.
    Si hay ciclos, opcionalmente marca aristas no pertenecientes al árbol BFS como punteadas.
    """
    # Dibujo base: líneas negras
    for u, v in G.edges():
        if u == v:
            continue
        x1, y1 = posiciones[u]
        x2, y2 = posiciones[v]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.0, zorder=2)

    if aristas_extra_punteadas:
        # Marcar aristas "extra" respecto a un BFS tree (para indicar anillos sin desordenar layout)
        try:
            T = nx.bfs_tree(G, source=1)
            tree_edges = set((min(a, b), max(a, b)) for a, b in T.edges())
            for u, v in G.edges():
                if u == v:
                    continue
                key = (min(u, v), max(u, v))
                if key not in tree_edges:
                    x1, y1 = posiciones[u]
                    x2, y2 = posiciones[v]
                    ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.2, linestyle="--", zorder=3)
        except Exception:
            pass


def dibujar_etiquetas_nodo(ax, posiciones, nodo_raiz: int):
    """
    Etiqueta el número del nodo sobre el punto (sobrio).
    """
    for n, (x, y) in posiciones.items():
        if n == nodo_raiz:
            # opcional: no etiquetar el nodo 1 (trafo)
            continue
        ax.text(x, y + 0.22, f"{n}", ha="center", va="bottom", fontsize=10, color="black")


def dibujar_usuarios_por_nodo(ax, posiciones, df_conexiones):
    """
    Escribe 'Usuarios: X' debajo de cada nodo final.
    Mantiene orientación limpia; si el nodo está arriba, coloca el texto hacia abajo; si está abajo, hacia arriba.
    """
    for _, row in df_conexiones.iterrows():
        nf = int(row["nodo_final"])
        normales = int(row.get("usuarios", 0))
        especiales = int(row.get("usuarios_especiales", 0) or 0)

        if nf not in posiciones:
            continue

        x, y = posiciones[nf]

        # dirección del texto (para no salirse del dibujo)
        dir_texto = -1 if y >= 0 else 1
        y_texto = y + dir_texto * 0.45

        # Línea punteada corta (estilo acometida)
        ax.plot([x, x], [y, y_texto], color="black", linewidth=1.0, linestyle="--", zorder=1)

        # Texto formal en negro
        ax.text(x, y_texto, f"Usuarios: {normales}", ha="center", va="center", fontsize=10, color="black")

        if especiales > 0:
            ax.text(x, y_texto + dir_texto * 0.22, f"Esp.: {especiales}", ha="center", va="center", fontsize=9, color="black")


def dibujar_distancias_tramos(ax, G: nx.Graph, posiciones):
    """
    Distancias centradas en cada tramo, en rojo, SIN rotación (más tipo plano).
    """
    for (u, v, d) in G.edges(data=True):
        if u == v:
            continue
        x1, y1 = posiciones[u]
        x2, y2 = posiciones[v]
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        distancia = float(d.get("distancia", 0.0))
        if distancia <= 0:
            continue

        # Texto encima del tramo (offset vertical leve)
        ym_text = ym + 0.22

        ax.text(
            xm, ym_text,
            f"{distancia:.1f} m",
            color="red",
            fontsize=10,
            ha="center",
            va="bottom"
        )


# =============================================================================
# Función principal para crear imagen ReportLab
# =============================================================================

def crear_grafico_nodos(
    nodos_inicio,
    nodos_final,
    usuarios,
    distancias,
    capacidad_transformador,
    df_conexiones,
    nodo_raiz: int = 1,
    etiqueta_ts: str = "TS",
    ancho_pulg: float = 6.0,
    alto_pulg: float = 3.4,
) -> Image:
    """
    Genera el diagrama formal (PNG) y lo devuelve como ReportLab Image.
    """
    _configurar_estilo_matplotlib()

    G = crear_grafo(nodos_inicio, nodos_final, usuarios, distancias)
    posiciones = calcular_posiciones_red_troncal(G, nodo_raiz=nodo_raiz)

    fig = plt.figure(figsize=(ancho_pulg, alto_pulg))
    ax = plt.gca()

    # Dibujo formal
    dibujar_aristas_formales(ax, G, posiciones, aristas_extra_punteadas=True)
    dibujar_nodos(ax, posiciones, nodo_raiz=nodo_raiz)
    dibujar_transformador(ax, posiciones, nodo_raiz, float(capacidad_transformador), etiqueta_ts=etiqueta_ts)
    dibujar_etiquetas_nodo(ax, posiciones, nodo_raiz=nodo_raiz)
    dibujar_usuarios_por_nodo(ax, posiciones, df_conexiones)
    dibujar_distancias_tramos(ax, G, posiciones)

    # Título discreto (formal)
    ax.set_title("Diagrama Unifilar de Nodos", fontsize=11, color="black", pad=8)

    # Limpieza
    ax.axis("off")

    # Márgenes suaves
    plt.tight_layout()

    # Exportar a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # ReportLab Image
    img = Image(buf, width=6.5 * inch, height=3.6 * inch)
    img.hAlign = "CENTER"
    return img


def crear_grafico_nodos_desde_archivo(ruta_excel: str) -> Image:
    """
    Genera el gráfico directamente a partir de un archivo Excel.
    """
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

    # Etiqueta TS-#
    etiqueta_ts = f"TS-{transformador_numero}" if transformador_numero not in (None, "", 0) else "TS"

    return crear_grafico_nodos(
        nodos_inicio=df_conexiones["nodo_inicial"].astype(int).tolist(),
        nodos_final=df_conexiones["nodo_final"].astype(int).tolist(),
        usuarios=df_conexiones["usuarios"].astype(int).tolist(),
        distancias=df_conexiones["distancia"].astype(float).tolist(),
        capacidad_transformador=float(capacidad_transformador),
        df_conexiones=df_conexiones,
        nodo_raiz=1,
        etiqueta_ts=etiqueta_ts,
    )
