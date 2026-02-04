# -*- coding: utf-8 -*-
"""
graficos_red.py

Diagrama formal tipo "unifilar CAD" para red secundaria.

Mejoras clave:
- Layout: troncal principal por distancia + ramas ordenadas por subárbol (menos colisiones)
- Estilo: sobrio (negro) + longitudes en rojo
- Etiquetas CAD: N-#, A, kVA, P-## (si existe)
- Salida: ReportLab Image (PNG)

Requisitos:
- networkx
- matplotlib
- reportlab
"""

from __future__ import annotations

import io
from typing import Dict, Tuple, Optional, Iterable, List, Any

import matplotlib.pyplot as plt
import networkx as nx
from reportlab.platypus import Image
from reportlab.lib.units import inch

# Importa función para cargar datos
try:
    from modulos.datos import cargar_datos_circuito
except ImportError:
    from datos import cargar_datos_circuito  # fallback


# =============================================================================
# Utilidades
# =============================================================================

def _configurar_estilo_matplotlib():
    # Estilo sobrio
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10


def _edge_dist(G: nx.Graph, u: int, v: int) -> float:
    return float(G[u][v].get("distancia", 0.0))


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


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
    Grafo no dirigido con atributos:
      - usuarios (int) en arista
      - distancia (float) en arista
    """
    G = nx.Graph()
    for ni, nf, u, d in zip(nodos_inicio, nodos_final, usuarios, distancias):
        ni_i = _safe_int(ni)
        nf_i = _safe_int(nf)
        u_i = _safe_int(u)
        d_f = _safe_float(d)

        # Evitar aristas basura
        if ni_i <= 0 or nf_i <= 0:
            continue

        G.add_edge(ni_i, nf_i, usuarios=u_i, distancia=d_f)
    return G


# =============================================================================
# Layout mejorado: troncal + ramas por subárbol
# =============================================================================

def _bfs_tree_y_distancias(G: nx.Graph, nodo_raiz: int) -> Tuple[nx.DiGraph, Dict[int, float]]:
    """
    Construye BFS tree (dirigido) y calcula distancia acumulada (usando distancias del grafo original).
    """
    T = nx.bfs_tree(G, source=nodo_raiz)

    dist_acum: Dict[int, float] = {nodo_raiz: 0.0}
    for u in nx.topological_sort(T):
        for v in T.successors(u):
            dist_acum[v] = dist_acum[u] + _edge_dist(G, u, v)
    return T, dist_acum


def _camino_troncal_mas_largo(T: nx.DiGraph, dist_acum: Dict[int, float], nodo_raiz: int) -> List[int]:
    """
    Troncal = camino raíz->hoja con mayor distancia acumulada.
    """
    hojas = [n for n in T.nodes if T.out_degree(n) == 0]
    if not hojas:
        return [nodo_raiz]

    hoja_lejana = max(hojas, key=lambda n: dist_acum.get(n, 0.0))
    return nx.shortest_path(T, source=nodo_raiz, target=hoja_lejana)


def _tamanio_subarbol(T: nx.DiGraph, raiz: int) -> int:
    """
    Tamaño (cantidad de nodos) del subárbol desde 'raiz' en el árbol dirigido.
    """
    return len(nx.descendants(T, raiz)) + 1


def calcular_posiciones_red_troncal_mejorado(
    G: nx.Graph,
    nodo_raiz: int = 1,
    escala_x: Optional[float] = None,
    sep_y: float = 1.15,
    margen_y: float = 0.85,
    estirar_x: float = 12.0,
) -> Dict[int, Tuple[float, float]]:
    """
    Layout CAD-like mejorado:
    - BFS tree para ordenar (estable)
    - Troncal principal por distancia acumulada (horizontal)
    - Ramas asignadas a 'slots' de Y según tamaño de subárbol (reduce colisiones)
    - X proporcional a distancia acumulada

    Nota: Para redes con anillos, el árbol “rompe” el ciclo para el layout; luego se pueden dibujar aristas extra punteadas.
    """
    if nodo_raiz not in G.nodes:
        return {n: (0.0, 0.0) for n in G.nodes}

    # Escala dinámica (si no se da)
    if escala_x is None:
        total_dist = sum(nx.get_edge_attributes(G, "distancia").values()) or 1.0
        escala_x = 10.0 / (total_dist + 1.0)

    T, dist_acum = _bfs_tree_y_distancias(G, nodo_raiz)
    troncal = _camino_troncal_mas_largo(T, dist_acum, nodo_raiz)
    troncal_set = set(troncal)

    pos: Dict[int, Tuple[float, float]] = {}

    # 1) Troncal (y=0)
    for n in troncal:
        x = dist_acum.get(n, 0.0) * escala_x * estirar_x
        pos[n] = (x, 0.0)

    # 2) Slots de Y para ramas
    ocupados_y: List[float] = []

    def y_libre(signo: int, bloque: int = 1) -> float:
        """
        Encuentra un y libre.
        'bloque' sirve para reservar más separación para ramas grandes.
        """
        k = 1
        while True:
            y = signo * (margen_y + (k - 1) * sep_y * bloque)
            if all(abs(y - yo) >= sep_y * 0.95 for yo in ocupados_y):
                ocupados_y.append(y)
                return y
            k += 1

    def colocar_subarbol(raiz_rama: int, y_base: float):
        """
        Coloca todo el subárbol con el mismo y_base (unifilar estilo CAD: ramas en "paralelo").
        """
        stack = [raiz_rama]
        while stack:
            n = stack.pop()
            x = dist_acum.get(n, 0.0) * escala_x * estirar_x
            pos[n] = (x, y_base)
            hijos = list(T.successors(n))
            for h in reversed(hijos):
                stack.append(h)

    # 3) Colgar ramas de la troncal ordenadas por tamaño (ramas grandes primero)
    signo = 1
    for n in troncal:
        hijos = list(T.successors(n))
        ramas = [h for h in hijos if h not in troncal_set]
        if not ramas:
            continue

        # Ordenar por tamaño del subárbol (grande primero) para asignar mejores slots
        ramas.sort(key=lambda r: _tamanio_subarbol(T, r), reverse=True)

        for r in ramas:
            tam = _tamanio_subarbol(T, r)
            # ramas grandes reservan más separación
            bloque = 2 if tam >= 6 else 1
            yb = y_libre(signo, bloque=bloque)
            signo *= -1
            colocar_subarbol(r, yb)

    # 4) Nodos fuera del BFS (si el grafo no está conectado)
    for n in G.nodes:
        if n not in pos:
            yb = y_libre(signo)
            signo *= -1
            pos[n] = (0.0, yb)

    return pos


# =============================================================================
# Dibujo formal (CAD-like)
# =============================================================================

def dibujar_aristas(ax, G: nx.Graph, pos: Dict[int, Tuple[float, float]], puntear_anillos: bool = True, nodo_raiz: int = 1):
    # Aristas base (negras)
    for u, v in G.edges():
        if u == v:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.0, zorder=2)

    if puntear_anillos:
        # Aristas extra vs BFS tree: punteadas (si hay ciclos)
        try:
            T = nx.bfs_tree(G, source=nodo_raiz)
            tree_edges = set((min(a, b), max(a, b)) for a, b in T.edges())
            for u, v in G.edges():
                if u == v:
                    continue
                key = (min(u, v), max(u, v))
                if key not in tree_edges:
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.2, linestyle="--", zorder=3)
        except Exception:
            pass


def dibujar_nodos(ax, pos: Dict[int, Tuple[float, float]], nodo_raiz: int = 1):
    # Nodos generales como puntos negros
    xs, ys = [], []
    for n, (x, y) in pos.items():
        if n == nodo_raiz:
            continue
        xs.append(x)
        ys.append(y)
    ax.scatter(xs, ys, s=18, c="black", zorder=4)


def dibujar_transformador(ax, pos: Dict[int, Tuple[float, float]], nodo_raiz: int, capacidad_kva: float, etiqueta_ts: str = "TS"):
    x, y = pos[nodo_raiz]
    ax.scatter([x], [y], marker="^", s=120, c="black", zorder=5)
    ax.text(
        x - 0.75, y + 0.10,
        f"{etiqueta_ts}\n{capacidad_kva:.0f} kVA",
        ha="right", va="bottom",
        fontsize=9, color="black"
    )


def _map_punto_por_nodo(df_conexiones) -> Dict[int, str]:
    """
    Intenta mapear un 'P-##' por nodo final si existe alguna columna compatible.
    Acepta: 'punto', 'Punto', 'p', 'P', 'estructura', etc. (si trae P-xx).
    """
    candidatos = [c for c in df_conexiones.columns]
    col = None
    for c in candidatos:
        cl = str(c).strip().lower()
        if cl in ("punto", "p", "p#", "p_num", "punto_num", "punto#"):
            col = c
            break
    if col is None:
        # buscar alguna que contenga "punto"
        for c in candidatos:
            if "punto" in str(c).strip().lower():
                col = c
                break

    mapa = {}
    if col is None:
        return mapa

    for _, row in df_conexiones.iterrows():
        nf = _safe_int(row.get("nodo_final", 0))
        val = str(row.get(col, "")).strip()
        if nf > 0 and val:
            mapa[nf] = val
    return mapa


def dibujar_etiquetas_cad(
    ax,
    pos: Dict[int, Tuple[float, float]],
    df_conexiones,
    nodo_raiz: int = 1,
    mostrar_punto: bool = True,
):
    """
    Etiqueta formal tipo CAD:
      N-#
      (corriente A si existe en df_conexiones: 'corriente' o 'I' o 'I_A')
      (kVA si existe en df_conexiones: 'kva' o 'kVA')
      P-## (si existe en df)
    """
    # Mapas opcionales
    mapa_punto = _map_punto_por_nodo(df_conexiones) if mostrar_punto else {}

    # Columnas opcionales para A/kVA por nodo final (si existen)
    cols = {str(c).strip().lower(): c for c in df_conexiones.columns}
    col_i = cols.get("i") or cols.get("corriente") or cols.get("i_a") or cols.get("corriente_a")
    col_kva = cols.get("kva") or cols.get("kva_nodo") or cols.get("demanda_kva")

    # Hacemos un mapa por nodo_final para I y kVA (si vienen)
    mapa_i: Dict[int, float] = {}
    mapa_kva: Dict[int, float] = {}

    for _, row in df_conexiones.iterrows():
        nf = _safe_int(row.get("nodo_final", 0))
        if nf <= 0:
            continue
        if col_i:
            mapa_i[nf] = _safe_float(row.get(col_i, 0.0))
        if col_kva:
            mapa_kva[nf] = _safe_float(row.get(col_kva, 0.0))

    for n, (x, y) in pos.items():
        if n == nodo_raiz:
            continue

        # Texto principal arriba del nodo
        lineas = [f"N-{n}"]

        # I (A) opcional
        if n in mapa_i and mapa_i[n] > 0:
            lineas.append(f"{mapa_i[n]:.0f} A")

        # kVA opcional
        if n in mapa_kva and mapa_kva[n] > 0:
            lineas.append(f"{mapa_kva[n]:.2f} kVA")

        # P-## opcional
        if mostrar_punto and n in mapa_punto:
            lineas.append(str(mapa_punto[n]))

        texto = "\n".join(lineas)

        ax.text(
            x, y + 0.26,
            texto,
            ha="center", va="bottom",
            fontsize=9, color="black"
        )


def dibujar_usuarios(ax, pos: Dict[int, Tuple[float, float]], df_conexiones, nodo_raiz: int = 1):
    """
    Usuarios por nodo final, colocados debajo (o arriba si el nodo está abajo).
    """
    for _, row in df_conexiones.iterrows():
        nf = _safe_int(row.get("nodo_final", 0))
        if nf <= 0 or nf == nodo_raiz:
            continue
        if nf not in pos:
            continue

        usuarios = _safe_int(row.get("usuarios", 0))
        especiales = _safe_int(row.get("usuarios_especiales", 0))

        x, y = pos[nf]
        dir_texto = -1 if y >= 0 else 1
        y_txt = y + dir_texto * 0.55

        # línea punteada corta (estilo acometida)
        ax.plot([x, x], [y, y_txt], color="black", linewidth=1.0, linestyle="--", zorder=1)

        # texto en negro (formal)
        ax.text(x, y_txt, f"Usuarios: {usuarios}", ha="center", va="center", fontsize=9, color="black")

        if especiales > 0:
            ax.text(x, y_txt + dir_texto * 0.22, f"Esp.: {especiales}", ha="center", va="center", fontsize=8, color="black")


def dibujar_distancias(ax, G: nx.Graph, pos: Dict[int, Tuple[float, float]]):
    """
    Distancia centrada por tramo con offset perpendicular (mejor lectura).
    """
    for u, v, d in G.edges(data=True):
        if u == v:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        dist_m = _safe_float(d.get("distancia", 0.0))
        if dist_m <= 0:
            continue

        # offset perpendicular pequeño
        dx, dy = (x2 - x1), (y2 - y1)
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        ox, oy = (-dy / norm) * 0.18, (dx / norm) * 0.18

        ax.text(
            xm + ox, ym + oy,
            f"{dist_m:.1f} m",
            color="red",
            fontsize=9,
            ha="center", va="center"
        )


# =============================================================================
# Función principal (ReportLab Image)
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
    ancho_pulg: float = 6.6,
    alto_pulg: float = 3.6,
    titulo: str = "Diagrama Unifilar de Nodos",
) -> Image:
    """
    Genera el diagrama formal (PNG) y lo devuelve como ReportLab Image.
    """
    _configurar_estilo_matplotlib()

    G = crear_grafo(nodos_inicio, nodos_final, usuarios, distancias)

    pos = calcular_posiciones_red_troncal_mejorado(
        G,
        nodo_raiz=nodo_raiz,
        escala_x=None,        # dinámica
        sep_y=1.15,           # separación vertical ramas
        margen_y=0.85,
        estirar_x=12.0,       # “lineal” tipo CAD
    )

    fig = plt.figure(figsize=(ancho_pulg, alto_pulg))
    ax = plt.gca()

    # Dibujo
    dibujar_aristas(ax, G, pos, puntear_anillos=True, nodo_raiz=nodo_raiz)
    dibujar_nodos(ax, pos, nodo_raiz=nodo_raiz)
    dibujar_transformador(ax, pos, nodo_raiz, _safe_float(capacidad_transformador), etiqueta_ts=etiqueta_ts)
    dibujar_etiquetas_cad(ax, pos, df_conexiones, nodo_raiz=nodo_raiz, mostrar_punto=True)
    dibujar_usuarios(ax, pos, df_conexiones, nodo_raiz=nodo_raiz)
    dibujar_distancias(ax, G, pos)

    # Título sobrio
    ax.set_title(titulo, fontsize=11, color="black", pad=8)
    ax.axis("off")
    plt.tight_layout()

    # Exportar a PNG
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=240, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # ReportLab Image
    img = Image(buf, width=6.5 * inch, height=3.6 * inch)
    img.hAlign = "CENTER"
    return img


def crear_grafico_nodos_desde_archivo(ruta_excel: str) -> Image:
    """
    Genera el gráfico a partir de Excel.
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
        titulo="Diagrama Unifilar de Nodos",
    )
