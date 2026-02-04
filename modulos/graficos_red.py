# -*- coding: utf-8 -*-
"""
graficos_red.py

Diagrama formal tipo "unifilar CAD" para red secundaria:
- Layout ortogonal: troncal horizontal + ramales verticales (estilo plano)
- Textos por nodo en bloque (N-#, A, kVA, P-## si existe)
- Distancias colocadas como plano (sobre tramos horizontales / al lado de verticales)
- Salida: ReportLab Image (PNG embebible en PDF)

Requisitos:
- networkx
- matplotlib
- reportlab
"""

from __future__ import annotations

import io
from typing import Dict, Tuple, Optional, Iterable, Any, List

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


def _configurar_estilo_matplotlib():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10


def _edge_dist(G: nx.Graph, u: int, v: int) -> float:
    return float(G[u][v].get("distancia", 0.0))


# =============================================================================
# Grafo
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

        if ni_i <= 0 or nf_i <= 0:
            continue

        G.add_edge(ni_i, nf_i, usuarios=u_i, distancia=d_f)
    return G


# =============================================================================
# Layout CAD ORTOGONAL
# =============================================================================

def _bfs_tree_y_dist(G: nx.Graph, raiz: int) -> Tuple[nx.DiGraph, Dict[int, float]]:
    T = nx.bfs_tree(G, source=raiz)
    dist: Dict[int, float] = {raiz: 0.0}
    for u in nx.topological_sort(T):
        for v in T.successors(u):
            dist[v] = dist[u] + _edge_dist(G, u, v)
    return T, dist


def _troncal_mas_larga(T: nx.DiGraph, dist: Dict[int, float], raiz: int) -> List[int]:
    hojas = [n for n in T.nodes if T.out_degree(n) == 0]
    if not hojas:
        return [raiz]
    hoja_lejana = max(hojas, key=lambda n: dist.get(n, 0.0))
    return nx.shortest_path(T, source=raiz, target=hoja_lejana)


def calcular_posiciones_cad_ortogonal(
    G: nx.Graph,
    nodo_raiz: int = 1,
    escala_x: Optional[float] = None,
    estirar_x: float = 1.0,
    sep_y: float = 1.0,
) -> Dict[int, Tuple[float, float]]:
    """
    Posiciona nodos con reglas "CAD":
    - Troncal principal (camino más largo) en horizontal (y=0)
    - Cualquier rama que sale de la troncal baja en vertical a un nivel y fijo,
      y continúa horizontal si sigue (sub-troncal)
    - Separación vertical por niveles (sep_y)

    Retorna pos[n] = (x, y) en unidades abstractas.
    """
    if nodo_raiz not in G.nodes:
        return {n: (0.0, 0.0) for n in G.nodes}

    if escala_x is None:
        total = sum(nx.get_edge_attributes(G, "distancia").values()) or 1.0
        escala_x = 1.0 / total  # base para normalizar, luego estiramos

    T, dist = _bfs_tree_y_dist(G, nodo_raiz)
    troncal = _troncal_mas_larga(T, dist, nodo_raiz)
    troncal_set = set(troncal)

    pos: Dict[int, Tuple[float, float]] = {}

    # 1) Troncal horizontal
    for n in troncal:
        x = dist.get(n, 0.0) * escala_x * 100.0 * estirar_x  # 100 para que "se sienta CAD"
        pos[n] = (x, 0.0)

    # 2) Colgar ramas por niveles (1,2,3...) alternando abajo/arriba
    nivel_usado = []  # y ocupados
    def siguiente_nivel(signo: int) -> float:
        k = 1
        while True:
            y = signo * (k * sep_y)
            if y not in nivel_usado:
                nivel_usado.append(y)
                return y
            k += 1

    signo = -1  # por defecto, primero hacia abajo como en planos
    troncal_index = {n: i for i, n in enumerate(troncal)}

    def colocar_rama(raiz_rama: int, y_nivel: float):
        """
        Coloca la rama manteniendo x por distancia acumulada,
        y constante (horizontal) excepto el primer tramo (vertical ideal).
        """
        stack = [raiz_rama]
        while stack:
            n = stack.pop()
            x = dist.get(n, 0.0) * escala_x * 100.0 * estirar_x
            pos[n] = (x, y_nivel)
            hijos = list(T.successors(n))
            for h in reversed(hijos):
                stack.append(h)

    # Recorremos troncal y asignamos niveles a ramas salientes
    for n in troncal:
        hijos = list(T.successors(n))
        ramas = [h for h in hijos if h not in troncal_set]
        for r in ramas:
            y_nivel = siguiente_nivel(signo)
            signo *= -1  # alternar
            colocar_rama(r, y_nivel)

    # nodos no conectados (raro)
    for n in G.nodes:
        if n not in pos:
            pos[n] = (0.0, siguiente_nivel(signo))
            signo *= -1

    return pos


# =============================================================================
# Estilo CAD: dibujo
# =============================================================================

def _calc_unidades_texto(pos: Dict[int, Tuple[float, float]]) -> float:
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xspan = (max(xs) - min(xs)) if xs else 1.0
    yspan = (max(ys) - min(ys)) if ys else 1.0
    # unidad para offsets de texto (proporcional al tamaño del dibujo)
    return max(xspan / 40.0, yspan / 8.0, 1.2)


def _map_por_nodo(df, colname_candidates: List[str]) -> Dict[int, Any]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    col = None
    for cand in colname_candidates:
        if cand in cols:
            col = cols[cand]
            break
    if col is None:
        return {}
    out = {}
    for _, row in df.iterrows():
        nf = _safe_int(row.get("nodo_final", 0))
        if nf > 0:
            out[nf] = row.get(col)
    return out


def dibujar_aristas(ax, G: nx.Graph, pos: Dict[int, Tuple[float, float]], lw: float = 2.4):
    for u, v in G.edges():
        if u == v:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=lw, zorder=2)


def dibujar_nodos(ax, pos: Dict[int, Tuple[float, float]], nodo_raiz: int = 1):
    # puntos negros pequeños
    xs = []
    ys = []
    for n, (x, y) in pos.items():
        if n == nodo_raiz:
            continue
        xs.append(x)
        ys.append(y)
    ax.scatter(xs, ys, s=28, c="black", zorder=4)


def dibujar_transformador(ax, pos: Dict[int, Tuple[float, float]], nodo_raiz: int, kva: float, etiqueta_ts: str = "TS"):
    x, y = pos[nodo_raiz]
    ax.scatter([x], [y], marker="^", s=220, c="black", zorder=5)
    ax.text(
        x - 2.2, y,
        f"{etiqueta_ts}\n{kva:.0f} kVA",
        ha="right", va="center",
        fontsize=13, color="black"
    )


def dibujar_distancias(ax, G: nx.Graph, pos: Dict[int, Tuple[float, float]]):
    """
    Estilo CAD:
    - si el tramo es horizontal: distancia arriba (y + off)
    - si es vertical: distancia al lado (x + off)
    - si es diagonal (debería ser raro): offset perpendicular
    """
    U = _calc_unidades_texto(pos)
    off = U * 0.35

    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        dist_m = _safe_float(d.get("distancia", 0.0))
        if dist_m <= 0:
            continue

        dx = x2 - x1
        dy = y2 - y1

        if abs(dy) < 1e-6:  # horizontal
            ax.text(xm, ym + off, f"{dist_m:.0f} m", color="black", fontsize=12,
                    ha="center", va="bottom")
        elif abs(dx) < 1e-6:  # vertical
            ax.text(xm + off, ym, f"{dist_m:.0f} m", color="black", fontsize=12,
                    ha="left", va="center")
        else:
            # diagonal: offset perpendicular
            norm = (dx*dx + dy*dy) ** 0.5 or 1.0
            ox, oy = (-dy/norm) * off, (dx/norm) * off
            ax.text(xm + ox, ym + oy, f"{dist_m:.0f} m", color="black", fontsize=12,
                    ha="center", va="center")


def dibujar_textos_por_nodo(ax, pos: Dict[int, Tuple[float, float]], df_conexiones, nodo_raiz: int = 1):
    """
    Bloque de texto al lado del nodo, tipo CAD, como tu ejemplo:
      N-#
      (A si existe)
      (kVA si existe)
      (P-## si existe)

    Si no tienes A/kVA/P en df_conexiones, solo saldrá N-#.
    """
    U = _calc_unidades_texto(pos)

    # mapeos opcionales (según nombres comunes)
    mapa_i = _map_por_nodo(df_conexiones, ["i", "corriente", "i_a", "corriente_a"])
    mapa_kva = _map_por_nodo(df_conexiones, ["kva", "demanda_kva", "kva_nodo"])
    mapa_p = _map_por_nodo(df_conexiones, ["punto", "p", "p#"])

    # Si tu P viene como "P-10" en otra columna, puedes añadirla arriba.

    for n, (x, y) in pos.items():
        if n == nodo_raiz:
            continue

        # En tu CAD, el bloque suele ir "cerca" del nodo, sin cruzarse con la línea:
        # - Si y==0 (troncal): texto debajo
        # - Si y != 0 (ramal): texto a la izquierda del nodo (y centrado un poco arriba)
        if abs(y) < 1e-9:
            tx = x
            ty = y - U * 0.95
            ha = "center"
            va = "top"
        else:
            tx = x - U * 0.55
            ty = y + U * 0.10
            ha = "right"
            va = "bottom"

        lineas = [f"N-{n}"]

        ii = mapa_i.get(n, None)
        if ii is not None and _safe_float(ii) > 0:
            lineas.append(f"{_safe_float(ii):.0f} A")

        kk = mapa_kva.get(n, None)
        if kk is not None and _safe_float(kk) > 0:
            lineas.append(f"{_safe_float(kk):.2f} kVA")

        pp = mapa_p.get(n, None)
        if pp not in (None, "", 0):
            lineas.append(str(pp))

        ax.text(
            tx, ty,
            "\n".join(lineas),
            ha=ha, va=va,
            fontsize=12, color="black"
        )


def dibujar_usuarios(ax, pos: Dict[int, Tuple[float, float]], df_conexiones, nodo_raiz: int = 1):
    """
    Si quieres mantener usuarios, que sea discreto.
    Si NO lo quieres (porque en CAD usas N/A/kVA/P), puedes comentar esta función en crear_grafico_nodos().
    """
    U = _calc_unidades_texto(pos)
    mapa_u = _map_por_nodo(df_conexiones, ["usuarios"])
    for n, (x, y) in pos.items():
        if n == nodo_raiz:
            continue
        if n not in mapa_u:
            continue

        u = _safe_int(mapa_u.get(n, 0))
        if u <= 0:
            continue

        # texto pequeño, debajo del nodo
        ax.text(x, y - U * 0.35, f"Usuarios: {u}", ha="center", va="top", fontsize=10, color="black")


# =============================================================================
# Principal
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
    ancho_pulg: float = 8.5,
    alto_pulg: float = 3.0,
    titulo: str = "",
) -> Image:
    """
    Genera el diagrama formal (PNG) y lo devuelve como ReportLab Image.
    """
    _configurar_estilo_matplotlib()

    G = crear_grafo(nodos_inicio, nodos_final, usuarios, distancias)

    # Layout ortogonal CAD
    pos = calcular_posiciones_cad_ortogonal(
        G,
        nodo_raiz=nodo_raiz,
        escala_x=None,
        estirar_x=1.35,   # sube “sensación CAD”
        sep_y=42.0,       # separación vertical en unidades (más parecido a plano)
    )

    fig = plt.figure(figsize=(ancho_pulg, alto_pulg))
    ax = plt.gca()

    # Dibujo
    dibujar_aristas(ax, G, pos, lw=3.0)
    dibujar_nodos(ax, pos, nodo_raiz=nodo_raiz)
    dibujar_transformador(ax, pos, nodo_raiz, _safe_float(capacidad_transformador), etiqueta_ts=etiqueta_ts)
    dibujar_distancias(ax, G, pos)
    dibujar_textos_por_nodo(ax, pos, df_conexiones, nodo_raiz=nodo_raiz)

    # Si NO quieres "Usuarios" porque en CAD no los pones, deja esto comentado:
    # dibujar_usuarios(ax, pos, df_conexiones, nodo_raiz=nodo_raiz)

    if titulo:
        ax.set_title(titulo, fontsize=13, color="black", pad=6)

    ax.axis("off")
    plt.tight_layout()

    # Exportar
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=240, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    img = Image(buf, width=6.9 * inch, height=2.6 * inch)
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
        titulo="",  # en CAD normalmente no se pone título grande
    )
