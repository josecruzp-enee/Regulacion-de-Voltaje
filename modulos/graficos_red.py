# -*- coding: utf-8 -*-
"""
graficos_red.py
Funciones para visualización de la red eléctrica y generación de diagramas
usando NetworkX y Matplotlib.
"""

import io
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.platypus import Image
from reportlab.lib.units import inch

# Importa función para cargar datos
try:
    from modulos.datos import cargar_datos_circuito
except ImportError:
    from datos import cargar_datos_circuito   # fallback si se ejecuta fuera del paquete


# ============================================================
# Creación de Grafo y Posiciones
# ============================================================

def crear_grafo(nodos_inicio, nodos_final, usuarios, distancias):
    G = nx.Graph()
    for ni, nf, u, d in zip(nodos_inicio, nodos_final, usuarios, distancias):
        try:
            G.add_edge(
                int(ni), int(nf),
                usuarios=int(u),
                distancia=float(d)
            )
        except Exception as e:
            print(f"⚠️ Error al agregar arista {ni}-{nf}: {e}")
    return G


def calcular_posiciones_red(G, nodo_raiz=1, escala=None, sep_y=1.2, margen_y=0.8):
    """
    Layout 'ingenieril' para red radial:
    - Calcula un árbol BFS desde el nodo_raiz
    - Encuentra una TRONCAL (camino más largo desde la raíz en el árbol)
    - Coloca la troncal en horizontal (y=0), con X proporcional a distancia acumulada
    - Cuelga las ramas en vertical (y positivos/negativos) evitando colisiones
    """

    # --- 0) Escala dinámica (si no se da) ---
    if escala is None:
        total_distancia = sum(nx.get_edge_attributes(G, "distancia").values()) or 1.0
        escala = 8.0 / (total_distancia + 1.0)

    # --- 1) Construir árbol BFS (si hay ciclos, el árbol los "rompe") ---
    T = nx.bfs_tree(G, source=nodo_raiz)

    # Si el grafo no es conectado o el nodo_raiz no existe, protegemos:
    if nodo_raiz not in G.nodes:
        return {n: (0.0, 0.0) for n in G.nodes}

    # --- 2) Encontrar hoja más lejana (en el árbol) para definir la troncal ---
    # Distancia acumulada en el árbol (por atributo 'distancia' del grafo original)
    def dist_arista(u, v):
        return float(G[u][v].get("distancia", 0.0))

    # acumuladas: usando recorrido topológico del árbol
    dist_acum = {nodo_raiz: 0.0}
    for u in nx.topological_sort(T):
        for v in T.successors(u):
            dist_acum[v] = dist_acum[u] + dist_arista(u, v)

    # hoja más lejana:
    hojas = [n for n in T.nodes if T.out_degree(n) == 0]
    if not hojas:
        return {nodo_raiz: (0.0, 0.0)}

    hoja_lejana = max(hojas, key=lambda n: dist_acum.get(n, 0.0))

    # Troncal = camino raíz -> hoja_lejana
    troncal = nx.shortest_path(T, source=nodo_raiz, target=hoja_lejana)

    # --- 3) Posiciones iniciales troncal (horizontal) ---
    pos = {}
    for n in troncal:
        x = dist_acum.get(n, 0.0) * escala
        pos[n] = (x, 0.0)

    # --- 4) Colocar ramas: asignar "slots" en Y para evitar amontonamiento ---
    # Usaremos una lista de niveles ya ocupados para escoger y libre.
    ocupados_y = []  # valores de y usados para colgar subárboles

    def siguiente_y_libre(signo=1):
        """Encuentra un y libre separado por sep_y."""
        k = 1
        while True:
            y = signo * (margen_y + (k - 1) * sep_y)
            # revisa si está muy cerca de otro ocupado
            if all(abs(y - yo) >= sep_y * 0.9 for yo in ocupados_y):
                ocupados_y.append(y)
                return y
            k += 1

    # Para alternar arriba/abajo
    signo = 1

    # Subárbol desde un nodo troncal hacia un hijo que NO está en troncal
    troncal_set = set(troncal)

    def colocar_subarbol(raiz_rama, y_base):
        """Coloca un subárbol con y fijo (estructura tipo 'peine')"""
        # recorrido DFS en el árbol
        stack = [(raiz_rama, None)]
        while stack:
            n, padre = stack.pop()
            # x proporcional a distancia acumulada (desde raíz del árbol)
            x = dist_acum.get(n, 0.0) * escala

            # Para ramas, inclinamos ligeramente por profundidad local para separar textos
            # (opcional) pequeño jitter según orden de aparición:
            pos[n] = (x, y_base)

            hijos = list(T.successors(n))
            # Poner hijos después
            for h in reversed(hijos):
                stack.append((h, n))

    # Recorremos la troncal y colgamos ramas
    for n in troncal:
        hijos = list(T.successors(n))
        ramas = [h for h in hijos if h not in troncal_set]
        for r in ramas:
            y_rama = siguiente_y_libre(signo=signo)
            signo *= -1
            colocar_subarbol(r, y_rama)

    # --- 5) Si hubiera nodos fuera del árbol BFS (grafo no conectado), acomodarlos aparte ---
    for n in G.nodes:
        if n not in pos:
            pos[n] = (0.0, siguiente_y_libre(signo=signo))
            signo *= -1

    return pos


# ============================================================
# Funciones de Dibujo
# ============================================================

def dibujar_nodos_transformador(ax, G, posiciones, capacidad_transformador):
    tamaño_transformador = 400
    nx.draw_networkx_nodes(
        G, posiciones, nodelist=[1],
        node_shape="^", node_color="orange",
        node_size=tamaño_transformador, label="Transformador (Nodo 1)"
    )
    x, y = posiciones[1]
    etiqueta = f"Transformador\n{capacidad_transformador} kVA"
    ax.text(
        x - 1, y, etiqueta, fontsize=9, ha="center", color="black",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
    )


def dibujar_nodos_generales(ax, G, posiciones):
    tamaño_nodos = 200
    otros_nodos = [n for n in G.nodes if n != 1]
    nx.draw_networkx_nodes(
        G, posiciones, nodelist=otros_nodos,
        node_shape="o", node_color="lightblue", node_size=tamaño_nodos
    )


def dibujar_aristas(ax, G, posiciones):
    aristas_normales = [(u, v) for u, v in G.edges() if u != v]
    nx.draw_networkx_edges(G, posiciones, edgelist=aristas_normales, width=2)

    bucles = [(u, v) for u, v in G.edges() if u == v and u != 1]
    for nodo, _ in bucles:
        x, y = posiciones[nodo]
        circle = plt.Circle((x, y), 0.1, color="red", fill=False,
                            linestyle="--", linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y + 0.12, "Bucle", fontsize=12, color="red", ha="center")


def dibujar_etiquetas_nodos(ax, G, posiciones):
    etiquetas_nodos = {n: str(n) for n in G.nodes}
    nx.draw_networkx_labels(G, posiciones, etiquetas_nodos,
                            font_size=12, font_weight="bold")


def dibujar_acometidas(ax, posiciones, df_conexiones):
    for _, row in df_conexiones.iterrows():
        nf = int(row["nodo_final"])
        normales = int(row["usuarios"])
        especiales = int(row.get("usuarios_especiales", 0))

        if nf in posiciones:
            x, y = posiciones[nf]
            x_u, y_u = x, y - 0.2
            ax.plot([x, x_u], [y, y_u], color="gray", linestyle="--", linewidth=1)

            # Texto de usuarios normales (arriba)
            ax.text(x_u, y_u - 0.03,
                    f"Usuarios: {normales}", fontsize=12,
                    color="blue", ha="center", va="top")

            # Texto de usuarios especiales (abajo, si existen)
            if especiales > 0:
                ax.text(x_u, y_u - 0.18,   # más abajo
                        f"Especiales: {especiales}", fontsize=12,
                        color="red", ha="center", va="top")


def dibujar_distancias_tramos(ax, G, posiciones):
    for (u, v, d) in G.edges(data=True):
        if u != v:
            x1, y1 = posiciones[u]
            x2, y2 = posiciones[v]
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            dist = (dx**2 + dy**2) ** 0.5
            offset_x, offset_y = -dy / dist * 0.1, dx / dist * 0.1
            ax.text(xm + offset_x, ym + offset_y, f"{d['distancia']} m",
                    color="red", fontsize=12, ha="center", va="center")


# ============================================================
# Funciones Principales
# ============================================================

def crear_grafico_nodos(nodos_inicio, nodos_final, usuarios, distancias,
                        capacidad_transformador, df_conexiones):
    """
    Genera un gráfico a partir de listas de datos y el DataFrame completo.
    """
    G = crear_grafo(nodos_inicio, nodos_final, usuarios, distancias)
    posiciones = calcular_posiciones_red(G, nodo_raiz=1)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    dibujar_nodos_transformador(ax, G, posiciones, capacidad_transformador)
    dibujar_nodos_generales(ax, G, posiciones)
    dibujar_aristas(ax, G, posiciones)
    dibujar_etiquetas_nodos(ax, G, posiciones)
    dibujar_acometidas(ax, posiciones, df_conexiones)
    dibujar_distancias_tramos(ax, G, posiciones)

    plt.title("Diagrama de Nodos del Transformador")
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    img = Image(buf, width=5 * inch, height=3 * inch)
    img.hAlign = "CENTER"
    return img
   


def crear_grafico_nodos_desde_archivo(ruta_excel):
    """
    Genera el gráfico directamente a partir de un archivo Excel.
    """
    (df_conexiones, df_parametros, df_info,
     tipo_conductor, area_lote, capacidad_transformador,
     proyecto_numero, proyecto_nombre, transformador_numero,
     usuarios, distancias, nodos_inicio, nodos_final) = cargar_datos_circuito(ruta_excel)

    return crear_grafico_nodos(
        nodos_inicio=df_conexiones["nodo_inicial"].astype(int).tolist(),
        nodos_final=df_conexiones["nodo_final"].astype(int).tolist(),
        usuarios=df_conexiones["usuarios"].astype(int).tolist(),
        distancias=df_conexiones["distancia"].astype(float).tolist(),
        capacidad_transformador=capacidad_transformador,
        df_conexiones=df_conexiones
    )

