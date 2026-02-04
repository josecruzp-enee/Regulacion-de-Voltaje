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


def calcular_posiciones_red(G, nodo_raiz=1, escala=None, dy=1.5):
    """
    Calcula posiciones en forma horizontal con ramificaciones.
    Si no se da `escala`, se calcula dinámicamente según el total de distancias.
    """
    posiciones = {}
    usados = set()

    # Escala dinámica
    if escala is None:
        total_distancia = sum(nx.get_edge_attributes(G, "distancia").values())
        escala = 5 / (total_distancia + 1)

    def asignar_posiciones(nodo, x, y):
        if nodo in usados:
            return
        usados.add(nodo)
        posiciones[nodo] = (x, y)

        vecinos = [v for v in G.neighbors(nodo) if v not in usados]
        vecinos.sort()

        for i, vecino in enumerate(vecinos):
            distancia = G[nodo][vecino].get("distancia", 0)
            dx = distancia * escala
            nuevo_x = x + dx
            nuevo_y = y - dy * (i - (len(vecinos) - 1) / 2)
            asignar_posiciones(vecino, nuevo_x, nuevo_y)

    asignar_posiciones(nodo_raiz, 0, 0)
    return posiciones


# ============================================================
# Funciones de Dibujo
# ============================================================

def dibujar_simbolo_transformador(ax, posiciones, capacidad_transformador, nodo_raiz=1):
    """
    Dibuja el símbolo del transformador a la par del nodo 1 (no reemplaza el nodo).
    """
    if nodo_raiz not in posiciones:
        return

    x, y = posiciones[nodo_raiz]

    # Offset hacia la izquierda del nodo 1 (ajustable)
    dx = 0.8
    dy = 0.0

    xt, yt = x - dx, y + dy

    # símbolo discreto (triángulo negro)
    ax.scatter([xt], [yt], marker="^", s=160, c="orange", edgecolors="black", zorder=5)

    # texto a la izquierda del símbolo
    ax.text(
        xt - 0.2, yt,
        f"Transformador\n{float(capacidad_transformador):.0f} kVA",
        fontsize=9, ha="right", va="center", color="black"
    )


def dibujar_nodos_generales(ax, G, posiciones):
    tamaño_nodos = 200
    # ✅ incluir también el nodo 1 como nodo normal
    nx.draw_networkx_nodes(
        G, posiciones, nodelist=list(G.nodes),
        node_shape="o", node_color="lightblue", node_size=tamaño_nodos,
        edgecolors="black", linewidths=1.0
    )


def dibujar_aristas(ax, G, posiciones):
    """
    Aristas ortogonales (sin diagonales), sin stub.
    """
    for (u, v, d) in G.edges(data=True):
        if u == v:
            continue

        x1, y1 = posiciones[u]
        x2, y2 = posiciones[v]

        # Codo H->V
        ax.plot([x1, x2], [y1, y1], color="black", linewidth=2, zorder=1)
        ax.plot([x2, x2], [y1, y2], color="black", linewidth=2, zorder=1)




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
    dibujar_nodos_generales(ax, G, posiciones)
    dibujar_aristas(ax, G, posiciones)
    dibujar_etiquetas_nodos(ax, G, posiciones)
    dibujar_acometidas(ax, posiciones, df_conexiones)
    dibujar_distancias_tramos(ax, G, posiciones)
    dibujar_simbolo_transformador(ax, posiciones, capacidad_transformador, nodo_raiz=1)


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




