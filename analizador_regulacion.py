# -*- coding: utf-8 -*-
"""
analizador_regulacion.py
Analizador específico para la app de Regulación de Voltaje (Streamlit).

- Escanea modulos/ y app.py
- Enumera funciones, imports y llamadas
- Detecta claves st.session_state
- Genera:
  * MAPA_REGULACION.txt
  * MAPA_REGULACION.json
  * imports.dot
  * flujo_regulacion.dot
"""

import os, re, ast, json, argparse
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional

# =========================
# Configuración
# =========================
BASES_PREDETERMINADAS = ["modulos"]
INCLUIR_PREDETERMINADOS = ["app.py"]
SALIDA_TXT = "MAPA_REGULACION.txt"

PATRONES_SESSION = [
    r"st\.session_state\[['\"]([^'\"]+)['\"]\]",
    r"st\.session_state\.get\(\s*['\"]([^'\"]+)['\"]",
]

# Clasificación semántica por nombre de archivo
CLASIFICACION = {
    "datos": "📊 Datos",
    "cargas": "📊 Datos",
    "demanda": "📊 Datos",
    "lineas": "⚡ Cálculo",
    "calculos": "⚡ Cálculo",
    "corrientes": "🔌 Corrientes",
    "matrices": "📐 Matrices",
    "pdf": "📄 Reportes",
    "graficos": "📄 Reportes",
    "secciones": "📄 Reportes",
    "app": "🎛️ UI",
}

# =========================
# Utilidades
# =========================
def leer(ruta: str) -> str:
    with open(ruta, "r", encoding="utf-8") as f:
        return f.read()

def modulo_relativo(ruta: str) -> str:
    rel = os.path.relpath(ruta, start=".")
    rel = rel.replace("\\", "/")
    if rel.endswith(".py"):
        rel = rel[:-3]
    return rel.replace("/", ".")

def recorrer_py(bases, incluir):
    out = []
    for b in bases:
        for r, _, files in os.walk(b):
            for f in files:
                if f.endswith(".py"):
                    out.append(os.path.join(r, f))
    for f in incluir:
        if os.path.isfile(f):
            out.append(f)
    return sorted(set(map(os.path.abspath, out)))

# =========================
# AST
# =========================
class Info(ast.NodeVisitor):
    def __init__(self, modulo, fuente):
        self.modulo = modulo
        self.funciones = set()
        self.imports = set()
        self.llamadas = {}
        self.session_keys = set()
        self._actual = None

        for p in PATRONES_SESSION:
            for m in re.findall(p, fuente):
                self.session_keys.add(m)

    def visit_Import(self, n):
        for a in n.names:
            self.imports.add(a.name)

    def visit_ImportFrom(self, n):
        if n.module:
            self.imports.add(n.module)

    def visit_FunctionDef(self, n):
        self.funciones.add(n.name)
        ant = self._actual
        self._actual = n.name
        self.generic_visit(n)
        self._actual = ant

    def visit_Call(self, n):
        nombre = None
        if isinstance(n.func, ast.Name):
            nombre = n.func.id
        elif isinstance(n.func, ast.Attribute):
            nombre = n.func.attr
        if nombre and self._actual:
            self.llamadas.setdefault(self._actual, set()).add(nombre)
        self.generic_visit(n)

# =========================
# Análisis
# =========================
def analizar_archivo(ruta):
    src = leer(ruta)
    mod = modulo_relativo(ruta)
    tree = ast.parse(src)
    info = Info(mod, src)
    info.visit(tree)
    return info

def clasificar(modulo: str) -> str:
    for k, v in CLASIFICACION.items():
        if k in modulo:
            return v
    return "📦 Otros"

def construir_proyecto():
    archivos = recorrer_py(BASES_PREDETERMINADAS, INCLUIR_PREDETERMINADOS)
    proyecto = {}

    for r in archivos:
        try:
            info = analizar_archivo(r)
            proyecto[info.modulo] = {
                "ruta": r,
                "tipo": clasificar(info.modulo),
                "funciones": sorted(info.funciones),
                "imports": sorted(info.imports),
                "llamadas": {k: sorted(v) for k, v in info.llamadas.items()},
                "session_state": sorted(info.session_keys),
            }
        except Exception as e:
            proyecto[modulo_relativo(r)] = {"error": str(e)}

    return proyecto

def grafo_imports(proyecto):
    mods = set(proyecto.keys())
    edges = set()
    for m, d in proyecto.items():
        for i in d.get("imports", []):
            if i in mods:
                edges.add((m, i))
            for lm in mods:
                if lm.startswith(i + "."):
                    edges.add((m, lm))
    return sorted(edges)

# =========================
# Salidas
# =========================
def escribir_txt(proyecto, imports):
    L = []
    L.append("MAPA DE LA APP – REGULACIÓN DE VOLTAJE")
    L.append(f"Generado: {datetime.now():%Y-%m-%d %H:%M}")
    L.append("=" * 100)

    for m, d in sorted(proyecto.items()):
        L.append(f"\n📄 {m}")
        L.append(f"Tipo: {d.get('tipo')}")
        L.append(f"Ruta: {d.get('ruta')}")
        if d.get("funciones"):
            L.append("Funciones:")
            for f in d["funciones"]:
                L.append(f"  - {f}")
        if d.get("session_state"):
            L.append("st.session_state:")
            for k in d["session_state"]:
                L.append(f"  - {k}")

    L.append("\n📊 IMPORTS INTERNOS")
    for a, b in imports:
        L.append(f"  {a} → {b}")

    with open(SALIDA_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

def escribir_dot(edges, ruta, label):
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("digraph G {\nrankdir=LR;\n")
        f.write(f'label="{label}"; labelloc="t";\n')
        for a, b in edges:
            f.write(f'"{a}" -> "{b}";\n')
        f.write("}\n")

# =========================
# Main
# =========================
def main():
    proyecto = construir_proyecto()
    imports = grafo_imports(proyecto)

    escribir_txt(proyecto, imports)

    with open("MAPA_REGULACION.json", "w", encoding="utf-8") as f:
        json.dump(proyecto, f, indent=2, ensure_ascii=False)

    escribir_dot(imports, "imports.dot", "Imports internos – Regulación de Voltaje")

    print("✅ Análisis completado")

if __name__ == "__main__":
    main()
