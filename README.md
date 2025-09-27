Análisis de Regulación en red secundaria. 

Proyecto: Análisis de regulación de voltaje

Este repositorio contiene scripts para calcular flujo de carga, pérdidas, regulación de voltaje y generar informes PDF (versión larga y versión corta).

Estructura y descripción de archivos

módulo_de_regulacion_de_voltaje.py
Lógica principal de análisis (carga de datos, flujo de carga, pérdidas, regulación, proyecciones).

Suele exponer una función tipo main_con_ruta_archivo(ruta_excel) o similar que genera el informe largo (PDF).

informe_corto.py
Genera el informe corto (1 página/3 columnas) en PDF usando ReportLab.
Lee resultados desde el Excel/procesamiento y arma tablas + gráficos.

utilidades_red.py
Funciones auxiliares (cálculos, limpieza de datos, formatos, etc.) usadas por los módulos anteriores.

aplicación.py
Script de entrada (launcher). Puedes ejecutarlo para correr el flujo completo o servir de ejemplo de uso.

datos_circuito.xlsx
Archivo de entrada con la definición del circuito, parámetros y datos para el análisis.

Imagen encabezada.jpg
Imagen usada como fondo/encabezado en los PDF.

Asegúrate de que el código usa exactamente este nombre (incluyendo mayúsculas/minúsculas y espacio).

requisitos.txt
Dependencias de Python (por ejemplo reportlab, pandas, numpy).

LÉAME.md
Este archivo de documentación.

LICENCIA
Licencia del proyecto.

.devcontainer/ y devcontainer.json
Configuración para GitHub Codespaces/Dev Containers (entorno reproducible).

Si usas Codespaces, coloca devcontainer.json dentro de .devcontainer/ (recomendado).
