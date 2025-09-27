Análisis de Regulación en red secundaria. 

Proyecto: Análisis de regulación de voltaje

Definición: 
Este repositorio contiene scripts para calcular flujo de carga, pérdidas, regulación de voltaje y generar informes PDF (versión larga y versión corta).

Estructura y descripción de archivos
1.módulo_de_regulacion_de_voltaje.py
  Lógica principal de análisis (carga de datos, flujo de carga, pérdidas, regulación, proyecciones).

2.informe_corto.py
  Genera el informe corto (1 página/3 columnas) en PDF usando ReportLab.
  Lee resultados desde el Excel/procesamiento y arma tablas + gráficos.

3.utilidades_red.py
  Funciones auxiliares (cálculos, limpieza de datos, formatos, etc.) usadas por los módulos anteriores.

4.aplicación.py
  Script de entrada (launcher). Puedes ejecutarlo para correr el flujo completo o servir de ejemplo de uso.

5.datos_circuito.xlsx
  Archivo de entrada con la definición del circuito, parámetros y datos para el análisis.

6.Imagen encabezada.jpg
  Imagen usada como fondo/encabezado en los PDF.

7.requisitos.txt
  Dependencias de Python (por ejemplo reportlab, pandas, numpy).

8.LÉAME.md
  Este archivo de documentación.

9.LICENCIA
  Licencia del proyecto.

10.devcontainer/ y devcontainer.json
  Configuración para GitHub Codespaces/Dev Containers (entorno reproducible).

