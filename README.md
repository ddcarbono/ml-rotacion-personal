# Rotacion voluntaria de personal - Entregable 1 (Machine Learning)

Autores: Daniel Carbonó y Luis Pérez.

Jupyter Book con el análisis exploratorio y el modelo base (línea base trivial y regresión logística). Los `.ipynb`
están ejecutados: se leen con sus salidas sin necesidad de datos.

Los datos no están en este repositorio: son datos de personas de la empresa, con autorización de uso académico
(capítulo 1). El panel `panel_rotacion_2025_2026.csv` (CSV anonimizado) se entrega por separado por el canal del
curso. Para volver a ejecutar:

```
set PANEL_ROTACION=C:\ruta\al\panel_rotacion_2025_2026.csv
python compilar.py          # ejecuta fuentes/*.py y escribe los .ipynb con salidas (kernel ml_venv)
jupyter-book build .        # arma el libro en _build/html
```

o copiar el panel a `datos/panel_rotacion_2025_2026.csv` junto a este archivo (la carpeta `datos/` no se versiona).

- `fuentes/`: el código de cada capítulo (formato percent). Se edita aquí, no en los `.ipynb`.
- `comun.py`: ruta del panel, semilla, partición, grupos de variables, estilo de los gráficos y funciones comunes.
- `diccionario.csv`: tipo, unidad, significado y disponibilidad de cada columna del panel.
- `referencias.bib`: bibliografía.
- `_static/identidad.css`: colores del libro.
- `requirements.txt`: versiones del entorno del curso (Python 3.9).
