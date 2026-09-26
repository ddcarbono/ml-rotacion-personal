# Rotacion voluntaria de personal - Entregable 1 (Machine Learning)

Autores: Daniel Carbonó y Luis Pérez.

Jupyter Book con el EDA y el modelo base. Los `.ipynb` estan ejecutados: se leen con sus salidas sin
necesidad de datos.

Los datos no estan en este repositorio: son datos de personas de la empresa, con autorizacion para
uso academico (capitulo 1). Para volver a ejecutar hace falta el panel
`panel_rotacion_2025_2026.csv`, que se entrega por separado:

```
set PANEL_ROTACION=C:\ruta\al\panel_rotacion_2025_2026.csv
python compilar.py          # ejecuta fuentes/*.py y escribe los .ipynb con salidas (kernel ml_venv)
jupyter-book build .        # arma el libro en _build/html
```

- `fuentes/`: el codigo de cada capitulo (formato percent). Se edita aqui, no en los `.ipynb`.
- `comun.py`: ruta del panel, semilla, particion, grupos de variables, estilo de los graficos y
  funciones de tasas y metricas que usan los capitulos.
- `_static/identidad.css`: colores del libro.
- `requirements.txt`: versiones del entorno del curso.
