# Introducción

**Autores:** Daniel Carbonó y Luis Pérez.

## El problema

El objetivo es predecir, para cada colaborador activo al inicio de un mes, la probabilidad de que
renuncie voluntariamente en ese mes. Es un problema de clasificación binaria sobre un panel
persona-mes: cada fila es una persona en un mes, y la variable objetivo `y_renuncia` vale 1 si en
ese mes su vínculo laboral termina por renuncia voluntaria.

El área de talento humano puede intervenir antes de una renuncia (condiciones, contrato,
acompañamiento) si sabe a quién dirigirse. El modelo se usaría como una lista mensual de las
personas con mayor riesgo; por eso la métrica principal es la proporción de renuncias que quedan
dentro de esa lista, y no la exactitud.

## Los datos

El panel proviene del sistema de nómina SAP de un grupo empresarial colombiano dedicado
principalmente a la producción y comercialización de oleaginosas, que autorizó su uso académico.
Tiene 85.223 persona-mes de 5.747 personas en 22 sociedades, de enero de 2025 a agosto de 2026, con
892 renuncias (prevalencia de 1,05 %). Los datos no se publican: aun seudonimizados permiten
reidentificar a las personas (capítulo 1). El libro contiene el código, las salidas agregadas y las
métricas.

Todo el análisis exploratorio se hace con el conjunto de entrenamiento. El conjunto de prueba
(*test*, los últimos cuatro meses) se usa una sola vez, al final del capítulo 7.

## Reproducir

Entorno `ml_venv` del curso (Python 3.9 y el `requirements.txt` del curso), con el panel en la ruta
que indique la variable de entorno `PANEL_ROTACION`:

```
python compilar.py          # ejecuta fuentes/*.py y escribe los .ipynb con salidas
jupyter-book build .        # arma el libro en _build/html
```

Semilla global: 2026.
