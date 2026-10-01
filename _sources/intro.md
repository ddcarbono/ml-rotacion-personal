# Introducción

**Autores:** Daniel Carbonó y Luis Pérez.

## Resumen

La renuncia voluntaria obliga a reemplazar y a volver a entrenar a las personas, y el área de talento humano solo
puede intervenir antes si sabe a quién dirigirse. Este trabajo prepara un problema de clasificación binaria para
predecir, al inicio de cada mes, la probabilidad de que cada colaborador renuncie en ese mes. Se usa un panel
persona-mes de un grupo agroindustrial colombiano: 85.068 observaciones de 5.738 personas en 22 sociedades, de enero
de 2025 a agosto de 2026, con 837 renuncias (prevalencia de 0,98 %). Cada fila combina atributos del trabajador y
del puesto con la historia laboral reciente (contrato, trayectoria, posición salarial relativa, jornada, ingreso,
ausencias no médicas, vacaciones, proyectos personales y origen); los datos sensibles (salud, riesgo psicosocial, sindicato) y los de clima,
desempeño y sanciones quedan fuera. El análisis exploratorio, hecho solo con el entrenamiento, muestra que la antigüedad reconocida y el tipo de
contrato concentran la señal, que el faltante es estructural e informativo y que hay deriva de nivel entre 2025 y
2026. Con partición cronológica (conjunto de prueba o test: mayo a agosto de 2026) y validación de ventana creciente, una regresión
logística regularizada alcanza en el test una PR-AUC de 0,028 frente a 0,008 de la línea base trivial y un ROC-AUC
de 0,764, y la lista mensual del 10 % de mayor riesgo contiene el 34 % de las renuncias.

**Palabras clave:** rotación de personal, renuncia voluntaria, datos de panel, clases desbalanceadas, regresión
logística, validación temporal, fuga de datos.

## El problema

Para cada colaborador activo al inicio de un mes se estima la probabilidad de que renuncie voluntariamente en ese
mes. Cada fila es una persona en un mes, y la variable objetivo `y_renuncia` vale 1 si en ese mes su vínculo termina
por renuncia voluntaria. El modelo se usaría como una lista mensual de las personas con mayor riesgo; por eso la
métrica principal es la PR-AUC y la proporción de renuncias que quedan dentro de esa lista, y no la exactitud.

## Metodología

| | |
|---|---|
| Población y muestra | colaboradores de 22 sociedades del grupo, sin aprendices, enero de 2025 a agosto de 2026 (capítulo 1) |
| Unidad de análisis | persona-mes; 5.738 personas son las entidades independientes |
| Variables | 18 atributos del trabajador y del puesto y 87 de historia laboral en nueve bloques (diccionario, capítulo 1) |
| Diseño | partición cronológica decidida antes del análisis exploratorio: entrenamiento de enero de 2025 a abril de 2026, test de mayo a agosto de 2026 (capítulo 2) |
| Análisis exploratorio | objetivo, univariado y bivariado con tamaños de efecto y corrección por comparaciones múltiples (capítulo 3), multivariado (capítulo 4), temporal (capítulo 5) y auditoría de fuga (capítulo 6), todo con el entrenamiento |
| Modelo base | `DummyClassifier` y regresión logística regularizada en un `Pipeline`, con hiperparámetros elegidos por validación de ventana creciente con un mes de separación (capítulo 7) |
| Validez | intervalos por *bootstrap* por personas, pruebas pareadas entre modelos, DeLong con covarianza, diagnóstico de sobreajuste, curva de aprendizaje, calibración y autocorrelación de residuos (capítulo 7) |

El conjunto de prueba se usa una sola vez, al final del capítulo 7. Las discusiones de resultados y limitaciones
están en la síntesis de cada capítulo y al cierre del capítulo 7.

## Los datos

El panel proviene del sistema de nómina y de los registros de personal de la empresa, que autorizó su uso académico.
La licencia es de uso académico, no redistribuible: el panel no se publica en este repositorio, porque aun
anonimizado conserva combinaciones de atributos que pueden describir a una persona (capítulo 1). Se entrega a la universidad como un archivo CSV
anonimizado por un canal privado. El libro contiene el código, las salidas agregadas y las métricas.

## Reproducir

Entorno de ejecución (Python 3.9 y las versiones de `requirements.txt`), con el panel en la ruta que indique la
variable de entorno `PANEL_ROTACION` (o en `datos/panel_rotacion_2025_2026.csv` junto a este libro):

```
python compilar.py          # ejecuta fuentes/*.py y escribe los .ipynb con salidas
jupyter-book build .        # arma el libro en _build/html
```

Semilla global: 2026.
