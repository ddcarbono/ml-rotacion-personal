# %% [markdown]
# # Partición
#
# ```{admonition} Alcance
# :class: tip
# El corte se decidió antes del análisis exploratorio: el conjunto de prueba (*test*) son los
# últimos cuatro meses, de mayo a agosto de 2026 (constante `CORTE` en `comun.py`). No se movió al
# cambiar la base.
# ```
#
# ## Por qué un corte en el tiempo
#
# El modelo se usaría para predecir, con la información disponible al inicio de un mes, quién
# renuncia en ese mes, y la evaluación debe reproducir ese uso. Una partición aleatoria por filas
# pondría meses futuros en el entrenamiento (el modelo aprendería de agosto para predecir marzo) y
# meses de una misma persona en los dos conjuntos. Una partición estratificada por la clase, la que
# el curso sugiere para clasificación desbalanceada, tiene el mismo problema. Con el corte
# cronológico, todo lo que el modelo ve es anterior a lo que predice; el desbalance se conserva de
# forma natural porque la prevalencia mensual es estable (tabla siguiente).
#
# Las mismas personas aparecen en entrenamiento y en test, porque es un panel y quien estaba en abril
# sigue en mayo. Esto no constituye una fuga: en producción el modelo también puntúa a personas que
# ya conoce. Habría fuga si la etiqueta de un mes de test entrara al entrenamiento, y el corte lo
# impide.
#
# ## Uso previo del panel completo
#
# Para construir el panel se exploraron todos los meses: se definieron las exclusiones, se unificaron
# los cargos, se corrigió el objetivo (capítulo 1) y se revisaron las fugas. Son decisiones de
# calidad de datos (qué es un aprendiz, qué salida es una renuncia, qué dato está mal registrado) y
# ninguna se tomó según su capacidad predictiva. A partir de aquí el test queda reservado: el
# análisis exploratorio, la selección de variables, las vallas de atípicos, la imputación y el ajuste
# usan solo el entrenamiento, y el test se usa una vez, en el capítulo 7.

# %%
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from comun import (cargar, particion, estilo, etiquetar, pliegues_temporales, tasa_ic, OBJETIVO,
                   PREDICTORAS, CORTE, SEMILLA, VERDE, ORO, BORDE)

estilo()
pd.set_option('display.width', 160)

p = cargar()
tr, te = particion(p)

resumen = pd.DataFrame({
    'meses': [tr.mes.nunique(), te.mes.nunique()],
    'desde': [tr.mes.min(), te.mes.min()],
    'hasta': [tr.mes.max(), te.mes.max()],
    'filas': [len(tr), len(te)],
    '% filas': [round(100 * len(tr) / len(p), 1), round(100 * len(te) / len(p), 1)],
    'personas': [tr.persona_id.nunique(), te.persona_id.nunique()],
    'renuncias': [int(tr[OBJETIVO].sum()), int(te[OBJETIVO].sum())],
    '% renuncias': [round(100 * tr[OBJETIVO].sum() / p[OBJETIVO].sum(), 1),
                    round(100 * te[OBJETIVO].sum() / p[OBJETIVO].sum(), 1)],
    'prevalencia %': [round(100 * tr[OBJETIVO].mean(), 2), round(100 * te[OBJETIVO].mean(), 2)],
}, index=['entrenamiento', 'test'])
resumen

# %% [markdown]
# El entrenamiento tiene 16 meses (enero de 2025 a abril de 2026), 67.416 filas y 688 renuncias; el
# test, 4 meses, 17.652 filas (20,8 %) y 149 renuncias (17,8 % de las renuncias). La prevalencia es
# algo menor en el test (0,84 % frente a 1,02 %): con menos positivos, la precisión esperable de
# cualquier modelo baja sin que el modelo haya empeorado. Por eso las métricas
# del capítulo 7 se comparan siempre con la línea base trivial del mismo test.

# %%
en_ambos = len(set(tr.persona_id) & set(te.persona_id))
solo_te = te.loc[~te.persona_id.isin(tr.persona_id), 'persona_id'].nunique()
print(f'personas en los dos conjuntos: {en_ambos:,} de {te.persona_id.nunique():,} del test '
      f'({100 * en_ambos / te.persona_id.nunique():.0f} %) | solo en test (ingresos nuevos): {solo_te:,}')
print(f'renuncias del test de personas que ya estaban en entrenamiento: '
      f'{int(te.loc[te.persona_id.isin(tr.persona_id), OBJETIVO].sum())} de {int(te[OBJETIVO].sum())}')

por_mes = p.groupby('mes').agg(filas=(OBJETIVO, 'size'), renuncias=(OBJETIVO, 'sum'))
colores = [VERDE if mes < CORTE else ORO for mes in por_mes.index]

fig, ax = plt.subplots(figsize=(11, 3.2))
barras = ax.bar(por_mes.index, por_mes.renuncias, color=colores, **BORDE)
etiquetar(ax, barras)
ax.set(title=f'Renuncias por mes (verde: entrenamiento, dorado: test; total {int(por_mes.renuncias.sum())})',
       ylabel='Número de renuncias')
ax.tick_params(axis='x', rotation=60)
plt.tight_layout()
plt.show()

# %% [markdown]
# De las 4.611 personas del test, 4.373 (95 %) ya estaban en el entrenamiento y 238 son ingresos
# nuevos; 139 de las 149 renuncias del test son de personas que el modelo ya vio en meses anteriores,
# que es exactamente la situación de uso.
#
# Nótese que el número de renuncias por mes varía bastante (el gráfico lo muestra mes a mes; el
# capítulo 5 estudia su estacionalidad). Con cuatro meses de test, la estimación de cualquier métrica
# depende de unas 150 renuncias, y su incertidumbre se mide con *bootstrap* en el capítulo 7.
#
# ## Validación dentro del entrenamiento
#
# La validación cruzada sigue la misma lógica de la partición: es cronológica, con ventana creciente
# por mes y un mes de separación (`gap`), implementada en `comun.pliegues_temporales`. Cada pliegue
# valida un mes, de septiembre de 2025 a abril de 2026, y entrena con todos los meses anteriores salvo
# el inmediatamente previo. La separación se debe a que una renuncia puede quedar registrada ya
# entrado el mes siguiente, de modo que al predecir el mes *t* la etiqueta de *t − 1* puede no estar
# disponible. Con esta validación se eligen el hiperparámetro y el tamaño de la lista (capítulo 7).

# %%
pl = pliegues_temporales(tr.mes.to_numpy(), primer_val='2025-09', gap=1)
y_tr = tr[OBJETIVO].to_numpy()
mes_tr = tr.mes.to_numpy()
tabla_pl = pd.DataFrame([{
    'mes validado': mes_tr[va][0],
    'entrena hasta': mes_tr[en].max(),
    'meses de entrenamiento': len(np.unique(mes_tr[en])),
    'filas entrenamiento': len(en),
    'renuncias entrenamiento': int(y_tr[en].sum()),
    'filas validación': len(va),
    'renuncias validación': int(y_tr[va].sum()),
} for en, va in pl], index=pd.RangeIndex(1, len(pl) + 1, name='pliegue'))
tabla_pl

# %% [markdown]
# Son 8 pliegues. El primero entrena con 7 meses (enero a julio de 2025) y valida septiembre; el
# último entrena con 14 meses y valida abril de 2026. Agosto de 2025 nunca entrena al pliegue que
# valida septiembre: es el mes de separación. Cada mes validado tiene entre 31 y 60 renuncias, pocas
# para una métrica estable en un solo pliegue; por eso el capítulo 7 promedia los ocho y mira también
# su dispersión.
#
# ### Por qué no `TimeSeriesSplit(gap=...)`
#
# `TimeSeriesSplit` de scikit-learn parte por **posición de fila**, no por fecha, y su `gap` cuenta
# filas. En un panel eso falla de dos maneras: si las filas están ordenadas por persona (como las deja
# `cargar()`), cada pliegue mezcla todos los meses; y aun ordenadas por mes, los cortes caen a mitad
# de un mes y `gap=1` separa una sola fila, cuando un mes tiene más de 4.000. La celda lo comprueba.

# %%
def meses_compartidos(orden, n_splits=8, gap=1):
    """Para TimeSeriesSplit sobre el entrenamiento en el orden dado: por pliegue, cuántos meses de la
    validación aparecen también en el entrenamiento y cuántas filas separa el gap."""
    d = tr.sort_values(orden).reset_index(drop=True)
    out = []
    for en, va in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(d):
        m_en, m_va = set(d.mes.iloc[en]), set(d.mes.iloc[va])
        out.append((len(m_va), len(m_va & m_en), va[0] - en[-1] - 1))
    return pd.DataFrame(out, columns=['meses en validación', 'de ellos también en entrenamiento',
                                      'filas de separación'])

comparacion = pd.concat({'orden persona, mes': meses_compartidos(['persona_id', 'mes']),
                         'orden mes, persona': meses_compartidos(['mes', 'persona_id'])}, axis=1)
comparacion.index = pd.RangeIndex(1, 9, name='pliegue')
print(f'filas por mes en el entrenamiento: {tr.groupby("mes").size().min():,} a {tr.groupby("mes").size().max():,}')
comparacion

# %% [markdown]
# Ordenado por persona, cada pliegue de validación contiene 16 meses y todos están también en el
# entrenamiento: es una partición aleatoria disfrazada. Ordenado por mes, los pliegues validan dos o
# tres meses y en todos uno de ellos queda partido entre entrenamiento y validación; la separación es
# de una fila, no de un mes. `pliegues_temporales` define los pliegues por el valor del mes, por eso
# no tiene ninguno de los dos problemas.
#
# ### Validación agrupada por persona
#
# Como complemento se usa una validación de cinco pliegues agrupada por persona (`GroupKFold`), en la
# que todos los meses de una persona caen en el mismo pliegue. Mide si el modelo generaliza a personas
# que no vio, aunque mezcla meses (no respeta el tiempo): por eso es complemento y no la validación
# principal.

# %%
gkf = GroupKFold(n_splits=5)
tabla_g = pd.DataFrame([{
    'personas validación': tr.persona_id.iloc[va].nunique(),
    'filas validación': len(va),
    'renuncias validación': int(y_tr[va].sum()),
    'prevalencia % validación': round(100 * y_tr[va].mean(), 2),
    'personas en ambos lados': len(set(tr.persona_id.iloc[en]) & set(tr.persona_id.iloc[va])),
} for en, va in gkf.split(tr, y_tr, groups=tr.persona_id)], index=pd.RangeIndex(1, 6, name='pliegue'))
tabla_g

# %% [markdown]
# Ninguna persona cae en dos lados de un pliegue, y las renuncias se reparten de forma pareja (entre
# 129 y 144 por pliegue, prevalencias de 0,96 % a 1,07 %).
#
# ## Duplicados y entidades entre particiones

# %%
print('filas persona-mes duplicadas en el panel:', int(p.duplicated(['persona_id', 'mes']).sum()))

pares_tr = set(zip(tr.persona_id, tr.mes))
pares_te = set(zip(te.persona_id, te.mes))
print('pares (persona, mes) en entrenamiento y test a la vez:', len(pares_tr & pares_te))

# vectores de predictoras del test que aparecen idénticos en el entrenamiento
vec_tr = set(map(tuple, tr[PREDICTORAS].astype(str).to_numpy()))
igual = te[PREDICTORAS].astype(str).apply(tuple, axis=1).isin(vec_tr).to_numpy()
print(f'filas de test con un vector de las {len(PREDICTORAS)} predictoras idéntico en entrenamiento: {int(igual.sum()):,} '
      f'de {len(te):,} ({100 * igual.mean():.2f} %)')
BASE18 = PREDICTORAS[:18]
vec_tr18 = set(map(tuple, tr[BASE18].astype(str).to_numpy()))
igual18 = te[BASE18].astype(str).apply(tuple, axis=1).isin(vec_tr18).to_numpy()
print(f'lo mismo con solo las 18 atributos: {int(igual18.sum()):,} ({100 * igual18.mean():.1f} %)')

# %% [markdown]
# Ninguna fila está en los dos conjuntos. Con las 18 atributos, el 10,1 % de las filas de test
# (1.785) tenía un vector idéntico a alguno del entrenamiento: personas distintas con el mismo perfil
# de puesto, frecuentes en fincas con muchos operarios del mismo tipo. Con las 105 predictoras solo
# 53 filas (0,30 %) lo tienen, porque la historia laboral (jornada, nómina, vacaciones) distingue a esas personas. En
# ningún caso hay fuga: la etiqueta de test corresponde a un mes posterior.
#
# ```{admonition} Sensibilidad del objetivo: el preaviso de no renovación
# :class: warning
# Una renuncia registrada cuando ya existía una decisión de no renovar el contrato (tomada antes del
# retiro, para un contrato que terminaba en ese mes o en los tres siguientes) no se cuenta como
# renuncia: la empresa ya había decidido terminar el contrato y no es rotación evitable. Al construir
# la base, 85 de las 892 renuncias anteriores pasaron a otra salida por esta regla, 19 de ellas en el
# test (capítulo 1).
#
# La columna `preaviso_no_renovacion` marca los meses con un preaviso vigente. **No es predictora:**
# con preaviso, el objetivo vale 0 por construcción, y un modelo que la usara aprendería la regla con
# que se definió *y*, no el comportamiento. Queda solo para describir y para la sensibilidad (celda
# siguiente). Tampoco son predictoras la bonificación de salida (según la construcción de la base, 16
# renuncias la recibieron; quedan en *y*, y la marca no está en el panel académico porque se conoce al
# salir) ni el tipo de retiro. Las renuncias con bonificación de salida o con indicios de haber sido
# pedidas se identificaron al construir la base con fuentes que el panel no incluye por privacidad,
# así que su análisis de sensibilidad no es reproducible con el panel publicado.
# ```

# %%
filas = []
for nombre, d in [('entrenamiento', tr), ('test', te)]:
    con = d.preaviso_no_renovacion.eq(1)
    filas.append({
        'conjunto': nombre, 'filas con preaviso': int(con.sum()), '% filas': round(100 * con.mean(), 2),
        'personas con preaviso': d.loc[con, 'persona_id'].nunique(),
        'renuncias con preaviso': int(d.loc[con, OBJETIVO].sum()),
        'otras salidas con preaviso': int(d.loc[con, 'evento'].eq('otra_salida').sum()),
        'prevalencia % (todas)': round(100 * d[OBJETIVO].mean(), 3),
        'prevalencia % (sin filas con preaviso)': round(100 * d.loc[~con, OBJETIVO].mean(), 3),
    })
pd.DataFrame(filas).set_index('conjunto')

# %% [markdown]
# Hay 469 filas con preaviso en el entrenamiento (300 personas, 0,70 %) y 161 en el test (127
# personas, 0,91 %). Tienen cero renuncias en los dos conjuntos, como exige la regla, y la mayoría
# termina en otra salida (199 y 66). Sacarlas del conjunto en riesgo mueve la prevalencia de 1,021 %
# a 1,028 % en el entrenamiento y de 0,844 % a 0,852 % en el test: la prevalencia apenas
# depende de esa decisión. La alternativa, dejar esas 85 renuncias como renuncias, las habría
# puesto en personas a las que ya se les había comunicado el fin del contrato: su "renuncia" es una
# forma administrativa de una salida decidida por la empresa.
#
# ## Síntesis
#
# - El test son los cuatro últimos meses (17.652 filas y 149 renuncias, prevalencia 0,84 %) y se usa
#   una sola vez; el entrenamiento, 16 meses (67.416 filas y 688 renuncias, 1,02 %).
# - La validación dentro del entrenamiento es temporal: 8 pliegues de ventana creciente, de
#   septiembre de 2025 a abril de 2026, con un mes de separación (`pliegues_temporales`, no
#   `TimeSeriesSplit`, cuyo `gap` cuenta filas). La agrupada por persona (`GroupKFold`) es un
#   complemento.
# - Las mismas personas están en los dos lados, sin pares persona-mes repetidos; es el uso real.
# - `preaviso_no_renovacion` no entra al modelo: con preaviso *y* = 0 por construcción.
# - Desde aquí, todas las tablas y figuras del análisis exploratorio usan solo el entrenamiento.
