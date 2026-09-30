# %% [markdown]
# # Auditoría de fuga
#
# ```{admonition} Alcance
# :class: tip
# Las secciones 6.1 a 6.6 se calculan con el entrenamiento (enero de 2025 a abril de 2026). La sección
# 6.7 compara entrenamiento y test solo por sus predictoras y personas, sin mirar las etiquetas del
# test. Las revisiones hechas al construir el panel sobre las tablas de origen se citan como tales:
# sus cifras no se pueden recalcular desde el panel, porque las columnas con fuga no están en él.
# ```
#
# Una variable tiene fuga cuando contiene información que no estaría disponible en el momento de
# predecir, casi siempre porque se registra a causa del desenlace. La predicción de este proyecto se
# hace **el día 1 del mes *t*** (o el primer día activo, si la persona ingresa a mitad de mes) y se
# refiere a la renuncia **durante el mes *t***. Cualquier dato que se conozca después de esa fecha es
# sospechoso. El capítulo revisa las 105 predictoras en cinco frentes: cuándo se conoce cada una,
# qué columnas derivan del desenlace, qué fugas se corrigieron al construir el panel, cuánto predice
# cada variable sola y si su señal se concentra justo antes de la salida.

# %%
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from comun import (cargar, particion, estilo, etiquetar, pliegues_temporales, OBJETIVO, NUM, BIN, CAT, FUERA,
                   PREDICTORAS, BLOQUE, BLOQUES, NOMBRE, SENSIBILIDAD, DESENLACE,
                   VERDE, ORO, GRIS, BORDE)

estilo()
pd.set_option('display.max_rows', 200)

p = cargar()
tr, te = particion(p)
y = tr[OBJETIVO].to_numpy()
print(f'entrenamiento: {len(tr):,} filas | {int(y.sum())} renuncias | predictoras: {len(PREDICTORAS)} '
      f'({len(NUM)} numéricas, {len(BIN)} binarias de atributos, {len(CAT)} categóricas)')
print('fuera de la historia por el capítulo 1:', FUERA)
print('columnas del panel que no son predictoras:',
      [c for c in p.columns if c not in PREDICTORAS])

# %% [markdown]
# ## Disponibilidad en el momento de la predicción
#
# La tabla siguiente dice, para cada una de las 105 predictoras, **de qué ventana sale** (columna
# `disponible` del diccionario del capítulo 1) y si se conoce el día 1 del mes *t*. La ventana se
# verificó en los programas que construyen cada bloque del panel. Hay tres reglas de fondo:
#
# - **Foto del día 1.** Los atributos, el contrato, la trayectoria, el salario relativo, el horario
#   teórico y el saldo de vacaciones se leen del registro vigente el día 1 (o el primer día activo).
#   Un cambio que empieza ese mismo día ya se conoce; uno que empieza el día 15 no cuenta hasta *t+1*.
# - **Rezago de un mes en nómina y marcaciones.** La nómina del mes *m* se liquida al cierre de *m* y
#   el mes en curso es, justamente, el de la salida: por eso las horas extra, recargos, bonos, ingreso
#   y marcaciones biométricas de la fila *t* terminan en *t−1* (ventanas de 1, 3 y 12 meses). La fecha
#   de las últimas vacaciones pagadas va con dos meses de rezago, porque se suelen tomar justo antes
#   de salir.
# - **Ausencias.** Los registros de novedades (licencias, capacitación, compensatorios, día de la
#   familia; nada de salud) se reparten por días entre los meses que cubren y van con rezago de un
#   mes. Los días de licencia no remunerada van con **dos** meses de rezago (sufijo `_r2`); la razón se
#   ve en la sección 6.2.

# %%
dic = pd.read_csv('diccionario.csv').set_index('variable')

# fuente de cada ventana (verificada en los programas de construcción de cada bloque)
FUENTE = {
    'día 1 del mes t': 'registro vigente el día 1 (o primer día activo)',
    'mes t-1 (cargo del mes anterior)': 'cargo del mes anterior: en el mes de salida el cargo queda vacío',
    'antes del día 1 del mes t': 'eventos con fecha anterior al día 1',
    'perfil del trabajador (fijo)': 'lugar de nacimiento del perfil; no cambia en el tiempo',
    'cierre de nómina del mes t-1': 'nómina liquidada, rezago de un mes',
    'cierres de nómina de t-3 a t-1': 'nómina liquidada, rezago de un mes',
    'cierres de nómina de t-12 a t-1': 'nómina liquidada, rezago de un mes',
    'marcaciones del mes t-1': 'marcaciones biométricas, rezago de un mes',
    'marcaciones de t-3 a t-1': 'marcaciones biométricas, rezago de un mes',
    'marcaciones de t-3 a t-1 y nómina': 'marcaciones y nómina, rezago de un mes',
    'registro de novedades de t-12 a t-1': 'novedades de ausencia, rezago de un mes',
    'registro de novedades de t-4 a t-2': 'novedades de ausencia, rezago de dos meses',
    'registro de novedades de t-13 a t-2': 'novedades de ausencia, rezago de dos meses',
    'nómina hasta t-2 (rezago de dos meses)': 'vacaciones pagadas, rezago de dos meses',
}
# las que se conocen el día 1 pero con una reserva documentada
RESERVA = {
    **{v: 'rezago de dos meses: el mes t-2 puede completarse después del día 1 (6,2 % de las licencias no remuneradas)'
       for v in ['dias_licencia_no_remunerada_3m_r2', 'dias_licencia_no_remunerada_12m_r2']},
    **{v: 'rezago de un mes: el mes t-1 puede completarse ya entrado t (registro tardío de novedades)'
       for v in ['licencias_no_remuneradas_12m', 'dias_licencia_remunerada_12m', 'dias_capacitacion_12m',
                 'dias_compensatorio_12m', 'dia_familia_12m']},
    **{v: 'saldo reconstruido; la compensación en dinero se fecha con la última modificación del registro'
       for v in ['dias_vacaciones_pendientes', 'periodos_vacaciones_pendientes', 'vacaciones_acumuladas_2p',
                 'dias_vacaciones_compensadas_12m']},
    'nacido_en_depto_sede': 'el departamento de la sede se estima con los perfiles actuales',
}

disp = pd.DataFrame({
    'bloque': [BLOQUE[v] for v in PREDICTORAS],
    'variable': PREDICTORAS,
    'ventana': [dic.disponible.get(v, '(sin diccionario)') for v in PREDICTORAS],
})
disp['fuente'] = disp.ventana.map(FUENTE).fillna('(revisar)')
disp['¿se conoce el día 1 de t?'] = np.where(disp.variable.isin(RESERVA), 'sí, con reserva', 'sí')
disp['reserva'] = disp.variable.map(RESERVA).fillna('')
print('variables sin fuente asignada:', int(disp.fuente.eq('(revisar)').sum()))
resumen = pd.crosstab(disp.bloque, disp['¿se conoce el día 1 de t?'], margins=True, margins_name='total')
resumen

# %% [markdown]
# Nótese que las 105 predictoras se conocen el día 1 del mes que se predice: ninguna usa un dato
# del mes *t* posterior a esa fecha. Noventa y tres lo hacen sin reserva. Doce llevan una reserva
# documentada, que no las saca del modelo pero sí las deja en observación (sección 6.8):
#
# **Las siete ausencias.** La validación de la fuente mide, para cada registro de novedad, si su
# último cambio cae en el mismo mes en que empieza la ausencia, a más tardar el mes siguiente, o
# después:
#
# | clase | mismo mes | a más tardar el mes siguiente | después |
# |---|---|---|---|
# | licencia no remunerada | 64,7 % | 93,8 % | 6,2 % |
# | licencia remunerada | 68,9 % | 92,7 % | 7,3 % |
# | capacitación | 62,6 % | 100,0 % | 0,0 % |
# | compensatorio | 72,7 % | 99,9 % | 0,1 % |
# | día de la familia | 90,0 % | 99,9 % | 0,1 % |
#
# Con rezago de un mes, el mes *t−1* de la ventana solo tiene el día 1 de *t* lo registrado dentro
# del mismo mes (entre el 63 % y el 90 %). Con rezago de dos, el mes *t−2* tiene lo registrado a más
# tardar el mes siguiente: el 93,8 % de las licencias no remuneradas, y queda un 6,2 % que llega
# después. Es una cota superior (el último cambio puede ser la corrección de un registro que ya
# existía) y en las ventanas de 12 meses afecta a uno de los doce meses. Por eso las licencias
# no remuneradas van con dos meses de rezago y las demás ausencias quedan con reserva.
#
# Las otras cinco con reserva:
#
# - **Cuatro de vacaciones.** El saldo del día 1 se reconstruye con la historia de días causados y
#   vacaciones disfrutadas; la compensación en dinero se fecha con la última modificación del
#   registro de días causados, porque la fuente no guarda la fecha de cada liquidación. Si esa modificación ocurre
#   al salir, el saldo de los meses previos queda algo alto. El cotejo del último mes contra el saldo
#   actual coincide (±1 día) en el 93 % de las personas.
# - **`nacido_en_depto_sede`.** El lugar de nacimiento no cambia, pero el departamento de la sede se
#   estima con los perfiles actuales de quienes trabajan en ella.
#
# La tabla completa, variable por variable, está plegada a continuación.

# %% tags=["hide-output"]
disp.set_index(['bloque', 'variable'])

# %% [markdown]
# ## Variables derivadas del objetivo, identificadores y proxies
#
# Además de las tres que el capítulo 1 dejó fuera por calidad (sección 6.8), seis columnas del panel
# no son predictoras: dos describen la salida, una es el propio objetivo, dos son identificadores y la
# última es un *proxy* que el objetivo contiene por construcción.

# %%
print('evento frente al objetivo (entrenamiento):')
print(pd.crosstab(tr.evento, tr[OBJETIVO]).to_string())
print('\ntipo_retiro (solo en otras salidas):')
print(tr.tipo_retiro.value_counts(dropna=False).to_string())
print('\npreaviso de no renovación frente al objetivo:')
print(pd.crosstab(tr.preaviso_no_renovacion, tr[OBJETIVO]).to_string())
pre = tr[tr.preaviso_no_renovacion.eq(1)]
print('\nfilas con preaviso, por evento del mes:')
print(pre.evento.value_counts().to_string())

# %% [markdown]
# Nótese que `evento` reproduce el objetivo exactamente: sus 688 renuncias son las 688 de `y_renuncia`,
# y las otras salidas y las renuncias administrativas (31 filas) quedan en 0. `tipo_retiro` solo existe
# en las 441 otras salidas. Ambas se conocen al cerrar el mes *t*, con la salida ya ocurrida: son el
# desenlace, no predictoras.
#
# `preaviso_no_renovacion` es el caso más peligroso porque **sí** se conoce el día 1: es la decisión,
# registrada antes, de no renovar un contrato que termina en *t* o después. Pero el objetivo se definió
# de modo que una salida con preaviso no cuenta como renuncia (capítulo 2): en las 469 filas con
# preaviso hay 0 renuncias, 199 terminan en otra salida en el mismo mes y 270 siguen activas (el
# contrato vence más adelante). Como predictora, la variable «adivinaría» ceros por construcción de la
# etiqueta, no por el comportamiento de la persona. Se descarta y queda solo para la sensibilidad del
# objetivo.
#
# `persona_id` es un seudónimo al azar: no entra al modelo y solo agrupa filas. `mes` tampoco entra
# como tal (identificaría el periodo y no se repite en el test); el calendario entra solo como una
# indicadora de enero, que es la decisión del capítulo 5.
#
# **Ausencias sin rezago.** El catálogo de construcción del panel tenía los días de licencia no
# remunerada con rezago de un mes. Su AUC univariado era de 0,643 (3 meses) y 0,606 (12 meses), pero
# bajaba a 0,591 y 0,561 con el valor de un mes antes, y a 0,585 y 0,553 con el de dos: la señal se
# concentraba en el mes previo a la salida, donde parte de la licencia es trámite de la propia salida
# y parte se registra después del día 1. Esas dos versiones se descartaron (quedaron solo para
# sensibilidad fuera del libro) y entran las de rezago de dos meses, `_r2`. La sección 6.5 repite la
# prueba con el panel del libro.

# %% [markdown]
# ## Fugas corregidas al construir el panel
#
# El AUC univariado solo detecta la fuga que permanece en el panel. Las más importantes se
# encontraron antes, al revisar cómo registra el sistema de nómina cada salida, y se corrigieron en la
# construcción:
#
# | campo | fuga | evidencia | corrección |
# |---|---|---|---|
# | cargo del mes *t* | en el mes de salida el sistema libera la posición y el cargo queda vacío | vacío en todos los meses de salida y en ninguno del resto | se usa el cargo del mes anterior, del que salen `familia_cargo` y `oficio`; el texto del cargo no está en el panel |
# | contrato | un reporte mensual toma el contrato vigente hoy y lo copia a todos los meses | ningún cambio de contrato dentro de una misma persona en toda la historia | contrato vigente el día 1 de cada mes según la historia de contratos |
# | tipo de personal | la base de costos de personal solo registra al personal activo, y la ausencia de fila delata la salida | faltantes concentrados en los meses de salida | se recalcula con la misma regla contable sobre el centro de costo del mes, que siempre existe |
# | ausencias y nómina del mes *t* | el mes en curso incluye los trámites de la propia salida (liquidación, licencia antes de irse) | la liquidación solo aparece al salir | ventanas que terminan en *t−1*; la liquidación no se usa |
# | licencia no remunerada | parte de la señal es trámite de la salida en el mes previo | AUC que cae al retroceder un mes (sección 6.2) | rezago de dos meses (`_r2`) |
# | pago de salida | un pago no recurrente puede ser el pago de la salida y no una bonificación | patrón revisado en la nómina | los pagos con ese patrón no se cuentan como bonificación |
#
# El panel tampoco trae los textos de cargo, función, departamento y área: de ellos se derivaron los
# ejes (`nivel`, `familia_cargo`, `oficio`, `tipo_unidad`, `area_funcional`) y se quitaron, porque
# son redundantes con esos ejes y porque son cuasi-identificadores.
#
# ## AUC univariado fuera de pliegue
#
# Cada variable se usa sola como puntaje y se evalúa **fuera de pliegue con la validación temporal del
# capítulo 2** (`pliegues_temporales`: cada pliegue valida un mes de septiembre de 2025 a abril de 2026
# y entrena con los meses anteriores, dejando un mes de separación). En cada pliegue, con los meses de
# entrenamiento se decide:
#
# - para una numérica o binaria, la mediana con que se imputa el faltante y el sentido (si el riesgo
#   crece o decrece con el valor);
# - para una categórica, la tasa de renuncia de cada categoría, suavizada hacia la tasa global con un
#   peso de 30 filas (una categoría con pocas filas no puede dar una tasa de 100 %); una categoría que
#   no aparece en el entrenamiento recibe la tasa global.
#
# El AUC se calcula con los puntajes de los ocho meses de validación juntos. Se agrega la indicadora
# de enero (`enero`) como una variable más, porque es el calendario que entra al modelo del
# capítulo 7. Como referencia, el AUC
# **en muestra** usa todo el entrenamiento para elegir el sentido o la tasa y para evaluarlo: la
# diferencia entre los dos mide cuánto de la señal aparente es sobreajuste de la propia variable.

# %%
tr['enero'] = tr.mes.str[5:].eq('01').astype(int)   # calendario del modelo (capítulo 5)
pliegues = pliegues_temporales(tr.mes)
M_SUAVE = 30


def puntaje_cat(x_ent, y_ent, x_val):
    glob = y_ent.mean()
    g = pd.DataFrame({'x': x_ent.to_numpy(), 'y': y_ent}).groupby('x').y.agg(['sum', 'size'])
    t = (g['sum'] + M_SUAVE * glob) / (g['size'] + M_SUAVE)
    return x_val.map(t).fillna(glob).to_numpy()


def auc_univariado(v):
    """AUC fuera de pliegue (temporal) y en muestra de una variable sola."""
    es_cat = v in CAT
    x = tr[v].astype(str).where(tr[v].notna(), '(nulo)') if es_cat else tr[v].astype(float)
    oof, yy, por_pliegue = [], [], []
    for ent, val in pliegues:
        if es_cat:
            s = puntaje_cat(x.iloc[ent], y[ent], x.iloc[val])
        else:
            med = x.iloc[ent].median()
            xe = x.iloc[ent].fillna(med)
            signo = 1 if roc_auc_score(y[ent], xe) >= 0.5 else -1
            s = signo * x.iloc[val].fillna(med).to_numpy()
        oof.append(s)
        yy.append(y[val])
        por_pliegue.append(roc_auc_score(y[val], s))
    auc_oof = roc_auc_score(np.concatenate(yy), np.concatenate(oof))
    # en muestra: parámetros con todo el entrenamiento, evaluado en las mismas filas de validación
    iv = np.concatenate([val for _, val in pliegues])
    if es_cat:
        a_in = roc_auc_score(y[iv], puntaje_cat(x, y, x.iloc[iv]))
        sentido = 'tasa de la categoría'
    else:
        a = roc_auc_score(y, x.fillna(x.median()))
        s_in = (1 if a >= 0.5 else -1) * x.iloc[iv].fillna(x.median()).to_numpy()
        a_in = roc_auc_score(y[iv], s_in)
        sentido = 'más alto, más riesgo' if a >= 0.5 else 'más bajo, más riesgo'
    return {'variable': v, 'bloque': BLOQUE.get(v, 'calendario'), 'AUC fuera de pliegue': auc_oof,
            'mín. pliegue': min(por_pliegue), 'máx. pliegue': max(por_pliegue), 'AUC en muestra': a_in,
            'sentido': sentido, 'nulos %': 100 * tr[v].isna().mean()}


auc = (pd.DataFrame([auc_univariado(v) for v in PREDICTORAS + ['enero']])
       .set_index('variable').sort_values('AUC fuera de pliegue', ascending=False))
auc['alerta'] = np.where(auc['AUC fuera de pliegue'] > 0.8, 'AUC > 0,8: revisar', '')
print(f'pliegues: {len(pliegues)} | variables: {len(auc)} | con AUC > 0,8: {int(auc.alerta.ne("").sum())}')
print(f'AUC fuera de pliegue: máximo {auc["AUC fuera de pliegue"].max():.3f} | '
      f'mediana {auc["AUC fuera de pliegue"].median():.3f}')
print('variables por tramo de AUC fuera de pliegue:')
print(pd.cut(auc['AUC fuera de pliegue'], [0, 0.55, 0.6, 0.65, 0.7, 0.8, 1],
             labels=['< 0,55', '0,55-0,60', '0,60-0,65', '0,65-0,70', '0,70-0,80', '> 0,80'])
      .value_counts(sort=False).to_string())

# %%
top = auc.head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(9.5, 7.5))
colores = [ORO if a > 0.8 else VERDE for a in top['AUC fuera de pliegue']]
etiquetas = [f'{NOMBRE.get(v, v)} ({v})' if NOMBRE.get(v, v) != v else v for v in top.index]
barras = ax.barh(etiquetas, top['AUC fuera de pliegue'], color=colores, **BORDE)
etiquetar(ax, barras, '{:.3f}')
ax.scatter(top['AUC en muestra'], range(len(top)), marker='|', s=90, c=GRIS, zorder=3,
           label='AUC en muestra')
ax.axvline(0.5, c=GRIS, lw=0.8)
ax.axvline(0.8, c=ORO, ls='--', lw=0.8)
ax.tick_params(axis='y', labelsize=7)
ax.set(xlim=(0.45, 0.85), xlabel='AUC',
       title='Las 25 variables con mayor AUC univariado (fuera de pliegue temporal)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()

# %% [markdown]
# Nótese que **ninguna variable supera 0,8, ni siquiera 0,75**. La máxima es la antigüedad reconocida
# (0,711) y la mediana de las 106 es 0,530: 60 variables quedan por debajo de 0,55 y solo siete pasan de
# 0,65. Las seis primeras son relojes de la relación laboral (antigüedad, meses en la función y en la
# posición, meses desde el último cambio de contrato) y la edad, más el oficio: todas miden lo mismo,
# cuánto lleva la persona, y la renuncia se concentra en el primer año (capítulo 3). Un AUC de 0,7 para
# esa variable es el esperado en rotación de personal y no es indicio de fuga: se conoce el día 1 sin
# ambigüedad y su señal no cambia de un mes a otro (sección 6.5).
#
# De las variables nuevas, las que más separan son las de jornada e ingreso (meses con extras, ingreso
# frente al pactado, recargo nocturno: menos extras, más riesgo, porque quien acaba de llegar aún no
# las cobra), la historia de cesantías para vivienda, el día de la familia y las vacaciones
# pendientes. El sentido de casi todas es «más bajo, más riesgo», que otra vez es antigüedad: el
# capítulo 4 mide esa colinealidad. La única ausencia con sentido «más alto, más riesgo» entre las 25
# primeras es la licencia no remunerada con rezago de dos meses (0,598).
#
# La tabla completa de las 106 variables, ordenada, está plegada a continuación.

# %% tags=["hide-output"]
auc.round(3)

# %%
por_bloque = auc.groupby('bloque').agg(
    variables=('AUC fuera de pliegue', 'size'),
    AUC_máximo=('AUC fuera de pliegue', 'max'),
    AUC_mediana=('AUC fuera de pliegue', 'median'),
    sobre_0_6=('AUC fuera de pliegue', lambda s: int((s > 0.6).sum())),
    mejor=('AUC fuera de pliegue', 'idxmax'),
).sort_values('AUC_máximo', ascending=False)
brecha = auc['AUC en muestra'] - auc['AUC fuera de pliegue']
print('mayor diferencia entre el AUC en muestra y fuera de pliegue:')
print(brecha.sort_values(ascending=False).head(6).round(3).to_string())
por_bloque.round(3)

# %% [markdown]
# Nótese que el AUC en muestra y el fuera de pliegue, evaluados sobre las mismas filas, difieren en
# 0,053 como máximo (`meses_desde_aumento_merito`, cuya imputación con la mediana mueve a casi la mitad
# de las filas y cambia según el pliegue). Entre las categóricas, las de muchas categorías (área
# funcional, ubicación, sociedad) pierden entre 0,03 y 0,04 fuera de pliegue: parte de
# su señal aparente es la tasa de categorías pequeñas, que no se repite en los meses siguientes. Es la
# razón para agruparlas con `min_frequency` en el capítulo 7.
#
# Por bloque, los atributos, la trayectoria y el contrato tienen las variables más fuertes (todas de
# antigüedad); la jornada y los atributos tienen seis variables sobre 0,6 cada uno. Ningún bloque trae una variable
# aislada muy por encima del resto, que es lo que se vería con una fuga.
#
# ## Señal que se concentra antes de la salida
#
# Una variable con fuga «por trámite» predice bien en el mes de la salida y mucho peor si se usa su
# valor de uno o dos meses antes: su señal no anticipa la renuncia, la acompaña. Para cada numérica se
# compara el AUC de su valor en *t* con el de su valor en *t−1* y *t−2* de la misma persona, sobre las
# mismas filas (las que tienen los dos meses previos en el panel). Se marca una caída de 0,05 o más en
# $|AUC - 0{,}5|$ entre *t* y *t−2*. Es la misma prueba con la que, al construir el panel, se descubrió
# el problema de la licencia no remunerada.

# %%
t_ord = tr.sort_values(['persona_id', 'mes'])
per = pd.PeriodIndex(t_ord.mes, freq='M')
gpid = t_ord.groupby('persona_id')
consec2 = ((per - pd.PeriodIndex(gpid.mes.shift(2).fillna('1900-01'), freq='M')).map(lambda d: d.n) == 2)
sub = t_ord[np.asarray(consec2)]
ys = sub[OBJETIVO].to_numpy()


def fuerza(x):
    x = x.astype(float)
    x = x.fillna(x.median())
    return abs(roc_auc_score(ys, x) - 0.5) + 0.5 if x.nunique() > 1 else 0.5


filas = []
for v in NUM + BIN:
    xt = sub[v]
    x1 = gpid[v].shift(1).loc[sub.index]
    x2 = gpid[v].shift(2).loc[sub.index]
    filas.append({'variable': v, 'bloque': BLOQUE[v], 'AUC en t': fuerza(xt), 'AUC en t-1': fuerza(x1),
                  'AUC en t-2': fuerza(x2)})
caida = pd.DataFrame(filas).set_index('variable')
caida['caída'] = caida['AUC en t'] - caida['AUC en t-2']
print(f'filas con los dos meses previos: {len(sub):,} ({int(ys.sum())} renuncias)')
print(f'variables con caída >= 0,05: {int((caida["caída"] >= 0.05).sum())} de {len(caida)}')
caida.sort_values('caída', ascending=False).head(10).round(3)

# %% [markdown]
# Nótese que, con 56.566 filas y 528 renuncias que tienen los dos meses previos, **solo una de las 88
# numéricas y binarias cae 0,05 o más**: `turnos_1m`, los turnos marcados en el mes anterior (0,595 en *t*, 0,533
# en *t−1*, 0,531 en *t−2*). No es fuga en sentido estricto: las marcaciones de *t−1* están completas
# el día 1 de *t*, porque las registra el biométrico, no una persona. Lo que dice es que quien va a
# renunciar marca menos turnos el mes antes (posiblemente vacaciones o permisos antes de irse); las
# variables de marcaciones de tres meses caen 0,02 o menos. Queda en observación.
#
# Las dos licencias no remuneradas `_r2` caen 0,044 y 0,046, por debajo del umbral. El rezago de dos
# meses deja fuera el mes del trámite (en la versión de un mes la caída era de 0,058 y 0,053), pero la
# licencia sigue anticipando la salida con uno o dos meses: esa parte es señal disponible el día 1, no
# fuga.
#
# ## Datos faltantes que delatan la salida
#
# La otra forma habitual de fuga es el faltante que aparece *porque* la persona se va: el registro se
# cierra o se vacía en el mes de la salida. Entre quienes renuncian y tienen fila tres meses antes, se
# compara el porcentaje de faltantes en el mes de la renuncia con el de tres meses antes. Un salto de
# cinco puntos o más sería una alerta. Se muestra también la tasa de renuncia con y sin dato, que no
# es fuga pero dice si el faltante es informativo (y justifica el indicador de faltante del capítulo 7).

# %%
ren = t_ord[t_ord[OBJETIVO].eq(1)]
per_r = pd.PeriodIndex(ren.mes, freq='M')
filas = []
for v in PREDICTORAS:
    if tr[v].isna().sum() == 0:
        continue
    atras = gpid[v].shift(3).loc[ren.index]
    mes_atras = gpid.mes.shift(3).loc[ren.index]
    ok = np.asarray((per_r - pd.PeriodIndex(mes_atras.fillna('1900-01'), freq='M')).map(lambda d: d.n) == 3)
    nul = tr[v].isna()
    filas.append({'variable': v, 'bloque': BLOQUE[v], 'nulos %': 100 * nul.mean(),
                  'sin dato al renunciar %': 100 * ren[v][ok].isna().mean(),
                  'sin dato 3 meses antes %': 100 * atras[ok].isna().mean(),
                  'renuncia sin dato %': 100 * tr.loc[nul, OBJETIVO].mean(),
                  'renuncia con dato %': 100 * tr.loc[~nul, OBJETIVO].mean()})
falt = pd.DataFrame(filas).set_index('variable')
falt['salto'] = falt['sin dato al renunciar %'] - falt['sin dato 3 meses antes %']
print(f'variables con faltantes: {len(falt)} | con salto >= 5 puntos: {int((falt.salto >= 5).sum())} | '
      f'salto máximo: {falt.salto.max():.1f} puntos')
print(f'renuncias con fila tres meses antes: {int(ok.sum())} de {len(ren)}')
falt['razón'] = falt['renuncia sin dato %'] / falt['renuncia con dato %']
cols_salto = ['bloque', 'nulos %', 'sin dato al renunciar %', 'sin dato 3 meses antes %', 'salto']
falt[falt['nulos %'] >= 1].sort_values('salto', ascending=False)[cols_salto].head(8).round(2)

# %%
# regla de privacidad del libro: el grupo sin dato necesita 300 filas y 5 renuncias para mostrarse
falt['filas sin dato'] = [int(tr[v].isna().sum()) for v in falt.index]
falt['renuncias sin dato'] = [int(tr.loc[tr[v].isna(), OBJETIVO].sum()) for v in falt.index]
visible = (falt['filas sin dato'] >= 300) & (falt['renuncias sin dato'] >= 5)
print(f'variables con menos de 300 filas o 5 renuncias sin dato (no se muestran): '
      f'{int((~visible).sum())}, con {int(falt.loc[~visible, "filas sin dato"].max())} filas sin dato como máximo')
informativo = falt[visible & ((falt.razón >= 1.5) | (falt.razón <= 1 / 1.5))].sort_values('razón', ascending=False)
print(f'variables con faltante informativo (razón de tasas >= 1,5 o <= 1/1,5): {len(informativo)} '
      f'de {int(visible.sum())}')
# muchas comparten el mismo patrón de faltante (la misma fuente ausente): se agrupan
informativo = informativo.round({'nulos %': 1, 'renuncia sin dato %': 2, 'renuncia con dato %': 2, 'razón': 2})
patrones = (informativo.reset_index()
            .groupby(['nulos %', 'renuncia sin dato %', 'renuncia con dato %', 'razón'], as_index=False)
            .agg(variables=('variable', 'size'), bloques=('bloque', lambda b: ', '.join(sorted(set(b)))),
                 ejemplo=('variable', 'first'))
            .sort_values('razón', ascending=False).set_index('ejemplo'))
patrones

# %% [markdown]
# Nótese que **ningún faltante aparece al salir**: el salto máximo entre el mes de la renuncia y tres
# meses antes es de 0,2 puntos, lejos de los 5 de la alerta. Así se comprueba que las correcciones del
# cargo y del tipo de personal (sección 6.3) siguen funcionando con el panel nuevo.
#
# De las 63 variables con faltantes, 4 (`nivel`, `tipo_unidad`, `area_funcional` y `estado_civil`)
# tienen 18 filas sin dato o menos y no se prueban ni se muestran. De las 59 restantes, el faltante es
# **informativo** en 54 (la tasa de renuncia sin dato es al menos 1,5 veces la tasa con dato, o a lo
# sumo dos tercios de ella) y no lo es en 5. Los patrones tienen explicación estructural:
#
# - *Sin nómina previa* (2,4 % de las filas: 20 variables de ingreso y jornada, y el cargo del mes
#   anterior): es el primer mes del episodio, con 1,66 % de renuncia frente a 1,00 %.
# - *Sin sueldo relativo* (34-39 %): sueldos que no son básicos o grupos de menos de cinco personas;
#   1,4 % frente a 0,8 %.
# - *Sin vacaciones pagadas desde 2024* (34,6 %) y *sin aumento de sueldo registrado* (44,9 %): sobre todo
#   personas nuevas; entre 2,6 y 3 veces la tasa.
# - *Sin marcación biométrica* (43-46 %) y *sin vencimiento* (47 %, contratos indefinidos): menos
#   riesgo, 0,63 % y 0,47 % frente a 1,33 % y 1,51 %.
#
# Imputar solo con la mediana borraría esa información. Por eso el capítulo 7 imputa **con indicador
# de faltante**.
#
# ## Duplicados y entidades entre entrenamiento y test

# %%
print('filas persona-mes duplicadas en el panel:', int(p.duplicated(['persona_id', 'mes']).sum()))
print('pares (persona, mes) en entrenamiento y test a la vez:',
      len(set(zip(tr.persona_id, tr.mes)) & set(zip(te.persona_id, te.mes))))
per_tr, per_te = set(tr.persona_id), set(te.persona_id)
print(f'personas: entrenamiento {len(per_tr):,} | test {len(per_te):,} | en ambos {len(per_tr & per_te):,} '
      f'| solo en test {len(per_te - per_tr):,}')
print(f'filas de test de personas ya vistas en entrenamiento: {100 * te.persona_id.isin(per_tr).mean():.1f} %')

vec_tr = tr[PREDICTORAS].astype(str).apply(tuple, axis=1)
vec_te = te[PREDICTORAS].astype(str).apply(tuple, axis=1)
print(f'filas de entrenamiento con un vector de {len(PREDICTORAS)} predictoras repetido dentro del entrenamiento: '
      f'{int(vec_tr.duplicated(keep=False).sum()):,}')
igual = vec_te.isin(set(vec_tr)).to_numpy()
print(f'filas de test con un vector idéntico en entrenamiento: {int(igual.sum()):,} de {len(te):,}')
base18 = [c for c in PREDICTORAS if BLOQUE[c] == 'atributos']
igual18 = te[base18].astype(str).apply(tuple, axis=1).isin(set(tr[base18].astype(str).apply(tuple, axis=1)))
print(f'  con solo las 18 de atributos: {int(igual18.sum()):,} ({100 * igual18.mean():.1f} %)')

# %% [markdown]
# Nótese que no hay filas persona-mes repetidas ni pares (persona, mes) en los dos lados: el corte es
# limpio. En cambio, **las personas se repiten**, como corresponde a un panel: 4.373 de las 4.611
# personas del test ya estaban en el entrenamiento, y el 96,8 % de las filas del test son suyas. No es
# fuga, porque la etiqueta del test es de meses posteriores y ninguna predictora usa la etiqueta de la
# persona (no hay «renunció antes» ni tasas por persona). Sí significa que el test mide si el modelo
# predice el futuro de la misma plantilla, que es el uso real, y no si generaliza a personas nuevas;
# para eso el capítulo 2 dejó la validación agrupada por persona como complemento.
#
# Con las 105 predictoras, solo 53 filas del test (0,3 %) repiten un vector del entrenamiento. Con las
# 18 de atributos eran 1.785 (10,1 %): la historia laboral distingue a personas que antes tenían el
# mismo perfil, como los operarios de una misma finca. Dentro del entrenamiento hay 301 filas con un
# vector repetido.
#
# ## Variables descartadas o en observación
#
# El panel de construcción tenía 262 columnas; el libro usa 105 predictoras. La diferencia se explica
# por identificadores, columnas del desenlace, versiones redundantes, controles de cobertura y, sobre
# todo, por **categorías completas excluidas por privacidad**: salud e incapacidades, riesgo
# psicosocial, clima individual, sindicato, hijos y familia, quejas, desempeño, sanciones, deudas y
# descuentos, y sueldo o devengado absolutos. Son datos sensibles (Ley 1581 de 2012) o de zona gris, y
# el proyecto no los usa.
#
# | variable | decisión | justificación |
# |---|---|---|
# | `persona_id`, `mes` | fuera del modelo | identificadores; `persona_id` solo agrupa y `mes` se reemplaza por la indicadora de enero |
# | `evento`, `tipo_retiro`, `y_renuncia` | fuera del modelo | desenlace: se conocen al cerrar el mes *t* (sección 6.2) |
# | `contrato_fijo` | descartada (capítulo 1) | idéntica a `contrato`: duplicaría la misma información |
# | `horas_bajo_legal` | descartada (capítulo 1) | vale 0 en el 99,98 % de las filas |
# | `horas_diarias_teoricas`, `plan_horario` | descartadas (capítulo 5) | siguen el calendario de la reducción legal de la jornada (46 a 44 horas en julio de 2025, 44 a 42 en julio de 2026, dentro del test), no a la persona; en el test toman valores que no existen en el entrenamiento |
# | `cambios_plan_12m` | descartada (capítulo 1) | sigue el calendario de la reducción legal de la jornada (cerca del 80 % de las filas marca cambio y cae a 20 % en julio de 2025), no a la persona |
# | `preaviso_no_renovacion` | descartada, solo sensibilidad | con preaviso el objetivo es 0 por construcción: 469 filas, 0 renuncias |
# | días de licencia no remunerada con rezago de un mes | descartadas | la señal decae de 0,643 a 0,585 al retroceder dos meses: trámite de la salida; entran las `_r2` |
# | textos de cargo, función, departamento y área | quitados del panel | redundantes con los ejes derivados y cuasi-identificadores |
# | otras marcas de la salida (reclasificación, bonificación de salida) | fuera del panel | describen la salida |
# | meses del episodio actual | fuera del panel | Spearman de 0,998 con `antig_meses` |
# | horas semanales teóricas, horas extra en salarios mínimos, horas por encima de la teórica, montos absolutos de bonificación | fuera del panel | casi duplicados (Spearman ≥ 0,98) de `horas_diarias_teoricas`, `horas_extra_*`, `horas_turno_3m` y `meses_con_bpr_12m` |
# | meses con nómina en la ventana, clase de personal, retiros de cesantías por otros motivos | fuera del panel | cobertura de la fuente o sin variación |
# | `turnos_1m` | en observación | única con caída ≥ 0,05 (0,595 → 0,531); disponible el día 1, pero su señal es del mes previo a la salida |
# | `dias_licencia_no_remunerada_3m_r2`, `_12m_r2` | en observación | caída de 0,044 y 0,046, bajo el umbral; el 6,2 % de las licencias se registra después del mes siguiente |
# | cinco ausencias con rezago de un mes | en observación | el mes *t−1* puede completarse después del día 1: solo entre el 63 % y el 90 % se registra en el mismo mes (sección 6.1) |
# | cuatro variables de vacaciones | en observación | saldo reconstruido y compensación fechada por aproximación |
# | `nacido_en_depto_sede` | en observación | departamento de la sede estimado con perfiles actuales; 24 % de nulos |
# | `ubicacion`, `sociedad`, `area_funcional` | en observación | pierden de 0,03 a 0,04 de AUC fuera de pliegue: categorías pequeñas; se agrupan con `min_frequency` |
# | `genero`, `nivel` | en observación | AUC fuera de pliegue de 0,484: sin señal propia; la regularización decide |
#
# «En observación» no quiere decir fuera: las variables entran al modelo del capítulo 7 y se vigila
# que no dominen los coeficientes.
#
# ## Resumen ejecutivo del EDA
#
# La tabla final sigue la convención del curso: calidad de los datos, variables prometedoras,
# problemas detectados y decisiones que pasan al modelo. Para que se sostenga sola, la celda siguiente
# recalcula con el entrenamiento los pocos indicadores de calidad que la tabla cita; el detalle está
# en los capítulos 3 a 5.

# %%
continuas = [v for v in NUM if tr[v].nunique() > 2]
# para las correlaciones se usan las numéricas y binarias, como el capítulo 4
num_bin = NUM + BIN
asim = tr[continuas].skew()
rho = tr[num_bin].corr(method='spearman').abs()
pares = [(a, b, rho.loc[a, b]) for i, a in enumerate(num_bin) for b in num_bin[i + 1:] if rho.loc[a, b] > 0.7]
# el nulo cuenta como una categoría más, como en el capítulo 3
raras = {v: int((tr.groupby(tr[v].fillna('(nulo)'), observed=True)[OBJETIVO].sum() < 5).sum()) for v in CAT}
print(f'predictoras con faltantes: {int((tr[PREDICTORAS].isna().sum() > 0).sum())} de {len(PREDICTORAS)} | '
      f'celdas faltantes: {100 * tr[PREDICTORAS].isna().to_numpy().mean():.1f} %')
print(f'numéricas continuas (más de dos valores): {len(continuas)} | con |asimetría| > 2: {int((asim.abs() > 2).sum())}')
print(f'pares de numéricas y binarias con |Spearman| > 0,7: {len(pares)}')
print(f'categóricas: {len(CAT)} | categorías (el nulo cuenta como una) con menos de 5 renuncias: {sum(raras.values())} '
      f'(en {sum(1 for r in raras.values() if r)} variables)')

# %% [markdown]
# | aspecto | hallazgo | de dónde sale | decisión que pasa al modelo |
# |---|---|---|---|
# | calidad: faltantes | 63 de 105 predictoras tienen faltantes (13,7 % de las celdas); ninguno aparece al salir (salto máximo de 0,2 puntos) y 54 son informativos | capítulos 1 y 3, sección 6.6 | imputación con la mediana **más indicador de faltante** |
# | calidad: colas largas | 44 de las 73 numéricas continuas tienen asimetría mayor que 2 en valor absoluto (horas extra, recargos, bonos, días de licencia) | capítulo 3 y celda anterior | **recorte p1-p99** y **log con signo** en las de cola larga, ajustados solo con el entrenamiento |
# | calidad: categorías raras | 64 categorías con menos de 5 renuncias (el nulo cuenta como una categoría), en 11 de las 17 categóricas | capítulo 3 y celda anterior | **one-hot con `min_frequency`** (las raras se agrupan) |
# | calidad: duplicados | ninguna fila persona-mes repetida ni compartida entre particiones; el 96,8 % de las filas de test son de personas vistas | capítulo 2 y sección 6.7 | validación agrupada por persona como complemento |
# | variables prometedoras | antigüedad (AUC 0,711), meses en la función y en la posición, contrato, edad, oficio; de la historia: extras, ingreso frente al pactado, licencia no remunerada `_r2`, vacaciones pendientes | capítulo 3 y sección 6.4 | antigüedad en logaritmo o por tramos (capítulo 3) |
# | problema: colinealidad | 50 pares de numéricas y binarias con Spearman mayor que 0,7 en valor absoluto; buena parte de la historia es otra medida de la antigüedad | capítulo 4 (VIF, PCA) y celda anterior | **regularización** (L1 frente a L2, rejilla de C hasta 1e-4) en vez de eliminar a mano |
# | problema: tiempo | la tasa oscila alrededor de 1,02 % con variación entre meses por encima del azar (p = 0,002) y baja de 1,24 % a 0,96 % entre enero-abril de 2025 y de 2026; la deriva grande viene de la reducción legal de la jornada, el ciclo del salario y cambios de cobertura; la cobertura de `tamano_equipo_jefe` cambia en 2025 | capítulo 5 | **validación temporal con gap** de un mes (`pliegues_temporales`); el test, una sola vez al final; horario teórico y plan horario fuera; revisar la calibración por mes |
# | problema: calendario | solo la indicadora de enero mejora a la constante fuera de muestra (8 de 8 meses); el mes del año y el seno-coseno no | capítulo 5 | entra **solo la indicadora de enero** |
# | problema: fuga | ninguna AUC sobre 0,8; `preaviso_no_renovacion` y las ausencias sin rezago descartadas; `turnos_1m` en observación | secciones 6.2 a 6.5 | se modela con las 105 y la indicadora de enero |
#
# Las filas que citan los capítulos 3 a 5 remiten a su detalle; las cifras de esta tabla se calculan
# en este capítulo con el entrenamiento.
#
# ## Síntesis
#
# - Las 105 predictoras se conocen el día 1 del mes que se predice: 93 sin reserva y doce con una
#   reserva documentada (las siete ausencias, por registro tardío; cuatro de vacaciones reconstruidas;
#   el departamento de la sede).
# - `preaviso_no_renovacion` (0 renuncias en 469 filas, por construcción) y las licencias no
#   remuneradas con rezago de un solo mes se descartan, igual que `contrato_fijo`, `horas_bajo_legal`
#   y `cambios_plan_12m` (capítulo 1) y `horas_diarias_teoricas` y `plan_horario` (capítulo 5);
#   `evento` y `tipo_retiro` son el desenlace.
# - Ninguna variable predice sola más de lo plausible: AUC máximo de 0,711 (antigüedad), ninguno sobre
#   0,8, y la diferencia con el AUC en muestra no pasa de 0,053.
# - Solo `turnos_1m` concentra su señal justo antes de la salida, y ningún faltante aparece al salir.
# - No hay filas compartidas entre entrenamiento y test; las personas sí se repiten, como en todo
#   panel, y ninguna predictora usa la etiqueta de la persona.
