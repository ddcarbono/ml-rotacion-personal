# %% [markdown]
# # EDA multivariado
#
# ```{admonition} Alcance
# :class: tip
# Este capítulo usa solo el entrenamiento (67.416 persona-mes, 688 renuncias) y la semilla 2026 en
# todo lo aleatorio. Toda transformación (mediana, percentiles de recorte, media y desviación) se
# estima con el entrenamiento; el test no se toca hasta el capítulo 7.
# ```
#
# El capítulo anterior miró cada variable frente al objetivo. Aquí se miran las variables entre sí,
# con cuatro preguntas que condicionan el modelo: qué variables repiten la misma información
# (correlación, V de Cramér y VIF), cuántas dimensiones tiene de verdad la historia laboral (PCA), si
# hay filas raras que renuncien distinto (atípicos multivariados) y si hay subpoblaciones naturales
# (k-medias). Con 85 numéricas repartidas en 8 bloques más los atributos, el análisis se organiza por
# bloque y se resume en tablas; los gráficos se reservan para lo que no cabe en una tabla.

# %%
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')   # otros procesos usan la CPU

import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, roc_auc_score)

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from comun import (cargar, particion, estilo, tasa_ic, v_cramer, OBJETIVO, NUM, BIN, CAT, BLOQUES,
                   BLOQUE, NOMBRE, SEMILLA, VERDE, ORO, TINTA, GRIS, GRIS_CL, BORDE)

estilo()
pd.set_option('display.width', 200)

tr, _ = particion(cargar())
y = tr[OBJETIVO].to_numpy()
rng = np.random.default_rng(SEMILLA)


def nom(c):
    return NOMBRE.get(c, c)


# numéricas ordenadas por bloque: primero los atributos (edad, antigüedad y las 3 binarias)
ORDEN_BLOQUES = ['atributos'] + list(BLOQUES)
NUMS = sorted(NUM + BIN, key=lambda c: (ORDEN_BLOQUES.index(BLOQUE[c]), (NUM + BIN).index(c)))
print(f'entrenamiento: {len(tr):,} filas | {int(y.sum())} renuncias | {len(NUMS)} numéricas y binarias '
      f'| {len(CAT)} categóricas')
print(pd.Series([BLOQUE[c] for c in NUMS]).value_counts().reindex(ORDEN_BLOQUES).to_string())

# %% [markdown]
# Son 88 columnas: las 85 numéricas (incluidas las binarias 0/1 de la historia) más las 3 binarias de
# atributos (primer mes, reingreso, traslado). Las 5 variables que el capítulo 1 dejó fuera (entre ellas
# el contrato fijo, idéntico al tipo de contrato, y las que siguen el calendario de la reducción legal
# de la jornada) no entran. Jornada es el bloque más grande (25), porque cada
# concepto de tiempo (horas extra, recargo nocturno, dominicales, sábados) viene en tres ventanas.
#
# ## Correlación entre numéricas (Spearman)
#
# Se usa Spearman porque la mayoría de estas variables son conteos con muchos ceros y colas largas
# (capítulo 3): la correlación de rangos no depende de la escala ni de unos pocos valores extremos. Los
# faltantes se excluyen par a par, de modo que una correlación con una variable de muchos faltantes (por
# ejemplo, meses desde que fue aprendiz, 95 % nulo) describe solo a las filas que la tienen.

# %%
R = tr[NUMS].corr(method='spearman')
cortes = np.cumsum([sum(BLOQUE[c] == b for c in NUMS) for b in ORDEN_BLOQUES])[:-1]

fig, ax = plt.subplots(figsize=(17, 15))
sns.heatmap(R, cmap='oro_verde', center=0, vmin=-1, vmax=1, ax=ax, square=True,
            xticklabels=[nom(c) for c in NUMS], yticklabels=[nom(c) for c in NUMS],
            cbar_kws={'shrink': 0.5, 'label': 'Spearman'})
for c in cortes:
    ax.axhline(c, color=TINTA, lw=0.8)
    ax.axvline(c, color=TINTA, lw=0.8)
ax.tick_params(labelsize=5.5)
posiciones = np.r_[0, cortes, len(NUMS)]
for b, a, z in zip(ORDEN_BLOQUES, posiciones[:-1], posiciones[1:]):
    ax.text(len(NUMS) + 1, (a + z) / 2, b.replace('_', '\n'), va='center', fontsize=8, color=VERDE, fontweight='bold')
ax.set(title=f'Correlación de Spearman entre las {len(NUMS)} numéricas, ordenadas por bloque')
plt.tight_layout()
plt.show()

# %%
# resumen por bloque: la mayor |r| dentro de cada bloque y entre cada par de bloques
A = R.abs().where(~np.eye(len(R), dtype=bool))
por_bloque = A.groupby(lambda c: BLOQUE[c]).max().T.groupby(lambda c: BLOQUE[c]).max()
por_bloque = por_bloque.loc[ORDEN_BLOQUES, ORDEN_BLOQUES]

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(por_bloque, annot=True, fmt='.2f', cmap='verde', vmin=0, vmax=1, ax=ax,
            annot_kws={'size': 8}, cbar_kws={'label': 'máx |Spearman|'})
ax.set(title='Mayor |Spearman| dentro de cada bloque y entre bloques', xlabel='', ylabel='')
plt.tight_layout()
plt.show()

# %%
i, j = np.triu_indices(len(R), 1)
pares = pd.DataFrame({'variable_1': R.index[i], 'variable_2': R.columns[j], 'spearman': R.values[i, j]})
pares['bloque_1'] = pares.variable_1.map(BLOQUE)
pares['bloque_2'] = pares.variable_2.map(BLOQUE)
pares['|r|'] = pares.spearman.abs()
fuertes = pares[pares['|r|'] > 0.7].sort_values('|r|', ascending=False)
print(f'pares con |r| > 0,7: {len(fuertes)} de {len(pares):,} | con |r| > 0,95: '
      f'{(fuertes["|r|"] > 0.95).sum()} | entre bloques distintos: '
      f'{(fuertes.bloque_1 != fuertes.bloque_2).sum()}')
print(f'variables involucradas: {len(set(fuertes.variable_1) | set(fuertes.variable_2))}')
fuertes[['variable_1', 'variable_2', 'bloque_1', 'bloque_2', 'spearman']].round(3).reset_index(drop=True)

# %% [markdown]
# Nótese que la redundancia está concentrada y tiene dos orígenes claros:
#
# 1. **Ventanas de un mismo concepto.** De los 50 pares con |r| > 0,7 (el umbral del curso), la mayoría
#    son el mismo concepto medido en 1, 3 y 12 meses: recargo nocturno (0,96 entre 1 y 3 meses; 0,96
#    entre 3 y 12), dominicales (0,89-0,91), horas extra (0,80-0,90), horas por turno (0,94) y sábados
#    (0,78). También son casi duplicados por construcción días y periodos de vacaciones pendientes
#    (0,97), estar en la mediana local y la distancia a ella (-0,97), meses con extras y su porcentaje
#    (0,95), y días y número de licencias no remuneradas (0,93). Los 4 pares por encima de 0,95 son
#    los de la mediana local, las vacaciones pendientes y dos de las ventanas de recargo nocturno.
# 2. **El reloj de la persona.** La antigüedad reconocida se mueve con todo lo que cuenta meses desde
#    algo: meses desde el cambio de contrato (0,88), en la función (0,85), en la posición (0,83), desde
#    que fue aprendiz (0,89) y el histórico de retiros de cesantías para vivienda (0,74). Estos
#    explican casi todos los 10 pares que cruzan bloques. El primer mes y el ingreso a mitad de mes (0,94) son la misma marca de ingreso.
#
# El resumen por bloque confirma que, fuera de ese reloj, **los bloques están poco correlacionados
# entre sí**: ningún par entre salario relativo, jornada, ingreso relativo, ausencias y vacaciones
# supera 0,7 (el mayor es 0,68, entre ingreso relativo y ausencias, por los meses con ausencia pagada).
# La historia no es una copia de los atributos, sino información nueva. Dentro de cada
# bloque la redundancia es mayor en jornada (las ventanas); en ingreso relativo ningún par pasa de 0,4.
# Las celdas en blanco del mapa grande son pares sin correlación definida: por ejemplo, el primer mes
# con las variables de nómina, que están vacías justo en el primer mes (llegan con un mes de rezago), de
# modo que en las filas donde existen el primer mes vale siempre 0.
#
# ## Asociación entre categóricas (V de Cramér)

# %%
variables = CAT + [OBJETIVO]
V = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
for a_i, a in enumerate(variables):
    for b in variables[a_i + 1:]:
        V.loc[a, b] = V.loc[b, a] = v_cramer(tr[a], tr[b])

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(V, annot=True, fmt='.2f', cmap='verde', vmin=0, vmax=1, ax=ax, annot_kws={'size': 7},
            xticklabels=[nom(c) for c in variables], yticklabels=[nom(c) for c in variables])
ax.set(title='V de Cramér entre variables categóricas')
plt.tight_layout()
plt.show()

k_i, k_j = np.triu_indices(len(CAT), 1)
pv = pd.DataFrame({'variable_1': np.array(CAT)[k_i], 'variable_2': np.array(CAT)[k_j],
                   'V': V.loc[CAT, CAT].values[k_i, k_j]})
print('pares de categóricas con V > 0,7:')
print(pv[pv.V > 0.7].sort_values('V', ascending=False).round(3).to_string(index=False))
print('\nV con el objetivo (las 5 mayores):')
print(V[OBJETIVO].drop(OBJETIVO).sort_values(ascending=False).head(5).round(3).to_string())

# %%
# anidamiento exacto: indicadoras de sociedad y de línea juntas
D = add_constant(pd.get_dummies(tr[['sociedad', 'linea']], drop_first=True, dtype=float))
print(f'sociedad + línea: {D.shape[1]} columnas, rango {np.linalg.matrix_rank(D.to_numpy())}')
D = add_constant(pd.get_dummies(tr[['oficio', 'familia_cargo']], drop_first=True, dtype=float))
print(f'oficio + familia de cargo: {D.shape[1]} columnas, rango {np.linalg.matrix_rank(D.to_numpy())}')

# %% [markdown]
# Los pares con V próxima a 1 son anidamientos por construcción: cada sociedad pertenece a una sola
# línea (V = 1,00) y cada oficio a una sola familia de cargo (0,995). Las dos comprobaciones de rango lo
# hacen exacto: las indicadoras de sociedad y línea suman 27 columnas pero solo 22 son independientes,
# y las de oficio y familia, 57 columnas con rango 44. En esas matrices el VIF de las indicadoras de la
# variable gruesa es infinito y los coeficientes de una logística sin penalización no están
# identificados. Le siguen anidamientos parciales (ubicación en línea, 0,88; oficio y ubicación en tipo
# de unidad, 0,76; oficio en área funcional y en la situación del control de tiempos, 0,72 y 0,70) que dicen
# lo mismo: la organización se describe con varias variables que se
# contienen unas a otras. Ninguna categórica tiene una asociación fuerte con el objetivo: la mayor es la
# del oficio (V = 0,075), como se espera con una clase del 1 %.
#
# ## Preparación de las numéricas
#
# El VIF, el PCA, los atípicos y los conglomerados necesitan una matriz sin faltantes y en una escala
# común. Se usa la misma preparación del modelo (capítulos 3 y 7): con los valores observados, recortar
# cada numérica en sus percentiles 1 y 99 y aplicar un logaritmo con signo,
# $\operatorname{sign}(x)\log(1 + |x|)$, a las que conservan una asimetría mayor que 2 (cola larga);
# luego imputar la mediana y estandarizar. Sin el recorte y el log, unas pocas filas con
# valores extremos dominarían la covarianza, y con ella el PCA, la distancia de Mahalanobis y
# k-medias.

# %%
def preparar(df, cols):
    """El mismo orden y criterio que el Recorte del capítulo 7: con los valores observados, recorta en
    p1-p99 y aplica log con signo si |asimetría| > 2 después del recorte; luego imputa la mediana. Todo
    se estima con el entrenamiento (este capítulo no toca el test). Las binarias solo se imputan."""
    X = df[cols].astype(float).copy()
    recortadas, con_log = [], []
    for c in cols:
        x = X[c]
        if x.dropna().nunique() > 2:
            bajo, alto = x.quantile([0.01, 0.99])
            if alto > bajo:
                x = x.clip(bajo, alto)
                recortadas.append(c)
            obs = x.dropna()
            if abs((((obs - obs.mean()) / obs.std(ddof=0)) ** 3).mean()) > 2:
                x = np.sign(x) * np.log1p(x.abs())
                con_log.append(c)
        X[c] = x.fillna(x.median())
    constantes = [c for c in cols if X[c].std() == 0]
    return X.drop(columns=constantes), recortadas, con_log, constantes


Xn, recortadas, con_log, constantes = preparar(tr, NUMS)
COLS = list(Xn.columns)
Zs = ((Xn - Xn.mean()) / Xn.std()).to_numpy()
print(f'recortadas en p1-p99: {len(recortadas)} | con log con signo (|asimetría| > 2 tras el recorte): '
      f'{len(con_log)} | constantes tras imputar (fuera): {constantes}')
print(f'matriz numérica: {Zs.shape[0]:,} x {Zs.shape[1]}')

# variables que comparten exactamente el patrón de faltantes
faltan = tr[NUMS].isna()
faltan = faltan.loc[:, faltan.any()]
patron = faltan.T.apply(lambda s: hash(tuple(s)), axis=1)
grupos_nulos = faltan.columns.groupby(patron)
print(f'\n{faltan.shape[1]} variables con faltantes forman {len(grupos_nulos)} patrones distintos; '
      'los que comparten 3 o más variables:')
for _, cols in sorted(grupos_nulos.items(), key=lambda kv: -len(kv[1])):
    if len(cols) >= 3:
        print(f'  {100 * faltan[cols[0]].mean():.1f} % faltante | {len(cols)} variables | '
              f'bloques {sorted(set(BLOQUE[c] for c in cols))}')

# %% [markdown]
# Se recortaron 72 de las 88 columnas (las demás son binarias o tienen el percentil 1 igual al 99) y 39
# recibieron el log con signo; ninguna quedó constante. Nótese la estructura de los faltantes: las 54
# variables con nulos forman solo 22 patrones, y los grandes son bloques enteros que faltan juntos. Las
# 5 y 3 variables que se calculan con las marcas biométricas faltan en el 44 % y el 43 % de las filas,
# las 3 de la mediana local en el 34 %, y 20 variables de jornada e ingreso que salen de la nómina
# faltan juntas en el mismo 2,4 % de las filas. Este último es el primer mes de cada episodio: la
# nómina y las marcaciones de la fila del mes t llegan hasta t-1, y en el primer mes no hay mes anterior
# que medir. Una consecuencia para los indicadores de faltante: como las 54 variables forman solo 22
# patrones, sus indicadores están muy correlacionados (dentro de un patrón son idénticos). No es un
# problema para una logística regularizada, que absorbe esa redundancia como la de las ventanas; el
# modelo del capítulo 7 usa un indicador por variable por simplicidad del Pipeline. El faltante es en sí
# una característica del puesto (marcar o no en biométrico). La imputación con la mediana crea además un
# pico artificial en esas columnas, que se verá en el PCA y en los atípicos.
#
# ## Factor de inflación de la varianza (VIF)
#
# El VIF de una columna mide cuánto se infla la varianza de su coeficiente porque las demás la
# explican: $\text{VIF}_j = 1 / (1 - R_j^2)$, con $R_j^2$ el de la regresión de la columna $j$ contra
# todas las demás y una constante (`add_constant`; sin constante el $R^2$ no está centrado y el VIF sale
# mal). Se lee con los umbrales del curso: 1 es ausencia de colinealidad, de 1 a 5 es baja, de 5 a 10
# moderada y más de 10 alta. Se calcula sobre las 88 numéricas ya imputadas, recortadas y estandarizadas
# del entrenamiento, que es la matriz que verá el modelo.

# %%
Xv = add_constant(pd.DataFrame(Zs, columns=COLS))
vif = pd.Series([variance_inflation_factor(Xv.values, k) for k in range(1, Xv.shape[1])], index=COLS)
clase = pd.cut(vif.rename('VIF'), [0, 5, 10, np.inf], labels=['1-5 (baja)', '5-10 (moderada)', '> 10 (alta)'],
               right=False)
tabla = pd.crosstab(pd.Series([BLOQUE[c] for c in COLS], index=COLS, name='bloque'), clase)
tabla = tabla.reindex(ORDEN_BLOQUES)
tabla.loc['total'] = tabla.sum()
print(f'VIF mediano {vif.median():.1f} | máximo {vif.replace(np.inf, np.nan).max():,.0f} | '
      f'infinitos: {np.isinf(vif).sum()}')
tabla

# %%
# las de VIF > 5 con su pareja más correlacionada (Spearman)
socio = R.loc[COLS, COLS].abs().where(~np.eye(len(COLS), dtype=bool))
altos = vif[vif >= 5].sort_values(ascending=False)
pd.DataFrame({'bloque': [BLOQUE[c] for c in altos.index], 'VIF': altos.round(1),
              'más correlacionada con': [socio[c].idxmax() for c in altos.index],
              '|Spearman|': [socio[c].max().round(2) for c in altos.index]})

# %%
# un representante por concepto: qué pasa con los VIF si se deja una sola ventana (la de 3 meses)
ventanas = [c for c in COLS if c.endswith('_1m') or c.endswith('_12m')]
duplicados = [c for c in ventanas if c.replace('_1m', '_3m').replace('_12m', '_3m') in COLS
              and c.replace('_1m', '_3m').replace('_12m', '_3m') != c]
resto = [c for c in COLS if c not in duplicados]
Xr = add_constant(pd.DataFrame(Zs, columns=COLS)[resto])
vif_r = pd.Series([variance_inflation_factor(Xr.values, k) for k in range(1, Xr.shape[1])], index=resto)
print(f'sin las ventanas de 1 y 12 meses que tienen su par de 3 meses ({len(duplicados)} variables):')
print(f'  VIF > 10: {(vif_r > 10).sum()} (antes {(vif > 10).sum()}) | VIF 5-10: '
      f'{((vif_r >= 5) & (vif_r < 10)).sum()} (antes {((vif >= 5) & (vif < 10)).sum()}) | '
      f'máximo {vif_r.max():.1f}')
print(vif_r[vif_r >= 5].sort_values(ascending=False).round(1).to_string())

# %% [markdown]
# Nótese que la colinealidad es real pero acotada: la mediana del VIF es 2,8, 61 de las 88 columnas
# quedan por debajo de 5, 14 entre 5 y 10 y 13 por encima de 10, y el máximo es 35 (ninguno infinito,
# porque entre numéricas no hay dependencias exactas). Los VIF altos son justo los casi duplicados de la
# correlación: las ventanas de recargo nocturno, dominicales y horas extra, meses con extras y su
# porcentaje, días y periodos de vacaciones pendientes (34), y el reloj de la persona (antigüedad
# reconocida 35, meses desde el cambio de contrato 33).
#
# El ejercicio de dejar una sola ventana por concepto (la de 3 meses) muestra que las ventanas explican
# cerca de la mitad del problema: al quitar esas 9 columnas, los VIF > 10 bajan de 13 a 7 y los de 5 a 10
# de 14 a 7, pero el máximo apenas se mueve (de 35 a 34,9). Lo que queda alto ya no son ventanas sino el reloj (antigüedad reconocida y
# meses desde el cambio de contrato), los
# pares construidos (vacaciones, meses con extras) y la desigualdad salarial del oficio y nivel (Gini y
# privación relativa, 11).
#
# ```{admonition} Qué hacer con los VIF altos en una logística regularizada
# :class: note
# El VIF mide la inestabilidad de los coeficientes, no la calidad de la predicción. El modelo del
# capítulo 7 es una logística penalizada, y la penalización es la respuesta estándar a la colinealidad:
# con L2 (ridge) el problema tiene solución única aunque dos columnas sean casi iguales, y el peso se
# reparte entre ellas; con L1 (lasso) tiende a quedarse con una del par y anular la otra. Por eso **no
# se eliminan variables a ciegas por su VIF**: quitar la antigüedad por tener VIF 35 descartaría una de
# las predictoras más fuertes de los atributos, y quitar ventanas antes de ver si la de 1 mes aporta algo
# que la de 12 no tiene sería decidir sin evidencia. La redundancia sí tiene dos consecuencias que se
# llevan al modelo: (1) los coeficientes de un grupo de casi duplicados (las tres ventanas de un
# concepto, el reloj de la persona) se interpretan en conjunto, nunca uno por uno; y (2) la elección
# entre L1 y L2 y la fuerza de la penalización se deciden por validación temporal, sabiendo que L1 hará
# una selección entre las ventanas que puede cambiar de un pliegue a otro sin que cambie la predicción.
# ```
#
# ## Componentes principales (PCA)
#
# El PCA busca las direcciones de máxima varianza de la matriz estandarizada. Aquí es exploratorio:
# sirve para medir cuántas dimensiones tiene de verdad la historia laboral y qué bloques la forman, no
# para reemplazar las variables en el modelo (el modelo del capítulo 7 usa las variables originales,
# que se pueden interpretar). Se ajusta con el entrenamiento, sobre las 88 numéricas ya preparadas, y
# para comparar también sobre la matriz completa con las categóricas en una columna por categoría. El
# curso sugiere retener los componentes que explican entre el 70 % y el 80 % de la varianza.

# %%
pca = PCA(random_state=SEMILLA).fit(Zs)
var = pca.explained_variance_ratio_
acum = var.cumsum()
lam = pca.explained_variance_
umbral = {f'{int(100 * u)} %': int(np.argmax(acum >= u) + 1) for u in [0.5, 0.7, 0.8, 0.9]}
print(f'{len(COLS)} columnas | componentes para ' + ', '.join(f'{k}: {v}' for k, v in umbral.items()))
print(f'Kaiser (autovalor > 1): {(lam > 1).sum()} | razón de participación (Σλ)²/Σλ²: '
      f'{lam.sum() ** 2 / (lam ** 2).sum():.1f} | PC1 explica {100 * var[0]:.1f} %')

# para comparar: la matriz de diseño completa (numéricas + categóricas en una columna por categoría)
prep = ColumnTransformer([
    ('num', 'passthrough', COLS),
    ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='(nulo)'),
                          OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=300,
                                        sparse_output=False)), CAT)])
Xd = prep.fit_transform(pd.concat([pd.DataFrame(Zs, columns=COLS, index=tr.index), tr[CAT]], axis=1))
Xd = StandardScaler().fit_transform(Xd)
acum_d = PCA(random_state=SEMILLA).fit(Xd).explained_variance_ratio_.cumsum()
print(f'matriz completa ({Xd.shape[1]} columnas): componentes para ' + ', '.join(
    f'{int(100 * u)} %: {int(np.argmax(acum_d >= u) + 1)}' for u in [0.5, 0.7, 0.8, 0.9]))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
n = 60
ax[0].bar(np.arange(1, n + 1), var[:n], color=GRIS_CL, label='por componente')
ax[0].plot(np.arange(1, n + 1), acum[:n], marker='.', color=VERDE, label='acumulada (numéricas)')
ax[0].plot(np.arange(1, n + 1), acum_d[:n], ls='--', color=ORO, label='acumulada (con categóricas)')
for u in [0.7, 0.8]:
    ax[0].axhline(u, ls=':', color=GRIS, lw=0.8)
ax[0].set(title='Varianza explicada', xlabel='Número de componentes', ylabel='Proporción')
ax[0].legend(fontsize=7)
ax[1].plot(np.arange(1, n + 1), lam[:n], marker='.', color=VERDE)
ax[1].axhline(1, ls=':', color=GRIS, lw=0.8)
ax[1].set(title='Autovalores (criterio de Kaiser en 1)', xlabel='Componente', ylabel='Autovalor')
plt.tight_layout()
plt.show()

# %%
# cargas: parte de la varianza de cada componente que aporta cada bloque (suma de cargas al cuadrado)
K = 6
cargas = pd.DataFrame(pca.components_[:K].T, index=COLS, columns=[f'PC{k + 1}' for k in range(K)])
aporte = (cargas ** 2).groupby(lambda c: BLOQUE[c]).sum().reindex(ORDEN_BLOQUES)

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.heatmap(aporte, annot=True, fmt='.2f', cmap='verde', vmin=0, vmax=1, ax=ax, annot_kws={'size': 8},
            cbar_kws={'label': 'suma de cargas²'})
ax.set(title='Aporte de cada bloque a los primeros componentes', xlabel='', ylabel='')
plt.tight_layout()
plt.show()

for k in cargas:
    mayores = cargas[k].reindex(cargas[k].abs().sort_values(ascending=False).index).head(6)
    print(f'{k} ({100 * var[int(k[2:]) - 1]:.1f} %): ' +
          ', '.join(f'{c} ({v:+.2f})' for c, v in mayores.items()))

# %%
S = pca.transform(Zs)
auc = pd.Series([roc_auc_score(y, S[:, k]) for k in range(10)], index=[f'PC{k + 1}' for k in range(10)])
print('AUC univariado de los 10 primeros componentes (max(AUC, 1 - AUC)):')
print(auc.map(lambda a: max(a, 1 - a)).round(3).to_frame('AUC').T.to_string())

fig, ax = plt.subplots(figsize=(6.5, 4.5))
renuncia = y == 1
idx = rng.choice(np.flatnonzero(~renuncia), 8000, replace=False)
ax.scatter(S[idx, 0], S[idx, 1], s=3, c=GRIS_CL, label='No renuncia (muestra de 8.000)')
ax.scatter(S[renuncia, 0], S[renuncia, 1], s=6, c=ORO, label='Renuncia')
ax.legend(fontsize=7, markerscale=2)
ax.set(title='PC1 frente a PC2', xlabel=f'PC1 ({100 * var[0]:.1f} %)', ylabel=f'PC2 ({100 * var[1]:.1f} %)')
plt.tight_layout()
plt.show()

# %% [markdown]
# Nótese que la varianza está muy repartida. PC1 explica solo el 11,6 %; hacen falta 11 componentes
# para el 50 %, 24 para el 70 % y 33 para el 80 %, el rango del curso. La dimensionalidad efectiva
# coincide por dos caminos: 25 autovalores mayores que 1 (Kaiser) y una razón de participación
# $(\sum \lambda)^2 / \sum \lambda^2$ de 26,7. Es decir, las 88 columnas equivalen a unas 25-33
# direcciones independientes: la redundancia de las
# ventanas y del reloj reduce la dimensión a un tercio, pero lo que queda sigue siendo alto. Con las
# categóricas (267 columnas) hacen falta 59 componentes para el 70 % y 85 para el 80 %: las indicadoras
# añaden muchas dimensiones casi independientes.
#
# Los primeros componentes se leen por bloque:
#
# - **PC1 (11,6 %), carga de trabajo pagada:** dominicales, recargo nocturno y horas extra en sus tres
#   ventanas. Es la dirección del bloque de jornada.
# - **PC2 (7,4 %), el reloj de la persona:** meses en la función y en la posición, antigüedad reconocida,
#   meses desde el cambio de contrato, la edad y el histórico de cesantías para vivienda, todos con el
#   mismo signo. Mezcla atributos, contrato, trayectoria y arraigo.
# - **PC3 (6,3 %), posición salarial:** elegibilidad para horas extra, estar en la mediana local o lejos
#   de ella, meses con bonificación por productividad, vacaciones pendientes y desigualdad del oficio y
#   nivel.
# - **PC4 (5,9 %), turnos largos:** porcentaje de turnos de 10 y de 12 horas, horas por turno y turnos
#   largos sin pago de extras. Es jornada, pero la de las marcas biométricas, distinta de la jornada
#   pagada de PC1.
# - **PC5 (4,0 %), ausencias:** licencias remuneradas y no remuneradas y meses con ausencia pagada, junto
#   con aumentos por mérito. **PC6 (3,4 %)** es de nuevo salario relativo (residuo del sueldo y
#   posicionamiento frente a privación relativa), mezclado con la marca de ingreso reciente.
#
# Ningún componente separa por sí solo a quienes renuncian: las renuncias se reparten por todo el plano
# PC1-PC2. El AUC univariado más alto es el de PC2 (0,67), que es básicamente la antigüedad; le siguen
# PC3 (0,65), PC4 (0,62) y PC1 (0,61). La señal está en varias direcciones a la vez y ninguna domina, lo que favorece
# un modelo que sume muchas contribuciones pequeñas sobre uno que dependa de un eje.
#
# ## Atípicos multivariados
#
# Un atípico multivariado es una fila cuya combinación de valores es rara aunque cada valor por separado
# no lo sea. Se usan dos métodos sobre las 88 numéricas estandarizadas y recortadas:
#
# - **Distancia de Mahalanobis**, $d^2 = (x - \mu)^\top \Sigma^{-1} (x - \mu)$, que mide la distancia al
#   centro teniendo en cuenta la covarianza. Si los datos fueran normales multivariados, $d^2$ seguiría una
#   $\chi^2$ con tantos grados de libertad como columnas, y el cuantil 0,999 marcaría el 0,1 % de las
#   filas. No se aplica a la matriz con las categóricas en indicadoras: las indicadoras no son
#   continuas, la normal no tiene sentido para ellas y, con todas las categorías, la covarianza es
#   singular (las indicadoras de una variable suman 1).
# - **Isolation Forest**, que mide qué tan fácil es aislar una fila con cortes al azar: las raras se
#   aíslan con pocos cortes. No supone ninguna distribución.
#
# Lo que interesa no es borrar filas, sino saber si las raras renuncian distinto.

# %%
iso = IsolationForest(n_estimators=300, contamination='auto', random_state=SEMILLA, n_jobs=4)
p_iso = -iso.fit(Zs).score_samples(Zs)

centrado = Zs - Zs.mean(axis=0)
S_inv = np.linalg.pinv(np.cov(Zs, rowvar=False))
d2 = np.einsum('ij,jk,ik->i', centrado, S_inv, centrado)
gl = np.linalg.matrix_rank(np.cov(Zs, rowvar=False))
corte = chi2.ppf(0.999, gl)
print(f'Mahalanobis: rango de la covarianza {gl} de {Zs.shape[1]} | '
      f'filas sobre el cuantil 0,999 de χ²({gl}): {100 * (d2 > corte).mean():.1f} % (se esperaría 0,1 %)')
print(f'Spearman entre los dos puntajes: {pd.Series(p_iso).corr(pd.Series(d2), method="spearman"):.2f} | '
      f'coincidencia del 1 % más raro: '
      f'{100 * ((p_iso >= np.quantile(p_iso, .99)) & (d2 >= np.quantile(d2, .99))).sum() / (0.01 * len(d2)):.0f} %')

etiquetas = [f'D{k}' for k in range(1, 11)]
tablas = {}
fig, ax = plt.subplots(1, 2, figsize=(13, 3.4), sharey=True)
for a, (titulo, p) in zip(ax, [('Isolation Forest', p_iso), ('Mahalanobis', d2)]):
    t = tasa_ic(pd.DataFrame({OBJETIVO: y, 'decil': pd.qcut(p, 10, labels=etiquetas)}), 'decil')
    tablas[titulo] = t
    error = [t['tasa_%'] - t.ic_bajo, t.ic_alto - t['tasa_%']]
    a.bar(t.index.astype(str), t['tasa_%'], yerr=error, color=VERDE, capsize=3, **BORDE)
    for xi, v, alto in zip(range(len(t)), t['tasa_%'], t.ic_alto):
        a.text(xi, alto, f'{v:.2f}', ha='center', va='bottom', fontsize=7, color=TINTA)
    a.axhline(100 * y.mean(), ls='--', c=GRIS, lw=0.8)
    a.set(title=f'Tasa por decil de anomalía: {titulo} (D10 = más raro)', xlabel='Decil',
          ylabel='Tasa mensual (%)')
plt.tight_layout()
plt.show()
pd.concat({k: v[['renuncias', 'tasa_%']].T for k, v in tablas.items()})

# %%
# perfil del 1 % más raro según Isolation Forest: variables estandarizadas con mayor |media|
top = p_iso >= np.quantile(p_iso, 0.99)
perfil = pd.Series(Zs[top].mean(axis=0), index=COLS)
perfil = perfil.reindex(perfil.abs().sort_values(ascending=False).index).head(10)
r_top = int(y[top].sum())
print(f'1 % más raro: {top.sum():,} filas, {tr[top].persona_id.nunique():,} personas, renuncias: '
      + (f'{r_top}, tasa {100 * y[top].mean():.2f} %' if r_top >= 5 else '<5 (tasa suprimida)'))
print('medias estandarizadas más alejadas de 0:')
print(pd.DataFrame({'bloque': [BLOQUE[c] for c in perfil.index], 'media z': perfil.round(2)}).to_string())
for v in ['linea', 'tipo_unidad', 'contrato']:
    reparto = (100 * tr[top][v].value_counts(normalize=True)).round(0).head(3)
    print(f'{v} en el 1 % más raro:', reparto.to_dict(), '| en todo el entrenamiento:',
          (100 * tr[v].value_counts(normalize=True)).round(0).head(3).to_dict())

# %% [markdown]
# Nótese primero que la normal multivariada no se sostiene: el 17,1 % de las filas supera el cuantil
# 0,999 de la $\chi^2(88)$, cuando se esperaría el 0,1 %. Con binarias, conteos llenos de ceros y los
# picos de la imputación, la distancia de Mahalanobis sirve para ordenar las filas de más típica a más
# rara, pero no para marcar atípicos con un corte de la $\chi^2$. Los dos puntajes ordenan de forma
# parecida (Spearman 0,84) pero no marcan las mismas filas extremas: solo el 16 % del 1 % más raro es
# común a los dos.
#
# Lo importante para el modelo: **las filas raras no renuncian más**. Por deciles de Isolation Forest la
# tasa oscila entre 0,74 % y 1,19 %, y los dos deciles más raros (0,86 % y 0,74 %) quedan algo por
# debajo de la media; por Mahalanobis oscila entre 0,82 % y 1,32 %, con los dos deciles más raros
# también por debajo (0,82 % y 0,83 %). El 1 % más raro según Isolation Forest (675 filas de 168
# personas) tiene
# menos de 5 renuncias. Su perfil es de movilidad interna reciente: ascenso reciente, ascensos y
# requisiciones de promoción en el último año, aumento de sueldo y lejanía de la mediana local, sobre
# todo en áreas funcionales y plantas industriales (61 % y 32 %, frente al 31 % y 12 % del total) y más
# de contrato indefinido (65 % frente a 47 %). Es un
# perfil raro pero estable. Las filas atípicas son combinaciones legítimas de puestos poco comunes, no
# errores de registro, y se conservan; el recorte en p1-p99 ya limita su influencia en el modelo.
#
# ## Conglomerados (k-medias)
#
# Se aplica k-medias sobre los 33 primeros componentes (el 80 % de la varianza), que quitan ruido y
# redundancia sin perder la estructura, con k de 2 a 20 y 10 arranques por k. Tres índices, porque
# ninguno basta solo:
#
# - **Silueta** (de -1 a 1, mayor es mejor): compara la distancia media de cada fila a su grupo con la
#   distancia al grupo vecino. Por debajo de 0,25 se lee como ausencia de estructura sustancial. Cuesta
#   $O(n^2)$, así que se calcula en una muestra fija de 10.000 filas.
# - **Calinski-Harabasz** (mayor es mejor): razón entre la dispersión entre grupos y dentro de ellos.
# - **Davies-Bouldin** (menor es mejor): similitud media de cada grupo con el más parecido.
#
# Si la mejor silueta cayera en el extremo superior del rango, el código lo amplía de 5 en 5.

# %%
m = umbral['80 %']
Sk = S[:, :m]
muestra = rng.choice(len(Sk), 10000, replace=False)


def evaluar(ks):
    filas = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=SEMILLA).fit(Sk)
        tam = np.bincount(km.labels_)
        filas.append({'k': k, 'silueta': silhouette_score(Sk[muestra], km.labels_[muestra]),
                      'Calinski-Harabasz': calinski_harabasz_score(Sk, km.labels_),
                      'Davies-Bouldin': davies_bouldin_score(Sk, km.labels_),
                      'inercia': km.inertia_, 'menor grupo': tam.min(), 'mayor grupo': tam.max()})
    return filas


filas = evaluar(range(2, 21))
while max(filas, key=lambda f: f['silueta'])['k'] == filas[-1]['k'] and filas[-1]['k'] < 30:
    filas += evaluar(range(filas[-1]['k'] + 1, filas[-1]['k'] + 6))   # la mejor silueta está en el borde
indices = pd.DataFrame(filas).set_index('k')
print(f'k-medias sobre los {m} primeros componentes (80 % de la varianza); silueta en una muestra de '
      f'10.000 filas, Calinski-Harabasz y Davies-Bouldin en todo el entrenamiento')
indices.round({'silueta': 3, 'Calinski-Harabasz': 0, 'Davies-Bouldin': 2, 'inercia': 0})

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
for a, (col, mejor) in zip(ax, [('silueta', 'mayor'), ('Calinski-Harabasz', 'mayor'),
                                ('Davies-Bouldin', 'menor')]):
    a.plot(indices.index, indices[col], marker='o', color=VERDE)
    a.set(title=f'{col} ({mejor} es mejor)', xlabel='k')
plt.tight_layout()
plt.show()

# %%
k = int(indices.silueta.idxmax())
km = KMeans(k, n_init=10, random_state=SEMILLA).fit(Sk)
otra = KMeans(k, n_init=10, random_state=SEMILLA + 1).fit(Sk)
print(f'k elegido por la silueta: {k} | estabilidad frente a otra semilla (ARI): '
      f'{adjusted_rand_score(km.labels_, otra.labels_):.2f}')


def perfil_grupos(etq):
    """Filas, personas y tasa por grupo; '<5' si hay menos de 5 renuncias y fuera los de < 300 filas."""
    d = tr.assign(grupo=etq)
    g = d.groupby('grupo').agg(filas=(OBJETIVO, 'size'), personas=('persona_id', 'nunique'),
                               renuncias=(OBJETIVO, 'sum'),
                               pct_fijo=('contrato', lambda s: round(100 * (s == 'Termino Fijo').mean(), 1)),
                               linea=('linea', lambda s: f'{s.value_counts().index[0]} '
                                                         f'({100 * s.value_counts(normalize=True).iloc[0]:.0f} %)'))
    g['tasa_%'] = (100 * g.renuncias / g.filas).round(2)
    z = pd.DataFrame(Zs, columns=COLS).groupby(etq).mean()
    g['rasgos (media z)'] = [', '.join(f'{c} {v:+.1f}' for c, v in
                                       z.loc[i].reindex(z.loc[i].abs().sort_values(ascending=False).index)
                                       .head(3).items()) for i in g.index]
    chicos = g.filas < 300
    g = g[~chicos].astype({'renuncias': object, 'tasa_%': object})
    pocas = g.renuncias.astype(int) < 5
    g.loc[pocas, 'renuncias'] = '<5'
    g.loc[pocas, 'tasa_%'] = '-'
    print(f'grupos con menos de 300 filas (fuera de la tabla): {int(chicos.sum())} | '
          f'con menos de 5 renuncias (tasa suprimida): {int(pocas.sum())}')
    return g.sort_values('filas', ascending=False)


perfil_grupos(km.labels_)

# %% [markdown]
# Nótese que **no hay estructura natural fuerte**. La mejor silueta es 0,19, con k = 2, y todas las
# demás quedan entre 0,11 y 0,13, muy por debajo de 0,25; la inercia baja sin un codo claro, y los tres
# índices no coinciden: la silueta y Calinski-Harabasz prefieren k = 2 (Calinski-Harabasz baja casi
# de forma monótona con k, como suele ocurrir sin grupos reales), mientras Davies-Bouldin mejora
# despacio hasta su mínimo de 1,90 en k = 11. Como la mejor silueta está en el extremo inferior del
# rango, ampliar k hacia arriba no la mejora (se probó hasta 20, y la silueta se queda en 0,13). Desde
# k = 16 aparecen grupos de 294 filas: k-medias empieza a aislar rincones pequeños.
#
# Los dos grupos de k = 2 son estables (ARI 1,00 entre dos semillas) pero no son subpoblaciones en el
# sentido de la renuncia: separan a quienes trabajan dominicales y horas extra (12.983 filas, la
# dirección de PC1) del resto (54.433), y los dos tienen casi la misma tasa, 0,96 % y 1,03 %. Es el primer
# componente cortado en dos, no un tipo de trabajador.

# %% [markdown]
# Con k = 11, la partición de menor Davies-Bouldin, los grupos son solo medianamente estables (ARI 0,69
# frente a otra semilla); ninguno tiene menos de 300 filas.

# %%
k_db = int(indices['Davies-Bouldin'].idxmin())
km_db = KMeans(k_db, n_init=10, random_state=SEMILLA).fit(Sk)
otra = KMeans(k_db, n_init=10, random_state=SEMILLA + 1).fit(Sk)
print(f'k con el menor Davies-Bouldin: {k_db} | estabilidad frente a otra semilla (ARI): '
      f'{adjusted_rand_score(km_db.labels_, otra.labels_):.2f}')
g = perfil_grupos(km_db.labels_)
g

# %% [markdown]
# Aquí sí aparece algo útil, aunque la geometría sea débil: la tasa varía mucho entre grupos, de 0,13 %
# a 3,08 %. Los de tasa alta son reconocibles: turnos largos sin pago de extras, casi todo en banano
# (3,08 %); el primer mes y el ingreso a mitad de mes (1,65 %); capacitación y parte variable del
# ingreso en palma (1,41 %); y poco tiempo en la función, con 85 % de contrato fijo (1,40 %). Los de
# tasa más baja son de contrato indefinido y mucha antigüedad (0,13 %) y de licencias remuneradas con
# aumentos por mérito (0,29 %). Pero cada uno de esos grupos es una combinación de rasgos que ya son predictores por sí
# solos (antigüedad, contrato, turnos, primer mes), y los grupos no están separados en el espacio (la
# silueta de k = 11 es 0,13). Son cortes descriptivos de un continuo, no tipos de trabajador. Para el
# modelo no conviene usarlos como variable: la logística ya puede combinar esas mismas variables, y un
# conglomerado estimado con todo el entrenamiento introduciría una decisión que no se validó por
# pliegues.
#
# ## Síntesis
#
# - **Redundancia.** Hay 50 pares de numéricas con |r| > 0,7 y 13 VIF por encima de 10 (mediana 2,8), pero tienen dos
#   orígenes identificables: las ventanas de 1, 3 y 12 meses de un mismo concepto de jornada, y el reloj
#   de la persona (antigüedad reconocida, meses desde el cambio de contrato, en la posición y en la
#   función). Entre bloques
#   de la historia (salario, jornada, ingreso, ausencias, vacaciones) no hay pares fuertes: la historia
#   laboral trae información nueva frente a los atributos. Entre categóricas, sociedad y línea, y oficio y
#   familia de cargo, están anidadas exactamente.
# - **Dimensionalidad.** Las 88 numéricas equivalen a unas 25-33 dimensiones (24 componentes para el
#   70 %, 33 para el 80 %, Kaiser 25, razón de participación 26,7). Los ejes principales son la carga de
#   trabajo pagada, el reloj de la persona, la posición salarial y los turnos largos. Ninguno separa por
#   sí solo a quienes renuncian (AUC máximo 0,67).
# - **Atípicos.** Las filas raras, por Isolation Forest o por Mahalanobis, no renuncian más; la
#   normalidad multivariada no se cumple, así que el corte de la $\chi^2$ no se usa. Se conservan todas
#   las filas y el recorte en p1-p99 limita la influencia de los valores extremos.
# - **Subpoblaciones.** Sin estructura natural fuerte: la silueta no pasa de 0,19 en k = 2 a 20. Los
#   grupos de k = 11 difieren mucho en tasa (0,13 % a 3,08 %), pero son combinaciones de predictores conocidos, no grupos
#   separados.
#
# **Consecuencias para el modelo (capítulo 7).** Se usa una logística **regularizada** con todas las
# variables, sin eliminar por VIF: la penalización resuelve la inestabilidad de los coeficientes que
# produce la colinealidad, y la predicción no depende de cómo se reparta el peso entre casi duplicados.
# Con L1 el modelo elegirá entre las ventanas de un mismo concepto, y esa elección puede variar entre
# pliegues sin que la predicción cambie; por eso los coeficientes de un grupo redundante se interpretan
# juntos. Las categóricas anidadas hacen singular la matriz sin penalización, otra razón para no usar la
# logística sin regularizar. Con unas 30 dimensiones efectivas y 688 renuncias de entrenamiento, la
# fuerza de la penalización importa: la rejilla de C debe llegar a valores pequeños y elegirse por
# validación temporal. La preparación de las numéricas (recorte p1-p99, log con signo en las de cola
# larga y mediana) es la misma de este capítulo, ajustada dentro de cada pliegue; los indicadores de
# faltante, uno por variable, repiten pocos patrones y la penalización absorbe su redundancia. No se añaden los componentes ni los conglomerados como variables.
