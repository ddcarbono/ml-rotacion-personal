# %% [markdown]
# # EDA del objetivo: univariado y bivariado
#
# ```{admonition} Alcance
# :class: tip
# Este capítulo usa solo el entrenamiento (enero de 2025 a abril de 2026). Las tasas son mensuales:
# la probabilidad de renunciar en un mes dado. Los intervalos de Wilson al 95 % describen las tasas;
# la incertidumbre del modelo se mide en el capítulo 7, remuestreando personas.
# ```
#
# Para una tasa $\hat p = a / n$, el intervalo de Wilson es
#
# $$
# \frac{\hat p + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat p (1 - \hat p)}{n} + \frac{z^2}{4n^2}}}
#   {1 + \frac{z^2}{n}}, \qquad z = 1{,}96 .
# $$
#
# Se prefiere al intervalo de Wald ($\hat p \pm z \sqrt{\hat p (1 - \hat p) / n}$) porque con tasas
# cercanas a cero este puede dar límites negativos y cubre menos de lo que dice.
#
# El capítulo sigue el orden del enunciado: primero el objetivo (2.1), luego cada predictora por
# separado (2.2) y después las relaciones de a dos, entre predictoras y con el objetivo (2.3). Son
# 105 predictoras: las 18 de atributos se analizan una por una y las 87 de la historia laboral, por
# bloque; todas entran en las tablas y solo las más relevantes de cada bloque tienen gráfico. Las
# cinco descartadas en los capítulos 1 y 5 no se analizan: `contrato_fijo` (idéntica a `contrato`),
# `horas_bajo_legal` (cero en el 99,98 % de las filas) y `cambios_plan_12m`, `horas_diarias_teoricas`
# y `plan_horario`, que siguen el calendario de la reducción legal de la jornada (capítulo 5) y no a
# la persona.
#
# Un recordatorio sobre el tiempo de la historia laboral: en la fila del mes *t*, la nómina, las
# marcaciones y las novedades llegan hasta *t* − 1; las variables con sufijo `_r2` y los meses desde
# las últimas vacaciones, hasta *t* − 2. Por eso en el primer mes de cada episodio (2,4 % de las
# filas) esas variables están vacías, y buena parte de los nulos de 2,4 % que se verán son esas filas.
#
# ```{admonition} Privacidad
# :class: important
# En toda tabla o gráfico por grupo, las categorías con menos de cinco renuncias, menos de cinco
# personas o (en tasas) menos de 300 persona-mes se juntan en "otras (agrupadas)", y ese grupo se amplía
# hasta cumplir los mismos mínimos; los tramos ordenados se unen con el vecino. Así ninguna celda
# pequeña se publica ni se puede obtener restando del total (688 renuncias). Una comprobación
# automática lo verifica en cada tabla (función `verificar`).
# ```

# %%
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from comun import (eje_llano, rotulo, cargar, particion, estilo, tasa, tasa_ic, grafico_tasa, etiquetar, puntos,
                   OBJETIVO, NUM, NUM_BASE, BIN, CAT, CAT_BASE, CAT_HISTORIA, BASE as BASE_VARS,
                   BLOQUES, BLOQUE, HISTORIA, PREDICTORAS, SEMILLA, NOMBRE, UNIDAD,
                   VERDE, ORO, GRIS, GRIS_CL, TINTA, BORDE)

estilo()
pd.set_option('display.max_columns', 30)

tr, _ = particion(cargar())
y = tr[OBJETIVO].to_numpy()
BASE = 100 * y.mean()
NOMBRE = {**NOMBRE, 'antig_meses': 'antigüedad reconocida'}
UNIDAD = {**UNIDAD, 'antig_meses': 'Antigüedad reconocida (meses)'}
_dic = pd.read_csv('diccionario.csv').set_index('variable')


def nom(v):
    """Nombre legible de una variable (diccionario del capítulo 1)."""
    return NOMBRE.get(v, v)


def ajustar(p):
    """Valores p corregidos por Holm (error de familia) y por Benjamini-Hochberg (FDR)."""
    p = np.asarray(p, float)
    return multipletests(p, method='holm')[1], multipletests(p, method='fdr_bh')[1]


OTRAS = 'otras (agrupadas)'
CONTROL = []          # registro del control de privacidad de cada tabla o gráfico por grupo


def agrupar(df, v, min_ren=5, min_per=5, min_filas=0):
    """Categorías de `v` listas para publicar: las que tienen menos de `min_ren` renuncias, menos de
    `min_per` personas o menos de `min_filas` filas se juntan en 'otras (agrupadas)', y si ese grupo
    no llega a los mínimos se le suman las categorías más pequeñas hasta que llegue. Así no queda
    ninguna celda pequeña ni una categoría oculta que se pueda obtener restando del total."""
    x = df[v].astype(object).where(df[v].notna(), '(nulo)').astype(str)
    g = df.groupby(x).agg(filas=(OBJETIVO, 'size'), renuncias=(OBJETIVO, 'sum'),
                          personas=('persona_id', 'nunique'))
    chicas = set(g.index[(g.renuncias < min_ren) | (g.personas < min_per) | (g.filas < min_filas)])

    def suficiente():
        m = x.isin(chicas)
        return (df.loc[m, OBJETIVO].sum() >= min_ren and df.loc[m, 'persona_id'].nunique() >= min_per
                and m.sum() >= min_filas)

    while chicas and not suficiente() and len(chicas) < len(g) - 1:
        resto = g.drop(index=list(chicas)).sort_values(['filas', 'renuncias'])
        chicas.add(resto.index[0])
    # con pocas categorías juntas se nombran; con muchas, 'otras (agrupadas)'
    etiqueta = ' + '.join(sorted(chicas, key=lambda c: -g.loc[c, 'filas'])) if len(chicas) <= 3 else OTRAS
    return x.where(~x.isin(chicas), etiqueta)


def fusionar(df, clave, orden, min_ren=5, min_per=5, min_filas=0):
    """Une tramos consecutivos (en el orden dado) hasta que cada uno tenga los mínimos; el último
    tramo insuficiente se une al anterior. Devuelve {tramo: grupo de tramos}."""
    grupos, actual = [], []
    for k in orden:
        if not (clave == k).any():
            continue
        actual.append(k)
        m = clave.isin(actual)
        if (df.loc[m, OBJETIVO].sum() >= min_ren and df.loc[m, 'persona_id'].nunique() >= min_per
                and m.sum() >= min_filas):
            grupos.append(actual)
            actual = []
    if actual:
        if grupos:
            grupos[-1] = grupos[-1] + actual
        else:
            grupos.append(actual)
    return {k: tuple(g) for g in grupos for k in g}


def etiqueta_tramo(g):
    """'0-2' + '3-5' -> '0-5'; '60-119' + '120+' -> '60+'."""
    if len(g) == 1:
        return str(g[0])
    ini, fin = str(g[0]).split('-')[0], str(g[-1])
    return ini + ('+' if fin.endswith('+') else '-' + fin.split('-')[-1])


def verificar(nombre, t, total, col='renuncias', minimo=5):
    """Control de privacidad: toda celda publicada tiene al menos `minimo` renuncias y las celdas
    suman el total, así que ninguna celda pequeña se puede obtener por resta."""
    t = pd.DataFrame(t)
    assert (t[col] >= minimo).all(), f'{nombre}: celda con menos de {minimo}'
    assert int(t[col].sum()) == int(total), f'{nombre}: las celdas no suman el total'
    CONTROL.append({'tabla': nombre, 'celdas': len(t), 'mínimo publicado': int(t[col].min()),
                    'suma': int(t[col].sum()), 'total': int(total)})


def publicable(df, v):
    """Tasa con IC por categoría para un gráfico: las categorías con < 300 filas, < 5 renuncias o
    < 5 personas van agrupadas."""
    t = tasa_ic(df, agrupar(df, v, min_filas=300))
    verificar(f'gráfico de {v}', t, df[OBJETIVO].sum())
    return t


print(f'entrenamiento: {len(tr):,} filas | {tr.persona_id.nunique():,} personas | '
      f'{int(y.sum())} renuncias | tasa mensual {BASE:.2f} %')

# %% [markdown]
# ## El objetivo
#
# El objetivo es binario: `y_renuncia` = 1 si la persona renuncia voluntariamente en el mes. Se
# reporta la frecuencia de cada clase, el grado de desbalance, el número de casos de la clase
# minoritaria y cómo se reparten en los meses del entrenamiento (el eje vertical del gráfico de
# clases está en escala logarítmica para que la barra de las renuncias se vea).

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={'width_ratios': [1, 2.2]})
conteo = tr[OBJETIVO].value_counts().sort_index()
barras = ax[0].bar(['0: no renuncia', '1: renuncia'], conteo.values, color=[VERDE, ORO], **BORDE)
etiquetar(ax[0], barras, textos=[f'{n:,} ({100 * n / len(tr):.2f} %)' for n in conteo.values])
ax[0].set(title='Clases del objetivo', ylabel='Persona-mes', yscale='log')
eje_llano(ax[0].yaxis)

por_mes = tr.groupby('mes')[OBJETIVO].agg(filas='size', renuncias='sum')
barras = ax[1].bar(por_mes.index, por_mes.renuncias, color=ORO, **BORDE)
etiquetar(ax[1], barras)
ax[1].set(title='Renuncias por mes del entrenamiento', ylabel='Renuncias', xlabel='')
ax[1].tick_params(axis='x', rotation=60, labelsize=7)
plt.tight_layout()
plt.show()

eventos = tr.evento.value_counts().to_frame('filas')
eventos['%'] = (100 * eventos.filas / len(tr)).round(2)
eventos['y'] = tr.groupby('evento')[OBJETIVO].max()
print(eventos.to_string())
n1, n0 = int(y.sum()), int(len(y) - y.sum())
print(f'\nrazón de desbalance: 1 renuncia por cada {n0 / n1:.0f} filas sin renuncia')
print(f'personas distintas que renuncian: {tr.loc[tr[OBJETIVO] == 1, "persona_id"].nunique()} '
      f'de {tr.persona_id.nunique():,}')
print(f'renuncias por mes: mínimo {por_mes.renuncias.min()}, mediana {por_mes.renuncias.median():.0f}, '
      f'máximo {por_mes.renuncias.max()}')
print(f'tasa anual equivalente: {100 * (1 - (1 - BASE / 100) ** 12):.1f} %')
print(f'exactitud de "nadie renuncia": {100 * n0 / len(y):.2f} %')

# %% [markdown]
# Nótese que la clase positiva son 688 de 67.416 filas (1,02 %): una renuncia por cada 97 filas sin
# renuncia. Las 688 renuncias son de 688 personas distintas, de las 5.500 del entrenamiento, así que
# nadie renuncia dos veces en el periodo. Una tasa mensual de 1,02 % equivale a que cerca de una de
# cada nueve personas (11,6 %) renuncie en un año. Por mes hay entre 27 y 60 renuncias, con mediana
# de 42. Las 441 otras salidas (despido, fin de contrato, pensión y otras) y las 31
# renuncias administrativas (renuncias registradas tras las cuales la persona sigue en el grupo,
# capítulo 1) cuentan como $y = 0$.
#
# ```{admonition} Implicaciones para la métrica y la validación
# :class: important
# - **La exactitud (*accuracy*) no sirve.** Predecir que nadie renuncia acierta el 98,98 % de las filas y no
#   identifica ninguna renuncia.
# - **Métrica principal: PR-AUC** (área bajo la curva de precisión-exhaustividad; en inglés también AUC-PR o *average precision*). Se concentra en la clase positiva y su piso
#   es la prevalencia (0,0102): un modelo que ordena al azar obtiene 0,0102, y cualquier ganancia se
#   lee como múltiplo de ese piso. La ROC-AUC se reporta como complemento, porque con 97 negativos
#   por positivo premia ordenar bien la gran masa de negativos.
# - **Métrica operativa: captura en el 10 % de mayor riesgo de cada mes**, la fracción de las
#   renuncias del mes que caen en la lista que se revisaría. Al azar vale 10 %.
# - **Validación.** Con unas 42 renuncias por mes, un mes de validación da una PR-AUC muy ruidosa;
#   por eso se promedian los ocho pliegues temporales (septiembre de 2025 a abril de 2026, capítulo
#   2) y las diferencias entre modelos se acompañan de intervalos *bootstrap*. La partición es
#   cronológica, no estratificada: la prevalencia de cada pliegue la da el mes, y estratificar
#   mezclaría meses. No se remuestrea (ni sobremuestreo ni SMOTE) en el EDA; el peso de las clases
#   (`class_weight`) se evalúa en el capítulo 7 dentro de la validación.
# ```

# %% [markdown]
# ## Análisis unidimensional
#
# ### Tipo y cardinalidad de las 105 predictoras

# %%
def tipo(v):
    if v in CAT:
        return 'categórica ordinal' if v == 'nivel' else 'categórica nominal'
    return _dic.tipo.get(v, 'numérica')


tipos = pd.DataFrame({
    'bloque': [BLOQUE[v] for v in PREDICTORAS],
    'tipo': [tipo(v) for v in PREDICTORAS],
    'valores distintos': [tr[v].nunique() for v in PREDICTORAS],
    '% nulos': [round(100 * tr[v].isna().mean(), 1) for v in PREDICTORAS],
}, index=pd.Index(PREDICTORAS, name='variable'))
resumen_tipos = pd.crosstab(tipos.bloque, tipos.tipo, margins=True, margins_name='total')
resumen_tipos

# %% tags=["hide-output"]
with pd.option_context('display.max_rows', 200):
    display(tipos.assign(nombre=[nom(v) for v in tipos.index]))

# %% [markdown]
# Se evidencia que de las 105 predictoras, 44 son numéricas continuas, 29 numéricas discretas (conteos y
# meses), 15 binarias y 17 categóricas (16 nominales y una ordinal, el nivel del cargo). Entre las
# categóricas está `estado_gestion_tiempos`, que viene como código numérico pero es una etiqueta (tres
# valores) y se trata como categórica. Las cardinalidades de las categóricas van de 2 (género,
# contrato) a 57 (ubicación). La tabla plegada da la de cada variable con su % de nulos: 63
# predictoras tienen algún nulo, en varias de ellas estructural (por ejemplo, `meses_al_vencimiento`
# es nulo exactamente en el 47,1 % de filas con contrato indefinido, que no vence); el mecanismo de
# los faltantes se describe en el capítulo 1 («Faltantes»).
#
# ### Asimetría y colas largas
#
# `pandas` reporta la curtosis como exceso (curtosis − 3), de modo que la normal vale 0. Se marca como
# **cola larga** toda numérica no binaria cuya asimetría sigue
# siendo \|g₁\| > 2 **después de recortarla a sus percentiles 1 y 99** (sobre los valores observados):
# es el criterio con el que los capítulos 4 y 7 deciden qué variables reciben el logaritmo con signo.
# Las tablas muestran también la asimetría cruda, como contexto.
#
# ### Edad y antigüedad reconocida
#
# La antigüedad de este panel es la **antigüedad reconocida** por la empresa: meses desde la fecha de
# antigüedad del sistema de nómina, con continuidad en los traslados entre sociedades del grupo. No
# es la antigüedad del episodio en la sociedad actual, de la que
# difiere en cerca del 10 % de las filas. El diagrama de caja muestra los atípicos (puntos dorados,
# más allá de 1,5 × IQR).

# %%
def asim_recortada(x):
    """Asimetría de los valores observados después de recortarlos a sus percentiles 1 y 99: el criterio
    del libro para el logaritmo con signo (el mismo del Pipeline del capítulo 7 y del capítulo 4).
    Como en la clase Recorte del capítulo 7: si p1 = p99 no se recorta (quedaría constante) y la
    asimetría es la de momentos, sin corrección por sesgo."""
    x = pd.Series(x).dropna().to_numpy(float)
    lo, hi = np.percentile(x, [1, 99])
    if hi > lo:
        x = np.clip(x, lo, hi)
    return stats.skew(x, bias=True)


def descriptivos(v):
    x = tr[v].dropna()
    q = x.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    iqr = q[0.75] - q[0.25]
    return {
        'n': len(x), '% nulos': 100 * tr[v].isna().mean(), 'media': x.mean(), 'desv.': x.std(),
        'p1': q[0.01], 'p5': q[0.05], 'p25': q[0.25], 'mediana': q[0.5], 'p75': q[0.75],
        'p95': q[0.95], 'p99': q[0.99], 'IQR': iqr,
        'asimetría': x.skew(), 'asimetría tras p1-p99': asim_recortada(x),
        'curtosis (exceso)': x.kurt(), '% ceros': 100 * (x == 0).mean(),
        '% fuera 1,5×IQR': 100 * ((x < q[0.25] - 1.5 * iqr) | (x > q[0.75] + 1.5 * iqr)).mean(),
    }


fig, ax = plt.subplots(2, 2, figsize=(11, 5.5), gridspec_kw={'height_ratios': [3, 1.2]})
for i, v in enumerate(NUM_BASE):
    sns.histplot(tr[v], bins=40, ax=ax[0, i], color=VERDE, **BORDE)
    ax[0, i].set(title=f'Distribución de la {nom(v)}', xlabel='', ylabel='Persona-mes')
    sns.boxplot(x=tr[v], ax=ax[1, i], color=GRIS_CL, showfliers=True,
                flierprops=dict(marker='o', ms=2, mfc=ORO, mec=ORO, alpha=0.5))
    ax[1, i].set(xlabel=UNIDAD[v])
plt.tight_layout()
plt.show()

filas = {}
for v in NUM_BASE:
    d = descriptivos(v)
    k2, p_norm = stats.normaltest(tr[v].dropna())
    d.update({"D'Agostino K²": k2, 'p normalidad': p_norm})
    filas[v] = d
pd.DataFrame(filas).T.drop(columns=['% ceros']).round(2)

# %% [markdown]
# Se observa que la edad es aproximadamente simétrica (asimetría 0,45) y platicúrtica (exceso −0,56):
# media 37,3 y mediana 36 años, IQR de 28 a 45 y solo 0,08 % de filas fuera de las vallas. La
# antigüedad reconocida es muy asimétrica a la derecha (1,45) y leptocúrtica (1,97): mediana de 37
# meses frente a una media de 64,6, IQR de 13 a 90 meses, y un 6,2 % de filas por encima de la valla
# superior (más de unos 205 meses), que son personas de muchos años y no errores. Con la antigüedad
# nueva aparece esa cola. Ninguno de los atípicos se elimina: son valores reales. La asimetría de la
# antigüedad justifica transformarla (logaritmo o tramos, capítulo 7).
#
# ```{admonition} Normalidad con cautela
# :class: note
# Con 67.416 filas cualquier desviación mínima de la normal resulta significativa, y la prueba de
# Shapiro-Wilk no se usa: su aproximación del valor p solo está validada hasta n = 5.000 y con
# muestras de este tamaño rechaza por potencia, no por forma. Se usa la prueba de D'Agostino-Pearson
# (combina asimetría y curtosis), y se lee junto con la forma de la distribución. En cualquier caso,
# la regresión logística no supone normalidad de las predictoras.
# ```
#
# ### Binarias de atributos

# %%
binarias = pd.DataFrame({v: tr[v].value_counts().sort_index() for v in BIN}).T
binarias.columns = ['0', '1']
binarias['% con 1'] = (100 * binarias['1'] / len(tr)).round(2)
binarias.index = [nom(v) for v in binarias.index]
binarias

# %% [markdown]
# Las tres binarias de atributos son poco frecuentes: el primer mes de un episodio es el
# 2,4 % de las filas, los reingresos el 8,1 % y los traslados en los últimos 12 meses el 2,7 %. Su
# relación con el objetivo se ve con las demás binarias, en la sección de razones de tasa.
#
# ### La historia laboral, bloque por bloque
#
# Para cada bloque, una tabla con los descriptivos de todas sus variables numéricas (media,
# desviación, percentiles, IQR, % de nulos, asimetría, curtosis, % de ceros y % fuera de las vallas
# de 1,5 × IQR) y un gráfico de las tres más asociadas con el objetivo en el entrenamiento (las de
# mayor $|P(\text{sup}) - 0{,}5|$, sección 3.3.3). El histograma corta en el percentil 99 para que
# se vea la forma; el diagrama de caja muestra todos los valores, con los atípicos en dorado. Por
# privacidad no se publican mínimos ni máximos: los percentiles 1 y 99 hacen su papel.

# %%
def p_sup(v):
    """P(sup) = P(X de quien renuncia > X de quien no) + ½ P(empate) = U / (n1 n0), el AUC univariado."""
    a = tr.loc[tr[OBJETIVO] == 1, v].dropna()
    b = tr.loc[tr[OBJETIVO] == 0, v].dropna()
    u, pv = stats.mannwhitneyu(a, b, alternative='two-sided')
    return u / (len(a) * len(b)), pv, a, b


NUMB = NUM + BIN                       # numéricas de comun (86, con las binarias de la historia) + las 3 binarias de atributos
PSUP = {v: p_sup(v)[0] for v in NUMB}
CONTINUAS = [v for v in NUM if tr[v].dropna().nunique() > 2]
# binarias con menos de 5 renuncias en el valor 1: sus medidas de asociación permitirían reconstruir el conteo
CHICAS = [v for v in NUMB if v not in CONTINUAS and tr.loc[tr[v] == 1, OBJETIVO].sum() < 5]


def top_bloque(b, k=3):
    vs = [v for v in BLOQUES[b] if v in CONTINUAS]
    return sorted(vs, key=lambda v: -abs(PSUP[v] - 0.5))[:k]


def tabla_bloque(b):
    vs = [v for v in BLOQUES[b] if v in NUM]
    t = pd.DataFrame({v: descriptivos(v) for v in vs}).T
    t.insert(0, 'tipo', [tipo(v) for v in vs])
    t['cola larga'] = np.where((t['asimetría tras p1-p99'].abs() > 2) & (t.tipo != 'binaria'), 'sí', '')
    t = t.drop(columns=['n', 'p5', 'p95'])
    t.index = [f'{v}' for v in vs]
    return t.round(2)


def figura_bloque(b):
    vs = top_bloque(b)
    fig, ax = plt.subplots(2, len(vs), figsize=(4 * len(vs), 4.2), squeeze=False,
                           gridspec_kw={'height_ratios': [3, 1.2]})
    for i, v in enumerate(vs):
        x = tr[v].dropna()
        sns.histplot(x[x <= x.quantile(0.99)], bins=30, ax=ax[0, i], color=VERDE, **BORDE)
        ax[0, i].set(title=f'{textwrap.shorten(nom(v), 40, placeholder="...")}\nP(sup) = {PSUP[v]:.3f}', xlabel='', ylabel='Persona-mes')
        ax[0, i].title.set_fontsize(9)
        sns.boxplot(x=x, ax=ax[1, i], color=GRIS_CL, showfliers=True,
                    flierprops=dict(marker='o', ms=2, mfc=ORO, mec=ORO, alpha=0.4))
        ax[1, i].set(xlabel=v)
    fig.suptitle(f'Bloque {b}: las {len(vs)} de mayor asociación con el objetivo', x=0.01, ha='left',
                 fontweight='bold', color=VERDE, fontsize=11)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# #### Contrato

# %%
display(tabla_bloque('contrato'))
figura_bloque('contrato')

# %% [markdown]
# Se evidencia que el bloque de contrato no tiene colas largas, pero sí nulos estructurales:
# `meses_al_vencimiento` solo existe en término fijo y `meses_desde_aprendiz` solo en el 5 % que fue
# aprendiz. La variable más asociada es `meses_desde_cambio_contrato` (P(sup) 0,329), cuya mediana de
# 8 meses frente a una media de 50 refleja la mezcla de renovaciones recientes con indefinidos antiguos.
#
# #### Trayectoria

# %%
display(tabla_bloque('trayectoria'))
figura_bloque('trayectoria')

# %% [markdown]
# Se observa que los meses en la posición y en la función son las más asociadas del bloque y, como se
# verá en la sección bidimensional, repiten buena parte de la antigüedad. El resto son conteos de cola
# larga casi siempre en cero (ascensos, movimientos laterales, jefatura formal, personas a cargo,
# postulaciones internas). `meses_desde_ascenso` es nulo en el 91,5 % (quien no ha ascendido) y está
# topada en 12 meses, y `tamano_equipo_jefe` tiene un 36 % de nulos.
#
# #### Salario relativo

# %%
display(tabla_bloque('salario_relativo'))
figura_bloque('salario_relativo')

# %% [markdown]
# El bloque salarial comparte un 34-39 % de nulos estructurales: el posicionamiento local es nulo
# exactamente en las filas con salario a destajo o integral, que no tienen un sueldo básico comparable
# (el 86 % de las filas de banano). Siete de sus diez numéricas no binarias tienen cola larga. Las más
# asociadas con el objetivo son los aumentos por mérito en 24 meses (P(sup) 0,389, menores en quienes
# renuncian) y la privación relativa y el Gini del oficio y nivel (0,604 y 0,603).
#
# #### Jornada

# %%
with pd.option_context('display.max_rows', 60):
    display(tabla_bloque('jornada'))
figura_bloque('jornada')

# %% [markdown]
# Se evidencia que la jornada es el bloque más grande (27 variables) y el de más colas largas (12 de 22
# numéricas no binarias después del recorte): horas extra, recargos y dominicales tienen de 49 % a 79 %
# de ceros, y las marcas biométricas tienen un 43-46 % de nulos, que son exactamente las personas sin
# marcación biométrica. La más asociada es una frecuencia y no un monto: los meses con extras o
# recargos en 12 meses (P(sup) 0,373), con distribución en U y valores menores en quienes renuncian,
# en parte porque llevan menos tiempo.
#
# #### Ingreso relativo

# %%
display(tabla_bloque('ingreso_relativo'))
figura_bloque('ingreso_relativo')

# %% [markdown]
# Se observa que el ingreso frente al pactado es la única variable del bloque aproximadamente simétrica
# (asimetría 0,55) y la más asociada con el objetivo (P(sup) 0,368: quien renuncia gana menos por
# encima de lo pactado). El resto son bonos y auxilios casi siempre en cero, con asimetrías de 5 a 34,
# las colas más largas del libro; la volatilidad y el cambio del ingreso tienen nulos por falta de
# historia.
#
# #### Ausencias

# %%
display(tabla_bloque('ausencias'))
figura_bloque('ausencias')

# %% [markdown]
# El bloque de ausencias no tiene nulos (un mes sin licencia es un cero) y seis de sus siete
# variables tienen cola larga. La más asociada son los días de licencia no remunerada de los tres
# meses previos (P(sup) 0,585, mayores en quienes renuncian); el día de la familia es simétrico y
# quien renuncia lo ha tomado menos (0,392), lo que refleja en parte la antigüedad.
#
# #### Vacaciones

# %%
display(tabla_bloque('vacaciones'))
figura_bloque('vacaciones')

# %% [markdown]
# Se evidencia que la mitad de las filas no tiene vacaciones pendientes y que la única cola larga del
# bloque son las vacaciones compensadas en dinero. Los meses desde las últimas vacaciones están topados
# en 12 y son nulos en el 35 % de las filas, sobre todo personas nuevas que aún no las han disfrutado.
# Las pendientes se asocian negativamente con la renuncia (P(sup) 0,377), porque para tenerlas hay que
# haber cumplido un año.
#
# #### Proyectos personales

# %%
display(tabla_bloque('proyectos_personales'))
figura_bloque('proyectos_personales')

# %% [markdown]
# Son conteos pequeños con cola larga: el 98 % no retiró cesantías para educación en 12 meses. El
# retiro histórico de cesantías para vivienda es el más asociado (P(sup) 0,360), también con un
# componente de antigüedad.
#
# #### Origen

# %%
display(tabla_bloque('origen'))

# %% [markdown]
# Haber nacido en el departamento de la sede vale 1 en el 64 % de las filas con dato y tiene un 24 % de
# nulos.
#
# #### Colas largas y valores extremos en toda la historia

# %%
todas = pd.concat([tabla_bloque(b).assign(bloque=b) for b in BLOQUES])
continuas_hist = todas[todas.tipo != 'binaria']
cola = continuas_hist[continuas_hist['asimetría tras p1-p99'].abs() > 2]
print(f'numéricas no binarias de la historia: {len(continuas_hist)} | |asimetría| > 2 cruda: '
      f'{(continuas_hist["asimetría"].abs() > 2).sum()} | tras recorte p1-p99 (criterio del libro): {len(cola)} '
      f'| con curtosis (exceso) > 10: {(continuas_hist["curtosis (exceso)"] > 10).sum()}')
print(f'mediana del % de ceros en las de cola larga: {cola["% ceros"].median():.0f} %')
normal = [stats.normaltest(tr[v].dropna())[1] for v in continuas_hist.index]
print(f"D'Agostino rechaza la normalidad al 1 % en {sum(p < 0.01 for p in normal)} de {len(normal)}")
resumen_cola = (continuas_hist.assign(larga=continuas_hist['asimetría tras p1-p99'].abs() > 2,
                                     cruda=continuas_hist['asimetría'].abs() > 2)
                .groupby('bloque').agg(numéricas=('larga', 'size'), asim_cruda_mayor_2=('cruda', 'sum'),
                                       cola_larga_tras_recorte=('larga', 'sum'),
                                       mediana_pct_ceros=('% ceros', 'median'),
                                       mediana_pct_nulos=('% nulos', 'median')).round(1))
resumen_cola

# %% [markdown]
# De las 71 numéricas no binarias de la historia, 44 tienen \|asimetría\| > 2 en crudo y 39
# la conservan después del recorte p1-p99: esas 39 son las de cola larga del libro. Las cinco que el
# recorte normaliza lo suficiente son los tres recargos nocturnos (1, 3 y 12 meses), la volatilidad
# del ingreso y el cambio del ingreso de 3 frente a 9 meses: su asimetría venía de unos pocos
# extremos, no de la masa de ceros. Además, 33 tienen una curtosis de exceso mayor que 10. No es ruido
# de medición: son conteos y montos que valen cero para la mayoría (mediana de 73 % de ceros en las de
# cola larga) y mucho para unos pocos. La prueba de D'Agostino rechaza la normalidad en las 71, como
# se esperaba con este n, así que la prueba no discrimina y lo que informa es la forma. Las colas
# largas se concentran en jornada (12 de 22), salario relativo (7 de 10), ingreso relativo (6 de 11)
# y ausencias (6 de 7); contrato no tiene ninguna. Los atípicos por la regla de 1,5 × IQR son en su mayoría valores reales (quien hizo muchas
# horas extra, quien tuvo una licencia larga) y no se eliminan.
#
# ```{admonition} Decisión para el capítulo 7: recorte y logaritmo con signo
# :class: important
# Con estas colas, una logística sobre los valores crudos deja que unas pocas filas extremas fijen
# el coeficiente, y la estandarización no lo arregla (la desviación la inflan los mismos extremos).
# Por eso el preprocesamiento del modelo (1) recorta todas las numéricas no binarias a sus
# percentiles 1 y 99, estimados en el entrenamiento de cada pliegue, y (2) aplica a las que siguen
# con \|asimetría\| > 2 después del recorte el logaritmo con signo, $\operatorname{sign}(x)\log(1 + |x|)$,
# que conserva el cero y el signo (hay variables negativas, como la variación del sueldo en mínimos).
# En el entrenamiento completo son 39; la celda final del capítulo las lista. El `Pipeline` repite
# la regla dentro de cada pliegue, con sus propias filas de ajuste.
# ```

# %% [markdown]
# ### Variables categóricas
#
# #### Frecuencias

# %%
def personas_por_categoria(v):
    return tr.groupby(tr[v].fillna('(nulo)'))['persona_id'].nunique()


filas = []
for v in CAT:
    conteo = tr[v].fillna('(nulo)').value_counts()
    per = personas_por_categoria(v)
    filas.append({
        'variable': v, 'bloque': BLOQUE[v], 'tipo': tipo(v), 'categorías': len(conteo),
        '% nulos': round(100 * tr[v].isna().mean(), 1),
        'más frecuente': conteo.index[0], '% más frecuente': round(100 * conteo.iloc[0] / len(tr), 1),
        'cat. < 300 filas': int((conteo < 300).sum()),
        'filas en esas cat.': int(conteo[conteo < 300].sum()),
        'cat. < 5 personas': int((per < 5).sum()),
    })
cat_resumen = pd.DataFrame(filas).set_index('variable')
cat_resumen

# %% [markdown]
# La tabla completa de frecuencias absolutas y relativas de las 17 categóricas está plegada debajo.
# Las categorías con menos de cinco personas distintas se agrupan en "otras (agrupadas)", ampliada con
# las categorías más pequeñas hasta reunir al menos cinco personas.

# %% tags=["hide-output"]
frec = []
for v in CAT:
    x = agrupar(tr, v, min_ren=0)
    assert tr.groupby(x).persona_id.nunique().min() >= 5, v
    c = x.value_counts()
    frec.append(pd.DataFrame({'variable': v, 'categoría': c.index, 'filas': c.values,
                              '%': (100 * c.values / len(tr)).round(2),
                              '% acumulado': (100 * c.values.cumsum() / len(tr)).round(1)}))
with pd.option_context('display.max_rows', 400):
    display(pd.concat(frec).set_index(['variable', 'categoría']))

# %%
PRINCIPALES_CAT = ['linea', 'contrato', 'nivel', 'tipo_unidad', 'estado_civil', 'familia_cargo',
                   'tipo_salario', 'situacion_minimo', 'jornada_vs_pago']
fig, ax = plt.subplots(3, 3, figsize=(14, 10))
for a, v in zip(ax.ravel(), PRINCIPALES_CAT):
    conteo = agrupar(tr, v, min_ren=0).value_counts()
    barras = a.barh([rotulo(c) for c in conteo.index[::-1]], conteo.values[::-1], color=VERDE, **BORDE)
    etiquetar(a, barras, textos=[f'{n:,} ({100 * n / len(tr):.1f} %)' for n in conteo.values[::-1]],
              fontsize=6)
    a.set(title=nom(v).capitalize(), xlabel='Persona-mes')
    a.tick_params(axis='y', labelsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# Se evidencia que las categóricas son muy desiguales: la línea de palma tiene el 71 % de las filas, el
# personal operativo el 69 % y los hombres el 86,5 %. La ubicación es la de más categorías raras: 30
# de sus 57 tienen menos de 300 filas (2.469 filas en total) y 12 tienen menos de cinco personas. En
# `oficio` y `familia_cargo` solo hay una categoría pequeña (115 filas), porque los oficios poco
# frecuentes ya se agruparon en `otro_<familia>` al construir el panel. Dos categóricas de la historia tienen nulos grandes y
# estructurales: `situacion_minimo` (51,9 %: todo el destajo y el integral y parte del sueldo básico) y `jornada_vs_pago`
# (44,4 %, las personas sin marcación biométrica).
#
# **Tratamiento de las categorías raras.** Una categoría con menos de 300 filas tiene unas tres
# renuncias esperadas y su tasa no se puede estimar. En el `Pipeline` se agrupan en una categoría de
# infrecuentes (`OneHotEncoder(min_frequency=300, handle_unknown='infrequent_if_exist')`), que también
# recibe las categorías que aparezcan por primera vez en validación o en test. El nulo se trata como
# una categoría más, porque aquí casi siempre significa algo (no marca, no tiene sueldo básico, primer
# mes).

# %% [markdown]
# ## Análisis bidimensional
#
# ### Numérica frente a numérica
#
# Se toman como principales la edad, la antigüedad reconocida y las diez numéricas no binarias de la
# historia con mayor asociación con el objetivo. Para cada par se calculan Pearson (relación lineal)
# y Spearman (relación monótona, sobre rangos) con los pares completos. Cuando Spearman supera
# claramente a Pearson, la relación es monótona pero no lineal, o la dominan unos pocos valores extremos.
#
# La matriz completa de correlaciones y el VIF quedan para el capítulo 4.

# %%
TOP_HIST = sorted([v for v in CONTINUAS if v not in NUM_BASE], key=lambda v: -abs(PSUP[v] - 0.5))[:10]
PRINC = NUM_BASE + TOP_HIST
rp = tr[PRINC].corr(method='pearson')
rs = tr[PRINC].corr(method='spearman')
pares = []
for i, a in enumerate(PRINC):
    for b in PRINC[i + 1:]:
        pares.append({'variable 1': a, 'variable 2': b, 'Pearson': rp.loc[a, b], 'Spearman': rs.loc[a, b]})
pares = pd.DataFrame(pares)
pares['|P − S|'] = (pares.Pearson - pares.Spearman).abs()
pares['lectura'] = pd.cut(pares['|P − S|'], [-1, 0.1, 0.2, 2],
                          labels=['aprox. lineal', 'moderada', 'no lineal / atípicos'])
print(f'{len(pares)} pares | lectura: {pares.lectura.value_counts().to_dict()}')
print(f'pares con |Spearman| > 0,7: {(pares.Spearman.abs() > 0.7).sum()}')
pares.sort_values('|P − S|', ascending=False).head(15).round(3)

# %% [markdown]
# Se observa que de los 66 pares, 52 tienen una relación aproximadamente lineal (\|P − S\| < 0,1), 13
# una no linealidad moderada y uno es claramente no lineal: antigüedad reconocida y meses en la
# función (Pearson 0,64, Spearman 0,85). Casi siempre Spearman supera a Pearson: son relaciones
# monótonas pero curvas o con techo (los meses en la función no pasan de unos 90, la antigüedad sí).
# Nueve pares superan \|Spearman\| > 0,7, el umbral usual de redundancia entre predictoras, y
# todos giran alrededor del tiempo en la empresa: antigüedad, meses en la posición y en la función,
# meses desde el último cambio de contrato y retiros históricos de cesantías para vivienda. Buena
# parte de lo que las variables de la historia dicen del objetivo es, entonces, antigüedad con otro
# nombre; el capítulo 4 lo cuantifica con el VIF.

# %%
masfuertes = (pares[pares['variable 1'].map(BLOQUE) != pares['variable 2'].map(BLOQUE)]
              .assign(a=lambda d: d.Spearman.abs()).nlargest(2, 'a'))
parejas = [('edad', 'antig_meses')] + list(zip(masfuertes['variable 1'], masfuertes['variable 2']))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, (u, w) in zip(ax, parejas):
    d = tr[[u, w]].dropna()
    d = d[(d[u] <= d[u].quantile(0.99)) & (d[w] <= d[w].quantile(0.99))]
    hb = a.hexbin(d[u], d[w], gridsize=35, cmap='verde', mincnt=1, bins='log')
    a.set(title=textwrap.fill(f'{nom(u)} frente a {nom(w)}', 45), xlabel=u, ylabel=w)
    a.title.set_fontsize(9)
    fig.colorbar(hb, ax=a, label='Persona-mes (log)')
    a.text(0.02, 0.96, f'Pearson {rp.loc[u, w]:.2f} | Spearman {rs.loc[u, w]:.2f}',
           transform=a.transAxes, fontsize=7, va='top', color=TINTA)
plt.tight_layout()
plt.show()

# %% [markdown]
# Se observa la forma de cada nube. Edad y antigüedad forman un triángulo (la antigüedad está acotada por
# la edad) con correlación 0,61, igual en Pearson y Spearman. La antigüedad reconocida y los meses
# desde el último cambio de contrato son casi la misma variable en los indefinidos (la diagonal: su
# último cambio fue el ingreso), con Pearson de 0,96; la nube de abajo son los de término fijo, que
# renuevan cada pocos meses. Los meses en la función crecen con la antigüedad hasta un techo, lo que
# explica la brecha entre Pearson (0,64) y Spearman (0,85).
#
# ### Categórica frente a numérica
#
# Diferencias de las numéricas principales entre líneas de negocio, tipos de contrato, tipos de
# unidad, niveles y tipos de personal, con la prueba de Kruskal-Wallis (alternativa no paramétrica al
# ANOVA, apropiada porque las distribuciones son asimétricas). El tamaño del efecto es
# $\varepsilon^2 = H / (n - 1)$, la fracción de la variabilidad de los rangos que explican los grupos.
# Los valores p de las pruebas se corrigen por Holm y por Benjamini-Hochberg dentro de la familia.

# %%
GRUPOS = ['linea', 'contrato', 'tipo_unidad', 'nivel', 'tipo_costos']
NUM_KW = NUM_BASE + TOP_HIST[:4]
filas = []
for v in NUM_KW:
    for g in GRUPOS:
        d = tr[[v, g]].dropna()
        h, pv = stats.kruskal(*[x[v] for _, x in d.groupby(g)])
        filas.append({'numérica': v, 'grupo': g, 'n': len(d), 'H': h, 'p': pv, 'ε²': h / (len(d) - 1)})
kw = pd.DataFrame(filas)
kw['p Holm'], kw['p BH'] = ajustar(kw.p)
kw_tabla = kw.pivot(index='numérica', columns='grupo', values='ε²').loc[NUM_KW, GRUPOS].round(3)
print(f'{len(kw)} pruebas de Kruskal-Wallis | significativas con Holm al 5 %: {(kw["p Holm"] < 0.05).sum()}')
print('ε² por numérica (filas) y grupo (columnas):')
kw_tabla

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.violinplot(data=tr, x='linea', y='edad', ax=ax[0], cut=0, color='#A3B8A9', inner='quartile')
ax[0].set(title='Edad por línea de negocio', xlabel='', ylabel='Edad (años)')
sns.violinplot(data=tr, x='contrato', y='antig_meses', ax=ax[1], cut=0, color='#A3B8A9', inner='quartile')
ax[1].set(title='Antigüedad reconocida por tipo de contrato', xlabel='', ylabel='Meses')
v3 = TOP_HIST[0]
d = tr[tr[v3] <= tr[v3].quantile(0.99)]
sns.boxplot(data=d, x='nivel', y=v3, ax=ax[2], color=GRIS_CL, order=sorted(d.nivel.dropna().unique()),
            flierprops=dict(marker='o', ms=2, mfc=ORO, mec=ORO, alpha=0.4))
ax[2].set(title=f'{nom(v3)} por nivel (hasta el p99)', xlabel='', ylabel=v3)
ax[2].tick_params(axis='x', rotation=30, labelsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# Las 30 pruebas son significativas después de Holm, como se espera con 67.416 filas, y el
# tamaño del efecto separa dos situaciones. El tipo de contrato explica entre el 27 % (edad) y el
# 67 % (meses desde el último cambio de contrato) de la variabilidad de los rangos de las numéricas
# de tiempo, y el 61 % en la antigüedad reconocida: el término fijo agrupa sobre todo a personas
# nuevas y jóvenes, así que los efectos del contrato y de la antigüedad se superponen. Línea, tipo de
# unidad, nivel y tipo de personal explican todos menos del 3 % (ε² ≤ 0,028): diferencias muy
# significativas y de magnitud despreciable. Los violines lo muestran: la antigüedad del término fijo
# se concentra en los primeros meses y la del indefinido alrededor de siete a ocho años.
#
# ### Predictoras numéricas frente al objetivo
#
# #### Mann-Whitney, P(sup) y correlación punto-biserial
#
# Para las 88 numéricas y binarias se compara la distribución entre
# quienes renuncian y quienes no con la prueba de Mann-Whitney. Como tamaño del efecto se reporta la
# probabilidad de superioridad
#
# $$
# P(\text{sup}) = \frac{U_1}{n_1 n_0} = P(X_1 > X_0) + \tfrac{1}{2} P(X_1 = X_0) = \frac{r + 1}{2},
# $$
#
# donde $r$ es la correlación biserial de rangos. $P(\text{sup})$ es exactamente el AUC de la variable
# usada sola como puntaje: 0,5 es no discriminar, 0,6 significa que en el 60 % de los pares
# (renuncia, no renuncia) quien renuncia tiene el valor mayor. Se prefiere a los umbrales de Cohen
# (0,1 / 0,3 / 0,5 para $r$), que fueron pensados para otras escalas y aquí no significan nada
# operativo. Por debajo de 0,5 la variable es menor en quienes renuncian. Se añade la correlación
# punto-biserial $r_{pb}$ (Pearson entre la variable y el objetivo 0/1), sensible a los atípicos, y la
# tasa de renuncia entre las filas con la variable nula. Los 88 valores p se corrigen por Holm y por
# Benjamini-Hochberg.

# %%
filas = []
for v in NUMB:
    ps, pv, a, b = p_sup(v)
    nulo = tr[v].isna()
    r_nulo = int(tr.loc[nulo, OBJETIVO].sum())
    filas.append({
        'variable': v, 'bloque': BLOQUE[v], 'tipo': tipo(v), '% nulos': 100 * nulo.mean(),
        'mediana renuncia': a.median() if len(a) >= 5 else np.nan, 'mediana no renuncia': b.median(),
        'r biserial rangos': 2 * ps - 1, 'P(sup)': ps,
        'r punto-biserial': stats.pointbiserialr(tr.loc[~nulo, OBJETIVO], tr.loc[~nulo, v])[0],
        'p': pv,
        'tasa % si nulo': (100 * r_nulo / nulo.sum()) if nulo.sum() and r_nulo >= 5 else np.nan,
        # si el grupo nulo es pequeño, la tasa de los no nulos lo delataría por resta
        'tasa % si no nulo': (100 * tr.loc[~nulo, OBJETIVO].mean()
                              if (r_nulo >= 5 or not nulo.any()) and v not in CHICAS
                              else np.nan),
    })
mw = pd.DataFrame(filas).set_index('variable')
mw['p Holm'], mw['p BH'] = ajustar(mw.p)
mw.loc[CHICAS, ['P(sup)', 'r biserial rangos', 'r punto-biserial', 'p', 'p Holm', 'p BH']] = np.nan   # privacidad
mw['|P(sup) − 0,5|'] = (mw['P(sup)'] - 0.5).abs()
mw = mw.sort_values('|P(sup) − 0,5|', ascending=False)
print(f'{len(mw)} pruebas | significativas con Holm al 5 %: {(mw["p Holm"] < 0.05).sum()} | '
      f'con BH al 5 %: {(mw["p BH"] < 0.05).sum()}')
print(f'P(sup) fuera de [0,45; 0,55]: {(mw["|P(sup) − 0,5|"] > 0.05).sum()} | '
      f'fuera de [0,40; 0,60]: {(mw["|P(sup) − 0,5|"] > 0.10).sum()}')
col_mw = ['bloque', '% nulos', 'mediana renuncia', 'mediana no renuncia', 'P(sup)', 'r biserial rangos',
          'r punto-biserial', 'p Holm', 'p BH', 'tasa % si nulo', 'tasa % si no nulo']
mw[col_mw].head(20).round(3)

# %% [markdown]
# Se evidencia que la antigüedad reconocida es la numérica más discriminante (P(sup) 0,302: en el 70 % de
# los pares, quien renuncia lleva menos tiempo; mediana de 13 frente a 38 meses) y que las siguientes
# son también variables de tiempo: meses en la función (0,321), meses desde el último cambio de
# contrato (0,329), meses en la posición (0,330) y edad (0,339; mediana de 30 frente a 36 años).
# Después vienen señales que no son antigüedad: quien renuncia gana menos por encima de lo pactado
# (ingreso frente al pactado, 0,368), ha tenido menos meses con extras (0,373), menos aumentos por
# mérito (0,389), está en oficios y niveles con más desigualdad salarial (privación relativa 0,604) y
# ha tomado más días de licencia no remunerada en los tres meses previos (0,585).
#
# De las 88 pruebas, 43 son significativas con Holm y 56 con BH, pero solo 15 variables tienen un
# P(sup) fuera de [0,40; 0,60]: las demás, aunque significativas, discriminan poco solas. La
# correlación punto-biserial es pequeña en todas (\|r_pb\| ≤ 0,06) porque con un objetivo tan raro
# su máximo posible es bajo, y es sensible a las colas; por eso se ordena por P(sup). La tasa de las
# filas con nulo difiere de la del resto: 1,66 % frente a 1,00 % en las variables de nómina, 1,40 %
# frente a 0,78 % en las salariales y 0,63 % frente a 1,33 % en las de marcación; el mecanismo de esos
# faltantes se describe en el capítulo 1 («Faltantes»).
#
# La tabla con las 88 está plegada debajo. Las dos binarias con menos de cinco renuncias en el valor
# 1 (`es_jefe_formal`, `auxilio_educativo_12m`) no publican sus medidas de asociación, que permitirían
# reconstruir el conteo; lo mismo en la tabla de información mutua.

# %% tags=["hide-output"]
with pd.option_context('display.max_rows', 100):
    display(mw[col_mw].round(4))

# %% [markdown]
# #### Densidades por clase

# %%
KDE = NUM_BASE + TOP_HIST[:8]
fig, ax = plt.subplots(2, 5, figsize=(16, 6))
for a, v in zip(ax.ravel(), KDE):
    d = tr[[v, OBJETIVO]].dropna()
    lo, hi = d[v].quantile([0.01, 0.99])
    d = d[d[v].between(lo, hi)]
    sns.kdeplot(data=d, x=v, hue=OBJETIVO, common_norm=False, ax=a, palette=[VERDE, ORO],
                fill=True, alpha=0.25, cut=0, warn_singular=False)
    a.set(title=f'{textwrap.shorten(nom(v), 34, placeholder="...")}\nP(sup) {PSUP[v]:.3f} | r_pb {mw.loc[v, "r punto-biserial"]:.3f}',
          xlabel=v, ylabel='Densidad')
    a.title.set_fontsize(8)
    a.legend(['renuncia', 'no renuncia'], fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# En casi todas las densidades la masa de quienes renuncian (dorado) está desplazada hacia
# los valores bajos: menos edad, menos antigüedad, menos meses en la función y en la posición, menos
# vacaciones pendientes, menos meses con extras. Las formas no son normales ni parecidas entre
# clases: en la antigüedad, los meses en la posición y los meses desde el cambio de contrato, los que
# no renuncian tienen un segundo pico alrededor de 80-90 meses (la cohorte grande de indefinidos) que
# quienes renuncian casi no tienen. En los meses con extras la forma es de U en los que no renuncian
# y decreciente en los que renuncian. Una relación así no es lineal en la variable cruda, lo que
# apoya los tramos o el logaritmo del capítulo 7.
#
# #### Binarias frente al objetivo: razón de tasas

# %%
def razon_binaria(v):
    t = tasa(tr, tr[v])
    a1, n1_ = t.loc[1, 'renuncias'], t.loc[1, 'filas']
    a0, n0_ = t.loc[0, 'renuncias'], t.loc[0, 'filas']
    rr = (a1 / n1_) / (a0 / n0_)
    ee = np.sqrt(1 / a1 - 1 / n1_ + 1 / a0 - 1 / n0_) if a1 > 0 else np.nan
    return {'variable': v, 'bloque': BLOQUE[v], 'filas con 1': int(n1_), 'renuncias con 1': int(a1),
            'tasa % con 1': 100 * a1 / n1_, 'tasa % con 0': 100 * a0 / n0_, 'razón': rr,
            'IC bajo': rr * np.exp(-1.96 * ee), 'IC alto': rr * np.exp(1.96 * ee),
            'p': fisher_exact([[a1, n1_ - a1], [a0, n0_ - a0]])[1]}


BINARIAS = [v for v in NUMB if set(tr[v].dropna().unique()) <= {0, 1}]
rb = pd.DataFrame([razon_binaria(v) for v in BINARIAS]).set_index('variable')
rb['p Holm'], rb['p BH'] = ajustar(rb.p)
rb = rb.sort_values('razón', ascending=False)
chico = rb['renuncias con 1'] < 5
for v in rb.index[~chico]:
    n1_v = int(rb.loc[v, 'renuncias con 1'])
    verificar(f'binaria {v}', pd.DataFrame({'renuncias': [n1_v, int(tr.loc[tr[v] == 0, OBJETIVO].sum())]}),
              tr.loc[tr[v].notna(), OBJETIVO].sum())
# con el grupo 1 pequeño no se publica nada que permita reconstruirlo: ni sus filas ni la tasa del grupo 0
rb_pub = rb.round(3).astype(object)
rb_pub.loc[chico, ['filas con 1', 'renuncias con 1', 'tasa % con 1', 'tasa % con 0', 'razón', 'IC bajo',
                   'IC alto', 'p', 'p Holm', 'p BH']] = '—'
print(f'{len(rb)} binarias | significativas con Holm al 5 %: {(rb["p Holm"] < 0.05).sum()}')
rb_pub

# %% [markdown]
# Se evidencia que 5 de las 15 binarias difieren con Holm. Marcar en el biométrico duplica la tasa (razón
# 2,10; IC de 1,78 a 2,49), y hacer turnos largos sin cobrar extras también (2,08; IC de 1,73 a 2,50):
# es la señal de jornada más clara del capítulo. En sentido contrario, tener dos periodos de
# vacaciones acumulados divide la tasa por cinco (0,18; IC de 0,12 a 0,28), ganar la mediana local la
# reduce (0,58) y haber nacido en el departamento de la sede también (0,68; IC de 0,58 a 0,80). El
# primer mes del episodio tiene una razón de 1,66 que no resiste a Holm (p 0,15), y el traslado en
# 12 meses no difiere (1,03). Ser jefe formal y recibir auxilio educativo tienen menos de cinco
# renuncias con valor 1 y no se publican.
#
# ### Categórica frente a categórica
#
# #### Tablas de contingencia

# %%
ct = pd.crosstab(tr.linea, tr.contrato, margins=True, margins_name='total')
ct['% término fijo'] = (100 * ct['Termino Fijo'] / ct['total']).round(1)
display(ct)
pd.crosstab(tr.nivel, tr.tipo_costos, normalize='index').mul(100).round(1)

# %% [markdown]
# Se observa que el término fijo es el 52,9 % de las filas, pero su peso varía por línea: 57,9 % en palma
# y 31,8 % en banano. El tipo de personal casi determina el nivel en los extremos (el 93,9 % del nivel
# operativo es operativo directo), y en los niveles medios y altos se reparte entre indirecto y
# administrativo.
#
# #### Chi-cuadrado y V de Cramér entre las 17 categóricas
#
# $V = \sqrt{\chi^2 / (n \,(\min(r, c) - 1))}$ va de 0 (independencia) a 1 (una determina a la otra).
# El nulo cuenta como una categoría. Son 136 pares; los valores p se corrigen por Holm y por BH.

# %%
filas = []
for i, a in enumerate(CAT):
    for b in CAT[i + 1:]:
        t = pd.crosstab(tr[a].fillna('(nulo)'), tr[b].fillna('(nulo)')).to_numpy()
        chi, pv, gl, _ = chi2_contingency(t, correction=False)
        filas.append({'variable 1': a, 'variable 2': b, 'chi2': chi, 'gl': gl, 'p': pv,
                      'V de Cramér': np.sqrt(chi / (t.sum() * (min(t.shape) - 1)))})
cc = pd.DataFrame(filas)
cc['p Holm'], cc['p BH'] = ajustar(cc.p)
print(f'{len(cc)} pares | significativos con Holm al 5 %: {(cc["p Holm"] < 0.05).sum()} | '
      f'V > 0,5: {(cc["V de Cramér"] > 0.5).sum()} | V > 0,8: {(cc["V de Cramér"] > 0.8).sum()}')
cc.sort_values('V de Cramér', ascending=False).head(15).round(3)

# %% [markdown]
# Los 136 pares son significativos con Holm (el valor p no ordena nada con este n) y
# 30 tienen V > 0,5. Tres son casi redundancias exactas: la sociedad determina la línea (V = 1,00),
# la familia de cargo casi determina el oficio (0,995) y la línea casi determina la ubicación (0,88).
# El oficio también está muy asociado con el tipo de unidad, el área funcional, el tipo de personal,
# el tipo de salario y el estado de gestión de tiempos (V ≈ 0,70). Las variables de organización están
# anidadas unas en otras; en un modelo con one-hot eso da columnas colineales, que la regularización
# tiene que repartir (capítulos 4 y 7).
#
# ### Predictoras categóricas frente al objetivo
#
# Para cada categórica, chi-cuadrado y V de Cramér con el objetivo (con un objetivo binario
# $V = \sqrt{\chi^2 / n}$), corregidos por Holm y BH sobre las 17 pruebas. La tasa mínima y máxima se
# calculan solo entre las categorías publicables (al menos 300 filas y 5 renuncias).

# %%
filas = []
for v in CAT:
    x = tr[v].fillna('(nulo)')
    chi, pv, gl, _ = chi2_contingency(pd.crosstab(x, tr[OBJETIVO]).to_numpy(), correction=False)
    r = tasa(tr, x)
    pub = r[(r.filas >= 300) & (r.renuncias >= 5)]
    filas.append({
        'variable': v, 'bloque': BLOQUE[v], 'categorías': x.nunique(),
        'cat. con < 5 renuncias': int((r.renuncias < 5).sum()),
        'chi2': chi, 'gl': gl, 'p': pv, 'V de Cramér': np.sqrt(chi / len(tr)),
        'tasa mín % (publicable)': pub['tasa_%'].min(), 'tasa máx % (publicable)': pub['tasa_%'].max(),
    })
asoc = pd.DataFrame(filas).set_index('variable').sort_values('V de Cramér', ascending=False)
asoc['p Holm'], asoc['p BH'] = ajustar(asoc.p)
asoc.round(4)

# %% [markdown]
# Se evidencia que 15 de las 17 categóricas se asocian con el objetivo después de Holm; las dos que no son
# el género (V 0,001; p Holm 0,82) y el estado de gestión de tiempos (V 0,007). Las de mayor V son
# las de muchas categorías (oficio 0,075, ubicación 0,069, sociedad 0,063, tipo de unidad 0,056),
# en parte porque V crece con los grados de libertad; el contrato, con una sola, tiene V 0,052, y la
# línea 0,050. El nivel del cargo queda en el borde (p Holm 0,071). Entre las categorías publicables
# la tasa va de 0,25 % a 3,5 % en oficio y ubicación. Como siempre en este capítulo, con 67.416 filas
# es el tamaño del efecto, no el valor p, lo que permite comparar.

# %%
fig, ax = plt.subplots(3, 2, figsize=(12, 12))
for a, v in zip(ax.ravel(), ['contrato', 'linea', 'nivel', 'tipo_costos', 'tipo_salario',
                             'situacion_minimo']):
    grafico_tasa(a, publicable(tr, v), f'Tasa de renuncia por {nom(v)}', BASE)
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 9))
for a, v in zip(ax, ['familia_cargo', 'oficio']):
    grafico_tasa(a, publicable(tr, v), f'Tasa de renuncia por {nom(v)}', BASE)
plt.tight_layout()
plt.show()

# %% [markdown]
# Se observa en los gráficos que el término fijo (1,51 %) renuncia tres veces más que el indefinido
# (0,47 %); que banano (2,44 %) es la única línea claramente por encima de la media, y que el destajo
# (junto con el integral, que es pequeño: 1,43 %) supera al básico (0,80 %). En la situación frente al mínimo, quienes vieron su sueldo
# igualar o superar el aumento del mínimo renuncian poco (0,42 %). En oficio, los de banano
# (producción de banano, polinización) están arriba y el operario de campo general, abajo. Los
# gráficos agrupan las categorías con menos de 300 filas, cinco renuncias o cinco personas.
#
# #### Comparación entre niveles: razones de tasa
#
# Cada categoría se contrasta con una referencia, la más frecuente de su variable, mediante la
# razón de tasas
#
# $$
# RR = \frac{a / n}{a_0 / n_0}, \qquad
# \operatorname{IC}_{95\%} = RR \cdot \exp\!\left(\pm 1{,}96 \sqrt{\tfrac{1}{a} - \tfrac{1}{n}
#   + \tfrac{1}{a_0} - \tfrac{1}{n_0}}\right),
# $$
#
# donde *a* y *n* son las renuncias y las filas de la categoría, y $a_0$ y $n_0$ las de la referencia.
# Cada contraste se prueba con la prueba exacta de Fisher y los valores p se corrigen con Holm y con
# BH sobre todos los contrastes. Entran todas las categorías: las que tienen menos de 300 filas, menos
# de cinco renuncias o menos de cinco personas se contrastan juntas como "otras (agrupadas)".

# %%
def contraste(df, v):
    """Razón de tasas de cada categoría frente a la más frecuente, con IC y prueba de Fisher."""
    t = tasa(df, agrupar(df, v, min_filas=300)).sort_values('filas', ascending=False)
    verificar(f'razones de tasa de {v}', t, df[OBJETIVO].sum())
    agrupada = t.index.str.contains(' + ', regex=False) | (t.index == OTRAS)
    ref = t.index[~agrupada][0]          # la referencia es la categoría real más frecuente
    a0, n0_ = t.loc[ref, 'renuncias'], t.loc[ref, 'filas']
    filas = []
    for nivel, r in t.iterrows():
        a, n = r.renuncias, r.filas
        fila = {'variable': nom(v), 'categoría': nivel, 'filas': int(n),
                'renuncias': int(a), 'tasa %': 100 * a / n}
        if nivel == ref:
            fila.update({'razón': 1.0, 'IC bajo': np.nan, 'IC alto': np.nan, 'p': np.nan})
        else:
            rr = (a / n) / (a0 / n0_)
            ee = np.sqrt(1 / a - 1 / n + 1 / a0 - 1 / n0_) if a > 0 else np.nan
            fila.update({'razón': rr, 'IC bajo': rr * np.exp(-1.96 * ee),
                         'IC alto': rr * np.exp(1.96 * ee),
                         'p': fisher_exact([[a, n - a], [a0, n0_ - a0]])[1]})
        filas.append(fila)
    return pd.DataFrame(filas)


niveles = pd.concat([contraste(tr, v) for v in CAT], ignore_index=True)
con_p = niveles.p.notna()
niveles.loc[con_p, 'p Holm'], niveles.loc[con_p, 'p BH'] = ajustar(niveles.p[con_p])
niveles['referencia'] = np.where(con_p, '', 'ref.')
niveles = niveles.set_index(['variable', 'categoría'])
n_agrup = niveles.index.get_level_values(1).str.contains(' + ', regex=False).sum() +     (niveles.index.get_level_values(1) == OTRAS).sum()
print(f'{con_p.sum()} contrastes; significativos con Holm al 5 %: {(niveles["p Holm"] < 0.05).sum()}; '
      f'con BH: {(niveles["p BH"] < 0.05).sum()}; categorías agrupadas: {n_agrup}')

columnas = ['filas', 'renuncias', 'tasa %', 'razón', 'IC bajo', 'IC alto', 'p Holm', 'p BH', 'referencia']
niveles_pub = niveles.round(3).astype({c: object for c in ['IC bajo', 'IC alto', 'p Holm', 'p BH']})
niveles_pub.loc[niveles_pub.referencia == 'ref.', ['IC bajo', 'IC alto', 'p Holm', 'p BH']] = ''
POCAS = ['contrato', 'linea', 'tipo_unidad', 'tipo_costos', 'nivel', 'estado_civil', 'genero',
         'tipo_salario', 'situacion_minimo']
niveles_pub.loc[[nom(v) for v in POCAS], columnas]

# %% tags=["hide-output"]
with pd.option_context('display.max_rows', 300):
    display(niveles_pub.drop(index=[nom(v) for v in POCAS])[columnas])

# %% [markdown]
# De 134 contrastes, 34 son significativos con Holm y 69 con BH. En 11 variables hubo que agrupar
# categorías pequeñas (por ejemplo, el destajo con el integral, o "ya en el mínimo" con "absorbida").
# Los significativos se concentran en cuatro ejes:
#
# - **Contrato.** El indefinido renuncia a 0,31 veces la tasa del fijo (IC de 0,26 a 0,37): el
#   término fijo renuncia 3,2 veces más. Es la diferencia más precisa del capítulo.
# - **Banano.** La línea de banano renuncia 3,0 veces más que palma (IC de 2,5 a 3,6) y la finca de
#   banano 4,0 veces más que la de palma. En oficio, producción de banano (8,9) y polinización (6,6)
#   tienen las razones más altas frente al operario de campo general, que es de riesgo bajo (0,33 %) y
#   por eso casi todas las razones de oficio son mayores que uno. Es una misma señal repartida entre
#   variables anidadas. Industrial, transporte, puerto y ganadería no se distinguen de palma.
# - **Salario y jornada.** El destajo (con el integral) renuncia 1,8 veces más que el básico (IC de 1,5 a 2,1); frente
#   al nulo de la situación frente al mínimo (que incluye todo el destajo), quien quedó igual o por encima del mínimo
#   renuncia a 0,30 y quien ya estaba en el mínimo o lo vio absorbido a 0,42. Hacer turnos largos sin cobrarlos multiplica
#   por 3,6 la tasa de quien no marca.
# - **Estado civil.** La categoría "otro" (los estados poco frecuentes, agrupados) renuncia a 0,30 de
#   la unión libre.
#
# Otras diferencias no resisten la corrección: el nivel del cargo, el género (0,97; IC de 0,77 a
# 1,20), el personal administrativo (0,59, significativo solo con BH), el proceso de planta y casi
# todas las familias de cargo. En sociedad y ubicación hay varias categorías con razones de 0,2 a 4:
# son la misma heterogeneidad que la línea y el oficio, vista a otra escala. El modelo del capítulo 7
# estima estos contrastes ajustando por las demás variables.

# %% [markdown]
# ### Información mutua frente al azar, para las 105 predictoras
#
# La información mutua con el objetivo mide cuánto reduce una variable la incertidumbre sobre la
# renuncia:
#
# $$
# I(X; Y) = \sum_{x} \sum_{y} p(x, y) \log \frac{p(x, y)}{p(x)\, p(y)} ,
# $$
#
# que vale cero si y solo si *X* y *Y* son independientes y, a diferencia de una correlación, capta
# relaciones no lineales. Las numéricas con más de 20 valores se discretizan en 20 cuantiles; si un
# valor pesa más del 5 % (los ceros de los conteos) va en su propia celda y el resto se reparte en 19
# cuantiles, y el nulo es una celda más. La
# información mutua estimada crece con el número de categorías aunque no haya relación; por eso se
# compara con su distribución cuando el objetivo se permuta al azar (200 permutaciones, las mismas
# para todas las variables) y se reporta el **exceso** sobre la media de esa distribución, en % de la
# entropía del objetivo $H(Y)$. La prueba formal usa que $G = 2nI$ sigue una $\chi^2$ con $k - 1$
# grados de libertad bajo independencia (prueba G); sus valores p se corrigen por Holm y por BH.
# El valor p de permutación ($\geq 1/201$) se da como control.

# %%
def codigos(v):
    """Celdas de la variable: 20 cuantiles para las numéricas con muchos valores (el valor más
    frecuente va en su propia celda si pesa más del 5 %, como los ceros de los conteos), las
    categorías para las demás, y el nulo como una celda más."""
    x = tr[v]
    if v in NUM and x.dropna().nunique() > 20:
        moda = x.mode()[0]
        if (x == moda).mean() > 0.05:
            c = pd.Series(0.0, index=x.index).where(x.notna())
            resto = x[x.notna() & (x != moda)]
            c[resto.index] = pd.qcut(resto, 19, labels=False, duplicates='drop') + 1
        else:
            c = pd.qcut(x, 20, labels=False, duplicates='drop')
    else:
        c = tr[v].astype(str) if v in CAT else tr[v]
        c = pd.Series(pd.factorize(c)[0], index=tr.index).where(tr[v].notna())
    return pd.Series(c).fillna(-1).astype(int).to_numpy() + 1


def im(x, yy, k):
    t = np.bincount(x * 2 + yy, minlength=2 * k).reshape(k, 2) / len(yy)
    px, py = t.sum(1, keepdims=True), t.sum(0, keepdims=True)
    nz = t > 0
    return float((t[nz] * np.log(t[nz] / (px @ py)[nz])).sum())


rng = np.random.default_rng(SEMILLA)
PERM = [rng.permutation(y) for _ in range(200)]
H = -(y.mean() * np.log(y.mean()) + (1 - y.mean()) * np.log(1 - y.mean()))

filas = []
for v in PREDICTORAS:
    x = codigos(v)
    x = pd.factorize(x)[0]
    k = x.max() + 1
    obs = im(x, y, k)
    azar = np.array([im(x, yp, k) for yp in PERM])
    g = 2 * len(y) * obs
    filas.append({'variable': v, 'bloque': BLOQUE[v], 'celdas': k, 'IM observada': obs,
                  'IM por azar': azar.mean(), 'exceso': obs - azar.mean(),
                  '% de H(y)': 100 * (obs - azar.mean()) / H,
                  'z': (obs - azar.mean()) / azar.std(),
                  'p permutación': (1 + (azar >= obs).sum()) / (1 + len(azar)),
                  'p prueba G': stats.chi2.sf(g, k - 1)})
imt = pd.DataFrame(filas).set_index('variable').sort_values('exceso', ascending=False)
imt['p prueba G'] = imt['p prueba G'].fillna(1.0)          # variable constante en el entrenamiento: k = 1
imt['p Holm'], imt['p BH'] = ajustar(imt['p prueba G'])
imt.loc[CHICAS, ['IM observada', 'IM por azar', 'exceso', '% de H(y)', 'z', 'p permutación', 'p prueba G',
                 'p Holm', 'p BH']] = np.nan                                               # privacidad
print(f'H(y) = {H:.4f} nats | exceso > 0 en {(imt.exceso > 0).sum()} de {len(imt)} | '
      f'significativas (prueba G, Holm 5 %): {(imt["p Holm"] < 0.05).sum()} | con BH: {(imt["p BH"] < 0.05).sum()}')
col_im = ['bloque', 'celdas', 'IM observada', 'IM por azar', '% de H(y)', 'z', 'p permutación', 'p Holm', 'p BH']
imt[col_im].head(25).round({'IM observada': 5, 'IM por azar': 5, '% de H(y)': 2, 'z': 1,
                            'p permutación': 4, 'p Holm': 4, 'p BH': 4})

# %% [markdown]
# Se observa que la antigüedad reconocida es la predictora con más información (4,8 % de la entropía del
# objetivo una vez descontado el azar), seguida de los meses desde el último cambio de contrato
# (4,0 %), el oficio (4,0 %), los meses en la posición y en la función (3,7 %), la ubicación (3,4 %),
# la edad (3,1 %) y la sociedad (3,0 %). La corrección por azar importa: la ubicación tiene la mayor
# información por azar (57 celdas) y, sin corregir, estaría por delante de los meses en la posición.
# La información mutua capta lo que Mann-Whitney no ve: los meses al vencimiento tienen P(sup) 0,500
# entre quienes tienen dato, pero son la décima variable en información (2,8 %) porque su nulo es el
# contrato indefinido; lo mismo los meses desde las últimas vacaciones (P(sup) 0,458 y 2,8 %), cuyo
# nulo son personas nuevas con una tasa de 1,8 % frente a 0,6 %. Ninguna variable pasa del 5 % de la
# entropía: la renuncia no la explica una variable sola, y los porcentajes no se suman porque las
# variables comparten información. Con la prueba G, 74 de las 105 son significativas con Holm.
#
# Las 105 predictoras, ordenadas, están plegadas debajo.

# %% tags=["hide-output"]
with pd.option_context('display.max_rows', 120):
    display(imt[col_im].round({'IM observada': 5, 'IM por azar': 5, '% de H(y)': 2, 'z': 1,
                               'p permutación': 4, 'p Holm': 4, 'p BH': 4}))

# %% [markdown]
# Todos los bloques tienen señal por encima del azar (tabla de la síntesis); la de trayectoria y
# contrato es en buena parte antigüedad, mientras que ausencias y origen aportan variables que no son
# tiempo (licencia no remunerada reciente, nacido en el departamento de la sede).
#
# ### Tasa de renuncia por antigüedad reconocida (*hazard*)
#
# La tasa por mes de antigüedad es la función de riesgo discreta (*hazard*): de quienes llegan al mes
# *k* de antigüedad, la fracción que renuncia en ese mes, $\hat h_k = d_k / n_k$, con $d_k$ renuncias
# entre las $n_k$ persona-mes que están en el mes *k*. La antigüedad es la **reconocida** por la
# empresa: cuenta desde la fecha de antigüedad del sistema de nómina y no se reinicia con un traslado
# entre sociedades del grupo. No es lo mismo que los
# meses del episodio en la sociedad actual: las dos difieren en cerca del 10 % de las filas, casi
# todas de personas trasladadas. Se calcula por separado para cada tipo de contrato; un tramo con
# menos de cinco renuncias o cinco personas (y, en el gráfico, menos de 300 persona-mes) se une con el
# siguiente.

# %%
ORDEN_TRAMOS = ['0-2', '3-5', '6-11', '12-23', '24-35', '36-59', '60-119', '120+']
tramos = pd.cut(tr.antig_meses, [-1, 2, 5, 11, 23, 35, 59, 119, 1e4], labels=ORDEN_TRAMOS).astype(str)


def tabla_tramos(df, tramo, orden, **minimos):
    """Tasa por tramo dentro de cada contrato, con los tramos pequeños unidos al vecino."""
    partes = []
    for contrato, s in df.groupby('contrato'):
        mapa = fusionar(s, tramo[s.index], orden, **minimos)
        grupo = tramo[s.index].map(lambda k: etiqueta_tramo(mapa[k]))
        orden_g = list(dict.fromkeys(etiqueta_tramo(mapa[k]) for k in orden if k in mapa))
        t = tasa_ic(s, grupo).reindex(orden_g)
        verificar(f'tramos de antigüedad, {contrato}', t, s[OBJETIVO].sum())
        partes.append(t.assign(contrato=contrato))
    return pd.concat(partes).rename_axis('tramo (meses)').reset_index().set_index(['contrato', 'tramo (meses)'])


display(tabla_tramos(tr, tramos, ORDEN_TRAMOS))

fig, ax = plt.subplots(figsize=(11, 3.8))
for contrato, color in [('Termino Fijo', ORO), ('Termino Indefinido', VERDE)]:
    s = tr[tr.contrato == contrato]
    tramo = (s.antig_meses // 3 * 3).clip(upper=60)
    mapa = fusionar(s, tramo, sorted(tramo.unique()), min_filas=300)
    t = tasa_ic(s, tramo.map(lambda k: mapa[k][0]))      # cada punto va en el inicio de su tramo
    verificar(f'curva de riesgo, {contrato}', t, s[OBJETIVO].sum())
    ax.plot(t.index, t['tasa_%'], marker='o', ms=3, label=contrato, color=color)
    ax.fill_between(t.index, t.ic_bajo, t.ic_alto, alpha=0.15, color=color)
ax.axhline(BASE, ls='--', c=GRIS, lw=0.8)
ax.set(title='Tasa de renuncia por antigüedad reconocida (tramos de 3 meses; 60 = 60 o más)',
       xlabel='Antigüedad reconocida (meses)', ylabel='Tasa mensual (%)')
ax.legend(title='Contrato')
plt.tight_layout()
plt.show()

# %% [markdown]
# Se evidencia que en el término fijo la tasa se mantiene alrededor de 2,0-2,3 % mensual durante el primer
# año (0-2, 3-5 y 6-11 meses), baja a 1,28 % en el segundo año, a 1,10 % en el tercero y a 0,70 % entre
# los 3 y 5 años (en 60 meses o más, que se publica junto, 0,32 %). En el indefinido casi no hay
# personas nuevas, y sus primeros 24 meses se publican juntos porque por separado tendrían menos de
# cinco renuncias; en los tramos comparables la tasa del indefinido no es menor que la del fijo: 1,62 %
# en sus primeros dos años (frente a 1,28 % del fijo entre 12 y 23 meses) y 1,39 % frente a 1,10 %
# entre 24 y 35. Desde los 3 años las dos se parecen (0,70 % y 0,73 %) y en los 10 años o más el
# indefinido baja a 0,20 %. Buena parte de la brecha entre contratos es, por tanto, antigüedad: el riesgo cae con el
# tiempo en la empresa, y el fijo agrupa a los nuevos. El riesgo no es monótono al inicio (meseta en
# el primer año) y cae con fuerza después, lo que una transformación logarítmica aproxima y unos
# tramos capturan mejor.
#
# ### Interacciones
#
# #### Contrato por línea de negocio

# %%
g = tr.groupby(['linea', 'contrato'])[OBJETIVO].agg(filas='size', renuncias='sum')
g['tasa_%'] = 100 * g.renuncias / g.filas
tasa_lc = g['tasa_%'].unstack()
ok = ((g.renuncias >= 5) & (g.filas >= 300)).unstack()
ok = ok.apply(lambda r: r & r.all(), axis=1)     # si una celda de la línea no se publica, la otra tampoco:
                                                 # el total de la línea es público y la delataría por resta
anot = np.where(ok, tasa_lc.round(2).astype(str) + ' %', '')
for linea in ok.index[ok.all(axis=1)]:
    verificar(f'línea x contrato, {linea}', g.loc[linea], tr.loc[tr.linea == linea, OBJETIVO].sum())

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.heatmap(tasa_lc.where(ok), annot=anot, fmt='', cmap='oro', ax=ax[0], cbar=False,
            linewidths=0.5, linecolor='white')
ax[0].set(title='Tasa de renuncia por línea y contrato\n(en blanco: la línea tiene una celda con < 5 renuncias)',
          xlabel='', ylabel='')

filas = []
for linea, s in tr.groupby('linea'):
    a = s[s.contrato == 'Termino Fijo'][OBJETIVO]
    b = s[s.contrato == 'Termino Indefinido'][OBJETIVO]
    if a.sum() >= 5 and b.sum() >= 5:
        rr = a.mean() / b.mean()
        ee = np.sqrt(1 / a.sum() - 1 / len(a) + 1 / b.sum() - 1 / len(b))
        filas.append({'línea': linea, 'razón fijo / indefinido': rr, 'IC bajo': rr * np.exp(-1.96 * ee),
                      'IC alto': rr * np.exp(1.96 * ee)})
    else:
        filas.append({'línea': linea, 'razón fijo / indefinido': np.nan, 'IC bajo': np.nan, 'IC alto': np.nan})
rr_linea = pd.DataFrame(filas).set_index('línea')
d = rr_linea.dropna().sort_values('razón fijo / indefinido')
yy = range(len(d))
ax[1].errorbar(d['razón fijo / indefinido'], yy, xerr=[d['razón fijo / indefinido'] - d['IC bajo'],
               d['IC alto'] - d['razón fijo / indefinido']], fmt='o', color=VERDE, ecolor=TINTA, lw=0.8)
ax[1].axvline(1, ls='--', c=GRIS, lw=0.8)
ax[1].set_yticks(list(yy))
ax[1].set_yticklabels([rotulo(c) for c in d.index])
ax[1].set_xscale('log')
eje_llano(ax[1].xaxis)
ax[1].set(title='Razón de tasas término fijo / indefinido, por línea (IC 95 %)', xlabel='Razón (escala log)')
plt.tight_layout()
plt.show()
rr_linea.round(2)

# %% [markdown]
# Se observa que el efecto del contrato depende de la línea: el término fijo multiplica la tasa por 6,1 en
# palma (1,25 % frente a 0,20 %), por 3,9 en industrial, por 3,2 en banano y por 2,0 en transporte,
# mientras que en puerto no hay diferencia (0,84 % frente a 0,86 %; razón 0,99, IC de 0,31 a 3,1). El
# grupo de mayor riesgo es banano con término fijo, con 4,6 % mensual, 4,5 veces la media. Un modelo
# aditivo en la escala logit supone la misma razón en todas las líneas; los intervalos de palma (4,5
# a 8,5) y de transporte (1,1 a 3,7) no se solapan, así que la interacción es real. En ganadería una
# de las dos celdas tiene menos de cinco renuncias; como el total de la línea es público, no se
# publica ninguna de las dos.
#
# #### Antigüedad reconocida por contrato, y la variable más asociada de la historia

# %%
v_int = TOP_HIST[0]
tramo3 = pd.cut(tr.antig_meses, [-1, 11, 35, 1e4], labels=['0-11', '12-35', '36+']).astype(str)
display(tabla_tramos(tr, tramo3, ['0-11', '12-35', '36+'])[['filas', 'renuncias', 'tasa_%']])
tercil = pd.qcut(tr[v_int], 3, duplicates='drop', labels=['bajo', 'medio', 'alto'])
t_int2 = tasa(tr, [tercil, tr.contrato])
for contrato in t_int2.index.get_level_values(1).unique():
    verificar(f'tercil x contrato, {contrato}', t_int2.xs(contrato, level=1),
              tr.loc[tr.contrato == contrato, OBJETIVO].sum())
t_int2.rename_axis([f'tercil de {v_int}', 'contrato']).round(2)

# %% [markdown]
# Se observa la interacción de antigüedad y contrato en la primera tabla: con menos de un año casi todo es
# término fijo (2,14 %); el indefinido con menos de 3 años (publicado en un solo tramo, porque su
# primer año tiene muy pocas renuncias) renuncia a 1,47 %, más que el fijo entre 1 y 3 años (1,21 %),
# y desde los 3 años el fijo vuelve a estar por encima (0,55 % frente a 0,37 %). El orden de
# los contratos se invierte según la antigüedad, algo que ningún modelo aditivo reproduce. La segunda
# tabla cruza el contrato con los terciles de los meses en la función, la variable de la historia más
# asociada: en el tercil bajo el fijo triplica al indefinido (1,90 % frente a 0,64 %), en el medio la
# brecha se reduce (1,06 % frente a 0,70 %) y en el alto desaparece (0,32 % frente a 0,33 %). De nuevo
# el efecto del contrato es sobre todo un efecto de ser nuevo.
#
# #### Género y familia de cargo

# %%
fam = tr.groupby('familia_cargo').agg(filas=('genero', 'size'),
                                      mujeres=('genero', lambda s: 100 * (s == 'Femenino').mean()))
fam['personas mujeres'] = tr[tr.genero == 'Femenino'].groupby('familia_cargo').persona_id.nunique()
fam['personas hombres'] = tr[tr.genero == 'Masculino'].groupby('familia_cargo').persona_id.nunique()
fam = fam[(fam.filas >= 300) & (fam['personas mujeres'].fillna(0) >= 5) & (fam['personas hombres'].fillna(0) >= 5)]
fam = fam.sort_values('mujeres')
fig, ax = plt.subplots(figsize=(7, 4))
barras = ax.barh([rotulo(c) for c in fam.index], fam.mujeres, color=VERDE, **BORDE)
etiquetar(ax, barras, '{:.0f} %')
ax.set(title='Porcentaje de mujeres por familia de cargo\n(familias con >= 300 filas y >= 5 personas de cada género)',
       xlabel='Mujeres (% de persona-mes)', ylabel='')
plt.tight_layout()
plt.show()
print(f'familias graficadas: {len(fam)} de {tr.familia_cargo.nunique()}')
print('tasa por género (%):', tasa(tr, 'genero')['tasa_%'].round(2).to_dict())

# %% [markdown]
# Las mujeres se concentran en las familias administrativas (74 % en finanzas y
# administración, 65 % en comercial y compras) y casi no están en las operativas (3 % a 9 % en
# mantenimiento técnico, transporte y campo). La tasa por género es prácticamente igual (0,99 % en
# mujeres y 1,03 % en hombres) y el género no aporta información por sí solo (V de Cramér 0,001).
# Como las mujeres están en familias de riesgo bajo, esa igualdad bruta no descarta una diferencia
# dentro de cada familia; el modelo, que ajusta por familia y oficio, lo decide. Se grafican 13 de las
# 17 familias con dato: las demás tienen menos de 300 filas o menos de cinco personas de algún
# género.

# %% [markdown]
# ## Síntesis
#
# La tabla reúne, para las tres variables con más información mutua de cada bloque, las medidas de
# asociación del capítulo; debajo se imprime la lista de las variables de cola larga que recibirán el
# logaritmo con signo en el capítulo 7.

# %%
sintesis = pd.DataFrame({
    'bloque': [BLOQUE[v] for v in PREDICTORAS],
    'P(sup)': [mw['P(sup)'].get(v, np.nan) for v in PREDICTORAS],
    'V de Cramér': [asoc['V de Cramér'].get(v, np.nan) for v in PREDICTORAS],
    'IM % de H(y)': [imt.loc[v, '% de H(y)'] for v in PREDICTORAS],
    'asimetría cruda': [tr[v].skew() if v in CONTINUAS else np.nan for v in PREDICTORAS],
    'asimetría tras p1-p99': [asim_recortada(tr[v]) if v in CONTINUAS else np.nan for v in PREDICTORAS],
    '% nulos': [100 * tr[v].isna().mean() for v in PREDICTORAS],
}, index=pd.Index(PREDICTORAS, name='variable'))
sintesis['cola larga'] = np.where(sintesis['asimetría tras p1-p99'].abs() > 2, 'sí', '')
sintesis['rango IM'] = sintesis['IM % de H(y)'].rank(ascending=False).astype('Int64')
top_sint = (sintesis.sort_values('rango IM').groupby('bloque').head(3)
            .sort_values(['bloque', 'rango IM']))
print(f'cola larga (|asimetría| > 2 tras recorte p1-p99): {(sintesis["cola larga"] == "sí").sum()} | '
      f'con la asimetría cruda serían {(sintesis["asimetría cruda"].abs() > 2).sum()} | '
      f'con nulos: {(sintesis["% nulos"] > 0).sum()} '
      f'({sum((sintesis["% nulos"] > 0) & ~sintesis.index.isin(CAT))} numéricas y '
      f'{sum((sintesis["% nulos"] > 0) & sintesis.index.isin(CAT))} categóricas)')
COLA_LARGA = sorted(sintesis.index[sintesis['cola larga'] == 'sí'])
print('cola larga:', ', '.join(COLA_LARGA))
top_sint.round(3)

# %% [markdown]
# **Conclusiones principales:**
#
# - **Antigüedad.** La antigüedad reconocida es la señal principal (P(sup) 0,302; 4,8 % de la
#   entropía del objetivo), y buena parte de la señal de contrato y trayectoria es antigüedad con otro
#   nombre (Spearman > 0,7 entre las variables de tiempo); el riesgo tiene una meseta el primer año y
#   luego cae, por lo que se representa en logaritmo o por tramos.
# - **Contrato.** El término fijo renuncia 3,2 veces más que el indefinido, sobre todo porque agrupa a
#   las personas nuevas.
# - **Interacciones.** La razón fijo/indefinido va de 0,99 en puerto a 6,1 en palma, y el orden de los
#   contratos se invierte entre 1 y 3 años de antigüedad; se evalúan como términos explícitos en la
#   validación del capítulo 7.
# - **Señales que no son tiempo.** Licencia no remunerada reciente (0,585), ingreso frente al pactado
#   (0,368), meses con extras (0,373), turnos largos sin pago (razón 2,1) y destajo (1,8 veces el
#   básico).
# - **Género.** No aporta información por sí solo (V de Cramér 0,001).
# - **Colas largas.** 39 numéricas conservan \|asimetría\| > 2 tras el recorte p1-p99 y reciben el
#   logaritmo con signo, según el recuadro de la sección de colas largas.
# - **Nulos.** Son estructurales e informativos (capítulo 1): indicador de faltante en las numéricas y
#   categoría propia en las categóricas. La cobertura de `tamano_equipo_jefe` cambia en el tiempo, por
#   lo que su coeficiente debe revisarse entre pliegues.
# - **Redundancias.** Sociedad y línea, familia y oficio y las variables de tiempo se dejan a la
#   regularización; el capítulo 4 las cuantifica con el VIF.

# %% tags=["remove-cell"]
# Control de privacidad (no se publica): cada tabla o gráfico por grupo pasó por `verificar`, que exige
# al menos 5 renuncias en toda celda publicada y que las celdas sumen el total, de modo que ninguna
# celda pequeña se pueda obtener por resta. Si alguna fallara, la compilación se habría detenido.
control = pd.DataFrame(CONTROL)
assert len(control) and (control['mínimo publicado'] >= 5).all() and (control.suma == control.total).all()
print(f'{len(control)} tablas o gráficos controlados; mínimo publicado: {control["mínimo publicado"].min()}')
