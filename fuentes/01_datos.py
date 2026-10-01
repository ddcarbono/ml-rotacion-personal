# %% [markdown]
# # Datos
#
# ```{admonition} Alcance
# :class: tip
# Este capítulo describe el panel completo: tamaño, estructura, diccionario y conteos de faltantes.
# Todo lo que depende del objetivo o fija un umbral (tasas por patrón de faltante, vallas de valores atípicos (*outliers*),
# eventos por parámetro) se calcula solo con el entrenamiento (enero de 2025 a abril de 2026), con la
# partición del capítulo 2 (`comun.particion`).
# ```
#
# ## Problema de investigación
#
# ¿Se puede estimar, con la información disponible el día 1 de cada mes, la probabilidad de que un
# trabajador renuncie voluntariamente durante ese mes? Es un problema de clasificación binaria sobre
# un panel persona-mes, con un evento raro (alrededor del 1 % de los meses) y una dimensión temporal:
# el modelo se entrena con meses pasados y se evalúa sobre meses futuros.
#
# ## Justificación
#
# La rotación voluntaria tiene un costo directo (selección, inducción, curva de aprendizaje) y se
# concentra en las operaciones de campo y de planta. Un puntaje mensual permite ordenar a quién mirar
# primero. Además, el conjunto reúne varias dificultades de modelado: un evento raro, un panel con
# entidades repetidas, una dimensión temporal, fugas de datos (*data leakage*) posibles en la fuente, categóricas
# de alta cardinalidad y 87 variables de historia laboral con faltantes estructurales.
#
# ## Fuente y licencia
#
# Los datos provienen del sistema de nómina y de gestión humana de un grupo agroindustrial colombiano
# (líneas de palma, banano, industrial, transporte, puerto y ganadería): registro de personal y de
# estados del trabajador, historia de contratos, sueldos pactados, nómina pagada, registro de tiempos,
# marcaciones biométricas, registro de novedades (licencias, capacitación, compensatorios),
# vacaciones, postulaciones internas y solicitudes de cesantías.
#
# **Licencia:** uso académico autorizado por la empresa, no redistribuible. No hay enlace público. La
# base se entrega a la universidad como CSV anonimizado por un canal privado, no en el repositorio;
# el libro lee su ruta de la variable de entorno `PANEL_ROTACION` (`comun.py`).
#
# ## Construcción del panel
#
# - **Unidad:** persona × mes. La persona se identifica por su documento, normalizado, y no por
#   registro de personal, porque una misma persona puede tener varios a lo largo del tiempo (traslados
#   entre sociedades, reingresos). El documento nunca sale de la extracción: se reemplaza por un
#   entero asignado al azar.
# - **Pertenencia al mes:** la persona está en el panel en el mes *t* si el registro de estados del
#   trabajador la muestra activa algún día de ese mes, y no según una foto del día 1, que perdería a
#   quien sale entre el 28 y el 31 (recuadro de abajo).
# - **Episodios laborales:** una ausencia de al menos un mes abre un episodio nuevo. Un traslado entre
#   sociedades sin pausa no lo abre.
# - **Objetivo:** `y_renuncia = 1` en el último mes de un episodio que termina por renuncia
#   voluntaria. Las otras salidas (despido, fin de contrato, pensión, fallecimiento, mutuo acuerdo)
#   valen 0 en su último mes, y después la persona sale del panel: estuvo en riesgo y no renunció
#   (riesgos competitivos). Una renuncia registrada cuando ya existía una decisión de no renovar el
#   contrato cuenta como no renovación, porque la empresa ya había decidido terminarlo (capítulo 2).
# - **Renuncia administrativa:** renuncia registrada tras la cual la persona sigue al mes siguiente
#   con otro registro de personal (un traslado). No es una salida (`y = 0`).
# - **Exclusiones:** aprendices, vicepresidencia y presidencia (pocas personas, identificables),
#   contratos de obra o labor, el mes de salida de los casos con etiqueta dudosa (salida sin registro,
#   terminación en periodo de prueba) y las personas con una edad al ingreso
#   fuera de 18 a 70 años, que indica una fecha de nacimiento mal registrada.
# - **Momento de medición:** los atributos (las 18 variables que describen al trabajador y su puesto:
#   demografía, contrato, cargo y ubicación) corresponden al día 1 del mes *t* (o al primer día activo,
#   si la persona ingresa a mitad de mes). El cargo entra con un mes de rezago, porque se vacía en el
#   mes de salida (capítulo 6). La nómina, las marcaciones y las novedades entran con los cierres
#   hasta *t − 1*, que son los conocidos el día 1 de *t*; la licencia no remunerada y las vacaciones
#   tomadas, con dos meses de rezago. La columna `disponible` del diccionario lo dice variable por
#   variable.
# - **Puesto:** los textos de cargo, función, departamento y centro de costo (cientos de valores) se
#   descomponen en ejes sin repetición: `nivel`, `familia_cargo`, `oficio`, `tipo_unidad`,
#   `area_funcional` y `proceso_planta`. Los textos originales no están en el panel.
# - **Antigüedad reconocida:** `antig_meses` es la antigüedad que reconoce la empresa (la fecha de
#   antigüedad del sistema de nómina), con continuidad en los traslados entre sociedades; una salida
#   real seguida de recontratación sí la reinicia.
#
# ```{admonition} Efecto de las reglas de construcción
# :class: note
# - **Preaviso de no renovación:** 85 renuncias registradas con un preaviso previo cuentan como no
#   renovación y no como renuncia; 19 de ellas caen en el test (capítulo 2).
# - **Salidas de fin de mes:** medir la pertenencia por la actividad del mes, y no por la foto del día
#   1, captura 30 renuncias con retiro entre el 28 y el 31 (6 en el test), que de otro modo no
#   tendrían mes de salida.
# - **Edad al ingreso:** la regla de 18 a 70 años excluye 7 personas (121 filas, ninguna renuncia).
# - **Códigos:** sociedad (S01 a S22) y ubicación (U001 a U057) se asignan al azar, no por tamaño.
#   `proceso_planta` queda en cinco grupos.
# - **Variables:** a los 18 atributos se suman 92 columnas de historia laboral en nueve bloques; cinco
#   se descartan (sección de duplicados y valores inconsistentes) y quedan 87.
# ```

# %%
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from comun import (cargar, particion, estilo, tasa_ic, OBJETIVO, NUM, BIN, CAT, CAT_BASE, NUM_BASE,
                   PREDICTORAS, FUERA, BLOQUE, BLOQUES, BASE, HISTORIA, SEMILLA, VERDE, ORO, GRIS, BORDE)

estilo()
pd.set_option('display.width', 160)
pd.set_option('display.max_colwidth', 110)
pd.set_option('display.max_rows', 200)

p = cargar()
tr, te = particion(p)
n_ren = int(p[OBJETIVO].sum())
n_no = int((p[OBJETIVO] == 0).sum())
print(f'filas: {len(p):,} | personas: {p.persona_id.nunique():,} | meses: {p.mes.nunique()} '
      f'({p.mes.min()} a {p.mes.max()}) | columnas: {p.shape[1]}')
print(f'predictoras: {len(PREDICTORAS)} ({len(BASE)} atributos + {len(HISTORIA)} de historia laboral) | '
      f'candidatas antes de la revisión de calidad: {len(PREDICTORAS) + len(FUERA)}')
print(f'clase positiva: {n_ren} renuncias frente a {n_no:,} no renuncias (1 a {n_no / n_ren:.0f}); '
      f'prevalencia {100 * p[OBJETIVO].mean():.2f} %')
print('evento:', p.evento.value_counts().to_dict())

# %% [markdown]
# El panel tiene 85.068 filas de 5.738 personas en 20 meses, y 115 columnas: el identificador, el
# mes, 110 candidatas a predictora (cinco se descartan en la revisión de calidad, más abajo), una columna que solo sirve para el
# análisis de sensibilidad del objetivo (`preaviso_no_renovacion`) y dos que describen el desenlace (`evento` y el objetivo). Hay una renuncia por cada 101 meses
# sin renuncia. Las 36 renuncias administrativas son traslados y no cuentan como salida.
#
# ### Por qué una fila por persona y mes
#
# El panel persona-mes es la forma habitual de un modelo de riesgo en tiempo discreto (Allison, 1982;
# Singer y Willett, 2003). Para la persona *i* en el mes *t*, el riesgo es la probabilidad de renunciar
# en ese mes dado que sigue activa al inicio:
#
# $$
# h_{it} = P(y_{it} = 1 \mid \text{activa al inicio de } t,\ x_{it}),
# \qquad \operatorname{logit} h_{it} = x_{it}^\top \beta .
# $$
#
# La verosimilitud del panel es
#
# $$
# L(\beta) = \prod_{i} \prod_{t \in R_i} h_{it}^{\,y_{it}} \, (1 - h_{it})^{1 - y_{it}},
# $$
#
# donde $R_i$ son los meses en que la persona *i* estuvo en riesgo. Es la misma verosimilitud de un
# modelo de supervivencia con censura: quien sigue activa al final de la ventana aporta sus meses sin
# renuncia, y quien sale por otra causa deja de aportar desde su salida. Como cada episodio tiene a lo
# sumo una renuncia, el producto sobre sus meses es la probabilidad de su historia completa, y que una
# persona aporte varias filas no sesga la estimación. Para medir la incertidumbre del modelo se
# remuestrean personas completas, con todos sus meses (capítulo 7). La probabilidad de seguir activa *k* meses es
# $S_{ik} = \prod_{t=1}^{k} (1 - h_{it})$.
#
# Se prefiere a una fila por persona, con la etiqueta "renunció o no", por cuatro razones:
#
# 1. **Tiempo de exposición.** Las personas se observan entre 1 y 20 meses. Con una fila por persona,
#    quien estuvo 3 meses tuvo menos oportunidad de renunciar que quien estuvo 20, y la etiqueta mezcla
#    el riesgo con el tiempo de observación. Con una tasa de 1 % mensual, la probabilidad de renunciar
#    en 3 meses es 1 − 0,99³ ≈ 3 % y en 16 meses 1 − 0,99¹⁶ ≈ 15 %, sin que el riesgo cambie. En el
#    panel cada fila vale un mes.
# 2. **Variables que cambian.** Edad, antigüedad, contrato, sueldo relativo, jornada y vacaciones
#    pendientes cambian de un mes a otro. Una fila por persona obliga a elegir un momento y descarta
#    el resto.
# 3. **Una misma referencia temporal.** Si los activos se describen en el último mes y los que renunciaron en su mes
#    de salida, las dos poblaciones se miden en momentos distintos, y quien se fue no pudo seguir
#    acumulando antigüedad: "los que renuncian tienen menos antigüedad" quedaría construido por el
#    diseño. En el panel todas las variables se miden con lo conocido al inicio del mes, antes del
#    desenlace. Por la misma razón no se usa el número total de meses que duró cada persona, que solo
#    se conoce cuando termina el episodio.
# 4. **El corte transversal es un caso particular.** La versión válida de una fila por persona toma a
#    las activas en una fecha y define *y* como la renuncia en los *k* meses siguientes. Usa las
#    variables de un solo mes y una ventana elegida a mano, y la foto del último mes no sirve para
#    entrenar porque su desenlace todavía no ocurrió. El panel usa todos los meses y da el riesgo de
#    cualquier mes con un solo modelo.
#
# **Estructura:** unidad de observación persona-mes, nivel de agregación mensual, tipo panel
# (longitudinal) desbalanceado: cada persona entra y sale en meses distintos. No hay componente
# espacial: la ubicación es un código de sede, sin coordenadas.
#
# ## Diccionario
#
# El diccionario completo está en `diccionario.csv` (una fila por columna del panel: nombre legible,
# bloque, tipo, unidad, significado y momento en que se conoce). La tabla siguiente, plegada, lo
# presenta completo y ordenado por bloque, con el número de valores distintos y el porcentaje de nulos
# calculados sobre el panel; a continuación se resume por bloque.

# %% tags=["hide-output"]
dic = pd.read_csv('diccionario.csv')
assert set(dic.variable) == set(p.columns) and len(dic) == p.shape[1]
dic['distintos'] = dic.variable.map(p.nunique())
dic['% nulos'] = dic.variable.map(100 * p.isna().mean()).round(1)
orden_bloques = ['identificador', 'atributos'] + list(BLOQUES) + ['analisis_sensibilidad', 'desenlace', 'objetivo']
dic['bloque'] = pd.Categorical(dic.bloque, orden_bloques, ordered=True)
COLS_DIC = ['variable', 'nombre', 'tipo', 'unidad', 'distintos', '% nulos', 'significado', 'disponible']
ESTILO_DIC = [{'selector': 'th, td', 'props': [('text-align', 'left'), ('vertical-align', 'top'), ('font-size', '0.8em')]},
              {'selector': 'td.col6', 'props': [('min-width', '420px'), ('white-space', 'normal')]},
              {'selector': 'td.col1, td.col7', 'props': [('min-width', '160px'), ('white-space', 'normal')]}]
dic.sort_values('bloque', kind='stable')[['bloque'] + COLS_DIC].style.hide(axis='index').format(
    {'% nulos': '{:.1f}'}).set_table_styles(ESTILO_DIC)

# %% [markdown]
# Resumen por bloque: cuántas variables tiene cada uno, de qué tipo y cuánto faltan.

# %%
pred = dic[dic.variable.isin(PREDICTORAS)].copy()
pred['clase'] = np.select([pred.variable.isin(CAT), pred.variable.map(p.nunique()) <= 2],
                          ['categórica', 'binaria'], 'numérica')
filas_con_nulo = {b: round(100 * p[[c for c in PREDICTORAS if BLOQUE[c] == b]].isna().any(axis=1).mean(), 1)
                  for b in ['atributos'] + list(BLOQUES)}
por_bloque = pred.groupby('bloque', observed=True).agg(
    variables=('variable', 'size'),
    numéricas=('clase', lambda s: int((s == 'numérica').sum())),
    binarias=('clase', lambda s: int((s == 'binaria').sum())),
    categóricas=('clase', lambda s: int((s == 'categórica').sum())),
    # se cuenta sobre los nulos sin redondear: cuatro atributos tienen entre 15 y 26 nulos (0,0 % al redondear)
    con_nulos=('variable', lambda s: int(p[list(s)].isna().any().sum())),
    max_pct_nulos=('% nulos', 'max'))
por_bloque['% filas con algún nulo'] = por_bloque.index.map(filas_con_nulo)
por_bloque.loc['total'] = por_bloque.sum(numeric_only=True)
por_bloque.loc['total', ['max_pct_nulos', '% filas con algún nulo']] = [
    pred['% nulos'].max(), round(100 * p[PREDICTORAS].isna().any(axis=1).mean(), 1)]
por_bloque

# %% [markdown]
# - Los **atributos** (18) describen la demografía, puesto y contrato del día 1.
# - **Contrato** (6): vencimiento del contrato fijo, renovaciones, paso previo por aprendizaje.
#   `contrato_fijo` se descarta porque repite `contrato` fila a fila.
# - **Trayectoria** (8): meses en la posición y la función, ascensos, movimientos laterales, jefatura
#   y tamaño del equipo del jefe inmediato.
# - **Salario relativo** (14): el sueldo frente al de sus pares (mediana, Gini, privación relativa,
#   residuo ajustado), frente al mínimo y los aumentos por mérito. No hay sueldo absoluto.
# - **Jornada** (27): horario teórico, horas extra y recargos pagados, y horas reales de las
#   marcaciones biométricas. Es el bloque más grande y el de más faltantes. Se descartan
#   `horas_bajo_legal`, `cambios_plan_12m`, `horas_diarias_teoricas` y `plan_horario` (más abajo).
# - **Ingreso relativo** (13): lo devengado frente a lo pactado, su parte variable, su volatilidad,
#   bonos y auxilios, siempre en salarios mínimos o en proporciones. No hay montos absolutos.
# - **Ausencias** (7): licencias remuneradas y no remuneradas, capacitación, compensatorios y día de
#   la familia. Ninguna es de salud.
# - **Vacaciones** (5): saldo pendiente, periodos acumulados, compensadas en dinero y meses desde las
#   últimas.
# - **Arraigo** (7): nacido en el departamento de la sede, postulaciones internas y retiros de
#   cesantías para vivienda o educación.
#
# Por qué no se muestran filas del panel (`head()`): aun sin nombres ni documentos, una fila completa
# con sociedad, sede, edad, antigüedad y cargo puede identificar a una persona (sección de ética). En
# su lugar va una **fila sintética**: cada valor se sorteó por separado de la distribución de su
# columna en el entrenamiento, así que la combinación no corresponde a nadie. Solo ilustra el formato.

# %%
rng = np.random.default_rng(SEMILLA)
sintetica = {c: tr[c].dropna().sample(1, random_state=int(rng.integers(1e9))).iloc[0] for c in p.columns}
sintetica['persona_id'] = 0
muestra_cols = ['persona_id', 'mes', 'sociedad', 'linea', 'edad', 'antig_meses', 'contrato', 'nivel', 'oficio',
                'meses_al_vencimiento', 'posicionamiento_local', 'horas_extra_3m', 'turnos_1m',
                'ingreso_vs_pactado_3m', 'dias_licencia_remunerada_12m', 'dias_vacaciones_pendientes',
                'nacido_en_depto_sede', 'y_renuncia']
pd.DataFrame([sintetica])[muestra_cols].T.rename(columns={0: 'fila SINTÉTICA (no es una persona)'})

# %% [markdown]
# ## Tamaño de la muestra y eventos por parámetro
#
# El tamaño se describe con *n*, *p*, *n/p*, los casos por clase y las entidades independientes. En una logística
# con un evento raro, la cantidad que importa no es *n/p* sino los **eventos por parámetro**: las
# renuncias del entrenamiento divididas por las columnas que el modelo estima de verdad, después de
# codificar las categóricas. La regla clásica pide al menos 10 (Peduzzi et al., 1996); por debajo, los
# coeficientes sin penalizar se inflan y el modelo se sobreajusta.

# %%
def columnas_modelo(df, cat, num):
    """Columnas que ve la logística: one-hot con infrecuentes agrupados (como en el Pipeline del
    capítulo 7), las numéricas y, aparte, un indicador de faltante por cada numérica con nulos."""
    enc = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=300).fit(df[cat].fillna('(nulo)'))
    indicadores = [c for c in num if df[c].isna().any()]
    return len(enc.get_feature_names_out()), len(num), len(indicadores)

ren_tr = int(tr[OBJETIVO].sum())
filas = []
# comun.NUM no incluye las 3 binarias de atributos (BIN): se suman aparte
for nombre, cat, num in [('18 atributos', CAT_BASE, NUM_BASE + BIN), (f'{len(PREDICTORAS)} predictoras', CAT, NUM + BIN)]:
    oh, nn, ind = columnas_modelo(tr, cat, num)
    for variante, k in [('sin indicadores', oh + nn), ('con indicadores de faltante', oh + nn + ind)]:
        filas.append((nombre, variante, len(cat) + len(num), oh, nn, ind if 'con' in variante else 0, k,
                      round(len(tr) / k), round(ren_tr / k, 1)))
epp = pd.DataFrame(filas, columns=['conjunto', 'variante', 'variables', 'columnas one-hot', 'numéricas',
                                   'indicadores', 'columnas', 'n/columnas', 'renuncias/columna']
                   ).set_index(['conjunto', 'variante'])

meses_persona = p.groupby('persona_id').size()
print(f'panel: n = {len(p):,} filas | p = {len(PREDICTORAS)} variables | n/p = {len(p) / len(PREDICTORAS):,.0f}')
print(f'casos por clase: {n_no:,} no renuncias (clase 0) y {n_ren} renuncias (clase 1)')
print(f'entidades independientes: {p.persona_id.nunique():,} personas '
      f'(mediana de {meses_persona.median():.0f} meses por persona); '
      f'personas con una renuncia: {int(p.groupby("persona_id")[OBJETIVO].max().sum())}')
print(f'entrenamiento: {len(tr):,} filas, {ren_tr} renuncias')
epp

# %% [markdown]
# *n/p* es holgado (810 filas por variable), pero no es la medida adecuada. Con las 105 predictoras, el one-hot deja 179
# columnas categóricas (con los infrecuentes agrupados) y, con las 88 numéricas y binarias y los 54
# indicadores de faltante, el modelo estima 321 coeficientes con 688 renuncias: unas **2 renuncias
# por columna** (2,1), muy por debajo de 10. Con solo los 18 atributos (162 columnas) serían 4,2. Esto tiene dos
# consecuencias para el capítulo 7: hace falta una regularización fuerte (la rejilla de *C* tiene que
# bajar lo suficiente, hasta 1e-4) y
# L1, que deja coeficientes en cero, es una candidata natural frente a L2.
#
# El tamaño efectivo está entre las 5.738 personas y las 85.068 filas, porque los meses de una misma
# persona están correlacionados. Para el evento lo que cuenta es que cada renuncia es de una persona
# distinta o de un episodio distinto: 837 renuncias de 835 personas (dos personas renunciaron,
# reingresaron y volvieron a renunciar).
#
# ## Faltantes
#
# Casi todos los nulos del panel son **estructurales**: la variable no aplica o no puede medirse para
# esa fila, y una regla observable dice cuándo. La base no los imputa; el tratamiento queda en el
# `Pipeline` del capítulo 7, ajustado con el entrenamiento.
#
# En la matriz, cada columna es una persona-mes (muestra de 3.000, ordenada por mes) y cada fila, una
# predictora con algún nulo, agrupadas por bloque.

# %%
con_nulos = [c for c in PREDICTORAS if p[c].isna().any()]
muestra = p.sample(3000, random_state=SEMILLA).sort_values('mes')
bloques_nulos = [BLOQUE[c] for c in con_nulos]

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(muestra[con_nulos].isna().T, cbar=False, cmap=['#FFFFFF', VERDE], ax=ax, xticklabels=False)
for i in range(1, len(con_nulos)):
    if bloques_nulos[i] != bloques_nulos[i - 1]:
        ax.axhline(i, color=ORO, lw=1.2)
ax.set_yticks(np.arange(len(con_nulos)) + 0.5)
ax.set_yticklabels([f'{c}  [{BLOQUE[c]}]' for c in con_nulos], fontsize=6.5)
ax.set(title=f'Matriz de nulos de las {len(con_nulos)} predictoras con faltantes (verde = nulo)',
       xlabel='Persona-mes (muestra de 3.000, ordenada por mes)', ylabel='')
plt.tight_layout()
plt.show()
print(f'predictoras con nulos: {len(con_nulos)} de {len(PREDICTORAS)} | filas con al menos un nulo: '
      f'{100 * p[PREDICTORAS].isna().any(axis=1).mean():.1f} %')

# %% [markdown]
# La matriz tiene franjas, no puntos sueltos: bloques enteros de variables faltan juntos en las mismas
# filas. Las franjas más anchas son las marcaciones biométricas (jornada), el vencimiento del contrato,
# el salario relativo y el tamaño del equipo del jefe. Las variables del tipo "meses desde X" están
# vacías casi siempre (último ascenso, aprendizaje, postulación) o en casi la mitad de las filas
# (aumento por mérito), porque el hecho no ocurrió. Casi ninguna fila (0,1 %) tiene todas las predictoras.
#
# ### Mecanismo
#
# La prueba de Little no sirve aquí: supone variables continuas con distribución conjunta normal, y
# los faltantes del panel no son un azar que haya que probar, sino reglas conocidas. La evidencia que
# se usa es doble: (1) contar cuántas filas se salen de la regla que explica el nulo; (2) comparar,
# en el entrenamiento, la tasa de renuncia y la antigüedad entre filas con y sin el nulo. Si la regla
# explica el nulo exactamente, el faltante depende de variables observadas (MAR, y en rigor
# "no aplica"); si además la tasa difiere, el nulo es informativo y no puede imputarse con la media o
# la moda sin perder señal.

# %%
REGLAS = [
    ('familia_cargo', 'primer mes del episodio (no hay cargo previo)', lambda d: d.primer_mes.eq(1)),
    ('ingreso_vs_pactado_3m', 'primer mes del episodio (no hay nómina previa)', lambda d: d.primer_mes.eq(1)),
    ('meses_al_vencimiento', 'contrato indefinido', lambda d: d.contrato_fijo.eq(0)),
    ('horas_extra_1m', 'no elegible para horas extra, o primer mes', lambda d: d.elegible_horas_extra.eq(0) | d.primer_mes.eq(1)),
    ('turnos_1m', 'no marca en biométrico', lambda d: d.marca_biometrico.eq(0)),
    ('horas_turno_3m', 'no marca, o sin turnos en 3 meses', lambda d: d.marca_biometrico.eq(0) | d.turnos_1m.eq(0)),
    ('posicionamiento_local', 'sueldo no básico (destajo o integral)', lambda d: d.tipo_salario.ne('basico')),
    ('posicionamiento', 'sueldo no básico, o grupo de oficio y nivel con menos de 5', lambda d: d.tipo_salario.ne('basico')),
    ('situacion_minimo', 'sueldo no básico, o sin sueldo de 12 meses antes', lambda d: d.tipo_salario.ne('basico')),
    ('volatilidad_ingreso_12m', 'menos de 6 meses de nómina detrás', lambda d: d.antig_meses.lt(6)),
    ('tamano_equipo_jefe', 'sin evaluación que identifique al jefe', None),
    ('nacido_en_depto_sede', 'sin dato de origen en el perfil', None),
    ('meses_desde_aumento_merito', 'nunca tuvo aumento por mérito', None),
    ('meses_desde_ascenso', 'sin ascenso desde 2022 o desde el ingreso', None),
]
filas = []
for v, regla, f in REGLAS:
    nulo = tr[v].isna()
    r = f(tr) if f else None
    t_n, t_c = tasa_ic(tr, nulo.map({True: 'nulo', False: 'no nulo'})).reindex(['nulo', 'no nulo'])['tasa_%']
    filas.append((v, BLOQUE[v], round(100 * nulo.mean(), 1), regla,
                  int((nulo & ~r).sum()) if f else np.nan, int((r & ~nulo).sum()) if f else np.nan,
                  t_n, t_c, tr.loc[nulo, 'antig_meses'].median(), tr.loc[~nulo, 'antig_meses'].median()))
mecanismo = pd.DataFrame(filas, columns=['variable', 'bloque', '% nulos', 'regla', 'nulo fuera de la regla',
                                         'regla sin nulo', 'tasa % (nulo)', 'tasa % (no nulo)',
                                         'antig. mediana (nulo)', 'antig. mediana (no nulo)']).set_index('variable')
mecanismo

# %% [markdown]
# Lectura de la tabla (entrenamiento):
#
# - **Reglas exactas.** El vencimiento falta exactamente en los contratos indefinidos, las marcaciones
#   exactamente en quien no marca y las horas extra exactamente en quien no es elegible o está en su
#   primer mes. La nómina del mes anterior falta en el primer mes del episodio salvo 4 filas, y el
#   cargo previo salvo 40. Son faltantes **MAR por construcción**: dependen de variables observadas,
#   y el valor ausente no existe (un contrato indefinido no tiene fecha de vencimiento).
# - **Reglas parciales.** `posicionamiento` falta además en 3.020 filas de grupos de oficio y nivel con
#   menos de cinco sueldos básicos, y `situacion_minimo` en 11.805 sin sueldo de 12 meses antes; la
#   regla de la volatilidad por antigüedad es una aproximación (557 filas con menos de seis meses sí la
#   tienen). Siguen siendo MAR: dependen del tamaño del grupo y de la historia, que se observan.
# - **Sin regla simple.** El jefe falta sobre todo en los ingresos recientes (antigüedad mediana de 10
#   meses frente a 80): las evaluaciones van por oleadas y quien entró después de la última no la
#   tiene. El origen falta en una de cada cuatro filas y, al revés, en personas antiguas (79 meses
#   frente a 27) y mucho más en ganadería. Son MAR con respecto a la antigüedad y la línea; no puede
#   descartarse del todo un componente MNAR (quien no llena su perfil puede ser distinto), y el
#   indicador de faltante lo recoge si lo hay.
# - **"Meses desde X" nulo = nunca ocurrió.** No es un dato perdido sino un valor fuera de la escala.
# - **No es MCAR.** Salvo en las horas extra (1,04 % frente a 1,02 %), la tasa de renuncia con nulo es
#   distinta de la tasa sin él: 1,6 % frente a 1,0 % en el primer mes, 0,47 % frente a 1,51 % sin
#   vencimiento (indefinidos), 0,63 % frente a 1,32 % sin marcaciones, 2,19 % frente a 0,86 % sin
#   volatilidad, 1,57 % frente a 0,71 % sin jefe identificado. El nulo lleva señal, casi siempre a
#   través de la antigüedad, e imputar solo la mediana la borraría.
#
# **Tratamiento (capítulo 7, ajustado con el entrenamiento):** en las numéricas, imputación con la
# mediana más un **indicador de faltante** por variable (`SimpleImputer(add_indicator=True)`), que le
# dice al modelo "no aplica" por separado del valor; en las categóricas, el nulo es una categoría
# propia. No se eliminan filas: casi ninguna está completa.
#
# ## Duplicados, valores imposibles y atípicos

# %%
print('filas persona-mes duplicadas:', int(p.duplicated(['persona_id', 'mes']).sum()))
casi = p[PREDICTORAS + [OBJETIVO]].duplicated(keep=False)
casi_base = p[BASE + [OBJETIVO]].duplicated(keep=False)
print(f'casi-duplicados (mismo vector de las {len(PREDICTORAS)} predictoras y objetivo): {int(casi.sum()):,} filas '
      f'({100 * casi.mean():.1f} %) | con solo los 18 atributos: {int(casi_base.sum()):,} ({100 * casi_base.mean():.0f} %)')
print('contrato_fijo idéntica a contrato:', bool((p.contrato.eq('Termino Fijo').astype(int) == p.contrato_fijo).all()))

edad_ingreso = p.edad - p.antig_meses / 12
print(f'edad: p1 {p.edad.quantile(0.01):.0f}, p99 {p.edad.quantile(0.99):.0f} años | mayores de 70: {int((p.edad > 70).sum())} filas')
print(f'edad al ingreso aproximada (edad entera - antigüedad) por debajo de 17: {int((edad_ingreso < 17).sum())} filas | '
      f'por encima de 70: {int((edad_ingreso > 70).sum())} filas')
print(f'antigüedad negativa: {int((p.antig_meses < 0).sum())} filas')
pct = [c for c in NUM if c.startswith('pct_') and c != 'pct_ultimo_aumento_merito']
print('porcentajes fuera de 0-100:', int(sum(((p[c] < 0) | (p[c] > 100)).sum() for c in pct)))
print('valores negativos por variable:', {c: int((p[c] < 0).sum()) for c in NUM if (p[c] < 0).any()})
print(f'contrato fijo con la fecha de fin ya pasada: {int((p.meses_al_vencimiento < 0).sum())} filas')
print(f'caída del ingreso de -100 % (devengado cero en 3 meses): {int((p.cambio_ingreso_3m_vs_9m <= -99.9).sum())} filas')
casi_const = {c: round(100 * p[c].value_counts(normalize=True).iloc[0], 1) for c in NUM
              if p[c].value_counts(normalize=True).iloc[0] > 0.97}
print('numéricas con un solo valor en más del 97 % de las filas (% de la moda):', casi_const)

# %% [markdown]
# - No hay filas persona-mes repetidas.
# - Con las 105 predictoras, solo el 0,4 % de las filas comparte todo el vector con otra; con las 18
#   atributos era el 20 %. Las variables de historia distinguen a personas con el mismo perfil de
#   puesto. No son la misma persona (su antigüedad cambia cada mes) y no se eliminan: cada fila es una
#   persona en riesgo ese mes.
# - `contrato_fijo` es idéntica a `contrato`: redundancia exacta, no error. **Se descarta**
#   (`comun.FUERA`) y queda `contrato`.
# - **Edades:** la base admite una edad al ingreso de 18 a 70 años (regla de exclusión). La edad al
#   ingreso se aproxima con la edad entera menos la antigüedad, así que puede quedar hasta un año por
#   debajo de la real; ninguna fila queda por debajo de 17 ni por encima de 70, lo que es compatible
#   con la regla. Sin ella quedarían ingresos entre los 15 y los 17 años, que con los aprendices ya
#   excluidos solo pueden ser fechas de nacimiento mal registradas (7 personas, recuadro de la
#   construcción del panel). Las 82 filas de mayores de 70 son
#   personas que ingresaron antes de esa edad y siguen activas; son valores reales.
# - **Otros rangos:** no hay antigüedades negativas ni porcentajes fuera de 0 a 100. Los negativos
#   que aparecen son legítimos por definición (residuo del sueldo, variación frente al mínimo, cambio
#   del ingreso), salvo los auxilios negativos, que son reversiones de un pago anterior, y seis filas
#   de contrato fijo con fecha de fin ya pasada sin renovación registrada (retraso del registro).
# - **Variables casi constantes:** varias binarias y conteos tienen la moda en más del 97 % de las
#   filas. Aportan poca información y el capítulo 4 las revisa; no se quitan aquí porque eso sería
#   seleccionar variables antes de tiempo. La excepción son las horas semanales bajo el máximo legal,
#   que valen 0 en el 99,98 % de las filas (13 filas distintas de cero): no es una variable sino una
#   constante con errores de registro, y **se descarta** sin mirar su relación con el objetivo.
#
# Una inconsistencia pide atención especial: `cambios_plan_12m`.

# %%
cambio_plan = (100 * p.cambios_plan_12m.gt(0).groupby(p.mes).mean()).round(1)
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(cambio_plan.index, cambio_plan.values, marker='o', color=VERDE)
ax.axvline(list(cambio_plan.index).index('2026-05') - 0.5, ls='--', color=ORO, lw=1)
ax.set(title='Filas con algún cambio de plan de horario en 12 meses, por mes (%; punteada: inicio del test)',
       ylabel='% de filas')
ax.tick_params(axis='x', rotation=60)
plt.tight_layout()
plt.show()

# %% [markdown]
# Cerca de cuatro de cada cinco filas registran un cambio de plan de horario en los 12 meses previos,
# y la proporción se desploma en dos meses concretos: julio de 2025 y agosto de 2026, al ritmo de las
# reasignaciones de planes por la reducción legal de la jornada (capítulo 5). La variable sigue el
# calendario y no a la persona, y su distribución cambia dentro del test; por eso **se descarta**
# (`comun.FUERA`), por su definición y no por su asociación con la renuncia. Por la misma razón se
# descartan `horas_diarias_teoricas` y `plan_horario` (capítulo 5).
#
# ### Atípicos
#
# Las vallas de Tukey (Q1 − 1,5·IQR, Q3 + 1,5·IQR) se calculan **solo con el entrenamiento**, porque
# si se decide un recorte con ellas es una decisión de preprocesamiento. La tabla cubre todas las
# numéricas que no son binarias, ordenadas por la fracción de filas fuera de las vallas.

# %% tags=["hide-output"]
cont = [c for c in NUM if tr[c].nunique() > 2]
filas = []
for v in cont:
    x = tr[v].dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    li, ls = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    filas.append((v, BLOQUE[v], len(x), q1, q3, li, ls, round(100 * ((x < li) | (x > ls)).mean(), 1),
                  x.quantile(0.01), x.quantile(0.99)))
atipicos = (pd.DataFrame(filas, columns=['variable', 'bloque', 'n', 'Q1', 'Q3', 'valla inf.', 'valla sup.',
                                         '% fuera', 'p1', 'p99'])
            .set_index('variable').sort_values('% fuera', ascending=False).round(2))
print(f'numéricas no binarias: {len(cont)} | con más del 5 % fuera de las vallas: '
      f'{int((atipicos["% fuera"] > 5).sum())} | con IQR = 0: {int((atipicos.Q1 == atipicos.Q3).sum())}')
atipicos

# %% [markdown]
# De las 73 numéricas no binarias, 41 tienen más del 5 % de las filas fuera de las vallas, pero por
# una razón que no es error: son distribuciones con una masa en un valor y una cola larga (horas
# extra, recargos, dominicales, bonos, días de licencia). En 19 variables Q1 = Q3 y las vallas
# colapsan en un punto, así que todo valor distinto queda "fuera": pasa con las que valen casi
# siempre 0 y con el posicionamiento salarial, que vale exactamente 1 en los cargos de tabla (quien
# gana la mediana de su grupo). `estado_gestion_tiempos` no está en la tabla: su código se
# guarda como número, pero es una categoría y el libro la trata como categórica. En esas variables
# el IQR no sirve como detector. Revisados los valores extremos, son posibles (turnos largos, varios
# periodos de vacaciones acumulados, antigüedades de décadas), no errores de digitación; los errores conocidos de la fuente (cantidades
# digitadas en los dominicales, reversiones) ya se corrigieron al construir los bloques.
#
# **Tratamiento:** no se eliminan filas. Para la logística, que es sensible a las colas, el capítulo 7
# recorta cada numérica entre sus percentiles 1 y 99 del entrenamiento y aplica un logaritmo con signo
# a las de cola larga, dentro del `Pipeline`. La forma de cada distribución (asimetría, curtosis) se
# estudia en el capítulo 3.
#
# ## Sesgos de muestreo y representatividad
#
# - **Cobertura:** 22 sociedades con datos completos en la ventana. El modelo describe las
#   operaciones de palma, banano, industria, transporte, puerto y ganadería, y no a todo el grupo.
# - **Solo nómina directa:** quien trabaja por contratista o como temporal no está en el sistema de
#   nómina. En campo esa población puede ser grande y rotar de otra manera.
# - **Exclusiones:** aprendices (rotación reglada por su contrato), vicepresidencia y presidencia,
#   obra o labor, meses de salida con etiqueta dudosa y personas con edad al ingreso no creíble. Las
#   conclusiones no se extienden a esos grupos.
# - **Cobertura desigual de las variables nuevas:** las marcaciones biométricas existen solo en las
#   sedes con biométrico (tabla siguiente), y el origen solo para quien llenó su perfil. Lo que el
#   modelo aprenda de la jornada real describe sobre todo a banano, industria y puerto.
# - **Ventana:** 20 meses. Los efectos de calendario se observan a lo sumo dos veces.
# - **Supervivencia:** quien lleva 10 años llegó ahí porque no renunció antes. La tasa baja de las
#   antigüedades altas mezcla el efecto del tiempo con la selección de quienes permanecen.

# %%
cobertura = p.groupby('linea').agg(filas=('mes', 'size'), personas=('persona_id', 'nunique'))
cobertura['% marca biométrico'] = (100 * p.groupby('linea').marca_biometrico.mean()).round(1)
cobertura['% con origen'] = (100 * p.groupby('linea').nacido_en_depto_sede.apply(lambda s: s.notna().mean())).round(1)
cobertura['% contrato fijo'] = (100 * p.groupby('linea').contrato_fijo.mean()).round(1)
cobertura.sort_values('filas', ascending=False)

# %% [markdown]
# La cobertura biométrica va del 88 % de las filas en banano al 10 % en ganadería (53 % en palma, la
# línea más grande), y en ganadería falta además el origen en el 71 % de las filas. Las variables de marcaciones son, en parte, un
# indicador de la sede; por eso su indicador de faltante no debe leerse como "no trabaja turnos
# largos", sino como "no se mide".
#
# ## Entidades y dependencia
#
# Las filas no son independientes: la misma persona aparece hasta 20 veces. Una partición aleatoria
# por filas dejaría meses de una misma persona en entrenamiento y en validación, y el modelo se
# evaluaría sobre personas que ya vio. Por eso el test es un corte en el tiempo, la validación
# principal del capítulo 7 es temporal y, como complemento, se usa una validación agrupada por
# `persona_id` (capítulo 2).

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
ax[0].hist(meses_persona, bins=20, color=VERDE, **BORDE)
ax[0].set(title='Meses en el panel por persona', xlabel='Meses en el panel', ylabel='Número de personas')
p.groupby('mes').size().plot(ax=ax[1], marker='o', color=VERDE)
ax[1].set(title='Personas activas por mes', xlabel='', ylabel='Número de personas')
ax[1].tick_params(axis='x', rotation=60)
plt.tight_layout()
plt.show()

completas = int((meses_persona == 20).sum())
print(f'meses por persona: mediana {meses_persona.median():.0f}, media {meses_persona.mean():.1f}; '
      f'personas con los 20 meses: {completas:,} ({100 * completas / len(meses_persona):.0f} %)')
print(f'personas activas por mes: {p.groupby("mes").size().min():,} a {p.groupby("mes").size().max():,}')

# %% [markdown]
# ## Consideraciones éticas y reidentificación
#
# **Anonimización.** El panel no tiene nombre, documento, registro de personal, fecha de nacimiento,
# textos de cargo ni centro de costo. `persona_id` es un entero asignado al azar; sociedad y ubicación
# son códigos sorteados, que no siguen el tamaño ni el orden alfabético y no pueden volver a mapearse
# sin la clave, que queda fuera de la entrega. Los procesos de planta muy específicos se agruparon. No hay sueldo ni montos absolutos: el salario y el
# ingreso entran como razones frente a los pares o en salarios mínimos.
#
# **Variables excluidas.** No se usan variables de salud e incapacidades, riesgo psicosocial,
# afiliación sindical ni de hijos y familia: son datos sensibles o de menores en el sentido de la
# Ley 1581 de 2012 (arts. 5 y 7). Tampoco se usan el clima laboral individual, las quejas y
# denuncias, el desempeño, las sanciones, las deudas y descuentos de nómina ni el sueldo absoluto. No
# son sensibles en ese sentido, pero su uso para estimar el riesgo de una persona excede la finalidad
# con que se registraron (principio de finalidad, art. 4 de la misma ley). Las ausencias que sí entran (licencias, capacitación, compensatorios, día de la familia)
# no son de salud ni disciplinarias.
#
# **Reidentificación.** Aun anonimizado, el panel conserva combinaciones de atributos (sociedad, sede,
# edad, antigüedad y puesto) que en una población de este tamaño pueden describir a una persona
# concreta, y quitarlas eliminaría lo que el modelo necesita. Por eso el panel no se publica: se
# entrega a la universidad por un canal privado, con uso académico, y el libro muestra solo código,
# tablas agregadas y métricas. En las tablas por grupo se suprimen las celdas con menos de 5 renuncias
# o personas, y los gráficos por categoría omiten los grupos con menos de 300 persona-mes.
#
# ## Síntesis
#
# - Panel persona-mes de 85.068 filas, 5.738 personas y 837 renuncias (0,98 %), con 105 predictoras
#   (18 atributos y 87 de historia laboral en nueve bloques). Uso académico autorizado, no redistribuible.
# - Eventos por parámetro: unas 2 renuncias de entrenamiento por columna del modelo. La logística
#   necesita regularización fuerte.
# - Los faltantes son estructurales (MAR por construcción) e informativos: se imputan con la mediana
#   más un indicador de faltante, y el nulo categórico es una categoría.
# - Sin duplicados. Se descartan cinco columnas: `contrato_fijo` (repite `contrato`),
#   `horas_bajo_legal` (constante) y `cambios_plan_12m`, `horas_diarias_teoricas` y `plan_horario`
#   (miden el calendario de la reducción legal de la jornada, no a la persona).
# - Atípicos: colas largas con masa en cero, no errores; se recortan en p1-p99 del entrenamiento
#   dentro del `Pipeline`.
# - Se excluyen los datos sensibles y los de uso no justificado por su finalidad. El panel se entrega
#   anonimizado y no se publica.
