"""Lo que comparten todos los capitulos: ruta del panel, semilla, particion, grupos de variables,
estilo de los graficos y las funciones de tasas y metricas.

El panel NO esta en el repo (datos de personas de la empresa): se entrega aparte y se lee desde la
variable de entorno PANEL_ROTACION o, si no esta definida, desde datos/ junto a este archivo."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')   # evita el aviso de joblib al contar nucleos en Windows

# Datos y particion
PANEL = Path(os.environ.get('PANEL_ROTACION',
                            Path(__file__).resolve().parent / 'datos' / 'panel_rotacion_2025_2026.csv'))
SEMILLA = 2026
CORTE = '2026-05'          # test = may-2026 a ago-2026 (4 meses), decidido antes del EDA

# Grupos de variables
OBJETIVO = 'y_renuncia'
# atributos del trabajador y del puesto (18)
NUM_BASE = ['edad', 'antig_meses']
BIN = ['primer_mes', 'reingreso', 'traslado_12m']
CAT_BASE = ['sociedad', 'linea', 'ubicacion', 'genero', 'estado_civil', 'contrato', 'nivel',
            'familia_cargo', 'oficio', 'tipo_unidad', 'area_funcional', 'proceso_planta', 'tipo_costos']
BASE = NUM_BASE + BIN + CAT_BASE
# historia laboral de la persona, por bloque (nomina, tiempos y novedades del mes y de los meses anteriores).
# Fuera, por privacidad: salud, riesgo psicosocial, clima individual, sindicato, hijos, quejas, desempeno,
# sanciones, deudas y descuentos, sueldo absoluto.
BLOQUES = {
    'contrato': ['meses_al_vencimiento', 'n_renovaciones', 'meses_desde_cambio_contrato', 'fue_aprendiz',
                 'meses_desde_aprendiz', 'ingreso_a_mitad'],
    'trayectoria': ['meses_en_posicion', 'meses_en_funcion', 'meses_desde_ascenso', 'n_ascensos_24m',
                    'n_movimientos_laterales_12m', 'es_jefe_formal', 'personas_a_cargo', 'tamano_equipo_jefe',
                    'postulaciones_internas_12m', 'requisicion_promocion_12m', 'meses_desde_postulacion'],
    'salario_relativo': ['tipo_salario', 'posicionamiento', 'posicionamiento_local', 'gini_oficio_nivel',
                         'privacion_oficio_nivel', 'residuo_sueldo', 'en_mediana_local', 'dist_mediana_local', 'gana_minimo',
                         'var_sueldo_smmlv_12m', 'situacion_minimo', 'meses_desde_aumento_merito',
                         'pct_ultimo_aumento_merito', 'n_aumentos_merito_24m'],
    'jornada': ['estado_gestion_tiempos',
                'horas_extra_1m', 'recargo_nocturno_1m', 'dominicales_1m', 'sabados_smmlv_1m', 'horas_extra_3m',
                'recargo_nocturno_3m', 'dominicales_3m', 'sabados_smmlv_3m', 'horas_extra_12m', 'recargo_nocturno_12m',
                'dominicales_12m', 'sabados_smmlv_12m', 'meses_con_extras_12m', 'pct_meses_con_extras_12m',
                'elegible_horas_extra', 'turnos_1m', 'horas_turno_1m', 'horas_turno_3m', 'pct_turnos_10h_3m',
                'pct_turnos_12h_3m', 'pct_turnos_nocturnos_3m', 'domingos_3m', 'pct_entradas_sin_salida_3m',
                'marca_biometrico', 'jornada_vs_pago', 'turnos_largos_sin_pago'],
    'ingreso_relativo': ['ingreso_vs_pactado_3m', 'pct_variable_3m', 'volatilidad_ingreso_12m', 'cambio_ingreso_3m_vs_9m',
                         'meses_con_ausencia_pagada_12m', 'bono_fiesta_12m', 'bono_seleccionado_12m', 'bonif_ocasional_12m',
                         'meses_con_bpr_12m', 'bono_productividad_12m', 'bonificacion_otra_12m', 'auxilio_educativo_12m',
                         'auxilios_3m'],
    'ausencias': ['dias_licencia_no_remunerada_3m_r2', 'dias_licencia_no_remunerada_12m_r2',
                  'licencias_no_remuneradas_12m', 'dias_licencia_remunerada_12m', 'dias_capacitacion_12m',
                  'dias_compensatorio_12m', 'dia_familia_12m'],
    'vacaciones': ['meses_desde_vacaciones', 'dias_vacaciones_pendientes', 'periodos_vacaciones_pendientes',
                   'vacaciones_acumuladas_2p', 'dias_vacaciones_compensadas_12m'],
    'proyectos_personales': ['cesantias_vivienda_12m', 'cesantias_educacion_12m', 'cesantias_vivienda_historico'],
    'origen': ['nacido_en_depto_sede'],
}
# orden de las columnas del modelo: el mismo de siempre, independiente de como se agrupan para describir
# (con L1 el orden de las columnas puede mover levemente la solucion)
_AL_FINAL = ['nacido_en_depto_sede', 'postulaciones_internas_12m', 'requisicion_promocion_12m', 'meses_desde_postulacion',
             'cesantias_vivienda_12m', 'cesantias_educacion_12m', 'cesantias_vivienda_historico']
HISTORIA = [c for c in sum(BLOQUES.values(), []) if c not in _AL_FINAL] + _AL_FINAL
CAT_HISTORIA = ['tipo_salario', 'situacion_minimo', 'estado_gestion_tiempos', 'jornada_vs_pago']
CAT = CAT_BASE + CAT_HISTORIA
NUM = NUM_BASE + [c for c in HISTORIA if c not in CAT_HISTORIA]     # incluye binarias 0/1 de la historia; NO incluye BIN
# fuera de la historia por el capitulo 1: contrato_fijo (identica a contrato), horas_bajo_legal (0 en el 99,98 %)
# y cambios_plan_12m; por el capitulo 5, horas_diarias_teoricas y plan_horario. Las tres ultimas siguen el
# calendario de la reduccion legal de la jornada (46 -> 44 -> 42 horas), no a la persona, y en jul-2026 (test)
# toman un valor que no existe en el entrenamiento
FUERA = ['contrato_fijo', 'horas_bajo_legal', 'cambios_plan_12m', 'horas_diarias_teoricas', 'plan_horario']
BLOQUE = {c: 'atributos' for c in BASE} | {c: b for b, v in BLOQUES.items() for c in v}
PREDICTORAS = BASE + HISTORIA
SENSIBILIDAD = ['preaviso_no_renovacion']   # con preaviso y = 0 por construccion: no entra; solo para la sensibilidad
DESENLACE = ['evento', OBJETIVO]           # describen la salida: no entran

# Nombre legible de cada variable, para titulos y ejes de los graficos
NOMBRE = {
    'edad': 'edad', 'antig_meses': 'antigüedad reconocida', 'linea': 'línea de negocio',
    'contrato': 'tipo de contrato', 'nivel': 'nivel del cargo', 'tipo_unidad': 'tipo de unidad',
    'estado_civil': 'estado civil', 'familia_cargo': 'familia de cargo', 'oficio': 'oficio',
    'sociedad': 'sociedad', 'tipo_costos': 'tipo de personal', 'proceso_planta': 'proceso de planta',
    'genero': 'género', 'ubicacion': 'ubicación', 'area_funcional': 'área funcional',
    'antig_t': 'tramo de antigüedad', 'primer_mes': 'primer mes', 'reingreso': 'reingreso',
    'traslado_12m': 'traslado en 12 meses',
}
UNIDAD = {'edad': 'Edad (años)', 'antig_meses': 'Antigüedad (meses)'}
# diccionario.csv (variable, bloque, tipo, unidad, significado, disponible): nombre legible de las variables de la
# historia laboral; las de arriba mandan
_DIC = Path(__file__).resolve().parent / 'diccionario.csv'
if _DIC.exists():
    _d = pd.read_csv(_DIC)
    NOMBRE = {**dict(zip(_d.variable, _d.nombre)), **NOMBRE} if 'nombre' in _d else NOMBRE

# Colores del libro: los de las sustentaciones y del EDA anterior
VERDE, ORO, TINTA, GRIS = '#405731', '#B08A2D', '#333333', '#5A5A5A'
VERDE_CL, ORO_CL, GRIS_CL = '#E8F0EB', '#F5EDE0', '#BBCAC1'
PALETA = [VERDE, ORO, '#7F9C6B', GRIS, '#D4B25F', '#A3B8A9', '#6E4F1F', '#2F3E25']
BORDE = dict(edgecolor=ORO, linewidth=0.6)   # filete dorado de las barras


def estilo():
    """Tema de los graficos. Registra los mapas de color 'verde', 'oro' y 'oro_verde'."""
    import matplotlib as mpl
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    sns.set_theme(style='whitegrid', font_scale=0.9, palette=PALETA)
    mpl.rcParams.update({
        'text.color': TINTA, 'axes.labelcolor': TINTA,
        'xtick.color': GRIS, 'ytick.color': GRIS,
        'axes.titleweight': 'bold', 'axes.titlecolor': VERDE,
        'axes.titlelocation': 'left', 'axes.titlesize': 11,
        'axes.edgecolor': '#CCCCCC', 'grid.color': '#EEEEEE',
        'legend.frameon': False, 'image.cmap': 'verde', 'figure.dpi': 100,
    })
    mapas = {
        'verde': ['#FFFFFF', VERDE_CL, '#7F9C6B', VERDE],
        'oro': ['#FFFFFF', ORO_CL, '#D4B25F', ORO],
        'oro_verde': [ORO, ORO_CL, '#FFFFFF', VERDE_CL, VERDE],
    }
    for nombre, colores in mapas.items():
        if nombre not in mpl.colormaps:
            mpl.colormaps.register(LinearSegmentedColormap.from_list(nombre, colores))
    tablas_en_una_fila()


def _encabezado_plano(df):
    """Sube el nombre del indice a la fila de las columnas: pandas lo pone en una segunda fila."""
    if any(df.index.names):
        df = df.copy()
        nombre = ' / '.join(str(n) for n in df.index.names if n is not None)
        ultimo = df.columns.names[-1]
        if ultimo is not None:                     # tabla cruzada: "filas / columnas" en una sola celda
            nombre = f'{nombre} / {ultimo}'
        df.columns = df.columns.set_names(list(df.columns.names[:-1]) + [nombre])
        df.index.names = [None] * df.index.nlevels
    return df._repr_html_()


def tablas_en_una_fila():
    """Registra el formato HTML de los DataFrame con el encabezado en una sola fila."""
    try:
        from IPython import get_ipython
    except ImportError:
        return
    ip = get_ipython()
    if ip is not None:
        ip.display_formatter.formatters['text/html'].for_type(pd.DataFrame, _encabezado_plano)


def cargar():
    p = pd.read_csv(PANEL, low_memory=False)
    # las categoricas guardadas como codigo numerico (estado_gestion_tiempos) pasan a texto: el OneHotEncoder
    # no acepta numeros mezclados con la etiqueta '(nulo)' de la imputacion
    for c in CAT:
        p[c] = p[c].map(lambda v: v if pd.isna(v) or isinstance(v, str) else str(int(v)) if float(v).is_integer() else str(v))
    return p.sort_values(['persona_id', 'mes']).reset_index(drop=True)


def particion(p):
    """Corte cronologico. Las mismas personas aparecen en los dos lados (es un panel): el test mide
    si el modelo predice meses futuros, que es el uso real."""
    return p[p.mes < CORTE].copy(), p[p.mes >= CORTE].copy()


def tasa(df, por, minimo=0):
    """Filas, renuncias y tasa (%) por grupo; descarta los grupos con menos de `minimo` filas."""
    t = df.groupby(por, dropna=False, observed=True)[OBJETIVO].agg(filas='size', renuncias='sum')
    t['tasa_%'] = 100 * t.renuncias / t.filas
    return t[t.filas >= minimo]


def wilson(k, n, z=1.96):
    """Intervalo de Wilson al 95 % para una proporcion; devuelve (bajo, alto) en %."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / n
    den = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / den
    radio = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return 100 * (centro - radio), 100 * (centro + radio)


def tasa_ic(df, por, minimo=0):
    """`tasa` con el intervalo de Wilson de cada grupo."""
    t = tasa(df, por, minimo)
    t['ic_bajo'], t['ic_alto'] = wilson(t.renuncias, t.filas)
    return t.round(2)


def v_cramer(a, b):
    """V de Cramer entre dos categoricas; el nulo cuenta como una categoria mas."""
    from scipy.stats import chi2_contingency

    t = pd.crosstab(a.fillna('(nulo)'), b.fillna('(nulo)')).to_numpy()
    if min(t.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))


def eje_llano(eje):
    """Eje en escala log con números corrientes (1.000, 10.000; 0,5, 1, 2) en vez de potencias de 10."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    def f(v, _):
        return f'{v:,.0f}'.replace(',', '.') if v >= 1000 else f'{v:g}'.replace('.', ',')
    eje.set_major_formatter(FuncFormatter(f))
    eje.set_minor_formatter(NullFormatter())


def es(texto):
    """Cifras con el formato del texto del libro: punto de miles y coma decimal (32.597; 48,4)."""
    import re
    return re.sub(r'\d[\d,.]*\d|\d', lambda m: m.group(0).translate(str.maketrans(',.', '.,')), str(texto))


def rotulo(categoria):
    """Nombre de una categoría para un eje: sin guiones bajos (finca_palma -> finca palma)."""
    return str(categoria).replace('_', ' ')


def etiquetar(ax, barras, fmt='{:,.0f}', textos=None, fontsize=7):
    """Escribe el valor al final de cada barra."""
    if textos is None:
        textos = [fmt.format(v) for v in barras.datavalues]
    textos = [es(t) for t in textos]
    ax.bar_label(barras, labels=textos, padding=2, fontsize=fontsize, color=TINTA)
    # margen proporcional al texto más largo, para que la cifra no se salga del recuadro
    largo = max((len(str(t)) for t in textos), default=0)
    if barras.orientation == 'horizontal':
        ax.margins(x=0.08 + 0.018 * largo)
    else:
        ax.margins(y=0.15)


def puntos(ax, x, y, fmt='{:.2f}', fontsize=7):
    """Escribe el valor encima de cada punto de una serie corta."""
    for xi, yi in zip(x, y):
        if pd.notna(yi):
            ax.annotate(fmt.format(yi), (xi, yi), textcoords='offset points', xytext=(0, 5),
                        ha='center', fontsize=fontsize, color=TINTA)


def grafico_tasa(ax, t, titulo, base=None):
    """Barras horizontales de tasa con su IC de Wilson; la linea punteada es la tasa global."""
    t = t.sort_values('tasa_%')
    y = range(len(t))
    ax.barh(y, t['tasa_%'], color=VERDE, **BORDE)
    error = [t['tasa_%'] - t.ic_bajo, t.ic_alto - t['tasa_%']]
    ax.errorbar(t['tasa_%'], y, xerr=error, fmt='none', ecolor=TINTA, lw=0.8)
    # la etiqueta va despues del IC para no taparlo
    for yi, v, alto in zip(y, t['tasa_%'], t.ic_alto):
        ax.text(alto, yi, es(f' {v:.2f} %'), va='center', fontsize=7, color=TINTA)
    ax.set_xlim(0, t.ic_alto.max() * 1.25)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f'{rotulo(i)} (n={es(f"{n:,}")})' for i, n in zip(t.index, t.filas)], fontsize=8)
    if base is not None:
        ax.axvline(base, ls='--', c=GRIS, lw=0.8)
    ax.set(title=titulo, xlabel='Tasa de renuncia mensual (%)')


def pliegues_temporales(meses, primer_val='2025-09', gap=1):
    """Validacion de ventana creciente por mes: cada pliegue valida un mes y entrena con todos los
    meses anteriores menos `gap` meses de separacion. gap=1 porque la renuncia de un mes puede quedar
    registrada en el sistema de nomina ya entrado el mes siguiente."""
    meses = np.asarray(meses)
    orden = sorted(np.unique(meses))
    pliegues = []
    for i in range(orden.index(primer_val), len(orden)):
        ultimo_entrenamiento = orden[i - gap - 1]
        entrena = np.where(meses <= ultimo_entrenamiento)[0]
        valida = np.where(meses == orden[i])[0]
        pliegues.append((entrena, valida))
    return pliegues


def marcar_top(s, meses, q):
    """1 para la fraccion q de mayor puntaje dentro de cada mes: la lista que se revisaria cada mes.
    Usa solo el orden de los puntajes del mes, no las etiquetas."""
    d = pd.DataFrame({'s': s, 'm': meses})
    marca = np.zeros(len(d), int)
    for _, g in d.groupby('m'):
        k = max(1, int(round(q * len(g))))
        marca[g.nlargest(k, 's').index.to_numpy()] = 1
    return marca


def metricas_umbral(y, marca):
    """Exactitud, precision, exhaustividad, especificidad, F1 y F2 de una marca 0/1."""
    tp = int(((marca == 1) & (y == 1)).sum())
    fp = int(((marca == 1) & (y == 0)).sum())
    fn = int(((marca == 0) & (y == 1)).sum())
    tn = int(((marca == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    exhaustividad = tp / (tp + fn) if tp + fn else 0.0

    def f_beta(b):
        if precision + exhaustividad == 0:
            return 0.0
        return (1 + b**2) * precision * exhaustividad / (b**2 * precision + exhaustividad)

    return {'exactitud': (tp + tn) / len(y), 'precisión': precision, 'exhaustividad': exhaustividad,
            'especificidad': tn / (tn + fp), 'F1': f_beta(1), 'F2': f_beta(2)}
