"""Lo que comparten todos los capitulos: ruta del panel, semilla, particion, grupos de variables,
estilo de los graficos y las funciones de tasas y metricas.

El panel NO esta en el repo (datos de la empresa): se lee desde la variable de entorno PANEL_ROTACION
o, si no esta definida, desde datos/rotacion/ en la carpeta raiz del proyecto."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')   # evita el aviso de joblib al contar nucleos en Windows

# Datos y particion
RAIZ = Path(__file__).resolve().parents[4]
PANEL = Path(os.environ.get('PANEL_ROTACION',
                            RAIZ / 'datos' / 'rotacion' / 'panel_rotacion_2025_2026.csv'))
SEMILLA = 2026
CORTE = '2026-05'          # test = may-2026 a ago-2026 (4 meses), decidido antes del EDA

# Grupos de variables
OBJETIVO = 'y_renuncia'
NUM = ['edad', 'antig_meses']
BIN = ['primer_mes', 'reingreso', 'traslado_12m']
CAT = ['sociedad', 'linea', 'ubicacion', 'genero', 'estado_civil', 'contrato', 'nivel',
       'familia_cargo', 'oficio', 'tipo_unidad', 'area_funcional', 'proceso_planta', 'tipo_costos']
PREDICTORAS = NUM + BIN + CAT
TRAZA = ['funcion', 'cargo_t1', 'departamento', 'area']   # originales de los ejes: no entran
DESENLACE = ['evento', 'tipo_retiro', OBJETIVO]           # describen la salida: no entran

# Nombre legible de cada variable, para titulos y ejes de los graficos
NOMBRE = {
    'edad': 'edad', 'antig_meses': 'antigüedad', 'linea': 'línea de negocio',
    'contrato': 'tipo de contrato', 'nivel': 'nivel del cargo', 'tipo_unidad': 'tipo de unidad',
    'estado_civil': 'estado civil', 'familia_cargo': 'familia de cargo', 'oficio': 'oficio',
    'sociedad': 'sociedad', 'tipo_costos': 'tipo de personal', 'proceso_planta': 'proceso de planta',
    'genero': 'género', 'ubicacion': 'ubicación', 'area_funcional': 'área funcional',
    'antig_t': 'tramo de antigüedad', 'primer_mes': 'primer mes', 'reingreso': 'reingreso',
    'traslado_12m': 'traslado en 12 meses',
}
UNIDAD = {'edad': 'Edad (años)', 'antig_meses': 'Antigüedad (meses)'}

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
    if df.columns.nlevels == 1 and df.columns.name is None and any(df.index.names):
        df = df.copy()
        df.columns.name = ' / '.join(str(n) for n in df.index.names if n is not None)
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
    p = pd.read_csv(PANEL)
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


def etiquetar(ax, barras, fmt='{:,.0f}', textos=None, fontsize=7):
    """Escribe el valor al final de cada barra."""
    if textos is None:
        textos = [fmt.format(v) for v in barras.datavalues]
    ax.bar_label(barras, labels=textos, padding=2, fontsize=fontsize, color=TINTA)
    if barras.orientation == 'horizontal':
        ax.margins(x=0.12)
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
        ax.text(alto, yi, f' {v:.2f} %', va='center', fontsize=7, color=TINTA)
    ax.set_xlim(0, t.ic_alto.max() * 1.25)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f'{i} (n={n:,})' for i, n in zip(t.index, t.filas)], fontsize=8)
    if base is not None:
        ax.axvline(base, ls='--', c=GRIS, lw=0.8)
    ax.set(title=titulo, xlabel='Tasa de renuncia mensual (%)')


def pliegues_temporales(meses, primer_val='2025-09', gap=1):
    """Validacion de ventana creciente por mes: cada pliegue valida un mes y entrena con todos los
    meses anteriores menos `gap` meses de separacion. gap=1 porque la renuncia de un mes puede quedar
    registrada en SAP ya entrado el mes siguiente."""
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
