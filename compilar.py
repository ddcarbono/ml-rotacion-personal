"""Convierte fuentes/*.py (formato percent: '# %%' codigo, '# %% [markdown]' texto) en .ipynb, los
ejecuta con el kernel del curso y deja los .ipynb con salidas junto al libro. Luego: jupyter-book build .
Uso (con ml_venv): python compilar.py [nombre_sin_extension ...]"""
import sys, re, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

AQUI = Path(__file__).resolve().parent
KERNEL = 'ml_venv'

def a_notebook(src):
    """Una celda por marcador '# %%'. Un marcador de codigo puede llevar etiquetas de Jupyter Book,
    por ejemplo '# %% tags=["hide-output"]' para plegar una salida larga."""
    celdas, tipo, buf, etiquetas = [], None, [], []
    def cerrar():
        if tipo is None:
            return
        txt = '\n'.join(buf).strip('\n')
        if tipo == 'md':
            txt = '\n'.join(re.sub(r'^# ?', '', l) for l in txt.split('\n'))
            celdas.append(nbformat.v4.new_markdown_cell(txt))
        elif txt.strip():
            celda = nbformat.v4.new_code_cell(txt)
            if etiquetas:
                celda.metadata['tags'] = etiquetas
            celdas.append(celda)
    for l in src.read_text(encoding='utf-8').split('\n'):
        if l.startswith('# %%'):
            cerrar()
            tipo = 'md' if '[markdown]' in l else 'code'
            buf = []
            m = re.search(r'tags=\[(.*?)\]', l)
            etiquetas = re.findall(r'"([^"]+)"', m.group(1)) if m else []
        else:
            buf.append(l)
    cerrar()
    nb = nbformat.v4.new_notebook(cells=celdas)
    nb.metadata['kernelspec'] = {'name': KERNEL, 'display_name': 'Python 3.9 (ml_venv)', 'language': 'python'}
    return nb

nombres = sys.argv[1:] or [f.stem for f in sorted((AQUI / 'fuentes').glob('*.py'))]
for n in nombres:
    t = time.time(); nb = a_notebook(AQUI / 'fuentes' / f'{n}.py')
    NotebookClient(nb, timeout=1800, kernel_name=KERNEL, resources={'metadata': {'path': str(AQUI)}}).execute()
    nbformat.write(nb, AQUI / f'{n}.ipynb')
    print(f'{n}: {len(nb.cells)} celdas, {time.time() - t:.0f} s')
