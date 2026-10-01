"""Une la introduccion y los siete capitulos ya ejecutados en un solo .ipynb con salidas (entrega de D2L).
Uso (con ml_venv): python unir_cuadernos.py [salida.ipynb]"""
import sys
import uuid
from pathlib import Path

import nbformat

AQUI = Path(__file__).resolve().parent
CAPITULOS = ['01_datos', '02_particion', '03_eda_objetivo', '04_eda_multivariado', '05_eda_temporal',
             '06_auditoria_fuga', '07_modelo_base']
salida = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI / 'entregable1_rotacion_personal.ipynb'

libro = nbformat.v4.new_notebook()
libro.cells.append(nbformat.v4.new_markdown_cell((AQUI / 'intro.md').read_text(encoding='utf-8')))
for cap in CAPITULOS:
    nb = nbformat.read(AQUI / f'{cap}.ipynb', as_version=4)
    if not libro.metadata:
        libro.metadata = nb.metadata
    for c in nb.cells:
        c['id'] = uuid.uuid4().hex[:8]  # ids unicos en el cuaderno unido
        libro.cells.append(c)
nbformat.write(libro, salida)
print(f'{salida.name}: {len(libro.cells)} celdas, {salida.stat().st_size / 1e6:.1f} MB')
