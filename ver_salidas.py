"""Imprime las salidas de texto de un .ipynb ejecutado (para revisar sin abrir Jupyter).
Uso: python ver_salidas.py <notebook.ipynb> [max_caracteres_por_salida]"""
import sys, json
nb = json.load(open(sys.argv[1], encoding='utf-8')); lim = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    for o in c.get('outputs', []):
        t = o.get('text') or o.get('data', {}).get('text/plain') or ''
        if o.get('output_type') == 'error': t = o['ename'] + ': ' + o['evalue']
        if 'image/png' in o.get('data', {}): t = (t if isinstance(t, str) else ''.join(t)) + ' [imagen]'
        t = ''.join(t) if isinstance(t, list) else t
        if t.strip(): print(f'--- celda {i}\n{t[:lim]}')
