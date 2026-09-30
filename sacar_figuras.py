"""Guarda las figuras de un .ipynb ejecutado como PNG (para revisarlas). Uso: python sacar_figuras.py <nb> <carpeta>"""
import sys, json, base64, os
nb = json.load(open(sys.argv[1], encoding='utf-8')); os.makedirs(sys.argv[2], exist_ok=True)
for i, c in enumerate(nb['cells']):
    for j, o in enumerate(c.get('outputs', [])):
        if 'image/png' in o.get('data', {}):
            open(f'{sys.argv[2]}/{os.path.basename(sys.argv[1])[:2]}_c{i}_{j}.png', 'wb').write(base64.b64decode(o['data']['image/png']))
