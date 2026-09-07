import json
from pathlib import Path

root = Path('data/rag-experiments/runs')
rows = []
for path in sorted(root.glob('*/metrics.json')):
    obj = json.loads(path.read_text(encoding='utf-8'))
    m = obj.get('metrics', {})
    rows.append((obj.get('experiment_id', path.parent.name), obj.get('index_strategy',''), obj.get('total_cases',0), m.get('hit_rate'), m.get('mrr'), m.get('recall'), m.get('precision')))
out = root / 'comparison.csv'
lines = ['experiment_id,index_strategy,total_cases,hit_rate,mrr,recall,precision']
lines += [','.join(map(str, row)) for row in rows]
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
md = root / 'comparison.md'
table = ['| Experiment | Strategy | Cases | HitRate | MRR | Recall | Precision |','|---|---|---:|---:|---:|---:|---:|']
table += [f'| {r[0]} | {r[1]} | {r[2]} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.4f} |' for r in rows]
md.write_text('\n'.join(table) + '\n', encoding='utf-8')
print(json.dumps({'count': len(rows), 'csv': str(out), 'markdown': str(md)}, ensure_ascii=False))
