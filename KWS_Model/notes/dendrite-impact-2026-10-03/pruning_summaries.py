from pathlib import Path
import importlib.util, json, sys, statistics, copy
from collections import defaultdict, Counter
BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ptp_audit_source',BASE/'scripts/analyze_ptp.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
inv=json.loads((OUT/'pruning_inventory.json').read_text())
all_rows=inv['candidates']; ptp=[]
def stats(values):
    v=[x for x in values if x is not None]
    return {'n':len(v),'mean':statistics.mean(v) if v else None,'sd':statistics.stdev(v) if len(v)>1 else None,'min':min(v) if v else None,'max':max(v) if v else None}
for family in ['sparknet-c16-ptp','sparknet-c16g16-ptp','sparknet-c18g8-ptp']:
    dirs=[BASE/c['dir'] for c in inv['cells'] if c['family']==family]
    runs=mod.load_runs(dirs)
    native=mod.analyze(runs)
    for r in runs:
        for p in r.budgets.values():p.params += 12*p.dendrites*(p.dendrites+1)//2
    clean=mod.analyze(runs)
    out={'family':family,'native':native,'clean':clean}
    for r in clean['runs']:
        orig=next(row for row in all_rows if row['dir']==r['run_dir'].replace(str(BASE)+'/','') and row['width']==r['width'])
        p=r['budgets'][3]
        r['report_final_matches_budget3']={'val_matches':abs(orig['final_val']-p['val_acc'])<1e-9,'params_matches':orig['final_params']==p['params'],'report_final_params':orig['final_params'],'report_final_val':orig['final_val']}
    ptp.append(out)
    print('\n',family)
    print(mod.format_table(clean))
    print('width | base/zero/final val mean(sd) | final test mean(sd) | deltas final-prune/PAIzero | parameters mean | d counts')
    for width in sorted(set(r['width'] for r in clean['runs']), reverse=True):
        group=[r for r in clean['runs'] if r['width']==width]; rows=[r for r in all_rows if r['family']==family and r['width']==width]
        vals=[]
        for key in ['prune_val','zero_val','final_val']:
            s=stats([100*r[key] for r in rows]); vals.append(f"{s['mean']:.3f}({s['sd']:.3f})")
        st=stats([100*r['budgets'][3]['test_acc'] for r in group])
        print(width,' | ','/'.join(vals),' | ',f"{st['mean']:.3f}({st['sd']:.3f})",' | ',stats([r['delta_final_prune_pp'] for r in rows])['mean'],stats([r['delta_pai_zero_pp'] for r in rows])['mean'],' | ',stats([r['final_params'] for r in rows])['mean'],' | ',Counter(r['budgets'][3]['dendrites'] for r in group))

summary=[]
for family in sorted(set(r['family'] for r in all_rows)):
    for w in sorted(set(r['width'] for r in all_rows if r['family']==family),reverse=True):
        rows=[r for r in all_rows if r['family']==family and r['width']==w]; comp=[r for r in rows if r['status']=='complete']
        row={'family':family,'width':w,'candidates':len(rows),'complete':len(comp),'stats':{k:stats([r[k] for r in comp]) for k in ['source_val','prune_val','zero_val','pai_val','final_val','final_params','final_macs','delta_final_prune_pp','delta_pai_zero_pp','delta_final_source_pp','delta_resume_pai_pp','pai_epochs','prune_epochs']},'retained':dict(Counter(r['retained_dendrites_copied_count'] for r in comp))}
        summary.append(row)
print('\nOLD FC SUMMARY')
for r in summary:
    if r['family']=='sparknet-c16-dendritic-prune-no-kd-fc-only-d3':print(r)
print('HASH DUPLICATES')
for key in ['prune_jsonl_sha256','pai_jsonl_sha256','final_clean_sha256']:
    d=defaultdict(list)
    for r in all_rows:
        if r[key]:d[r[key]].append((r['family'],r['seed'],r['width']))
    print(key,[v for v in d.values() if len(v)>1])
print('LOGGED SEEDS')
for f in sorted(set(r['family'] for r in all_rows)):
    print(f,{r['seed']:r['logged_seed_values'] for r in all_rows if r['family']==f and r['pai_epochs']})
print('SOURCE PARENT STATS')
for f in sorted(set(c['family'] for c in inv['cells'])):
    cs=[c for c in inv['cells'] if c['family']==f]
    print(f,stats([(c['source'] or {}).get('validation_accuracy') for c in cs]),[c['source'] for c in cs[:1]])
(OUT/'pruning_summaries.json').write_text(json.dumps({'ptp':ptp,'family_width_summary':summary},indent=2))
