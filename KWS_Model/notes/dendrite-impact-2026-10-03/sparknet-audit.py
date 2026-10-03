"""Read-only original-artifact audit; outputs restricted to this new notes folder."""
from pathlib import Path
import json, re, statistics, math, csv
import yaml
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
families=['sparknet-dendritic-study-v2','sparknet-grow-dendrites-v3','sparknet-pai-documented','sparknet-lowdata02','sparknet-lowdata10','sparknet-lowdata20','sparknet-native-dendrites','sparknet-paper-replication']
rows=[]; logs={}
def load(p):
    return yaml.safe_load(p.read_text()) if p.exists() else {}
def avg(xs): return statistics.mean(xs) if xs else None
def sd(xs): return statistics.stdev(xs) if len(xs)>1 else None
for family in families:
    for manifest in sorted((ROOT/'outputs'/family).rglob('manifest.yaml')):
        rr=manifest.parent
        manifest_data=load(manifest)
        rel=rr.relative_to(ROOT/'outputs'/family)
        parts=rel.parts
        seedmatch=re.search(r'seed(\d+)',rr.name)
        seed=int(seedmatch[1]) if seedmatch else None
        row={'family':family,'run':str(rr.relative_to(ROOT)),'label':str(rel),'seed':seed,'arm':(parts[1] if family=='sparknet-dendritic-study-v2' and parts[0]=='arms' else parts[0]) if len(parts)>1 else 'scratch','kind':'scratch'}
        metrics=[]
        for p in sorted((rr/'metrics').rglob('*.jsonl')):
            for line in p.read_text().splitlines():
                if line.strip():
                    try: metrics.append(dict(json.loads(line),_file=str(p.relative_to(ROOT))))
                    except json.JSONDecodeError: pass
        logs[row['run']]=metrics
        row['metric_rows']=len(metrics); row['metric_seed_values']=sorted({m.get('seed') for m in metrics if m.get('seed') is not None})
        grow=rr/'reports/grow_summary.yaml'; prune=rr/'reports/sparknet_dendritic_prune_experiment.yaml'
        if grow.exists():
            d=load(grow); row['kind']='PAI_grow'; row['report']=str(grow.relative_to(ROOT)); row['status']=d.get('status'); row['seed_report']=d.get('seed'); row['arm']=d.get('arm',parts[0]);row['model']=d.get('model_name');row['variant']=d.get('variant');row['schedule']=d.get('schedule');row['checks']=d.get('checks');row['dendrite']=d.get('dendrite');row['diagnostics']=d.get('dendrite_diagnostics');row['results']=d.get('results');row['cost']=d.get('cost');row['pb_scores']=d.get('pb_scores_at_integration');row['test_split_used']=d.get('test_split_used');row['selection_split']=d.get('selection_split');row['data_config']=d.get('data_config');row['train_config']=d.get('train_config')
            row['best_val']=d.get('results',{}).get('best_val_acc_overall');row['final_val']=d.get('results',{}).get('final_val_acc');row['params']=d.get('cost',{}).get('deployed',{}).get('params');row['macs']=d.get('cost',{}).get('deployed',{}).get('macs')
        elif prune.exists():
            d=load(prune); row['kind']='PAI_posthoc';row['report']=str(prune.relative_to(ROOT));row['status']=d.get('status');row['seed_report']=d.get('seed');row['source']=d.get('source');row['pai_config']=d.get('perforatedai');row['candidates']=d.get('candidates');row['test_split_used']=d.get('test_split_used');row['selection_split']=d.get('selection_split');
            if d.get('candidates'):
                c=d['candidates'][0]; row['best_val']=c.get('dendritic',c.get('baseline',{})).get('validation_accuracy');row['params']=c.get('dendritic',c.get('baseline',{})).get('deployed_params');row['macs']=c.get('dendritic',c.get('baseline',{})).get('macs')
            row['model']='sparknet_'+re.sub(r'-seed\d+','',rr.name)+'_paper'
        else:
            summaries=load(rr/'metrics/summaries.yaml')
            phases=summaries.get('phases',{})
            good=[v for v in phases.values() if isinstance(v,dict) and 'best_val_acc' in v]
            if good:
                d=good[-1];row['status']='complete' if d.get('completed_epoch',0)>=200 else 'incomplete';row['best_val']=d.get('best_val_acc');row['final_val']=d.get('final_val_acc');row['summary']=d
            else: row['status']='missing_summary'
            if family=='sparknet-grow-dendrites-v3': row['kind']='PAI_grow_no_report'
            row['model']=next((m.get('phase') for m in metrics if str(m.get('phase','')).startswith('sparknet')), 'sparknet_'+re.sub(r'-seed\d+','',rr.name)+'_paper')
            if metrics:
                row['params']=metrics[-1].get('parameter_count');row['max_epoch']=max(m.get('epoch',0) for m in metrics)
        row['segments']={seg:sum(m.get('segment')==seg for m in metrics) for seg in sorted({m.get('segment') for m in metrics if m.get('segment')})}
        row['max_epoch']=max((m.get('epoch',0) for m in metrics),default=0)
        row['best_metric_val']=max((m.get('val_acc',m.get('val_accuracy',-1)) for m in metrics),default=None)
        row['manifest_status']=manifest_data.get('status')
        row['manifest_command']=manifest_data.get('command')
        if family=='sparknet-paper-replication' and row['status']=='missing_summary':
            row['kind']='released_evaluation' if 'evaluate' in str(row['manifest_command']) else 'scratch_no_metrics'
            row['status']=row['manifest_status']
        rows.append(row)
(OUT/'sparknet-all-runs.json').write_text(json.dumps(rows,indent=2))
# Pair growth with the exact model-name and seed scratch within the same data family.
controls={(r['family'],r['model'],r['seed']):r for r in rows if r['kind']=='scratch' and r['status']=='complete' and r.get('best_val') is not None}
pairs=[]
for r in rows:
    if r['kind']!='PAI_grow' or r['status']!='complete': continue
    cf='sparknet-dendritic-study-v2' if r['family']=='sparknet-grow-dendrites-v3' else r['family']
    c=controls.get((cf,r['model'],r['seed']))
    if c:
        a=logs[r['run']];b=logs[c['run']]; pre=[m for m in a if m.get('segment')=='pre_switch']; bmap={m['epoch']:m for m in b}
        diffs=[abs(m.get('val_acc',m.get('val_accuracy'))-bmap[m['base_epoch']].get('val_acc',bmap[m['base_epoch']].get('val_accuracy'))) for m in pre if m.get('base_epoch') in bmap]
        pairs.append({'family':r['family'],'arm':r['arm'],'model':r['model'],'seed':r['seed'],'run':r['run'],'control_run':c['run'],'val_gain_pp':100*(r['best_val']-c['best_val']),'final_gain_pp':100*(r['final_val']-c['final_val']),'dendritic_best_val':r['best_val'],'control_best_val':c['best_val'],'params':r['params'],'macs':r['macs'],'control_params':c.get('params'),'pre_switch_val_max_abs_diff':max(diffs,default=None),'pre_switch_matched_epochs':len(diffs)})
(OUT/'sparknet-paired-growth.json').write_text(json.dumps(pairs,indent=2))
groups={}
for r in pairs: groups.setdefault((r['family'],r['arm'],r['model']),[]).append(r)
agg=[]
tvals={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262}
for (fam,arm,model),g in sorted(groups.items()):
    vals=[x['val_gain_pp'] for x in g];s=sd(vals);margin=float(t.ppf(.975,len(g)-1))*s/math.sqrt(len(g)) if s is not None else None
    agg.append({'family':fam,'arm':arm,'model':model,'n':len(g),'seeds':[x['seed'] for x in g],'mean_gain_pp':avg(vals),'sd_gain_pp':s,'ci95_low':avg(vals)-margin if margin is not None else None,'ci95_high':avg(vals)+margin if margin is not None else None,'positive_seeds':sum(x>0 for x in vals),'mean_val_pct':100*avg([x['dendritic_best_val'] for x in g]),'control_val_pct':100*avg([x['control_best_val'] for x in g]),'params':g[0]['params'],'macs':g[0]['macs'],'max_pre_switch_val_diff':max(x['pre_switch_val_max_abs_diff'] or 0 for x in g)})
(OUT/'sparknet-growth-aggregates.json').write_text(json.dumps(agg,indent=2))
with (OUT/'sparknet-growth-aggregates.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=agg[0].keys());w.writeheader();w.writerows(agg)
for family in families:
    rr=[r for r in rows if r['family']==family]
    from collections import Counter
    print(family,'runs',len(rr),'kinds',dict(Counter(r['kind'] for r in rr)),'status',dict(Counter(r['status'] for r in rr)))
print('growth pairs',len(pairs),'aggregates',len(agg),'pre-switch mismatches',sum((p['pre_switch_val_max_abs_diff'] or 0)>0 for p in pairs))
for a in agg:
    if a['n']>=3: print(a['family'],a['arm'],a['model'],'n',a['n'],'gain',round(a['mean_gain_pp'],3),'CI',tuple(round(a[k],3) for k in ['ci95_low','ci95_high']),'val',round(a['mean_val_pct'],3),'cost',a['params'],a['macs'],'pre diff',a['max_pre_switch_val_diff'])
