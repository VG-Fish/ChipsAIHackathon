"""Read-only audit of the seven assigned SparkNet pruning families."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, statistics
import yaml

BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FAMILIES=['sparknet-c12-dendritic-prune-no-kd-fc-only-d3','sparknet-c12-dendritic-prune-no-kd-unlimited','sparknet-c16-dendritic-prune-no-kd-fc-only-d3','sparknet-c16-dendritic-prune-no-kd-gate-conv-d3','sparknet-c16-ptp','sparknet-c16g16-ptp','sparknet-c18g8-ptp']

def yload(p):
    return yaml.safe_load(p.read_text()) if p.exists() else {}
def jload(p):
    if not p.exists():return []
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def csvload(p):
    if not p.exists():return []
    return list(csv.DictReader(p.open()))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

def short_records(records):
    by={r['epoch']:r for r in records}
    return [by[k] for k in sorted(by)]

def transitions(records):
    last=None; out=[]
    for r in records:
        mode=r.get('pai_mode',r.get('mode'))
        if mode!=last:out.append({'epoch':r['epoch'],'mode':mode});last=mode
    return out

cells=[]; raw=[]
for family in FAMILIES:
    root=BASE/'outputs'/family
    dirs=sorted(p for p in root.glob('seed*') if p.is_dir()) if any(p.is_dir() for p in root.glob('seed*')) else [root]
    for cell in dirs:
        report_path=cell/'reports/sparknet_dendritic_prune_experiment.yaml'
        rep=yload(report_path); man=yload(cell/'manifest.yaml')
        candmap={int(c['width']):c for c in rep.get('candidates',[])}
        widths=set(candmap)
        for p in (cell/'metrics/sparsity').glob('*'):
            import re
            m=re.search(r'sparknet_c(\d+)',p.name)
            if m:widths.add(int(m[1]))
        artifact_counts=Counter(p.suffix for p in cell.rglob('*') if p.is_file())
        input_checks=[]
        for inp in man.get('inputs',[]):
            p=Path(inp['path'])
            input_checks.append({'role':inp.get('role'),'path':str(p),'stored_sha256':inp.get('sha256'),'current_sha256':sha(p),'matches':sha(p)==inp.get('sha256')})
        cells.append({'family':family,'dir':str(cell.relative_to(BASE)),'report_status':rep.get('status'),'manifest_status':man.get('status'),'seed':rep.get('seed') if rep.get('seed') is not None else (int(cell.name[4:]) if cell.name.startswith('seed') and cell.name[4:].isdigit() else man.get('seed')),'source':rep.get('source'),'perforatedai':rep.get('perforatedai'),'knowledge_distillation':rep.get('knowledge_distillation'),'budget':rep.get('budget'),'budget_enforced':rep.get('budget_enforced'),'widths':sorted(widths),'candidate_statuses':dict(Counter(c.get('status') for c in candmap.values())),'artifact_counts':dict(artifact_counts),'input_checks':input_checks,'invocations':[{k:v for k,v in x.items() if k in ['status','started_at','ended_at','error','argv','log']} for x in man.get('invocations',[])]})
        for w in sorted(widths):
            c=candmap.get(w,{})
            b=c.get('baseline') or {}; d=c.get('dendritic') or {}; comp=c.get('comparison') or {}; res=d.get('resume') or {}
            name=f'sparknet_c{w}_multilayer'; pai_dir=cell/'pai/candidates'/name
            logs={phase:short_records(jload(cell/'metrics/sparsity'/candidate/f'{phase}.jsonl')) for phase,candidate in [('prune_supervised',f'sparknet_c{w}'),('pai',name),('resume_supervised',name)]}
            epochs=logs['pai']; arch=csvload(pai_dir/f'{name}_best_arch_scores.csv'); switch=csvload(pai_dir/f'{name}switch_epochs.csv'); pb=csvload(pai_dir/f'{name}Best PBScores.csv')
            pdeltas=sorted(set(r.get('evaluated_parameter_count',r.get('parameter_count')) for r in epochs))
            pp=(b.get('deployed_params')); projection=d.get('one_dendrite_cost_projection') or {}; copied=projection.get('copied_params_per_dendrite')
            retained=None
            if pp is not None and d.get('pai_deployed_params') is not None and copied:
                retained=(d['pai_deployed_params']-pp)/copied
            z=d.get('zero_dendrite_validation_accuracy')
            if z is None and arch:
                z=float(min(arch,key=lambda row:float(row.get('Param Counts',float('inf'))))['Max Valid Scores'])
            row={'family':family,'dir':str(cell.relative_to(BASE)),'seed':rep.get('seed') if rep.get('seed') is not None else (int(cell.name[4:]) if cell.name.startswith('seed') and cell.name[4:].isdigit() else man.get('seed')),'width':w,'status':c.get('status'),'report_path':str(report_path.relative_to(BASE)),'source_val':(rep.get('source') or {}).get('validation_accuracy'),'source_params':(rep.get('source') or {}).get('deployed_params'),'source_macs':(rep.get('source') or {}).get('macs'),'prune_rate':(c.get('group_prune') or {}).get('prune_rate'),'prune_val':b.get('validation_accuracy'),'base_params':pp,'base_macs':b.get('macs'),'pai_val':d.get('pai_search_validation_accuracy'),'zero_val':z,'final_val':d.get('validation_accuracy'),'final_params':d.get('deployed_params'),'final_macs':d.get('macs'),'pai_params':d.get('pai_deployed_params'),'retained_dendrites_copied_count':retained,'resume_status':res.get('status'),'resume_val':res.get('resume_best_val_acc'),'prune_epochs':len(logs['prune_supervised']),'pai_epochs':len(epochs),'resume_epochs':len(logs['resume_supervised']),'logged_seed_values':sorted(set(r.get('seed') for phase in logs.values() for r in phase), key=str),'pai_logged_counts':pdeltas,'pai_modes':transitions(epochs),'pai_architecture_rows':arch,'switch_epochs':switch,'pb_scores':pb,'has_final_clean':(pai_dir/'final_clean_pai.pt').exists(),'has_cycle_meta':(pai_dir/'cycle_metadata.yaml').exists(),'failed_retry_files':len(list(pai_dir.glob('*noImprove_lr*'))),'test_logging_epochs':sum(r.get('test_accuracy') is not None for r in epochs),'train_eval_epochs':sum(r.get('train_eval_accuracy') is not None for r in epochs),'prune_jsonl_sha256':sha(cell/'metrics/sparsity'/f'sparknet_c{w}'/'prune_supervised.jsonl'),'pai_jsonl_sha256':sha(cell/'metrics/sparsity'/name/'pai.jsonl'),'final_clean_sha256':sha(pai_dir/'final_clean_pai.pt'),'comparison':comp}
            for key, a,zv in [('delta_final_prune_pp',row['final_val'],row['prune_val']),('delta_pai_zero_pp',row['pai_val'],row['zero_val']),('delta_final_source_pp',row['final_val'],row['source_val']),('delta_resume_pai_pp',row['final_val'],row['pai_val'])]:
                row[key]=100*(a-zv) if a is not None and zv is not None else None
            raw.append(row)

(OUT/'pruning_inventory.json').write_text(json.dumps({'cells':cells,'candidates':raw},indent=2))
fields=[k for k,v in raw[0].items() if not isinstance(v,(dict,list))]
with (OUT/'pruning_candidates.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(raw)
print('CELLS')
for c in cells:
    print(c['dir'], 'seed',c['seed'],'report',c['report_status'],'manifest',c['manifest_status'],c['candidate_statuses'])
print('CANDIDATES')
for r in raw:
    print(r['family'].replace('sparknet-',''),r['seed'],r['width'],r['status'],'val',r['prune_val'],r['zero_val'],r['pai_val'],r['final_val'],'cost',r['base_params'],r['final_params'],r['final_macs'],'d',r['retained_dendrites_copied_count'],'ep',r['prune_epochs'],r['pai_epochs'],r['resume_epochs'])
