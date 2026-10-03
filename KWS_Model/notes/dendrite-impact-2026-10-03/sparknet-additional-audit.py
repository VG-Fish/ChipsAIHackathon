from pathlib import Path
import json,csv,statistics,re,math
from collections import defaultdict,Counter
from scipy.stats import t
import yaml
P=Path(__file__).resolve().parent; ROOT=P.parents[1]
rows=json.loads((P/'sparknet-all-runs.json').read_text());pairs=json.loads((P/'sparknet-paired-growth.json').read_text())
post=[]
for r in rows:
 if r['kind']!='PAI_posthoc':continue
 c=(r.get('candidates') or [{}])[0];base=c.get('baseline',{});den=c.get('dendritic',{});item={'family':r['family'],'arm':r['arm'],'model':r['model'],'seed':r['seed'],'run':r['run'],'status':r['status'],'source_val':r.get('source',{}).get('validation_accuracy'),'baseline_val':base.get('validation_accuracy'),'final_val':den.get('validation_accuracy'),'search_val':den.get('pai_search_validation_accuracy'),'zero_val':den.get('zero_dendrite_validation_accuracy'),'base_params':base.get('deployed_params'),'deployed_params':den.get('deployed_params'),'macs':den.get('macs'),'comparison':c.get('comparison'),'resume_status':den.get('resume',{}).get('status') if den.get('resume') else None}
 p=ROOT/r['run']; archfiles=[f for f in p.rglob('*_best_arch_scores.csv') if f.name==f.parent.name+'_best_arch_scores.csv']
 item['best_arch_files']=[str(f.relative_to(ROOT)) for f in archfiles]
 item['best_arch_rows']=[]
 for f in archfiles:
  with f.open() as h:
   for x in csv.DictReader(h):item['best_arch_rows'].append({k:float(v) for k,v in x.items() if v})
 switches=[f for f in p.rglob('*switch_epochs.csv') if f.name==f.parent.name+'switch_epochs.csv']
 item['switch_data']=[{'path':str(f.relative_to(ROOT)),'text':f.read_text()} for f in switches]
 post.append(item)
(P/'sparknet-posthoc-runs.json').write_text(json.dumps(post,indent=2))
def stats(xs):
 n=len(xs);m=statistics.mean(xs);s=statistics.stdev(xs) if n>1 else None;half=float(t.ppf(.975,n-1))*s/math.sqrt(n) if s is not None else None
 return {'n':n,'mean':m,'sd':s,'ci95':[m-half,m+half] if half is not None else None,'wins':sum(x>0 for x in xs)}
groups=defaultdict(list)
for r in post:
 if r['status']=='complete':groups[(r['family'],r['arm'],r['model'])].append(r)
controls={(r['family'],r['model'],r['seed']):r for r in post if r['arm']=='control' and r['status']=='complete'}
postagg=[]
for key,rr in sorted(groups.items()):
 a={'family':key[0],'arm':key[1],'model':key[2],'n':len(rr),'final_val_pct':100*statistics.mean(r['final_val'] for r in rr),'params':rr[0]['deployed_params'],'macs':rr[0]['macs'],'retained_growth_runs':sum(r['deployed_params']>r['base_params'] for r in rr),'higher_arch_attempt_runs':sum(any(x.get('Param Counts',0)>r['base_params'] for x in r['best_arch_rows']) for r in rr)}
 for label,field in [('vs_source','source_val'),('vs_finetune','baseline_val'),('vs_pai_zero','zero_val')]:
  xs=[100*(r['final_val']-r[field]) for r in rr if r[field] is not None]
  if xs:a[label]=stats(xs)
 xs=[100*(r['final_val']-controls[(r['family'],r['model'],r['seed'])]['final_val']) for r in rr if (r['family'],r['model'],r['seed']) in controls]
 if xs:a['vs_control']=stats(xs)
 postagg.append(a)
(P/'sparknet-posthoc-aggregates.json').write_text(json.dumps(postagg,indent=2))
# Sham-adjusted growth, exact arm/model/seed matched.
lookup={(r['family'],r['arm'],r['model'],r['seed']):r for r in rows if r['kind']=='PAI_grow' and r['status']=='complete'}
shams=[]
for r in lookup.values():
 if (r.get('variant') or {}).get('sham'):continue
 sham=lookup.get((r['family'],r['arm']+'-sham',r['model'],r['seed']))
 if sham:
  shams.append({'family':r['family'],'arm':r['arm'],'model':r['model'],'seed':r['seed'],'real_run':r['run'],'sham_run':sham['run'],'gain_pp':100*(r['best_val']-sham['best_val']),'final_gain_pp':100*(r['final_val']-sham['final_val'])})
sg=defaultdict(list)
for r in shams:sg[(r['family'],r['arm'],r['model'])].append(r)
sha=[dict(family=k[0],arm=k[1],model=k[2],stats=stats([r['gain_pp'] for r in v]),final_stats=stats([r['final_gain_pp'] for r in v])) for k,v in sorted(sg.items())]
(P/'sparknet-sham-comparisons.json').write_text(json.dumps({'pairs':shams,'aggregates':sha},indent=2))
for a in postagg:print('POST',json.dumps(a))
for a in sha:print('SHAM',json.dumps(a))
print('V3 uncaught seed mismatch',[(r['run'],r['seed'],r.get('seed_report'),r.get('metric_seed_values')) for r in rows if r.get('seed_report') is not None and (r['seed']!=r['seed_report'] or r['metric_seed_values']!=[r['seed']])])
