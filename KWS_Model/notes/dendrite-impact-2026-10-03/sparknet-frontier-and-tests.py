from pathlib import Path
from collections import defaultdict
import json,re,statistics,math
from scipy.stats import t
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
R=json.loads((P/'sparknet-all-runs.json').read_text());G=json.loads((P/'sparknet-growth-aggregates.json').read_text())
def stats(xs):
 n=len(xs);mean=statistics.mean(xs);sd=statistics.stdev(xs) if n>1 else None;h=float(t.ppf(.975,n-1))*sd/math.sqrt(n) if sd is not None else None
 return dict(n=n,mean=mean,sd=sd,ci95=[mean-h,mean+h] if h is not None else None)
def costs(model):
 match=re.fullmatch(r'sparknet_c(\d+)(?:g(\d+))?_paper',model)
 if not match:return None
 c=int(match[1]);g=int(match[2] or 32)
 return dict(channels=c,gate_channels=g,params=6*c*c+(109+g)*c+364+15*g,macs=606*c*c+101*(95+g)*c+35552+12*g)
scratch=defaultdict(list);lookup={}
for r in R:
 if r['kind']=='scratch' and r['status']=='complete' and r.get('best_val') is not None:
  co=costs(r['model']);r['analytic_cost']=co
  if co and r.get('params') is not None:assert co['params']==r['params']
  scratch[(r['family'],r['model'])].append(r)
  lookup[(r['family'],r['model'],r['seed'])]=r
S=[]
for (family,model),rr in sorted(scratch.items()):
 vals=[100*r['best_val'] for r in rr];a=dict(family=family,model=model,stats=stats(vals),seeds=[r['seed'] for r in rr],cost=rr[0]['analytic_cost'],runs=[r['run'] for r in rr]);S.append(a)
(P/'sparknet-scratch-aggregates.json').write_text(json.dumps(S,indent=2))
# Mean frontier with exactly the same seed set per growth group, no cross-protocol test/val mixing.
F=[]
for gr in G:
 cf='sparknet-dendritic-study-v2' if gr['family']=='sparknet-grow-dendrites-v3' else gr['family']
 dominators=[]
 for (fam,model),ss in scratch.items():
  if fam!=cf:continue
  co=costs(model)
  if not co or co['params']>gr['params'] or co['macs']>gr['macs']:continue
  selected=[lookup.get((fam,model,seed)) for seed in gr['seeds']]
  if any(x is None for x in selected):continue
  mean=100*statistics.mean(x['best_val'] for x in selected)
  if mean>=gr['mean_val_pct']:
   dominators.append(dict(model=model,mean_val_pct=mean,params=co['params'],macs=co['macs'],gain_over_growth_pp=mean-gr['mean_val_pct']))
 F.append(dict(family=gr['family'],arm=gr['arm'],model=gr['model'],n=gr['n'],mean_val_pct=gr['mean_val_pct'],params=gr['params'],macs=gr['macs'],conventional_dominators=sorted(dominators,key=lambda x:x['params'])))
(P/'sparknet-conventional-dominance.json').write_text(json.dumps(F,indent=2))
# Unique heldout data; no re-evaluation. Derive FAR/FRR from stored confusion matrices.
T=[]
for file in [ROOT/'outputs/sparknet-dendritic-study-v2/selection/test_report.json',ROOT/'outputs/plots/sparknet-dendritic-comparison/broader_v3_test_accuracy_all.json']:
 d=json.loads(file.read_text())
 for e in d['evaluations']:
  cm=e['metrics']['confusion_matrix'];n=sum(map(sum,cm));keyword=sum(map(sum,cm[:10]));nonkey=sum(map(sum,cm[10:]));correct=sum(cm[i][i] for i in range(len(cm)));false_accept=sum(sum(row[:10]) for row in cm[10:]);false_reject=keyword-sum(cm[i][i] for i in range(10))
  acc=correct/n;assert abs(acc-e['test_accuracy'])<1e-12
  T.append(dict(source=str(file.relative_to(ROOT)),arm=e['arm'],width=e['width'],seed=e['seed'],checkpoint=e['checkpoint'],test_accuracy=acc,raw_far=e['test_far'],raw_frr=e['test_frr'],corrected_keyword10_far=false_accept/nonkey,corrected_keyword10_frr=false_reject/keyword,n_samples=n,non_keyword_samples=nonkey,keyword_samples=keyword,selection_validation_accuracy=e.get('validation_accuracy'),num_params=e.get('num_params')))
# Published-port c16 test reports preserved separate.
for f in sorted((ROOT/'outputs/sparknet-paper-replication').glob('c16-seed*/test_report_fixedseed0.json')):
 d=json.loads(f.read_text());seed=int(re.search(r'seed(\d+)',str(f))[1]);T.append(dict(source=str(f.relative_to(ROOT)),arm='paper_c16',seed=seed,test_accuracy=d['accuracy'],raw_far=d['far'],raw_frr=d['frr'],num_params=d['num_params']))
(P/'sparknet-test-evidence.json').write_text(json.dumps(T,indent=2))
for x in F:
 if x['n']>=3 and (x['arm'].startswith('pointwise_b2') or x['family'].startswith('sparknet-lowdata')):
  print('DOMINANCE',x['family'],x['arm'],x['model'],[(d['model'],round(d['gain_over_growth_pp'],3)) for d in x['conventional_dominators']])
for x in T:
 if 'broader' in x['source']:print('TEST',x['arm'],'accuracy',100*x['test_accuracy'],'FAR',100*x['corrected_keyword10_far'],'FRR',100*x['corrected_keyword10_frr'])
# Exact v2 scratch versus conventional continuation test: group width and seed.
v2=[x for x in T if 'selection/test_report' in x['source']];ctr={(x['width'],x['seed']):x for x in v2 if x['arm']=='control'}
for width in sorted(set(x['width'] for x in v2)):
 for arm in sorted(set(x['arm'] for x in v2)):
  rr=[x for x in v2 if x['arm']==arm and x['width']==width];g=[100*(x['test_accuracy']-ctr[(width,x['seed'])]['test_accuracy']) for x in rr if (width,x['seed']) in ctr]
  print('V2TEST',width,arm,'mean',100*statistics.mean(x['test_accuracy'] for x in rr),'vs_control',stats(g))
