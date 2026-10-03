from pathlib import Path
from collections import defaultdict
import json,statistics,re,math,sys,hashlib
import yaml
from safetensors import safe_open
import torch
from scipy.stats import t
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(ROOT/'src'))
from kws.models.sparknet import build_sparknet
from kws.utils.profile import count_macs
R=json.loads((P/'sparknet-all-runs.json').read_text())
C=[]
for r in R:
 if r['status']!='complete' or r['kind'] not in ['PAI_grow','PAI_posthoc']:continue
 rr=ROOT/r['run'];paths=[]
 if r['kind']=='PAI_grow':
  d=yaml.safe_load((ROOT/r['report']).read_text())
  paths=[(label,rr/d['artifacts'][key]) for label,key in [('final','final_clean'),('best','best_clean')] if key in d.get('artifacts',{})]
 else:
  cand=(r.get('candidates') or [{}])[0];den=cand.get('dendritic') or {};cp=den.get('checkpoint')
  if cp and cp.endswith('final_clean_pai.pt'):paths=[('final',rr/cp)]
  else:paths=[('final',p) for p in (rr/'pai').rglob('final_clean_pai.pt') if '/best_dendritic/' not in str(p)]
 for label,path in paths:
  entry=dict(run=r['run'],family=r['family'],arm=r['arm'],model=r['model'],seed=r['seed'],label=label,path=str(path.relative_to(ROOT)),reported_params=r.get('params'))
  if not path.exists():entry['status']='missing';C.append(entry);continue
  try:
   with safe_open(path,framework='pt',device='cpu') as sf:
    shapes={k:tuple(sf.get_slice(k).get_shape()) for k in sf.keys()}
    dendriteprefix={k.split('.layer_array.')[0] for k in shapes if '.layer_array.' in k}
    entry['status']='read';entry['modules']={}
    for pref in sorted(dendriteprefix):
     ids=sorted({int(k[len(pref+'.layer_array.'):].split('.')[0]) for k in shapes if k.startswith(pref+'.layer_array.')})
     skips=[sf.get_tensor(k).float().flatten() for k in shapes if k.startswith(pref+'.skip_weights.')]
     entry['modules'][pref]=dict(n_dendrites=len(ids)-1,skip_mean_abs=float(torch.cat(skips).abs().mean()) if skips else None,skip_max_abs=float(torch.cat(skips).abs().max()) if skips else None)
    entry['non_bookkeeping_tensor_elements']=sum(math.prod(shape) for k,shape in shapes.items() if not any(s in k for s in ['tracker_string','module_id','node_index','num_cycles','view_tuple','num_batches_tracked','running_mean','running_var']))
    entry['retained_dendrites_total']=sum(x['n_dendrites'] for x in entry['modules'].values());entry['nonzero_skip_modules']=sum((x['skip_max_abs'] or 0)>0 for x in entry['modules'].values())
  except Exception as e:entry['status']='error';entry['error']=str(e)
  C.append(entry)
(P/'sparknet-clean-artifact-audit.json').write_text(json.dumps(C,indent=2))
# Native architectures are constructed from config only; no licensed PAI imported.
N=[];groups=defaultdict(list)
for r in R:
 if r['family']!='sparknet-native-dendrites':continue
 groups[r['model']].append(r)
for model,rr in sorted(groups.items()):
 cfgpath=ROOT/'configs/model'/f'{model}.yaml';cfg=yaml.safe_load(cfgpath.read_text());m=build_sparknet(cfg,(32,101),12)
 params=sum(p.numel() for p in m.parameters());macs=count_macs(m,(32,101));assert all(r['params']==params for r in rr)
 vals=[100*r['best_val'] for r in rr];mean=statistics.mean(vals);sd=statistics.stdev(vals) if len(vals)>1 else None
 N.append(dict(model=model,n=len(rr),seeds=[r['seed'] for r in rr],mean_val_pct=mean,sd_val_pct=sd,params=params,macs=macs,model_config=str(cfgpath.relative_to(ROOT)),native_config={k:cfg[k] for k in ['channels','gate_channels','dendrites','dendrite_fan_in','dendrite_mode','dendrite_blocks'] if k in cfg},runs=[r['run'] for r in rr]))
(P/'sparknet-native-aggregates.json').write_text(json.dumps(N,indent=2))
print('CLEAN',len(C),'read',sum(x['status']=='read' for x in C),'errors',[x for x in C if x['status']!='read'])
for family in sorted(set(x['family'] for x in C)):
 cc=[x for x in C if x['family']==family and x['label']=='final'];print(family,'final artifacts',len(cc),'has_retained',sum(x.get('retained_dendrites_total',0)>0 for x in cc),'nonzero skip',sum(x.get('nonzero_skip_modules',0)>0 for x in cc),'parameter mismatches',sum(x.get('non_bookkeeping_tensor_elements')!=x['reported_params'] for x in cc))
for x in C:
 if x['family']=='sparknet-pai-documented':print('DOC',x)
for x in N: print('NATIVE',x['model'],x['n'],round(x['mean_val_pct'],4),round(x['sd_val_pct'] or 0,4),x['params'],x['macs'])
print('PAI not imported',not any(k.startswith('perforatedai') or k.startswith('perforatedbp') for k in sys.modules))
