from pathlib import Path
import json, re, statistics, importlib.util, sys
from collections import defaultdict,Counter
from safetensors import safe_open
BASE=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ptp_audit_extra',BASE/'scripts/analyze_ptp.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
inv=json.loads((OUT/'pruning_inventory.json').read_text()); summaries=json.loads((OUT/'pruning_summaries.json').read_text())
scratch=[r for r in json.loads((OUT/'sparknet-all-runs.json').read_text()) if r['family']=='sparknet-dendritic-study-v2' and r['kind']=='scratch' and r['status']=='complete']

def parse(model):
 m=re.search(r'c(\d+)(?:g(\d+))?',model);return int(m[1]),int(m[2] or 32)
def mac(c,g):return 606*c*c+(9595+101*g)*c+35552+12*g
sg=defaultdict(list)
for r in scratch:
 w,g=parse(r['model']);r['width']=w;r['gate']=g;r['macs']=mac(w,g);sg[(w,g)].append(r)
scratch_points=[]
for (w,g),rs in sg.items():
 scratch_points.append({'width':w,'gate':g,'params':rs[0]['params'],'macs':rs[0]['macs'],'n':len(rs),'val':statistics.mean(r['best_val'] for r in rs),'sd':statistics.stdev(r['best_val'] for r in rs) if len(rs)>1 else None,'runs':[r['run'] for r in rs]})

def agg(vals):return mod.t_summary(list(vals))

# Audit every actual final clean structure (all completed and any partial final artifacts).
structures=[]
for row in inv['candidates']:
 if not row['has_final_clean']:continue
 p=BASE/row['dir']/'pai/candidates'/f"sparknet_c{row['width']}_multilayer"/'final_clean_pai.pt'
 with safe_open(p,framework='pt',device='cpu') as f:
  keys=list(f.keys());branchkeys=[k for k in keys if re.fullmatch(r'fc.layer_array.\d+.weight',k)]
  skips=[k for k in keys if k.startswith('fc.skip_weights.')]
  skip_count=sum(f.get_tensor(k).numel() for k in skips)
  norms={k:float(f.get_tensor(k).norm()) for k in skips}
  branches=len(branchkeys); d=branches-1 if branches else 0
  if branches:g=int(f.get_slice(branchkeys[0]).get_shape()[1])
  else:
   g=int(f.get_slice('fc.weight').get_shape()[1]) if 'fc.weight' in keys else None
  overhead=d*(g*12+12)+skip_count if g else None
  structures.append({'family':row['family'],'seed':row['seed'],'width':row['width'],'file':str(p.relative_to(BASE)),'fc_branch_count':branches or 1,'dendrites':d,'gate':g,'skip_count':skip_count,'skip_norms':norms,'overhead':overhead,'report_overhead':(row['final_params']-row['base_params']) if row['final_params'] is not None else None,'overhead_matches_report':row['final_params']-row['base_params']==overhead if row['final_params'] is not None else None})

extra=[]
for family in summaries['ptp']:
 f=family['family'];g=32 if f=='sparknet-c16-ptp' else 16 if f=='sparknet-c16g16-ptp' else 8
 runs=family['clean']['runs'];lo,hi=[family['clean']['zero_dendrite_reference'][i]['params'] for i in [0,-1]]
 byseed=defaultdict(list);strictseed=defaultdict(list);byrate=defaultdict(list)
 report_rows=[r for r in inv['candidates'] if r['family']==f]
 curve=mod.LogCurve.from_points((p['params'],p['val']) for p in scratch_points if p['gate']==g)
 for r in runs:
  d=r['decomposition'];r['strict_zero_curve_covered']=lo<=d['n_final']<=hi
  byseed[r['seed']].append(d['gain'])
  if r['strict_zero_curve_covered']:strictseed[r['seed']].append(d['gain'])
  byrate[r['prune_rate']].append(r)
 pair=[]
 for r in report_rows:
  st=[s for s in scratch if s['width']==r['width'] and s['gate']==g and s['seed']==r['seed']]
  e={'seed':r['seed'],'width':r['width'],'same_width_scratch':st[0] if st else None,'final_vs_same_width_scratch_pp':100*(r['final_val']-st[0]['best_val']) if st else None,'final_vs_scratch_param_curve_pp':100*(r['final_val']-curve(r['final_params'])),'final_cost_in_scratch_range':curve.points[0][0]<=r['final_params']<=curve.points[-1][0]}
  e['same_gate_parameter_dominators']=[p for p in scratch_points if p['gate']==g and p['params']<=r['final_params'] and p['val']>=r['final_val']]
  e['same_gate_param_mac_dominators']=[p for p in e['same_gate_parameter_dominators'] if p['macs']<=r['final_macs']]
  pair.append(e)
 rate=[]
 for pr,rows in sorted(byrate.items()):
  rate.append({'rate':pr,'width':rows[0]['width'],'n':len(rows),'raw_gain_pp':agg(100*r['decomposition']['raw_gain'] for r in rows),'cost_pp':agg(100*r['decomposition']['param_cost'] for r in rows),'gain_pp':agg(100*r['decomposition']['gain'] for r in rows),'b0_params':rows[0]['budgets']['0']['params'],'b3_params_mean':statistics.mean(r['budgets']['3']['params'] for r in rows),'b3_params_range':[min(r['budgets']['3']['params'] for r in rows),max(r['budgets']['3']['params'] for r in rows)],'strict_coverage':sum(r['strict_zero_curve_covered'] for r in rows),'final_vs_same_width_scratch_pp':agg(p['final_vs_same_width_scratch_pp'] for p in pair if p['width']==rows[0]['width'] and p['final_vs_same_width_scratch_pp'] is not None),'final_vs_scratch_param_curve_pp':agg(p['final_vs_scratch_param_curve_pp'] for p in pair if p['width']==rows[0]['width'])})
 common_rates={rate for rate, rs in byrate.items() if all(r['strict_zero_curve_covered'] for r in rs)}
 strict_common_byseed={seed:[r['decomposition']['gain'] for r in runs if r['seed']==seed and r['prune_rate'] in common_rates] for seed in byseed}
 ex={'strict_common_rates':sorted(common_rates),'strict_common_clustered_seed_gain_pp':agg(100*statistics.mean(v) for v in strict_common_byseed.values()),'family':f,'gate':g,'zero_curve_domain':[lo,hi],'seed_mean_gain_pp':{s:100*statistics.mean(v) for s,v in byseed.items()},'clustered_seed_gain_pp':agg(100*statistics.mean(v) for v in byseed.values()),'strict_seed_mean_gain_pp':{s:100*statistics.mean(v) for s,v in strictseed.items()},'strict_clustered_seed_gain_pp':agg(100*statistics.mean(v) for v in strictseed.values()),'rates':rate,'scratch_pairs':pair}
 extra.append(ex)
 print('\n',f,'ZERO RANGE',lo,hi,'CLUSTERED',ex['clustered_seed_gain_pp'],'STRICT COMMON',ex['strict_common_clustered_seed_gain_pp'])
 for r in rate:print('rate',r['rate'],'width',r['width'],'raw/cost/gain',*[round(r[k]['mean'],3) for k in ['raw_gain_pp','cost_pp','gain_pp']], 'cleanparams',r['b3_params_mean'],r['b3_params_range'],'strict',r['strict_coverage'],'scratch deltas',r['final_vs_same_width_scratch_pp']['mean'],r['final_vs_scratch_param_curve_pp']['mean'])
 print('dominance parameter',sum(bool(p['same_gate_parameter_dominators']) for p in pair),'param+mac',sum(bool(p['same_gate_param_mac_dominators']) for p in pair),'total',len(pair))
print('STRUCTURE',len(structures),'MISMATCHES',[r for r in structures if r['overhead_matches_report'] is False])
print('actual retained counts',Counter((s['family'],s['dendrites']) for s in structures))
(OUT/'pruning_extended.json').write_text(json.dumps({'ptp':extra,'structures':structures,'scratch_points':scratch_points},indent=2))
