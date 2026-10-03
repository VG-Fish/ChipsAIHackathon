"""Descriptive seed-paired effects from existing JSON; no training or PAI imports."""
from pathlib import Path
import csv,json,statistics,math
OUT=Path(__file__).resolve().parent;KWS=OUT.parents[1]
rows=list(csv.DictReader((OUT/'faithful-run-inventory.csv').open()))
# 95% Student t multipliers for descriptive paired intervals, n=3 or n=5.
T={3:4.3026527299,5:2.7764451052}
def select(batch,arm,frac=1,opt='tracked'):
 return {int(r['seed']):r for r in rows if r['status']=='evaluated' and r['batch']==batch and r['arm']==arm and float(r['train_fraction'])==frac and r['phase0_optimizer']==opt}
def spec(batch,arm,frac=1,opt='tracked'):return (batch,arm,frac,opt)
B='pai-faithful';D='pai-faithful-b21';E='pai-faithful-b22';F='pai-faithful-b23';G='pai-faithful-b24'
pairs=[]
def add(label,a,b):pairs.append((label,a,b))
for arm in ('pb-all-max1-tanh-sw25-c4x8','gd-all-max1-tanh-sw25-c4x8','pb-all-max1-relu-sw25-c4x8'):
 add(arm+' vs same base',spec(B,arm),spec(B,'none-c4x8'))
 add(arm+' vs width',spec(B,arm),spec(B,'none-c8x16'))
add('PB vs GD low width',spec(B,'pb-all-max1-tanh-sw25-c4x8'),spec(B,'gd-all-max1-tanh-sw25-c4x8'))
for control in ('none-c8x16','none-r25-c8x16','none-c16x32','none-r25-c16x32'):
 add('main PB c8x16 vs '+control,spec(B,'pb-all-max1-tanh-sw25-c8x16'),spec(B,control))
for control in ('none-c8x16','none-r25-c8x16','none-c16x32','none-c28x56'):
 add('main linear max3 vs '+control,spec(B,'pb-linear-max3-tanh-sw25-c8x16'),spec(B,control))
add('full tracked ReLU vs base',spec(D,'pb-all-max1-relu-sw25-c8x16'),spec(B,'none-c8x16'))
add('full detached ReLU vs detached base',spec(D,'pb-all-max1-relu-sw25-c8x16',opt='detached'),spec(D,'none-c8x16',opt='detached'))
add('full ReLU detached minus tracked',spec(D,'pb-all-max1-relu-sw25-c8x16',opt='detached'),spec(D,'pb-all-max1-relu-sw25-c8x16'))
for arm in ('pb-all-max1-tanh-sw25-c8x16','pb-all-max1-relu-sw25-c8x16','gd-all-max1-tanh-sw25-c8x16'):
 for control in ('none-c8x16','none-r25-c8x16','none-c16x32','none-r25-c16x32'):
  add('20pct '+arm+' vs '+control,spec(D,arm,.2),spec(D,control,.2))
add('20pct PB vs GD tanh',spec(D,'pb-all-max1-tanh-sw25-c8x16',.2),spec(D,'gd-all-max1-tanh-sw25-c8x16',.2))
add('20pct detached ReLU vs detached base',spec(D,'pb-all-max1-relu-sw25-c8x16',.2,'detached'),spec(D,'none-c8x16',.2,'detached'))
add('20pct PB c4 vs base',spec(D,'pb-all-max1-tanh-sw25-c4x8',.2),spec(D,'none-c4x8',.2))
add('20pct PB c4 vs width',spec(D,'pb-all-max1-tanh-sw25-c4x8',.2),spec(D,'none-c8x16',.2))
add('CPU b22 sigmoid minus tanh',spec(E,'pb-all-max1-sigmoid-sw25-c8x16'),spec(E,'pb-all-max1-tanh-sw25-c8x16'))
add('CPU b22 PB c8 vs width c16x32',spec(E,'pb-all-max1-tanh-sw25-c8x16'),spec(E,'none-c16x32'))
add('CPU b22 PB c8 vs 3conv same params',spec(E,'pb-all-max1-tanh-sw25-c8x16'),spec(E,'none-c12x32x64'))
add('CPU b22 PB c12 vs base',spec(E,'pb-all-max1-tanh-sw25-c12x24'),spec(E,'none-c12x24'))
add('CPU b22 PB c12 vs 3conv same params',spec(E,'pb-all-max1-tanh-sw25-c12x24'),spec(E,'none-c16x42x84'))
for width,wide in [('c6x12x24','c8x20x40'),('c8x20x40','c12x32x64')]:
 add('CPU b23 PB '+width+' vs base',spec(F,'pb-all-max1-tanh-sw25-'+width),spec(F,'none-'+width))
 add('CPU b23 PB '+width+' vs width',spec(F,'pb-all-max1-tanh-sw25-'+width),spec(F if 'c8x20' in wide else E,'none-'+wide))
for conversion,wide in [('all','c17x50x100'),('linear','c16x41x82')]:
 arm='pb-'+conversion+'-max1-tanh-sw25-c12x32x64'
 add('CPU b24 '+conversion+' PB vs base b22',spec(G,arm),spec(E,'none-c12x32x64'))
 add('CPU b24 '+conversion+' PB vs near-matched width',spec(G,arm),spec(G,'none-'+wide))
add('CPU b24 all PB vs 4conv near-matched params',spec(G,'pb-all-max1-tanh-sw25-c12x32x64'),spec(G,'none-c12x32x64x94'))
add('CPU b24 all PB minus linear PB',spec(G,'pb-all-max1-tanh-sw25-c12x32x64'),spec(G,'pb-linear-max1-tanh-sw25-c12x32x64'))
result=[];budgets=[]
for label,a,b in pairs:
 aa=select(*a);bb=select(*b);seeds=sorted(aa.keys()&bb.keys());n=len(seeds)
 if not n:raise ValueError(label)
 for metric in ('val_accuracy','test_accuracy'):
  delta=[100*(float(aa[s][metric])-float(bb[s][metric])) for s in seeds]
  mean=statistics.mean(delta);sd=statistics.stdev(delta) if n>1 else None
  half=T[n]*sd/math.sqrt(n) if n in T else None
  result.append(dict(comparison=label,metric=metric,n=n,seeds=','.join(map(str,seeds)),
   delta_mean_pp=mean,delta_sd_pp=sd,ci95_low_pp=mean-half if half is not None else None,
   ci95_high_pp=mean+half if half is not None else None,wins=sum(d>0 for d in delta),ties=sum(d==0 for d in delta),
   paired_deltas_pp=','.join(f'{d:.6f}' for d in delta),arm_spec=json.dumps(a),control_spec=json.dumps(b),
   arm_params=','.join(sorted({aa[s]['params_live_selected_numel'] for s in seeds})),control_params=','.join(sorted({bb[s]['params_live_selected_numel'] for s in seeds})),
   arm_macs_proxy=','.join(sorted({aa[s]['selected_conv_linear_plus_top_macs_proxy'] for s in seeds})),
   control_macs_proxy=','.join(sorted({bb[s]['selected_conv_linear_plus_top_macs_proxy'] for s in seeds}))))
 for budget in (100,200,300,400,500,600):
  vals=[]
  for seed in seeds:
   ds=[]
   for rr in (aa[seed],bb[seed]):
    d=json.loads((KWS/rr['path']/'result.json').read_text())
    if d['epochs_run']<=budget:ds.append(d)
    else:ds.append(next((x for x in d.get('budget_snapshots',[]) if x['budget_epochs']==budget),None))
   if all(x is not None and x.get('test_accuracy') is not None for x in ds):
    vals.append((ds[0]['test_accuracy']-ds[1]['test_accuracy'])*100)
  if vals:budgets.append(dict(comparison=label,budget_epochs=budget,n=len(vals),test_delta_pp=statistics.mean(vals)))
for name,rs in [('faithful-paired-effects.csv',result),('faithful-equal-epoch-budget-effects.csv',budgets)]:
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rs[0].keys());w.writeheader();w.writerows(rs)
print('Descriptive pairs',len(pairs),'metric rows',len(result),'budget rows',len(budgets))
for r in result:
 if r['metric']=='test_accuracy' and any(t in r['comparison'] for t in ('CPU','PB vs GD','main PB','20pct PB vs GD')):
  print(r['comparison'],f"{r['delta_mean_pp']:+.3f}pp [{r['ci95_low_pp']:+.3f},{r['ci95_high_pp']:+.3f}]",'wins',r['wins'],'cost',r['arm_params'],r['control_params'],'MAC',r['arm_macs_proxy'],r['control_macs_proxy'])
