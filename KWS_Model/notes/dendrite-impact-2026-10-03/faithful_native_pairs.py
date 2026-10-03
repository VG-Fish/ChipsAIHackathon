"""Descriptive native architecture seed-paired validation comparisons."""
from pathlib import Path
import csv,statistics,math
O=Path(__file__).resolve().parent
rs=list(csv.DictReader((O/'faithful-native-students-inventory.csv').open()))
cost={r['model_config']:r for r in csv.DictReader((O/'faithful-native-model-costs.csv').open())}
def get(group,model):return {int(r['seed']):r for r in rs if r['group']==group and r['model']==model}
pairs=[]
def add(g,a,b,why):pairs.append((g,a,b,why))
g='students-dnn'
for a,b in [('dnn_h16_d2f32','dnn_h16'),('dnn_h16_d2f32','dnn_h18'),('dnn_h16_d2f32','dnn_h16x59'),('dnn_h16_d4f32','dnn_h16'),('dnn_h16_d4f32','dnn_h20'),('dnn_h16_d4f32','dnn_h16x112'),('dnn_h16_d2f32','dnn_h16_d2f32lin'),('dnn_h16_d4f32','dnn_h16_d4f32lin'),('dnn_h16_d4f32','dnn_h16_d4f32rnd'),('dnn_h16_d4f32head','dnn_h16'),('dnn_h16_d4f32head','dnn_h18'),('dnn_h16_d4f32hid','dnn_h16'),('dnn_h32_d2f32','dnn_h32'),('dnn_h32_d2f32','dnn_h35'),('dnn_h16x59_hd2','dnn_h16x59'),('dnn_h16x59_hd2','dnn_h16x87'),('dnn_h16x59_hd2','dnn_h16x59x21'),('dnn_h16x59_hd4','dnn_h16x59'),('dnn_h16x59_hd4','dnn_h16x112'),('dnn_h16x59_hd4','dnn_h16x59x32')]:add(g,a,b,'native additive/local activation/depth comparison')
g='students-dscnn'
for a,b in [('ds_cnn_w20d1f4_mfcc','ds_cnn_w20_mfcc'),('ds_cnn_w20d2f4_mfcc','ds_cnn_w20_mfcc'),('ds_cnn_w24d2f4_mfcc','ds_cnn_w24_mfcc'),('ds_cnn_w24d2f4_mfcc','ds_cnn_w28_mfcc'),('ds_cnn_w24fcd4f8_mfcc','ds_cnn_w24_mfcc'),('ds_cnn_w24fcd4f8_mfcc','ds_cnn_w28_mfcc')]:add(g,a,b,'native additive vs base/width')
g='students-dtnet'
for a,b in [('dtnet_a_het','dtnet_a_point'),('dtnet_a_het','dtnet_a_point3'),('dtnet_b_het','dtnet_b_point'),('dtnet_a_het','dtnet_a_lin'),('dtnet_a_het','dtnet_a_shared'),('dtnet_a_het','dtnet_a_notau'),('dtnet_a_het','dtnet_a_rnd'),('dtnet_a_het','dtnet_a_mfcc')]:add(g,a,b,'native timescale/locality/frontend ablation; widths differ in budget controls')
g='students-msd'
for a,b in [('sparknet_msd_a_relu','sparknet_msd_a_lin'),('sparknet_msd_a_relu','sparknet_msd_a_dense'),('sparknet_msd_a_relu','sparknet_msd_a_1scale'),('sparknet_msd_a13_relu','sparknet_msd_a13_lin'),('sparknet_msd_a13_relu','sparknet_msd_a13_dense'),('sparknet_msd_b_relu','sparknet_msd_b_lin'),('sparknet_msd_b_relu','sparknet_msd_b_dense')]:add(g,a,b,'native temporal receptive field control; widths/gate may differ')
for g in ('students-ei','students-ei-adamw'):
 for a,b in [('ei_c4x8_hd2f32','ei_c4x8'),('ei_c8x16_hd2f64','ei_c8x16'),('ei_c8x16_hd2f64','ei_c10x20')]:add(g,a,b,'native head; single seed')
for a,b in [('ei_c4x8_hd1f32','ei_c4x8'),('ei_c4x8_hd4f32','ei_c4x8'),('ei_c8x16_hd1f64','ei_c8x16'),('ei_c8x16_hd1f64','ei_c9x18')]:add('students-ei',a,b,'native head; single seed')
result=[]
for g,a,b,why in pairs:
 aa=get(g,a);bb=get(g,b);ss=sorted(aa.keys()&bb.keys());n=len(ss)
 delta=[100*(float(aa[s]['best_val_accuracy'])-float(bb[s]['best_val_accuracy'])) for s in ss]
 mean=statistics.mean(delta);sd=statistics.stdev(delta) if n>1 else None
 half={3:4.3026527299,5:2.7764451052}[n]*sd/math.sqrt(n) if n in (3,5) else None
 ap=aa[ss[0]];bp=bb[ss[0]]
 result.append(dict(group=g,arm=a,control=b,n=n,delta_mean_pp=mean,delta_sd_pp=sd,
 ci95_low_pp=mean-half if half is not None else None,ci95_high_pp=mean+half if half is not None else None,
 wins=sum(d>0 for d in delta),deltas_pp=','.join(f'{d:.6f}' for d in delta),
 arm_params=ap['params'],control_params=bp['params'],arm_macs=cost[ap['model_config']]['macs'],
 control_macs=cost[bp['model_config']]['macs'],mac_method=cost[ap['model_config']]['method'],interpretation=why))
with (O/'faithful-native-paired-validation-effects.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=result[0].keys());w.writeheader();w.writerows(result)
for r in result:print(r['group'],r['arm'],'vs',r['control'],f"{r['delta_mean_pp']:+.3f}pp",f"[{r['ci95_low_pp']:+.3f},{r['ci95_high_pp']:+.3f}]" if r['ci95_low_pp'] is not None else 'n=1','params',r['arm_params'],r['control_params'],'MAC',r['arm_macs'],r['control_macs'])
