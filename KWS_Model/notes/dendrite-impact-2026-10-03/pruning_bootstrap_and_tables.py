from pathlib import Path
import json,math,statistics,random
from collections import defaultdict,Counter
BASE=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
inv=json.loads((OUT/'pruning_inventory.json').read_text());sums=json.loads((OUT/'pruning_summaries.json').read_text());ex=json.loads((OUT/'pruning_extended.json').read_text())

def zcurve(runs):
 groups=defaultdict(list)
 for r in runs:groups[r['prune_rate']].append(r)
 pts=sorted((rs[0]['budgets']['0']['params'], statistics.mean(r['budgets']['0']['test_acc'] for r in rs)) for rs in groups.values())
 def curve(n):
  if n<=pts[0][0]:return pts[0][1]
  if n>=pts[-1][0]:return pts[-1][1]
  for (n0,a0),(n1,a1) in zip(pts,pts[1:]):
   if n0<=n<=n1:return a0+(math.log10(n)-math.log10(n0))/(math.log10(n1)-math.log10(n0))*(a1-a0)
 return curve

def q(v,p):
 i=p*(len(v)-1);lo=math.floor(i);hi=math.ceil(i);return v[lo]*(hi-i)+v[hi]*(i-lo) if lo!=hi else v[lo]
boot=[];rng=random.Random(3102026)
for family,extra in zip(sums['ptp'],ex['ptp']):
 runs=family['clean']['runs'];seeds=sorted(set(r['seed'] for r in runs));byseed={s:[r for r in runs if r['seed']==s] for s in seeds};v=[];strictv=[]
 for _ in range(10000):
  ids=rng.choices(seeds,k=len(seeds));sel=[r for s in ids for r in byseed[s]];z=zcurve(sel)
  gains=[100*(r['budgets']['3']['test_acc']-z(r['budgets']['3']['params'])) for r in sel]
  sg=[100*(r['budgets']['3']['test_acc']-z(r['budgets']['3']['params'])) for r in sel if r['prune_rate'] in extra['strict_common_rates']]
  v.append(statistics.mean(gains));strictv.append(statistics.mean(sg))
 v.sort();strictv.sort()
 boot.append({'family':family['family'],'seed_units':len(seeds),'bootstrap_replicates':10000,'bootstrap_gain_ci95_pp':[q(v,.025),q(v,.975)],'strict_common_rates':extra['strict_common_rates'],'strict_common_gain_ci95_pp':[q(strictv,.025),q(strictv,.975)]})

lines=['\n## Completed PTP: raw gain versus the parameter bill\n','These tables summarize budget3 epochs selected offline on validation with corrected clean costs. `Raw` is budget3 TEST minus budget0 TEST; `Cost` is the zero-curve accuracy increase associated with spending those added parameters; `Matched` is TEST minus the zero-curve at the final clean count. All differences are pp. Entries with `strict n=0` are wholly above the observed zero-curve domain and use the paper\'s flat extrapolation. Test cohorts are seed-specific. No unpruned parent is included in Z by the analysis definition.\n']
for fam,extra in zip(sums['ptp'],ex['ptp']):
 lines+=['\n### '+fam['family']+'\n','| Target prune | Width | Base params | Selected clean params mean (range) | B0 test % | B3 test mean ± sd % | Raw pp | Cost pp | Matched pp | Strict n |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for pr,row in zip(fam['clean']['per_rate'],extra['rates']):
  rs=[r for r in fam['clean']['runs'] if r['width']==pr['width']];acc=[100*r['budgets']['3']['test_acc'] for r in rs];lo,hi=row['b3_params_range']
  lines.append(f"| {100*row['rate']:.0f}% | C{row['width']} | {row['b0_params']} | {row['b3_params_mean']:.1f} ({lo}–{hi}) | {100*pr['a_pre']:.3f} | {statistics.mean(acc):.3f} ± {statistics.stdev(acc):.3f} | {row['raw_gain_pp']['mean']:+.3f} | {row['cost_pp']['mean']:+.3f} | {row['gain_pp']['mean']:+.3f} | {row['strict_coverage']}/{row['n']} |")
 lines+=['\nThe final report and offline budget selection disagree in some cells; reported final architecture costs are in `pruning_candidates.csv`, selected epoch costs in `pruning_summaries.json`.\n']
lines+=['\n### Uncertainty using independent parent seeds\n','Rate rows from the same source seed are correlated, so the pooled t intervals over42/30/30 rows from `analyze_ptp.py` are not used as independent evidence. The next table first averages rates within each source seed, then computes a95% t interval across6/5/5 seed units. The bootstrap resamples **entire source seeds**, refits the seed-mean zero curve in each replicate, and recomputes the overall gain10,000 times; its interval also reflects uncertainty in Z. Small seed counts and a coarse interpolated curve limit precision.\n','| Source | Seed units | Mean gain pp with flat extrapolation | Seed-mean 95% t CI | Refitted-Z bootstrap 95% CI | Strict common-rate mean pp | Strict common rates | Strict refitted-Z bootstrap CI |','|---|---:|---:|---|---|---:|---|---|']
for extra,b in zip(ex['ptp'],boot):
 t=extra['clustered_seed_gain_pp'];st=extra['strict_common_clustered_seed_gain_pp']
 lines.append(f"| {extra['family']} | {t['n']} | {t['mean']:+.3f} | [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] | [{b['bootstrap_gain_ci95_pp'][0]:+.3f}, {b['bootstrap_gain_ci95_pp'][1]:+.3f}] | {st['mean']:+.3f} | {', '.join(str(round(100*p))+'%' for p in extra['strict_common_rates'])} | [{b['strict_common_gain_ci95_pp'][0]:+.3f}, {b['strict_common_gain_ci95_pp'][1]:+.3f}] |")
lines+=['\nAt C16 the severe C3 pruning cases account for much of the large pooled deficit; do not generalize−7.41pp to a typical light-pruning candidate. The C3 pruning-FT validation mean is42.07% ±22.78pp; seed3 stops after9FT epochs at7.58%, seed4 after12 at18.99%. Ordinary pre-dendrite PAI retraining raises them to54.11% and32.98%. These failures stay in the inventory and comparison rather than being censored. For common in-range C16 rates40–70%, average clean-cost-matched deficit is−11.985pp, because the added classifier budget crosses the steep ordinary pruning curve.\n','\n### Comparison with scratch models and conventional parents\n','Validation comparisons below use currently completed same-gate scratch runs. They are independent architecture controls, not perfectly time-matched continued-training controls. Missing exact widths use an explicitly labeled log-parameter interpolation of the measured same-gate mean scratch curve. Some curve widths have3 rather than5 seeds; exact run lists and seed counts are stored in `pruning_extended.json`. No g16/g8 scratch test curve is available, so these comparisons are validation only.\n','| Source | Width | Final report−same-width, same-seed scratch val pp | Final report−same-gate scratch parameter curve val pp |','|---|---:|---:|---:|']
for extra in ex['ptp']:
 for r in extra['rates']:
  a=r['final_vs_same_width_scratch_pp'];b=r['final_vs_scratch_param_curve_pp']
  lines.append(f"| {extra['family']} | C{r['width']} | {a['mean']:+.3f} (n={a['n']}) | {b['mean']:+.3f} |" if a['mean'] is not None else f"| {extra['family']} | C{r['width']} | unavailable | {b['mean']:+.3f} |")
lines+=['\nSame-gate conventional scratch mean rows dominate the final **individual** report candidate in accuracy/parameters for40/42 C16,28/30 C16g16,17/30 C18g8 cases; requiring no greater MAC count reduces these counts to35/42,26/30,16/30. These are descriptive comparisons to an architecture mean, not paired statistical tests, and favorable seeds should not be mistaken for a method-level frontier. Parent-level examples and exact nondominated candidates follow.\n']
(OUT/'sparknet-pruning-and-ptp.md').open('a').write('\n'.join(lines))
(OUT/'pruning_bootstrap.json').write_text(json.dumps(boot,indent=2))
print(boot)
