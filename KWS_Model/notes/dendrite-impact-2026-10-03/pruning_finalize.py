"""Finalize read-only audit, including deployable neuron-mode sensitivity."""
from pathlib import Path
from collections import defaultdict, Counter
import importlib.util, sys, json, statistics, csv, copy, random, math

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ptp_final_audit', BASE / 'scripts/analyze_ptp.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
inv = json.loads((OUT / 'pruning_inventory.json').read_text())
summaries = json.loads((OUT / 'pruning_summaries.json').read_text())
extra = json.loads((OUT / 'pruning_extended.json').read_text())

mode_audit = []
neuron = []
canonical = []
canonical_mappings = []
for family in summaries['ptp']:
    dirs = [BASE / c['dir'] for c in inv['cells'] if c['family'] == family['family']]
    runs = mod.load_runs(dirs)
    canonical_runs = copy.deepcopy(runs)
    for run in runs:
        log = Path(run.run_dir) / 'metrics/sparsity' / f'sparknet_c{run.width}_multilayer' / 'pai.jsonl'
        records = mod.load_jsonl(log)
        counts = mod.evaluated_parameter_counts(records)
        dendrites = mod.dendrite_counts(counts, run.step)
        clean = [run.pruned_params + c - counts[0] + 12 * d * (d + 1) // 2 for c, d in zip(counts, dendrites)]
        ix = [i for i, r in enumerate(records) if r.get('pai_mode') == 'n']
        run.budgets = mod.select_budgets([records[i] for i in ix], [dendrites[i] for i in ix], [clean[i] for i in ix])
        report = next(r for r in inv['candidates'] if r['dir'] == run.run_dir.replace(str(BASE) + '/', '') and r['width'] == run.width)
        all_epoch = next(r for r in family['clean']['runs'] if r['seed'] == run.seed and r['width'] == run.width)
        selected = run.budgets[3]
        prior = all_epoch['budgets']['3']
        prior_mode = next(r['pai_mode'] for r in records if r['epoch'] == prior['epoch'])
        mode_audit.append({'family':family['family'], 'seed':run.seed, 'width':run.width, 'all_epoch_selected_mode':prior_mode, 'all_epoch_selected_epoch':prior['epoch'], 'n_only_selected_epoch':selected.epoch, 'n_only_val_matches_export':abs(selected.val_acc-report['final_val']) < 1e-9, 'n_only_cost_matches_export':selected.params == report['final_params'], 'all_epoch_val_matches_export':all_epoch['report_final_matches_budget3']['val_matches'], 'all_epoch_cost_matches_export':all_epoch['report_final_matches_budget3']['params_matches']})
        name = f'sparknet_c{run.width}_multilayer'
        with (Path(run.run_dir) / 'pai/candidates' / name / f'{name}_best_arch_scores.csv').open() as stream:
            arch = list(csv.DictReader(stream))
        mapped = []
        for a in arch:
            matches = [i for i, e in enumerate(records) if e['pai_mode']=='n' and counts[i]==int(a['Param Counts']) and abs(e['val_acc']-float(a['Max Valid Scores']))<1e-9 and abs(e['train_accuracy']-float(a['Train']))<1e-9]
            assert len(matches)==1, (run.run_dir,run.width,a,matches)
            mapped.append(matches[0])
            canonical_mappings.append({'family':family['family'], 'seed':run.seed,'width':run.width,'native_params':int(a['Param Counts']),'epoch':records[matches[0]]['epoch'],'mode':'n','val':float(a['Max Valid Scores']),'train':float(a['Train'])})
        cr = next(r for r in canonical_runs if r.seed==run.seed and r.width==run.width)
        cr.budgets = mod.select_budgets([records[i] for i in mapped], [dendrites[i] for i in mapped], [clean[i] for i in mapped])
        assert abs(cr.budgets[3].val_acc-report['final_val'])<1e-9
        assert cr.budgets[3].params==report['final_params']
    result = mod.analyze(runs)
    cluster = defaultdict(list)
    for r in result['runs']:
        cluster[r['seed']].append(100 * r['decomposition']['gain'])
    result['seed_clustered_gain_pp'] = mod.t_summary([statistics.mean(v) for v in cluster.values()])
    result['family'] = family['family']
    neuron.append(result)
    result = mod.analyze(canonical_runs)
    cluster = defaultdict(list)
    rates = defaultdict(list)
    z = result['zero_dendrite_reference']
    for r in result['runs']:
        cluster[r['seed']].append(100*r['decomposition']['gain'])
        rates[r['prune_rate']].append(r)
    strict = [rate for rate, rs in rates.items() if all(z[0]['params']<=r['budgets'][3]['params']<=z[-1]['params'] for r in rs)]
    strict_cluster = [statistics.mean(100*r['decomposition']['gain'] for r in result['runs'] if r['seed']==seed and r['prune_rate'] in strict) for seed in cluster]
    result['seed_clustered_gain_pp'] = mod.t_summary([statistics.mean(v) for v in cluster.values()])
    result['strict_common_rates'] = sorted(strict)
    result['strict_seed_clustered_gain_pp'] = mod.t_summary(strict_cluster)
    result['family'] = family['family']
    seeds = sorted(cluster)
    rng = random.Random(3102026)
    boot, strict_boot = [], []
    for _ in range(10000):
        ids = rng.choices(seeds,k=len(seeds))
        selected = [r for seed in ids for r in result['runs'] if r['seed']==seed]
        zero = mod.LogCurve.from_points((r['budgets'][0]['params'],r['budgets'][0]['test_acc']) for r in selected)
        boot.append(statistics.mean(100*(r['budgets'][3]['test_acc']-zero(r['budgets'][3]['params'])) for r in selected))
        strict_boot.append(statistics.mean(100*(r['budgets'][3]['test_acc']-zero(r['budgets'][3]['params'])) for r in selected if r['prune_rate'] in strict))
    def percentile(v,p):
        v=sorted(v);i=p*(len(v)-1);lo=math.floor(i);hi=math.ceil(i)
        return v[lo]*(hi-i)+v[hi]*(i-lo) if lo!=hi else v[lo]
    result['refitted_z_bootstrap_ci_pp']=[percentile(boot,.025),percentile(boot,.975)]
    result['strict_refitted_z_bootstrap_ci_pp']=[percentile(strict_boot,.025),percentile(strict_boot,.975)]
    canonical.append(result)
(OUT / 'pruning_neuron_mode.json').write_text(json.dumps({'mode_audit': mode_audit, 'neuron_mode': neuron,'canonical_export':canonical,'canonical_epoch_mappings':canonical_mappings}, indent=2))

lines = []
lines.append('\n## Final export versus candidate-training snapshots\n')
lines.append('The discrepancy has a specific cause: **all14 discrepant budget3 selections are p-mode epochs**, while all88 selections that agree with export are n-mode epochs. There are13 different validation values and9 different parameter counts (union14). `analyze_ptp.py` currently searches every epoch; the runner\'s canonical architecture CSV records neuron-mode maxima and exports that architecture. During p mode the current candidate is being trained and ordinary base weights are absent from the optimizer; PTP does not freeze BatchNorm running statistics. A transient p-mode snapshot is therefore not interchangeable with the retained exported model.\n')
lines.append('A read-only sensitivity recomputes all four budgets using n-mode epochs, assigns the corrected clean cost, and refits the zero reference. It matches101/102 exports. The exceptional C18g8 seed0 C17 records a higher zero-dendrite score at the n→p switch boundary (epoch12) that is omitted from its canonical architecture row. A stronger reconstruction maps **each canonical CSV row uniquely to an n-mode epoch using native count + validation + recorded TRAIN accuracy**, then selects budgets only among those rows. All102 reconstructed budget3 costs and validation scores now match final reports exactly. The mapped TEST scores reconstruct the selected architecture epoch; final-clean test inference has not been rerun. Detailed unique epoch mappings and both sensitivities are in `pruning_neuron_mode.json`.\n')
lines.append('| Source | N-only mean matched TEST pp | Seed-clustered 95% t CI | All-epoch mean pp |\n|---|---:|---|---:|')
for f, x in zip(neuron, extra['ptp']):
    s = f['seed_clustered_gain_pp']
    lines.append(f"| {f['family']} | {s['mean']:+.3f} | [{s['ci_low']:.3f}, {s['ci_high']:.3f}] | {x['clustered_seed_gain_pp']['mean']:+.3f} |")
lines.append('\nThe earlier per-rate tables and bootstrap retain the historical **all-epoch** analysis definition. The n-only check changes small numbers and resolves export identity; it preserves the unfavorable same-cost conclusion. No global zero-pruning source checkpoint is added to either Z curve.\n')
lines.append('### Canonical export-aligned TEST reconstruction (preferred evidence)\n')
lines.append('| Source | Mean matched TEST pp | Seed-clustered t CI | Refitted-Z bootstrap CI | Strict common mean pp | Strict common rates | Strict bootstrap CI |\n|---|---:|---|---|---:|---|---|')
for f in canonical:
    s=f['seed_clustered_gain_pp'];st=f['strict_seed_clustered_gain_pp'];b=f['refitted_z_bootstrap_ci_pp'];sb=f['strict_refitted_z_bootstrap_ci_pp']
    lines.append(f"| {f['family']} | {s['mean']:+.3f} | [{s['ci_low']:.3f}, {s['ci_high']:.3f}] | [{b[0]:.3f}, {b[1]:.3f}] | {st['mean']:+.3f} | {', '.join(str(round(100*r))+'%' for r in f['strict_common_rates'])} | [{sb[0]:.3f}, {sb[1]:.3f}] |")
for f in canonical:
    lines.append('\n#### '+f['family']+' canonical budgets\n')
    lines.append('| Prune / width | B0 params | Final clean mean (range) | B0 test % | Final mapped test mean ± SD % | Raw pp | Cost pp | Matched pp |\n|---|---:|---:|---:|---:|---:|---:|---:|')
    for pr in f['per_rate']:
        rs=[r for r in f['runs'] if r['prune_rate']==pr['prune_rate']]
        ps=[r['budgets'][3]['params'] for r in rs];ts=[100*r['budgets'][3]['test_acc'] for r in rs]
        lines.append(f"| {100*pr['prune_rate']:.0f}% / C{pr['width']} | {rs[0]['budgets'][0]['params']} | {statistics.mean(ps):.1f} ({min(ps)}–{max(ps)}) | {100*pr['a_pre']:.3f} | {statistics.mean(ts):.3f} ± {statistics.stdev(ts):.3f} | {100*pr['raw_gain']:+.3f} | {100*pr['param_cost']:+.3f} | {100*pr['gain']['mean']:+.3f} |")

lines.append('## Complete disposition of the 28 run roots\n')
lines.append('Folders named `seedN.log` are launcher logs, not runs. A report may say running after interruption; the artifact disposition below takes precedence. The138 serialized/planned candidate rows contain118 final clean exports. Gate-conv manifests plan candidates without serializing completed result rows. No incomplete arm is treated as an accuracy failure or silently dropped from this inventory.\n')
lines.append('| Run root under outputs/ | Report / manifest | Actual result disposition |\n|---|---|---|')
for c in inv['cells']:
    rows = [r for r in inv['candidates'] if r['dir'] == c['dir']]
    complete = [r for r in rows if r['has_final_clean']]
    partial = [r for r in rows if r['pai_epochs'] and not r['has_final_clean']]
    if c['family'].endswith('gate-conv-d3'):
        if c['seed'] in [0,1,2]:
            disposition = 'C12 PAI partial; no final clean export; later widths not completed'
        else:
            disposition = f"C12 pruning FT only ({6 if c['seed']==3 else 1} epochs); no PAI or final export"
    else:
        disposition = f"{len(complete)}/{len(rows)} final clean exports"
        if complete:
            disposition += ' (' + ', '.join('C'+str(r['width']) for r in complete) + ')'
        if partial:
            disposition += '; partial PAI ' + ', '.join('C'+str(r['width']) for r in partial)
        if len(rows) > len(complete) + len(partial):
            disposition += '; remaining candidates uncompleted'
    lines.append(f"| `{c['dir'].removeprefix('outputs/')}` | {c['report_status']} / {c['manifest_status']} | {disposition} |")

lines.append('\n### What the partial runs establish\n')
lines.append('- C12 FC-only: C10 completed at92.3240% validation,4114 clean parameters, three retained dendrites. C8 has40 pruning-FT epochs and132 PAI epochs, still at its zero-dendrite native count; no final export exists. The source C12 is92.6133% at3400 parameters, so the completed C10 candidate is larger and less accurate than its parent.\n- C12 unlimited: C10 has1788 unique PAI epochs and native architecture counts2854 through13078 in eight1278-parameter increments. Canonical validation rises from91.5718% at the zero row to93.2305% at the largest native architecture. That is+0.6172pp above its source validation, using at least3.85× the parent native parameter count; exact clean cost, retained final architecture and final test are unavailable. This is a partial high-compute observation, not a completed frontier result.\n- C16 gate-conv: seeds0–2 have557/552/537 PAI epochs on C12. Canonical zero-to-best improvements are+0.2475/+0.3600/+0.3825pp; native counts3400/3816/4232/4648 show attempted growth up to three copies. Seeds3–4 stop during pruning fine-tuning. Every seed lacks a final clean export; no placement-level comparison to completed FC arms is valid.\n')

lines.append('## Parent-relative deployment tradeoffs\n')
lines.append('These comparisons use the **final reports**, not offline snapshots. Positive size reductions mean smaller. Every completed PTP width has lower mean validation than its own unpruned source. More capacity and more training recover some post-pruning accuracy, but recovery is not preservation of the original model. Gate-width families change both encoder width and gate width, pruning criterion/recipe families also differ, and this is not a randomized placement study.\n')
lines.append('| Source family | Final width | Final−source val pp | Mean parameter reduction | Mean MAC reduction |\n|---|---:|---:|---:|---:|')
for r in summaries['family_width_summary']:
    if r['family'] not in [f['family'] for f in neuron]: continue
    ss = r['stats']
    candidate = next(c for c in inv['candidates'] if c['family'] == r['family'] and c['width'] == r['width'])
    p_red = 100*(1-ss['final_params']['mean']/candidate['source_params'])
    m_red = 100*(1-ss['final_macs']['mean']/candidate['source_macs'])
    lines.append(f"| {r['family']} | C{r['width']} | {ss['delta_final_source_pp']['mean']:+.3f} | {p_red:+.2f}% | {m_red:+.2f}% |")
lines.append('\nThe unpruned C16/g16 parent is itself a strong conventional option:94.9966% ±0.3067pp validation,4140 parameters,370256 MACs, versus C16/g32 at94.9306% ±0.2743pp,4636 parameters,396304 MACs and C18/g8 at94.9921% ±0.3595pp,4534 parameters,419246 MACs. Its mean accuracy difference is within seed variation, while its parameter and MAC advantage is exact. Thus a comparison restricted to one original gate32 parent misses a conventional architectural improvement available before dendrites.\n')

lines.append('### Best individual completed candidates, with exact evidence paths\n')
lines.append('These are family-specific maximum final validation values and are **selected seed envelopes**, not matched-seed mean wins. Report and checkpoint paths are relative to `KWS_Model/`.\n')
lines.append('| Family / seed / width | Final val % | Parameters | MACs | Retained D | Evidence |\n|---|---:|---:|---:|---:|---|')
for fam in sorted({r['family'] for r in inv['candidates']}):
    rows = [r for r in inv['candidates'] if r['family']==fam and r['has_final_clean']]
    if not rows: continue
    r = max(rows, key=lambda r:r['final_val'])
    checkpoint = r['dir'] + f"/pai/candidates/sparknet_c{r['width']}_multilayer/final_clean_pai.pt"
    lines.append(f"| {fam} / {r['seed']} / C{r['width']} | {100*r['final_val']:.4f} | {r['final_params']} | {r['final_macs']} | {r['retained_dendrites_copied_count']} | report `{r['report_path']}`; checkpoint `{checkpoint}` |")

lines.append('\n## Accounting, controls, and limits\n')
lines.append('- The completed118 clean checkpoints were inspected using safetensors without importing PerforatedAI. Every clean branch/skip parameter overhead matches its report. All42 C16,30 g16 and30 g8 PTP cells completed; the only PTP zero-dendrite final is C16 seed1 C15. A budget of three means at most three, not exactly three.\n- Canonical `noImprove_lr` artifacts record failed retries, not whole-run zero retention. Thirteen of15 completed historical C16 FC cells have these artifacts although14 retain dendrites. Every placement claim must inspect final branch structure or clean overhead. PB correlation magnitudes describe candidate fit, and no fixed threshold or two-observation trend proves a useful placement.\n- Recovery versus immediate pruning FT conflates a long ordinary zero-dendrite continuation with branch growth. Old C16 FC runs use40 FT +520–854 PAI +8 resume epochs; zero-phase recovery dominates. Their final branch-associated canonical best−zero changes are only+0.193/+0.346/+0.661pp at C12/C10/C8. A C10 seed0 zero-branch model recovers+4.364pp before resume. A matched ordinary continuation at identical optimizer, data cohort, wall time and checkpoint selection is absent.\n- PTP uses validation for search, logs clean train and test every epoch, and has no post-PAI resume. Its zero reference has already received substantial ordinary PAI neuron training. It is a meaningful within-search zero control, although shorter than the full growing lifecycle. Held-out TEST is never an optimizer/PAI score input in current code; repeated observational access still differs from a one-time frozen test evaluation.\n- No configured hardware budget is enforced in any assigned run. FC PTP head overhead is1260/684/396 clean parameters at D3 for gate32/16/8, a large fraction of these tiny models. The supplied paper\'s largest classification budget adds at most10.4%, under3% through70% pruning; that scale assumption does not transfer to SparkNet. The paper\'s use of a cheap head on larger models is relevant context, not positive evidence for these tiny models.\n- Reports contain logical8-bit weight bytes, conservative activation accounting and host CPU latency. They are not board deployment measurements or evidence of quantized dendritic equivalence. The old singleton C12 FC reported MAC225646 is below the clean formula226030 for C10/D3/g32, a384-MAC missing base-branch discrepancy; preserve it as historical reported cost and avoid precise cross-family MAC attribution.\n- Group pruning in PTP physically rebuilds a narrower dense network, manually propagating producer and consumer dependency groups with fixed gate width. Old pruning ranks producer L1 magnitudes; PTP uses dependency-group L2 scores. Target rates round to realizable widths and achieved rates differ. The severe C3 early-stopped fits remain included. No claim here treats masks, nominal requested rates, or stopped poor fits as equivalent achieved dense cost.\n')
lines.append('## Reading coverage and corrections to historical notes\n')
lines.append('Read the root AGENTS/README and KWS README; supplied Pruning Then Perforating text; `PROJECT_FINDINGS.md`; and `dendrite-study-v2/{DATA_INVENTORY,INTEGRATION_AUDIT,DSCNN_DENDRITIC_PIPELINES,agent-dscnn-pipeline-designs,ENHANCEMENTS,PICO_COMPRESSION_PIPELINE_JOURNAL,review-next-runs-evidence,review-next-runs-pai,agent-repo-results-audit}.md`. Parent and sibling agents cover core PAI theory, primary papers and the other output families. The historical notes are useful development records, not independently verified results.\n')
lines.append('Relevant lessons: an optimizer handoff can harm a previously good checkpoint; identity-prune controls expose that before any branch is added; sham switches preserve lifecycle while adding no branch and are stronger controls for switch effects; scratch equal-cost curves and actual no-branch controls answer different questions; class-output scales and direct module forward calls affect clean counting/profiling; normalized export equivalence should be measured rather than inferred; heldout performance, hardware cost and timing must be reported separately. The historical journal\'s proposal to apply a DSCNN prune/KD/resume recipe to SparkNet is motivation, not causal evidence.\n')
lines.append('Corrections made in this audit: current configs are not authoritative for historical saved manifests; native counts omit clean skip scales; the clean scale count is triangular, not D²×classes; noImprove files do not prove zero retained branches; a PB score threshold is not a performance guarantee; old C16 pruning directories do not provide five independent continuation seeds; all-epoch offline selection can choose candidate p-mode snapshots; and endpoint-flat extrapolation cannot establish a measured cost-matched win. Prior notes that predate late-September PTP cannot be treated as a complete current inventory.\n')
lines.append('## Audit completion\n')
lines.append('Read-only analyses completed with standard Python/YAML/scipy/safetensors. No training, licensed PAI import, original artifact mutation, or `.env` read was performed. The machine-readable inventory and all computations are retained beside this notebook. Tests of model behavior were not run because this is an artifact analysis; exact clean overheads were independently checked against every saved final tensor graph, and reconstructed canonical epoch costs/validation match all102 PTP final reports.\n')

with (OUT / 'sparknet-pruning-and-ptp.md').open('a') as stream:
    stream.write('\n'.join(lines))
print('Neuron-mode matched final validation',sum(r['n_only_val_matches_export'] for r in mode_audit),'/',len(mode_audit))
print('Neuron-mode matched final cost',sum(r['n_only_cost_matches_export'] for r in mode_audit),'/',len(mode_audit))
for f in neuron: print(f['family'], f['seed_clustered_gain_pp'])
for f in canonical: print('CANONICAL',f['family'],f['seed_clustered_gain_pp'],'strict',f['strict_seed_clustered_gain_pp'],'bootstrap',f['refitted_z_bootstrap_ci_pp'],'strict_bootstrap',f['strict_refitted_z_bootstrap_ci_pp'])
