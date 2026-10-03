"""Read-only artifact audit. Run from repository root with KWS_Model/.venv/bin/python.

Only imports stdlib and YAML; it does not load checkpoints or import PerforatedAI.
Outputs are restricted to this new research directory.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import re
import statistics
import yaml

ROOT = Path(__file__).resolve().parents[3]
KWS = ROOT / 'KWS_Model'
OUT = Path(__file__).resolve().parent

def load_jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def cnn_macs(c):
    """Analytical Conv2d/Linear MAC count for the unperforated EICNN.

    Excludes CMVN, nonlinearities, pooling, bias adds, noise and PAI skip edges.
    Not a hardware benchmark. Only used with one retained PAI dendrite.
    """
    h, w, cin = 13, 101, 1
    convs = []
    for cout in (c['c1'], c['c2'], c.get('c3', 0), c.get('c4', 0)):
        if not cout:
            continue
        convs.append(h * w * cin * cout * 9)
        h = math.ceil(h / 2) if c['pool'] == 'same' else h // 2
        w = math.ceil(w / 2) if c['pool'] == 'same' else w // 2
        cin = cout
    return sum(convs), h * w * cin * 12

def cnn_one_dendrite_skip_macs(c):
    if c['conversion']=='linear':
        return 12
    h,w=13,101
    outputs=12
    for cout in (c['c1'],c['c2'],c.get('c3',0),c.get('c4',0)):
        if not cout:continue
        outputs+=h*w*cout
        h=math.ceil(h/2) if c['pool']=='same' else h//2
        w=math.ceil(w/2) if c['pool']=='same' else w//2
    return outputs

faithful = []
raw_results = {}
for group in sorted((KWS / 'outputs').glob('pai-faithful*')):
    for run in sorted(p for p in group.iterdir() if p.is_dir()):
        p = run / 'result.json'
        history = load_jsonl(run / 'epochs.jsonl')
        if p.exists():
            d = json.loads(p.read_text())
            raw_results[str(run.relative_to(KWS))] = d
            c = d['config']
            conv, fc = cnn_macs(c)
            count = d['dendrites']['structural_max_per_module']
            macs = conv + fc
            if count == 1:
                macs += conv + fc if c['conversion'] == 'all' else fc
            elif count > 1:
                macs = None
            row = dict(
                path=str(run.relative_to(KWS)), batch=group.name, status='evaluated',
                arm=d['arm'], seed=c['seed'], data_config=c['data_config'],
                train_fraction=0.2 if 'frac20' in c['data_config'] else 0.1 if 'frac10' in c['data_config'] else 1.0,
                device=d['device'], phase0_optimizer=c['phase0_optimizer'],
                method=c['dendrites'], conversion=c['conversion'], activation=c['forward'],
                c1=c['c1'], c2=c['c2'], c3=c.get('c3', 0), c4=c.get('c4', 0),
                restart_every=c.get('restart_every', 0), stop_lr=c.get('stop_lr', 0),
                epochs=d['epochs_run'], stop_reason=d['stop_reason'],
                params_base=d['params']['base_model'], params_clean=d['params']['clean_best'],
                params_live_selected_numel=d['params']['numel_best'],
                single_dendrite_clean_missing_coefficients=d['params']['numel_best']-d['params']['clean_best'] if count==1 else 0,
                params_pai_count=d['params']['pai_count_best'], params_numel=d['params']['numel_best'],
                clean_error=d['params']['clean_error'],
                base_conv_linear_macs=conv + fc, selected_conv_linear_macs_proxy=macs,
                selected_conv_linear_plus_top_macs_proxy=macs+cnn_one_dendrite_skip_macs(c) if count==1 else macs,
                dendrites_attempted=d['dendrites']['max_num_dendrites_added_during_run'],
                dendrites_retained=count, val_accuracy=d['val_acc_selected'],
                test_accuracy=d['test_accuracy'], test_samples=d['test_samples'],
                train_acc_at_best_val=next((h['train_acc'] for h in history if h['epoch']==d['max_val_acc_epoch']),None),
                seconds_total=d['seconds_total'], epochs_sha256=hashlib.sha256((run/'epochs.jsonl').read_bytes()).hexdigest(),
                result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
        else:
            match = re.search(r'seed(\d+)', run.name)
            log_path = group / (run.name + '.log')
            log = log_path.read_text() if log_path.exists() else ''
            failed = 'RuntimeError: PerforatedAI entered pdb.set_trace()' in log
            row = dict.fromkeys(faithful[0].keys()) if faithful else {}
            row.update(path=str(run.relative_to(KWS)), batch=group.name, status='failed' if failed else 'partial',
                       arm=run.name, seed=int(match[1]) if match else None,
                       epochs=len(history), val_accuracy=max((h.get('val_acc',0) for h in history),default=None),
                       test_accuracy=None, stop_reason='initial_correlation_batches_40_exceeds_epoch_batches_29' if failed else 'no_final_result',
                       dendrites_attempted=max((h.get('dendrites_added') or 0 for h in history),default=None),
                       epochs_sha256=hashlib.sha256((run/'epochs.jsonl').read_bytes()).hexdigest() if history else None)
        faithful.append(row)

write_csv('faithful-run-inventory.csv', faithful)
grouped = collections.defaultdict(list)
for r in faithful:
    if r['status']=='evaluated':
        grouped[(r['batch'],r['train_fraction'],r['phase0_optimizer'],r['arm'])].append(r)
summary = []
for key, rows in sorted(grouped.items()):
    rows.sort(key=lambda r:r['seed'])
    def avg(field): return statistics.mean(r[field] for r in rows)
    summary.append(dict(batch=key[0],train_fraction=key[1],phase0_optimizer=key[2],arm=key[3],n=len(rows),
        device=rows[0]['device'],seeds=','.join(str(r['seed']) for r in rows),
        params_clean_values=','.join(str(p) for p in sorted(set(r['params_clean'] for r in rows))),
        params_live_selected_numel_values=','.join(str(p) for p in sorted(set(r['params_live_selected_numel'] for r in rows))),
        selected_conv_linear_macs_proxy_values=','.join(str(p) for p in sorted(set(r['selected_conv_linear_macs_proxy'] for r in rows),key=str)),
        selected_conv_linear_plus_top_macs_proxy_values=','.join(str(p) for p in sorted(set(r['selected_conv_linear_plus_top_macs_proxy'] for r in rows),key=str)),
        val_mean_pct=avg('val_accuracy')*100,
        test_mean_pct=avg('test_accuracy')*100,
        test_sd_pct=statistics.stdev(r['test_accuracy']*100 for r in rows) if len(rows)>1 else None,
        retained_counts=','.join(str(r['dendrites_retained']) for r in rows),
        epochs_mean=avg('epochs'),seconds_mean=avg('seconds_total')))
write_csv('faithful-arm-summary.csv',summary)

native=[]
for group in sorted((KWS/'outputs').glob('students-*')):
    for run in sorted(p for p in group.iterdir() if p.is_dir()):
        manifest=yaml.safe_load((run/'manifest.yaml').read_text())
        summary_path=run/'metrics/summaries.yaml'
        summ=yaml.safe_load(summary_path.read_text()) if summary_path.exists() else {}
        phases=summ.get('phases',{})
        best=max((p.get('best_val_acc',0) for p in phases.values()),default=None)
        hist=[row for p in run.glob('metrics/*/*.jsonl') for row in load_jsonl(p)]
        inputs={item['role']:item for item in manifest.get('inputs',[])}
        model_path=Path(inputs['model_config']['path'])
        cfg=yaml.safe_load(model_path.read_text()) if model_path.exists() else {}
        native.append(dict(path=str(run.relative_to(KWS)),group=group.name,model=run.name.rsplit('-seed',1)[0],
            seed=manifest.get('seed'),status=manifest.get('status'),
            epochs=max((p.get('epochs',0) for p in phases.values()),default=0),
            params=hist[-1].get('parameter_count') if hist else None,best_val_accuracy=best,
            test_accuracy=None,train_final_accuracy=hist[-1].get('train_accuracy') if hist else None,
            final_val_accuracy=hist[-1].get('val_acc') if hist else None,
            family=cfg.get('family','ds_cnn'),
            native_dendrites=bool(cfg.get('dendrites') or cfg.get('head_dendrites') or cfg.get('fc_dendrites')
                or (cfg.get('msd_blocks') and cfg.get('msd_pointwise','dendritic')!='dense')
                or (cfg.get('family')=='dtnet' and cfg.get('branches',4)!=1)),
            model_config=str(model_path.relative_to(KWS)),
            model_config_matches_recorded_sha=hashlib.sha256(model_path.read_bytes()).hexdigest()==inputs['model_config']['sha256'] if model_path.exists() else None,
            train_config=inputs['train_config']['path'].split('KWS_Model/')[-1],
            train_config_sha=inputs['train_config']['sha256'],data_config=inputs['data_config']['path'].split('KWS_Model/')[-1],
            git_commit=manifest.get('git',{}).get('commit'),git_dirty=manifest.get('git',{}).get('dirty'),
            seconds=sum(h.get('elapsed_seconds',0) for h in hist)))
write_csv('faithful-native-students-inventory.csv',native)
native_groups=collections.defaultdict(list)
for r in native:native_groups[(r['group'],r['model'],r['train_config_sha'])].append(r)
native_summary=[]
for key,rows in sorted(native_groups.items()):
    vals=[r['best_val_accuracy']*100 for r in rows if r['best_val_accuracy'] is not None]
    native_summary.append(dict(group=key[0],model=key[1],n=len(rows),seeds=','.join(str(r['seed']) for r in rows),
        statuses=','.join(sorted(set(r['status'] for r in rows))),params=rows[0]['params'],
        native_dendrites=rows[0]['native_dendrites'],train_config=rows[0]['train_config'],
        val_mean_pct=statistics.mean(vals) if vals else None,val_sd_pct=statistics.stdev(vals) if len(vals)>1 else None,
        train_final_mean_pct=statistics.mean(r['train_final_accuracy']*100 for r in rows),
        seconds_mean=statistics.mean(r['seconds'] for r in rows)))
write_csv('faithful-native-students-summary.csv',native_summary)

queue=[]
queue_status=[]
for group in sorted((KWS/'outputs').glob('research-*')):
    jobs=group/'jobs.all.txt'
    status_path=group/'status.log'
    status=status_path.read_text() if status_path.exists() else ''
    started=re.findall(r'START (outputs/\S+)',status)
    completed=re.findall(r'DONE rc=0 (outputs/\S+)',status)
    failed=re.findall(r'FAIL rc=\d+ (outputs/\S+)',status)
    queue_status.append(dict(queue=group.name,archived_job_file=jobs.exists(),start_events=len(started),
        done_events=len(completed),fail_events=len(failed),
        started_targets=','.join(started),failed_targets=','.join(failed),
        comments='empty jobs.txt after consumed queue is not evidence of no jobs'))
    if not jobs.exists():
        # Four consumed native queues retained only status and manifests.
        for name in dict.fromkeys(started):
            target=KWS/name
            queue.append(dict(queue=group.name,run_path=name,artifact_present=target.exists(),
                result_present=(target/'result.json').exists(),manifest_present=(target/'manifest.yaml').exists(),
                command='not archived; recover from target manifest if present'))
        continue
    for line in jobs.read_text().splitlines():
        if not line.strip():continue
        name,_,command=line.partition('\t')
        target=KWS/name
        queue.append(dict(queue=group.name,run_path=name,artifact_present=target.exists(),
                         result_present=(target/'result.json').exists(),manifest_present=(target/'manifest.yaml').exists(),
                         command=command))
write_csv('faithful-research-queue-provenance.csv',queue)
write_csv('faithful-research-queue-status.csv',queue_status)

counts=collections.Counter((r['batch'],r['status']) for r in faithful)
print(json.dumps({'faithful_status_counts':{str(k):v for k,v in counts.items()},
                  'native_status_counts':dict(collections.Counter(r['status'] for r in native)),
                  'native_model_config_changed':sum(r['model_config_matches_recorded_sha'] is False for r in native),
                  'research_queues':len(set(r['queue'] for r in queue)),'queue_jobs':len(queue)},indent=2))
