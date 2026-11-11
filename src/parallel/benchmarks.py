"""Parallel benchmark suite using existing task generators and linear readouts."""
import numpy as np
from src.parallel.common import parser,validate,cases,simulate,Results,write_csv
from src.task_evaluation import history_features

from src.tasks.benchmarks.suite import TASKS, tasks, evaluate


def main(argv=None):
    p=parser('benchmarks',TASKS)
    p.add_argument('--windows',type=int,nargs='+',default=[2,3,5,10])
    p.add_argument('--delays',type=int,nargs='+',default=[1,2,5,10])
    p.add_argument('--horizons',type=int,nargs='+',default=[1,5,10,20])
    p.add_argument('--history-taps',type=int,nargs='+',default=[1,10,16])
    a=p.parse_args(argv);validate(a)
    if any(n<1 for n in a.delays+a.horizons+a.history_taps) or min(a.windows)<2:
        p.error('invalid delay, horizon, window, or history count')
    if max(a.history_taps)-1>a.washout:
        p.error('washout must cover history taps')
    out=Results(a)
    for case in cases(a):
        for family in a.tasks:
            try:
                variants=list(tasks(family,a,case.seed))
            except Exception as exc:
                out.fail(case,family,exc);continue
            # Same generator seed yields the same input for XOR/parity variants.
            cached_u,cached_X=None,None
            for name,task in variants:
                try:
                    if cached_u is None or not np.array_equal(task.u,cached_u):
                        cached_u=task.u.copy()
                        cached_X=simulate(case,task.u,a,cycles=CYCLES[family]).X
                    gap=task.target_offset if family=='mackey_glass' else 0
                    values,arrays=evaluate(cached_X,task,a,gap)
                    d=out.directory(case,name)
                    np.savez_compressed(d/'predictions.npz',**arrays)
                    np.savez_compressed(d/'input_targets.npz',u=task.u,y=task.y)
                    import json
                    (d/'task.json').write_text(json.dumps(dict(
                        name=task.name,target_offset=task.target_offset,washout=task.washout,
                        encoding='zoh',cycles_per_symbol=(a.cycles_per_symbol if a.cycles_per_symbol is not None
                        else CYCLES[family])),indent=2))
                    if 'test_nrmse_per_output' in arrays:
                        write_csv(d/'per_output.csv',[dict(output=i,train_nrmse=float(tr),test_nrmse=float(te))
                            for i,(tr,te) in enumerate(zip(arrays['train_nrmse_per_output'],arrays['test_nrmse_per_output']))])
                    out.add(case,name,status='ok',**values)
                    # Baselines are repeated per case for identical-split comparisons.
                    baseline_rows=[]
                    for taps in a.history_taps:
                        baseline_values,_=evaluate(history_features(task.u,taps),task,a,gap)
                        baseline_rows.append(dict(method=f'input_history{taps}',**baseline_values))
                    write_csv(d/'baselines.csv',baseline_rows)
                except Exception as exc:
                    out.fail(case,name,exc)
    if any(row.get('status')=='failed' for row in out.rows):
        raise SystemExit('Some benchmark cases failed; see summary.csv')

if __name__=='__main__':
    main()
