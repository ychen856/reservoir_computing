"""Parallel benchmark suite using existing task generators and linear readouts."""
import numpy as np
from src.parallel.common import parser,validate,cases,simulate,Results,write_csv
from src.simulation.evaluation import evaluate_continuous_regression,evaluate_independent_classification
from src.task_evaluation import history_features
from src.tasks.benchmarks.sine_cube import generate_sine_cube
from src.tasks.benchmarks.narma10 import generate_narma10
from src.tasks.benchmarks.henon import generate_henon
from src.tasks.benchmarks.rossler import generate_rossler
from src.tasks.benchmarks.parity import generate_parity
from src.tasks.benchmarks.xor import generate_xor
from src.tasks.benchmarks.mackey_glass import generate_mackey_glass

TASKS=('sine_cube','narma10','henon','rossler','parity','xor','mackey_glass')
# Preserve the current run_xx.py timing choices. Encoding is explicitly ZOH:
# NARMA's old masked-input experiment is a DIFFERENT protocol, not silently copied.
CYCLES=dict(sine_cube=3.05,narma10=2.,henon=10.,rossler=8.,parity=1.,xor=1.,mackey_glass=1.)


def tasks(name,a,seed):
    kw=dict(n_samples=a.symbols,washout=a.washout,train_ratio=a.train_ratio)
    if name=='parity':
        for window in a.windows:
            yield f'parity_window{window}',generate_parity(window=window,seed=seed,**kw)
    elif name=='xor':
        for delay in a.delays:
            yield f'xor_delay{delay}',generate_xor(delay=delay,seed=seed,**kw)
    elif name=='mackey_glass':
        for horizon in a.horizons:
            yield f'mackey_glass_horizon{horizon}',generate_mackey_glass(horizon=horizon,**kw)
    else:
        fn=dict(sine_cube=generate_sine_cube,narma10=generate_narma10,
                henon=generate_henon,rossler=generate_rossler)[name]
        if name=='narma10':
            kw['seed']=seed
        yield name,fn(**kw)


def evaluate(X,task,a,gap=0):
    # Align ALL methods on identical original symbol indices, including purge.
    valid=np.arange(task.washout,len(task.u))
    cut=int(len(valid)*task.train_ratio)
    train,test=valid[:cut],valid[cut+gap:]
    if min(len(train),len(test))<3:
        raise ValueError('too few train/test samples')
    if task.task_type=='classification':
        ev=evaluate_independent_classification(X[train],task.y[train],X[test],task.y[test],
                                              ridge=a.ridge,normalize=True,n_classes=2)
        values=dict(metric='accuracy',train=float(ev.train_accuracy),test=float(ev.test_accuracy))
        extras=dict(confusion_matrix=ev.confusion_matrix)
    else:
        joined_X=np.vstack([X[train],X[test]])
        joined_y=np.concatenate([task.y[train],task.y[test]],axis=0)
        # This ratio reconstructs the explicit split, with rounding protected.
        ratio=(len(train)+.25)/len(joined_X)
        ev=evaluate_continuous_regression(joined_X,joined_y,washout=0,
                        train_ratio=ratio,ridge=a.ridge,normalize=True)
        if ev.n_train != len(train):
            raise AssertionError('split alignment mismatch')
        values=dict(metric='nrmse',train=float(ev.train_nrmse),test=float(ev.test_nrmse))
        extras=dict(train_nrmse_per_output=ev.train_nrmse_per_output,
                    test_nrmse_per_output=ev.test_nrmse_per_output,
                    train_r_per_output=ev.train_r_per_output,test_r_per_output=ev.test_r_per_output)
    values.update(feature_dim=X.shape[1],n_train=len(train),n_test=len(test),purge_symbols=gap)
    arrays=dict(train_indices=train,test_indices=test,y_train=ev.y_train,y_test=ev.y_test,
                y_train_pred=ev.y_train_pred,y_test_pred=ev.y_test_pred,
                W=ev.W,mu=ev.mu,std=ev.std,**extras)
    return values,arrays


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
