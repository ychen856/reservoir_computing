import numpy as np
from src.simulation.evaluation import evaluate_continuous_regression,evaluate_independent_classification
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


def evaluate(X, task, a, gap=0, *, split=None):
    # Align ALL methods on identical original symbol indices, including purge.
    valid=np.arange(task.washout,len(task.u))
    cut=int(len(valid)*task.train_ratio)
    train, test = (valid[:cut], valid[cut+gap:]) if split is None else split
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
