"""Shared experiment harness; does not modify existing NARMA or property code."""
from pathlib import Path
import csv
import numpy as np
from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import simulate_continuous_task, simulate_independent_trials
from src.task_evaluation import evaluate, history_features, sequential_indices, stratified_indices

FC=48e6
TS_RATIO=0.5
TS=TS_RATIO/FC
K=8
POINTS=16
INPUT_SCALE=1e-2
RIDGE=1e-6
CONFIGS=((0.,0.),(1.,0.),(1.,8.33))
OUTPUT=Path('results/remaining_tasks')


def save_rows(filename, rows):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not rows: return
    fields=list(rows[0].keys())
    with (OUTPUT/filename).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print('Saved',OUTPUT/filename)


def continuous_experiment(task, task_name, history=10, polynomial=False):
    from sklearn.preprocessing import PolynomialFeatures
    train, test=sequential_indices(task)
    baseline={'Current input':task.u.reshape(-1,1),
              f'{history}-tap history':history_features(task.u,history)}
    if polynomial:
        baseline[f'{history}-tap polynomial']=PolynomialFeatures(degree=2,include_bias=False).fit_transform(baseline[f'{history}-tap history'])
    rows=[]
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for eta,alpha in CONFIGS:
        res=PTSingleReservoir(eta=eta,alpha=alpha)
        sim=simulate_continuous_task(res,task,TS,points_per_symbol=POINTS,
                                     k=K,amplitude_scale=INPUT_SCALE,
                                     encoding='zoh',solver='exact_zoh' if alpha==0 else 'ivp')
        methods={'PT reservoir':sim.X, **baseline}
        if polynomial:
            methods['PT polynomial']=PolynomialFeatures(degree=2,include_bias=False).fit_transform(sim.X)
        for method,X in methods.items():
            result=evaluate(X,task.y,train,test,task.task_type,RIDGE)
            row={'task':task_name,'eta':eta,'alpha':alpha,'ts_ratio':TS_RATIO,
                 'method':method,'train':result['train'],'test':result['test']}
            rows.append(row)
            print(f'{task_name:20s} eta={eta:g} alpha={alpha:g} {method:22s} train={result["train"]:.5f} test={result["test"]:.5f}')
            if 'confusion' in result:
                np.savetxt(OUTPUT/f'{task_name}_eta{eta:g}_alpha{alpha:g}_{method.replace(" ","_")}_confusion.csv',
                           result['confusion'],fmt='%d',delimiter=',')
    save_rows(task_name+'_summary.csv',rows)


def waveform_experiment(task):
    """Each waveform is an independent IVP trial, reservoir reset to zero."""
    train,test=stratified_indices(task.y,task.train_ratio)
    rows=[]
    OUTPUT.mkdir(parents=True,exist_ok=True)
    # Input-only waveform baseline uses the same K temporal samples as reservoir.
    indices=np.linspace(0,task.u.shape[1]-1,K,dtype=int)
    raw_features=task.u[:,indices]
    for eta,alpha in CONFIGS:
        res=PTSingleReservoir(eta=eta,alpha=alpha)
        # runner independent-trial path uses IVP and interpolates supplied waveform.
        # Scale input at source to preserve a consistent current amplitude.
        from src.tasks.base import TaskData
        scaled=TaskData(task.name,task.u*INPUT_SCALE,task.y,task.task_type,
                        train_ratio=task.train_ratio,metadata=task.metadata)
        sim=simulate_independent_trials(res,scaled,k=K)
        for method,X in [('PT reservoir',sim.X),('Input temporal samples',raw_features)]:
            result=evaluate(X,task.y,train,test,'classification',RIDGE)
            rows.append({'task':'waveform_classification','eta':eta,'alpha':alpha,
                         'method':method,'train':result['train'],'test':result['test']})
            print(f'waveform eta={eta:g} alpha={alpha:g} {method} train={result["train"]:.4f} test={result["test"]:.4f}')
            np.savetxt(OUTPUT/f'waveform_eta{eta:g}_alpha{alpha:g}_{method.replace(" ","_")}_confusion.csv',
                       result['confusion'],fmt='%d',delimiter=',')
    save_rows('waveform_classification_summary.csv',rows)
