import numpy as np
from src.tasks.properties._common import split,score,MemoryResult

def evaluate_cross_memory(X,u,X_test=None,u_test=None,pairs=None,max_delay=20,washout=200,train_ratio=.8,ridge=1e-6,normalize=True):
    if pairs is None: pairs=[(i,j) for i in range(1,max_delay+1) for j in range(i+1,max_delay+1)]
    pairs=list(pairs)
    if not pairs or any(i<1 or j<=i for i,j in pairs): raise ValueError('Require 1<=i<j')
    Xa,ua,ia,Xb,ub,ib=split(X,u,X_test,u_test,max(j for i,j in pairs),washout,train_ratio)
    def targets(u,ids): return np.column_stack([u[ids-i]*u[ids-j] for i,j in pairs])
    tr,te=score(Xa,targets(ua,ia),Xb,targets(ub,ib),ridge,normalize)
    return MemoryResult([f'u[k-{i}]*u[k-{j}]' for i,j in pairs],tr,te,float(te.sum()))
