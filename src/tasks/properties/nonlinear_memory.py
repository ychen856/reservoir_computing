import numpy as np
from numpy.polynomial.legendre import legval
from src.tasks.properties._common import split,score,MemoryResult

def evaluate_nonlinear_memory(X,u,X_test=None,u_test=None,delays=range(1,51),degrees=(2,3),washout=200,train_ratio=.8,ridge=1e-6,normalize=True):
    delays=list(delays); degrees=list(degrees)
    if not delays or min(delays)<1 or not degrees or min(degrees)<2: raise ValueError('Invalid delays/degrees')
    if np.max(np.abs(u))>1.00001 or (u_test is not None and np.max(np.abs(u_test))>1.00001): raise ValueError('Uniform input [-1,1] required')
    Xa,ua,ia,Xb,ub,ib=split(X,u,X_test,u_test,max(delays),washout,train_ratio)
    def targets(u,ids):
        cols=[]
        for degree in degrees:
            c=np.zeros(degree+1); c[degree]=1
            for d in delays: cols.append(legval(u[ids-d],c))
        return np.column_stack(cols)
    tr,te=score(Xa,targets(ua,ia),Xb,targets(ub,ib),ridge,normalize)
    return MemoryResult([f'P{p}(u[k-{d}])' for p in degrees for d in delays],tr,te,float(te.sum()))
