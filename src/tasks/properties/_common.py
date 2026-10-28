import numpy as np
from dataclasses import dataclass
from src.readout.ridge import train_ridge_regression, predict_ridge_regression

@dataclass
class MemoryResult:
    labels: list
    train_scores: np.ndarray
    test_scores: np.ndarray
    total_test: float

def split(X,u,X_test,u_test,max_delay,washout,train_ratio):
    X=np.asarray(X,float); u=np.asarray(u,float).ravel()
    if X.ndim!=2 or len(X)!=len(u): raise ValueError('X/u mismatch')
    start=max(max_delay,washout)
    if X_test is None or u_test is None:
        cut=start+int((len(u)-start)*train_ratio)
        if cut<=start+2 or cut+max_delay>=len(u)-2: raise ValueError('Not enough samples')
        return X[start:cut],u,np.arange(start,cut),X[cut+max_delay:],u,np.arange(cut+max_delay,len(u))
    Xt=np.asarray(X_test,float); ut=np.asarray(u_test,float).ravel()
    if Xt.ndim!=2 or len(Xt)!=len(ut) or Xt.shape[1]!=X.shape[1]: raise ValueError('Test X/u mismatch')
    if min(len(u),len(ut))<=start+2: raise ValueError('Not enough samples')
    return X[start:],u,np.arange(start,len(u)),Xt[start:],ut,np.arange(start,len(ut))

def score(Xa,Ya,Xb,Yb,ridge=1e-6,normalize=True):
    if normalize:
        mu=Xa.mean(0); sd=Xa.std(0); sd[sd<1e-12]=1
        Xa=(Xa-mu)/sd; Xb=(Xb-mu)/sd
    W=train_ridge_regression(Xa,Ya,ridge=ridge)
    def calc(Y,P):
        ans=[]
        for i in range(Y.shape[1]):
            a,b=Y[:,i],P[:,i]
            if np.std(a)<1e-12 or np.std(b)<1e-12: ans.append(0.); continue
            r=np.corrcoef(a,b)[0,1]; ans.append(float(r*r) if np.isfinite(r) else 0.)
        return np.asarray(ans)
    return calc(Ya,predict_ridge_regression(Xa,W)),calc(Yb,predict_ridge_regression(Xb,W))
