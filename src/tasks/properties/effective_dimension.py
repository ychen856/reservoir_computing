import numpy as np
from dataclasses import dataclass
@dataclass
class EffectiveDimensionResult:
    dimension: float
    eigenvalues: np.ndarray
    feature_dim: int

def evaluate_effective_dimension(X,standardize=False):
    X=np.asarray(X,float)
    if X.ndim!=2 or len(X)<2 or not np.isfinite(X).all(): raise ValueError('Invalid X')
    A=X-X.mean(0)
    if standardize:
        sd=A.std(0); sd[sd<1e-12]=1; A=A/sd
    eig=np.linalg.svd(A,compute_uv=False)**2/(len(A)-1)
    denom=np.dot(eig,eig)
    dim=float(eig.sum()**2/denom) if denom>0 else 0.
    return EffectiveDimensionResult(dim,eig,X.shape[1])
