import numpy as np
from dataclasses import dataclass
@dataclass
class RankResult:
    rank: int
    singular_values: np.ndarray
    feature_dim: int
    n_trials: int

def numerical_rank(X,relative_tolerance=1e-6):
    X=np.asarray(X,float)
    if X.ndim!=2 or X.shape[0]<2 or not np.isfinite(X).all(): raise ValueError('Invalid feature matrix')
    if relative_tolerance<=0: raise ValueError('Invalid tolerance')
    s=np.linalg.svd(X-X.mean(0),compute_uv=False)
    r=int(np.sum(s>s[0]*relative_tolerance)) if len(s) and s[0]>0 else 0
    return RankResult(r,s,X.shape[1],len(X))

def evaluate_kernel_rank(trial_features,relative_tolerance=1e-6):
    """Each row is final state after an independent random input history."""
    return numerical_rank(trial_features,relative_tolerance)
