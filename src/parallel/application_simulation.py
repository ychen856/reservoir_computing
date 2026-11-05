"""Continuous-carrier and independent-waveform integration with explicit routing edges."""
from types import SimpleNamespace
import numpy as np
from scipy.integrate import solve_ivp
from src.parallel.common import simulate
from src.utils.features import extract_temporal_features,reorder_features_by_age


def integrate(case,current,t_span,t_eval,switch_period,knots=(),x0=None,rtol=1e-8,atol=1e-11):
    """Route one known current; split at switch and input interpolation knots.

    Every branch evolves at all times. Independent calls reset to zero unless
    x0 is supplied. Clock origin is t_span[0], never a label or trial index.
    """
    start,end=map(float,t_span);t=np.asarray(t_eval,float)
    if (not np.isfinite(switch_period) or switch_period<=0 or end<=start
        or t.ndim!=1 or not len(t) or not np.isfinite(t).all()
        or np.any(np.diff(t)<=0) or t[0]<start or t[-1]>end):
        raise ValueError('Invalid time grid or switch period')
    reservoir=case.reservoir()
    branches=reservoir.branches if case.architecture=='parallel' else [reservoir]
    state=np.zeros(4*case.n_pairs) if x0 is None else np.asarray(x0,float).reshape(-1).copy()
    if state.shape!=(4*case.n_pairs,) or not np.isfinite(state).all():
        raise ValueError('x0 must contain four finite states per pair')
    knots=np.asarray(knots,float)
    if not np.isfinite(knots).all():raise ValueError('Nonfinite input knots')
    edges=np.unique(np.r_[start,np.arange(1,int(np.ceil((end-start)/switch_period)))*switch_period+start,
                           knots[(knots>start)&(knots<end)],end])
    states=np.empty((len(state),len(t)))
    for left,right in zip(edges[:-1],edges[1:]):
        symbol=int(np.floor(((left+right)/2-start)/switch_period))
        selected=symbol%case.n_pairs
        def rhs(time,x):
            u=float(current(np.nextafter(right,left) if time>=right else time))
            return np.concatenate([branch.tank.nonlinear_rhs(time,x[4*j:4*j+4],
                case.eta,case.alpha,lambda _,j=j:u if case.mode!='interleaved' or j==selected else 0.)
                for j,branch in enumerate(branches)])
        sol=solve_ivp(rhs,(left,right),state,method='DOP853',dense_output=True,rtol=rtol,atol=atol)
        if not sol.success:raise RuntimeError(sol.message)
        lo,hi=np.searchsorted(t,[left,right])
        if hi>lo:states[:,lo:hi]=sol.sol(t[lo:hi])
        state=sol.y[:,-1]
    if t[-1]==end:states[:,-1]=state
    if not np.isfinite(states).all():raise RuntimeError('Nonfinite reservoir state')
    return SimpleNamespace(t=t,y=states,success=True,message='Piecewise waveform integration')


def continuous(case,u,a,encoding,amplitude):
    ts=(a.cycles_per_symbol if a.cycles_per_symbol is not None else .5)/48e6
    if encoding=='zoh':
        # Keep the already validated properties/benchmarks ZOH integration.
        from copy import copy
        cfg=copy(a);cfg.amplitude=amplitude
        return simulate(case,u,cfg).X
    if encoding not in ('carrier','pulse'):raise ValueError('Unsupported encoding')
    u=np.asarray(u,float);end=len(u)*ts
    t=np.arange(len(u)*a.points_per_symbol)*ts/a.points_per_symbol
    boundaries=np.arange(len(u)+1)*ts
    def current(time):
        n=min(int(np.searchsorted(boundaries,time,side="right")-1),len(u)-1)
        elapsed=time-boundaries[n]
        if encoding=='pulse':return amplitude*u[n] if elapsed<a.pulse_duty*ts else 0.
        return amplitude*u[n]*np.cos(2*np.pi*48e6*time)
    knots=(np.arange(len(u))+a.pulse_duty)*ts if encoding=='pulse' else ()
    sol=integrate(case,current,(0,end),t,ts,knots=knots)
    indices=tuple(i for j in range(case.n_pairs) for i in (4*j,4*j+1))
    X=extract_temporal_features(sol,len(u),ts,k=a.k,state_indices=indices)
    if case.ordering=='age':X=reorder_features_by_age(X,case.n_pairs,a.k,np.arange(len(u)))
    return X


def waveform_features(case,waves,time,a,observation=None,cache=None):
    """Sample K times per pair; age alignment follows physical sample time.

    Unlike sequence age ordering, a waveform can span many switch periods.
    Each temporal node therefore uses its own switch phase.
    """
    waves=np.asarray(waves,float);time=np.asarray(time,float)
    if waves.ndim!=2 or waves.shape[1]!=len(time) or not np.isfinite(waves).all():
        raise ValueError('Expected finite (trials,time) waveforms')
    start,stop=(time[0],time[-1]) if observation is None else observation
    if not time[0]<=start<stop<=time[-1]:raise ValueError('Invalid observation window')
    sample=np.linspace(start,stop,a.k)
    period=a.switch_cycles/48e6
    # Independent inputs reset the switch clock to the same origin every trial.
    phases=np.floor((sample-time[0])/period+1e-10).astype(int)
    pair_ids=(phases[:,None]-np.arange(case.n_pairs))%case.n_pairs
    X=[]
    cache={} if cache is None else cache
    for wave in waves:
        key=wave.tobytes()
        if key in cache:
            X.append(cache[key].copy())
            continue
        sol=integrate(case,lambda t:np.interp(t,time,wave),
                      (time[0],time[-1]),sample,period,knots=time[1:-1])
        block=sol.y.reshape(case.n_pairs,4,a.k)[:,:2,:].transpose(2,0,1)
        if case.ordering=='age':block=np.take_along_axis(block,pair_ids[:,:,None],axis=1)
        feature=block.reshape(-1)
        cache[key]=feature.copy()
        X.append(feature)
    return np.asarray(X),sample
