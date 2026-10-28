"""Smoke tests for six property evaluators. Run: python -m src.run_property_validation"""
import numpy as np
from src.tasks.properties.nonlinear_memory import evaluate_nonlinear_memory
from src.tasks.properties.cross_memory import evaluate_cross_memory
from src.tasks.properties.kernel_rank import evaluate_kernel_rank
from src.tasks.properties.generalization_rank import evaluate_generalization_rank
from src.tasks.properties.effective_dimension import evaluate_effective_dimension
from src.tasks.properties.voltage_decay import evaluate_voltage_decay

def main():
    rng=np.random.default_rng(42); n=1200
    u=rng.uniform(-1,1,n); v=rng.uniform(-1,1,n)
    def make_features(z):
        x=np.zeros((len(z),4))
        x[1:,0]=z[:-1]**2-1/3
        x[2:,1]=z[:-2]**3
        x[2:,2]=z[1:-1]*z[:-2]
        x[:,3]=z
        return x
    X=make_features(u); Xt=make_features(v)
    nm=evaluate_nonlinear_memory(X,u,Xt,v,delays=[1,2],degrees=[2,3],washout=30)
    cm=evaluate_cross_memory(X,u,Xt,v,pairs=[(1,2)],washout=30)
    kr=evaluate_kernel_rank(rng.normal(size=(40,4)))
    gr=evaluate_generalization_rank(np.tile(np.arange(4.),(40,1))+1e-4*rng.normal(size=(40,4)))
    ed=evaluate_effective_dimension(X)
    t=np.linspace(0,2e-6,4000); wave=np.exp(-t/0.5e-6)*np.cos(2*np.pi*48e6*t)
    vd=evaluate_voltage_decay(t,wave,fit_start=.15e-6,fit_end=1.5e-6)
    assert 0<=nm.total_test<=len(nm.labels)
    assert 0<=cm.total_test<=len(cm.labels)
    assert 0<=kr.rank<=4 and 0<=gr.rank<=4
    assert 0<=ed.dimension<=4+1e-8
    assert np.isfinite(vd.tau_seconds) and vd.tau_seconds>0
    print('Nonlinear memory:',nm.total_test)
    print('Cross memory:',cm.total_test)
    print('Kernel rank:',kr.rank,'Generalization rank:',gr.rank)
    print('Effective dimension:',ed.dimension,'Voltage decay tau (s):',vd.tau_seconds)
    print('PASS: basic synthetic sanity checks')

if __name__=='__main__': main()
