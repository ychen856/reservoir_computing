"""Parallel versions of the existing property evaluators and physical diagnostics."""
import numpy as np
from scipy.integrate import solve_ivp
from src.parallel.common import parser, validate, cases, simulate, Results, write_csv
from src.tasks.properties.kernel_rank import evaluate_kernel_rank
from src.tasks.properties.generalization_rank import evaluate_generalization_rank
from src.tasks.properties.effective_dimension import evaluate_effective_dimension
from src.tasks.properties.voltage_decay import evaluate_voltage_decay

from src.tasks.properties.suite import TASKS, memory


def rank(case, name, a, out):
    rng = np.random.default_rng(case.seed+2)
    base = rng.uniform(-1,1,a.trial_symbols)
    values = []
    for _ in range(a.trials):
        independent = rng.uniform(-1,1,a.trial_symbols)
        perturbed = np.clip(base+rng.normal(0,a.perturbation_std,a.trial_symbols), -1,1)
        u = independent if name == 'kernel_rank' else perturbed
        values.append(simulate(case,u,a).X[-1])
    X = np.asarray(values)
    fn = evaluate_kernel_rank if name == 'kernel_rank' else evaluate_generalization_rank
    ev = fn(X, relative_tolerance=a.rank_tolerance)
    np.savez_compressed(out.directory(case,name)/'spectrum.npz', X=X, singular_values=ev.singular_values)
    write_csv(out.directory(case,name)/'spectrum.csv', [dict(component=i+1, singular_value=float(v))
        for i,v in enumerate(ev.singular_values)])
    out.add(case,name,status='ok',feature_dim=X.shape[1],rank=ev.rank,n_trials=a.trials,
            rank_ceiling=min(a.trials-1,X.shape[1]),rank_tolerance=a.rank_tolerance)


def convergence(case,a,out):
    ts = (a.cycles_per_symbol or .5)/48e6
    t = np.arange(a.convergence_symbols*a.points_per_symbol)*ts/a.points_per_symbol
    u = np.random.default_rng(case.seed).uniform(-1,1,a.convergence_symbols)
    scale = np.tile([1.,1.,1e-3,1e-3],case.n_pairs)
    initial = np.tile([.1,-.1,0.,0.],case.n_pairs)
    indices = [i for j in range(case.n_pairs) for i in (4*j,4*j+1)]
    for forcing in ('zero','driven'):
        def current(time):
            return 0. if forcing == 'zero' else a.amplitude*u[min(int(time/ts),len(u)-1)]
        def solve(x0):
            return case.reservoir().simulate(current,(0,a.convergence_symbols*ts),t,
                    x0=x0,solver='exact_zoh',symbol_period=ts).y
        A, B = solve(np.zeros(4*case.n_pairs)), solve(initial)
        dr = np.linalg.norm((A-B)/scale[:,None],axis=0)
        dv = np.linalg.norm((A-B)[indices],axis=0)
        np.savez_compressed(out.directory(case,'fading_memory_convergence')/f'{forcing}.npz',
                            t=t,states_A=A,states_B=B,voltage_distance=dv,state_distance=dr)
        out.add(case,'fading_memory_convergence',status='ok',forcing=forcing,
            initial_distance=float(dr[0]),final_distance=float(dr[-1]),
            tail_rms_distance=float(np.sqrt(np.mean(dr[int(.9*len(dr)):]**2))))


def decay(case,a,out):
    # Exact same smooth pulse/cutoff/fit window as src.run_voltage_decay.
    # Integrate separately at every switch boundary and the hard input cutoff.
    ts = (a.cycles_per_symbol or .5)/48e6
    end, cutoff = 1.5e-6, .26e-6
    t = np.linspace(0,end,6000,endpoint=False)
    branches = (case.reservoir().branches if case.architecture == 'parallel'
                else [case.reservoir()])
    edges = np.unique(np.r_[np.arange(0,end,ts),cutoff,end])
    state = np.zeros(4*case.n_pairs)
    states = np.empty((len(state),len(t)))
    for left,right in zip(edges[:-1],edges[1:]):
        n = int(np.floor((left+right)/2/ts))
        driven = (left+right)/2 < cutoff
        def rhs(time,x):
            current = (a.amplitude*np.exp(-.5*((time-.12e-6)/.035e-6)**2)
                       *np.cos(2*np.pi*48e6*(time-.12e-6))) if driven else 0.
            return np.concatenate([branch.tank.nonlinear_rhs(time,x[4*j:4*j+4],
                case.eta,case.alpha,lambda _,j=j: current if
                case.mode != 'interleaved' or n % case.n_pairs == j else 0.)
                for j,branch in enumerate(branches)])
        sol = solve_ivp(rhs,(left,right),state,method='DOP853',dense_output=True,rtol=1e-8,atol=1e-11)
        if not sol.success:
            raise RuntimeError(sol.message)
        lo,hi = np.searchsorted(t,[left,right])
        states[:,lo:hi] = sol.sol(t[lo:hi])
        state = sol.y[:,-1]
    np.savez_compressed(out.directory(case,'voltage_decay')/'waveforms.npz',t=t,states=states)
    rows=[]
    for pair in range(case.n_pairs):
        for channel in range(2):
            try:
                ev=evaluate_voltage_decay(t,states[4*pair+channel],.30e-6,1.25e-6)
                row=dict(pair=pair,channel=channel+1,tau_s=ev.tau_seconds,
                         slope=ev.slope,fit_r_squared=ev.r_squared,n_fit=ev.n_fit)
            except ValueError as exc:
                row=dict(pair=pair,channel=channel+1,tau_s=float('nan'),error=str(exc))
            rows.append(row)
            out.add(case,'voltage_decay',status='ok',**row)
    write_csv(out.directory(case,'voltage_decay')/'by_channel.csv',rows)


def main(argv=None):
    p=parser('properties',TASKS)
    p.add_argument('--max-delay',type=int,default=50)
    p.add_argument('--nonlinear-max-delay',type=int,default=20)
    p.add_argument('--cross-max-delay',type=int,default=20)
    p.add_argument('--permutations',type=int,default=100)
    p.add_argument('--null-seed',type=int,default=2026)
    p.add_argument('--trials',type=int,default=80)
    p.add_argument('--trial-symbols',type=int,default=48)
    p.add_argument('--rank-tolerance',type=float,default=1e-3)
    p.add_argument('--perturbation-std',type=float,default=.02)
    p.add_argument('--convergence-symbols',type=int,default=480)
    a=p.parse_args(argv); validate(a)
    if min(a.max_delay,a.nonlinear_max_delay,a.permutations,a.trial_symbols,a.convergence_symbols)<1 or a.cross_max_delay<2 or a.trials<2:
        p.error('invalid memory/rank/null parameters')
    # _common.split purges max_delay samples at the train/test boundary.
    start=max(a.washout,a.max_delay,a.nonlinear_max_delay,a.cross_max_delay)
    if (a.symbols-start)*(1-a.train_ratio) <= max(a.max_delay,a.nonlinear_max_delay,a.cross_max_delay)+3:
        p.error('too few test samples after memory-delay purge')
    out=Results(a)
    for case in cases(a):
        cache={}
        for name in a.tasks:
            try:
                if name in ('kernel_rank','generalization_rank'):
                    rank(case,name,a,out)
                elif name=='voltage_decay':
                    decay(case,a,out)
                elif name=='fading_memory_convergence':
                    convergence(case,a,out)
                else:
                    distribution='uniform' if name.startswith('nonlinear') or name=='effective_dimension' else 'bipolar'
                    if distribution not in cache:
                        rng=np.random.default_rng(case.seed)
                        u=(rng.uniform(-1,1,a.symbols) if distribution=='uniform'
                           else rng.choice([-1.,1.],a.symbols))
                        cache[distribution]=(u,simulate(case,u,a).X)
                    u,X=cache[distribution]
                    if name=='effective_dimension':
                        ev=evaluate_effective_dimension(X[a.washout:],standardize=False)
                        np.savez_compressed(out.directory(case,name)/'spectrum.npz',eigenvalues=ev.eigenvalues)
                        write_csv(out.directory(case,name)/'spectrum.csv',[
                            dict(component=i+1,eigenvalue=float(v)) for i,v in enumerate(ev.eigenvalues)])
                        out.add(case,name,status='ok',feature_dim=X.shape[1],dimension=ev.dimension)
                    else:
                        memory(case,name,a,out,(u,X))
            except Exception as exc:
                out.fail(case,name,exc)
    if any(row.get('status')=='failed' for row in out.rows):
        raise SystemExit('Some property cases failed; see summary.csv')

if __name__=='__main__':
    main()
