"""Parallel PAM4 equalization, noisy M-PSK and waveform classification."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from src.parallel.common import parser,validate,cases,Results,write_csv
from src.parallel.application_simulation import continuous,waveform_features
from src.tasks.applications.pam4_nonlinear import generate_pam4_nonlinear_task
from src.tasks.applications.pam4 import quantize_pam4_class
from src.tasks.applications.mpsk import generate_mpsk_task
from src.tasks.applications.waveform_classification import generate_waveform_classification,EXPERIMENTS
from src.readout.preprocessing import fit_standardizer,transform_standardizer
from src.readout.ridge import train_ridge_regression,predict_ridge_regression
from src.readout.metrics import nrmse
from src.simulation.evaluation import evaluate_independent_classification

TASKS=('pam4','pam4_nonlinear','mpsk','waveform_classification')


def fingerprint(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def future_window(X,lookahead):
    X=np.asarray(X,float)
    if X.ndim!=2 or lookahead<0 or lookahead>=len(X):raise ValueError('Invalid feature window')
    n=len(X)-lookahead
    return np.concatenate([X[j:j+n] for j in range(lookahead+1)],axis=1)


def window_indices(n,washout,ratio,max_window,radius=2):
    """Same target rows for all windows, with channel/future-feature isolation."""
    ids=np.arange(max(washout,radius),n-radius)
    cut=int(len(ids)*ratio)
    if cut<=0 or cut>=len(ids):raise ValueError('Empty split')
    boundary=ids[cut]
    train=ids[:cut];test=ids[cut:]
    # The symmetric ISI channel sees source symbols through n+2.
    train=train[train+max_window+radius<boundary]
    test=test[test+max_window<n-radius]
    if min(len(train),len(test))<3:raise ValueError('Too few samples after window/channel purge')
    return train,test,boundary


def clock_features(X,mode):
    parity=(1-2*(np.arange(len(X))%2))[:,None]
    if mode=='plain':return X
    if mode=='parity_only':return parity*X
    if mode=='parity_aware':return np.concatenate([X,parity*X],axis=1)
    raise ValueError('Unknown clock feature transform')


def fit_pam4(X,y,train,test,ridge,readout):
    mu,sd=fit_standardizer(X[train])
    A=transform_standardizer(X[train],mu,sd);B=transform_standardizer(X[test],mu,sd)
    if readout=='scalar':targets=y[train].astype(float)
    elif readout=='onehot':targets=np.eye(4)[y[train]]
    elif readout=='ordinal':targets=(y[train,None]>=np.arange(1,4)).astype(float)
    else:raise ValueError('Unknown PAM4 readout')
    W=train_ridge_regression(A,targets,ridge=ridge)
    pa=np.asarray(predict_ridge_regression(A,W));pb=np.asarray(predict_ridge_regression(B,W))
    def decode(p):
        if readout=='scalar':return quantize_pam4_class(p.reshape(-1))
        if readout=='onehot':return np.argmax(p,axis=1)
        return np.sum(p>=.5,axis=1).astype(int)
    ha,hb=decode(pa),decode(pb)
    cm=np.zeros((4,4),int);np.add.at(cm,(y[test],hb),1)
    values=dict(train_accuracy=float(np.mean(ha==y[train])),test_accuracy=float(np.mean(hb==y[test])),
                train_ser=float(np.mean(ha!=y[train])),test_ser=float(np.mean(hb!=y[test])),
                feature_dim=X.shape[1],n_train=len(train),n_test=len(test))
    if readout=='scalar':values.update(train_nrmse=float(nrmse(y[train],pa.reshape(-1))),test_nrmse=float(nrmse(y[test],pb.reshape(-1))))
    arrays=dict(train_indices=train,test_indices=test,y_train=y[train],y_test=y[test],
        train_scores=pa,test_scores=pb,y_train_pred=ha,y_test_pred=hb,confusion_matrix=cm,W=W,mu=mu,std=sd)
    return values,arrays


def save_prediction(directory,label,arrays):
    np.savez_compressed(directory/f'{label}_predictions.npz',**arrays)
    cm=arrays['confusion_matrix']
    write_csv(directory/f'{label}_confusion.csv',[dict(actual=i,predicted=j,count=int(cm[i,j]))
        for i in range(len(cm)) for j in range(len(cm))])


def pam4(case,family,a,out):
    strengths=[0.] if family=='pam4' else a.strengths
    max_window=max(a.windows)
    for strength in strengths:
        for snr in a.snr_db:
            clean=np.isposinf(snr)
            tag=f'{family}_strength{strength:g}_snr{snr:g}'
            try:
                task=generate_pam4_nonlinear_task(n_symbols=a.symbols,seed=case.seed,
                    washout=a.washout,train_ratio=a.train_ratio,quadratic=.036*strength,
                    cubic=-.011*strength,add_awgn=not clean,snr_db=snr,
                    noise_seed=case.seed+a.noise_seed_offset)
                train,test,boundary=window_indices(len(task.u),a.washout,a.train_ratio,max_window)
                encoding=(('carrier' if family=='pam4_nonlinear' else 'zoh')
                           if a.encoding=='task-default' else a.encoding)
                amp=a.nonlinear_amplitude if family=='pam4_nonlinear' else a.amplitude
                ridge=a.nonlinear_ridge if family=='pam4_nonlinear' else a.ridge
                X=continuous(case,task.u,a,encoding,amp)
                directory=out.directory(case,tag)
                np.savez_compressed(directory/'input_states.npz',u=task.u,y=task.y,X=X)
                settings=dict(family=family,strength=strength,snr_db='clean' if clean else snr,
                    encoding=encoding,amplitude=amp,ridge=ridge,cycles_per_symbol=a.cycles_per_symbol or .5,
                    input_sha256=fingerprint(task.u),channel_radius=2,
                    max_window=max_window,train_test_boundary=int(boundary),
                    measured_snr_db=task.metadata['measured_snr_db'],input_scale=task.metadata['input_scale'])
                (directory/'task.json').write_text(json.dumps(settings,indent=2))
                q=task.metadata['linear_channel_output'][2:-2]
                derivative=1+2*.036*strength*q+3*(-.011)*strength*q*q
                write_csv(directory/'channel_diagnostics.csv',[dict(strength=strength,
                    derivative_min=float(derivative.min()),derivative_max=float(derivative.max()),
                    fraction_nonpositive=float(np.mean(derivative<=0)))])
                baseline_rows=[]
                for L in a.windows:
                    # Age ordering occurs BEFORE temporal window concatenation.
                    Z=future_window(X,L);raw=future_window(task.u[:,None],L)
                    for readout in a.pam4_readouts:
                        feature_modes=a.pam4_feature_modes or (
                            ['plain','parity_only','parity_aware'] if encoding=='carrier' else ['plain'])
                        for transform in feature_modes:
                            label=f'L{L}_{transform}_{readout}'
                            features=clock_features(Z,transform)
                            values,arrays=fit_pam4(features,task.y,train,test,ridge,readout)
                            save_prediction(directory,label,arrays)
                            out.add(case,tag,status='ok',lookahead_symbols=L,readout=readout,
                                feature_transform=transform,encoding=encoding,amplitude=amp,ridge=ridge,
                                strength=strength,snr_db='clean' if clean else snr,
                                train_test_boundary=int(boundary),input_sha256=fingerprint(task.u),**values)
                        for degree in a.input_degrees:
                            base=raw if degree==1 else PolynomialFeatures(degree=degree,include_bias=False).fit_transform(raw)
                            bv,_=fit_pam4(base,task.y,train,test,ridge,readout)
                            baseline_rows.append(dict(lookahead_symbols=L,readout=readout,
                                method=f'input_window_degree{degree}',nonlinear_readout_diagnostic=degree>1,**bv))
                write_csv(directory/'baselines.csv',baseline_rows)
            except Exception as exc:out.fail(case,tag,exc)


def trial_score(Xtrain,ytrain,Xtest,ytest,ridge,n_classes):
    if Xtrain.shape[1]>len(Xtrain) and ridge>0:
        mu,sd=fit_standardizer(Xtrain)
        A=transform_standardizer(Xtrain,mu,sd);B=transform_standardizer(Xtest,mu,sd)
        targets=np.eye(n_classes)[ytrain];mean=targets.mean(0)
        # Center both sides to preserve the unregularized intercept exactly.
        center=A.mean(0);Ac=A-center
        coef=Ac.T@np.linalg.solve(Ac@Ac.T+ridge*np.eye(len(A)),targets-mean)
        bias=mean-center@coef;W=np.vstack([bias,coef])
        ha=np.argmax(A@coef+bias,axis=1);hb=np.argmax(B@coef+bias,axis=1)
        cm=np.zeros((n_classes,n_classes),int);np.add.at(cm,(ytest,hb),1)
        ev=SimpleNamespace(train_accuracy=float(np.mean(ha==ytrain)),test_accuracy=float(np.mean(hb==ytest)),
            train_ser=float(np.mean(ha!=ytrain)),test_ser=float(np.mean(hb!=ytest)),feature_dim=Xtrain.shape[1],
            n_train=len(ytrain),n_test=len(ytest),y_train=ytrain,y_test=ytest,y_train_pred=ha,y_test_pred=hb,
            confusion_matrix=cm,W=W,mu=mu,std=sd)
    else:
        ev=evaluate_independent_classification(Xtrain,ytrain,Xtest,ytest,
                ridge=ridge,normalize=True,n_classes=n_classes)
    values=dict(train_accuracy=float(ev.train_accuracy),test_accuracy=float(ev.test_accuracy),
                train_ser=float(ev.train_ser),test_ser=float(ev.test_ser),feature_dim=ev.feature_dim,
                n_train=ev.n_train,n_test=ev.n_test)
    arrays=dict(y_train=ev.y_train,y_test=ev.y_test,y_train_pred=ev.y_train_pred,
                y_test_pred=ev.y_test_pred,confusion_matrix=ev.confusion_matrix,W=ev.W,mu=ev.mu,std=ev.std)
    return values,arrays


def independent(case,family,a,out):
    test_seed=case.seed+a.test_seed_offset
    if test_seed==case.seed:raise ValueError('Independent trial train/test seeds must differ')
    variants=a.psk_orders if family=='mpsk' else a.waveform_experiments
    noise_cases=a.psk_snr_db if family=='mpsk' else [None]
    for variant in variants:
        for snr in noise_cases:
            tag=(f'mpsk_M{variant}_snr{snr:g}' if family=='mpsk' else f'waveform_{variant}')
            try:
                if family=='mpsk':
                    n=int(np.ceil(1.2e-6*48e6*a.psk_samples_per_cycle))
                    t=np.linspace(0,1.2e-6,n+1)
                    kw=dict(M=variant,t_eval=t,t_center=.5e-6,sigma=(8/48e6)/6,
                        Iin_peak=a.amplitude,fc=48e6,add_noise=not np.isposinf(snr),snr_db=snr,
                        adc_enabled=not a.no_adc,fs_adc=192e6,adc_bits=a.adc_bits)
                    tr=generate_mpsk_task(n_repeat=a.train_repeats,seed=case.seed,**kw)
                    te=generate_mpsk_task(n_repeat=a.test_repeats,seed=test_seed,**kw)
                    obs=(.5e-6,t[-1]);n_classes=variant
                else:
                    kw=dict(n_points=a.waveform_points,duration=.5e-6,train_ratio=a.train_ratio,**EXPERIMENTS[variant])
                    tr=generate_waveform_classification(n_per_class=a.waveform_train_per_class,seed=case.seed,**kw)
                    te=generate_waveform_classification(n_per_class=a.waveform_test_per_class,seed=test_seed,**kw)
                    # Matched nuisance-factor draws are confined to each independent data set.
                    tr.u=tr.u*a.amplitude;te.u=te.u*a.amplitude
                    t=tr.metadata['t_eval'];obs=(t[0],t[-1]);n_classes=4
                feature_cache={}
                A,samples=waveform_features(case,tr.u,t,a,obs,cache=feature_cache)
                B,_=waveform_features(case,te.u,t,a,obs,cache=feature_cache)
                values,arrays=trial_score(A,tr.y,B,te.y,a.ridge,n_classes)
                directory=out.directory(case,tag);save_prediction(directory,'reservoir',arrays)
                np.savez_compressed(directory/'features.npz',X_train=A,X_test=B,
                    y_train=tr.y,y_test=te.y,sample_times=samples,
                    train_seed=case.seed,test_seed=test_seed)
                if a.save_waveforms:np.savez_compressed(directory/'waveforms.npz',t=t,train=tr.u,test=te.u)
                (directory/'task.json').write_text(json.dumps(dict(family=family,variant=variant,
                    train_seed=case.seed,test_seed=test_seed,switch_period_s=a.switch_cycles/48e6,
                    reset_between_trials=True,within_trial_interleaving=True,
                    interpolation='piecewise-linear (same input reconstruction as existing trial runners)',
                    sample_times_s=samples.tolist(),train_input_sha256=fingerprint(tr.u),
                    test_input_sha256=fingerprint(te.u),amplitude=a.amplitude,ridge=a.ridge,
                    adc_enabled=(not a.no_adc) if family=='mpsk' else False,
                    adc_bits=a.adc_bits if family=='mpsk' and not a.no_adc else None,
                    fs_adc=192e6 if family=='mpsk' and not a.no_adc else None,
                    snr_db=('clean' if snr is not None and np.isposinf(snr) else snr)),indent=2))
                out.add(case,tag,status='ok',readout='onehot',within_trial_interleaving=True,
                    train_seed=case.seed,test_seed=test_seed,switch_period_s=a.switch_cycles/48e6,
                    amplitude=a.amplitude,ridge=a.ridge,
                    train_input_sha256=fingerprint(tr.u),test_input_sha256=fingerprint(te.u),**values)
                baseline_rows=[]
                # For PSK compare exactly the same observation window as the reservoir.
                compact_train=np.asarray([np.interp(samples,t,w) for w in tr.u])
                compact_test=np.asarray([np.interp(samples,t,w) for w in te.u])
                matched_times=np.linspace(obs[0],obs[1],A.shape[1])
                matched_train=np.asarray([np.interp(matched_times,t,w) for w in tr.u])
                matched_test=np.asarray([np.interp(matched_times,t,w) for w in te.u])
                methods=[('input_K',compact_train,compact_test,False),
                         ('input_matched_dimension',matched_train,matched_test,False),
                         ('input_full',tr.u,te.u,False)]
                if family=='waveform_classification':
                    methods.append(('input_FFT_magnitude_diagnostic',np.abs(np.fft.rfft(tr.u,axis=1)),
                                    np.abs(np.fft.rfft(te.u,axis=1)),True))
                for label,Xa,Xb,diagnostic in methods:
                    # Full waveform input is intentionally a higher-bandwidth baseline.
                    bv,ba=trial_score(Xa,tr.y,Xb,te.y,a.ridge,n_classes)
                    baseline_rows.append(dict(method=label,nonlinear_diagnostic=diagnostic,**bv))
                    save_prediction(directory,label,ba)
                write_csv(directory/'baselines.csv',baseline_rows)
            except Exception as exc:out.fail(case,tag,exc)


def main(argv=None):
    p=parser('applications',TASKS)
    p.add_argument('--windows',type=int,nargs='+',default=[0,20])
    p.add_argument('--strengths',type=float,nargs='+',default=[.5,1.,2.5])
    p.add_argument('--snr-db',type=float,nargs='+',default=[float('inf')],help='inf means clean, e.g. --snr-db inf 40 20')
    p.add_argument('--psk-snr-db',type=float,nargs='+',default=[20.],help='PSK noise settings, separate from PAM4 --snr-db')
    p.add_argument('--encoding',choices=['task-default','zoh','carrier','pulse'],default='task-default')
    p.add_argument('--pulse-duty',type=float,default=1.)
    p.add_argument('--nonlinear-amplitude',type=float,default=3e-4)
    p.add_argument('--nonlinear-ridge',type=float,default=.1)
    p.add_argument('--pam4-readouts',nargs='+',choices=['scalar','onehot','ordinal'],default=['scalar','onehot','ordinal'])
    p.add_argument('--pam4-feature-modes',nargs='+',choices=['plain','parity_only','parity_aware'],default=None,help='Default: all three for carrier; plain otherwise')
    p.add_argument('--input-degrees',type=int,nargs='+',default=[1,2])
    p.add_argument('--noise-seed-offset',type=int,default=81)
    p.add_argument('--psk-orders',type=int,nargs='+',default=[4,8,16,32])
    p.add_argument('--train-repeats',type=int,default=20)
    p.add_argument('--test-repeats',type=int,default=10)
    p.add_argument('--test-seed-offset',type=int,default=1000003)
    p.add_argument('--psk-samples-per-cycle',type=int,default=40)
    p.add_argument('--no-adc',action='store_true')
    p.add_argument('--adc-bits',type=int,default=3)
    p.add_argument('--waveform-experiments',nargs='+',choices=list(EXPERIMENTS),default=list(EXPERIMENTS))
    p.add_argument('--waveform-points',type=int,default=128)
    p.add_argument('--waveform-train-per-class',type=int,default=40)
    p.add_argument('--waveform-test-per-class',type=int,default=20)
    p.add_argument('--switch-cycles',type=float,default=.5,help='Fixed within-trial switch period / carrier period')
    p.add_argument('--save-waveforms',action='store_true')
    a=p.parse_args(argv);validate(a)
    if a.smoke:
        a.windows=[0,2];a.strengths=[1.];a.psk_orders=[4]
        a.train_repeats,a.test_repeats=2,1
        a.psk_samples_per_cycle=8;a.waveform_points=32
        a.waveform_train_per_class=a.waveform_test_per_class=5
        a.waveform_experiments=['E']
    if (min(a.windows)<0 or max(a.windows)>=a.symbols-a.washout-10
        or min(a.strengths)<0 or not 0<a.pulse_duty<=1 or a.switch_cycles<=0
        or min(a.psk_orders)<2 or min(a.train_repeats,a.test_repeats,a.adc_bits)<1
        or a.psk_samples_per_cycle<4 or a.waveform_points<32
        or min(a.waveform_train_per_class,a.waveform_test_per_class)<5
        or a.nonlinear_ridge<0 or a.nonlinear_amplitude<=0
        or a.test_seed_offset==0 or any(v not in (1,2,3) for v in a.input_degrees)
        or any(np.isnan(x) or np.isneginf(x) for x in a.snr_db+a.psk_snr_db)):
        p.error('Invalid channel, trial, noise or readout settings')
    if not np.isfinite([a.switch_cycles,a.nonlinear_amplitude,a.nonlinear_ridge,a.pulse_duty]).all():
        p.error('Physical/readout settings must be finite')
    out=Results(a)
    config=json.loads((out.root/'config.json').read_text())
    config['protocol']='PAM4 continuous ZOH/carrier/pulse; independent trial within-waveform interleaving; fixed ridge; train-only standardization'
    for name in ['parallel/applications.py','parallel/application_simulation.py']:
        source=Path(__file__).resolve().parents[1]/name
        config['source_sha256'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
    config['snr_db']=['clean' if np.isposinf(v) else v for v in a.snr_db]
    config['psk_snr_db']=['clean' if np.isposinf(v) else v for v in a.psk_snr_db]
    (out.root/'config.json').write_text(json.dumps(config,indent=2,allow_nan=False))
    for case in cases(a):
        for family in a.tasks:
            if family in ('pam4','pam4_nonlinear'):pam4(case,family,a,out)
            else:independent(case,family,a,out)
    if any(row.get('status')=='failed' for row in out.rows):
        raise SystemExit('Some application cases failed; inspect summary.csv')

if __name__=='__main__':main()
