import numpy as np
from numpy.polynomial.legendre import legval
from src.tasks.properties.memory_capacity import evaluate_linear_memory_capacity
from src.tasks.properties.nonlinear_memory import evaluate_nonlinear_memory
from src.tasks.properties.cross_memory import evaluate_cross_memory
from src.tasks.properties._common import split, score
from src.utils.csv_output import write_csv


TASKS = ('memory_capacity', 'memory_capacity_null', 'nonlinear_memory_capacity',
         'nonlinear_memory_capacity_null', 'cross_memory_capacity',
         'cross_memory_capacity_null', 'kernel_rank', 'generalization_rank',
         'effective_dimension', 'voltage_decay', 'fading_memory_convergence')


def memory(case, name, a, out, sim):
    u, X = sim
    kw = dict(washout=a.washout, train_ratio=a.train_ratio, ridge=a.ridge, normalize=True)
    if name.startswith('memory_capacity'):
        result = evaluate_linear_memory_capacity(X, u, max_delay=a.max_delay, **kw)
        labels = [f'delay{d}' for d in result.delays]
        train, test = result.train_capacity, result.test_capacity
        start = max(a.washout, a.max_delay)
        ids = np.arange(start, len(u))
        Y = np.column_stack([u[ids-d] for d in range(1, a.max_delay+1)])
        cut = int(len(ids)*a.train_ratio)
        Xa, Xb, Ya, Yb = X[start:start+cut], X[start+cut:], Y[:cut], Y[cut:]
    else:
        nonlinear = name.startswith('nonlinear')
        maximum = a.nonlinear_max_delay if nonlinear else a.cross_max_delay
        if nonlinear:
            result = evaluate_nonlinear_memory(X, u, delays=range(1, maximum+1), degrees=(2,3), **kw)
        else:
            result = evaluate_cross_memory(X, u, max_delay=maximum, **kw)
        labels, train, test = result.labels, result.train_scores, result.test_scores
        Xa, ua, ia, Xb, ub, ib = split(X, u, None, None, maximum, a.washout, a.train_ratio)
        def targets(values, ids):
            if nonlinear:
                return np.column_stack([legval(values[ids-d], [0]*degree+[1])
                    for degree in (2,3) for d in range(1, maximum+1)])
            return np.column_stack([values[ids-i]*values[ids-j]
                for i in range(1, maximum+1) for j in range(i+1, maximum+1)])
        Ya, Yb = targets(ua, ia), targets(ub, ib)
    checked_train, checked_test = score(Xa, Ya, Xb, Yb, a.ridge, True)
    np.testing.assert_allclose(checked_train, train, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(checked_test, test, rtol=1e-5, atol=1e-7)
    detail = [dict(target=l, train=float(tr), test=float(te)) for l,tr,te in zip(labels,train,test)]
    totals = dict(status='ok', feature_dim=X.shape[1], n_train=len(Xa), n_test=len(Xb),
                  train_mc=float(np.sum(train)), test_mc=float(np.sum(test)))
    d = out.directory(case, name)
    if name.endswith('_null'):
        rng = np.random.default_rng(a.null_seed)
        null = []
        for _ in range(a.permutations):
            if name == 'memory_capacity_null':
                # Original linear-MC null permutes each delay independently.
                A = np.column_stack([rng.permutation(Ya[:,j]) for j in range(Ya.shape[1])])
                B = np.column_stack([rng.permutation(Yb[:,j]) for j in range(Yb.shape[1])])
            else:
                # Match original nonlinear/CMC null: shuffle complete target rows.
                A, B = Ya[rng.permutation(len(Ya))], Yb[rng.permutation(len(Yb))]
            null.append(score(Xa, A, Xb, B, a.ridge, True)[1])
        null = np.asarray(null)
        null_totals = null.sum(1)
        mean, p95 = null.mean(0), np.quantile(null, .95, axis=0)
        for row, nm, q in zip(detail, mean, p95):
            row.update(null_mean=float(nm), null_p95=float(q), excess_test=row['test']-float(nm))
        totals.update(null_total=float(null_totals.mean()),
            excess_total=float(np.sum(test)-null_totals.mean()),
            null_total_p95=float(np.quantile(null_totals,.95)),
            p_total=float((1+np.count_nonzero(null_totals >= np.sum(test)))/(1+len(null))))
        np.savez_compressed(d/'null_scores.npz', null_test=null, real_train=train, real_test=test)
        write_csv(d/'null_totals.csv', [dict(permutation=i+1, total=float(v)) for i,v in enumerate(null_totals)])
    write_csv(d/'by_target.csv', detail)
    out.add(case, name, **totals)
