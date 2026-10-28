"""PT single-tank linearized EP/stability explorer.

Run from project root:
    python3 pt_ep_explorer_v2.py --eta-min 0 --eta-max 1.3 --n 1301 --eta-select 1.0

Requires src.circuits.pt_single.PTSingleReservoir.
The minimum-gap point is only an EP candidate, not a proven EP.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, linear_sum_assignment, minimize_scalar

from src.circuits.pt_single import PTSingleReservoir


def eig_diagnostics(A):
    values, vectors = np.linalg.eig(A)
    vectors /= np.linalg.norm(vectors, axis=0)
    pairs = [(i, j) for i in range(len(values)) for j in range(i + 1, len(values))]
    gaps = np.array([abs(values[i] - values[j]) for i, j in pairs])
    k = int(np.argmin(gaps))
    i, j = pairs[k]
    overlap = abs(np.vdot(vectors[:, i], vectors[:, j]))
    return float(np.max(values.real)), float(gaps[k]), float(overlap), (i, j)


def tracked_eigenvalues(matrices):
    values = np.empty((len(matrices), matrices[0].shape[0]), dtype=complex)
    values[0] = np.linalg.eigvals(matrices[0])
    values[0] = values[0][np.lexsort((values[0].imag, values[0].real))]
    for k in range(1, len(matrices)):
        current = np.linalg.eigvals(matrices[k])
        rows, cols = linear_sum_assignment(abs(values[k - 1, :, None] - current[None, :]))
        values[k, rows] = current[cols]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eta-min', type=float, default=0.0)
    parser.add_argument('--eta-max', type=float, default=1.3)
    parser.add_argument('--n', type=int, default=1301)
    parser.add_argument('--eta-select', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=8.33,
                        help='Reference only: zero-input linearization is independent of alpha')
    parser.add_argument('--out', default='results/pt_ep_explorer_v2.png')
    args = parser.parse_args()
    if args.n < 3 or args.eta_min >= args.eta_max:
        parser.error('Require n >= 3 and eta-min < eta-max')
    if not args.eta_min <= args.eta_select <= args.eta_max:
        parser.error('eta-select must lie inside the sweep range')

    tank = PTSingleReservoir(eta=args.eta_select, alpha=args.alpha).tank

    def matrix(e):
        return tank.build_state_matrix(float(e))

    def spectral(e):
        return eig_diagnostics(matrix(e))[0]

    def gap(e):
        return eig_diagnostics(matrix(e))[1]

    eta = np.linspace(args.eta_min, args.eta_max, args.n)
    matrices = [matrix(e) for e in eta]
    values = tracked_eigenvalues(matrices)
    spectral_values = np.max(values.real, axis=1)
    gaps = np.array([eig_diagnostics(A)[1] for A in matrices])

    # Refine only around the best grid candidate; this is not proof of a true EP.
    k = int(np.argmin(gaps))
    if 0 < k < len(eta) - 1:
        result = minimize_scalar(gap, bounds=(eta[k - 1], eta[k + 1]),
                                 method='bounded', options={'xatol': 1e-12})
        candidate_eta = float(result.x) if result.success else float(eta[k])
        if gap(candidate_eta) > gaps[k]:
            candidate_eta = float(eta[k])
    else:
        candidate_eta = float(eta[k])
    ep_s, ep_gap, ep_overlap, ep_pair = eig_diagnostics(matrix(candidate_eta))

    # Detect ALL sign changes in the sampled range; stability need not be monotonic.
    boundaries = []
    for a, b, sa, sb in zip(eta[:-1], eta[1:], spectral_values[:-1], spectral_values[1:]):
        if sa == 0:
            root = float(a)
        elif sa * sb < 0:
            root = float(brentq(spectral, float(a), float(b), xtol=1e-13))
        else:
            continue
        if not boundaries or abs(root - boundaries[-1]) > 1e-8:
            boundaries.append(root)

    selected_s, selected_gap, selected_overlap, selected_pair = eig_diagnostics(matrix(args.eta_select))
    scale = 1e-6  # 1/s -> 1/us, rad/s -> Mrad/s
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for j in range(values.shape[1]):
        ax[0, 0].plot(eta, values[:, j].real * scale, lw=1.3)
        ax[0, 1].plot(eta, values[:, j].imag * scale, lw=1.3)
    ax[0, 0].axhline(0, color='black', ls='--', lw=1)
    ax[0, 0].set(title='Eigenvalue real parts', ylabel='Re(lambda) [1/us]')
    ax[0, 1].set(title='Eigenvalue imaginary parts', ylabel='Im(lambda) [Mrad/s]')

    ax[1, 0].plot(eta, spectral_values * scale, lw=1.8, label='Spectral abscissa')
    ax[1, 0].axhline(0, color='black', ls='--', lw=1)
    ax[1, 0].fill_between(eta, spectral_values * scale, 0,
                          where=spectral_values < 0, alpha=0.13, label='Linearly stable')
    ax[1, 0].set(title='Linear stability', ylabel='max Re(lambda) [1/us]')

    ax[1, 1].plot(eta, gaps * scale, lw=1.6, label='Minimum eigenvalue gap')
    ax[1, 1].scatter([candidate_eta], [ep_gap * scale], marker='x', s=100,
                     linewidths=2, color='tab:orange', zorder=5)
    ax[1, 1].set(title='Minimum-gap EP candidate (not proof of EP)',
                 ylabel='min |lambda_i - lambda_j| [Mrad/s]')

    for panel in ax.ravel():
        panel.axvline(candidate_eta, color='tab:orange', ls='--', lw=1.3,
                      label=f'Gap minimum: {candidate_eta:.6f}')
        for index, boundary in enumerate(boundaries):
            panel.axvline(boundary, color='tab:purple', ls='-.', lw=1.3,
                          label=f'Stability boundary: {boundary:.6f}' if index == 0 else None)
        panel.axvline(args.eta_select, color='tab:red', ls=':', lw=1.5,
                      label=f'Selected: {args.eta_select:.6f}')
        panel.set_xlabel('eta = G / G_EP (ideal reference)')
        panel.grid(alpha=0.3)
    ax[1, 0].legend(loc='best', fontsize=8)
    ax[1, 1].legend(loc='best', fontsize=8)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print('\nPT CIRCUIT: LINEARIZED EP AND STABILITY ANALYSIS')
    print('-' * 58)
    print(f'Ideal reference G_EP      : {tank.G_EP:.9g} S')
    print(f'Common damping delta      : {tank.delta:g}')
    print(f'Minimum-gap candidate eta : {candidate_eta:.9f}')
    print(f'Minimum eigenvalue gap    : {ep_gap:.9g} rad/s ({ep_gap * scale:.6g} Mrad/s)')
    print(f'Candidate vector overlap  : {ep_overlap:.9f} (1 = parallel)')
    print(f'Candidate eigenvalue pair : {ep_pair}')
    print('Stability boundaries      : ' +
          (', '.join(f'{b:.9f}' for b in boundaries) if boundaries else 'none found in scan'))
    print(f'Selected eta              : {args.eta_select:.9f}')
    print(f'Selected max Re(lambda)   : {selected_s:.9g} 1/s')
    print(f'Selected linear stability : {"stable" if selected_s < 0 else "not asymptotically stable"}')
    print(f'Selected gap / overlap    : {selected_gap:.9g} rad/s / {selected_overlap:.9f}')
    print(f'Selected pair             : {selected_pair}')
    print(f'Figure                    : {path}')
    print('-' * 58)
    print('NOTE: Gap minimization and vector overlap only screen EP candidates.')
    print('NOTE: This is zero-input linear stability, not nonlinear driven stability.')


if __name__ == '__main__':
    main()
