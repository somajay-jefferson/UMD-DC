# Run stage4_clusso_fdr.py once per (n, q) grid cell, for figures in the layout
# of the Random CLUSSO paper's Figs 1-5 (see plot_grid.py, GRID_PLAN.md).
#
# Each cell is an ordinary stage 4 run with its own JSON and `meta` block, named
# for exactly what ran, so cells can be filled in over days and in any order:
# a cell whose JSON already exists is skipped. The grid does not have to be
# complete to plot.
#
#     python run_grid.py --n 200 300 400 500 1000 --q 20 50 100 --reps 50 \
#         --arm b --dry-run
#
# Cells run one after another; each uses --jobs workers internally.

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


def cell_name(n, q, rho, a):
    hold = f'_hold{int(round(a.holdout * 100))}' if a.holdout > 0 else ''
    arm = {'a': 'exactk', 'b': 'onelam', 'both': 'botharms'}[a.arm]
    # The design's own data knobs go in the name: sparsity for core's design,
    # rho for Random CLUSSO's (whose real features are fixed at 40%).
    shape = (f'rc_rho{rho:g}' if a.design == 'random_clusso'
             else f'cl_s{a.sparsity:g}')
    return (f'grid_n{n}_q{q}_{shape}_t{a.thresh:g}_{arm}_{a.null_ladder}'
            f'_B{a.B}M{a.M}_r{a.reps}{hold}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, nargs='+', default=[200, 300, 400, 500, 1000])
    ap.add_argument('--q', type=int, nargs='+', default=[20, 50, 100])
    ap.add_argument('--design', choices=('clusso', 'random_clusso'),
                    default='random_clusso',
                    help="data generator; random_clusso reproduces the paper's "
                         'Figs 1-5 design (needed for rho > 0)')
    ap.add_argument('--rho', type=float, nargs='+', default=[0.0],
                    help="feature correlation(s); the paper's are 0 0.3 0.6 0.9")
    ap.add_argument('--sparsity', type=float, default=0.8,
                    help="design 'clusso' only")
    ap.add_argument('--thresh', type=float, default=0.01,
                    help="coefficient cut; 0.01 is the Random CLUSSO paper's")
    ap.add_argument('--reps', type=int, default=50)
    ap.add_argument('--B', type=int, default=50)
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--arm', choices=('a', 'b', 'both'), default='b',
                    help="a = paper's rule (exact-k), b = one-lambda (about 10x "
                         'cheaper), both')
    ap.add_argument('--null-ladder', choices=('fine', 'coarse'), default='fine')
    ap.add_argument('--holdout', type=float, default=0.0,
                    help="0 = in-sample MSE, as the paper's simulation figures "
                         'are; 0.2 = its real-data 80/20 validation')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20260929)
    ap.add_argument('--out', default=os.path.join(_HERE, 'grid'))
    ap.add_argument('--dry-run', action='store_true',
                    help='print the cells and commands without running')
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.design == 'clusso' and any(r != 0 for r in a.rho):
        sys.exit("--rho > 0 needs --design random_clusso")
    cells = [(n, q, r) for r in a.rho for q in a.q for n in a.n]
    todo = []
    for n, q, r in cells:
        name = cell_name(n, q, r, a)
        js = os.path.join(a.out, name + '.json')
        status = 'done' if os.path.exists(js) else 'todo'
        print(f'  [{status}] n={n:<5} q={q:<4} rho={r:<4g} -> {js}')
        if status == 'todo':
            todo.append((n, q, r, name, js))
    print(f'{len(cells)} cells, {len(todo)} to run')

    for n, q, r, name, js in todo:
        cmd = [sys.executable, '-u', os.path.join(_HERE, 'stage4_clusso_fdr.py'),
               '--n', str(n), '--q-features', str(q), '--sparsity', str(a.sparsity),
               '--reps', str(a.reps), '--B', str(a.B), '--M', str(a.M),
               '--arm', a.arm, '--null-ladder', a.null_ladder,
               '--holdout', str(a.holdout), '--jobs', str(a.jobs),
               '--design', a.design, '--rho', str(r), '--thresh', str(a.thresh),
               '--seed', str(a.seed), '--json', js,
               '--run-date', time.strftime('%Y-%m-%d')]
        print(' '.join(cmd) if a.dry_run else f'\n=== {name} ===', flush=True)
        if a.dry_run:
            continue
        log = open(os.path.join(a.out, name + '.log'), 'w')
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        log.close()
        # stage4 exits 1 when its FDR gate fails, which is a result, not an
        # error. Only a missing JSON means the cell did not finish.
        ok = os.path.exists(js)
        print(f'  {"finished" if ok else "FAILED (no JSON)"}, exit {rc}', flush=True)
        if not ok:
            sys.exit(f'stopping: {name} did not write its JSON; see its .log')


if __name__ == '__main__':
    main()
