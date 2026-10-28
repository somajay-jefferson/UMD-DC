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


def cell_name(n, q, a):
    hold = f'_hold{int(round(a.holdout * 100))}' if a.holdout > 0 else ''
    arm = {'a': 'exactk', 'b': 'onelam', 'both': 'botharms'}[a.arm]
    return (f'grid_n{n}_q{q}_rho0_s{a.sparsity:g}_{arm}_{a.null_ladder}'
            f'_B{a.B}M{a.M}_r{a.reps}{hold}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, nargs='+', default=[200, 300, 400, 500, 1000])
    ap.add_argument('--q', type=int, nargs='+', default=[20, 50, 100])
    ap.add_argument('--sparsity', type=float, default=0.8)
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
    cells = [(n, q) for q in a.q for n in a.n]
    todo = []
    for n, q in cells:
        name = cell_name(n, q, a)
        js = os.path.join(a.out, name + '.json')
        status = 'done' if os.path.exists(js) else 'todo'
        print(f'  [{status}] n={n:<5} q={q:<4} -> {js}')
        if status == 'todo':
            todo.append((n, q, name, js))
    print(f'{len(cells)} cells, {len(todo)} to run')

    for n, q, name, js in todo:
        cmd = [sys.executable, '-u', os.path.join(_HERE, 'stage4_clusso_fdr.py'),
               '--n', str(n), '--q-features', str(q), '--sparsity', str(a.sparsity),
               '--reps', str(a.reps), '--B', str(a.B), '--M', str(a.M),
               '--arm', a.arm, '--null-ladder', a.null_ladder,
               '--holdout', str(a.holdout), '--jobs', str(a.jobs),
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
