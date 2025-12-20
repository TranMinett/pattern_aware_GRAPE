#!/usr/bin/env python
"""
Run pattern-aware GRAPE experiments.

Examples
--------
python main.py
python main.py --fast
python main.py --datasets annealing hepatitis --seeds 1 2 3
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import get_config, METHODS, DATASETS
from src import (DatasetLoader, run_experiment, prepare_splits,
                 set_seed, drop_allnan_cols)
from src.utils import aggregate_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS)
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    cfg = get_config(fast=args.fast)
    if args.datasets:
        cfg['datasets'] = args.datasets
    if args.methods:
        cfg['methods'] = args.methods
    if args.seeds:
        cfg['seeds'] = args.seeds
    if args.epochs:
        cfg['epochs'] = args.epochs
    cfg['save_dir'] = args.output_dir

    os.makedirs(cfg['save_dir'], exist_ok=True)

    print(f"datasets: {cfg['datasets']}")
    print(f"methods: {len(cfg['methods'])}")
    print(f"seeds: {cfg['seeds']}")
    print(f"device: {cfg['device']}\n")

    all_results, detailed = [], []

    for dname in tqdm(cfg['datasets'], desc='datasets'):
        try:
            X, y = DatasetLoader.load(dname)
            X, _ = drop_allnan_cols(X, dname)
        except Exception as e:
            print(f'failed to load {dname}: {e}')
            continue

        for method in tqdm(cfg['methods'], desc=f'  {dname}', leave=False):
            mresults = []
            for seed in cfg['seeds']:
                set_seed(seed)
                if not args.quiet:
                    print(f'[{dname}] {method} seed={seed}')
                try:
                    splits = prepare_splits(X, y, seed)
                    t0 = time.time()
                    metrics = run_experiment(*splits, method, cfg['device'],
                                             {**cfg, 'seed': seed})
                    metrics.update({'dataset': dname, 'method': method,
                                    'seed': seed, 'time': time.time() - t0})
                    mresults.append(metrics)
                    detailed.append(metrics)
                    if not args.quiet:
                        print(f"  bal_acc: {metrics['balanced_accuracy']:.4f}")
                except Exception as e:
                    print(f'error {dname}/{method}/seed={seed}: {e}')

            if mresults:
                agg = {'dataset': dname, 'method': method, 'n': len(mresults)}
                agg.update(aggregate_seeds(mresults))
                times = [r['time'] for r in mresults]
                agg['time_mean'], agg['time_std'] = np.mean(times), np.std(times)
                all_results.append(agg)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(f"{cfg['save_dir']}/results.csv", index=False)
        pd.DataFrame(detailed).to_csv(f"{cfg['save_dir']}/detailed.csv", index=False)
        print(f"\nresults saved to {cfg['save_dir']}/")

        print('\nsummary:')
        for d in cfg['datasets']:
            sub = df[df['dataset'] == d]
            if len(sub):
                print(f'\n{d}:')
                for _, r in sub.iterrows():
                    if 'balanced_accuracy_mean' in r:
                        print(f"  {r['method']}: "
                              f"{r['balanced_accuracy_mean']:.4f} "
                              f"+/- {r['balanced_accuracy_std']:.4f}")


if __name__ == '__main__':
    main()
