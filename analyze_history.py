# analyze_history.py
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path


def load_all_histories(results_dir: str, dataset: str = None) -> pd.DataFrame:
    pattern = os.path.join(results_dir, dataset, "**", "all_history.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"No all_history.csv found in {results_dir}/{dataset}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # 날짜 디렉토리를 run_id로
        run_id = str(Path(f).parent.name)  # 2026-03-16-21-37-39
        df['run_id'] = run_id
        dfs.append(df)
        print(f"Loaded: {f} ({len(df)} rows)")

    return pd.concat(dfs, ignore_index=True)


def get_best_epoch_per_seed(df: pd.DataFrame, metric: str = 'Test/VUS-PR') -> pd.DataFrame:
    if metric not in df.columns:
        print(f"Metric '{metric}' not found.")
        print(f"Available Test metrics: {[c for c in df.columns if c.startswith('Test/')]}")
        return pd.DataFrame()

    idx = df.groupby(['run_id', 'Seed'])[metric].idxmax()
    return df.loc[idx].reset_index(drop=True)


def summarize_runs(df_best: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df_best.columns if c.startswith('Test/')]

    rows = []
    for run_id, group in df_best.groupby('run_id'):
        row = {'run_id': run_id, 'n_seeds': len(group)}
        for col in metric_cols:
            row[f'{col}_mean'] = group[col].mean()
            row[f'{col}_std']  = group[col].std()
        rows.append(row)

    return pd.DataFrame(rows)


def rank_runs(df_summary: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    col = f'{primary_metric}_mean'
    if col not in df_summary.columns:
        print(f"{col} not found")
        return df_summary
    return df_summary.sort_values(col, ascending=False).reset_index(drop=True)


def print_summary(df_ranked: pd.DataFrame, df_best: pd.DataFrame, top_n: int = 5):
    metrics = ['Test/VUS-PR', 'Test/VUS-ROC', 'Test/AUC-ROC',
               'Test/AUC-PR', 'Test/Standard-F1']

    print(f"\n{'='*60}")
    print(f"  Top {top_n} runs (by {PRIMARY_METRIC})")
    print(f"{'='*60}")

    for i, row in df_ranked.head(top_n).iterrows():
        run_id = row['run_id']
        print(f"\n[Rank {i+1}] run_id: {run_id} (seeds: {int(row['n_seeds'])})")

        # seed별 best epoch 출력
        seed_rows = df_best[df_best['run_id'] == run_id].sort_values('Seed')
        epochs = seed_rows['Epoch'].tolist()
        seeds  = seed_rows['Seed'].tolist()
        print(f"  Best epochs: {dict(zip(seeds, epochs))}")
        print(f"  Epoch mean: {seed_rows['Epoch'].mean():.1f} ± {seed_rows['Epoch'].std():.1f}")

        for m in metrics:
            mean_col = f'{m}_mean'
            std_col  = f'{m}_std'
            if mean_col in row.index:
                print(f"  {m}: {row[mean_col]:.4f} ± {row.get(std_col, 0):.4f}")


RESULTS_DIR    = 'results/'
DATASET        = 'PSM'
PRIMARY_METRIC = 'Test/VUS-PR'
TOP_N          = 5

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',     default=RESULTS_DIR)
    parser.add_argument('--dataset',         default=DATASET)
    parser.add_argument('--primary_metric',  default=PRIMARY_METRIC)
    parser.add_argument('--top_n',           type=int, default=TOP_N)
    args = parser.parse_args()

    df_all = load_all_histories(args.results_dir, args.dataset)
    if df_all.empty:
        exit()

    print(f"\nTotal rows: {len(df_all)}, runs: {df_all['run_id'].nunique()}")
    print(f"Columns: {df_all.columns.tolist()}")

    df_best    = get_best_epoch_per_seed(df_all, metric=args.primary_metric)
    df_summary = summarize_runs(df_best)
    df_ranked  = rank_runs(df_summary, args.primary_metric)

    print_summary(df_ranked, df_best, top_n=args.top_n)

    out = os.path.join(args.results_dir, args.dataset, 'run_ranking.csv')
    df_ranked.to_csv(out, index=False)
    print(f"\nSaved: {out}")