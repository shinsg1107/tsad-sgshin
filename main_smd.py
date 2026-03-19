import os
import numpy as np
import pandas as pd
from pathlib import Path

from models.build import build_model
from utils.parser import parse_args, load_config
from trainer import build_trainer
from utils.misc import mkdir, set_seeds, set_devices
from utils.smd_ranking import rank_machines


SMD_ALL_MACHINES = [
    'machine-1-1', 'machine-1-2', 'machine-1-3', 'machine-1-4', 'machine-1-5',
    'machine-1-6', 'machine-1-7', 'machine-1-8',
    'machine-2-1', 'machine-2-2', 'machine-2-3', 'machine-2-4', 'machine-2-5',
    'machine-2-6', 'machine-2-7', 'machine-2-8', 'machine-2-9',
    'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4', 'machine-3-5',
    'machine-3-6', 'machine-3-7', 'machine-3-8', 'machine-3-9', 'machine-3-10',
    'machine-3-11',
]


def run_single_seed(cfg, seed):
    """단일 seed에 대해 학습 및 평가"""
    cfg.SEED = seed
    set_seeds(seed)

    model   = build_model(cfg)
    trainer = build_trainer(cfg, model)
    trainer.train()

    return trainer.history


def run_single_machine(args, machine, seeds, run_date, base_result_dir, base_ckpt_dir):
    print(f"\n{'='*60}")
    print(f"  Machine: {machine}")
    print(f"{'='*60}")

    machine_history = []
    last_epoch_rows = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # 매 seed마다 cfg를 새로 로드하되 동일한 run_date 사용
        cfg, _ = load_config(args, date=run_date)
        set_devices(cfg.VISIBLE_DEVICES)

        # machine 설정
        cfg.DATA.SMD_ENTITY = machine

        # seed별 경로 설정
        cfg.RESULT_DIR           = os.path.join(base_result_dir, machine, f"seed_{seed}")
        cfg.TRAIN.CHECKPOINT_DIR = os.path.join(base_ckpt_dir,   machine, f"seed_{seed}")

        mkdir(cfg.RESULT_DIR)
        mkdir(cfg.TRAIN.CHECKPOINT_DIR)

        # machine별 causal graph 경로 설정
        causal_enable = bool(getattr(
            getattr(cfg.ORACLEAD, "CAUSAL_ENCODER", None),
            "ENABLE", False
        ))
        if causal_enable:
            machine_id = machine.replace("machine-", "")  # "1-1"
            graph_path = os.path.join(
                "/home/sgshin/workspace/SGTSAD/data/Causal_graph/SMD",
                f"SMD_{machine_id}_graph.npy"
            )
            if os.path.isfile(graph_path):
                cfg.ORACLEAD.CAUSAL_ENCODER.GRAPH_PATH = graph_path
                print(f"  Causal graph: {graph_path}")
            else:
                print(f"  [Warning] Causal graph not found: {graph_path}")
                print(f"  Disabling causal encoder for {machine}")
                cfg.ORACLEAD.CAUSAL_ENCODER.ENABLE = False

        history = run_single_seed(cfg, seed)

        if history:
            for record in history:
                record['Machine'] = machine
                record['Seed']    = seed
            machine_history.extend(history)

            last = history[-1].copy()
            last['Machine'] = machine
            last['Seed']    = seed
            last_epoch_rows.append(last)

    return machine_history, last_epoch_rows


def summarize_machine(last_epoch_rows, machine, save_dir):
    if not last_epoch_rows:
        return None

    df = pd.DataFrame(last_epoch_rows)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {'Seed', 'Epoch'}
    metric_cols  = [c for c in numeric_cols if c not in exclude_cols]

    mean_row = {col: df[col].mean() for col in metric_cols}
    std_row  = {col: df[col].std()  for col in metric_cols}
    mean_row.update({'Machine': machine, 'Seed': 'mean'})
    std_row.update( {'Machine': machine, 'Seed': 'std'})

    df_summary = pd.concat([
        df,
        pd.DataFrame([mean_row]),
        pd.DataFrame([std_row]),
    ], ignore_index=True)

    os.makedirs(save_dir, exist_ok=True)
    summary_csv = os.path.join(save_dir, 'seed_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"[{machine}] seed summary saved: {summary_csv}")

    # 터미널 출력
    print(f"\n=== [{machine}] Mean ± Std over seeds ===")
    for col in metric_cols:
        if col.startswith("Test/"):
            print(f"  {col}: {mean_row[col]:.4f} ± {std_row[col]:.4f}")

    return mean_row


def main():
    args = parse_args()

    # 첫 번째 load에서 run_date 결정
    cfg, run_date = load_config(args, None)
    set_devices(cfg.VISIBLE_DEVICES)

    base_result_dir = cfg.RESULT_DIR
    base_ckpt_dir   = cfg.TRAIN.CHECKPOINT_DIR
    seeds           = getattr(cfg, 'SEEDS', [0, 1, 2, 3, 4])
    top_n           = getattr(cfg, 'TOP_N', 4)

    mkdir(base_result_dir)
    with open(Path(base_result_dir) / 'config.txt', 'w') as f:
        f.write(cfg.dump())

    # ------------------------------------------------------------------ #
    # 전체 machine 순회
    # ------------------------------------------------------------------ #
    all_history       = []
    machine_mean_rows = []

    for machine in SMD_ALL_MACHINES:
        machine_history, last_epoch_rows = run_single_machine(
            args            = args,
            machine         = machine,
            seeds           = seeds,
            run_date        = run_date,
            base_result_dir = base_result_dir,
            base_ckpt_dir   = base_ckpt_dir,
        )

        all_history.extend(machine_history)

        machine_save_dir = os.path.join(base_result_dir, machine)
        mean_row = summarize_machine(last_epoch_rows, machine, machine_save_dir)
        if mean_row is not None:
            machine_mean_rows.append(mean_row)

    # ------------------------------------------------------------------ #
    # 1. 전체 history CSV
    # ------------------------------------------------------------------ #
    if all_history:
        df_all  = pd.DataFrame(all_history)
        all_csv = os.path.join(base_result_dir, 'SMD_all_history.csv')
        df_all.to_csv(all_csv, index=False)
        print(f"\nAll history saved: {all_csv}")

    # ------------------------------------------------------------------ #
    # 2. machine ranking
    # ------------------------------------------------------------------ #
    if machine_mean_rows:
        ranking_save_dir = os.path.join(base_result_dir, 'rankings')
        df_ranking = rank_machines(
            machine_mean_rows = machine_mean_rows,
            save_dir          = ranking_save_dir,
            top_n             = top_n,
        )

        if df_ranking is not None and 'Rank/Overall' in df_ranking.columns:
            top_machines = df_ranking.sort_values('Rank/Overall').head(top_n)['Machine'].tolist()
            print(f"\n=== Recommended Top {top_n} Machines (Overall) ===")
            for i, m in enumerate(top_machines, 1):
                print(f"  {i}. {m}")


if __name__ == '__main__':
    main()