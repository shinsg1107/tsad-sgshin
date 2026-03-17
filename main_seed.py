import os
import numpy as np
import pandas as pd
from pathlib import Path

from models.build import build_model
from utils.parser import parse_args, load_config
from trainer import build_trainer
from utils.misc import mkdir, set_seeds, set_devices
from models.oracle.detector import DetectorOracleAD


def run_single_seed(cfg, seed):
    """단일 seed에 대해 학습 및 평가 수행"""
    base_result_dir = cfg.RESULT_DIR
    base_ckpt_dir   = cfg.TRAIN.CHECKPOINT_DIR

    # seed별 저장 경로 분리
    cfg.SEED = seed
    cfg.RESULT_DIR          = os.path.join(base_result_dir, f"seed_{seed}")
    cfg.TRAIN.CHECKPOINT_DIR = os.path.join(base_ckpt_dir,  f"seed_{seed}")

    mkdir(cfg.RESULT_DIR)
    mkdir(cfg.TRAIN.CHECKPOINT_DIR)

    set_seeds(seed)

    model   = build_model(cfg)
    trainer = build_trainer(cfg, model)
    trainer.train()

    return trainer.history  # list of dicts


def main():
    args = parse_args()
    cfg, date = load_config(args, None)

    set_devices(cfg.VISIBLE_DEVICES)

    base_result_dir = cfg.RESULT_DIR
    with open(mkdir(base_result_dir) / 'config.txt', 'w') as f:
        f.write(cfg.dump())

    seeds = getattr(cfg, 'SEEDS', [0, 1, 2, 3, 4])

    all_history   = []   # 모든 seed의 전체 history
    last_epoch_rows = [] # 각 seed의 마지막 epoch 결과만

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"  Seed {seed}")
        print(f"{'='*50}")

        # cfg는 매 seed마다 원본에서 다시 복사
        cfg_seed, _ = load_config(args, date=date)
        set_devices(cfg_seed.VISIBLE_DEVICES)

        history = run_single_seed(cfg_seed, seed)

        if history:
            all_history.extend(history)

            # 마지막 epoch 결과 추출
            last = history[-1].copy()
            last['Seed'] = seed
            last_epoch_rows.append(last)

    # ------------------------------------------------------------------ #
    # 1. 전체 history CSV
    # ------------------------------------------------------------------ #
    if all_history:
        df_all = pd.DataFrame(all_history)
        all_csv = os.path.join(base_result_dir, 'all_history.csv')
        df_all.to_csv(all_csv, index=False)
        print(f"\nAll history saved: {all_csv}")

    # ------------------------------------------------------------------ #
    # 2. seed별 마지막 epoch 결과 + 평균/std CSV
    # ------------------------------------------------------------------ #
    if last_epoch_rows:
        df_last = pd.DataFrame(last_epoch_rows)

        # 숫자형 컬럼만 평균/std 계산
        numeric_cols = df_last.select_dtypes(include=[np.number]).columns.tolist()
        # Seed, Epoch 등 집계 의미 없는 컬럼 제외
        exclude_cols = {'Seed', 'Epoch'}
        metric_cols  = [c for c in numeric_cols if c not in exclude_cols]

        mean_row = {col: df_last[col].mean() for col in metric_cols}
        std_row  = {col: df_last[col].std()  for col in metric_cols}

        mean_row['Seed'] = 'mean'
        std_row['Seed']  = 'std'

        df_summary = pd.concat([
            df_last,
            pd.DataFrame([mean_row]),
            pd.DataFrame([std_row]),
        ], ignore_index=True)

        summary_csv = os.path.join(base_result_dir, 'seed_summary.csv')
        df_summary.to_csv(summary_csv, index=False)
        print(f"Seed summary saved: {summary_csv}")

        # 터미널 출력
        print("\n=== Mean ± Std over seeds ===")
        for col in metric_cols:
            if col.startswith("Test/"):
                print(f"  {col}: {mean_row[col]:.4f} ± {std_row[col]:.4f}")


if __name__ == '__main__':
    main()
