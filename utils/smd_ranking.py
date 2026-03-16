# utils/ranking.py

import os
import numpy as np
import pandas as pd


def rank_machines(
    machine_mean_rows: list,
    save_dir: str,
    top_n: int = 4,
) -> pd.DataFrame:
    """
    모든 metric에 대해 machine ranking을 계산하고 저장.
    
    Args:
        machine_mean_rows: 각 machine의 mean 결과 dict 리스트
        save_dir: 결과 저장 디렉토리
        top_n: 상위 N개 machine 선택
    
    Returns:
        df_ranking: metric별 ranking이 추가된 DataFrame
    """
    if not machine_mean_rows:
        print("No machine results to rank.")
        return None

    df = pd.DataFrame(machine_mean_rows)

    # ranking 대상 metric 컬럼 (Test/ 로 시작하는 숫자형 컬럼)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {'Seed', 'Epoch'}
    metric_cols  = [c for c in numeric_cols
                    if c not in exclude_cols and c.startswith('Test/')]

    if not metric_cols:
        print("No Test/* metrics found for ranking.")
        return df

    # ------------------------------------------------------------------ #
    # 1. metric별 ranking 컬럼 추가
    # ------------------------------------------------------------------ #
    df_ranking = df.copy()
    for metric in metric_cols:
        rank_col = f"Rank/{metric.replace('Test/', '')}"
        df_ranking[rank_col] = df_ranking[metric].rank(ascending=False, method='min').astype(int)

    # ------------------------------------------------------------------ #
    # 2. metric별 정렬된 ranking 테이블 저장
    # ------------------------------------------------------------------ #
    os.makedirs(save_dir, exist_ok=True)
    all_rankings = []

    for metric in metric_cols:
        rank_col    = f"Rank/{metric.replace('Test/', '')}"
        metric_name = metric.replace('Test/', '')

        df_sorted = df_ranking[['Machine', metric, rank_col]].sort_values(
            metric, ascending=False
        ).reset_index(drop=True)
        df_sorted.columns = ['Machine', metric_name, 'Rank']
        df_sorted['RankMetric'] = metric_name

        all_rankings.append(df_sorted)

        # metric별 csv 저장
        metric_csv = os.path.join(save_dir, f"ranking_{metric_name}.csv")
        df_sorted.to_csv(metric_csv, index=False)

        # 터미널 출력
        print(f"\n=== Ranking by {metric_name} ===")
        print(df_sorted.to_string(index=False))

        # 상위 top_n 출력
        print(f"\n  Top {top_n}: {df_sorted['Machine'].head(top_n).tolist()}")
        top_mean = df_sorted.head(top_n)[metric_name].mean()
        print(f"  Top {top_n} mean {metric_name}: {top_mean:.4f}")

    # ------------------------------------------------------------------ #
    # 3. 전체 ranking 통합 csv 저장
    # ------------------------------------------------------------------ #
    df_all_rankings = pd.concat(all_rankings, ignore_index=True)
    all_ranking_csv = os.path.join(save_dir, "all_metric_rankings.csv")
    df_all_rankings.to_csv(all_ranking_csv, index=False)
    print(f"\nAll metric rankings saved: {all_ranking_csv}")

    # ------------------------------------------------------------------ #
    # 4. 종합 ranking: 모든 metric rank의 평균으로 계산
    # ------------------------------------------------------------------ #
    rank_cols = [c for c in df_ranking.columns if c.startswith('Rank/')]
    if rank_cols:
        df_ranking['Rank/Overall'] = df_ranking[rank_cols].mean(axis=1)
        df_overall = df_ranking[['Machine', 'Rank/Overall'] + metric_cols + rank_cols]\
            .sort_values('Rank/Overall', ascending=True)\
            .reset_index(drop=True)

        overall_csv = os.path.join(save_dir, "ranking_overall.csv")
        df_overall.to_csv(overall_csv, index=False)

        print(f"\n=== Overall Ranking (mean of all metric ranks) ===")
        print(df_overall[['Machine', 'Rank/Overall']].to_string(index=False))
        print(f"\n  Top {top_n} overall: {df_overall['Machine'].head(top_n).tolist()}")

    # ------------------------------------------------------------------ #
    # 5. 전체 ranking 포함 DataFrame 저장
    # ------------------------------------------------------------------ #
    full_csv = os.path.join(save_dir, "machine_full_ranking.csv")
    df_ranking.to_csv(full_csv, index=False)
    print(f"\nFull ranking saved: {full_csv}")

    return df_ranking