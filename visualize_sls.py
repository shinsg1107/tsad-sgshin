# visualize_sls.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse

def load_causal_graph(path: str) -> np.ndarray:
    """causal graph 로드 (.pt 또는 .npy)"""
    if path.endswith('.pt'):
        cg = torch.load(path, map_location='cpu').float().numpy()
    elif path.endswith('.npy'):
        cg = np.load(path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {path}")
    print(f"Causal graph loaded: {path}, shape={cg.shape}, "
          f"mean={cg.mean():.4f}, max={cg.max():.4f}")
    return cg


def plot_sls_vs_causal(epochs, matrices, causal_graph: np.ndarray,
                       save_path: str = None, dataset: str = ""):
    """
    각 epoch의 SLS와 causal graph 비교:
    1. SLS vs Causal graph 나란히
    2. epoch별 SLS-Causal 차이(diff) 추이
    3. correlation 추이
    """
    # causal graph 정규화 [0,1]
    cg = causal_graph.copy()
    cg_min, cg_max = cg.min(), cg.max()
    if cg_max > cg_min:
        cg_norm = (cg - cg_min) / (cg_max - cg_min)
    else:
        cg_norm = cg.copy()

    # epoch별 통계 계산
    mean_diffs   = []  # SLS와 causal의 평균 절대 차이
    correlations = []  # SLS와 causal의 Pearson correlation
    frob_dists   = []  # Frobenius distance

    for mat in matrices:
        # SLS도 [0,1] 정규화 후 비교
        m_min, m_max = mat.min(), mat.max()
        if m_max > m_min:
            mat_norm = (mat - m_min) / (m_max - m_min)
        else:
            mat_norm = mat.copy()

        diff = np.abs(mat_norm - cg_norm)
        mean_diffs.append(diff.mean())
        frob_dists.append(np.linalg.norm(mat_norm - cg_norm, 'fro'))

        # Pearson correlation (flatten)
        corr = np.corrcoef(mat_norm.flatten(), cg_norm.flatten())[0, 1]
        correlations.append(corr)

    # ------------------------------------------------------------------ #
    # Figure 1: 마지막 epoch SLS vs Causal graph 나란히 비교
    # ------------------------------------------------------------------ #
    last_mat = matrices[-1]
    m_min, m_max = last_mat.min(), last_mat.max()
    last_norm = (last_mat - m_min) / (m_max - m_min) if m_max > m_min else last_mat

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(last_norm, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title(f'SLS (Epoch {epochs[-1]}, normalized)', fontsize=10)
    axes[0].set_xlabel('Variable j'); axes[0].set_ylabel('Variable i')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(cg_norm, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title('Causal Graph (normalized)', fontsize=10)
    axes[1].set_xlabel('Variable j (Effect)')
    axes[1].set_ylabel('Variable i (Cause)')
    plt.colorbar(im1, ax=axes[1])

    diff_last = last_norm - cg_norm
    im2 = axes[2].imshow(diff_last, cmap='RdBu', 
                          vmin=-1, vmax=1, aspect='auto')
    axes[2].set_title(f'SLS - Causal (Epoch {epochs[-1]})\n'
                      f'mean|diff|={mean_diffs[-1]:.4f}, '
                      f'corr={correlations[-1]:.4f}', fontsize=9)
    axes[2].set_xlabel('Variable j'); axes[2].set_ylabel('Variable i')
    plt.colorbar(im2, ax=axes[2], label='SLS - Causal')

    fig.suptitle(f'SLS vs Causal Graph {dataset}', fontsize=12)
    plt.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_comparison.png')
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f"Saved comparison: {p}")
    plt.show()

    # ------------------------------------------------------------------ #
    # Figure 2: epoch별 SLS-Causal 유사도 추이
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, mean_diffs, 'b-o', markersize=4)
    axes[0].set_title('Mean |SLS - Causal| over Epochs')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Mean Abs Diff')
    axes[0].grid(True)

    axes[1].plot(epochs, correlations, 'r-o', markersize=4)
    axes[1].set_title('Pearson Correlation (SLS vs Causal)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Correlation')
    axes[1].set_ylim(-1, 1)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].grid(True)

    axes[2].plot(epochs, frob_dists, 'g-o', markersize=4)
    axes[2].set_title('Frobenius Distance (SLS vs Causal)')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Frobenius Dist')
    axes[2].grid(True)

    fig.suptitle(f'SLS-Causal Similarity over Epochs {dataset}', fontsize=12)
    plt.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_similarity.png')
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f"Saved similarity: {p}")
    plt.show()

    # ------------------------------------------------------------------ #
    # Figure 3: 모든 epoch의 SLS-Causal diff grid
    # ------------------------------------------------------------------ #
    n = len(epochs)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (epoch, mat) in enumerate(zip(epochs, matrices)):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        m_min, m_max = mat.min(), mat.max()
        mat_norm = (mat - m_min) / (m_max - m_min) if m_max > m_min else mat
        diff = mat_norm - cg_norm
        im = ax.imshow(diff, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax.set_title(f'Ep{epoch} corr={correlations[idx]:.2f}', fontsize=8)
        ax.tick_params(labelsize=6)

    for idx in range(len(epochs), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f'SLS - Causal Graph per Epoch {dataset}', fontsize=12)
    plt.colorbar(im, ax=axes, shrink=0.6, label='SLS - Causal')
    plt.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_diff_grid.png')
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f"Saved diff grid: {p}")
    plt.show()

    # 최종 수치 출력
    print(f"\n=== SLS vs Causal Summary ===")
    print(f"  Final epoch {epochs[-1]}:")
    print(f"    Mean |diff|:       {mean_diffs[-1]:.4f}")
    print(f"    Pearson corr:      {correlations[-1]:.4f}")
    print(f"    Frobenius dist:    {frob_dists[-1]:.4f}")
    print(f"  Best corr epoch:     {epochs[np.argmax(correlations)]} "
          f"({max(correlations):.4f})")
    print(f"  Best diff epoch:     {epochs[np.argmin(mean_diffs)]} "
          f"({min(mean_diffs):.4f})")
    
def load_sls_sequence(sls_dir: str):
    """epoch순으로 sls 파일 로드"""
    files = sorted(
        [f for f in os.listdir(sls_dir) if f.startswith('sls_epoch_') and f.endswith('.pt')],
        key=lambda x: int(x.replace('sls_epoch_', '').replace('.pt', ''))
    )

    if not files:
        print(f"No sls_epoch_*.pt files found in {sls_dir}")
        return [], []

    epochs = []
    matrices = []
    for f in files:
        epoch = int(f.replace('sls_epoch_', '').replace('.pt', ''))
        sls = torch.load(os.path.join(sls_dir, f), map_location='cpu').numpy()
        epochs.append(epoch)
        matrices.append(sls)
        print(f"Loaded {f}: shape={sls.shape}, mean={sls.mean():.4f}, max={sls.max():.4f}")

    return epochs, matrices


def plot_sls_grid(epochs, matrices, save_path: str = None, dataset: str = ""):
    """모든 epoch을 grid로 한번에 시각화"""
    n = len(epochs)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)

    # 전체 color scale 통일
    vmin = min(m.min() for m in matrices)
    vmax = max(m.max() for m in matrices)

    for idx, (epoch, mat) in enumerate(zip(epochs, matrices)):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        im = ax.imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'Epoch {epoch}', fontsize=9)
        ax.set_xlabel('Variable j', fontsize=7)
        ax.set_ylabel('Variable i', fontsize=7)
        ax.tick_params(labelsize=6)

    # 빈 subplot 숨기기
    for idx in range(len(epochs), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f'SLS Evolution {dataset}', fontsize=12)
    plt.colorbar(im, ax=axes, shrink=0.6, label='Distance')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved grid: {save_path}")
    plt.show()


def plot_sls_diff(epochs, matrices, save_path: str = None, dataset: str = ""):
    """epoch간 변화량(diff) 시각화"""
    if len(matrices) < 2:
        return

    diffs = [np.abs(matrices[i+1] - matrices[i]) for i in range(len(matrices)-1)]
    diff_epochs = [f"{epochs[i]}→{epochs[i+1]}" for i in range(len(epochs)-1)]

    n = len(diffs)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    vmax = max(d.max() for d in diffs)

    for idx, (label, diff) in enumerate(zip(diff_epochs, diffs)):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        im = ax.imshow(diff, cmap='hot', vmin=0, vmax=vmax, aspect='auto')
        ax.set_title(f'Δ {label}', fontsize=9)
        ax.tick_params(labelsize=6)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f'SLS Change Between Epochs {dataset}', fontsize=12)
    plt.colorbar(im, ax=axes, shrink=0.6, label='|Δ Distance|')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diff: {save_path}")
    plt.show()


def plot_sls_stats(epochs, matrices, save_path: str = None, dataset: str = ""):
    """epoch별 SLS 통계 (mean, std, max) 추이"""
    means = [m.mean() for m in matrices]
    stds  = [m.std()  for m in matrices]
    maxs  = [m.max()  for m in matrices]

    # epoch간 변화량
    diffs = [np.abs(matrices[i+1] - matrices[i]).mean()
             for i in range(len(matrices)-1)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0][0].plot(epochs, means, 'b-o', markersize=4)
    axes[0][0].set_title('SLS Mean over Epochs')
    axes[0][0].set_xlabel('Epoch')
    axes[0][0].set_ylabel('Mean Distance')
    axes[0][0].grid(True)

    axes[0][1].plot(epochs, stds, 'r-o', markersize=4)
    axes[0][1].set_title('SLS Std over Epochs')
    axes[0][1].set_xlabel('Epoch')
    axes[0][1].set_ylabel('Std Distance')
    axes[0][1].grid(True)

    axes[1][0].plot(epochs, maxs, 'g-o', markersize=4)
    axes[1][0].set_title('SLS Max over Epochs')
    axes[1][0].set_xlabel('Epoch')
    axes[1][0].set_ylabel('Max Distance')
    axes[1][0].grid(True)

    axes[1][1].plot(epochs[1:], diffs, 'm-o', markersize=4)
    axes[1][1].set_title('Mean |ΔSLS| Between Epochs')
    axes[1][1].set_xlabel('Epoch')
    axes[1][1].set_ylabel('Mean Change')
    axes[1][1].grid(True)

    fig.suptitle(f'SLS Statistics {dataset}', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved stats: {save_path}")
    plt.show()


def make_animation(epochs, matrices, save_path: str = None, dataset: str = ""):
    """GIF 애니메이션 생성"""
    vmin = min(m.min() for m in matrices)
    vmax = max(m.max() for m in matrices)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrices[0], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label='Distance')
    title = ax.set_title(f'SLS Epoch {epochs[0]}')

    def update(idx):
        im.set_data(matrices[idx])
        title.set_text(f'SLS Epoch {epochs[idx]} | mean={matrices[idx].mean():.3f}')
        return [im, title]

    ani = animation.FuncAnimation(
        fig, update, frames=len(epochs), interval=500, blit=True
    )

    if save_path:
        ani.save(save_path, writer='pillow', fps=2)
        print(f"Saved animation: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sls_dir',      type=str, required=True)
    parser.add_argument('--dataset',      type=str, default='')
    parser.add_argument('--out_dir',      type=str, default=None)
    parser.add_argument('--gif',          action='store_true')
    parser.add_argument('--causal_graph', type=str, default='/home/sgshin/workspace/SGTSAD/data/Causal_graph/PSM_graph.npy',
                        help='Path to causal graph (.pt or .npy)')
    args = parser.parse_args()

    out_dir = args.out_dir or args.sls_dir
    os.makedirs(out_dir, exist_ok=True)

    epochs, matrices = load_sls_sequence(args.sls_dir)
    if not epochs:
        return

    print(f"\nLoaded {len(epochs)} SLS matrices, shape={matrices[0].shape}")

    plot_sls_grid(epochs, matrices,
                  save_path=os.path.join(out_dir, 'sls_grid.png'),
                  dataset=args.dataset)
    plot_sls_diff(epochs, matrices,
                  save_path=os.path.join(out_dir, 'sls_diff.png'),
                  dataset=args.dataset)
    plot_sls_stats(epochs, matrices,
                   save_path=os.path.join(out_dir, 'sls_stats.png'),
                   dataset=args.dataset)

    # causal graph 비교
    if args.causal_graph:
        causal_graph = load_causal_graph(args.causal_graph)
        plot_sls_vs_causal(epochs, matrices, causal_graph,
                           save_path=os.path.join(out_dir, 'sls_vs_causal.png'),
                           dataset=args.dataset)

    if args.gif:
        make_animation(epochs, matrices,
                       save_path=os.path.join(out_dir, 'sls_evolution.gif'),
                       dataset=args.dataset)


if __name__ == '__main__':
    main()