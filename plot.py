import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm

def load_sls(sls_path: str) -> torch.Tensor:
    sls = torch.load(sls_path, map_location="cpu", weights_only=True)
    if isinstance(sls, dict):  # 혹시 dict로 저장된 경우
        sls = sls["sls"]
    return sls.detach().float()

def preprocess_sls(sls: torch.Tensor, sym=True, zero_diag=True, mode="distance"):
    """
    mode:
      - "distance": 원래 SLS(거리)를 그대로 시각화 (값 클수록 밝게)
      - "affinity": exp(-d/tau)로 변환해서 '관계 강함(가까움)'이 밝게
      - "inv": 1/(d+eps)로 변환 (affinity 비슷)
    """
    m = sls.clone()

    if sym:
        m = 0.5 * (m + m.t())

    if zero_diag:
        m.fill_diagonal_(0)

    a = m.numpy()

    if mode == "affinity":
        eps = 1e-6
        pos = a[a > 0]
        tau = np.percentile(pos, 50) if pos.size > 0 else 1.0  # median scale
        a = np.exp(-a / (tau + eps))
        np.fill_diagonal(a, 0)

    elif mode == "inv":
        eps = 1e-6
        a = 1.0 / (a + eps)
        np.fill_diagonal(a, 0)

    return a

def robust_vmin_vmax(a: np.ndarray, low=1, high=99):
    vals = a[np.isfinite(a)]
    return np.percentile(vals, low), np.percentile(vals, high)

def plot_matrix(
    a: np.ndarray,
    out_path="sls_matrix.pdf",
    title="SLS matrix",
    cmap="magma",
    low=1, high=99,
    norm_type="power",   # "power" or "log" or None
    gamma=0.5,           # power norm gamma (<1이면 대비 증가)
    dpi=300,
):
    vmin, vmax = robust_vmin_vmax(a, low=low, high=high)

    if norm_type == "power":
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
        imshow_kwargs = dict(norm=norm)
    elif norm_type == "log":
        eps = 1e-12
        norm = LogNorm(vmin=max(vmin, eps), vmax=max(vmax, eps))
        imshow_kwargs = dict(norm=norm)
    else:
        imshow_kwargs = dict(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(6, 5))
    plt.imshow(a, cmap=cmap, interpolation="nearest", **imshow_kwargs)
    plt.title(title)
    plt.xlabel("Effect (j)")
    plt.ylabel("Cause (i)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

if __name__ == "__main__":
    sls = load_sls("results/SWaT/sls_latest.pt")   # 경로 맞게 수정
    a = preprocess_sls(sls, mode="distance")      # or "affinity"
    plot_matrix(
        a,
        out_path="sls_matrix_1.pdf",
        title="SLS (robust + PowerNorm)",
        cmap="magma",
        low=1, high=99,       # 필요하면 5/95로 더 강하게
        norm_type="power",
        gamma=0.5             # 0.3~0.7 사이로 조절 추천
    )
