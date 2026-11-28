# -*- coding: utf-8 -*-
"""只读 CSV 画斑噪曲线，供队友复现用。"""

import csv
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
CSV_PATH = BASE / "degraded" / "degraded_accuracy.csv"
OUT_PNG = BASE / "degraded_curve_from_csv.png"

L_LIST = [0, 1, 2, 4, 8, 16, 32]
SERIES = [
    ("clean", "普通训练", "#4C72B0", "o"),
    ("noise-p050", "噪声增强 p=0.5", "#DD8452", "s"),
    ("noise-p100", "噪声增强 p=1.0", "#55A868", "^"),
]

# 普通训练的无噪点取第 4 章表 3 的口径
S0_OVERRIDE = {"clean": (98.45, 0.59)}

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def load():
    """按组和视数聚合均值、标准差。"""

    data = {prefix: ([], []) for prefix, _, _, _ in SERIES}

    with CSV_PATH.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    for prefix, _, _, _ in SERIES:
        accs, stds = [], []
        for L in L_LIST:
            vals = [float(r["accuracy"]) for r in rows if r["group"] == prefix and int(r["L"]) == L]
            if not vals:
                accs.append(None)
                stds.append(0)
                continue
            accs.append(statistics.mean(vals))
            stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
        data[prefix] = (accs, stds)

    # 覆盖无噪点
    idx0 = L_LIST.index(0)
    for prefix, (mean, std) in S0_OVERRIDE.items():
        data[prefix][0][idx0] = mean
        data[prefix][1][idx0] = std

    return data


def intensity(L):
    """噪声强度 1/sqrt(L)。"""

    return 0.0 if L == 0 else 1.0 / math.sqrt(L)


def main():
    data = load()
    noisy = [32, 16, 8, 4, 2, 1]
    xs = [-0.12] + [intensity(L) for L in noisy]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # 左图：准确率
    for prefix, label, color, marker in SERIES:
        accs, stds = data[prefix]
        ys = [accs[0]] + [accs[L_LIST.index(L)] for L in noisy]
        err = [stds[0]] + [stds[L_LIST.index(L)] for L in noisy]
        ax1.errorbar(xs, ys, yerr=err, label=label, color=color, marker=marker,
                     capsize=3, linewidth=1.8, markersize=6, clip_on=False)

    ax1.axhline(95, color="#999999", linestyle="--", linewidth=1)
    ax1.text(xs[-1], 95.15, "95% 核对线", ha="right", va="bottom",
             fontsize=9, color="#666666")
    ax1.axvline(-0.03, color="#cccccc", linestyle=":", linewidth=1)
    ax1.text(0.55, 93.7, "相干斑强度递增 →", ha="center", fontsize=9, color="#666666")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["无噪"] + [f"L={L}" for L in noisy], fontsize=9)
    ax1.set_xlim(-0.22, 1.06)
    ax1.set_ylim(93.2, 99.9)
    ax1.set_xlabel("功率域标准差 $1/\\sqrt{L}$", fontsize=10)
    ax1.set_ylabel("测试准确率（%）", fontsize=10)
    ax1.set_title("(a) 各档准确率", fontsize=11, loc="left")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend(fontsize=9, loc="lower left")

    # 右图：相对普通训练的增益
    clean = data["clean"][0]
    width = 0.028
    for i, (prefix, label, color, _) in enumerate(SERIES[1:]):
        accs = data[prefix][0]
        gains = [accs[L_LIST.index(L)] - clean[L_LIST.index(L)] for L in noisy]
        offset = -width / 2 if i == 0 else width / 2
        ax2.bar([x + offset for x in xs[1:]], gains, width=width,
                color=color, label=label, edgecolor="none")

    ax2.axhline(0, color="#333333", linewidth=1)
    ax2.set_xticks(xs[1:])
    ax2.set_xticklabels([f"L={L}" for L in noisy], fontsize=9)
    ax2.set_xlim(0.10, 1.06)
    ax2.set_ylabel("相对普通训练的增益（百分点）", fontsize=10)
    ax2.set_title("(b) 增强带来的提升", fontsize=11, loc="left")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.legend(fontsize=9, loc="upper right")

    top = data["noise-p050"][0][L_LIST.index(1)] - clean[L_LIST.index(1)]
    ax2.set_ylim(0, top + 0.55)
    ax2.annotate(f"L=1 增益 +{top:.2f}", xy=(1.0 - width / 2, top),
                 xytext=(0.52, top + 0.12), fontsize=9, color="#DD8452",
                 arrowprops=dict(arrowstyle="->", color="#DD8452", linewidth=1))

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    print("已写入", OUT_PNG)


if __name__ == "__main__":
    main()
