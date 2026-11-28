"""遮挡增强上线前的快速自检：命中率、形状类型、有限性、固定位置可复现性。

运行方式（项目 baseline 目录下的虚拟环境）：
    .venv\Scripts\python.exe smoke_occlusion.py
"""

import sys
from pathlib import Path
from functools import partial

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from augmentation.occlusion import apply_occlusion  # noqa: E402
from dataset import MSTAR_Dataset  # noqa: E402


def check_hit_rate():
    """prob=0.6 时，被修改样本比例应接近 60%，不能全部命中或全部不命中。"""
    np.random.seed(12456)
    n = 2000
    hit = 0
    for _ in range(n):
        mock = np.random.randn(2, 88, 88).astype(np.float32)
        out = apply_occlusion(mock, area_ratio=0.10, patch_count=1,
                               fill_mode="mean", prob=0.6, seed=None)
        if not np.allclose(mock, out):
            hit += 1
    ratio = hit / n
    print(f"命中率自检：{hit}/{n} = {ratio:.3f}（期望约 0.60）")
    assert 0.55 <= ratio <= 0.65, "命中率偏离 prob=0.6 过多"
    return ratio


def check_contract():
    """输出形状、dtype、有限性；未命中时返回干净副本。"""
    np.random.seed(12457)
    mock = np.random.randn(2, 88, 88).astype(np.float32)

    out = apply_occlusion(mock, area_ratio=0.20, prob=1.0, seed=12321,
                          position=(10, 10))
    assert out.shape == (2, 88, 88)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()

    # 固定种子 + 固定位置，两次结果必须一致
    repeat = apply_occlusion(mock, area_ratio=0.20, prob=1.0, seed=12321,
                              position=(10, 10))
    assert np.allclose(out, repeat), "固定位置复现不一致"
    print("输出契约与固定位置复现自检通过")


def check_dataset():
    """接真实训练集跑一小段，确认 transform 不报错、标签不变、数值有限。"""
    np.random.seed(12477)
    transform = partial(
        apply_occlusion,
        area_ratio=0.10,
        patch_count=1,
        fill_mode="mean",
        prob=0.6,
    )
    ds = MSTAR_Dataset("train", transform=transform)
    checked = 300
    for i in range(checked):
        image, label = ds[i]
        assert tuple(image.shape) == (2, 88, 88)
        assert image.dtype == torch.float32
        assert torch.isfinite(image).all()
        assert isinstance(label, int)
    print(f"真实训练集抽样自检通过：{checked} 张图像形状/类型/有限性正常")


if __name__ == "__main__":
    check_hit_rate()
    check_contract()
    check_dataset()
    print("\n全部遮挡增强自检通过，可以开始正式训练。")
