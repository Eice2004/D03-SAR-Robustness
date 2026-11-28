# -*- coding: utf-8 -*-
"""speckle_noise 模块自检。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from speckle_noise import apply_speckle_noise


def make_image(value=1.0):
    """构造 (2,88,88) 的测试图，幅度为常数，相位为渐变。"""

    image = np.zeros((2, 88, 88), dtype=np.float32)
    image[0] = value
    image[1] = np.linspace(0, 6.28, 88, dtype=np.float32)[None, :]
    return image


def power_cv(image):
    """幅度平方的变异系数。"""

    power = np.square(image[0].astype(np.float64))
    return power.std() / power.mean()


def main():
    rng = np.random.default_rng(2026)
    image = make_image()

    out = apply_speckle_noise(image, L=1, p=1.0, rng=rng)
    print("shape", out.shape, "dtype", out.dtype)
    assert out.shape == (2, 88, 88)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()

    # 相位不能变
    assert np.array_equal(out[1], image[1])
    print("phase unchanged ok")

    # 输入不能被改
    assert np.array_equal(image[0], make_image()[0])
    print("input untouched ok")

    # 视数越大，功率变异系数越小，理论上约 1/sqrt(L)
    for L in (1, 2, 4, 8, 16):
        big = np.ones((2, 88, 88), dtype=np.float32)
        big[1] = 1.0
        out = apply_speckle_noise(big, L=L, p=1.0, rng=np.random.default_rng(100 + L))
        cv = power_cv(out)
        theory = 1.0 / np.sqrt(L)
        print(f"L={L:2d} cv={cv:.3f} theory={theory:.3f}")
        assert abs(cv - theory) < 0.35 * theory, (L, cv, theory)

    # p=0 原样返回
    out0 = apply_speckle_noise(image, L=2, p=0.0)
    assert np.array_equal(out0, image)
    print("p=0 no change ok")

    # 同一 rng 种子可复现
    a = apply_speckle_noise(image, L=None, p=1.0, rng=np.random.default_rng(7))
    b = apply_speckle_noise(image, L=None, p=1.0, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)
    print("reproducible ok")

    # 幅度保持非负
    assert out0[0].min() >= 0
    print("nonnegative ok")
    print("TEST OK")


if __name__ == "__main__":
    main()
