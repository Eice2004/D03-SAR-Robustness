# -*- coding: utf-8 -*-
"""相干斑噪声增强模块。

功率域乘性模型：强度乘 Gamma 噪声，幅度开方回原量纲。
"""

import numpy as np


# 训练默认档位。视数越小噪声越强。
DEFAULT_L_POOL = (1, 2, 4, 8)

# 输入输出约定
IMAGE_SHAPE = (2, 88, 88)


def _sample_L(L_pool, rng):
    """从档位中随机取一个视数。"""

    if rng is None:
        index = np.random.randint(len(L_pool))
    else:
        index = int(rng.integers(len(L_pool)))

    return float(L_pool[index])


def _check_image(image):
    """检查输入形状、类型和数值范围。"""

    if not isinstance(image, np.ndarray):
        raise TypeError("image 必须是 NumPy 数组")

    if image.shape != IMAGE_SHAPE:
        raise ValueError(f"image 形状必须为 {IMAGE_SHAPE}，实际为 {image.shape}")

    if not np.isfinite(image).all():
        raise ValueError("image 不能包含 NaN 或 Inf")

    if image[0].min() < 0:
        raise ValueError("幅度通道不能为负，乘性噪声要求非负输入")


def apply_speckle_noise(image, L=None, p=1.0, L_pool=DEFAULT_L_POOL, rng=None):
    """给幅度通道加相干斑噪声，返回新的 (2, 88, 88) float32 数组。

    参数
    L       固定视数。None 表示从 L_pool 随机抽。
    p       施加概率。0 表示不加，1 表示每张都加。
    L_pool  候选视数，默认 1、2、4、8。
    rng     np.random.Generator。None 时使用全局 np.random。

    只改第 0 通道幅度，第 1 通道相位保持原值。
    """

    _check_image(image)

    # 复制输入，避免影响原数组
    result = image.copy()

    if p <= 0:
        return result

    if p < 1.0:
        draw = np.random.random() if rng is None else float(rng.random())
        if draw >= p:
            return result

    if L is None:
        L = _sample_L(L_pool, rng)

    if L <= 0:
        raise ValueError("视数 L 必须大于 0")

    # 幅度平方得到强度
    amplitude = result[0].astype(np.float32)
    power = np.square(amplitude)

    # 强度噪声，均值保持 1
    if rng is None:
        noise = np.random.gamma(shape=L, scale=1.0 / L, size=amplitude.shape)
    else:
        noise = rng.gamma(shape=L, scale=1.0 / L, size=amplitude.shape)

    # 强度乘噪声后再开方，回到幅度量纲
    result[0] = np.sqrt(power * noise).astype(np.float32)

    return result
