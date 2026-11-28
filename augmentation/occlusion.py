import numpy as np


def _fill_value(image, channel, fill_mode, rng):
    """按指定方式计算被遮挡区域的填充值。

    image: 双通道数组，形状 (2, H, W)
    channel: 当前通道索引，0 为幅度，1 为相位
    fill_mode: 填充方式，"mean" 通道均值；"zero" 置零；"noise" 同分布噪声
    rng: 已指定种子的 NumPy 随机数生成器
    """
    if fill_mode == "zero":
        return 0.0
    if fill_mode == "noise":
        mean_value = image[channel].mean()
        std_value = image[channel].std()
        return rng.normal(mean_value, std_value)
    # 默认按通道均值填充，避免直接置 0 破坏相位通道分布
    return image[channel].mean()


def _area_to_side(area_ratio, height, width):
    """把面积占比换算成正方形遮挡块的边长像素。

    area_ratio: 遮挡块面积占整幅图像面积的比例，0~1
    正方形边长 = 图像边长 * sqrt(面积占比)，高宽按各自尺寸等比换算。
    """
    side_ratio = np.sqrt(area_ratio)
    block_h = max(4, int(round(height * side_ratio)))
    block_w = max(4, int(round(width * side_ratio)))
    return min(block_h, height), min(block_w, width)


def apply_occlusion(
    image,
    area_ratio=0.10,
    patch_count=1,
    fill_mode="mean",
    prob=1.0,
    seed=None,
    position=None,
):
    """对双通道 SAR 图像施加随机块状遮挡。

    输入：np.ndarray，形状 (2, 88, 88)，dtype float32
    输出：np.ndarray，形状 (2, 88, 88)，dtype float32，数值全部有限

    参数：
        area_ratio: 单个遮挡块面积占整幅图像面积的比例，0~1
            正方形边长 = 图像边长 * sqrt(面积比例)
        patch_count: 每张图遮挡方块的数量
        fill_mode: 填充方式，"mean" / "zero" / "noise"
        prob: 单张图被遮挡的概率，1.0 表示全部遮挡，0.0 表示不遮挡
        seed: 随机种子；传入整数可复现遮挡结果，None 时使用系统随机源
        position: 固定遮挡位置 (y0, x0)，左上角像素坐标；
            传入 None 时位置随机；传入坐标时跳过随机位置采样，用于多固定位置评估

    约定：不交换、不删除幅度/相位通道；不修改标签；先复制输入再处理。
    """
    # 拷贝副本，不能直接修改原数组
    image = image.copy()
    if image.ndim != 3 or image.shape[0] != 2:
        raise ValueError(f"遮挡输入形状必须为 (2, H, W)，实际为 {image.shape}")
    if fill_mode not in ("mean", "zero", "noise"):
        raise ValueError(f"不支持的填充方式：{fill_mode}")
    if not 0.0 <= prob <= 1.0:
        raise ValueError("遮挡概率 prob 必须在 0 到 1 之间")
    if not 0.0 <= area_ratio <= 1.0:
        raise ValueError("面积比例 area_ratio 必须在 0 到 1 之间")
    if not isinstance(patch_count, int) or patch_count < 1:
        raise ValueError("遮挡块数量 patch_count 必须为不小于 1 的整数")

    _, height, width = image.shape

    # 命中判定使用全局随机源（由训练脚本 set_random_seed 统一播种）。
    # 不能在每次调用时用固定种子新建 RNG 再取首抽，否则同种子下每张图的判定都相同，
    # 会出现“目录名是遮挡增强、实际输入全为干净图像”的问题。
    if np.random.random_sample() > prob:
        return image.astype(np.float32, copy=False)

    # 遮挡位置与填充噪声使用独立 RNG；seed 仅用于评估阶段固定复现，不参与命中判定。
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    for _ in range(patch_count):
        # 由固定面积占比换算成正方形遮挡块边长
        block_h, block_w = _area_to_side(area_ratio, height, width)

        if position is None:
            # 随机方块左上角位置
            y0 = rng.integers(0, height - block_h + 1)
            x0 = rng.integers(0, width - block_w + 1)
        else:
            # 使用固定位置，并裁剪到图像范围内
            y0, x0 = position
            y0 = max(0, min(y0, height - block_h))
            x0 = max(0, min(x0, width - block_w))

        y1 = y0 + block_h
        x1 = x0 + block_w

        # 分别对幅度、相位通道填充遮挡区域
        for channel in range(2):
            fill_value = _fill_value(
                image, channel, fill_mode, rng
            )
            image[channel, y0:y1, x0:x1] = fill_value

    # 保证输出为 float32 且不含 NaN / Inf
    image = image.astype(np.float32, copy=False)
    if not np.isfinite(image).all():
        raise ValueError("遮挡增强输出存在非有限数值")
    return image


if __name__ == "__main__":
    # 自检：生成一张模拟双通道图像，检查形状、类型、有限性和可复现性
    mock = np.random.randn(2, 88, 88).astype(np.float32)
    result = apply_occlusion(mock, seed=12321)
    repeat = apply_occlusion(mock, seed=12321)

    print("输入形状：", mock.shape)
    print("输出形状：", result.shape)
    print("输出类型：", result.dtype)
    print("数值全部有限：", np.isfinite(result).all())
    print("相同种子结果一致：", np.allclose(result, repeat))

    assert result.shape == (2, 88, 88)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.allclose(result, repeat)

    # 验证面积比例到边长的换算：10% 面积 → 边长约为原图 sqrt(0.1) ≈ 0.316
    h, w = 88, 88
    bh, bw = _area_to_side(0.10, h, w)
    print("面积占比 10% 对应块尺寸：", bh, "×", bw)
    assert 24 <= bh <= 29 and 24 <= bw <= 29
    print("遮挡增强自检通过！")
