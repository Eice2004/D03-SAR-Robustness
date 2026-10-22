
from pathlib import Path
import numpy as np


# ========== 1. 数据集配置 ==========

DATA_ROOT = Path(
    r"C:\Users\Administrator\Desktop\MSTAR-dataset-main\soc_complex"
)

CLASS_NAMES = (
    "2S1", "BMP2", "BRDM2", "BTR60", "BTR70",
    "D7", "T62", "T72", "ZIL131", "ZSU234"
)

CLASS_TO_IDX = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}


# ========== 2. 扫描十类数据文件 ==========

def collect_samples(split):
    """收集文件路径和类别标签，不读取图像数据。"""

    if split not in ("train", "test"):
        raise ValueError("split 必须是 train 或 test")

    samples = []

    for class_name in CLASS_NAMES:
        class_dir = DATA_ROOT / split / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(class_dir)

        label = CLASS_TO_IDX[class_name]

        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file():
                samples.append((file_path, label))

    return samples


# ========== 3. 读取单张 MSTAR 原始图像 ==========

def read_mstar(file_path):
    """读取 Phoenix 文件，返回幅度和相位双通道数组。"""

    file_path = Path(file_path)
    header = {}

    with file_path.open("rb") as f:

        # 解析 Phoenix 文件头
        while True:
            raw_line = f.readline()

            if not raw_line:
                raise ValueError("未找到 Phoenix 文件头结束标记")

            line = raw_line.decode("ascii").strip()

            if "EndofPhoenixHeader" in line:
                break

            if "=" in line:
                key, value = line.split("=", 1)
                header[key.strip()] = value.strip()

        header_length = int(header["PhoenixHeaderLength"])
        rows = int(header["NumberOfRows"])
        cols = int(header["NumberOfColumns"])

        # 按文件头规定的偏移位置读取
        f.seek(header_length)
        raw_data = f.read()

    pixel_count = rows * cols

    # 大端 float32：幅度整块 + 相位整块
    values = np.frombuffer(raw_data, dtype=">f4")

    if values.size != pixel_count * 2:
        raise ValueError(f"数据长度错误：{file_path}")

    image = values.reshape(2, rows, cols).astype(np.float32)

    if not np.isfinite(image).all():
        raise ValueError(f"存在非有限数值：{file_path}")

    return image


# ========== 4. 中心裁剪 ==========

def center_crop(image, size):
    """对双通道图像进行中心裁剪。"""

    _, height, width = image.shape

    if height < size or width < size:
        raise ValueError("原始图像尺寸小于裁剪尺寸")

    start_h = (height - size) // 2
    start_w = (width - size) // 2

    return image[
        :,
        start_h:start_h + size,
        start_w:start_w + size
    ]


# ========== 5. 主程序：验证数据集和样本 ==========

if __name__ == "__main__":

    train_samples = collect_samples("train")
    test_samples = collect_samples("test")

    print("训练集数量：", len(train_samples))
    print("测试集数量：", len(test_samples))
    print("类别映射：", CLASS_TO_IDX)

    # 先验证之前已经检查过的样本
    sample_path = DATA_ROOT / "train" / "2S1" / "HB19377.000"

    image = read_mstar(sample_path)
    image_128 = center_crop(image, 128)
    image_88 = center_crop(image_128, 88)

    print("原始图像形状：", image.shape)
    print("128裁剪形状：", image_128.shape)
    print("88裁剪形状：", image_88.shape)
    print("幅度范围：", image[0].min(), image[0].max())
    print("相位范围：", image[1].min(), image[1].max())
