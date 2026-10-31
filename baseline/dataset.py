import time  # Python标准库，用于测量数据加载耗时
import torch
from torch.utils.data import Dataset, DataLoader

# 复用已经完成的 MSTAR 数据读取与裁剪函数
from read_mstar import (
    collect_samples,
    read_mstar,
    center_crop,
    random_crop
)


# ========== 1. 定义 MSTAR 数据集类 ==========

class MSTAR_Dataset(Dataset):

    def __init__(self, split="train"):
        """初始化数据集，收集原始文件路径和类别标签。"""

        # 只允许使用训练集或测试集
        if split not in ("train", "test"):
            raise ValueError("split 必须是 train 或 test")

        # 保存数据集类型，供其他方法判断训练或测试模式
        self.split = split

        # 收集原始文件路径和标签
        # 每个元素为 (file_path, label)
        # 此时仅收集路径，不读取实际图像
        self.samples = collect_samples(split)

    def __len__(self):
        """返回数据集能够提供的样本数量。"""

        if self.split == "train":
            # 每张原始训练图像对应49个Patch
            # 2747 × 49 = 134603
            return len(self.samples) * 49

        else:
            # 测试集不扩展Patch，每张原始图像对应一个样本
            return len(self.samples)

    def __getitem__(self, index):
        """根据索引读取并处理一张SAR图像，返回张量和标签。"""

        # ---------- 1. 确定原始图像及Patch编号 ----------

        if self.split == "train":

            # 整除49：定位第几张原始图像
            image_index = index // 49

            # 对49取余：定位该图像中的第几个Patch
            patch_index = index % 49

        else:
            # 测试集每张图像只对应一个样本
            image_index = index

        # 从样本列表中获取文件路径和类别标签
        file_path, label = self.samples[image_index]

        # ---------- 2. 读取原始SAR图像 ----------

        # 读取幅度、相位双通道数据，形状为 (2, H, W)
        image = read_mstar(file_path)

        # 先中心裁剪成128×128，与基线预处理保持一致
        image = center_crop(image, 128)

        # ---------- 3. 训练集和测试集分别处理 ----------

        if self.split == "train":

            # 训练集：128×128 → 中心100×100
            chip = center_crop(image, 100)

            # 49个Patch排列成7×7网格
            # 通过Patch编号计算滑动窗口的行、列起点
            y = patch_index // 7
            x = patch_index % 7

            # 按需提取一个94×94 Patch
            # 不必每次生成全部49个Patch，节省计算和内存
            patch = chip[:, y:y + 94, x:x + 94]

            # 训练时随机裁剪成88×88
            image = random_crop(patch, 88)

        else:

            # 测试集采用固定中心裁剪
            # 不进行随机增强，确保评估结果可复现
            image = center_crop(image, 88)

        # ---------- 4. 转换为PyTorch张量 ----------

        # copy()创建独立、连续的NumPy数组
        # torch.from_numpy()将其转换成PyTorch张量
        image = torch.from_numpy(image.copy())

        # 返回一张图像及其整数类别标签
        # 图像形状：(2, 88, 88)，数据类型：float32
        return image, label


# ========== 2. 主程序：验证Dataset图像读取 ==========

# 直接运行dataset.py时执行，作为模块导入时不执行
if __name__ == "__main__":

    # 创建训练数据集和测试数据集对象
    train_dataset = MSTAR_Dataset("train")
    test_dataset = MSTAR_Dataset("test")

    # ---------- 1. 检查数据集长度 ----------

    print("训练集长度：", len(train_dataset))
    print("测试集长度：", len(test_dataset))

    # ---------- 2. 检查训练样本 ----------

    # 索引50对应第二张原始图像的第二个Patch
    train_image, train_label = train_dataset[50]

    print("训练图像形状：", train_image.shape)
    print("训练图像类型：", type(train_image))
    print("训练图像数据类型：", train_image.dtype)
    print("训练标签：", train_label)

    # ---------- 3. 检查测试样本 ----------

    # 测试集不进行Patch扩展和随机裁剪
    test_image, test_label = test_dataset[0]

    print("测试图像形状：", test_image.shape)
    print("测试图像类型：", type(test_image))
    print("测试标签：", test_label)

    # ---------- 4. 断言检查 ----------

    # 训练集和测试集均应输出双通道88×88图像
    assert train_image.shape == (2, 88, 88)
    assert test_image.shape == (2, 88, 88)

    # 检查图像数值是否全部有限（没有NaN或Inf）
    assert torch.isfinite(train_image).all()
    assert torch.isfinite(test_image).all()

    print("Dataset图像读取检查通过！")

    # ----------  5. 测试DataLoader批量加载 ----------
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=100,
        shuffle=True,
        num_workers=0
    )

    # 只读取一个批次，暂时不遍历整个训练集
    images, labels = next(iter(train_loader))

    print("批量图像形状：", images.shape)
    print("批量标签形状：", labels.shape)
    print("批量图像数据类型：", images.dtype)
    print("批量标签数据类型：", labels.dtype)

    # 验证是否满足A-ConvNet的输入要求
    assert images.shape == (100, 2, 88, 88)
    assert labels.shape == (100,)

    print("DataLoader批量加载检查通过！")
    # ---------- 6. 测试在线数据加载速度 ----------

    # 创建迭代器，连续读取10个批次
    loader_iter = iter(train_loader)

    start_time = time.perf_counter()

    for i in range(10):
        images, labels = next(loader_iter)

    end_time = time.perf_counter()

    elapsed = end_time - start_time

    print("读取批次数量：", 10)
    print("读取图像数量：", 10 * 100)
    print("总耗时（秒）：", round(elapsed, 3))
    print("平均每批耗时（秒）：", round(elapsed / 10, 3))