
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import random
import json

from pathlib import Path

# 导入已经完成的数据集类和模型
from dataset import MSTAR_Dataset
from model import AConvNet


# ========== 1. 训练参数配置 ==========

# 随机种子：与原作者SOC实验保持一致
RANDOM_SEED = 12321

# 每批读取的图像数量
BATCH_SIZE = 100

# 目前只训练1轮，用于检查完整训练流程
# 正式基线训练时改为100
EPOCHS = 100

# 优化器参数
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.004

# 学习率调整参数
# 第50轮结束后，将学习率降低为原来的0.1倍
LR_MILESTONES = [50]
LR_DECAY = 0.1

# 实验名称
MODEL_NAME = "AConvNet-SOC"


# ========== 2. 设置随机种子 ==========

def set_random_seed(seed):
    """固定Python、NumPy和PyTorch的随机种子。"""

    # Python标准库随机数
    random.seed(seed)

    # NumPy随机数
    np.random.seed(seed)

    # PyTorch随机数
    torch.manual_seed(seed)

    # 固定GPU随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 设置CuDNN，尽量保证计算结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 在创建数据集和模型之前设置随机种子
set_random_seed(RANDOM_SEED)


# ========== 3. 设置训练设备 ==========

# 如果有NVIDIA GPU，就使用GPU；否则使用CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前训练设备：", device)
print("随机种子：", RANDOM_SEED)


# ========== 4. 设置模型和实验结果保存路径 ==========

# 获取项目根目录，也就是D:\D03Test
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 模型权重保存目录
# 不同随机种子使用不同目录，避免实验结果相互覆盖
CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / f"{MODEL_NAME}-seed{RANDOM_SEED}"
)

# 训练记录保存目录
HISTORY_DIR = PROJECT_ROOT / "experiments" / "history"

# 创建保存目录，如果目录已经存在则不重复创建
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# 训练记录文件
HISTORY_PATH = (
    HISTORY_DIR
    / f"history-{MODEL_NAME}-seed{RANDOM_SEED}.json"
)


# ========== 5. 加载训练数据和测试数据 ==========

# 创建MSTAR训练集
train_dataset = MSTAR_Dataset("train")

# 创建训练集DataLoader
# 每批读取100张图像，每轮打乱样本顺序
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# 创建MSTAR测试集
test_dataset = MSTAR_Dataset("test")

# 创建测试集DataLoader
# 测试时不打乱样本顺序
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("训练集样本数量：", len(train_dataset))
print("测试集样本数量：", len(test_dataset))
print("每轮训练批次数量：", len(train_loader))


# ========== 6. 创建A-ConvNet模型 ==========

# 创建模型，并将模型放到指定设备
model = AConvNet().to(device)


# ========== 7. 定义损失函数 ==========

# 交叉熵损失：衡量预测结果与真实类别之间的差距
criterion = nn.CrossEntropyLoss()


# ========== 8. 定义优化器 ==========

# 使用随机梯度下降优化器
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)


# ========== 9. 定义学习率调整器 ==========

# 使用多阶段学习率调整
# 到达第50轮后，学习率乘以0.1
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=LR_MILESTONES,
    gamma=LR_DECAY
)


# ========== 10. 定义测试集准确率计算函数 ==========

# 测试阶段不需要计算梯度，可以节省显存
@torch.no_grad()
def evaluate(model, test_loader, device):

    # 切换到测试模式，关闭Dropout
    model.eval()

    # 记录预测正确的图像数量
    correct = 0

    # 记录测试图像总数量
    total = 0

    # 遍历整个测试集
    for images, labels in test_loader:

        # 将图像和标签移动到GPU或CPU
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播，得到每张图像的10个类别分数
        outputs = model(images)

        # 找到每张图像分数最高的类别
        predictions = torch.argmax(outputs, dim=1)

        # 统计预测正确的图像数量
        correct += (predictions == labels).sum().item()

        # 累计测试图像数量
        total += labels.size(0)

    # 计算分类准确率
    accuracy = 100 * correct / total

    return accuracy


# ========== 11. 初始化训练记录 ==========

# 使用字典保存实验参数和每轮训练结果
history = {

    # 实验基本参数
    "random_seed": RANDOM_SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "initial_learning_rate": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    "lr_milestones": LR_MILESTONES,
    "lr_decay": LR_DECAY,

    # 每轮训练结果
    "loss": [],
    "accuracy": [],
    "learning_rate": []
}


# ========== 12. 完整训练过程 ==========

# 记录训练前第一层卷积权重
# 后面用于检查模型参数是否发生变化
weight_before = model.conv1[0].weight.detach().clone()

# 记录最后保存的模型文件路径
last_checkpoint_path = None

# 开始训练
for epoch in range(1, EPOCHS + 1):

    # 设置为训练模式，启用Dropout
    model.train()

    # 记录本轮训练开始时的学习率
    current_lr = optimizer.param_groups[0]["lr"]

    # 用列表记录每个批次的损失
    batch_losses = []

    print(f"\n========== 第 {epoch}/{EPOCHS} 轮训练 ==========")
    print(f"当前学习率：{current_lr:.6f}")

    # 遍历整个训练集
    for batch_index, (images, labels) in enumerate(
        train_loader, start=1
    ):

        # 将图像和标签移动到GPU或CPU
        images = images.to(device)
        labels = labels.to(device)

        # ---------- 1. 前向传播 ----------

        # 将图像输入A-ConvNet，得到预测分数
        outputs = model(images)

        # ---------- 2. 计算损失 ----------

        # 比较预测结果与真实标签
        loss = criterion(outputs, labels)

        # ---------- 3. 清除旧梯度 ----------

        optimizer.zero_grad()

        # ---------- 4. 反向传播 ----------

        # 根据损失计算模型参数的梯度
        loss.backward()

        # ---------- 5. 更新模型权重 ----------

        # 优化器根据梯度调整模型参数
        optimizer.step()

        # 记录当前批次的损失
        batch_losses.append(loss.item())

        # 每完成100个批次输出一次训练进度
        if batch_index % 100 == 0 or batch_index == len(train_loader):

            print(
                f"第 {epoch} 轮，"
                f"批次：{batch_index}/{len(train_loader)}，"
                f"当前损失：{loss.item():.4f}"
            )

    # ---------- 6. 计算本轮平均损失 ----------

    average_loss = sum(batch_losses) / len(batch_losses)

    print(f"\n第 {epoch} 轮训练完成！")
    print(f"平均损失：{average_loss:.4f}")

    # ---------- 7. 在测试集上评估模型 ----------

    # 使用完整测试集计算分类准确率
    test_accuracy = evaluate(
        model,
        test_loader,
        device
    )

    print(f"测试集准确率：{test_accuracy:.2f}%")

    # ---------- 8. 保存本轮模型权重 ----------

    # 每轮训练完成后保存一次模型
    # 例如：model-001.pth、model-002.pth
    checkpoint_path = (
        CHECKPOINT_DIR
        / f"model-{epoch:03d}.pth"
    )

    # 只保存模型参数，与原作者的保存方式保持一致
    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    last_checkpoint_path = checkpoint_path

    print("模型已保存：", checkpoint_path)

    # ---------- 9. 保存本轮训练记录 ----------

    # 将本轮结果添加到历史记录
    history["loss"].append(average_loss)
    history["accuracy"].append(test_accuracy)
    history["learning_rate"].append(current_lr)

    # 每轮结束后立即更新JSON文件
    # 即使训练中途停止，也能保留已经完成的轮次记录
    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("训练记录已保存：", HISTORY_PATH)

    # ---------- 10. 调整下一轮学习率 ----------

    # 根据MultiStepLR设置调整学习率
    # 第50轮结束后，下一轮使用0.0001
    scheduler.step()

    next_lr = optimizer.param_groups[0]["lr"]

    print(f"下一轮学习率：{next_lr:.6f}")
    print("-" * 60)


# ========== 13. 检查模型训练结果 ==========

# 获取训练后的第一层卷积权重
weight_after = model.conv1[0].weight.detach()

# 检查权重是否发生变化
weight_changed = not torch.equal(
    weight_before,
    weight_after
)

print("\n模型权重是否更新：", weight_changed)

# 确认模型参数确实发生了更新
assert weight_changed


# ========== 14. 检查模型保存与加载 ==========

# 从刚刚保存的模型文件中读取权重
saved_weights = torch.load(
    last_checkpoint_path,
    map_location=device
)

# 将保存的权重重新加载到模型
model.load_state_dict(saved_weights)

print("模型权重保存与加载检查通过！")


# ========== 15. 最终结果 ==========

print("\n========== 训练流程检查完成 ==========")

print("训练轮数：", EPOCHS)
print("模型保存目录：", CHECKPOINT_DIR)
print("训练记录文件：", HISTORY_PATH)

print("完整训练、测试、保存与加载检查通过！")
