
import torch
from torch.utils.data import DataLoader

# 导入之前写好的数据集类和A-ConvNet模型
from dataset import MSTAR_Dataset
from model import AConvNet


# ========== 1. 创建训练集和测试集 ==========

train_dataset = MSTAR_Dataset("train")
test_dataset = MSTAR_Dataset("test")


# ========== 2. 创建DataLoader ==========

# 每批读取4张图像，这次只做模型测试
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)


# ========== 3. 创建A-ConvNet模型 ==========

model = AConvNet()

# 设置为测试模式，关闭Dropout
model.eval()


# ========== 4. 使用真实MSTAR数据测试模型 ==========

# 测试阶段不需要计算梯度，可以节省内存
with torch.no_grad():

    # 获取训练集的第一个批次
    train_images, train_labels = next(iter(train_loader))

    # 将真实训练图像输入模型
    train_outputs = model(train_images)

    # 获取测试集的第一个批次
    test_images, test_labels = next(iter(test_loader))

    # 将真实测试图像输入模型
    test_outputs = model(test_images)


# ========== 5. 输出测试结果 ==========

print("训练图像形状：", train_images.shape)
print("训练标签形状：", train_labels.shape)
print("训练模型输出：", train_outputs.shape)

print("测试图像形状：", test_images.shape)
print("测试标签形状：", test_labels.shape)
print("测试模型输出：", test_outputs.shape)


# ========== 6. 检查结果 ==========

assert train_outputs.shape == (4, 10)
assert test_outputs.shape == (4, 10)

print("真实MSTAR数据模型测试通过！")
