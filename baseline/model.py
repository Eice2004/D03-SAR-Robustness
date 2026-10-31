
import torch
import torch.nn as nn


# ========== 1. 定义 A-ConvNet 模型 ==========

class AConvNet(nn.Module):

    def __init__(self):
        """初始化A-ConvNet网络结构。"""

        # 初始化父类nn.Module
        super().__init__()

        # ========== 第一层卷积模块 ==========
        # 按顺序执行：卷积 → ReLU激活 → 最大池化

        self.conv1 = nn.Sequential(

            # 1. 卷积操作
            # 输入2个通道（幅度、相位）
            # 输出16个特征通道
            # 使用5×5卷积核，步长为1，不进行边缘填充
            nn.Conv2d(
                in_channels=2,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0
            ),

            # 2. ReLU激活函数
            # 将负数变成0，正数保持不变
            nn.ReLU(inplace=True),

            # 3. 最大池化
            # 使用2×2窗口，每次移动2个像素
            # 将特征图的长和宽缩小一半
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )

            
        )


        # ========== 第二层卷积模块 ==========
        # 按顺序执行：卷积 → ReLU激活 → 最大池化

        self.conv2 = nn.Sequential(

            # 1. 卷积操作
            # 输入16个通道（来自第一层）
            # 输出32个特征通道
            # 使用5×5卷积核
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=0
            ),

            # 2. ReLU激活函数
            nn.ReLU(inplace=True),

            # 3. 最大池化
            # 使用2×2窗口，步长为2
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

    
        # ========== 第三层卷积模块 ==========
        # 按顺序执行：卷积 → ReLU激活 → 最大池化

        self.conv3 = nn.Sequential(

            # 1. 卷积操作
            # 输入32个通道（来自第二层）
            # 输出64个特征通道
            # 使用6×6卷积核
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=6,
                stride=1,
                padding=0
            ),

            # 2. ReLU激活函数
            nn.ReLU(inplace=True),

            # 3. 最大池化
            # 使用2×2窗口，步长为2
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )


        # ========== 第四层卷积模块 ==========
        # 按顺序执行：卷积 → ReLU激活
        # 注意：这一层没有最大池化

        self.conv4 = nn.Sequential(

            # 1. 卷积操作
            # 输入64个通道（来自第三层）
            # 输出128个特征通道
            # 使用5×5卷积核
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=0
            ),

            # 2. ReLU激活函数
            nn.ReLU(inplace=True)
        )


        # ========== Dropout：随机失活 ==========
        # 放在第四层和第五层之间
        # 训练时随机将50%的元素置零，减少过拟合
        self.dropout = nn.Dropout(p=0.5)

        # ========== 第五层卷积模块 ==========
        # 最后一层卷积，将特征转换为10个类别的分数

        self.conv5 = nn.Conv2d(
            in_channels=128,
            out_channels=10,
            kernel_size=3,
            stride=1,
            padding=0
        )

        # 将卷积输出展平，方便后续计算分类损失
        self.flatten = nn.Flatten()

        # ========== 6. 卷积层权重初始化 ==========

        # 遍历当前模型中的所有子模块
        for layer in self.modules():

            # 判断当前模块是否为二维卷积层
            if isinstance(layer, nn.Conv2d):

                # 使用Kaiming正态分布初始化卷积权重
                nn.init.kaiming_normal_(
                    layer.weight,
                    nonlinearity="relu"
                )

                # 前四层卷积的偏置初始化为0.1
                # 第五层卷积的偏置初始化为0
                if layer is self.conv5:
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.constant_(layer.bias, 0.1)

    
    # ========== 2. 定义前向传播过程 ==========

    def forward(self, x):
        """规定输入图像在A-ConvNet中的处理顺序。"""

        # 第一层：卷积 → ReLU → 最大池化
        x = self.conv1(x)

        # 第二层：卷积 → ReLU → 最大池化
        x = self.conv2(x)

        # 第三层：卷积 → ReLU → 最大池化
        x = self.conv3(x)

        # 第四层：卷积 → ReLU
        x = self.conv4(x)

        # Dropout：训练时随机失活，减少过拟合
        x = self.dropout(x)

        # 第五层：卷积，输出10个类别的分数
        x = self.conv5(x)

        # 展平：将每张图像的10个分数整理成一维
        x = self.flatten(x)

        # 返回最终分类分数
        return x


# ========== 3. 测试A-ConvNet网络 ==========

if __name__ == "__main__":

    # 1. 创建A-ConvNet模型
    model = AConvNet()

    # 设置为测试模式
    # 此时Dropout不再随机丢弃数据
    model.eval()

    # 2. 随机生成4张模拟SAR图像
    # 每张图像有2个通道，尺寸为88×88
    test_images = torch.randn(4, 2, 88, 88)

    # 3. 将图像输入模型，进行前向传播
    outputs = model(test_images)

    # 4. 输出输入和输出的形状
    print("输入图像形状：", test_images.shape)
    print("模型输出形状：", outputs.shape)

    # 5. 检查输出形状是否正确
    assert outputs.shape == (4, 10)

    print("A-ConvNet前向传播检查通过！")
