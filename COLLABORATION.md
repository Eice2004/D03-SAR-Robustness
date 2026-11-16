# 三人协作指南

我们共用一套基线代码。每位成员只开发自己负责的增强模块，不需要复制或重写 `dataset.py`、`model.py`、`train.py`。

## 1. 你负责什么

| 成员 | 工作内容 | 自己的分支 | 自己的模块 |
|---|---|---|---|
| A | 遮挡增强 | `feature-occlusion` | `augmentation/occlusion.py` |
| B | 相干斑噪声增强 | `feature-speckle-noise` | `augmentation/speckle_noise.py` |
| C | 联合增强，调用 A、B 的函数 | `feature-joint-augmentation` | `augmentation/joint_augmentation.py` |

上表中的增强模块是待开发文件。C 先与 A、B 约定参数和调用顺序，等它们合并到 `develop` 后同步使用，不复制两人的实现。

`baseline/`、`README.md`、本指南、`.gitignore` 和共享配置由团队指定的公共代码维护者统一维护。需要改这些文件或公共接口时，先沟通原因和影响。成员提交以自己的增强模块、必要测试和说明为主。

## 2. 增强函数怎么写

统一写成“接收一张图像，返回一张图像”的函数。

| 项目 | 必须满足的要求 |
|---|---|
| 输入、输出 | 都是 NumPy 数组，形状 `(2, 88, 88)`，类型 `float32` |
| 两个通道 | 第 0 通道是幅度，第 1 通道是相位，不交换、不删除 |
| 数值 | 不能含 NaN 或 Inf |
| 标签 | 不修改标签，不改变样本数量 |
| 修改方式 | 先复制输入，再对副本处理，避免影响原数组 |

函数名分别使用 `apply_occlusion`、`apply_speckle_noise`、`apply_joint_augmentation`。参数写清名称、范围和默认值；说明增强作用于哪些通道、遮挡填什么值、噪声采用什么分布。

下面是接口示意，函数体待成员实现，**不能当成已完成的增强算法**：

```python
import numpy as np


def apply_occlusion(image: np.ndarray) -> np.ndarray:
    """接收并返回 (2, 88, 88) 的 float32 数组。"""
    result = image.copy()
    # 在这里实现遮挡算法。
    return result
```

文件名和函数名使用小写加下划线；代码用 UTF-8、四空格缩进。保留公共代码原有注释，不做无关格式调整。导入增强模块时不能自动训练、读写文件或重设随机种子。增加第三方依赖前先与团队沟通。

## 3. 怎么接到基线中

Dataset 已支持 `transform` 参数。它会在**训练随机裁剪之后、转成 Tensor 之前**调用增强，并检查返回值。

```python
# 调用示意：假设 MSTAR_Dataset 和增强函数已正确导入
train_dataset = MSTAR_Dataset("train", transform=apply_occlusion)
test_dataset = MSTAR_Dataset("test")

# 不使用增强时，保留原基线行为
train_dataset = MSTAR_Dataset("train", transform=None)
```

`train.py` 参数区的 `TRAIN_TRANSFORM` 是训练增强配置位置：默认 `None` 表示不使用增强；函数写好并正确导入后，由维护者将它设为函数名（例如 `apply_occlusion`，不加括号）。需要指定增强参数时，可用 `functools.partial` 绑定参数。成员无需修改训练循环或自行维护另一套训练脚本。

**独立验证时的导入示意**：从项目根目录运行，且已经创建相应增强模块后，可以这样导入：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "baseline"))
from dataset import MSTAR_Dataset
from augmentation.occlusion import apply_occlusion
```

联合增强由 C 组合调用。下面只示意“先遮挡、后噪声”，实际顺序和参数须共同确定并写进实验记录：

```python
from augmentation.occlusion import apply_occlusion
from augmentation.speckle_noise import apply_speckle_noise


def apply_joint_augmentation(image):
    image = apply_occlusion(image)
    return apply_speckle_noise(image)
```

训练仍按原流程读取、裁剪和提取 Patch，只在末尾增加可选增强。测试集始终使用固定中心裁剪，**不调用训练增强**。以后开展退化数据上的鲁棒性测试，要另行统一评估方法，不把训练随机增强直接用于测试，也不根据测试集结果反复调整增强参数。

## 4. Git 怎么配合

- `main`：只保存验收通过的版本。
- `develop`：集成三人的工作，进行整体检查。
- 三个 `feature-*` 分支：各自开发，各自推送。

**不要直接推送 `main` 或 `develop`。** 功能分支通过 PR（合并请求）进入 `develop`；整体检查通过后，再由维护者提交 `develop → main` 的 PR。

以下是成员 A 的操作示例，B、C 替换分支名和文件名。执行前先检查 `git status`，妥善保存自己的未提交工作；假设远程名为 `origin`，且维护者已准备好 `develop`。

```powershell
# 第一次开始开发：从最新 develop 创建自己的分支
git status
git fetch origin
git switch develop
git pull --ff-only origin develop
git switch -c feature-occlusion

# 写好并测试后：只提交自己负责的文件
git diff
git add augmentation/occlusion.py
# 测试和说明文件也按具体路径逐个添加
git diff --cached
git commit -m "feat: add occlusion augmentation"
git push -u origin feature-occlusion
```

已有本地分支就直接切换，不重复创建；只有远程分支时，可用 `git switch --track origin/feature-occlusion`。推送后，在 GitHub 创建目标为 `develop` 的 PR，写清改了什么、如何使用、测了什么，由另一位成员审阅，维护者确认后合并。

开发过程中，在自己的分支且工作区干净时同步公共更新：

```powershell
git fetch origin
git merge origin/develop
```

发生冲突时，先与相关文件负责人确认两边的改动，再解决冲突、重新测试并推送。不要整文件覆盖对方内容，不要用强制推送或 `reset --hard` 清除别人的工作。公共代码修复同样走 PR。

## 5. 实验怎么保持一致

- **参数明确**：增强概率、遮挡比例、噪声强度等由调用方传入或集中配置，不在函数内部散落实验常量。
- **固定种子**：沿用公共入口的随机种子设置，增强函数内部不要反复调用 `seed()`。若使用 `np.random.default_rng`，需单独指定并记录种子；它不受 `np.random.seed` 控制。
- **记录随机源**：增强与裁剪共用 NumPy 随机源时，会影响后续裁剪序列。不同策略不能仅因初始种子相同就声称所有裁剪一致。多进程加载的种子设置由维护者统一处理。
- **公平比较**：除增强方法外，保持数据划分、类别映射、模型、训练轮数、优化器、学习率和尾批次处理一致。
- **每次留记录**：写明成员、日期、代码版本、增强参数及顺序、种子、环境版本、训练设置和结果。多种子实验保留每次结果，不只保留最好的一次。

新结果建议采用“模型－策略－配置编号－种子－时间”命名，例如：

```text
history-AConvNet-occlusion-cfg01-seed12321-20261116-153000.json
```

配置编号要对应完整参数，不能只写 `cfg01`。这是命名约定，当前脚本不会自动生成该名称。不要覆盖已有记录；小型历史 JSON 可经审核放入 `experiments/history/`，权重保存在本地 `checkpoints/`。

## 6. 本机数据和环境怎么处理

数据集、权重、虚拟环境、缓存和大型结果文件不提交 Git。`.gitignore` 已忽略 `data/`、`checkpoints/`、`*.pt`、`*.pth` 等，但其他路径仍需自己检查。提交前查看暂存文件，避免使用 `git add .` 把无关内容全部加入。

当前数据路径在 `baseline/read_mstar.py` 的 `DATA_ROOT` 中，还是本机绝对路径。你的存放位置不同时，先与维护者确认本地配置方法，不提交自己的路径覆盖公共文件。独立验证中可以在创建 Dataset 前临时设置路径，示意如下：

```python
# 接前面的导入示例；替换为自己的实际目录
import read_mstar

read_mstar.DATA_ROOT = Path(r"D:\local_data\soc_complex")
```

这只对当前 Python 进程有效，现有训练脚本没有自动读取个人路径配置的功能。团队统一记录 Python、NumPy、PyTorch、CUDA 版本；个人解释器路径、环境目录和账号凭据不进仓库。共享大文件使用团队约定的存储位置，并记录版本。

## 7. 提交前检查

- [ ] 只改了自己负责的模块和必要测试、说明，未擅自修改公共接口。
- [ ] 输入输出符合 `(2, 88, 88)`、NumPy `float32` 约定，没有 NaN/Inf，不修改输入或标签。
- [ ] 检查了参数边界、增强图像和双通道数值，固定种子可以复现。
- [ ] 与 Dataset 联调过少量样本或小批次；测试集不增强，关闭增强仍保持原基线行为。
- [ ] 联合增强说明了顺序和依赖；没有把空函数当作已完成算法。
- [ ] 提交中没有数据、权重、个人路径或无关改动；PR 写清测试结果和未验证事项。

先完成这些小范围检查，再按团队计划安排完整训练。遇到接口、结果或合并问题，带上复现步骤、参数和报错一起沟通。
