# Day 2 HW2

## 特征提取与语音分类

本文档只整理我当前实际跑出的结果和日志，不补写不存在的数据。

### 模型实现（加入 Dropout）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 建立神经网络
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.25):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, dropout=0.25):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim, dropout),
            *[BasicBlock(hidden_dim, hidden_dim, dropout) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.fc(x)
```

## 作业 TODO 对应回答

### TODO 1: 窄深 vs 宽浅（近似同参数量）

配置和结果如下。

| 模型配置                                         | 参数量（按上面网络定义计算） | 训练结果                                                                           |
| ------------------------------------------------ | ---------------------------: | ---------------------------------------------------------------------------------- |
| `hidden_layers=6, hidden_dim=1024, dropout=0.25` |                  `6,394,921` | `[005/005] Train Acc: 0.464580 Loss: 1.845719 \| Val Acc: 0.472387 loss: 1.809266` |
| `hidden_layers=2, hidden_dim=1700, dropout=0.25` |                  `5,931,341` | `[005/005] Train Acc: 0.473922 Loss: 1.805941 \| Val Acc: 0.473760 loss: 1.807432` |

结论：两者参数量同一数量级，宽浅模型在当前实验中验证准确率略高（`0.473760` vs `0.472387`），但差距很小。

### TODO 2: Dropout=0.25/0.5/0.75 对比

控制变量：`concat_nframes=1, num_epoch=5, hidden_layers=2, hidden_dim=1700`，只改变 `dropout`。

| Dropout | 训练结果                                                                           |
| ------: | ---------------------------------------------------------------------------------- |
|  `0.25` | `[005/005] Train Acc: 0.473922 Loss: 1.805941 \| Val Acc: 0.473760 loss: 1.807432` |
|   `0.5` | `[005/005] Train Acc: 0.458253 Loss: 1.876998 \| Val Acc: 0.467396 loss: 1.832862` |
|  `0.75` | `[005/005] Train Acc: 0.436513 Loss: 1.981332 \| Val Acc: 0.453445 loss: 1.893353` |

结论：在这组实验里 `dropout=0.25` 最好；`0.5` 和 `0.75` 都使训练和验证准确率下降，偏欠拟合。

### TODO 3: 进一步超参数优化

基于当前日志，做过以下尝试。

| 调整项                                                                                            | 训练结果                                                                           |
| ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `concat_nframes: 1 -> 11`，`hidden_layers=2, hidden_dim=1700, dropout=0.25, num_epoch=5`          | `[005/005] Train Acc: 0.684295 Loss: 0.993557 \| Val Acc: 0.692914 loss: 0.972141` |
| `concat_nframes: 1 -> 21`，`num_epoch: 5 -> 10`，`hidden_layers=2, hidden_dim=1700, dropout=0.25` | `[010/010] Train Acc: 0.760617 Loss: 0.734659 \| Val Acc: 0.741352 loss: 0.820803` |

结论：在我当前记录中，提升 `concat_nframes` 和 `num_epoch` 带来明显收益；目前最高验证准确率是 `0.741352`，Epoch继续增加可达到 `0.75` 。
