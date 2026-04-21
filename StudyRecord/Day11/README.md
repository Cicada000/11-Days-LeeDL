# Day 11 Meta Learning

## HW 15

## 数据预处理对齐

原 notebook 默认 few-shot 图像已经是 `28x28`，但标准 Omniglot 原图实际是 `105x105`。

因此在 `Omniglot` 数据集里补成了显式预处理：

```python
self.transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x),
])
```

这里做了两件事：

- 把输入统一缩放到模型原本预期的 `28x28`
- 增加反相 `1 - x`，让字符变成高亮前景，训练更容易收敛

## 修正 meta-batch 组装逻辑

原始 `get_meta_batch` 把输入尺寸硬编码成：

```python
reshape(-1, 1, 28, 28)
```

这会把数据读取阶段的尺寸假设直接锁死。现在改成从张量本身读取尺寸：

```python
channels, height, width = task_data.shape[2:]
train_data = task_data[:, :k_shot].reshape(-1, channels, height, width)
val_data = task_data[:, k_shot:].reshape(-1, channels, height, width)
```

这样 `support/query` 拆分逻辑和图像尺寸解耦，训练流程本身就更稳了。

## 训练配置优化

### 减小 `meta_batch_size`

原始配置是：

```python
meta_batch_size = 32
```

在当前数据规模下，这会导致每个 epoch 实际只有很少的 meta-update。  
现在改成：

```python
meta_batch_size = 8
```

这样每个 epoch 的更新次数明显增加，训练 acc 抬升更快。

### 增加验证阶段 inner steps

验证阶段从：

```python
val_inner_train_step = 3
```

改成：

```python
val_inner_train_step = 5
```

这更符合 few-shot 评估时“先在 support 上多做几步适应，再看 query 表现”的设定。

### 修正优化器初始化顺序

原 notebook 先构建模型和优化器，再在后面按 `FO / MAML` 分支修改 `meta_lr`，这会导致学习率配置和优化器实际使用值不完全同步。

现在把初始化改成：

```python
def model_init(lr=None):
    if lr is None:
        lr = meta_lr
    meta_model = Classifier(1, n_way).to(device)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=lr)
    ...
```

并在确定 `meta_lr` 后再创建 optimizer。

## 当前结果

由于未找到OmnigloTest数据集，故使用训练输出的acc来评估模型水平。

这次修改后，训练流程已经能正常跑通，且训练侧 acc 和验证 acc 都明显改善。

当前一次训练结果里，验证输出已经达到：

```text
Validation accuracy: 97.000 %
```