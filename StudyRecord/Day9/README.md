# Day 9 Adaptation & NetworkCompress

## HW 11 Adaptation

本次实验基于原始的 `HW11-Adaptation.ipynb` 进行优化，目标是提升 `real_or_drawing` 数据集在 Kaggle leaderboard 上的分类分数。

原始 notebook 直接运行并提交 `DaNN_submission.csv` 时，得分大约在 `0.52` 左右。

---

## 原始 `HW11-Adaptation.ipynb` 的设置

原始 notebook 的核心流程是标准 `DaNN`：

### 1. 数据预处理

源域 `source` 使用 `Canny` 边缘提取，目标域 `target` 使用灰度图：

```python
src_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])

tgt_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])
```

这里有两个明显问题：

- `target` 的训练与测试共用了同一套 transform
- 推理阶段仍然会执行 `RandomHorizontalFlip` 和 `RandomRotation`，导致提交结果具有随机性

### 2. 训练策略

原始 notebook 的训练损失如下：

```python
loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
```

并且固定：

```python
lamb = 0.1
```

训练 `200` 个 epoch，并且每一轮都直接覆盖保存：

```python
torch.save(feature_extractor.state_dict(), f'extractor_model.bin')
torch.save(label_predictor.state_dict(), f'predictor_model.bin')
```

也就是说，原始 notebook 实际上只保留了“最后一轮权重”，没有做验证集筛选，也没有做后期 checkpoint 对比。

---

## 优化过程

本次对 `HW11-Adaptation.ipynb` 的探索主要分成三步：先修复推理阶段的随机增强问题，再比较不同源域预处理方式，最后采用更适合 leaderboard 的灰度输入 + 晚期 checkpoint 导出方案。

### 1. 拆分目标域训练与测试变换

首先将 `target` 的训练增强与测试增强分开：

```python
tgt_train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])

tgt_test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
```

**说明**：这一步的目的是让提交 `csv` 时的推理变成确定性过程，避免同一个模型多次提交分数不一致。

### 2. 比较三种源域预处理方式

为了判断到底是“模型不够强”还是“输入形式不对”，对源域 `source` 进行了三种预处理对比：

- `canny`
- `blur_canny`
- `gray`

其中：

```python
source-mode = "canny" / "blur_canny" / "gray"
```

实验结论非常明确：

- `canny` 路线分数最低
- `blur_canny` 比 `canny` 更差
- `gray` 明显优于前两者

这说明本题最大的误区不是模型深度不够，而是**持续把 source 域硬转换成边缘图**。  
目标域虽然是 drawing，但它更接近“干净的灰度线稿”，而不是 source 照片经过 `Canny` 后产生的大量背景噪声边缘。

### 3. 最终采用的训练脚本

最终将实验整理为独立脚本 `hw11_adaptation_optimized.py`，核心方向如下：

- 源域输入改为 `gray`
- 不再按 source 验证集选“最终模型”
- 使用全量 source 数据参与训练
- 训练前期使用短暂 warmup
- 中后期使用更强的 `DaNN`
- 固定导出晚期 checkpoint：`120, 140, 160, 180, 200`
- 同时导出单 checkpoint 提交文件与 ensemble 提交文件

---

## 实验结果记录

### 1. 早期三组预处理对比

三组源域预处理的主要结果如下：

- `canny`
  - source-only 分数约 `0.21`
  - adaptation 分数约 `0.47`

- `blur_canny`
  - source-only 分数约 `0.17`
  - adaptation 分数约 `0.44`

- `gray`
  - source-only 分数约 `0.35`
  - adaptation 分数约 `0.60`

从这里可以看出，`gray` 是后续继续优化的唯一合理方向。

### 2. 最终灰度输入方案的 leaderboard 结果

最终脚本导出了以下文件：

- `outputs/hw11_gray_dann/submission_gray_epoch120.csv`
- `outputs/hw11_gray_dann/submission_gray_epoch140.csv`
- `outputs/hw11_gray_dann/submission_gray_epoch160.csv`
- `outputs/hw11_gray_dann/submission_gray_epoch180.csv`
- `outputs/hw11_gray_dann/submission_gray_epoch200.csv`
- `outputs/hw11_gray_dann/submission_gray_ensemble.csv`

对应 Kaggle 分数如下：

![result](pic/result-1.png)

### 3. 后续继续训练的结果

在 `epoch 200` 取得最佳结果后，又继续尝试了更晚的 checkpoint：

- `epoch 220`
- `epoch 240`

但这两次提交的分数都回落到了 `0.68` 左右，因此没有继续采用。

这说明：

- `epoch 200` 附近已经是当前训练配置下的最佳点
- 后续继续训练会开始伤害 target 域表现

---

## 最终结论

最终最佳结果为：

- `submission_gray_epoch200.csv`
- `Private Score = 0.74078`
- `Public Score = 0.74378`

对于当前仓库状态，这是本次实验得到的最佳方案。
