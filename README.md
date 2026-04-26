# 11-Days-LeeDL

> [!NOTE]  
> 每日详细内容见 [StudyRecord](https://github.com/Cicada000/11-Days-LeeDL/tree/main/StudyRecord) 文件夹下每个文件夹内的 `README.md`。

本项目记录了在 11 天内完成李宏毅 2022 深度学习课程 15 个作业的实战历程。

## 概括

### Day 1: 基础工具与回归分析 (HW1)
*   **特征工程**: 对 COVID-19 数据集进行了相关性分析，筛选出前 20 个强相关特征（如 `tested_positive` 的历史值和 `hh_cmnty_cli`），显著降低了特征维度。
*   **模型优化**: 将默认的 SGD 优化器替换为 **Adam**，并加入 **L2 正则化 (weight_decay)**。
*   **监控**: 使用 **Tensorboard** 实时跟踪训练与验证损失，成功观察到模型收敛过程。

### Day 2: 语音音位分类 (HW2)
*   **架构实验**: 进行了“窄而深”（6层/1024维）与“宽而浅”（2层/1700维）模型的性能对比，结果显示在参数量相近时，宽浅模型在语音特征提取上略具优势。
*   **过拟合探究**: 测试了 0.25/0.5/0.75 三组 Dropout 率，确定 0.25 为最佳平衡点。
*   **上下文增强**: 将输入帧的拼接长度（concat_nframes）从 1 提升至 21，使验证集准确率从 0.45 飙升至 0.74+。

### Day 3: CNN 进阶与食物图像分类 (HW3)
*   **数据增广**: 构建了包含 `RandomResizedCrop`、`ColorJitter` 和 `AutoAugment` 的高强度数据流水线。
*   **迁移学习**: 实现了支持 ResNet18/50、DenseNet121 等多种骨干网络的工厂函数。
*   **集成学习**: 实现了测试时增强 (**TTA**) 结合加权**软投票 (Soft Vote)** 策略，有效提升了 Leaderboard 的排名。

### Day 4: 说话人识别与 Conformer (HW4)
*   **架构升级**: 将基础 Transformer 编码器重构为 **Conformer** 结构（FFN -> MHSA -> Conv -> FFN），更契合音频的时序特性。
*   **特征池化**: 引入 **Self-Attention Pooling**，配合 Padding Mask 自动学习有效帧权重，取代了粗暴的平均池化。
*   **推理优化**: 解决了长序列位置编码越界问题，并采用分段预测平均法提升了推理稳定性。

### Day 5: Transformer 机器翻译 (HW5)
*   **模型构建**: 补全了基于 Transformer 的 Encoder-Decoder 架构。
*   **调度策略**: 实现了 **Noam 学习率调度器**（Inverse Square Root）和标签平滑（Label Smoothing）。
*   **回译 (Back-translation)**: 完成了从单语语料生成伪平行数据并进行二次清洗的完整流程，验证 BLEU 值达到 15.68。

### Day 6: 生成模型与高级异常检测 (HW6 + HW8)
*   **GAN 增强**: 将 DCGAN 升级为 **WGAN-GP**，通过 InstanceNorm 代替 BatchNorm 并引入梯度惩罚，成功解决了动漫人脸生成的模式坍塌。
*   **异常检测**: 修复了 VAE 潜变量层使用 ReLU 的错误，最终采用 **ImageNet 预训练 ResNet18 特征 + 马氏距离 (Mahalanobis)** 的方案，Public Score 达到 0.87，排名第 4。

### Day 7: BERT 中文问答微调 (HW7)
*   **环境兼容**: 针对新版 Transformers 库编写了 `sitecustomize.py` 补丁以兼容旧代码。
*   **QA 策略**: 将传统的简单 `argmax` 预测优化为 **Top-k 跨度搜索**，并加入了最大答案长度限制，得分提升至 0.796。

### Day 8: 可解释性与对抗攻击 (HW10)
*   **攻击对比**: 实现了 **FGSM** 和 **I-FGSM**。实证发现，迭代式攻击（I-FGSM）能将模型准确率从 95% 彻底摧毁至 1%。
*   **防御观察**: 探索了 JPEG 压缩对对抗扰动的削弱作用，观察了分类结果的恢复现象。

### Day 9: 领域适应与模型压缩 (HW11 + HW13)
*   **迁移学习**: 发现 `real_or_drawing` 任务中，灰度图处理比 Canny 边缘检测能保留更多有效语义，使 DaNN 模型表现大幅提升。
*   **网络压缩**: 设计了参数量限制在 100k 以内的 **Inverted Residual (MobileNetV2)** 结构。
*   **知识蒸馏**: 应用了 EMA（指数移动平均）和两阶段训练法，配合 TTA 最终获得 0.78+ 的高分。

### Day 10: 强化学习控制 (HW12)
*   **环境适配**: 解决了 `Gym 0.26.2` 与原代码的 API 冲突（如 `reset` 返回值）。
*   **稳定性增强**: 将策略网络改为输出 **Logits**（Categorical 分布），引入了 Entropy 正则和 **Advantage Normalization**，显著降低了训练方差。

### Day 11: 终身学习与元学习 (HW14 + HW15)
*   **元学习**: 在少样本学习（Omniglot）任务中，修正了图像反相预处理与 Meta-batch 组装逻辑，验证准确率稳定在 97%。
*   **终身学习**: 使用 EWC 算法探究了模型在学习新任务时如何通过约束参数更新来保护旧任务的记忆。
