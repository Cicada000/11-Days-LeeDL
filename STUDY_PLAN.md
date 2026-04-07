# 深度学习 11 天进阶实战计划 (基于李宏毅 2022 课程)

本计划根据 `Homework/` 目录下的 15 个作业任务及预备知识精心编排，旨在通过“理论+实战”的方式，带你从基础神经网络走入前沿大模型与强化学习。

---

## 计划总览

| 天数       | 主题                          | 涉及作业     |
| :--------- | :---------------------------- | :----------- |
| **Day 1**  | **PyTorch 基础与回归预测**    | Warmup + HW1 |
| **Day 2**  | **特征提取与语音分类**        | HW2          |
| **Day 3**  | **CNN 图像分类与数据扩增**    | HW3          |
| **Day 4**  | **Transformer 与音频分类**    | HW4          |
| **Day 5**  | **机器翻译 (seq2seq)**        | HW5          |
| **Day 6**  | **生成模型 (GAN) 与异常检测** | HW6 + HW8    |
| **Day 7**  | **大模型微调 (BERT)**         | HW7          |
| **Day 8**  | **AI 安全与可解释性**         | HW9 + HW10   |
| **Day 9**  | **迁移学习与模型压缩**        | HW11 + HW13  |
| **Day 10** | **强化学习 (RL)**             | HW12         |
| **Day 11** | **终身学习与元学习**          | HW14 + HW15  |

---

## 每日详细安排

### Day 1: 基础工具与回归分析
*   **重点文件**: 
    *   `Homework/Warmup/Google_Colab_Tutorial.ipynb`
    *   `Homework/HW1_Regression/HW1_Regression.ipynb`
*   **核心任务**:
    *   配置 PyTorch 环境，掌握张量 (Tensor) 操作。
    *   补全 HW1 中的 `My_Model` 结构，使用 DNN 预测 COVID-19 病例。
    *   学会使用 **Tensorboard** 观察 Loss 曲线。

### Day 2: 深度/宽度网络与语音分类
*   **重点文件**: `Homework/HW2_Classification/HW2_Classification.ipynb`
*   **核心任务**:
    *   理解语音特征 **MFCC** 的提取原理。
    *   **实验对比**: 实现一个“深而窄” vs “浅而宽”的模型，记录 Acc 差异。
    *   掌握 **Dropout** 层对缓解过拟合的作用。

### Day 3: CNN 进阶与图像分类
*   **重点文件**: `Homework/HW3_CNN/HW3_CNN.ipynb`
*   **核心任务**:
    *   补全 CNN 架构，尝试引入 `BatchNorm`。
    *   **关键点**: 实现 **Data Augmentation** (数据扩增) 和模型集成 (**Ensemble**)。
    *   尝试使用现成的框架（如 ResNet）突破 Baseline。

### Day 4: 自注意力机制 (Self-Attention)
*   **重点文件**: `Homework/HW4_Self-Attention/HW4_Self-Attention.ipynb`
*   **核心任务**:
    *   学习 Transformer 的 Encoder 结构。
    *   **进阶**: 将 Transformer 替换为 **Conformer** 结构。
    *   理解 **Warmup** 学习率策略在 Transformer 训练中的重要性。

### Day 5: 序列到序列 (seq2seq) 翻译
*   **重点文件**: `Homework/HW5_seq2seq/HW05_seq2seq.ipynb`
*   **核心任务**:
    *   实现英译中繁体的机器翻译模型。
    *   补全 RNN 或 Transformer 架构的 Decoder。
    *   了解 **Back-translation** (回译) 提升翻译质量的方法。

### Day 6: 生成式 AI 与异常检测
*   **重点文件**: 
    *   `Homework/HW6_GAN/HW06_GAN.ipynb`
    *   `Homework/HW8_AbnormalDetect/HW08_AbnormalDetect.ipynb`
*   **核心任务**:
    *   **GAN**: 动漫人脸生成，理解判别器与生成器的对抗博弈。
    *   **异常检测**: 利用 **VAE** (变分自编码器) 重建图像，根据重建 Loss 识别异常样本。

### Day 7: BERT 预训练模型应用
*   **重点文件**: `Homework/HW7_Bert/HW07-Bert.ipynb`
*   **核心任务**:
    *   学习使用 HuggingFace 的 `transformers` 库。
    *   在中文抽取式问答任务上微调 `bert-base-chinese`。
    *   掌握 `doc_stride` 处理长文本的技巧。

### Day 8: 可解释性与对抗攻击
*   **重点文件**: 
    *   `Homework/HW9_ExplainableAI/HW09-ExplainableAI.ipynb`
    *   `Homework/HW10_AdversarialAttack/HW10-AdversarialAttack.ipynb`
*   **核心任务**:
    *   **Explainable AI**: 使用 Saliency Map 查看 CNN 到底在“看”图片的哪里。
    *   **对抗攻击**: 学习如何通过微小的扰动使分类模型失效。

### Day 9: 领域适应与模型压缩
*   **重点文件**: 
    *   `Homework/HW11_Adaptation/HW11-Adaptation.ipynb`
    *   `Homework/HW13_NetworkCompress/HW13-networkCompress.ipynb`
*   **核心任务**:
    *   **Adaptation**: 实现 **DaNN** 解决训练域与测试域分布不一致的问题。
    *   **压缩**: 学习 **Knowledge Distillation** (知识蒸馏) 和 **Depthwise/Pointwise 卷积** 实现轻量化。

### Day 10: 强化学习 (RL)
*   **重点文件**: `Homework/HW12_RL/HW12-RL.ipynb`
*   **核心任务**:
    *   配置 OpenAI Gym 环境。
    *   实现 Policy Gradient 算法控制 LunarLander。

### Day 11: 终身学习与元学习
*   **重点文件**: 
    *   `Homework/HW14_LifeLongML/HW14-LifeLongMachineLearning.ipynb`
    *   `Homework/HW15_MetaLearning/HW15-MetaLearning.ipynb`
*   **核心任务**:
    *   **LifeLong ML**: 使用 **EWC** 算法解决神经网络的“灾难性遗忘”。
    *   **Meta Learning**: 掌握 **MAML** 算法，让模型学会“如何快速学习新任务”。
