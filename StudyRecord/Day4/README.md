# Day 4 Self-Attention

本节记录 HW4 说话人分类任务中对 baseline notebook 的几项主要修改：将原本的 `TransformerEncoderLayer` 替换为更贴近语音任务的 Conformer 编码器，加入 `Self-Attention Pooling` 取代简单平均池化，并修复测试阶段超长语音超过位置编码上限导致的推理报错。

## Conformer Encoder

在 `Day4/HW4_Self-Attention.ipynb` 中，新增了 `SinusoidalPositionalEncoding`、`FeedForwardModule`、`ConformerConvModule` 和 `ConformerBlock`，再用多层 `ConformerBlock` 组成新的编码器。整体结构采用 Conformer 常见的 `FFN -> MHSA -> Conv -> FFN` 残差堆叠方式，同时保留较轻量的参数规模，方便直接在作业环境中训练。

```python
class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=256, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, hidden_dim=dim_feedforward, dropout=dropout)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2 = FeedForwardModule(d_model, hidden_dim=dim_feedforward, dropout=dropout)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        x = x + 0.5 * self.ffn1(x)
        attn_input = self.self_attn_layer_norm(x)
        attn_output, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + self.self_attn_dropout(attn_output)
        x = x + self.conv_module(x, padding_mask=padding_mask)
        x = x + 0.5 * self.ffn2(x)
        x = self.final_layer_norm(x)
        return x
```

## Self-Attention Pooling

原始 notebook 在 encoder 后直接做 `mean pooling`。这次改成 `Self-Attention Pooling`，让模型自己学习每个时间步的重要性，并配合 `padding mask` 避开 `pad_sequence(..., padding_value=-20)` 补出来的无效帧。

```python
class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x, padding_mask=None):
        attn = self.attention(x).squeeze(-1)
        if padding_mask is not None:
            attn = attn.masked_fill(padding_mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1).unsqueeze(-1)
        return torch.sum(x * attn, dim=1)
```

在最终 `Classifier.forward` 中，先根据 `mel == -20` 生成 `padding_mask`，再送入 Conformer 和 attention pooling，避免补齐位置影响时序建模与句级表征。

## 当前模型结构

新的 `Classifier` 流程如下：

```python
padding_mask = mels.eq(self.mel_padding_value).all(dim=-1)
out = self.input_norm(mels)
out = self.pre_net(out)
out = self.positional_encoding(out)
out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
for encoder_layer in self.encoder_layers:
    out = encoder_layer(out, padding_mask=padding_mask)
stats = self.pooling(out, padding_mask=padding_mask)
return self.pred_layer(stats)
```

相比原始版本：

- encoder 从单层 `TransformerEncoderLayer` 改成多层 Conformer。
- pooling 从 `out.mean(dim=1)` 改成可学习的 attention pooling。
- 对补齐帧显式构造了 `padding_mask`，减少无效位置对注意力和池化的干扰。

## 训练结果

本次训练沿用 notebook 默认配置 `n_epochs=35`、`batch_size=64`、`learning_rate=1e-3`。从日志来看，模型在后半段稳定提升，最佳验证结果出现在第 31 个 epoch。

- 最佳 `Valid loss`：`1.2334`
- 最佳 `Valid acc`：`0.7069`
- 对应 epoch：`31`
- 最终 epoch (`35`)：`Train loss=1.1821`，`Train acc=0.7008`，`Valid loss=1.2364`，`Valid acc=0.7063`
- 推理阶段额外修复：测试集中存在长度超过 `4096` 的语音，因此将位置编码改成按需动态扩展，避免 `4300 vs 4096` 的维度报错

从训练曲线看，模型在前 10 个 epoch 主要完成从随机初始化到可用表示的收敛，验证集准确率由 `0.0533` 提升到 `0.5195`；20 epoch 之后进入稳定打磨阶段，最终将验证集准确率提升到 `0.70+`，说明 Conformer 编码器配合 attention pooling 对时序建模和句级表征是有效的。

## 推理与提交

修复超长序列的位置编码问题后，测试阶段已能够完整跑通，并成功生成提交文件。

- 推理完成度：`8000 / 8000`
- Kaggle `Public Score`：`0.68025`

![result](pic/result.png)

这次分数主要来自推理阶段的修正：不再直接把整段测试语音一次性送进模型，而是按固定长度切成多段后分别预测，再对 logits 做平均，从而缓解训练时只见过短语音片段、测试时却输入整段语音带来的分布不一致问题。
