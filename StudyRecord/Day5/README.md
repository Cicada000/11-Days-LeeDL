# Day 5 Seq2Seq

## 基础 Seq2Seq 训练框架

作业前半部分已经把一个完整的 seq2seq 训练流程搭起来，包括：

- 使用 fairseq 的 `TranslationTask` 读取 `DATA/data-bin/ted2020`
- 实现 `RNNEncoder`、`AttentionLayer`、`RNNDecoder` 和 `Seq2Seq` 封装
- 实现 `LabelSmoothedCrossEntropyCriterion`
- 实现 `NoamOpt` 学习率调度
- 接入 fairseq 的 beam search 做验证和测试推理
- 支持 checkpoint 保存、最佳模型保存和测试集预测生成

虽然 notebook 里仍然保留了 RNN 版 `Encoder` / `Decoder` 代码，方便和后续 transformer 版本对照，但当前实际训练入口已经切换到 transformer 结构。

## TransformerEncoder / TransformerDecoder 替换

`模型初始化` 这一段原本的 TODO 已经补完。现在 `build_model()` 中不再实例化 `RNNEncoder` 和 `RNNDecoder`，而是直接使用 fairseq 提供的 `TransformerEncoder` 与 `TransformerDecoder`，同时补齐了 transformer 所需的默认参数。

```python
def build_model(args, task):
    from fairseq.models.transformer import base_architecture

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    base_architecture(args)

    encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
    return Seq2Seq(args, encoder, decoder)
```

从 notebook 日志可以确认当前实际训练的模型已经是 transformer：

- `encoder: TransformerEncoder`
- `decoder: TransformerDecoder`
- `num. model params: 6,203,648`

## Noam 学习率公式

优化器部分的 TODO 也已经补完，`get_rate()` 按照 notebook 给出的公式实现了 transformer 常用的 inverse square root learning rate schedule：

```python
def get_rate(d_model, step_num, warmup_step):
    lr = np.power(d_model, -0.5) * min(
        np.power(step_num, -0.5),
        step_num * np.power(warmup_step, -1.5),
    )
    return lr
```

这部分和 `NoamOpt` 包装器一起使用，已经真正接入训练循环。

## 当前训练结果

根据 notebook 现有输出，当前 transformer 版本已经完整训练了 `15` 个 epoch，并成功完成验证与测试集预测。

- 最佳验证 BLEU：`15.68`
- 对应 epoch：`13`
- 第 `15` 个 epoch 验证损失：`4.2349`
- 第 `15` 个 epoch 验证 BLEU：`15.02`
- 测试集预测完成度：`4000 / 4000`

从训练日志看，BLEU 从第 1 个 epoch 的 `0.44` 持续提升到中后期的 `15+`，说明 transformer 版本已经能够稳定跑通完整训练和推理流程。

## Back-translation

- 下载并解压中文单语语料 `ted_zh_corpus.deduped.gz`
- 对单语中文做清洗，生成 `DATA/rawdata/mono/mono.clean.zh`
- 复用反向模型训练阶段的 sentencepiece 模型，对单语数据做 subword 编码，生成 `mono.tok.zh` 和 `mono.tok.en`
- 使用反向模型 `zh -> en` 在 `mono` split 上生成英文伪平行数据 `mono.pred.en`
- 将生成的英文预测与原始中文单语重新配对并再次清洗，得到 `mono.pred.clean.en` 和 `mono.pred.clean.zh`
- 对 synthetic parallel data 做 sentencepiece 编码，生成 `mono.synthetic.en` 和 `mono.synthetic.zh`
- 使用 fairseq 再次二进制化，并把结果写成正向训练可直接合并读取的 `DATA/data-bin/ted2020/train1.en-zh.*`

这里使用 `train1.en-zh.*` 是为了和原始 `train.en-zh.*` 配合；前面的训练代码使用了 `task.load_dataset(split="train", combine=True)`，因此在切回正向模型 `en -> zh` 后，可以直接把 back-translation 生成的数据与原始平行语料一起训练。