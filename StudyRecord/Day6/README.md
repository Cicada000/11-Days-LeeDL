# Day 6 GAN & AbnormalDetect

## GAN 结果

使用默认超参数设置，效果如下：

![result](pic/result-1.png)

## 模型优化 (WGAN-GP)

为了解决默认 GAN 模型（基础 DCGAN 架构）训练中出现的生成质量差、细节模糊以及模式坍塌（Mode Collapse）等问题，对 `HW06_GAN.ipynb` 进行了以下优化，将其升级为 **WGAN-GP** 架构：

### 数据增强
在 `get_dataset` 的图像预处理管道中加入了随机水平翻转：
```python
transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(), # 新增的数据增强
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
```
**说明**：由于动漫人脸具有较高的对称性，水平翻转相当于将有效训练数据量扩大了一倍，极大地增加了模型在训练过程中看到的数据多样性，提高了生成器的泛化能力。

### 判别器结构修改
去除了 Sigmoid，并替换了 Normalization：
```python
# 1. 移除最后一层的 Sigmoid 激活函数
nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0)  # output -> (batch, 1, 1, 1)
# nn.Sigmoid() removed for WGAN-GP

# 2. 将 BatchNorm2d 替换为 InstanceNorm2d
def conv_bn_lrelu(self, in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 4, 2, 1),
        nn.InstanceNorm2d(out_dim, affine=True), # 替换了 nn.BatchNorm2d
        nn.LeakyReLU(0.2),
    )
```
**说明**：
- 移除了 `Sigmoid` 函数，因为 WGAN 的判别器（Critic）输出的是一个无界的 Wasserstein 距离分数，而不是 [0, 1] 之间的概率值。
- 将所有的 `BatchNorm2d` 替换为 `InstanceNorm2d`（保留 affine 可学习参数）。这是 WGAN-GP 的数学硬性要求，因为 Batch Normalization 会导致同一个 Batch 内的样本相互影响，破坏梯度惩罚（Gradient Penalty）的独立计算条件。

### 损失函数与梯度惩罚
在 `TrainerGAN` 中添加了计算梯度惩罚的 `gp()` 函数：
```python
def gp(self, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).cuda()
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = self.D(interpolates)
    fake = torch.ones(d_interpolates.size()).cuda()
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
```
并更新了训练过程中的损失函数计算：
```python
# 判别器损失 (Discriminator Loss) 包含梯度惩罚项
gradient_penalty = self.gp(r_imgs, f_imgs)
loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + 10 * gradient_penalty

# 生成器损失 (Generator Loss)
loss_G = -torch.mean(self.D(f_imgs))
```
**说明**：这是 WGAN-GP 稳定训练的核心。通过在真实样本和生成样本之间的随机插值点上，强行限制梯度的范数（使其趋近于 1），满足 1-Lipschitz 连续性条件。这使得即便判别器被训练得非常完美，生成器也能持续获得稳定且不消失的梯度，从根本上缓解了模式坍塌问题。

### 优化器与超参数调整
更新了 `config` 并在 `TrainerGAN` 中修改了 Adam 优化器：
```python
config = {
    "model_type": "WGAN-GP",
    "batch_size": 512, # 从 64 提升至 512
    "lr": 2e-4,        # 从 1e-4 提升至 2e-4
    "n_epoch": 50,     # 从 15 提升至 50
    "n_critic": 5,     # 从 1 提升至 5
    "z_dim": 100,
    "workspace_dir": workspace_dir, 
}
```
修改 Adam 优化器的 `betas` 参数：
```python
self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr"], betas=(0.0, 0.9))
self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr"], betas=(0.0, 0.9))
```
修改之后的结果如下：

![result](pic/result-2.jpg)