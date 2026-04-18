# Day 10 RL

## HW 12

## 增加稳定评估工具

原始 notebook 的测试主要依赖：

```python
test_total_reward, action_list = test_agent(agent, env, max_steps=300, plot_flag=True)
print(np.mean(test_total_reward))
```

这个流程有两个问题：

- 测试 episode 太少，只有 5 个
- `eval()` 后依然在按策略分布随机采样动作，结果波动很大

因此我补充了更完整的评估函数。

### 给策略增加 deterministic 模式

`REINFORCE` 与 `ActorCritic` 都增加了：

```python
def policy(self, state, deterministic: bool=False):
    action_prob = self.actor(torch.FloatTensor(state).to(self.device))
    if deterministic:
        action = torch.argmax(action_prob, dim=-1)
    else:
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
    return action.detach().cpu().item()
```

### 新增汇总评估函数

```python
def summarize_scores(scores):
    return {
        'mean': ...,
        'std': ...,
        'min': ...,
        'max': ...,
        'success_rate_200': ...
    }
```

```python
def evaluate_agent(agent, env_name, seeds, num_test=5, max_steps=300, deterministic=False):
    ...
```

它会输出：

- 总均值 `mean`
- 标准差 `std`
- 最小/最大分数
- `success@200`，即分数达到 200 的比例
- 每个 seed 的详细结果

## 保存提交文件与服务器模拟验证改造

原始保存单元依赖外部状态，改动后变成：

```python
SELECTED_DETERMINISTIC = True
selected_scores, action_list, selected_summary = collect_action_list(
    agent, env_name,
    seed=NOTEBOOK_SEED,
    num_test=5,
    max_steps=300,
    deterministic=SELECTED_DETERMINISTIC
)

PATH = "Action_List.npy"
np.save(PATH, np.array(action_list, dtype=object), allow_pickle=True)
```

服务器模拟验证也改成了独立构造环境并固定每局 seed：

```python
env = gym.make(env_name)
all_seed(seed=NOTEBOOK_SEED, env=env)

for idx, actions in enumerate(action_list):
    state, _ = env.reset(seed=NOTEBOOK_SEED + idx)
    ...
```

## 当前改动的核心结论

这次修改没有改变强化学习算法本身的目标，而是把 notebook 从“能偶尔跑起来”提升到“可重复、可评估、可提交”的状态：

- 兼容 `gym 0.26.2`
- 兼容当前本地环境下的 `NumPy`
- 正确处理变长 `action_list`
- 区分 `sample` 与 `greedy` 评估
- 支持多 seed 统计，避免被单次高分误导

如果后续还要继续冲分，建议优先沿着这三个方向改：

- 优先只优化 `ActorCritic`
- 增大每个 batch 的采样 episode 数，降低梯度方差
- 在多 seed 稳定评估下比较超参数，而不是只看一次测试均值

## 训练稳定性增强

在前一版完成环境兼容与稳定评估后，本轮继续对 `ActorCritic` 和策略网络做了“小改动但有效”的增强，目标是降低训练方差、提升 `sample` 模式表现，并尽量让 `greedy` 也能跟上。

### 策略网络改成输出 logits

原始写法是网络最后一层直接做 `Softmax`：

```python
nn.Linear(32, action_dim),
nn.Softmax(dim=-1)
```

现在改成输出 logits：

```python
self.feature_net = nn.Sequential(
    nn.Linear(state_dim, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.Tanh(),
)
self.policy_head = nn.Linear(hidden_dim, action_dim)
```

并直接返回：

```python
return self.policy_head(x)
```

### 采样改成 `Categorical(logits=...)`

原始版本使用：

```python
action_prob = self.actor(state)
action_dist = Categorical(action_prob)
```

现在统一改成：

```python
logits = self.actor(state)
action_dist = Categorical(logits=logits)
```

这样数值更稳定，也更适合后续加入 entropy 正则。

### REINFORCE 增强

为 `REINFORCE` 增加了：

- `hidden_dim`
- `entropy_weight`
- `grad_clip`

并改成向量化写法：

```python
returns = ...
returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

logits = self.policy_net(states)
dist = Categorical(logits=logits)
log_probs = dist.log_prob(actions)
entropy = dist.entropy().mean()

loss = -(log_probs * returns).mean() - self.entropy_weight * entropy
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
```

### Actor-Critic 增强

本轮改动重点在 `ActorCritic`，新增了：

- `entropy_weight`
- `advantage_norm`
- `grad_clip`
- `critic_coef`

并把原来逐步反向传播改成整条 episode 向量化：

```python
returns = ...
values = self.critic(states).squeeze(-1)
advantages = returns - values.detach()

if self.advantage_norm:
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

logits = self.actor(states)
dist = Categorical(logits=logits)
log_probs = dist.log_prob(actions)
entropy = dist.entropy().mean()

actor_loss = -(log_probs * advantages).mean() - self.entropy_weight * entropy
critic_loss = self.critic_coef * F.smooth_l1_loss(values, returns)
```

训练时增加梯度裁剪：

```python
torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
```

### 训练超参数调整

`ActorCritic` 训练单元默认参数调整为：

```python
actor_lr=0.0015
critic_lr=0.0020
entropy_weight=0.01
advantage_norm=True
grad_clip=5.0
critic_coef=0.5
episode_per_batch=8
```

相较于原先：

- 增大了每个 batch 的采样 episode 数
- 加入 entropy 稳定探索
- 加入 advantage normalization 降低方差
- 使用 Huber loss 训练 critic，更稳一些

## 本轮重跑结果与相对上一轮的变化

### 上一轮 README 里记录的旧稳定评估结果

#### sample 模式

```text
mean=115.20, std=54.58, min=-12.91, max=196.19, success@200=0.00%
```

#### greedy 模式

```text
mean=61.71, std=28.16, min=17.72, max=108.16, success@200=0.00%
```

### 本轮重新训练后的结果

训练进度条末尾显示：

```text
Total=216.9, Recent=219.3, RecentBest=273.3, Final=65.1, Steps=208.6
```

说明模型在训练后期已经能够比较稳定地进入 `200+` 区间。

### 单次 5 局 sample 测试

```text
[262.72, 261.29, 298.64, 252.63, 46.04]
mean = 224.26
```

这个结果已经明显优于上一轮单次测试均值 `136.31`。

### 多 seed 稳定评估

#### sample 模式

```text
mean=210.07, std=102.95, min=-19.81, max=320.06, success@200=68.00%
```

#### greedy 模式

```text
mean=154.14, std=45.55, min=116.91, max=294.25, success@200=12.00%
```

### 实际提升总结

- `sample mean` 从 `115.20` 提升到 `210.07`
- `sample success@200` 从 `0.00%` 提升到 `68.00%`
- `greedy mean` 从 `61.71` 提升到 `154.14`
- `greedy success@200` 从 `0.00%` 提升到 `12.00%`

