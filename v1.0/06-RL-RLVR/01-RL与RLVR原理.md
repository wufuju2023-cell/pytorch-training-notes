# 06 · RL 与 RLVR 原理（从策略梯度到可验证奖励）

> 本篇是全书的“后训练”第一站。前面 01–05 章把模型、注意力、SFT、价值头、LoRA 拆开讲清楚了；
> 从这里开始，模型不再只是“模仿数据”，而是**用奖励去搜索更好的证明策略**。
>
> 读完本篇你会：
> 1. 完整推导策略梯度定理、REINFORCE、基线/优势；
> 2. 说清 PPO 的 clip 在做什么、GRPO 为什么不需要 critic；
> 3. 理解 RLVR（可验证奖励）为什么是 Lean 证明训练的主路线；
> 4. 把公式和 AlphaProof 的三个真实实现对上号（`/ttt_step`、`gpu_runtime`、`nanoproof/rl.py`）。
>
> 配套代码见 [`code/from_scratch/grpo.py`](code/from_scratch/grpo.py)（`F-b06-grpo`）与 [`code/with_api/trl_grpo.py`](code/with_api/trl_grpo.py)（`F-b06-trl_grpo`）；
> 配套实验见 notebooks `N13_GRPO最小实现`（`N-13`）、`N14_RLVR可验证奖励`（`N-14`）。
>
> **【文档｜DOC-RL】**（doccode = `RL`）｜编号与 Tag 规范见《00-风格与编号规范》。

---

## 0. 文件对照速查

**【注 RL.1.1｜R-RL.1.1】（源码对照速查）**

| 概念 | AlphaProof 源码位置 | 说明 |
|---|---|---|
| RTTT 一步更新 | `app/policy_server.py:416` `ttt_step` | loss = `-r*(logp-logp_old) + BETA_KL*(dlogp)^2 + value_coeff*MSE` |
| KL 防忘系数 | `app/policy_server.py:46` `BETA_KL = 0.05` | 见 `v1-spec 02` |
| 价值头 MSE/TD | `app/policy_server.py:356` `train_value`、`:467-475` | `value_coefficient * mse_loss` |
| GPU 后端 learn | `v1-1-agentic-tool/gpu/gpu_runtime/verified_backend.py:216`、`qwen35_backend.py:464`、`search_backend.py:276`、`real_backend.py:569`、`toy_backend.py:74` | 各 backend 的 `learn(session_id, event)` |
| 混合目标（SFT+replay+价值分类） | `gpu_runtime/mixed_objective.py` | 返回按距离类别的分类目标 |
| nanoproof RL | `nanoproof/nanoproof/rl.py:983-1003` | token-mean CE 正样本 + unlikelihood 负样本 |
| 经验回放 | `nanoproof/nanoproof/experience_collection.py` | `ReplayBuffer` / `NegativeBuffer` / `Matchmaker` |
| MCTS 收集 | `nanoproof/nanoproof/search.py` | `run_mcts` / `expand_node` / `backpropagate` |

既有数学文档（只读参考）：

- [`/home/a/文档/RLVR-OPSD-参考资料/02-RLVR-数学理论详解.md`](/home/a/文档/RLVR-OPSD-参考资料/02-RLVR-数学理论详解.md)：token-level MDP、策略梯度、GAE、PPO、GRPO、DAPO、GSPO、KL 估计器、方差分解。
- [`/home/a/文档/ttt-参考资料/01-问题定义与框架.md`](/home/a/文档/ttt-参考资料/01-问题定义与框架.md)：测试时训练（TTT）的分布偏移分类与协议。
- [`/home/a/文档/信息几何理论参考资料/00-导读与目录.md`](/home/a/文档/信息几何理论参考资料/00-导读与目录.md)：自然梯度、Fisher 信息与 KL 的几何解释。

---

## 1. 把“写证明”建模成 MDP

**【定义 RL.2.1｜D-RL.2.1】（token 级 MDP 与轨迹概率）**

我们把一次证明交互写成一个**token 级马尔可夫决策过程**（token-level MDP）。令

- 状态 $s_t = (q, a_{1:t-1})$：题目 $q$ 加上已经生成的 token（或 tactic）前缀；
- 动作 $a_t \in \mathcal{V}$：下一个 token（$\mathcal{V}$ 是词表）；在 tactic 层面，动作也可以是一条完整 tactic；动作分布 $\pi_\theta(\cdot\mid s_t)$ 即语言模型在该位置的 softmax 分类分布（【定义 1.2.1】）；
- 转移 $P(s_{t+1} \mid s_t, a_t)$：自回归模型里是**确定性拼接** $s_{t+1} = s_t \circ a_t$（环境侧还可能执行 Lean 得到新目标）；
- 奖励：通常只在终止步给 $r_T$。RLVR 里 $r_T \in \{0,1\}$（Lean 内核验证通过为 1），也可以给形状奖励（证明步数越短越高）。

一条轨迹 $\tau = (s_0, a_0, s_1, a_1, \dots, s_{T-1}, a_{T-1}, s_T)$ 的概率为

$$
p_\theta(\tau) = \mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\, P(s_{t+1} \mid s_t, a_t),
$$

其中 $\mu$ 是初始状态分布，$\pi_\theta$ 是我们要优化的策略（语言模型），$P$ 是环境动力学（与 $\theta$ 无关）。
目标是最大化期望回报

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)],
\qquad
R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_{t+1}.
$$

**【注 RL.2.2｜R-RL.2.2】（确定性转移：随机性全部来自策略采样）**

> 关键区别：在普通 RL 里状态转移含随机性；在自回归生成里 $P$ 基本是确定的，
> **随机性全部来自策略采样**。这让策略梯度有非常干净的形式（见 §2）。

---

## 2. 策略梯度定理（完整推导）

**【定理 RL.3.1｜T-RL.3.1】（策略梯度定理）**

**【证明｜Pf-RL.3.1】**

我们对 $\theta$ 求梯度。先算 $\nabla_\theta p_\theta(\tau)$。由对数导数技巧（log-derivative trick），

$$
\nabla_\theta p_\theta(\tau)
= p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau).
$$

再展开 $\log p_\theta(\tau)$：

$$
\log p_\theta(\tau)
= \log \mu(s_0) + \sum_{t=0}^{T-1} \Big[\log \pi_\theta(a_t \mid s_t) + \log P(s_{t+1} \mid s_t, a_t)\Big].
$$

因为 $\mu$ 和 $P$ 都不含 $\theta$，求梯度后只剩策略项：

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t).
$$

代入期望：

$$
\nabla_\theta J(\theta)
= \nabla_\theta \mathbb{E}_{\tau \sim p_\theta}[R(\tau)]
= \int \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau
= \int p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, R(\tau)\, d\tau.
$$

于是得到**策略梯度定理**：

$$
\boxed{\;\;
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim p_\theta}\!\left[\, R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right].
\;\;}
$$

常用等价形式（回报写成动作价值函数）：

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{s \sim d^{\pi}, a \sim \pi_\theta}\!\left[\, Q^{\pi}(s,a)\, \nabla_\theta \log \pi_\theta(a \mid s) \right],
$$

其中 $d^{\pi}$ 是策略 $\pi$ 下的状态占用分布，$Q^{\pi}(s,a) = \mathbb{E}[R \mid s, a]$。

---

## 3. REINFORCE、基线与优势

### 3.1 REINFORCE

**【定理 RL.4.1｜T-RL.4.1】（REINFORCE 更新）**

用一条采样轨迹估计上面的期望，就得到 **REINFORCE** 更新：

$$
\nabla_\theta J \approx R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t).
$$

直觉：如果这条轨迹回报高（$R>0$），就提高它的所有动作概率；否则降低。

### 3.2 基线为什么不引入偏差

**【命题 RL.4.2｜P-RL.4.2】（基线为什么不引入偏差）**

直接乘 $R(\tau)$ 方差很大。我们减去一个**只依赖状态**的基线 $b(s)$：

$$
\nabla_\theta J
= \mathbb{E}\!\left[ (Q^{\pi}(s,a) - b(s))\, \nabla_\theta \log \pi_\theta(a\mid s) \right].
$$

**【证明｜Pf-RL.4.2】**

它**无偏**，因为

$$
\mathbb{E}_{a \sim \pi_\theta}\!\left[ b(s)\, \nabla_\theta \log \pi_\theta(a\mid s) \right]
= b(s)\, \nabla_\theta \sum_a \pi_\theta(a\mid s)
= b(s)\, \nabla_\theta 1 = 0.
$$

**【定义 RL.4.3｜D-RL.4.3】（优势函数）**

取 $b(s) = V^{\pi}(s)$，乘积变成**优势函数**

$$
A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s),
$$

它在“这个动作比该状态下的平均好多少”这个意义上是最优基线的近似。在 LLM 的 RLHF/RLVR 里，
$V$ 通常由一个**价值头**（第 04 章）估计，$r_T$ 加上价值 bootstrap 构成优势。

### 3.3 GAE：优势的方差–偏差折中

**【定义 RL.4.4｜D-RL.4.4】（GAE：优势的方差–偏差折中）**

广义优势估计（GAE）用 $\lambda$ 在偏差与方差之间插值：

$$
\hat{A}_t = \sum_{l=0}^{T-1-t} (\gamma\lambda)^l\, \delta_{t+l},
\qquad
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t).
$$

- $\lambda = 0$：$\hat{A}_t = \delta_t$（低方差、高偏差）；
- $\lambda = 1$：$\hat{A}_t = \sum \gamma^l r_{t+l+1} - V(s_t)$（高方差、低偏差）。

---

## 4. 重要性采样与 off-policy 修正

**【定义 RL.5.1｜D-RL.5.1】（重要性比与单步代理目标）**

策略梯度要求样本来自**当前** $\pi_\theta$。若数据由旧策略 $\pi_{\text{old}}$ 采样，就需要**重要性采样**（importance sampling）。

对一条轨迹，重要性比是 $\prod_t \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}$。
在 token 级做**一阶近似**（只保留该步的比值），得到单步代理目标：

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)},
\qquad
J^{\text{IS}}(\theta) = \mathbb{E}_{t, a \sim \pi_{\text{old}}}\!\left[ \rho_t(\theta)\, \hat{A}_t \right].
$$

$\rho_t$ 偏离 1 越远，估计方差越大——这正是 PPO clip 要控制的量。

---

## 5. PPO：clip 的随机代理目标

**【定义 RL.6.1｜D-RL.6.1】（PPO clip 目标）**

PPO 用**截断**阻止新策略一次走太远。对每个 $(s_t, a_t)$：

$$
L^{\text{CLIP}}(\theta)
= \mathbb{E}\!\left[ \min\!\Big( \rho_t(\theta)\, \hat{A}_t,\; \operatorname{clip}\big(\rho_t(\theta), 1-\epsilon, 1+\epsilon\big)\, \hat{A}_t \Big) \right].
$$

**【注 RL.6.2｜R-RL.6.2】（clip 行为解读）**

理解方式（取 $\epsilon = 0.2$）：

- $\hat{A}_t > 0$（好动作）：想增大 $\rho_t$，但一旦 $\rho_t > 1+\epsilon$，$L$ 被钉在 $(1+\epsilon)\hat A_t$，再增大无梯度 → 不贪心；
- $\hat{A}_t < 0$（坏动作）：想减小 $\rho_t$，但一旦 $\rho_t < 1-\epsilon$，$L$ 被钉在 $(1-\epsilon)\hat A_t$ → 不激进。

**【定义 RL.6.3｜D-RL.6.3】（完整 PPO 目标）**

完整的 PPO 目标还减去 KL 惩罚：

$$
J^{\text{PPO}}(\theta)
= \mathbb{E}\!\left[ L^{\text{CLIP}}(\theta) - \beta\, D_{\mathrm{KL}}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big) \right].
$$

**【注 RL.6.4｜R-RL.6.4】（与 TRPO 的关系）**

> **与 TRPO 的关系**：TRPO 在“KL 约束”下做单调改进，等价于一阶近似求解一个带信赖域的优化；
> PPO 的 clip 是它的无约束、可微、易实现版本。二者都源于对**替代目标**（surrogate objective）
> 的单调改进保证（参见 `RLVR-OPSD-参考资料/02` §5）。

---

## 6. GRPO：组内相对优势（不需要 critic）

**【定义 RL.7.1｜D-RL.7.1】（GRPO 组内相对优势）**

**GRPO（Group Relative Policy Optimization）** 的核心创新：**丢弃价值网络**，用“同一道题的一组采样”内的统计量当基线。

对每道题 $q$，采样一组 $G$ 个回答 $\{o_1, \dots, o_G\}$，得到奖励 $\{r_1, \dots, r_G\}$。组内优势定义为

$$
\hat{A}_i = \frac{r_i - \operatorname{mean}(\{r_j\})}{\operatorname{std}(\{r_j\}) + \varepsilon_{\text{std}}}.
$$

**【注 RL.7.2｜R-RL.7.2】（均值基线与归一化的偏差）**

- 均值 $\operatorname{mean}(r)$ 充当 $V(q)$ 的蒙特卡洛估计 → 无偏基线（同 §3.2）；
- 除以标准差是**归一化**，让不同难度的题梯度尺度一致，但它**引入小的偏差**（Dr. GRPO 指出这一点，见 `RLVR-OPSD-参考资料/02` §7）。

**【定义 RL.7.3｜D-RL.7.3】（GRPO 目标）**

GRPO 的目标（带 clip 的 token 级版本）：

$$
J^{\text{GRPO}}(\theta)
= \mathbb{E}_{q,\{o_i\}}\!\left[
\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\min\!\Big( \rho_{i,t}(\theta)\, \hat{A}_i,\; \operatorname{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\, \hat{A}_i \Big)
\right]
- \beta\, D_{\mathrm{KL}}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big).
$$

**【注 RL.7.4｜R-RL.7.4】（为什么能省掉 critic）**

**为什么能省掉 critic**：在 RLVR 里奖励是**可验证**的 0/1 信号，同一道题的组内奖励就提供了充分的状态价值估计；
训练一个 60M+ 的价值头既贵又难校准。GRPO 用“同题多样本 + 组内归一化”换掉了它。

**【注 RL.7.5｜R-RL.7.5】（全同组退化与难度课程）**

> 要点：GRPO 对**同一 prompt 的多个完成**做相对比较。如果整组奖励全相同（全对/全错），
> $\operatorname{std} \to 0$，优势退化 → 该题几乎无梯度。这解释了为什么需要**难度适中的采样池**（课程/难度筛选）。

---

## 7. KL 正则：防遗忘与信赖域

**【定义 RL.8.1｜D-RL.8.1】（KL 正则与三种估计器）**

语言模型 RL 用 KL 惩罚把新策略拴在参考策略 $\pi_{\text{ref}}$（通常是 SFT 模型）附近，防止：
（i）为拿奖励而“胡言乱语”导致语言能力崩溃；（ii）灾难性遗忘。

常见三种估计器（都来自 $D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}}) = \mathbb{E}_{\pi_\theta}[\log \frac{\pi_\theta}{\pi_{\text{ref}}}]$ 的蒙特卡洛变形）：

$$
\text{k1} = \log\frac{\pi_\theta}{\pi_{\text{ref}}},
\qquad
\text{k2} = \tfrac{1}{2}\Big(\log\frac{\pi_\theta}{\pi_{\text{ref}}}\Big)^2,
\qquad
\text{k3} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1.
$$

**【注 RL.8.2｜R-RL.8.2】（三种估计器比较与信息几何）**

- **k1** 方差大但无偏；**k2** 方差不那么敏感，是“一阶近似”；**k3** 恒非负、低方差，被后续工作（如 DeepSeek / Kimi）采用。
- 从信息几何看，KL 是统计流形上的**局部二次型**，其 Hessian 就是 Fisher 信息矩阵；自然梯度正是用 Fisher 做预条件
  （见 `信息几何理论参考资料/07-自然梯度与Fisher效率.md`）。

**【定义 RL.8.3｜D-RL.8.3】（RTTT 的平方对数比 KL）**

在 AlphaProof 的 `/ttt_step` 里，KL 惩罚被实现为一个**平方对数比**：

$$
\text{kl} = \beta_{\mathrm{KL}}\, \big(\log \pi_\theta(a) - \log \pi_{\text{old}}(a)\big)^2,
\qquad \beta_{\mathrm{KL}} = 0.05,
$$

即 k2 形式，作用在**本次更新前**的旧策略上（防止在线 TTT 一步走太远）。

---

## 8. RLVR：可验证奖励

**【定义 RL.9.1｜D-RL.9.1】（RLVR：可验证奖励）**

**RLVR（Reinforcement Learning with Verifiable Rewards）** 指奖励来自一个**确定性验证器**，而不是人类偏好模型：

- 数学：答案字符串 / 数值 / 符号等价（SymPy、`mathlib` 判定）；
- 代码：单元测试；
- 定理证明：**Lean 内核**把 tactic 脚本编译成 `Expr` 并 `checkProof`。验证通过 = 奖励 1，否则 0。

相比 RLHF，RLVR 的奖励**不可被 reward model 攻击**（没有可学习的奖励模型），信号稀疏但干净：

$$
r(\tau) = \mathbb{1}\big[\text{verifier}(\tau) = \text{pass}\big].
$$

**【注 RL.9.2｜R-RL.9.2】（稀疏奖励的工程解法）**

难点是**稀疏 + 全 0 组**：证明题往往整组失败，优势全为 0。工程解法：
（i）用**形状奖励**（证明越短、步骤越少越高）；（ii）对未解出的组用搜索中间信号（value 头 / 部分进展）；
（iii）用课程（matchmaker）保证题目难度。

**【注 RL.9.3｜R-RL.9.3】（RLVR 的能力边界）**

**RLVR 的能力边界**（`RLVR-OPSD-参考资料/02` §13）：RLVR 只是**重加权已有采样分布**，提高 pass@1，
但**不会创造新解法**——它把 pass@k 的覆盖度重新分配。对新定理，必须靠预训练 + 搜索（第 07 章 MCTS）来扩展覆盖。

---

## 9. on-policy 稳定性、熵坍塌与奖励攻击

**【注 RL.10.1｜R-RL.10.1】（on-policy 稳定性与熵坍塌）**

- **on-policy 的意义**：策略梯度对数据分布敏感。数据越偏离当前策略，重要性比越偏，方差越大。
  PPO/GRPO 复用同批数据做多次更新（epoch），所以必须靠 clip + KL 把偏离限制在信赖域内。
- **熵坍塌**：策略过早收敛到少数高回报模式，$\mathcal{H}(\pi) \to 0$，探索消失。GRPO 里表现为组内答案几乎相同、$\operatorname{std}\to 0$。
  对策：熵正则、$\epsilon$ 更高（DAPO 的 clip-higher）、动态采样（丢弃全对/全错的组）。
**【注 RL.10.2｜R-RL.10.2】（奖励攻击与长度/难度偏差）**

- **奖励攻击的数学解释**：当 $D_{\mathrm{KL}}$ 约束松弛，最优策略会无限放大奖励高但语言上退化的输出，
  即 $\pi^{*} \propto \pi_{\text{ref}} e^{R/\beta}$；$\beta$ 越小越“偏”。见 `RLVR-OPSD-参考资料/02` §12。
- **长度/难度偏差**：Dr. GRPO 证明按 token 平均的损失会偏向短回答、按序列平均会带来长度偏差（§7）。

---

## 10. RTTT / TTRL：测试时训练的两个近亲

**【定义 RL.11.1｜D-RL.11.1】（TTRL 与 RTTT）**

- **TTRL（Test-Time Reinforcement Learning）**：在**推理阶段**对无标签测试题用多数投票等构造**伪标签**，
  再用 RL 目标（如 GRPO）更新模型。它是“自训练 + 可验证/自洽奖励”的结合。
- **RTTT（Reinforcement Test-Time Training）**：比 TTRL 更进一步——在**一次搜索过程中**，
  把搜索/验证得到的中间信号（value、visit 统计、子目标成功）当作奖励，**在线更新策略**，
  使同一次搜索后面的展开用到更新后的策略。AlphaProof 的 `/ttt_step` 与 `cpu_runtime/online_ttt.py` 都是这个思想。

**【算法 RL.11.2｜A-RL.11.2】（online_ttt 闭环）**

`cpu_runtime/online_ttt.py` 的闭环（见 `run_online`，`online_ttt.py:395`）：

1. 启动 `lake env lean theorem.lean` 子进程，用环境变量把 `REAP_SESSION_ID` / `REAP_POLICY_ENDPOINT` 注入 Lean；
2. Lean 搜索内核（V1 MCTS）每展开一个叶子就发一次 policy+value 请求；
3. 搜索过程中产生的 observer 事件（`observer.jsonl`）被 `OnlineCoordinator.accept` 消费，聚合成一个 **learn event**；
4. 调用 `POST /sessions/{sid}/learn/v1`（即 GPU 侧 `GpuRuntime.learn`）执行一步 `/ttt_step`；
5. 之后新的生成就用更新后的策略（`post_update_generations`），并把结果写进 `online-result.json`。

**【注 RL.11.3｜R-RL.11.3】（V1 闭环）**

> 这就是“**策略网络采样 → PUCT → 展开 → 回传 → 在线更新**”的 V1 闭环：搜索负责产生数据，RL 负责更新策略，二者交替。
> 详见第 07 章。

---

## 11. AlphaProof 三个实现逐行对照

### 11.1 `app/policy_server.py` 的 `/ttt_step`（RTTT 一步）

**【注 RL.12.1｜R-RL.12.1】（`policy_server.py` 的 `/ttt_step`）**

对每条 item `{prompt, target, r, logprob_old}`，代码在 `:462-463` 计算

$$
\mathcal{L}_{\text{policy}} = -r\,(\log p - \log p_{\text{old}}),
\qquad
\mathcal{L}_{\text{kl}} = \beta_{\mathrm{KL}}\,(\log p - \log p_{\text{old}})^2,
$$

其中 $\log p$ 是模型对 target 的**序列对数概率**（`out.loss` 取负，`:452-453`），$\log p_{\text{old}}$ 是采样时的对数概率（detach 的常数）。
再加上价值头 MSE（`value_coeff * mse_loss`，`:472`），最后

$$
\mathcal{L} = \frac{\mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{kl}} + c_v \mathcal{L}_{\text{value}}}{\text{count}}.
$$

**与 §3/§7 的对应**：
- 把 $r$ 看作优势估计（RLVR 里 $r = \pm 1$ 或 shaped reward），$\nabla_\theta \mathcal{L}_{\text{policy}} = -r \nabla_\theta \log p$，
  这正是 REINFORCE（$\log p_{\text{old}}$ 是常数，梯度为 0，只作数值上的“锚点”）。
- KL 项是 k2 形式，$\beta_{\mathrm{KL}} = 0.05$（`:46`）。
- 价值项是 scalar 价值头的 TD/MSE 目标（见《04-价值头原理》；MSE 与高斯 MLE 的对应见【命题 1.2.14】）。

> 注意：这**不是** PPO（没有 clip、没有重要性比 $\rho$），而是更接近“带 KL 的 REINFORCE + value 回归”。
> 因为它是**严格 on-policy 的单步 TTT**：每条数据只用一次，且用的就是刚采样的策略。

### 11.2 `gpu_runtime` 各 backend 的 `learn()`

**【注 RL.12.2｜R-RL.12.2】（`gpu_runtime` 各 backend 的 `learn()`）**

| backend | `learn` 位置 | 目标 |
|---|---|---|
| `verified_backend.py` | `:216` | 已验证轨迹（verified replay）上的策略/价值更新 |
| `qwen35_backend.py` | `:464` | Qwen3.5 QLoRA 后端的逐个 learn |
| `search_backend.py` | `:276` | 搜索过程中收集的经验 |
| `real_backend.py` | `:569` | 真实模型后端 |
| `toy_backend.py` | `:74` | CPU 玩具后端（教学/自测） |
| `mixed_objective.py` | `prepare_mixed_event` / `next_mixed_batch` | 混合 SFT + replay + 价值分类目标 |

所有 learn 都经过 `GpuRuntime.learn`（`runtime.py:425`）：它带 `expected_policy_version` 做**乐观并发控制**，
只有版本匹配才更新，返回含 `loss / policy_loss / kl / value_loss / grad_norm / value_target` 的 receipt
（`online_ttt.py:102` `validate_learn_receipt` 会检查 `finite_loss / finite_gradients / base_parameters_frozen` 等标志）。

价值语义是**分类**而非标量：`mixed_objective.py:72-82` 把 `return ∈ [-max_distance, -1]` 映射为
`value_class = -value-1`，用 64 个 bin 的 categorical 价值头（第 04 章），并强调 **“never clip or relabel”**。

### 11.3 `nanoproof/nanoproof/rl.py`（token-mean CE + unlikelihood）

**【注 RL.12.3｜R-RL.12.3】（`nanoproof/rl.py`：token-mean CE + unlikelihood）**

这是一个**离线/回放式**的 RL 循环（不是在线 TTT）。核心损失在 `:983-1003`：

正样本（成功 tactic）用**加权 token-mean 交叉熵**（交叉熵见【定义 1.2.4】，其对 logits 的梯度见【命题 1.2.9】）：

$$
\mathcal{L}_{\text{pos}}
= \frac{1}{\sum_t \text{mask}_{b,t}} \sum_{b,t} \text{CE}_{b,t}\cdot w_b \cdot \text{mask}_{b,t},
$$

其中 $w_b$ 是 value 样本的 `value_weight`（`:983-985`）。

负样本（失败的 tactic）用 **unlikelihood** 损失：先由 CE 还原概率 $p = e^{-\text{CE}}$（截断保证 $\log(1-p)$ 有限），

$$
\mathcal{L}_{\text{neg}} = \frac{1}{N_{\text{neg}}}\sum_{b \in \text{neg}} \frac{1}{|b|}\sum_{t} \big[-\log(1 - p_{b,t})\big].
$$

总损失

$$
\mathcal{L} = \mathcal{L}_{\text{pos}} + \lambda_{\text{ul}}\, \mathcal{L}_{\text{neg}}.
$$

**直觉**：正样本做 next-token 模仿，负样本**降低**生成失败 tactic 的概率（不是硬性抑制，而是“别再说它”）。
`NegativeBuffer`（`:457`）维护失败 tactic 的 FIFO 窗口，`ReplayBuffer`（`:410`）回放成功转移，
`Matchmaker`（`:289`）按题目难度/证明大小分配搜索预算。这就是“**难度课程 + 正负经验回放**”的工程实现。

---

## 12. 小结与思考题

**【注 RL.13.1｜R-RL.13.1】（小结）**

**一句话总结**：策略梯度是“用回报给动作概率加权”；PPO 用 clip 控制步长；GRPO 用组内相对优势替代 critic；
RLVR 用 Lean 内核提供干净奖励；KL 把新策略拴在 SFT 参考上；RTTT 把这一切搬进单次搜索的在线循环。

**【练习 RL.13.2｜Ex-RL.13.2】（思考题）**

思考题（可跑 `N-13`/`N-14` 验证）：

1. 当一组 GRPO 样本奖励全为 1 时，$\hat A_i$ 是多少？梯度会怎样？代码里应如何检测并丢弃这种组？
2. `/ttt_step` 的 KL 项用 $(\log p - \log p_{\text{old}})^2$，请写出它对 $\theta$ 的梯度，并解释它与 k2 估计器的关系。
3. 为什么 `nanoproof/rl.py` 的负样本用 unlikelihood 而不是直接最大化负样本的 CE 的相反数（即 $-\text{CE}$）？二者梯度差别在哪？
4. 从信息几何角度说明：为什么 $D_{\mathrm{KL}}$ 的 Hessian 是 Fisher 信息矩阵，从而自然梯度是 KL 约束下最速下降方向？

---

## 参考

**【注 RL.14.1｜R-RL.14.1】（参考与源码索引）**

- `RLVR-OPSD-参考资料/02-RLVR-数学理论详解.md`（策略梯度/PPO/GRPO/DAPO/GSPO/KL 估计器）
- `RLVR-OPSD-参考资料/06-Lean证明-RL与蒸馏训练资料.md`、`09-面向Lean证明模型的训练方案建议.md`
- `ttt-参考资料/01-问题定义与框架.md`、`04-测试时适应的理论与优化.md`
- `信息几何理论参考资料/05-Bregman散度与投影定理.md`、`07-自然梯度与Fisher效率.md`
- AlphaProof 源码：`app/policy_server.py`、`nanoproof/nanoproof/rl.py`、`gpu_runtime/*`
