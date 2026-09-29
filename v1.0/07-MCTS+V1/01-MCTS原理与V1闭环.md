# 07 · MCTS 原理与 V1 闭环

> 本篇把“策略网络 + 价值头”变成真正的**搜索**：AlphaZero 式 PUCT 在 OR/AND 证明树上选择、展开、回传。
> 配套代码：[`code/mcts.py`](code/mcts.py)（`F-b07-mcts`）、[`code/v1_tiny/`](code/v1_tiny/)（`F-b07-policy_value_eval` / `F-b07-run_demo` / `F-b07-self_play`）。
> 实验：notebooks `N15_极简MCTS_PUCT`（`N-15`）、`N16_AlphaProof_V1_闭环tiny`（`N-16`）。
>
> **【文档｜DOC-MCTS】**（doccode = `MCTS`）｜编号与 Tag 规范见《00-风格与编号规范》。
>
> **前置（预备篇）**：见【定义 PY3.1.1】（虚拟环境）、【定义 PY6.3.1】（生态库速览）。

---

## 0. 一句话与对照

**【注 MCTS.1.1｜R-MCTS.1.1】（一句话与对照）**

一句话：**MCTS 是跑在 CPU 上的搜索内核，GPU 只在“展开一个叶子”时被调用一次**；每轮 = 下降（PUCT argmax）→ 叶子展开（一次策略/价值前向）→ 折扣回传，直到根 solved 或预算耗尽。

| 层 | 对照位置 |
|---|---|
| Lean 搜索内核 | `reap-upstream/Tactic/TreeSearch.lean`：`monteCarloTreeSearch:470`、`reapMCTSStep:439`、`computePUCTScores:312`、`shouldProgressiveSample:333`、`updateNode/backpropValueTowardsMin:386-412` |
| GPU 边界 | `reap-upstream/Tactic/Generator.lean:149` `generatePolicyValue`（3 个 HTTP 端点） |
| 编排 | `python-driver/v1_run.py`（BatchSolver） |
| 在线训练闭环 | `v1-result/.../cpu_runtime/online_ttt.py` `run_online` |
| 逐轮手算示例 | `/home/a/文档/mcts-theory-learning/MCTS算法流程/V1-MCTS完整运行流程与算例.md` |

---

## 1. OR / AND 树与 focus 子目标

**【定义 MCTS.2.1｜D-MCTS.2.1】（OR / AND 节点）**

- **OR 节点**：多选一（tactic 二选一），只要一个孩子 solved 就 solved（$\exists$，`any`）。
- **AND 节点**：全都要（一条 tactic 分裂出多个子目标），全部 solved 才 solved（$\forall$，`all`）。

**【代码 MCTS.2.2｜Cd-MCTS.2.2】（focus 孩子构造）**

`nanoproof/search.py:727-751`：tactic 成功且 `len(new_branches) > 1` 时创建 AND 节点，并给每个子目标挂
`prior = 1/len` 的 **focus 孩子**（伪动作 `i`，`reward=0`，边长代价 0），使搜索能专攻某个子目标：

```python
child = Node(parent=node, action=tactic, prior=p, state=new_branches,
             to_play=Player.AND if len(new_branches) > 1 else Player.OR, reward=-1.0)
if len(new_branches) > 1:
    for i, branch in enumerate(new_branches):
        child.children[i] = Node(parent=child, action=i, prior=1/len(new_branches),
                                 state=[branch], to_play=Player.OR, reward=0.0)
```

**【注 MCTS.2.3｜R-MCTS.2.3】（solved 判定）**

`Node.calculate_solved:134-148`：OR 取 any、AND 取 all、叶子（`len(state)==0`）为 True。

---

## 2. PUCT 选择

**【定义 MCTS.3.1｜D-MCTS.3.1】（PUCT 选择分数）**

$$
\mathrm{score}(s,a) = Q(s,a) + c(N)\,\frac{p_a}{\sum_b p_b}\,\frac{\sqrt{N}}{N(s,a)+1},
\qquad
c(N) = c_{\mathrm{init}} + \ln\!\left(\frac{N + c_{\mathrm{base}} + 1}{c_{\mathrm{base}}}\right).
$$

$N$ 父访问数、$N(s,a)$ 边访问数、$p_a$ 先验（LLM token logprob 和，即 softmax 分布下的对数概率，见【定义 1.2.1】；温度 $\tau=50$），
$c_{\mathrm{init}}=0.001$、$c_{\mathrm{base}}=3200$。nanoproof 等价实现 `search.py:616-640` `ucb_score`。
**【注 MCTS.3.2｜R-MCTS.3.2】（未访问值与 AND 选择分数）**

未访问孩子的值：论文 $V(s,a)=\hat V_{\text{net}}(s)-c_{\text{pen}}$；V1 代码直接用 0。
AND 节点选择分数取 $1-\text{valueScore}$，已 solved 孩子置 $-\infty$（`search.py:634-639`）。

---

## 3. Q 变换、$\gamma$ 折扣与回传

**【定义 MCTS.4.1｜D-MCTS.4.1】（Q 变换与 $\gamma$ 折扣）**

V1 把值头输出**取负**（越小越难，价值头定义见《04-价值头原理》）：$v = \text{child.value} - \text{stepCost}$，

$$
\text{valueScore} = \gamma^{-1-v}, \qquad \gamma = 0.99.
$$

- OR：$Q=\text{valueScore}$；AND：$Q=1-\text{valueScore}$（已解孩子 $-\infty$）；
- 终局解出：$\gamma^0=1$，每多一步多远一个 $\gamma$ → **短证明优先**；
- 普通 tactic 边 `stepCost=1`，focus 边 `stepCost=0`。

**【注 MCTS.4.2｜R-MCTS.4.2】（回传规则）**

回传（`search.py:861-887`）：向上 `value_sum += value; visit_count += 1`；父是 AND 用
`backprop_value_towards_min`（未解且有访问孩子的最小值，从 1.0 起），否则 `value = node.reward + value`。

---

## 4. Progressive sampling、Dirichlet、循环检测

**【定义 MCTS.5.1｜D-MCTS.5.1】（progressive sampling 条件）**

$$
\text{命中} \iff \text{node 是 OR}\;\wedge\; n_{\mathrm{eval}} \le C\cdot N^{\alpha},
\qquad C=0.01,\ \alpha=0.6
$$

命中时不选已有孩子，而是对同一节点**再调一次策略网络**并入新 tactic（去重合并先验）。
默认参数下 $C\cdot N^{0.6}$ 要 $N\approx2154$ 才到 1，几乎不触发；论文值 $C=1.0,\alpha=0.5$。

**【注 MCTS.5.2｜R-MCTS.5.2】（Dirichlet 旁注与循环检测）**

**Dirichlet 旁注**：AlphaZero 在根加 $p'_a=(1-\epsilon)p_a+\epsilon\,\mathrm{Dir}(\alpha)$；
**V1 / nanoproof 均未实现**（无 Dirichlet / noise / virtual loss），选择是确定性 argmax，
随机性只在“策略候选生成”。`is_cycling:644-653` 把回到祖先状态的 tactic 判为 `cycle` 丢弃。

---

## 5. 单轮迭代伪代码（`search.py:475-594`）

**【代码 MCTS.6.1｜Cd-MCTS.6.1】（单轮迭代伪代码）**

```
for i in range(num_simulations):
    node, path = root, [root]
    while node.expanded and node.children and not progressive_sample(node):
        node = select_child(node)              # PUCT argmax
        path.append(node)
    tactics, logprobs, value = model.sample_tactic(node.state)   # 一次 GPU 前向
    value = -value
    expand_node(node, tactics, logprobs, temperature)            # 在 Lean 里真实执行
    backpropagate(path, value, config)         # γ 折扣回传
    if root.is_solved: break
```

**【注 MCTS.6.2｜R-MCTS.6.2】（expand_node 与复验）**

`expand_node:658-755` 逐条调 `branch.try_apply_tactic`，区分 `error`/`cycle`/`success`；
成功且子目标为空 → terminal，标记孩子和父 solved。解出后还要 `replaySolvedNode` + `checkProof` 复验。

---

## 6. 与 Lean `reap` 引擎的对应

**【代码 MCTS.7.1｜Cd-MCTS.7.1】（与 Lean `reap` 引擎的对应）**

```
v1_run.py → Lean: reapMCTS (Tactic/Syntax.lean:233)
          → runMCTS → monteCarloTreeSearch (:470)
          → reapMCTSStep (:439) ── PUCT computePUCTScores (:312)
                                ── progressive shouldProgressiveSample (:333)
                                ── backup updateEdge/updateNode/backpropValueTowardsMin (:386-412)
          → 叶子 visitNode → evalPolicyValue = generatePolicyValue (Tactic/Generator.lean:149)
          → 三端点: /premises, /v1/chat/completions (n=6,temp=0.99,logprobs), /value
```

一次展开：`/premises` 取 16 条引理；`/v1/chat/completions` 取 6 个候选 tactic 及 log 先验；`/value` 取 1 个分数（Lean 取负）。

---

## 7. V1 训练侧闭环

**【算法 MCTS.8.1｜A-MCTS.8.1】（BatchSolver）**

`app/v1_run.py`（BatchSolver）：每题生成临时 `.lean`（`theorem ... := by reapMCTS` + `%%TASK_<id>_DONE%%`），
`podman run ... lake env lean`，以是否打印 `DONE` 判定；checkpoint `state/<id>.done`（`--continue` 跳过）。
`DONE` 只表示 Lean 跑完，是否真证明成功仍需内核复验。

**【算法 MCTS.8.2｜A-MCTS.8.2】（online_ttt `run_online`）**

`cpu_runtime/online_ttt.py` `run_online`：起 `lake env lean` 子进程（注入 `REAP_SESSION_ID / REAP_POLICY_ENDPOINT /
REAP_VALUE_ENDPOINT`）→ Lean 把 observer 写 `observer.jsonl` → `OnlineCoordinator.accept` 聚合成 learn event
→ `POST /sessions/{sid}/learn/v1`（`GpuRuntime.learn`，`runtime.py:425`）执行一步 TTT → 新策略服务后续展开
（`post_update_generations`），结果写 `online-result.json`。receipt 字段见 `online_ttt.py:102`。

**【注 MCTS.8.3｜R-MCTS.8.3】（V1 闭环全景）**

全景：

```
   GPU: πθ 采样 n 条 tactic(带 logprob) + Vφ 输出 value(取负)
        ▲                                          │ HTTP
        └────────────── 3 个端点 ──────────────────┘
   Lean: PUCT 下降 → expand_node(真实执行 tactic) → γ 折扣回传 → 根 solved?
        │ observer.jsonl
        ▼
   GpuRuntime.learn → /ttt_step: -r(logp-logp_old)+β_KL(logp-logp_old)^2+c_v·MSE
        │ 更新参数
        └────────────► 服务后续搜索（闭环闭合）
```

---

## 8. 本目录可运行代码

**【注 MCTS.9.1｜R-MCTS.9.1】（本目录可运行代码）**

- [`code/mcts.py`](code/mcts.py)：极简 PUCT/OR-AND；`evaluator(state) -> (priors, value)` 是**价值头占位接口**；
  内含复刻流程文档 §4 的 toy 环境（`constructor` 分裂 `P∧Q`），`python3 mcts.py` 直接跑。
- [`code/v1_tiny/policy_value_eval.py`](code/v1_tiny/policy_value_eval.py)：tiny 策略+价值网络，输出 tactic 先验与剩余步数。
- [`code/v1_tiny/self_play.py`](code/v1_tiny/self_play.py)：用 MCTS 解题并收集 `(state, tactic, value_target)` 经验。
- [`code/v1_tiny/run_demo.py`](code/v1_tiny/run_demo.py)：合成可验证任务上端到端训练+搜索，打印闭环轨迹。

---

## 9. 小结与思考题

**【注 MCTS.10.1｜R-MCTS.10.1】（小结）**

- OR=选择取 any，AND=合取取 all；focus 让孩子让 AND 专攻瓶颈。
- PUCT = Q + 先验×访问奖励；$c(N)$ 对数增长，早期几乎纯按 Q 选。
- 备份用 $\gamma$ 折扣 + AND 的 min，偏向短证明与最难分支。
- GPU 只在叶子展开时调用一次；V1 无 Dirichlet、无 virtual loss、确定性 argmax。
- 闭环：搜索产数据 → observer → learn event → 在线更新 → 后续生成用新策略。

**【练习 MCTS.10.2｜Ex-MCTS.10.2】（思考题）**

思考题：

1. 流程文档迭代 2 时 $c\approx0.001625$，为什么此时几乎忽略探索项？$N$ 多大时探索项才与 Q 同量级？
2. 把 AND 的 `backprop_value_towards_min` 改成包含已解孩子（值为 1）会怎样？
3. `shouldProgressiveSample` 的 $C=0.01$ 意味着什么？调到 $C=1.0$ 后行为如何变（结合 N15）？
4. 为什么 V1 不加 root Dirichlet 也能工作？候选采样随机性起了什么作用？
5. `online-result.json` 的 `post_update_generations=0` 时闭环哪里断了？

---

## 10. 参考

**【注 MCTS.11.1｜R-MCTS.11.1】（参考与源码索引）**

- `mcts-theory-learning/MCTS算法流程/V1-MCTS完整运行流程与算例.md`
- `mcts-theory-learning/5-mcts-rl.md`、`6-alpha-proof-mcts-rl.md`、`8-6-q-transform-reason-1.md`、`9-mainrl-ttrl-llm-value-head.md`
- 源码：`nanoproof/nanoproof/search.py`、`app/v1_run.py`、`v1-result/reproduction/code/src/cpu_runtime/online_ttt.py`
