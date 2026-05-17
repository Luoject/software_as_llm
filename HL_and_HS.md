# HL（Heuristic Learning）与 HS（Heuristic System）详解

> 基于 [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/)（翁家翌，2026）及本仓库 `learning-beyond-gradients` 代码与实验产物整理。

---

## 1. 核心概念

### 1.1 问题背景

传统 **Continual Learning（持续学习）** 在神经网络上面临 **灾难性遗忘**：新任务梯度更新会覆盖旧能力。Deep RL 通过固定 reward 与梯度下降更新 **神经网络参数**，样本效率低、可解释性差，且难以把旧能力固化为可回归的测试。

翁家翌提出的 **HL（Heuristic Learning，启发式学习）** 范式将「被优化的对象」从 **权重** 换成 **可维护的软件系统**：用 **纯软件工程系统 HS** 充当策略与记忆载体，由 **Coding Agent（大模型编程智能体）** 阅读环境反馈、日志、测试、回放等 context，**直接改代码** 而非反向传播，通过多轮迭代使 HS 达到可与 Deep RL 媲美的任务表现。

### 1.2 定义

| 术语 | 含义 |
|------|------|
| **HL** | 学习与更新的 **过程**：吸收反馈 → 修改 HS（策略、检测器、测试、配置、memory）→ 再运行 → 记录 trials |
| **HS** | 被 HL 长期维护的 **Heuristic System**：不仅是单个 `policy.py`，至少包含程序策略、状态表示、反馈入口、实验记录（JSONL/CSV）、回放/测试、memory、以及由 Agent 执行的更新管线 |

一句话：**HS 是「软件形态的策略+基础设施」；HL 是「用 Agent 养大 HS」的闭环。**

---

## 2. 代码仓库中如何实现 HL / HS

本仓库 `learning-beyond-gradients` 中的典型 HS 结构如下（以 `mujoco/halfcheetah/heuristic_halfcheetah_v5.py` 为例）：

```
heuristic_<task>.py     # 程序策略（NumPy 规则/PD/CEM 参数向量，非神经网络）
*_trials.jsonl          # 每轮搜索/改动的结构化记录
*_trials_summary.csv    # 压缩后的试验摘要
*_log.md                # 人类可读的迭代日志
测试/回放脚本、prompt 模板（Atari57 批量 Codex 运行）
```

### 2.1 闭环与 Deep RL 的对应关系

| 环节 | Deep RL（如 PPO） | HL（本仓库实践） |
|------|-------------------|------------------|
| **Policy** | `ActorCritic` 网络参数 θ | Python 函数/规则/参数向量（如 PD+步态、Pong 几何启发式） |
| **State** | `env.observation` 张量 | 显式变量：关节角、球坐标、RAM 字节、缓存检测器等 |
| **Action** | `π(a\|s)` 采样 | `policy.act(obs)` 执行分支逻辑 |
| **Feedback** | 标量 reward + done | reward + **日志、失败片段、回归测试、视频、diff、人类批注** |
| **Update** | `loss.backward()` + Adam | **Coding Agent 编辑源码/配置**，或脚本内 **CEM** 调参（仍属 HS 内工程搜索） |
| **Memory** | Rollout buffer / replay | **trials.jsonl、summary、版本 diff、压缩摘要** |

### 2.2 代表性 HS 组件（代码级）

1. **程序策略**  
   - `atari/pong/heuristic_pong.py`：从像素/RAM 检测球与挡板，预测拦截点。  
   - `mujoco/halfcheetah/heuristic_halfcheetah_v5.py`：Fourier 步态 + 反射项 + 可选 `--search` CEM。  
   - `atari/breakout/heuristic_breakout.py`：状态机 + 落点预测 + 卡死检测。

2. **反馈入口**  
   - 环境 `reward`；CEM 一轮的 `mean_return`、`std`；Montezuma 的宏动作失败原因写入 jsonl。

3. **实验记录**  
   - `append_jsonl()` 写入每轮 `SearchIteration`（见 HalfCheetah `search()`）。  
   - Atari57：`heuristic_*_trials.jsonl` + `*_summary.csv`。

4. **更新机制**  
   - **人工/Agent 改代码**：文章主路径（Codex 读日志改 policy）。  
   - **CEM 自动调参**：在固定程序骨架上搜索参数向量（仍属 HS 演进，Agent 可随后「压缩」为更简规则）。

5. **压缩历史**  
   - 文章强调：只增不删的 HS 会腐化；需把局部 patch 折回更简单表示。仓库中用 CSV summary、提取 `*_min_policy.py`（Ant）体现这一点。

---

## 3. 与传统大模型（LLM）的区别与共同点

### 3.1 区别

| 维度 | 传统 LLM（预训练+推理） | HL 中的角色 |
|------|-------------------------|-------------|
| **推理时是否改权重** | 否（仅 context 内学习） | Agent **改外部代码/配置**，不改 LLM 权重 |
| **策略载体** | 隐式在参数里 | **显式 HS 代码** |
| **输出** | token 序列 | 可执行程序 + 测试 + 日志 |
| **可验证性** | 难对单步决策做 golden test | 可对 `policy.act` 写单元测试与固定 seed 回放 |
| **样本效率** | 需海量 pretrain | 一次好的代码补丁可 **跳跃** 到新策略 |

### 3.2 共同点

- **都依赖大模型能力**：HL 的更新者通常是 Coding Agent（GPT/Codex 等），负责读 context、写 patch、跑测试。  
- **都有上下文与记忆**：LLM 用 context window；HS 用 jsonl、summary、diff。  
- **都可做 Agent 编排**：HL 可把 LLM 放在 System 2，HS 放在 System 1（文章机器人分层示意）。

**关键区分**：传统 LLM 是 **「一个模型搞定推理」**；HL 是 **「LLM 养 HS，HS 才是落地策略」**——类似「大模型作编译器/维护者，软件作运行时」。

---

## 4. 与 Deep RL 的区别与共同点

### 4.1 区别

| 维度 | Deep RL | HL |
|------|---------|-----|
| 策略表示 | 神经网络 | 代码/规则/可解释控制器 |
| 更新 | 梯度、固定 reward | Agent 改代码；reward 可与测试、日志混合 |
| 遗忘 | 参数覆盖 | 规则冲突、测试漏洞；靠 **回归测试** 缓解 |
| 感知极限 | 端到端可学复杂视觉 | 纯 Python 难以做 ImageNet 级感知 |
| 维护成本 | 调超参、训很久 | 过去 heuristic 难维护；**Agent 降低维护成本** |

### 4.2 共同点

- **MDP 闭环**：状态、动作、转移、奖励一致。  
- **目标**：最大化累积回报或任务成功率。  
- **可并列基准**：仓库 Atari57 在固定步数下比较 HNS 与 PPO 曲线；本工程 `lunar_lander_HS` 以 LunarLander 平均回报对标 PPO。

### 4.3 文章中的实证结论（摘要）

- Breakout、Ant、HalfCheetah、VizDoom、Atari57 等任务上，HS 经 HL 迭代可达 **与常见 Deep RL 同量级** 的回报。  
- 优势：**可解释、样本效率高、可回归**；劣势：**耦合复杂度** 受 Agent 能力与模块化限制（Montezuma 宏动作反例）。

---

## 5. 在 LLM Agent 中的应用场景

### 5.1 自迭代优化 Agent Prompt

- **做法**：将 Prompt 模板、工具说明、few-shot 示例视为 HS 的一部分；用任务成功率、单元测试、人工评分作为 feedback；Agent 每轮修改 prompt 文件并跑评测。  
- **与 RLHF 区别**：不更新 base model，更新的是 **可版本化的 prompt 资产**。  
- **风险**：Prompt 堆叠导致耦合复杂度上升 → 需 **压缩**（合并规则、删冗余示例、A/B 摘要进 memory）。

### 5.2 自迭代优化 Agent Harness（工具链与编排）

- **做法**：Harness（重试逻辑、子 Agent 调度、日志格式、MCP 配置）即 HS；失败 trace、diff、CI 结果作为 feedback。  
- **价值**：Harness 比权重更易 **diff、回滚、回归**；适合「长运行 Agent 系统」的持续改进。  
- **与 HL 一致点**：`feature request → 写代码 → test → 下一轮` 与文章中的自动闭环同构。

### 5.3 与 Neural + HL 混合（文章展望）

- HL 在线产生经验 → 筛选为训练数据 → **周期性更新小网络**（System 1 感知 + HL 规则 + LLM System 2）。  
- 适合机器人、游戏 AI 等 **需要可解释安全层** 的场景。

### 5.4 不适用场景

- 需要端到端高维感知且无法手写检测器的任务（如 ImageNet 纯 heuristic）。  
- 反馈极其稀疏且无法做回放/测试的环境（需宏动作、搜索层 HS，见 Montezuma）。

---

## 6. 本 monorepo 中的落点

| 目录 | 范式 | 说明 |
|------|------|------|
| `learning-beyond-gradients/` | HL / HS 参考实现与论文产物 | 多环境 heuristic + trials |
| `lunar_lander_v2_PPO/` | Deep RL（PPO） | 神经网络基线 |
| `lunar_lander_v2_PPO/lunar_lander_HS/` | HL / HS | 月球登陆器启发式系统 + CEM/Agent 迭代 |

---

## 7. 参考文献

- 博文：[Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/)  
- 代码：[learning-beyond-gradients](https://github.com/Trinkle23897/learning-beyond-gradients)  
- BibTeX 见该仓库 `README.md` 中 `@misc{weng2026learning_beyond_gradients,...}`
