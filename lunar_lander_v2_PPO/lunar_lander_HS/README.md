# lunar_lander_HS — Heuristic System for LunarLander-v2

用 **HL（Heuristic Learning）** 范式替代 Deep RL PPO：策略为 `heuristic_lunar_lander.py` 中的可解释控制律 + 参数向量；由 **CEM** 与 **hl_agent**（程序化 Coding Agent）迭代更新 HS。

## 运行

```bash
cd lunar_lander_v2_PPO/lunar_lander_HS
pip install numpy gym[box2d] pillow
python hl_train.py
python test_hs.py --episodes 20
python make_gif_hs.py
```

## 目录

| 路径 | 说明 |
|------|------|
| `heuristic_lunar_lander.py` | HS 核心策略 |
| `hl_train.py` | HL 主循环（CEM + Agent patch） |
| `hl_eval.py` | 环境评估 |
| `hl_memory.py` | trials 压缩与快照 |
| `hl_agent.py` | 根据回报诊断打补丁 |
| `HS_checkpoints/` | 最佳参数 JSON |
| `HL_logs/` | jsonl 试验与 diff |
| `HL_memory/` | 压缩归档与策略历史 |

## 对标 PPO

LunarLander-v2 **solved** 约 **200** 平均回报；见 `PPO_logs` 与 `Deep_RL_PPO.md`。
