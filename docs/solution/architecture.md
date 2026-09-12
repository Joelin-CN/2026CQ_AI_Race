# 解决方案设计（讨论结论存档）

> 2026-09-13 讨论、仓库初始化时落档。本文是方案的唯一权威记录：
> 后续实现与偏离此处决策的改动，请更新本文或在 `docs/decisions/` 补 ADR。

## 1. 赛题

**全国通用人工智能挑战赛 · 高校组「通用智能体任务挑战赛」**（通知见
`docs/problem/`，官网 <https://nagic.bigai.ai/>）。

- 平台：TongSIM（通界平台）仿真，评测环境感知、逻辑推理、长程规划与具身交互。
- 初赛五项任务：整理房间（tidyroom）、分类计数（counting）、NPC 对话（npc）、
  瑞文测试（raven）、拼图（jigsaw）。
- 决赛：五项任务串联为综合任务，重点考查长程交互式推理 + 技术答辩。
- 流程：赛题软件内置 5 项任务 → 智能体作答 → 导出结果数据文件 → 官网自动评分。

### 时间线（很紧）

| 节点 | 时间 |
|---|---|
| 报名与初赛结果文件提交截止 | **2026-09-20** |
| 初赛评审 | 09-20 ~ 09-25 |
| 晋级公示 / 决赛赛题发布 | 09-25 ~ 10-01 |
| 决赛与颁奖（含技术答辩） | 10 月下旬 |

## 2. 官方 baseline 要点（事实，来自源码阅读）

- 纯 Python ≥3.12 + uv，无 ROS。两条 gRPC 链路：
  - **任务系统** `127.0.0.1:50051`（`AgentBase`：connect → 领题 → 收动作 → 评分）；
  - **TongSim 仿真** `127.0.0.1:50060`（感知 / 移动 / 操作 / 对话）。
- 核心回路（`VLMAgent.run_step`）：第一视角复合图（左 RGB + 右带数字编号的
  语义分割图）→ 拼 prompt → VLM → 解析**单动作 JSON** → TongSim 执行 →
  结果写入历史（最近 15 条 + 动作历史 10 条回填 prompt）。
- **关键杠杆**：感知 API 除图像外还返回结构化 `objects` 元数据
  （`object_id`/颜色/形状/世界坐标 `place_location`/`world_aabb`），
  `object_id` 与分割图上的编号一致。这是确定性计算的入口，不必纯靠 VLM 看图。
- 动作空间：25 个动作（`move_and_take_object`、`move_to_location`、
  `speak_to_npc`、`submit_answer`、`finish_task` 等）。
- 已内置 ~60 个 VLM 配置类（GPT/Claude/Gemini/Qwen/GLM/Doubao…，默认
  `VLMGPT5Config` = gpt-5.1 via Azure），以及瑞文测试的 ResNet18+MLP 专家模型
  （`vlm_agent/skills/`，含 pth 权重与硬编码裁剪坐标）。
- 官方明示："不强制要求必须使用大模型，可以使用任意算法实现任务"
  —— 官方自己给 raven 配了非 LLM 专家模型，混合方案是被鼓励的方向。

## 3. 决策记录

| # | 日期 | 决策 | 被否备选 | 理由 |
|---|---|---|---|---|
| D1 | 2026-09-13 | **独立自研包 + 官方库 editable 复用**：官方代码原样保留在 `official_source/`，我们只写 `src/cqairace`，import 复用其协议与基础设施 | fork 直改；完全独立实现 | 官方可能发决赛更新/修复，保持可同步；协议对齐零风险；一周初赛窗口不允许重造轮子 |
| D2 | 2026-09-13 | **多模型按任务混合路由**：provider 无关路由表（config `[routing]` 段），对话用强语言模型、视觉用强视觉模型、可确定性解决的不用 VLM | 全程单一模型 | 五任务能力侧重不同；官方客户端矩阵现成；便于 A/B 与成本控制 |
| D3 | 2026-09-13 | **最小骨架起步**：本次只初始化目录 + 文档 + 包配置，逻辑代码按阶段逐个落地 | 一次性写全骨架代码 | 先跑通系统采基线，再按任务迭代；避免无实测数据时凭空写逻辑 |

## 4. 架构分层

```
┌───────────────── 自研大脑层  src/cqairace（我们写）─────────────────┐
│ main.py      自研 runner（加载 config → 构建 agent → 对接任务系统）  │
│ agent.py     CQAgent(VLMAgent)：技能接管点/感知增强/场景记忆挂载处   │
│ routing.py   任务类型 → VLM 配置路由表（[routing] 配置驱动）         │
│ prompts/     任务专属 prompt 片段（<task_type>.txt）                │
│ skills/      确定性技能（规划：counting 聚合、jigsaw 匹配等）        │
└───────────────┬────────────────────────────────────────────────────┘
                │ 继承 / import（editable 挂载，不修改官方代码）
┌───────────────▼──────── 官方基础设施层 arenaagent（不改）───────────┐
│ AgentBase        ↔ 任务系统 gRPC :50051（生命周期/领题/交动作/评分） │
│ VLMAgent         ReAct 回路：复合图 → prompt → VLM → 单动作 JSON    │
│ TongSimGrpcClient↔ 仿真 gRPC :50060（感知/移动/操作/对话，心跳保活） │
│ vlm_config       ~60 个 VLM 配置类 + ClientFactory 多厂商客户端     │
│ raven_skill      ResNet18+MLP 瑞文专家模型（含权重）                │
└────────────────────────────────────────────────────────────────────┘
```

## 5. 核心策略：混合决策 + 感知双通道

1. **感知双通道**：图像走 VLM，结构化 `objects` 元数据（含世界坐标）走
   确定性计算。能算的不让模型猜。
2. **场景记忆**：跨步骤累积场景地图（已探索方位、已放置物体），弥补单帧
   FOV 120° 的信息缺口 —— tidyroom / counting 都需要全局信息。
3. **确定性优先**：能用规则/算法精确解决的动作不消耗 VLM 调用；
   VLM 负责语义理解（分类、对话）与兜底。
4. **失败反馈闭环**：沿用官方 action_histories 机制，把失败原因写回 prompt。

## 6. 分任务路线

| 任务 | 主要路线 | 关键杠杆 | 状态 |
|---|---|---|---|
| counting 分类计数 | 确定性聚合：转身 360° 扫描 + `objects` 按 object_id/世界坐标去重 + 颜色/形状统计 | objects 元数据 | 规划 |
| raven 瑞文测试 | 复用官方 ResNet18+MLP 专家模型 | 权重已带；注意 `/tmp` 路径在 Windows 的兼容问题 | 规划 |
| npc 对话 | 强 LLM 多轮对话收集线索，`submit_answer` 提交 | `speak_to_npc` 返回 `npc_reply`+`hints` 注入下一轮 prompt（官方已有） | 规划 |
| tidyroom 整理房间 | 规则化空间规划（`place_location`/`world_aabb`/高度约束）+ VLM 语义分类"什么放哪" | 官方 task_prompt 提示按包围盒与放置高度规划 | 规划 |
| jigsaw 拼图 | 分割图形状匹配做确定性放置 + VLM 兜底 | 官方提示：拼图块 X=837、观察点 (750,191)，场景固定 | 规划 |
| （决赛）长程串联 | 任务分解 planner + 全局记忆 + 阶段状态机 | 复用上述五技能 | 预留 |

## 7. 已知风险与坑（提前记录）

- **raven_skill 硬编码 `/tmp/raven_input_images`**：Windows 不兼容，
  复用时需在我们层处理（不改官方文件，运行前设置/软链或拷贝逻辑）。
- **决赛 baseline 未发布**：官方 usage_guide 引用 `final_baseline_agent`，
  但当前 repo 只有初赛版；决赛赛题发布后需同步官方仓库（`official_source`
  重新 pull 即可，这正是 D1 的价值）。
- **pb2 代码不入官方库**：`arenaagent/generated/` 需手动跑官方
  `scripts/generate_pb2.py` 生成，README「快速开始」已写明。
- **评分细则在 PDF 手册**（网盘分发，系统下载完成后）：`score_msg.proto`
  与 `TaskDifficultyMsg` 表明分数含难度系数，具体公式待手册确认。
- **赛题系统尚未下载完成**：端到端联调 blocked，先用官方文档与协议文件
  做离线设计；日志回放（`logs/prompts/`）是系统就绪前后的主要调试手段。

## 8. 实施路线（阶段划分）

| 阶段 | 内容 | 退出标准 |
|---|---|---|
| P0 ✅ | 仓库骨架 + 本设计文档 + git/GitHub | 本文档合入 main |
| P1 | 系统下载完成后：跑通官方 `preliminary_baseline_agent` 采基线 | 五任务各产出一份日志 + 结果文件 |
| P2 | 落地 `main.py`/`agent.py`/`routing.py` 最小闭环（行为=官方基线） | 我们的 runner 跑通五任务，成绩≥基线 |
| P3 | counting + raven 确定性技能 | 对应任务得分显著超 VLM 直跑 |
| P4 | npc 对话强化 + tidyroom / jigsaw 混合策略 | 初赛五任务总分达标，提交 |
| P5 | 决赛：长程 planner + 记忆，按决赛手册调整 | 决赛测试 + 技术答辩材料 |
