# 解决方案设计（讨论结论存档）

> 2026-09-13 讨论、仓库初始化时落档。本文是方案的唯一权威记录：
> 后续实现与偏离此处决策的改动，请更新本文或在 `docs/decisions/` 补 ADR。
> v2 赛题系统实测行为（一题一提交/一题一连接等）见
> [v2-system-notes.md](v2-system-notes.md)。

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

### 评分规则与题量（官方《初赛系统使用指南》PDF v1.0 + 2026-09-12 更新说明）

**精确公式**（从 PDF 第 11 页公式图 OCR 提取，2026-09-13 验证）：

- 整理房间 / 拼图：`得分 = 0.8 × 任务完成度 + 0.2 × 剩余时间/总时间`
- 分类计数 / NPC 对话 / 瑞文测试：`得分 = 剩余时间/总时间`（答对才有分，越快越高）
- 每题满分 100；任务分 = 该任务全部题目的平均分；**总成绩 = 五任务平均分（每任务权重 20%）**
- 每题限时 400 秒；超时自动结算：完成度部分（×0.8）保留，时间部分归零
- 每任务取历史有效提交最好成绩，可反复 test 刷分
- 注意：计时起点疑似为 agent 接入后的某个时刻（非任务系统启动时刻），
  train 实测成绩与朴素公式计算略有出入，精确计时基准待 P2 实测校准

| 任务 | 每轮题量 | 单题满分占总分权重 |
|---|---|---|
| 整理房间 tidyroom | 1 题 | 20% |
| 拼图 jigsaw | 1 题 | 20% |
| NPC 对话 npc | 5 题 | 每题 4% |
| 分类计数 counting | 10 题 | 每题 2% |
| 瑞文测试 raven | 10 题 | 每题 2% |

- 提交链路：`start_test.bat test --task <id>` 产出加密 `result_{Task}.bin`（五任务齐全）→
  `start_test.bat package-results` 打包 `result_package.bin` → 上传官网。
  `train` 模式产出的明文 `eval_res.json` 仅可调试、不可提交。
- ⚠️ 版本注意：release 包内《使用指南说明.md》是旧版（"每任务 3 题"），以 PDF 版
  （1/1/5/10/10 题，与 run_times 说明一致）为准；最终以 train 模式
  `_get_num_subjects_from_task()` 实测为准。
- ⚠️ 运维注意：Agent 运行出错后**必须重启赛题系统**，否则结果作废——稳定性 = 分数
  （counting/raven 需连跑 10 题）。test 模式下赛题在仿真环境中隐藏，无法从客户端偷看。
- 官方更新（09-05/09-08/09-12）：删除了直接获取仿真环境信息的接口、隐藏可动物体白名单、
  修复漏题风险；**初赛结束后将在最新系统上复验决赛队代码**，违规会被处理。
- 提交规则（官网+文档综合，2026-09-13 确认）：材料中无提交次数/频率限制；只认最新版
  赛题系统（2026.09.12）的结果；官网登录后的提交入口如另有频率限制需以页面/答疑群为准。
- 晋级与决赛结构（官网首页）：**初赛前 16 名晋级**；决赛成绩 = **竞速 60% + 技术答辩 40%**
  ——过程留档（每轮日志、改动登记）从现在起就是答辩材料。

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
| raven 瑞文测试 | **已破局（2026-09-14）**：官方 ResNet 首选 + 直接提交，v2 题库 train 实测 13/13 对（97~99.7 分）。实现见 `src/cqairace/raven_proto.py`；旧系统 0 分系 v1 判卷/题库缺陷 | 一题一提交/一题一连接的 v2 语义 | ✅ 验证通过 |
| npc 对话 | 强 LLM 多轮对话收集线索，`submit_answer` 提交 | `speak_to_npc` 返回 `npc_reply`+`hints` 注入下一轮 prompt（官方已有） | 规划 |
| tidyroom 整理房间 | **已落地初版（2026-09-14，train 32 分 = 基线翻倍）**：三阶段——360° 扫描建图 → VLM 一次性并行分类（物品→容器映射，规则兜底）→ 零 VLM 元数据执行（`world_aabb` 中心+顶高放置、not pickup 秒拒拉黑、30s 看门狗 finish）。实现 `src/cqairace/tidyroom_agent.py`；待多轮 train 调优 + test | `objects` 元数据 + 官方"按包围盒放置"提示 | ✅ 首版验证（32/100） |
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
- **评分细则在 PDF 手册**：已获取并入库摘要（见 1.1 评分规则与题量）；
  时间效率的具体公式（得分-时间曲线）仍是图片形式，精确公式待
  train 模式 `eval_res.json` 实测反推。
- **接口已收紧（2026-09-05 后）**：感知仅剩 `acquire_first_person_perception`
  （组合图 + objects 元数据）与 `has_object_in_hand` 两个接口；
  《上手指南》PDF 第 8 节接口表中的 `get_object_basic_info` 等单查接口为
  旧版残留，以 proto 为准。`objects` 元数据保留 → 感知双通道策略仍成立。
  `move_and_take_puzzle_piece` 已无作用、`movable_object_ids` 已失效。
- **官方文档版本混乱**：release 内 md 为旧版（3 题/任务），网盘 PDF 为新版；
  以实测为准。

## 8. 本地系统布局与环境（2026-09-13 记录）

### 官方完整比赛系统位置

**`E:\2026\cqAIRace\Windows`**（网盘下载，2026.09.12 版赛题系统）：

| 路径 | 内容 | 状态 |
|---|---|---|
| `客户端\Windows\Windows\` | UE 仿真客户端（TongTestUE5.exe，run.bat 启动） | 已解压（约 53GB） |
| `赛题系统\2026.09.12\release\release\` | 赛题系统：`start_test.bat` → `arena_offline.exe`（任务系统 :50051）+ `tongsim_server\`（TongSim 代理 :50060 → UE :50052/:5056） | 已解压 |
| `基准Agent\baseline-agent.zip` | 官方 baseline 打包 | 与 gitee 克隆功能一致（已逐文件核对，见下） |
| `文档\` | 四份 PDF：初赛系统使用指南 / ArenaAgentPro 上手指南 / Agent 模型配置表 / 新增自定义 Agent 配置 | 已通读，摘要见本文档 |
| `更新说明.txt` | 09-05 / 09-08 / 09-12 三次更新记录（接口收紧、白名单隐藏、反漏题） | 已读 |

版本核对结论（2026-09-13）：`baseline-agent.zip` 与 `official_source/gitee/baseline-agent`
文件清单完全一致；唯一实质差异是 `tongsim_interface.py` 的 docstring 详略（gitee 版更全、
更新）。**`official_source` 无需更换，继续以 gitee pull 方式跟进官方更新。**

启动顺序：①双击客户端 `run.bat`；②`start_test.bat train --task <task-id>`（调试）
或 `test`（正式）；③切换任务时客户端不用重启，Agent 出错则必须重启赛题系统（Ctrl+C）。

### Python 环境

- conda 环境 **`cqairace`**（`E:\Softwares\Anaconda\envs\cqairace`）：
  Python 3.12.14 + uv 0.12.13（pip 经清华镜像安装于环境内）。
- 用法：`conda activate cqairace` 后在仓库根目录执行 `uv sync` 等命令；
  首次 `uv sync` 前先 `uv venv --python %CONDA_PREFIX%\python.exe` 让项目
  `.venv` 绑定 conda 解释器（避免 uv 另行下载托管 Python）。

### 模型接入（2026-09-13 实测）

- 主力模型：**`deepseek-flash`**（DeepSeek，OpenAI 兼容接口
  `https://api.deepseek.com/v1`；同账号的 `deepseek-v4-pro` 已实测出局、
  全线禁用——2026-09-14 tidyroom 视觉分类压测：233~298s/次（flash 为
  43~71s）、单帧准确率更低，见 roadmap §6）。
- 实测：支持视觉输入（OpenAI `image_url` base64 格式）；能按要求在 `content`
  返回单动作 JSON 数组；为推理模型（`reasoning_content` 与正文分离，基线只读
  正文，兼容）；小 prompt 约 3s/步。
- 接入方式：官方"通用环境变量覆盖"（`VLM_CLIENT_TYPE=openai` +
  `VLM_CLIENT_CFG_NAME/API_BASE/API_KEY`），零代码改动，模板见根 README
  「模型接入」一节。key 只走环境变量，不入库。
- 影响：延迟对 counting/raven 无影响（确定性路线不调模型）；npc（时间分）
  与 tidyroom/jigsaw（VLM 路线）每步多约 3~8s；`deepseek-v4-pro` 已测
  出局（2026-09-14 压测：慢 4 倍且更不准，全线禁用），flash 的单帧错分
  靠多帧投票压制（tidyroom 已落地）。

## 9. 实施路线（阶段划分）

> 执行层细化（按天排布、决策门、每日检查清单、命令速查）见 [roadmap.md](roadmap.md)。

| 阶段 | 内容 | 退出标准 |
|---|---|---|
| P0 ✅ | 仓库骨架 + 本设计文档 + git/GitHub | 本文档合入 main |
| P1 | 跑通官方 `preliminary_baseline_agent` 采基线（train 模式；实测各任务题量与评分曲线） | 五任务各产出一份日志 + eval_res.json 基线成绩 |
| P2 | 落地 `main.py`/`agent.py`/`routing.py` 最小闭环（行为=官方基线） | 我们的 runner 跑通五任务，成绩≥基线 |
| P3 | counting + raven 确定性技能 | 对应任务得分显著超 VLM 直跑 |
| P4 | npc 对话强化 + tidyroom / jigsaw 混合策略 | 初赛五任务总分达标，提交 |
| P5 | 决赛：长程 planner + 记忆，按决赛手册调整 | 决赛测试 + 技术答辩材料 |
