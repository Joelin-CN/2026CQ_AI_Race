# 初赛冲刺路线图（2026-09-13 ~ 2026-09-20）

> 配套文档：[architecture.md](architecture.md)（方案设计与决策记录，含评分规则/系统布局/模型接入）。
> 本文是**执行层计划**：按天排布，含里程碑、决策门、每日检查清单与命令速查。
> D0 = 09-13（周日），提交截止 D7 = 09-20。

## 0. 总览

```
日期        D0~D1         D1~D2         D2~D3         D3~D5         D5~D7
            09-13/14      09-14/15      09-15/16      09-16~18      09-18~20
            ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
阶段        │ P1 打通+保底│ │ P2 确定性  │ │ P3 NPC    │ │ P4 完成度  │ │ P5 冲刺   │
            │ 全链路跑通  │ │ raven+计数│ │ 对话优化  │ │ 整理+拼图  │ │ 提交+留档 │
            └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
里程碑      合法           20/31 道题   5 道题时间   2 道题完成   最终成绩
            result_package 走代码路径   分提升      度提升       上传官网
```

**铁律：先保底，再优化。** 打包上传要求五个任务的 `.bin` 全齐；每任务取历史
最好成绩。任何时刻都必须手持一个可提交的 `result_package.bin`，之后每优化
一个任务就重跑重传一次。

**分工约定**：AI 助手负责代码、prompt、日志分析与文档；人负责跑系统
（客户端/赛题系统）、看画面、观察异常、拍板决策门。

---

## 1. 阶段详解

### P1 打通与保底（D0~D1）——最高优先级

**目标**：全链路跑通 + 拿到第一个合法提交 + 摸清全部情报。

| # | 任务 | 说明 |
|---|---|---|
| 1.1 | 环境落地 | `uv venv` 绑定 conda 解释器 → `uv sync` → 生成 pb2 → import 冒烟 |
| 1.2 | 首跑基线 | 客户端 `run.bat` → `start_test.bat train --task competition-preliminary-tidy-room-task` → DeepSeek 环境变量 → 跑官方基线 1 题 |
| 1.3 | 建立直觉 | 精读 `logs/prompts/`：感知图（左 RGB/右编号分割）+ prompt 文本 + 日志 `parsed json message` / `action run result` |
| 1.4 | 五任务基线 | train 模式逐任务跑完（run_times 按 1/1/5/10/10），记录 `eval_res.json` 基线分 |
| 1.5 | **保底提交** | test 模式跑全五任务 → `package-results` → 保存第一个合法 `result_package.bin` |

**退出标准**：`result_package.bin` 存在且已上传一次；五任务基线分记录在案。

**分析**：零优化但价值密度最高——把所有不确定变成已知：基线每题几分、哪类
任务崩、DeepSeek 格式错误率、400 秒能走几步、评分曲线形状（`eval_res.json`
明文可反推时间效率公式）。

**风险**：基线报错必须重启赛题系统（Ctrl+C）；counting/raven 各连跑 10 题，
全流程最坏 3 小时+。D0 必须完成 `uv sync`。

### P2 确定性得分基本盘（D1~D2）——raven + counting

**目标**：20/31 道题（counting 10 + raven 10，纯时间分）从 VLM 循环切换为
代码路径，拉到接近满分档。

| 任务 | 做法 | 量级 |
|---|---|---|
| raven | 在自研层激活官方 `solve_raven`（ResNet18 秒答），修 `/tmp` 路径 Windows 坑；train 实测 10 题正确率，必要时调候选顺序或加 VLM 复核 | 0.5~1 天 |
| counting | 确定性技能：原地 360° 分段转身 → 逐朝向取感知 → `object_id` 跨帧去重 → 颜色/形状聚合 → 直接 `submit_answer`，零 VLM 调用 | 1 天 |

**前置验证（P2 第一件事）**：`object_id` 跨步骤是否稳定（转回同一朝向，
同物体 ID 不变）——去重逻辑成立的前提；不稳定则改世界坐标聚类。

**同步落地**：第一批自研代码（`main.py`/`agent.py`/`routing.py` 最小闭环 +
两个 skill），架构见 architecture.md §4。

**退出标准**：train 模式 raven/counting 得分显著超基线；test 重跑并重传。

### P3 NPC 对话优化（D2~D3）

**目标**：5 道时间分题，减少对话轮数、提高答题正确率。

- 看基线 npc 失败日志 → 改 `task_spec_prompt.json` 的 `npc` 项：
  先问谁、问什么、几轮收敛、信息够了立刻 `submit_answer`；
- 用好 `speak_to_npc` 返回的 `npc_reply` + `hints` 注入机制；
- 每少一轮对话省 15~20 秒，prompt 里明确"够用即提交"。

**退出标准**：npc train 分超基线且 5 题稳定不崩。

**分析**：prompt 工程直接变现，无新知识门槛；收益上限低于 P2 但工作量也小。
若 flash 延迟成瓶颈（决策门 4），换 `deepseek-v4-pro` 对比。

### P4 完成度任务（D3~D5）——tidyroom + jigsaw

**目标**：两道完成度比例制题目，提升摆放/放置正确率且全程稳定不崩。

| 任务 | 主策略 | 保底策略 |
|---|---|---|
| jigsaw | 确定性：解析分割图右半做形状/位置匹配 → `move_and_put_down` 到参考包围盒（官方提示：拼图块 X=837） | 强 task prompt + 官方提示注入，放对一块是一块 |
| tidyroom | VLM 循环 + 元数据辅助：`place_location`/AABB 做放置规划，VLM 只判"什么放哪"；临近 400s 主动 `finish_task` 锁住完成度 | 同左，重点是不崩 |

**工程加固（本阶段重点）**：每题超时看门狗、异常捕获不炸循环、run_times
断点可重跑——Agent 崩一次 = 重启系统 = 丢一轮。

**退出标准**：两任务完成度较基线提升；连续两轮 test 不崩。

### P5 冲刺提交（D5~D7）

- 每天至少一轮完整 test（五任务）→ 打包 → **当天上传**，不攒到 D7；
- 代码与日志归档（决赛要复现代码 + 合规检查，只用官方接口）；
- D7 中午前完成最后一次提交，留半天缓冲。

---

## 2. 跨阶段工程纪律

1. **稳定性 = 分数**：所有代码第一天起就要异常捕获、超时看门狗、可断点重跑。
2. **一次只改一处**：每轮改动对应一份日志，能回答"分数为什么变了"；
   改动记录追加到本文 §6。
3. **秘密不入库**：key 只走环境变量。
4. **不修改 official_source**：官方代码只 import；我们的改动全在 `src/cqairace`。

## 3. 收益/成本总账

| 阶段 | 投入 | 预期收益（占总分权重） | 确定性 |
|---|---|---|---|
| P1 | 1 天 | 从"0 或无效"到"有成绩"+ 全部情报 | 必得 |
| P2 | 1.5 天 | counting+raven ≈ 40% 权重拉满 | 高 |
| P3 | 1 天 | npc ≈ 20% 权重中等提升 | 中高 |
| P4 | 2 天 | tidyroom+jigsaw ≈ 40% 权重的完成度爬坡 | 中 |
| P5 | 1 天 | 落袋 + 抗风险 | 必得 |

## 4. 决策门

| # | 时点 | 触发条件 | 动作 |
|---|---|---|---|
| G1 | P1 末 | 某任务基线已接近满分 | 对应优化阶段降级，时间挪给弱项 |
| G2 | P2 中 | raven 官方模型 10 题正确率 < 60% | 启用"模型候选 + VLM 复核"混合 |
| G3 | P2 中 | counting 的 object_id 跨步骤不稳定 | 去重改世界坐标聚类 |
| G4 | P3 末 | DeepSeek flash 格式错误率 > 20% 或延迟成瓶颈 | 换 deepseek-v4-pro / 调解析容错对比 |
| G5 | P4 初 | 时间余量 < 1.5 天 | 砍 jigsaw 确定性路线，纯 prompt 保完成度 |

## 5. 每日检查清单

- [ ] 当天的 `result_package.bin` 是否仍是"全五任务最好成绩"组合？
- [ ] 新增改动是否已在 §6 登记（改了什么、对应哪份日志、分数变化）？
- [ ] 关键日志是否归档到本机（决赛复验材料）？
- [ ] （D5 起）今天是否完成一轮 test + 上传？

## 6. 改动与结果登记

> 格式：`日期 | 任务 | 改动 | 日志/证据 | 分数变化`。跑一轮记一行。

### 2026-09-13 基线成绩（P1.4，train 模式，DeepSeek-flash，官方 preliminary_baseline_agent）

| 任务 | 成绩 | 明细 | 关键观察 |
|---|---|---|---|
| tidyroom | **16.0** | 1 题，超时结算 | 31 步/400s（13s/步），10 次抓取失败（"not pickup"，疑似抓人脚上的鞋），仅归位 4 件 ≈ 完成度 16% |
| counting | **53.14** | 10/10 全对：86.75/75.6/65.05/45.65/58.85/69.65/29.75/57.95/3.15/39.0 | 八选一选择题；**答错可重试**；耗时越长分越低（40s≈87 → 400s≈3）；题面提示需探索遮挡物品 |
| npc | **88.61** | 5/5 全对：93.7/90.85/94.0/94.3/70.2 | 说谎者推理题四选一；DeepSeek 交叉验证证词能力强；第 5 题错 2 次后仍 70.2；格式错误（提交拼音 zhangnainai）会被拒但不致死 |
| raven | **≈0** | 前若干题全 0 | solve_raven 反复返回同一候选（推进机制未生效），模型暴力换数字耗尽 400s；裁剪坐标与实际图（2376×1200 横排三题）吻合 → 锅在模型/答案映射，非裁剪；素材已抢救至 temp/baseline_records/raven_images |
| jigsaw | 待跑 | | |

**证据归档**：`temp/baseline_records/{counting,npc}/`（eval json + 运行日志）。
**初步基线总分（4/5 任务）**：(16.0+53.14+88.61+0)/5 ≈ 31.5，jigsaw 待补。

### 改动登记

| 日期 | 任务 | 改动 | 证据 | 分数变化 |
|---|---|---|---|---|
| 2026-09-13 | 全部 | 建立基线（无改动，官方 agent + DeepSeek-flash） | temp/baseline_records/ | 见上表 |

## 附录：命令速查

```powershell
# --- 模型环境变量（每会话） ---
$env:VLM_CLIENT_TYPE="openai"
$env:VLM_CLIENT_CFG_NAME="deepseek-flash"
$env:VLM_CLIENT_CFG_API_BASE="https://api.deepseek.com/v1"
$env:VLM_CLIENT_CFG_API_KEY="<DEEPSEEK_API_KEY>"

# --- 本项目环境（每会话） ---
conda activate cqairace   # 仓库根目录执行后续命令

# --- 赛题系统（E:\2026\cqAIRace\Windows\赛题系统\2026.09.12\release\release\）---
.\start_test.bat train --task competition-preliminary-tidy-room-task     # 调试（明文 eval_res.json）
.\start_test.bat test  --task competition-preliminary-counting-task      # 正式（加密 .bin）
.\start_test.bat package-results                                         # 打包上传件
# 任务 ID：tidy-room / jigsaw / counting / npc / raven（前缀均为 competition-preliminary-…-task）

# --- 基线 Agent（official_source/gitee/baseline-agent/ 或本项目 .venv）---
uv run arenaagent --agent_name preliminary_baseline_agent --config config.toml `
    --vlm_model VLMGPT4o1120Config --run_times <题数: 1/1/5/10/10>
```

运维备忘：Agent 报错必须重启赛题系统（Ctrl+C）；切换任务客户端不用重启；
客户端启动 = 双击 `客户端\Windows\Windows\run.bat`。
