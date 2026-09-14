# AGENTS.md — 面向 AI 编码助手的工作规范

本仓库是比赛项目（见 [README.md](README.md)），人在回路推进。AI 助手在此
仓库工作时遵守以下约定。

## 硬性规则

1. **绝不修改 `official_source/` 内任何文件**。官方 baseline-agent 只读：
   可以 import、可以引用其代码与文档，改动一律发生在 `src/cqairace/`。
   官方更新通过重新 `git pull` 该目录同步。
2. **秘密不入库**：API key、内网地址等只走环境变量；发现疑似泄漏立即提醒。
3. **方案变更先改文档**：偏离 `docs/solution/architecture.md` 决策记录的
   改动，先更新该文档（或在 `docs/decisions/` 补 ADR）再写代码。
4. `docs/problem/` 是官方原件，只读不改。

## 目录语义

| 路径 | 语义 |
|---|---|
| `docs/solution/architecture.md` | 方案唯一权威记录（决策、架构、评分规则、任务路线、阶段计划） |
| `src/cqairace/` | 自研智能体代码（规划布局见其目录内 README） |
| `official_source/gitee/baseline-agent/` | 官方库（gitignore，另行克隆；接口语义以其源码与 docs/ 为准） |
| `E:\2026\cqAIRace\Windows` | 官方完整比赛系统（gitignore 外部路径）：客户端/赛题系统/PDF 文档，2026.09.12 版，详见架构文档第 8 节 |
| `temp/` | 本地临时区（gitignore）：赛题系统安装包、下载中间物 |
| `temp/p4_tidyroom/test_sessions/` | tidyroom 逐轮感知档案（gitignore）：每轮一目录 `{YYYYMMDD_HHMMSS[_标签]}/`，内含 `frame_*.jpg`（行车记录仪）+ `session_summary.json`（placed/黑名单/分类）；同目录平级存放该轮 `test_round_*.bin` 备份 |
| `logs/`（运行时生成） | 运行日志与 prompt/感知图落盘（gitignore） |

## 工具链

- Python 环境：conda 环境 **`cqairace`**（Python 3.12.14 + uv 0.12.13）。
  先 `conda activate cqairace` 再执行 uv 命令；`.python-version` 固定 3.12；
  tuna 镜像已在 pyproject 配置。常用命令：`uv sync`、`uv run <cmd>`、`uv run pytest`。
- 官方包以 editable 路径依赖挂载（`[tool.uv.sources] arenaagentpro`），
  安装后 `import arenaagent` 直接可用；若未装，入口代码需带 sys.path 兜底。
- pb2 生成：`uv run python official_source/gitee/baseline-agent/scripts/generate_pb2.py`
  （`arenaagent.generated.*` 未生成时 import 会失败，属正常现象而非代码 bug）。

## 官方接口速查（详见官方 docs/simulation_interface_guide.md）

- 任务系统 gRPC `127.0.0.1:50051`；TongSim 仿真 gRPC `127.0.0.1:50060`。
- **v2 系统语义（实测，务必遵守）**：问答题一题**一提交**（无重试）、一题**一连接**
  （答完即断开，靠 run_times 循环领下一题）；答对才在 eval 归档揭示正确答案。
  详见 [docs/solution/v2-system-notes.md](docs/solution/v2-system-notes.md)。
- 感知：`acquire_first_person_perception` → `{image: base64 复合图(左RGB/右带编号分割), objects: [{object_id, color, shape, place_location, world_aabb}]}`；
  `object_id` 与分割图编号一致。
- 决策输出：单动作 JSON 数组 `[{"think", "action", "parameters", "output"}]`，
  问答题 `submit_answer`，行为题 `finish_task` 收尾。
- NPC 对话固定角色：江淑艳 / 刘伟东 / 赵爷爷 / 张奶奶。
- 已验证的生产级路径：raven 用 `cqairace.raven_proto`（ResNet 首选秒答，13/13 对）。

## 写作约定

- 文档与注释用中文；代码标识符用英文。
- 引用官方事实（接口、参数、规则）时给出文件路径或文档出处，便于复核。
- 比赛时间紧（初赛提交截止 2026-09-20）：优先低垂果实（counting/raven），
  改动求小步可回退，每步可运行。
