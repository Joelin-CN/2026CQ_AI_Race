# 2026CQ_AI_Race · 全国通用人工智能挑战赛（高校组）

全国通用人工智能挑战赛 · 高校组「通用智能体任务挑战赛」参赛仓库。
平台 TongSIM（通界平台）；初赛五项任务：整理房间 / 分类计数 / NPC 对话 /
瑞文测试 / 拼图，决赛串联为长程综合任务。

- 官网报名：<https://nagic.bigai.ai/>（**报名与初赛提交截止 2026-09-20**）
- 赛题通知：[docs/problem/](docs/problem/)
- **解决方案设计（必读）**：[docs/solution/architecture.md](docs/solution/architecture.md)

## 仓库结构

```
├── docs/
│   ├── problem/            # 赛题通知（教委会文件）
│   └── solution/           # 我们的方案设计与决策记录
├── src/cqairace/           # 自研智能体代码（当前骨架，见目录内 README）
│   └── prompts/            # 任务专属 prompt 片段（规划）
├── tests/                  # 测试（当前占位，见目录内 README）
├── official_source/        # 官方资料（不入库，另行克隆，见下）
│   └── gitee/baseline-agent
├── temp/                   # 本地临时区：赛题系统下载等（不入库）
├── config.toml.example     # 运行配置模板（复制为 config.toml 使用）
└── pyproject.toml          # uv 项目定义；官方包以 editable 路径依赖挂载
```

## 本地环境与官方系统位置

- **Python 环境**：conda 环境 `cqairace`（Python 3.12.14，内含 uv 0.12.13）。
  ```bash
  conda activate cqairace     # 而后所有 uv 命令在此环境中执行
  ```
- **官方完整比赛系统**：`E:\2026\cqAIRace\Windows`（2026.09.12 版）——
  客户端（UE，已解压约 53GB）/ 赛题系统（`start_test.bat`）/ 基准 Agent /
  四份 PDF 文档 / 更新说明。详见
  [docs/solution/architecture.md](docs/solution/architecture.md) 第 8 节。
- 启动顺序：①客户端 `run.bat`；②`start_test.bat train --task <task-id>`；
  Agent 出错必须重启赛题系统，切换任务不需要。

## 快速开始

```bash
conda activate cqairace

# 1. 官方 baseline 已克隆于 official_source/gitee/baseline-agent（新机器按 README 仓库结构说明重新克隆）

# 2. 让项目 .venv 绑定 conda 解释器，再安装依赖（含官方 arenaagentpro，首次较大，含 torch）
uv venv --python %CONDA_PREFIX%/python.exe   # PowerShell: uv venv --python $env:CONDA_PREFIX/python.exe
uv sync

# 3. 生成 gRPC pb2 代码（官方脚本，写入官方包的 arenaagent/generated/）
uv run python official_source/gitee/baseline-agent/scripts/generate_pb2.py

# 4. 赛题系统就绪后运行（系统启动方式见官方 docs/usage_guide.md）
cp config.toml.example config.toml
uv run arenaagent --agent_name preliminary_baseline_agent \
    --config config.toml --vlm_model VLMGPT5Config   # 官方基线
```

官方文档（克隆后位于 `official_source/gitee/baseline-agent/docs/`）：
`usage_guide.md`（上手指南）、`simulation_interface_guide.md`（感知/动作 API）、
`vlm_config_table.md`（可选模型与所需环境变量）。

## 模型接入（DeepSeek，2026-09-13 实测）

当前主力模型 **`deepseek-flash`**（DeepSeek 账号下与 `deepseek-v4-pro` 二选一，用 flash）。
实测结论：支持视觉输入（OpenAI `image_url` 格式）；接口为 OpenAI 兼容
（`https://api.deepseek.com/v1`），对应官方客户端 `client_type="openai"`；
推理模型（回复含 `reasoning_content`，不影响基线解析，只读 `content`）；
小 prompt 约 3 秒/步。

接入用官方"通用环境变量覆盖"方式（零代码改动，覆盖一切配置类）：

```powershell
# PowerShell，每次会话设置；key 只走环境变量，绝不写入文件/仓库
$env:VLM_CLIENT_TYPE="openai"
$env:VLM_CLIENT_CFG_NAME="deepseek-flash"
$env:VLM_CLIENT_CFG_API_BASE="https://api.deepseek.com/v1"
$env:VLM_CLIENT_CFG_API_KEY="<DEEPSEEK_API_KEY>"
```

设置后运行基线（入口类任意，实际生效的是上面的环境变量）：

```bash
uv run arenaagent --agent_name preliminary_baseline_agent \
    --config config.toml --vlm_model VLMGPT4o1120Config --run_times 1
```

## 关键约定

- **不修改 `official_source/` 内任何文件**：官方代码只 import 复用，
  我们的改进全部放在 `src/cqairace/`（理由见架构文档决策 D1）。
- API key 一律走环境变量（`OPENAI_API_KEY` / `DASHSCOPE_API_KEY` /
  `ZHIPU_API_KEY` / `ARK_API_KEY` 等），不入库、不写进 config.toml。
- `config.toml`、`logs/`、`temp/` 均不入库（见 .gitignore）。

## 赛程时间线

| 节点 | 时间 |
|---|---|
| 报名与初赛提交截止 | 2026-09-20 |
| 初赛评审 | 09-20 ~ 09-25 |
| 晋级公示 / 决赛赛题发布 | 09-25 ~ 10-01 |
| 决赛与颁奖 | 10 月下旬 |
