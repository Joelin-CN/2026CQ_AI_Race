# cqairace 包（自研智能体代码）

本包承载我们的参赛智能体"大脑层"，**当前为骨架占位**，具体代码待
`docs/solution/architecture.md` 中的路线逐阶段落地。规划模块布局：

```
cqairace/
├── main.py        # 自研 runner：加载 config、构建 agent、对接任务系统（镜像官方 builder.main）
├── agent.py       # CQAgent(VLMAgent)：技能接管点 + 感知增强 + 场景记忆的挂载位置
├── routing.py     # 任务类型 → VLM 模型配置 的路由表（provider 无关，按任务混跑）
└── prompts/       # 任务专属 prompt 片段（<task_type>.txt），见目录内 README
```

约定：

- 官方基础设施（`arenaagent`）一律 import 复用，不复制、不修改；
  官方仓库以 editable 路径依赖挂载（见根 pyproject.toml）。
- 每个模块落地时先读 `official_source/gitee/baseline-agent/docs/usage_guide.md`
  对应章节，接口语义以官方文档与源码为准。
- 调试产物统一落 `logs/`（不入库）；prompt 调优素材看 `logs/prompts/`。
