# 任务专属 Prompt

本目录存放按任务类型覆盖的 prompt 片段，命名约定：`<task_type>.txt`
（`tidyroom` / `counting` / `npc` / `raven` / `jigsaw`）。

- 官方基线机制：`arenaagent/vlm_agent/prompts/` 下的 `react.txt`（系统角色与
  动作 JSON 约束）、`interaction_info.txt`（每轮变量渲染模板）等构成基础
  prompt；`preliminary_baseline_agent` 通过 `_build_prompt_variables` 注入
  `task_prompt` 变量实现按任务覆盖（参考其 `prompts/task_spec_prompt.json`）。
- 我们的加载点：在 `CQAgent._build_prompt_variables` 中读取本目录的
  `<task_type>.txt` 作为 `task_prompt`（TODO，见 agent.py）。
- 调试素材：每轮实际 prompt 与感知图会落盘到 `logs/prompts/`
  （`prompt_*.txt` / `perception_*.jpg`），调 prompt 前先看真实样本。
