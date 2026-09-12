# tests/

测试目录（当前占位）。代码落地阶段的首批测试计划：

1. **冒烟测试**：官方包可 import（含 pb2 生成物）、`cq_agent` 注册成功、
   路由表能解析 `vlm_config.__ALL__` 中的类名。
2. **离线回放**：用 `logs/prompts/` 录制的真实感知图与 prompt 样本回放
   prompt 生成与 JSON 解析，不依赖赛题系统即可调"大脑"。
3. **技能单测**：counting 聚合、jigsaw 匹配等确定性技能对录制数据断言。

前置：`official_source/gitee/baseline-agent` 已克隆并运行过其
`scripts/generate_pb2.py`（见根 README「快速开始」）。
