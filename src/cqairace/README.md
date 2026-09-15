# cqairace 包（自研智能体代码）

本包承载我们的参赛智能体"大脑层"，落地于 `docs/solution/architecture.md`
规划的各阶段。当前模块布局：

```
cqairace/
├── tidyroom_agent.py       # P4 整理房间 agent（v4：识别 gate→分类锁定→纯执行
│                           #   + 歧义件到场确认；TIDYROOM_V4=0 回退 v3 流水线）
├── stamp_perceive.py       # v4 生产感知：戳定位→VLM 转写→洪泛色块→裁剪自标
│                           #   （绑号由代码完成，VLM 不再读 8-10px 小号）
├── vlm_direct.py           # 直连 VLM 客户端（deepseek-flash 思考关 + temperature=0；
│                           #   读 VLM_CLIENT_CFG_* 环境变量，官方客户端回退。
│                           #   背景：官方 client 无思考控制，大 prompt 42~68s 且
│                           #   reasoning 吃满 token 正文为空；直连后 0.5~1.0s）
├── dashcam.py              # 行车记录仪 YOLO 式标注（复盘旁路，不影响任务；
│                           #   _label_region 洪泛被 stamp_perceive 生产复用）
├── dashcam_templates.json  # 数字位图模板表（当前为空=模板 OCR 停用，复盘走
│                           #   vision 整组读数）
├── raven_proto.py          # P2 瑞文秒答（官方 ResNet 首选直接提交）
├── counting_agent.py       # P2 分类计数（扫描+VLM 分类+聚类去重）
├── jigsaw_skill.py         # P3 拼图技能（三层降级：几何/VLM+CV/执行校验）
├── jigsaw_agent.py         # P3 拼图薄壳 agent
└── prompts/                # prompt 片段，见目录内 README
```

约定：

- 官方基础设施（`arenaagent`）一律 import 复用，不复制、不修改；
  官方仓库以 editable 路径依赖挂载（见根 pyproject.toml）。
- 每个模块落地时先读 `official_source/gitee/baseline-agent/docs/usage_guide.md`
  对应章节，接口语义以官方文档与源码为准。
- 调试产物统一落 `logs/`（不入库）；prompt 调优素材看 `logs/prompts/`。
- tidyroom 感知/复盘的离线实验脚本在 `temp/p4_tidyroom/`（gitignore），
  关键结论回写 `docs/solution/v2-system-notes.md` §7。
