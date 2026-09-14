# v2 赛题系统实测行为笔记（2026-09-14）

> 记录对 2026.09.12 补丁版赛题系统（release_v2）的实测行为——全部来自 train/test
> 模式一手观察，官方文档未写明。**自定义 agent 的行为必须匹配这些语义。**
> 后续新发现继续追加到本文。

## 1. 版本与校验事件时间线

| 时间 | 事件 |
|---|---|
| 09-13 02:28 | 下载网盘 release.zip（MD5 `1876f072…`，后被证实为旧构建） |
| 09-13 深夜 | 首次打包上传被官网拒收："答题文件校验未通过" |
| 09-13 23:55 | 重新下载 release.zip（MD5 `cca6cde4…`，3 文件补丁：arena_offline.exe / arena_resources.pack / tongsim_server.exe，即"更新了赛题题库"） |
| 09-14 08:06 | 再次下载核对，MD5 不变 |
| 09-14 上午 | 技术支持确认：新版服务端校验 09-14 中午 12:00 生效（此前拒收是服务端未切换） |

经验：结果文件头部 `ARENA_RESULT_V4c91d6869a96a2bc4…` 新旧相同（格式+密钥常量），
官网校验的是**结果内容里的题库标识**；网盘同版本号会原地换文件，**每次提交前
重比 MD5**，被拒先怀疑版本、再问答疑群。

## 2. v2 核心语义（与 v1 的差异）

1. **一题一提交（one-shot）**：问答题（raven/counting/npc）提交一次即结束该题，
   无论对错，**没有重试机会**。v1 时代"答错继续试"的循环不再存在。
   → 策略含义：首选答案的正确率就是一切；基线 counting 的 10/10 全对含重试修正，
   在 v2 下不可复现，确定性计数更加重要。
   （注：raven 已实测确认；counting/npc 待复核——若仍有重试，行为会不同。）
2. **一题一连接**：agent 答完自己的题，其 session 立即 FINISHED(199)。系统推进到
   下一题后等待**新的 agent 连接**。官方 builder 的 `--run_times N` 循环天然匹配；
   自定义 `run()` 里常驻轮询 session 状态会永久停在 RUNNING(101) 并把系统拖入
   等待死锁。
3. train 与 test 行为一致（都遵循上述两条）；test 额外隐藏赛题、结果加密。

## 3. 判卷与归档行为

- 归档位置：`<release>/arena_offline/eval_res*.json`（train 明文；test 不落明文）。
- **答对时**：归档含 `correct_answer`（这就是我们的标注来源）。
- **答错时**：`correct_answer: null`（v2 起防漏题，不再泄露答案）。
- 每次新结果写入前，旧 `eval_res.json` 被重命名为 `eval_res.<时间戳>.json`。
- SessionStatus：101=RUNNING、199=FINISHED、99=TERMINATED（agent 视角，题答完即 199）。
- 任务系统日志（start_test 控制台）是 train 模式下最全的信息源。

## 4. 打包与运维坑

- `package-results --package-results-dir .` 的**基准目录是 `arena_offline`**，
  bins 就在其中时参数传 `.`；传错会报"result directory does not exist"。
- 换任务：杀 `arena_offline.exe` + `tongsim_server.exe` → 重跑 `start_test.bat`；
  UE 客户端全程不用重启。无 agent 连接时客户端显示空白天空盒属正常待机。
- Agent 进程出错/挂死后必须重启赛题系统，否则后续结果不可用。
- `uv run arenaagent` 若报 `No module named 'arena'`：重跑 pb2 生成脚本后
  `uv pip install --no-deps -e official_source/gitee/baseline-agent` 刷新 editable。

## 5. raven 任务的可用解法（已验证）

- 题面：一张 2376×1200 大图，横排 3 道图形推理（3×3 缺角 + 8 候选，候选按行
  编号 1-8），答案=三题编号连成三位数。
- 官方 `solve_raven`（ResNet18+MLP）在 v2 题库上**首选命中**：train 实测 13/13
  （6 种题型），秒级提交 97.0~99.7 分。
- v1 上 0/10 全错是旧版判卷/题库缺陷（同一套模型与裁剪代码，仅换系统即全对）。
- 实现与运行见 `src/cqairace/raven_proto.py`：
  `uv run python -m cqairace.raven_proto --run_times 10`
  （需带 VLM_CLIENT_* 环境变量以通过官方 client 构建断言，实际不调用模型）。

## 6. 素材库索引

| 路径 | 内容 |
|---|---|
| `temp/p2_raven/collected/` | 16 张唯一题图（彩色原图）+ 裁剪子图目录 |
| `temp/p2_raven/trace.jsonl` | 全部提交轨迹（题图哈希/名次/答案） |
| `temp/p2_counting/trace.jsonl` | counting 提交轨迹 |
| `temp/p4_tidyroom/trace.jsonl` | tidyroom 每件 take/put 轨迹 |
| `temp/baseline_records/` | 各任务 eval 归档 + 运行日志 + 历次提交包 |
| `temp/pdf_images/` | 官方 PDF 中提取的评分公式图 |

## 7. tidyroom 实测语义（2026-09-14，自研 agent train 实跑）

1. **"can not take this object for not pickup" 是服务端本地判断**：
   不触发导航、0.0s 返回——对不可抓物体（人脚上的鞋等）失败重试的代价
   几乎为零，"失败即拉黑"策略零成本；据此把物品候选尺寸上限放宽到 80cm
   （大件失败免费，成功一件赚 4 分）。
2. **train 每轮随机场景布局**：与基线同 task 的物体 object_id/位置完全
   不同——任何按 ID/坐标硬编码的策略在 test 必挂，运行时识别是唯一路径。
3. **评分线性可加**：5 件归位 + 234s 主动 finish = 32.0 分（基线 4 件超时
   16.0）。粗算 1 件≈4 分（0.8×5%），时间分≈0.2×剩余时间比；完成度
   分母精确值待多样本回归。
4. **deepseek-flash 视觉大 prompt（30+ 物体元数据+复合图）单次推理
   ~68s**：串行多帧分类必超时；解法是帧级并行（总耗时≈最慢单次）。
   模型还不守输出格式约定（返回 类别→ID列表 而非 ID→类别），解析器
   必须两种都兼容。
5. **UE 客户端被前台全屏程序（游戏）占用时感知退化**：可见物体从 52
   跌到 3，spawn 返回空 ID；重启 tongsim/arena 均无效，必须释放前台
   GPU 占用。跑验证前确认机器没人在用（本条是 2026-09-14 run3-5 三轮
   白跑的教训）。
