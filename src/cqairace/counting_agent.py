"""P2 计数 agent：360° 扫描 + 逐帧目标识别 + 确定性聚合，一次性提交（one-shot）。

策略（对应 v2 一题一提交语义）：
- 问句若按颜色/形状计数 → 纯元数据过滤，零 VLM，秒答；
- 问句按语义类别（苹果/碗等）→ 每个朝向取一帧复合图，VLM 标注该帧哪些
  object_id 属于目标类别，跨帧按世界坐标聚类去重后计数；
- 计数与八选项精确匹配则选之，否则取最接近选项。

运行（赛题系统 train counting 已启动，仓库根目录）：
    uv run python -m cqairace.counting_agent --run_times 10
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_official() -> None:
    try:
        import arenaagent  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(_REPO_ROOT / "official_source" / "gitee" / "baseline-agent"))


_ensure_official()

from loguru import logger  # noqa: E402

from arenaagent.builder import Register  # noqa: E402
from arenaagent.utils.configclass import configclass  # noqa: E402
from arenaagent.vlm_agent.json_parsor import extract_last_json_from_text  # noqa: E402
from arenaagent.vlm_agent.vlm_agent import VLMAgent, VLMAgentCfg  # noqa: E402

TRACE_DIR = _REPO_ROOT / "temp" / "p2_counting"

_COLOR_WORDS = {"红": "red", "绿": "green", "蓝": "blue", "黄": "yellow", "黑": "black", "白": "white", "灰": "gray", "紫": "purple", "橙": "orange", "粉": "pink", "棕": "brown", "米": "beige"}
_SHAPE_WORDS = {"圆柱": "cylinder", "圆形": "round", "圆": "round", "矩形": "rectangle", "立方": "box", "三角": "triangle", "线": "line", "椅子": "chair"}


@configclass
class CountingAgentCfg(VLMAgentCfg):
    name: str = "counting_agent"
    sleep_between_steps: float = 0.3


@Register("counting_agent")
class CountingAgent(VLMAgent):
    """计数 one-shot：扫描 → 聚合 → 匹配选项 → 提交。"""

    _SCAN_TURNS = 4  # 每次转 90°，共覆盖 360°
    _QUESTION_BUDGET = 150.0  # 整题预算（秒），超时用部分结果提交
    _OP_TIMEOUT = 40.0  # 单次外部调用超时（感知/转身/VLM）

    _pool = ThreadPoolExecutor(max_workers=2)  # 超时兜底用（挂起线程允许泄漏）

    @classmethod
    def _call_with_timeout(cls, fn: Callable, *args, timeout: float | None = None, default=None):
        """官方客户端均无超时，一次挂起即卡死全题；此处用线程+超时兜底。"""
        fut = cls._pool.submit(fn, *args)
        try:
            return fut.result(timeout=timeout or cls._OP_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            logger.warning("call timeout/error {}: {}", getattr(fn, "__name__", fn), exc)
            return default

    def __init__(self, stub, channel, cfg: CountingAgentCfg | None = None, sleep_between_steps: float = 0.3) -> None:
        super().__init__(stub=stub, channel=channel, cfg=cfg or CountingAgentCfg(), sleep_between_steps=sleep_between_steps)
        self._q_state: dict[str, dict] = {}  # 题目 -> {order, tried}（同题重提交换选项）

    def run_step(self, subject, task_response: dict[str, Any]) -> dict[str, Any]:
        question = str(subject.get("question", ""))
        options = subject.get("options") or {}
        key = self.action_space.get("key") or "action"
        logger.info("counting 题目: {}", question)

        # 1) 解析目标类别
        category, color_filter, shape_filter = self._parse_target(question)

        # 2) 扫描采集（首帧不动，之后每 90° 一帧）；整题预算看门狗
        deadline = time.time() + self._QUESTION_BUDGET
        frames = []
        for i in range(self._SCAN_TURNS):
            perception = self._call_with_timeout(
                self.tongsim.acquire_first_person_perception, self.character_id, 1280, 720,
                default={},
            )
            objects = perception.get("objects", []) or []
            frames.append({"image": perception.get("image", ""), "objects": objects})
            if i < self._SCAN_TURNS - 1:
                if time.time() > deadline - 30:  # 留 30s 给聚合与提交
                    logger.warning("扫描预算告急，提前结束于第 {} 帧", i + 1)
                    break
                self._call_with_timeout(self.tongsim.turn_in_degree, self.character_id, 90, default=None)
                time.sleep(0.3)

        # 3) 逐帧识别 + 跨帧去重（世界坐标 10cm 桶 + 颜色形状为键）
        matched: dict[tuple, dict] = {}
        per_frame_ids = []
        for fi, frame in enumerate(frames):
            ids = self._match_frame(category, color_filter, shape_filter, frame)
            per_frame_ids.append(ids)
            coord = {}
            for obj in frame["objects"]:
                oid = str(obj.get("object_id", ""))
                if oid in ids:
                    coord[oid] = obj
            for oid, obj in coord.items():
                k = self._dedup_key(obj)
                matched.setdefault(k, {"oid": oid, "obj": obj, "frames": []})
                matched[k]["frames"].append(fi)

        count = len(matched)
        # 4) 跨帧聚类去重（同色同形 20cm 半径贪心合并，防坐标漂移分裂/重复计数）
        count = self._cluster_count(matched)
        # 4) 生成选项顺序：精确匹配优先，其余按 |选项值-计数| 升序；一题一提交
        order = self._option_order(options, count)
        chosen = order[0] if order else "A"
        logger.info("counting 聚合: 目标={} 帧命中={} 计数={} 选项序={} -> 提交 {}",
                    category or color_filter or shape_filter, per_frame_ids, count, order, chosen)

        self._trace(question, category, color_filter, shape_filter, per_frame_ids, count, options, chosen)
        return {key: str(chosen)}

    # ------------------------------------------------------------------ #

    def _parse_target(self, question: str) -> tuple[str | None, str | None, str | None]:
        """返回 (语义类别, 颜色过滤, 形状过滤)。"""
        color_filter = None
        for cn, en in _COLOR_WORDS.items():
            if cn in question:
                color_filter = en
                break
        shape_filter = None
        for cn, en in _SHAPE_WORDS.items():
            if cn in question:
                shape_filter = en
                break
        m = re.search(r"多少[个只条块](.+?)(?:在|存|有|$)", question)
        category = m.group(1).strip() if m else None
        if category:
            for cn in list(_COLOR_WORDS) + list(_SHAPE_WORDS):
                category = category.replace(cn, "")
            category = category.strip("的 ") or None
        return category, color_filter, shape_filter

    def _match_frame(self, category, color_filter, shape_filter, frame) -> set[str]:
        objects = frame["objects"]
        if color_filter or shape_filter:
            ids = set()
            for obj in objects:
                c = str(obj.get("color", "")).lower()
                s = str(obj.get("shape", "")).lower()
                if color_filter and c != color_filter:
                    continue
                if shape_filter and s != shape_filter:
                    continue
                ids.add(str(obj.get("object_id", "")))
            return ids

        # 语义类别 → VLM 逐帧标注
        compact = [
            {"object_id": str(o.get("object_id", "")), "color": o.get("color"), "shape": o.get("shape")}
            for o in objects
        ]
        prompt = (
            f"这是仿真环境第一视角复合图（左半 RGB，右半是带数字编号的语义分割图，编号与下方元数据的 object_id 一一对应）。\n"
            f"任务：找出当前画面中属于「{category}」的物体。\n"
            f"当前帧可见物体元数据：{json.dumps(compact, ensure_ascii=False)}\n"
            f"请结合左图与元数据判断哪些 object_id 属于「{category}」。\n"
            f"注意：只把确定的成员算入，相似但不确定的物体不要算；被部分遮挡的物体若可辨认也应算入。\n"
            f"重要：不要思考过程，回答的第一个字符必须是 {{，格式为 {{\"ids\": [\"3\", \"7\"]}}；"
            f"若没有则输出 {{\"ids\": []}}。"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + frame["image"]}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        ids = set()
        for attempt in range(2):  # 推理模型可能返回空正文，重试一次
            try:
                resp = self._call_with_timeout(self.vlm_client.invoke, messages, timeout=self._OP_TIMEOUT)
                if resp is None:
                    continue
                text = getattr(resp, "text", "") or ""
                if not text.strip():
                    logger.warning("VLM 返回空文本（第 {} 次），重试", attempt + 1)
                    continue
                parsed = extract_last_json_from_text(text)
                data = json.loads(parsed) if isinstance(parsed, str) else parsed
                ids = {str(x) for x in data.get("ids", [])}
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLM 分类失败(第{}次): {}", attempt + 1, exc)
                continue
        return ids

    @staticmethod
    def _dedup_key(obj: dict) -> tuple:
        loc = obj.get("place_location") or {}
        try:
            x, y = int(float(loc.get("X", 0)) / 10), int(float(loc.get("Y", 0)) / 10)
        except Exception:
            x, y = 0, 0
        return (x, y, str(obj.get("color")), str(obj.get("shape")))

    @staticmethod
    def _cluster_count(matched: dict) -> int:
        """跨帧同物体坐标漂移会分裂成多条记录；按 (color, shape) 分组后
        在 20cm 半径内贪心合并，聚类数即计数。"""
        items = []
        for v in matched.values():
            loc = (v["obj"].get("place_location") or {})
            try:
                items.append((float(loc.get("X", 0)), float(loc.get("Y", 0)),
                              str(v["obj"].get("color")), str(v["obj"].get("shape"))))
            except Exception:
                continue
        groups: dict[tuple, list] = {}
        for x, y, c, s in items:
            groups.setdefault((c, s), []).append((x, y))
        total = 0
        for pts in groups.values():
            unused = sorted(pts)
            while unused:
                x0, y0 = unused.pop()
                total += 1
                keep = []
                for x, y in unused:
                    if (x - x0) ** 2 + (y - y0) ** 2 > 20 ** 2:
                        keep.append((x, y))
                unused = keep
        return total

    @staticmethod
    def _option_order(options: dict, count: int) -> list[str]:
        """精确匹配优先，其余按 |值-计数| 升序；返回字母序（重提交时按序换）。"""
        items = []
        for letter, value in options.items():
            try:
                v = int(float(value))
            except Exception:
                continue
            items.append((abs(v - count), letter))
        items.sort(key=lambda t: (t[0], t[1]))
        return [letter for _, letter in items]

    def _trace(self, *args) -> None:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.strftime("%H:%M:%S"),
            "question": args[0],
            "category": args[1],
            "color_filter": args[2],
            "shape_filter": args[3],
            "per_frame_ids": [sorted(s) if isinstance(s, set) else s for s in args[4]],
            "count": args[5],
            "options": {str(k): v for k, v in (args[6] or {}).items()},
            "chosen": args[7],
        }
        with open(TRACE_DIR / "trace.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def run(self) -> None:
        """官方式收尾：答完保持连接轮询 session 状态至终态（带 120s 上限）再断开。

        推测：服务端在 session 终态迁移时写判卷，agent 过早断开会跳过写盘
        （官方 agent 恰好总是轮询到 199；raven 采集器因答案全对走了同步写盘路径）。
        """
        from google.protobuf import struct_pb2

        from arenaagent.agent_base import parse_struct_to_data
        from arenaagent.generated.arena.message import session_msg_pb2

        done_states = (
            session_msg_pb2.SessionStatus.Value("FINISHED"),
            session_msg_pb2.SessionStatus.Value("TERMINATED"),
            session_msg_pb2.SessionStatus.Value("ERROR"),
        )

        if not self.connected:
            return
        self.action_space = parse_struct_to_data(
            self._call_struct("get_action_space", {"agent_id": self.agent_id}, struct_pb2.Struct.FromString)
        )
        while not self._task_ready():
            time.sleep(self.sleep_between_steps)
        self._run_subject()
        deadline = time.time() + 120
        while time.time() < deadline:
            status = self._get_task_status().get("session_status")
            if status in done_states:
                logger.info("session 终态 {}，断开", status)
                break
            time.sleep(1)
        else:
            logger.warning("session 轮询超时，强制断开")
        self._disconnect()


def main() -> None:
    import argparse

    import grpc

    from arenaagent.generated.arena.agent.arena_agent_service_pb2_grpc import TongTestAgentServiceStub
    from arenaagent.utils.config import load_config, set_logger_level

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--run_times", type=int, default=10)
    args = parser.parse_args()

    config = load_config(config_path=args.config)
    set_logger_level(level="debug", log_dir=config.get("counting_agent", {}).get("log_dir", "logs"))
    params = dict(config.get("counting_agent", {}) or {})

    for _ in range(max(1, args.run_times)):
        channel = grpc.insecure_channel(
            (config.get("grpc", {}) or {}).get("endpoint", "127.0.0.1:50051"),
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        stub = TongTestAgentServiceStub(channel)
        time.sleep(2)
        try:
            agent = CountingAgent(stub=stub, channel=channel)
            agent.load(params)
            agent.run()
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning("counting run error: {}", exc)
            continue
        finally:
            channel.close()
            time.sleep(2)


if __name__ == "__main__":
    main()
