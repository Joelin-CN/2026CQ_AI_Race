"""P2 瑞文采集器/原型 agent：跳过 VLM 循环，直接用官方 ResNet 排序候选快速试答。

用途（train 模式数据采集 + 技能原型验证）：
- 每步：物化题图 → 彩色原图归档到 temp/p2_raven/collected/ → ResNet 全量排序 →
  按名次逐一提交三位数答案（每题最多试 `_MAX_ATTEMPTS` 次）；
- 答对即结束该题（系统揭示我们提交的就是真值）；试答轨迹记 jsonl 供离线分析
  （ResNet 真值名次、每题结构类型）。

运行（仓库根目录，赛题系统 train raven 已启动）：
    uv run python -m cqairace.raven_proto [run_times]
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

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
from arenaagent.vlm_agent.raven_skill import (  # noqa: E402
    group_image_to_raven_list,
    resolve_raven_image_path,
    run_raven_inference,
)
from arenaagent.vlm_agent.vlm_agent import VLMAgent, VLMAgentCfg  # noqa: E402

COLLECT_DIR = _REPO_ROOT / "temp" / "p2_raven" / "collected"
TRACE_PATH = _REPO_ROOT / "temp" / "p2_raven" / "trace.jsonl"


@configclass
class RavenCollectorCfg(VLMAgentCfg):
    name: str = "raven_collector"
    sleep_between_steps: float = 0.5


@Register("raven_collector")
class RavenCollectorAgent(VLMAgent):
    """不走感知与 VLM，按 ResNet 候选名次快速试答。"""

    _MAX_ATTEMPTS = 120  # 400s / (约3s/次) 的安全上限
    _max_subjects = 10  # 单连接最多答多少题（防 session 常驻）

    def __init__(self, stub, channel, cfg: RavenCollectorCfg | None = None, sleep_between_steps: float = 0.5) -> None:
        super().__init__(stub=stub, channel=channel, cfg=cfg or RavenCollectorCfg(), sleep_between_steps=sleep_between_steps)
        self._ranked_cache: dict[str, list[list[int]]] = {}
        self._attempt: dict[str, int] = {}
        self._label_written: set[str] = set()
        self._last_submit_ts: dict[str, float] = {}
        self._last_question_key: str | None = None

    def run_step(self, subject, task_response: dict[str, Any]) -> dict[str, Any]:
        self._before_prompt_hook(subject)  # 物化 task_data 题图
        image_path = resolve_raven_image_path(self._raven_image_temp_path)
        if not image_path:
            logger.warning("raven canvas 不存在，退回 finish")
            return {"action": "finish_task", "output": 0}

        # 彩色原图按内容哈希归档（供离线 VLM 评测与复盘）
        raw = Path(image_path).read_bytes()
        canvas_hash = hashlib.sha256(raw).hexdigest()[:12]
        archive = COLLECT_DIR / f"{canvas_hash}.png"
        if not archive.exists():
            COLLECT_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_path, archive)
        self._last_question_key = canvas_hash

        # 同题限速：题间空转或系统重发同题时，最多 5 秒提交一次
        now = time.time()
        last_ts = self._last_submit_ts.get(canvas_hash, 0.0)
        if now - last_ts < 5.0:
            time.sleep(5.0 - (now - last_ts))
        self._last_submit_ts[canvas_hash] = time.time()

        ranked = self._ranked_for(str(archive))
        if ranked is None:
            logger.warning("ResNet 推理失败：{}", canvas_hash)
            return {"action": "finish_task", "output": 0}

        idx = self._attempt.get(canvas_hash, 0)
        if idx >= min(len(ranked), self._MAX_ATTEMPTS):
            logger.warning("候选耗尽仍未答对：{} rank_of_truth>{}", canvas_hash, idx)
            return {"action": "finish_task", "output": 0}
        self._attempt[canvas_hash] = idx + 1

        chosen = ranked[idx]
        answer = "".join(str(d) for d in chosen)
        key = self.action_space.get("key") or "action"

        # 判卷反馈里若显示答对，则该答案即真值，落档
        right = self._extract_right(task_response)
        self._trace(canvas_hash, idx, answer, ranked, right)
        logger.info("raven try#{} canvas={} answer={} right={}", idx + 1, canvas_hash, answer, right)

        return {key: answer}

    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """单连接答完整场：外层循环连续领题，直到 session 结束或达到题数上限。

        覆盖官方 run()：v2 赛题系统 train 模式下一个 session 连续出 10 题，
        官方 run() 答完一题后轮询 session 状态会永久停在 RUNNING(101)。
        这里答完一题稍等即继续领下一题，结束条件：FINISHED/TERMINATED/ERROR
        或答满 _max_subjects 题。
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

        answered = 0
        last_key = None
        while answered < self._max_subjects:
            self._run_subject()  # 内部结束时已调用 _evaluate_subject()
            answered += 1
            last_key = self._last_question_key
            logger.info("subject {} done ({}/{}), last={}", answered, answered, self._max_subjects, last_key)
            time.sleep(3)  # 给评估器留出写盘时间
            # 等下一题真正就绪（题目变化），避免在已完题上空转提交
            deadline = time.time() + 120
            while time.time() < deadline:
                status = self._get_task_status().get("session_status")
                if status in done_states:
                    break
                s = self._get_subject_from_task()
                key = self._subject_key(s)
                if key and key != last_key:
                    break
                time.sleep(1)
            status = self._get_task_status().get("session_status")
            if status in done_states:
                logger.info("session 终态 {}，收工", status)
                break
        self._disconnect()

    @staticmethod
    def _subject_key(subject: Any) -> str | None:
        if not isinstance(subject, dict):
            return None
        for k in ("question", "subject", "task_id", "id"):
            v = subject.get(k)
            if v:
                return hashlib.sha256(str(v).encode()).hexdigest()[:12]
        return None

    def _ranked_for(self, archive_path: str) -> list[list[int]] | None:
        h = hashlib.sha256(Path(archive_path).read_bytes()).hexdigest()[:12]
        if h in self._ranked_cache:
            return self._ranked_cache[h]
        image_list = group_image_to_raven_list(archive_path)
        if not image_list:
            return None
        ranked = run_raven_inference(image_list=image_list, structure=[])
        if ranked:
            self._ranked_cache[h] = ranked
        return ranked

    @staticmethod
    def _extract_right(task_response: Any) -> bool | None:
        try:
            text = json.dumps(task_response, ensure_ascii=False, default=str)
        except Exception:
            return None
        if '"answer_right": true' in text or "'answer_right': True" in text:
            return True
        if '"answer_right": false' in text or "'answer_right': False" in text:
            return False
        return None

    def _trace(self, canvas: str, idx: int, answer: str, ranked: list[list[int]], right: bool | None) -> None:
        COLLECT_DIR.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.strftime("%H:%M:%S"),
            "canvas": canvas,
            "attempt": idx,
            "answer": answer,
            "right": right,
        }
        with open(TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


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
    set_logger_level(level="debug", log_dir=config.get("raven_collector", {}).get("log_dir", "logs"))

    params = dict(config.get("raven_collector", {}) or {})
    run_times = max(1, args.run_times)
    for _ in range(run_times):
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
            agent = RavenCollectorAgent(stub=stub, channel=channel)
            agent.load(params)
            agent.run()
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning("collector run error: {}", exc)
            continue
        finally:
            channel.close()
            time.sleep(3)


if __name__ == "__main__":
    main()
