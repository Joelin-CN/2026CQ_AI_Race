"""P4 拼图 agent：薄壳挂载队友的确定性推演技能（cqairace.jigsaw_skill）。

技能核心（jigsaw_skill.JigsawSolver，队友 2026-09-15 提供，含 09-14 联调实测）：
- 三层降级：几何推演(格点/空缺恢复) → 语义匹配(VLM 主/CV 兜底) → 执行校验；
- 实测结论已内化：move_and_take_puzzle_piece 未实现(走通用抓取)、
  一步式 move_and_put_down 放置不被评测认可(必须 move_to + put_down_sth
  force_locate 两步式)、assign_mode=fallback(顺序放)最快总分最高；
- 状态机 seek → plan → execute → verify → done，异常降级返回 None。

本壳职责：每 run_step 推进 solver 一步；solver 降级时直接 finish 保底
（不回官方逐 VLM 循环——400s 内逐步 VLM 只会超时低分）。

运行（赛题系统 train jigsaw 已启动，仓库根目录）：
    uv run python -m cqairace.jigsaw_agent --run_times 1
匹配策略可用环境变量 ARENA_JIGSAW_ASSIGN=fallback|auto 覆盖（默认 fallback）。
"""
from __future__ import annotations

import os
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
from arenaagent.vlm_agent.vlm_agent import VLMAgent, VLMAgentCfg  # noqa: E402

from cqairace.jigsaw_skill import JigsawSolver  # noqa: E402


@configclass
class JigsawAgentCfg(VLMAgentCfg):
    name: str = "jigsaw_agent"
    sleep_between_steps: float = 0.3


@Register("jigsaw_agent")
class JigsawAgent(VLMAgent):
    """拼图：JigsawSolver 状态机逐 run_step 推进，降级即 finish 保底。"""

    def __init__(self, stub, channel, cfg: JigsawAgentCfg | None = None,
                 sleep_between_steps: float = 0.3) -> None:
        super().__init__(stub=stub, channel=channel, cfg=cfg or JigsawAgentCfg(),
                         sleep_between_steps=sleep_between_steps)
        self._solver: JigsawSolver | None = None

    def run_step(self, subject, task_response: dict[str, Any]) -> dict[str, Any]:
        if self._solver is None:
            assign = os.environ.get("ARENA_JIGSAW_ASSIGN", "fallback").strip().lower()
            self._solver = JigsawSolver(self, assign_mode=assign)
            logger.info("jigsaw solver 启动 assign_mode={}", assign)
        try:
            action = self._solver.step(subject, task_response)
        except Exception as exc:  # noqa: BLE001  技能整体异常：保底收尾
            logger.opt(exception=True).warning("jigsaw solver 异常: {}", exc)
            action = None
        if action is not None:
            return action
        logger.warning("jigsaw solver 降级，finish 保底")
        return self._do_action({"action": "finish_task",
                                "think": "jigsaw solver 降级保底收尾",
                                "output": 0})

    # 官方式收尾：保持连接轮询 session 至终态再断开（v2 判卷落盘前提）
    def run(self) -> None:
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
            self._call_struct("get_action_space", {"agent_id": self.agent_id},
                              struct_pb2.Struct.FromString)
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
    parser.add_argument("--run_times", type=int, default=1)
    args = parser.parse_args()

    config = load_config(config_path=args.config)
    set_logger_level(level="debug",
                     log_dir=config.get("jigsaw_agent", {}).get("log_dir", "logs"))
    params = dict(config.get("jigsaw_agent", {}) or {})

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
            agent = JigsawAgent(stub=stub, channel=channel)
            agent.load(params)
            agent.run()
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning("jigsaw run error: {}", exc)
            continue
        finally:
            channel.close()
            time.sleep(2)


if __name__ == "__main__":
    main()
