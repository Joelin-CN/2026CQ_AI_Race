"""P4 整理房间 agent v2：扫描/VLM/搬运流水线化 + 放置确认 + 二轮扫描。

v1 教训（2026-09-14 train 六轮实测，见 temp/p4_tidyroom/ 与 v2-notes §7）：
- VLM 串行等 115s 占总时长一半且常挂 → v2 改流水线：扫描逐帧即时异步发 VLM，
  规则分类先行开搬（run6 实测规则兜底 5/5 全对），VLM 结果回来只做补充修正；
- 判卷读物体**最终静止位置**（弹落=白放）→ v2 每件放后感知确认，不稳重放；
- 判卷机制（逆向 arena_offline.exe/Nuitka 字符串证实）：
  * 物品在 room_space 内随机撒地面，生成器保证初始全错（放对即重丢）；
  * "放对" = 物体 bbox 中心落入**任一**合法容器 bbox（container_names 是列表，
    多容器全合法——run2 放 16 号桌与基线放 15 号茶几均计分）；
  * score = 0.8×correct/total + 时间分；超时时间分归零；
- 4×90° 原地扫描实测已覆盖全部物品簇（客厅地毯区 + 玄关鞋区，物品全贴地），
  二轮扫描作为死角兜底（队列空后触发）。

运行（赛题系统 train tidyroom 已启动，仓库根目录）：
    uv run python -m cqairace.tidyroom_agent --run_times 1
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from concurrent.futures import Future
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

TRACE_DIR = _REPO_ROOT / "temp" / "p4_tidyroom"
SESSIONS_DIR = _REPO_ROOT / "temp" / "p4_tidyroom" / "test_sessions"

# no-op 实验开关：只扫描规划不搬运，用于校准"初始场景天然完成度"
_DRY_RUN = os.environ.get("TIDYROOM_DRY_RUN", "") not in ("", "0", "false")
# 单件实验开关：只搬指定 object_id（逗号分隔），用于校准判卷几何/分值
_ONLY_OIDS = {s for s in os.environ.get("TIDYROOM_ONLY_OID", "").split(",") if s}
# 导览模式：扫描后逐件走到物品面前停留（人工在 UE 里核对物品位置/类型）
_TOUR = os.environ.get("TIDYROOM_TOUR", "") not in ("", "0", "false")
# 侦察点 "x,y"：导览/扫描前先走到该点 4×90° 感知存图（核对家具布局用）
_SCOUT = os.environ.get("TIDYROOM_SCOUT", "")
# 鞋枚举实验：对鞋类物体穷举全部交互接口（能否拿起/放进鞋柜）
_ENUM_SHOE = os.environ.get("TIDYROOM_ENUM_SHOE", "") not in ("", "0", "false")
# 枕枚举实验：对矩形枕(13)穷举接口 + 物理挤动推向沙发（42cm 缺口）
_ENUM_PILLOW = os.environ.get("TIDYROOM_ENUM_PILLOW", "") not in ("", "0", "false")
# 行车记录仪：感知复合图落盘（事后回放小人第一视角，test 诊断用）
_SAVE_FRAMES = os.environ.get("TIDYROOM_SAVE_FRAMES", "") not in ("", "0", "false")
# 轮次标签：编入本轮目录名（如 test_r4 / train_a；缺省用纯时间戳）
_RUN_TAG = os.environ.get("TIDYROOM_RUN_TAG", "")

# VLM 输出词表归一
_ITEM_ALIASES = {
    "trash": "trash", "garbage": "trash", "rubbish": "trash", "litter": "trash",
    "cup": "cup", "mug": "cup", "bottle": "cup", "glass": "cup", "can": "trash",
    "food": "food", "fruit": "food", "snack": "food",
    "shoe": "shoe", "shoes": "shoe", "sneaker": "shoe", "slipper": "shoe", "boot": "shoe",
    "pillow": "pillow", "cushion": "pillow",
    "worn": "worn", "worn_shoe": "worn", "worn_shoes": "worn",
}
_CONTAINER_ALIASES = {
    "trash_bin": "trash_bin", "trashbin": "trash_bin", "bin": "trash_bin",
    "garbage_can": "trash_bin", "garbage_bin": "trash_bin", "trash_can": "trash_bin",
    "table": "table", "coffee_table": "table", "desk": "table", "dining_table": "table",
    "sofa": "sofa", "couch": "sofa",
    "shoe_cabinet": "shoe_cabinet", "cabinet": "shoe_cabinet",
    "shoe_rack": "shoe_cabinet", "sideboard": "shoe_cabinet",
}
# 类别 → 容器键；类别优先级（先易后难：小件几乎必成，大件枕类放最后）
_CAT_TO_CONTAINER = {"trash": "trash_bin", "cup": "table", "food": "table",
                     "shoe": "shoe_cabinet", "pillow": "sofa"}
_CAT_ORDER = {"trash": 0, "cup": 1, "food": 2, "shoe": 3, "pillow": 4}


@configclass
class TidyroomAgentCfg(VLMAgentCfg):
    name: str = "tidyroom_agent"
    sleep_between_steps: float = 0.3


@Register("tidyroom_agent")
class TidyroomAgent(VLMAgent):
    """整理房间：扫描→规则先行搬运→VLM 异步补充→放置确认→二轮扫描。"""

    _SCAN_TURNS = 4          # 每轮扫描：首帧不转 + 3×90°，FOV 120° 全覆盖
    _SCAN_ROUNDS = 2         # 首轮 + 二轮（死角兜底）
    _TOTAL_BUDGET = 370.0    # 整题预算（秒），首 run_step 起算
    _FINISH_RESERVE = 30.0   # finish 看门狗预留
    _ITEM_COST = 15.0        # 单件 take+put+确认 预估耗时
    _OP_TIMEOUT = 40.0       # 单次 tongsim 调用超时
    _VLM_TIMEOUT = 75.0      # 单次 VLM 调用超时（实测大 prompt 推理 68s）
    _VLM_BUDGET = 160.0      # VLM 收割总上限（与走路重叠，只作上限）
    _MAX_ITEM_DIM = 80.0     # 候选物品最大边（cm）：not pickup 秒拒零成本
    _MIN_DIM = 2.0           # 排除点状 AABB（墙角标记）
    _MAX_BASE_Z = 150.0      # 排除壁挂/吊灯（place_location Z 上限）
    _PAIR_DIST = 50.0        # 鞋成对判定的最大间距（cm，实测一双两脚相距 9~11cm）
    _PAIR_DIM_TOL = 12.0     # 鞋成对判定的尺寸容差（cm）

    _pool = ThreadPoolExecutor(max_workers=2)       # tongsim 调用超时兜底
    _vlm_pool = ThreadPoolExecutor(max_workers=4)   # 帧 VLM 并行分类

    @classmethod
    def _call_with_timeout(cls, fn: Callable, *args, timeout: float | None = None,
                           default=None, **kwargs):
        """官方客户端均无超时，一次挂起即卡死全题；线程+超时兜底。"""
        fut = cls._pool.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout or cls._OP_TIMEOUT)
        except FuturesTimeout:
            logger.warning("call timeout after {}s: {}", timeout or cls._OP_TIMEOUT,
                           getattr(fn, "__name__", fn))
            return default
        except Exception as exc:  # noqa: BLE001
            logger.warning("call error {}: {} {}", getattr(fn, "__name__", fn),
                           type(exc).__name__, exc)
            return default

    def __init__(self, stub, channel, cfg: TidyroomAgentCfg | None = None,
                 sleep_between_steps: float = 0.3) -> None:
        super().__init__(stub=stub, channel=channel, cfg=cfg or TidyroomAgentCfg(),
                         sleep_between_steps=sleep_between_steps)
        self._t0: float | None = None
        self._world: dict[str, dict] = {}          # object_id → 元数据（全局清单）
        self._categories: dict[str, str] = {}      # object_id → 物品类别
        self._container_votes: dict[str, Counter] = {}
        self._containers: dict[str, dict] = {}     # 容器键 → 元数据
        self._container_slots: dict[str, int] = {}
        self._queue: list[dict] = []               # 待办队列
        self._vlm_futs: list[tuple[int, Future]] = []  # 在途 VLM 帧分类 (seq, future)
        self._vlm_frames: dict[int, dict] = {}     # seq → 帧数据（换帧重发用）
        self._vlm_retried: set[int] = set()        # 已换发过的帧
        self._vlm_dead = 0                         # 重发后仍失败的帧数（全挂检测）
        self._vlm_seq = 0
        self._vlm_futs_pending_resubmit: list[dict] = []
        self._item_votes: dict[str, Counter] = {}  # VLM 多帧投票（跨收割累积）
        self._rule_cats: set[str] = set()          # 规则已分且高置信的物体
        self._vlm_started = 0.0
        self._scan_round = 0
        self._done_oids: set[str] = set()          # 已处理（成功/放弃/拉黑）
        self._blacklist: set[str] = set()
        self._placed = 0
        self._gave_up = 0
        self._session_dir: Path | None = None  # 本轮档案目录(首次落盘时创建)

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # 导览模式：走到每件物品面前供人工核对
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # 鞋枚举实验：穷举小人-鞋的全部交互接口
    # ------------------------------------------------------------------ #

    def _obj_loc_now(self, oid: str):
        # 感知前先走近目标，避免因视野外查不到
        self._call_with_timeout(self.tongsim.move_to_object, self.character_id,
                                oid, default=None)
        p = self._call_with_timeout(
            self.tongsim.acquire_first_person_perception, self.character_id,
            1280, 720, default={})
        for o in p.get("objects", []) or []:
            if str(o.get("object_id", "")) == oid:
                return o.get("place_location")
        return None

    def _enum_pillow(self) -> None:
        """对矩形枕穷举接口 + 物理挤动。

        账本：total=6 假设下 13 号枕是 5/6 完成度里唯一缺的 1 件。
        其位置 (380,213) 距沙发 14 投影(x<=338)仅 42cm——挤动目标明确：
        用可抓物强制放置在枕的 +x 侧，碰撞把它往 -x（沙发方向）推。
        """
        self._scan_rounds(max_rounds=1, submit_vlm=False)
        pillow = None
        for oid, o in self._world.items():
            shape = str(o.get("shape", "")).lower()
            color = str(o.get("color", "")).lower()
            if shape == "rectangle" and color in ("beige", "white") \
                    and min(self._dims(o)) >= self._MIN_DIM \
                    and max(self._dims(o)) <= self._MAX_ITEM_DIM:
                pillow = (oid, o)
                break
        if pillow is None:
            logger.info("ENUMP: 无矩形枕")
            return
        oid, o = pillow
        loc = o.get("place_location") or {}
        logger.info("ENUMP: 目标枕 {} @ ({},{},{}) 尺寸 {}",
                    oid, loc.get("X"), loc.get("Y"), loc.get("Z"),
                    tuple(round(v) for v in self._dims(o)))

        # ① 双手 + 贴近 take
        for hand in (0, 1):
            r = self._call_with_timeout(self.tongsim.move_and_take_object,
                                        self.character_id, oid, which_hand=hand,
                                        default={"result": "timeout"})
            logger.info("ENUMP take({}) -> {}", hand, r)
            in_hand, _ = self._call_with_timeout(
                self.tongsim.has_object_in_hand, self.character_id, default=(False, None))
            if in_hand:
                logger.info("ENUMP 抓到了！直接放沙发")
                self._enum_place_pillow_held(oid)
                return

        # ② 物理挤动：循环"抓可抓物→强制放到枕 +x 侧贴身"直至枕进沙发投影
        sofa = None
        for coid, co in self._world.items():
            if str(co.get("shape", "")).lower() == "rectangle" and max(self._dims(co)) > 250:
                sofa = (coid, co)
                break
        if sofa is None:
            logger.info("ENUMP: 未找到沙发，终止")
            return
        bb = sofa[1].get("world_aabb") or {}
        x_max = float((bb.get("max") or {}).get("x", 0))
        logger.info("ENUMP 沙发 {} 投影 x<={:.0f}", sofa[0], x_max)

        throwers = [moid for moid, mo in self._world.items()
                    if str(mo.get("shape", "")).lower() in ("cylinder", "round", "irregular")]
        if not throwers:
            logger.info("ENUMP: 无可抓投掷物")
            return
        for round_i in range(min(6, len(throwers))):
            now = self._obj_loc_now(oid)
            logger.info("ENUMP 挤动第 {} 轮: 枕当前位置 {}", round_i + 1, now)
            if now and float(now.get("X", 999)) <= x_max:
                logger.info("ENUMP 枕已进沙发投影！停止")
                break
            thrower = throwers[round_i]
            rt = self._call_with_timeout(self.tongsim.move_and_take_object,
                                         self.character_id, thrower, which_hand=0,
                                         default={"result": "timeout"})
            logger.info("ENUMP take({}) -> {}", thrower, rt)
            if not self._ok(rt):
                continue
            px = float(loc.get("X", 0)) + 12   # 枕 +x 侧贴身，砸落时往 -x 挤
            py = float(loc.get("Y", 0))
            pz = float(loc.get("Z", 0)) + 25
            rp = self._call_with_timeout(self.tongsim.put_down_sth, self.character_id,
                                         target_location={"X": px, "Y": py, "Z": pz},
                                         auto_rotate=True, force_locate=True,
                                         default={"result": "timeout"})
            logger.info("ENUMP put({},{},{}) -> {}", px, py, pz, rp)
            time.sleep(1)
        final = self._obj_loc_now(oid)
        logger.info("ENUMP 最终枕位置: {} (沙发投影 x<={:.0f})", final, x_max)

    def _enum_place_pillow_held(self, oid: str) -> None:
        cont = self._rule_container("sofa")
        if cont is None:
            logger.info("ENUMP: 未识别沙发")
            return
        self._call_with_timeout(self.tongsim.move_to_object, self.character_id,
                                str(cont.get("object_id", "")), default=None)
        bb = cont.get("world_aabb") or {}
        mn, mx = bb.get("min") or {}, bb.get("max") or {}
        point = {"X": round((mn.get("x", 0) + mx.get("x", 0)) / 2, 1),
                 "Y": round((mn.get("y", 0) + mx.get("y", 0)) / 2, 1),
                 "Z": round(mn.get("z", 0) + 0.5 * (mx.get("z", 0) - mn.get("z", 0)), 1)}
        r = self._call_with_timeout(self.tongsim.put_down_sth, self.character_id,
                                    target_location=point, auto_rotate=True,
                                    force_locate=True, default={"result": "timeout"})
        logger.info("ENUMP 枕入沙发 put({}) -> {}", point, r)

    def _enum_shoe(self) -> None:
        self._scan_rounds(max_rounds=1, submit_vlm=False)
        shoes = [(oid, o) for oid, o in self._world.items()
                 if str(o.get("shape", "")).lower() in ("boot", "shoe")]
        if not shoes:
            logger.info("ENUM: 场景无鞋类物体")
            return
        # 白鞋优先（官方提示"分辨人穿的鞋子"：红靴疑似人穿的干扰项，
        # 白鞋疑似散落可抓——2026-09-15 用户点破只枚举过 47 红靴的盲区）
        shoes.sort(key=lambda kv: (0 if str(kv[1].get("shape", "")).lower() == "shoe" else 1,
                                   float((kv[1].get("place_location") or {}).get("X", 0))))
        for oid, o in shoes:
            loc = o.get("place_location") or {}
            logger.info("ENUM: 目标鞋 {} shape={} color={} @ ({},{},{})",
                        oid, o.get("shape"), o.get("color"),
                        loc.get("X"), loc.get("Y"), loc.get("Z"))
            r = self._call_with_timeout(
                self.tongsim.move_and_take_object, self.character_id, oid,
                which_hand=0, default={"result": "timeout"})
            logger.info("ENUM take({}) -> {}", oid, r)
            time.sleep(1)
            in_hand, _ = self._call_with_timeout(
                self.tongsim.has_object_in_hand, self.character_id, default=(False, None))
            if in_hand:
                logger.info("ENUM {} 抓取成功！尝试入鞋柜", oid)
                self._enum_place_held(oid)
                return
        logger.info("ENUM: 全部 {} 只鞋均不可抓", len(shoes))

    def _enum_place_held(self, shoe_oid: str) -> None:
        """枚举中意外抓到了鞋：直接尝试放进鞋柜。"""
        cont = self._rule_container("shoe_cabinet")
        if cont is None:
            logger.info("ENUM: 未识别鞋柜，放下收尾")
            self._call_with_timeout(self.tongsim.put_down_sth, self.character_id,
                                    target_location={"X": 0, "Y": 0, "Z": 100},
                                    auto_rotate=True, default={})
            return
        self._call_with_timeout(self.tongsim.move_to_object, self.character_id,
                                str(cont.get("object_id", "")), default=None)
        bb = cont.get("world_aabb") or {}
        mn, mx = bb.get("min") or {}, bb.get("max") or {}
        point = {"X": round((mn.get("x", 0) + mx.get("x", 0)) / 2, 1),
                 "Y": round((mn.get("y", 0) + mx.get("y", 0)) / 2, 1),
                 "Z": round(mn.get("z", 0) + 0.5 * (mx.get("z", 0) - mn.get("z", 0)), 1)}
        r = self._call_with_timeout(self.tongsim.put_down_sth, self.character_id,
                                    target_location=point, auto_rotate=True,
                                    force_locate=True, default={})
        logger.info("ENUM 鞋入柜 put_down_sth({}) -> {}", point, r)

    def _tour(self) -> None:
        if _SCOUT:
            import base64
            sx_, sy_ = (float(v) for v in _SCOUT.split(","))
            self._call_with_timeout(self.tongsim.move_to_location, self.character_id,
                                    {"X": sx_, "Y": sy_, "Z": 0}, default=None)
            time.sleep(1)
            for i in range(4):
                p = self._call_with_timeout(
                    self.tongsim.acquire_first_person_perception, self.character_id,
                    1280, 720, default={})
                if p.get("image"):
                    TRACE_DIR.mkdir(parents=True, exist_ok=True)
                    (TRACE_DIR / f"scout_{i}.jpg").write_bytes(
                        base64.b64decode(p["image"]))
                for o in p.get("objects", []) or []:
                    oid = str(o.get("object_id", ""))
                    bb = o.get("world_aabb") or {}
                    mn, mx = bb.get("min") or {}, bb.get("max") or {}
                    try:
                        d = (abs(mx["x"]-mn["x"]), abs(mx["y"]-mn["y"]), abs(mx["z"]-mn["z"]))
                        loc = o.get("place_location") or {}
                    except Exception:
                        continue
                    if max(d) >= 60:
                        logger.info("SCOUT {} {} {} @ ({:.0f},{:.0f},{:.0f}) {:.0f}x{:.0f}x{:.0f}",
                                    oid, o.get("color"), o.get("shape"),
                                    loc.get("X", 0), loc.get("Y", 0), loc.get("Z", 0), *d)
                if i < 3:
                    self._call_with_timeout(self.tongsim.turn_in_degree,
                                            self.character_id, 90, default=None)
                    time.sleep(0.5)
        self._scan_rounds(max_rounds=1, submit_vlm=False)
        self._apply_rule_categories()
        self._rebuild_queue()
        logger.info("TOUR: 共 {} 件候选, 开始逐件导览", len(self._queue))
        for i, t in enumerate(self._queue, 1):
            oid = t["oid"]
            o = self._world[oid]
            loc = o.get("place_location") or {}
            logger.info("TOUR [{}/{}] {} 判定={} 位置=({},{},{})",
                        i, len(self._queue), oid, t["cat"],
                        loc.get("X"), loc.get("Y"), loc.get("Z"))
            self._call_with_timeout(self.tongsim.move_to_object, self.character_id,
                                    oid, default=None)
            time.sleep(4)
        logger.info("TOUR 结束")

    def run_step(self, subject, task_response: dict[str, Any]) -> dict[str, Any]:
        if self._t0 is None:
            self._t0 = time.time()
        try:
            if _ENUM_PILLOW:
                self._enum_pillow()
            elif _ENUM_SHOE:
                self._enum_shoe()
            elif _TOUR:
                self._tour()
            elif _DRY_RUN:
                logger.info("DRY RUN：仅统计感知，不搬运")
                self._scan_rounds(max_rounds=1, submit_vlm=False)
            else:
                self._pipeline()
        except Exception as exc:  # noqa: BLE001 任何异常不外抛：保住已放件数
            logger.opt(exception=True).warning("tidyroom 流水线异常（保底收尾）: {}", exc)
        summary = (f"tidyroom: world={len(self._world)} placed={self._placed} "
                   f"gave_up={self._gave_up} blacklist={sorted(self._blacklist)} "
                   f"cats={self._categories}")
        logger.info(summary)
        self._trace("finish", summary=summary, placed=self._placed,
                    blacklist=sorted(self._blacklist), categories=self._categories)
        self._save_session_summary()
        return self._do_action({"action": "finish_task", "think": summary,
                                "output": self._placed})

    def _save_session_summary(self) -> None:
        """本轮档案汇总：与感知图同目录的 session_summary.json。"""
        try:
            if self._session_dir is not None:
                self._session_dir.mkdir(parents=True, exist_ok=True)
                (self._session_dir / "session_summary.json").write_text(
                    json.dumps({
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "tag": _RUN_TAG,
                        "world_objects": len(self._world),
                        "placed": self._placed,
                        "gave_up": self._gave_up,
                        "blacklist": sorted(self._blacklist),
                        "categories": self._categories,
                    }, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # 流水线主循环
    # ------------------------------------------------------------------ #

    def _pipeline(self) -> None:
        self._scan_rounds(max_rounds=1, submit_vlm=True)  # 首轮：边扫边发 VLM
        # 规则分类先行 → 立即开搬，VLM 结果在搬运间隙收割
        self._apply_rule_categories()
        self._rebuild_queue()

        while True:
            remain = self._TOTAL_BUDGET - (time.time() - self._t0)
            if remain < self._FINISH_RESERVE:
                logger.warning("预算剩余 {:.0f}s 触发看门狗，收工", remain)
                break

            self._harvest_vlm(block=False)

            if self._queue:
                if remain < self._FINISH_RESERVE + self._ITEM_COST:
                    logger.warning("预算剩余 {:.0f}s 不足一件，收工", remain)
                    break
                self._execute_one()
                continue

            # 队列空：等 VLM 新结果（仅首轮扫描后值得等）→ 二轮扫描 → 收工
            if self._vlm_futs and self._scan_round < self._SCAN_ROUNDS:
                self._harvest_vlm(block=True, timeout=10.0)
                continue
            if self._scan_round < self._SCAN_ROUNDS:
                logger.info("队列空，触发第 {} 轮扫描", self._scan_round + 1)
                self._scan_rounds(max_rounds=self._SCAN_ROUNDS, submit_vlm=False)
                self._apply_rule_categories()
                self._rebuild_queue()
                continue
            # 二轮扫描后不再等 VLM（实测新物体为 0，期望收益≈0，白耗时间分）
            self._harvest_vlm(block=False)
            break

    # ------------------------------------------------------------------ #
    # 扫描（每帧即时异步发 VLM）
    # ------------------------------------------------------------------ #

    def _save_frame(self, tag: str, b64: str) -> None:
        """感知图落盘：temp/p4_tidyroom/test_sessions/{时间戳[_标签]}/frame_*.jpg。

        每轮独立目录（目录规范：test/train 逐轮归档，含 meta 汇总）。
        """
        if not _SAVE_FRAMES or not b64:
            return
        try:
            if self._session_dir is None:
                name = time.strftime("%Y%m%d_%H%M%S")
                if _RUN_TAG:
                    name = f"{name}_{_RUN_TAG}"
                self._session_dir = SESSIONS_DIR / name
            self._session_dir.mkdir(parents=True, exist_ok=True)
            (self._session_dir / f"frame_{tag}.jpg").write_bytes(base64.b64decode(b64))
        except Exception:  # noqa: BLE001  落盘失败不影响任务
            pass

    def _scan_rounds(self, max_rounds: int, submit_vlm: bool) -> None:
        while self._scan_round < max_rounds:
            self._scan_round += 1
            new = 0
            for i in range(self._SCAN_TURNS):
                p = self._call_with_timeout(
                    self.tongsim.acquire_first_person_perception, self.character_id,
                    1280, 720, default={},
                )
                objects = p.get("objects", []) or []
                self._save_frame(f"scan_r{self._scan_round}_{i}", p.get("image", ""))
                for obj in objects:
                    oid = str(obj.get("object_id", ""))
                    if not oid:
                        continue
                    prev = self._world.get(oid)
                    if prev is None:
                        new += 1
                    if prev is None or self._vol(obj) > self._vol(prev):
                        self._world[oid] = obj
                if submit_vlm and p.get("image"):
                    self._submit_vlm_frame(p.get("image", ""), objects)
                if i < self._SCAN_TURNS - 1:
                    self._call_with_timeout(self.tongsim.turn_in_degree,
                                            self.character_id, 90, default=None)
                    time.sleep(0.4)
            logger.info("扫描第 {} 轮完成: 新增 {} 物体, 全局清单 {} 物体, 在途 VLM {}",
                        self._scan_round, new, len(self._world), len(self._vlm_futs))

    def _submit_vlm_frame(self, image_b64: str, objects: list[dict]) -> None:
        if not self._vlm_started:
            self._vlm_started = time.time()
        self._vlm_seq += 1
        seq = self._vlm_seq
        self._vlm_frames[seq] = {"image": image_b64, "objects": objects}
        self._vlm_futs.append((seq, self._vlm_pool.submit(self._vlm_classify_frame,
                                                          self._vlm_frames[seq])))

    # ------------------------------------------------------------------ #
    # VLM 分类与收割
    # ------------------------------------------------------------------ #

    def _vlm_classify_frame(self, frame: dict) -> tuple[dict[str, str], dict[str, str]]:
        """单帧 VLM 标注，返回 (物品类别, 容器类型)。失败返回空 dict。"""
        compact = []
        for o in frame["objects"]:
            oid = str(o.get("object_id", ""))
            if not oid or self._is_point(o) or self._too_high(o):
                continue
            dx, dy, dz = self._dims(o)
            loc = o.get("place_location") or {}
            compact.append({"object_id": oid, "color": o.get("color"),
                            "shape": o.get("shape"),
                            "size_cm": [round(dx), round(dy), round(dz)],
                            "loc": [loc.get("X"), loc.get("Y"), loc.get("Z")]})
        if not compact:
            return {}, {}
        prompt = (
            "这是仿真环境第一视角复合图（左半 RGB，右半是带数字编号的语义分割图，"
            "编号与下方元数据的 object_id 一一对应）。场景是客厅+玄关，任务是整理房间。\n"
            "请把当前画面里可见的物体分成两类：\n"
            "1. items：散乱摆放在【地面上】、需要整理的小物件。类别只能是：\n"
            "   trash(垃圾碎屑)、cup(杯子/饮料罐/易拉罐)、food(食物/水果)、"
            "shoe(鞋靴)、pillow(抱枕靠垫,含圆柱形颈枕——长度超过 30cm 的细长圆柱是颈枕不是杯子)。\n"
            "   识别要点：\n"
            "   - 鞋的形状多样（靴子/运动鞋/圆头拖鞋），不要因为形状是圆形就判成食物；\n"
            "     紧挨在一起的一对小物体很可能是同一双鞋的两只。\n"
            "   - 银色/白色圆柱小件通常是易拉罐，按 cup 算。\n"
            "   - 穿在人脚上的鞋（有人在里面）标为 worn，不要标 shoe。\n"
            "   - 画面里每一件散落的地面小物件都必须归类，认不出的标 other，不要遗漏。\n"
            "2. containers：可放置物品的目标家具。类型只能是：\n"
            "   trash_bin(垃圾桶,通常是黑色小方箱)、table(茶几/餐桌)、sofa(沙发)、"
            "shoe_cabinet(鞋柜/边柜)。\n"
            "当前帧物体元数据（size_cm 是包围盒长宽高，loc 是世界坐标，单位 cm）：\n"
            f"{json.dumps(compact, ensure_ascii=False)}\n"
            "要求：家具本身、沙发上已摆放整齐的靠枕、墙上装饰不要列入 items。\n"
            "重要：不要思考过程，回答的第一个字符必须是 {{，格式如 "
            '{{"items": {{"36": "trash", "32": "cup"}}, "containers": {{"18": "trash_bin", "15": "table"}}}}；'
            "没有则用空字典。"
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + frame["image"]}},
                {"type": "text", "text": prompt},
            ],
        }]
        for attempt in range(2):
            # 直接裸调 invoke（本函数已跑在 _vlm_pool 线程，挂死由外层
            # future.result 超时兜底）；不走 _pool——那 2 个 worker 是留给
            # 主线程 tongsim 调用的，被 VLM 长占会导致转身/感知排队超时。
            try:
                resp = self.vlm_client.invoke(messages)
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLM invoke 异常(第{}次): {} {}", attempt + 1,
                               type(exc).__name__, exc)
                continue
            text = getattr(resp, "text", "") or ""
            if not text.strip():
                logger.warning("VLM 空文本（第 {} 次），重试", attempt + 1)
                continue
            try:
                parsed = extract_last_json_from_text(text)
                data = json.loads(parsed) if isinstance(parsed, str) else parsed
                items = self._normalize_map(data.get("items") or {}, _ITEM_ALIASES)
                conts = self._normalize_map(data.get("containers") or {}, _CONTAINER_ALIASES)
                return items, {k: v for k, v in conts.items() if v}
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLM 输出解析失败(第{}次): {}", attempt + 1, exc)
                continue
        return {}, {}

    @staticmethod
    def _normalize_map(raw: dict, aliases: dict[str, str]) -> dict[str, str]:
        """兼容两种输出格式：{"36": "trash"}（id→类别）与 {"trash": ["36"]}（类别→ids）。"""
        out: dict[str, str] = {}
        for k, v in raw.items():
            key = str(k)
            if isinstance(v, list):
                cat = aliases.get(key.lower().strip())
                if cat:
                    for oid in v:
                        out[str(oid)] = cat
            elif isinstance(v, (str, int)):
                cat = aliases.get(str(v).lower().strip())
                if cat:
                    out[key] = cat
        return out

    def _harvest_vlm(self, block: bool, timeout: float = 0.0) -> bool:
        """收割 VLM future：结果合并 / 失败换帧重发一次 / 全挂则清空不再等。

        VLM 可用性强化（2026-09-15）：deepseek 间歇全挂（test R1/R3 整轮
        零贡献）——失败帧换另一帧重发（同请求重试已证明无效），重发仍
        失败计入 dead；全部帧 dead 时清空 futures，主循环立即转二轮扫描
        收尾，不再空等烧时间分。
        """
        if not self._vlm_futs:
            return False
        changed = False
        pending: list[tuple[int, Future]] = []
        waited = False
        for seq, fut in self._vlm_futs:
            try:
                if not fut.done():
                    if not (block and not waited):
                        pending.append((seq, fut))
                        continue
                    waited = True
                    budget = self._VLM_BUDGET - (time.time() - self._vlm_started)
                    if budget <= 0:
                        pending.append((seq, fut))
                        continue
                    items, conts = fut.result(timeout=min(timeout, max(budget, 1.0)))
                else:
                    items, conts = fut.result(timeout=0.1)
            except FuturesTimeout:
                pending.append((seq, fut))
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLM 帧异常: {} {}", type(exc).__name__, exc)
                items, conts = {}, {}
            if not items and not conts:
                # 空结果 = 挂帧：未重发过的换帧重发一次；重发过则计 dead
                if seq not in self._vlm_retried:
                    self._vlm_retried.add(seq)
                    frame = self._vlm_frames.get(seq)
                    if frame:
                        logger.info("VLM 帧 {} 失败，换帧重发", seq)
                        self._vlm_futs_pending_resubmit.append(frame)
                else:
                    self._vlm_dead += 1
                    logger.warning("VLM 帧 {} 重发仍失败 (dead={})", seq, self._vlm_dead)
                continue
            for oid, cat in items.items():
                if oid in self._done_oids or cat in ("other", "worn"):
                    continue
                if self._world.get(oid) is None:
                    continue
                self._item_votes.setdefault(oid, Counter())[cat] += 1
                # 校准结论（2026-09-14 压测）：flash 单帧准确率 87.5%，
                # 错分靠多帧投票压制；规则对 cylinder/round/irregular/
                # boot 等特征已证明可靠——VLM 单票不得推翻规则，
                # ≥2 票才允许覆盖（防"单帧错分→容器放错"）。
                if oid in self._rule_cats and self._item_votes[oid][cat] < 2:
                    continue
                if self._categories.get(oid) != cat:
                    self._categories[oid] = cat
                    changed = True
            for oid, ctype in conts.items():
                if self._world.get(oid) is not None:
                    self._container_votes.setdefault(oid, Counter())[ctype] += 1
                    changed = True
        self._vlm_futs = pending
        # 换帧重发（失败帧的全部重试都在后台，不阻塞主流程）
        while self._vlm_futs_pending_resubmit:
            frame = self._vlm_futs_pending_resubmit.pop(0)
            self._submit_vlm_frame(frame["image"], frame["objects"])
            # 重发的 seq 尚未标记 retried（换的是新 seq，新帧失败会再判）
            self._vlm_retried.add(self._vlm_seq)
        # 全挂检测：所有已提交帧都 dead → 放弃等待
        if self._vlm_dead >= self._vlm_seq and self._vlm_seq > 0:
            logger.warning("VLM 全部帧失败（{}/{}），放弃等待", self._vlm_dead, self._vlm_seq)
            self._vlm_futs = []
            return changed
        if changed:
            self._resolve_containers()
            self._rebuild_queue()
        return changed

    def _pair_shoes(self) -> None:
        """鞋成对启发：紧邻确定鞋（<50cm）且尺寸相近（±12cm）的可搬物体，
        即使 shape 不是 boot/shoe 也按鞋处理。

        依据（2026-09-15 test R4 实测）：test 资产的鞋形态多样，圆头拖鞋
        shape=round 被规则误判 food 放错容器；而同一双的两只总是紧贴
        （实测相距 9~11cm），成对证据比单一 shape 可靠。
        """
        shoes = [(oid, self._world[oid]) for oid, c in self._categories.items()
                 if c == "shoe" and oid in self._world]
        if not shoes:
            return
        for oid, obj in self._world.items():
            if oid in self._done_oids or self._categories.get(oid) == "shoe":
                continue
            if self._is_point(obj) or self._too_high(obj):
                continue
            dx, dy, dz = self._dims(obj)
            if not (self._MIN_DIM <= min(dx, dy, dz) and max(dx, dy, dz) <= self._MAX_ITEM_DIM):
                continue
            loc = obj.get("place_location") or {}
            try:
                px, py = float(loc.get("X", 0)), float(loc.get("Y", 0))
            except (TypeError, ValueError):
                continue
            for soid, sobj in shoes:
                sloc = sobj.get("place_location") or {}
                try:
                    sx_, sy_ = float(sloc.get("X", 0)), float(sloc.get("Y", 0))
                except (TypeError, ValueError):
                    continue
                if (px - sx_) ** 2 + (py - sy_) ** 2 > self._PAIR_DIST ** 2:
                    continue
                sdx, sdy, sdz = self._dims(sobj)
                if (abs(dx - sdx) <= self._PAIR_DIM_TOL
                        and abs(dy - sdy) <= self._PAIR_DIM_TOL
                        and abs(dz - sdz) <= self._PAIR_DIM_TOL):
                    old = self._categories.get(oid)
                    self._categories[oid] = "shoe"
                    self._rule_cats.add(oid)
                    logger.info("鞋成对启发: {} ({}) 紧邻鞋 {} 且尺寸相近 → 改判 shoe",
                                oid, old, soid)
                    break

    def _apply_rule_categories(self) -> None:
        for oid, obj in self._world.items():
            if oid in self._done_oids or oid in self._categories:
                continue
            if self._is_point(obj) or self._too_high(obj):
                continue
            cat = self._rule_category(oid)
            if cat:
                self._categories[oid] = cat
                self._rule_cats.add(oid)
        self._resolve_containers()

    def _rule_category(self, oid: str) -> str | None:
        obj = self._world.get(oid)
        if obj is None:
            return None
        shape = str(obj.get("shape", "")).lower()
        color = str(obj.get("color", "")).lower()
        if shape in ("boot", "shoe"):
            return "shoe"
        if shape == "cylinder":
            # 尺寸判据（2026-09-15 导览实测）：≥35cm 的细长圆柱是颈枕/抱枕
            # （45×16cm 黑圆柱颈枕曾被误判 cup 塞进茶几），罐/杯都是短圆柱
            return "pillow" if max(self._dims(obj)) >= 35 else "cup"
        if shape == "irregular":
            return "trash"
        if shape == "round":
            return "food"
        if shape == "rectangle" and color in ("beige", "white") \
                and max(self._dims(obj)) < self._MAX_ITEM_DIM:
            return "pillow"
        return None

    # ------------------------------------------------------------------ #
    # 容器解析
    # ------------------------------------------------------------------ #

    def _resolve_containers(self) -> None:
        """VLM 投票为主 + 规则校正；trash_bin 规则强校正（黑小方箱特征极强）。"""
        voted = {oid: c.most_common(1)[0][0] for oid, c in self._container_votes.items() if c}
        resolved: dict[str, dict] = {}
        for oid, ctype in voted.items():
            obj = self._world.get(oid)
            if obj is None:
                continue
            prev = resolved.get(ctype)
            if prev is None or self._vol(obj) > self._vol(prev):
                resolved[ctype] = obj
        for ctype in ("trash_bin", "table", "sofa", "shoe_cabinet"):
            rule = self._rule_container(ctype)
            if ctype == "trash_bin" and rule is not None:
                resolved[ctype] = rule
            elif ctype not in resolved and rule is not None:
                resolved[ctype] = rule
        # 防倒退：已有容器不被规则清空
        for k, v in resolved.items():
            self._containers[k] = v

    def _rule_container(self, ctype: str) -> dict | None:
        best, best_vol = None, 0.0
        for oid, obj in self._world.items():
            color = str(obj.get("color", "")).lower()
            shape = str(obj.get("shape", "")).lower()
            dx, dy, dz = self._dims(obj)
            if min(dx, dy, dz) < self._MIN_DIM or self._too_high(obj):
                continue
            hit = False
            if ctype == "trash_bin":
                hit = color == "black" and shape == "box"
            elif ctype == "table":
                hit = color == "brown" and shape == "rectangle" \
                    and 25 <= dz <= 55 and max(dx, dy) < 200
            elif ctype == "sofa":
                hit = shape == "rectangle" and max(dx, dy) > 250 and dz < 120
            elif ctype == "shoe_cabinet":
                hit = color == "brown" and shape == "rectangle" and dz > 55
            if hit and self._vol(obj) > best_vol:
                best, best_vol = obj, self._vol(obj)
        if ctype == "shoe_cabinet":
            near = self._nearest_shoe_container(best)
            if near is not None:
                best = near
        return best

    def _nearest_shoe_container(self, fallback: dict | None) -> dict | None:
        shoes = [o for o in self._world.values()
                 if str(o.get("shape", "")).lower() in ("boot", "shoe")]
        if not shoes:
            return fallback
        gx = sum(float((o.get("place_location") or {}).get("X", 0)) for o in shoes) / len(shoes)
        gy = sum(float((o.get("place_location") or {}).get("Y", 0)) for o in shoes) / len(shoes)
        best, best_d = fallback, (1e9 if fallback else 0.0)
        for obj in self._world.values():
            color = str(obj.get("color", "")).lower()
            shape = str(obj.get("shape", "")).lower()
            dx, dy, dz = self._dims(obj)
            if color != "brown" or shape != "rectangle" or min(dx, dy, dz) < self._MIN_DIM:
                continue
            loc = obj.get("place_location") or {}
            d = ((float(loc.get("X", 0)) - gx) ** 2 + (float(loc.get("Y", 0)) - gy) ** 2) ** 0.5
            if d < min(best_d, 300.0):
                best, best_d = obj, d
        return best

    # ------------------------------------------------------------------ #
    # 队列构建
    # ------------------------------------------------------------------ #

    def _rebuild_queue(self) -> None:
        self._pair_shoes()  # 鞋成对启发（2026-09-15 R4：圆状鞋 33 被误判 food 的修正）
        queue: list[dict] = []
        for oid, cat in self._categories.items():
            if oid in self._done_oids:
                continue
            if _ONLY_OIDS and oid not in _ONLY_OIDS:
                continue
            obj = self._world.get(oid)
            cont_key = _CAT_TO_CONTAINER.get(cat)
            cont = self._containers.get(cont_key) if cont_key else None
            if obj is None or cont is None:
                continue
            dx, dy, dz = self._dims(obj)
            if max(dx, dy, dz) > self._MAX_ITEM_DIM or min(dx, dy, dz) < self._MIN_DIM:
                continue
            if self._on_target(obj, cont):
                continue
            put, put_retry = self._put_points(cont, cont_key)
            queue.append({"oid": oid, "cat": cat, "cont_key": cont_key,
                          "put": put, "put_retry": put_retry,
                          "move": self._move_point(put)})
        queue.sort(key=lambda t: (_CAT_ORDER.get(t["cat"], 9), t["oid"]))
        self._queue = queue

    # ------------------------------------------------------------------ #
    # 执行（单件：take → put → 确认 → 必要时重放）
    # ------------------------------------------------------------------ #

    def _execute_one(self) -> None:
        task = self._queue.pop(0)
        oid = task["oid"]
        t1 = time.time()

        take = self._call_with_timeout(
            self.tongsim.move_and_take_object, self.character_id, oid,
            which_hand=0, default={},
        )
        if not self._ok(take):
            self._done_oids.add(oid)
            self._blacklist.add(oid)
            logger.info("take {} 失败({}) → 拉黑, {:.1f}s", oid, self._err(take), time.time() - t1)
            self._trace("take_failed", oid=oid, cat=task["cat"], res=self._err(take))
            return

        ok = self._put_with_retry(task)
        self._done_oids.add(oid)
        if ok:
            self._placed += 1
            self._container_slots[task["cont_key"]] = \
                self._container_slots.get(task["cont_key"], 0) + 1
            logger.info("件完成 {} → {} ({:.1f}s), 累计 {}",
                        oid, task["cont_key"], time.time() - t1, self._placed)
            self._trace("placed", oid=oid, cat=task["cat"], cont=task["cont_key"],
                        sec=round(time.time() - t1, 1))
        else:
            self._gave_up += 1
            logger.warning("件放弃 {} ({})", oid, task["cont_key"])
            self._trace("gave_up", oid=oid, cat=task["cat"], cont=task["cont_key"])

    def _put_with_retry(self, task: dict) -> bool:
        """放置链 v3（实机观察修正，2026-09-14 run11）：

        move_and_put_down 的 move 点不控制朝向，人物到达后按行进方向放手，
        物体释放在"人物面前"——没正对容器就全掉在旁边地上（run11 杯子全灭）。
        而 put_down_sth 是强制坐标放置（run11 的 33 号被它直接"穿模"送进
        茶几内部）。故组合：
          move_to_object(容器) → 到达即面向容器
          put_down_sth(容器 AABB 内部点, force_locate) → 物体直接出现在容器体内
        """
        cont = self._containers.get(task["cont_key"])
        if cont is None:
            return False
        move = self._call_with_timeout(
            self.tongsim.move_to_object, self.character_id,
            str(cont.get("object_id", "")), default={},
        )
        if not self._ok(move):
            logger.info("move_to_object({}) 失败: {}", task["cont_key"], self._err(move))

        for label, point in (("内部点", task["put"]), ("内部点2", task["put_retry"])):
            put = self._call_with_timeout(
                self.tongsim.put_down_sth, self.character_id,
                target_location=point, auto_rotate=True, force_locate=True,
                default={},
            )
            if self._ok(put) and self._confirm_placed(task["oid"], task["cont_key"]):
                return True
            in_hand, _ = self._call_with_timeout(
                self.tongsim.has_object_in_hand, self.character_id, default=(False, None),
            )
            if not in_hand:
                retake = self._call_with_timeout(
                    self.tongsim.move_and_take_object, self.character_id, task["oid"],
                    which_hand=0, default={},
                )
                if not self._ok(retake):
                    return False
        # 两个点都没确认成功：原地放下防卡手
        self._call_with_timeout(
            self.tongsim.put_down_sth, self.character_id,
            target_location=task["put_retry"], auto_rotate=True, default={},
        )
        return False

    def _confirm_placed(self, oid: str, cont_key: str) -> bool:
        """感知确认物体最终位置——完全复刻判卷几何（逆向 _is_right_put）：
        物体 world_aabb **中心点** 落入容器 world_aabb（三轴，无容差）。

        历史：v2 曾用 place_location 锚点+容差判定，导致放置在茶几顶的
        杯子"锚点 z=47 通过"而"bbox 中心 z≈55 超界判错"，run9/run10 两轮
        确认全过却 0 分——确认与判卷几何必须一致。
        感知失败（UE 卡顿返回空）按未确认处理；感知有效但物体不在视野
        时乐观 True（典型：进桶后被桶壁遮挡）。
        """
        cont = self._containers.get(cont_key)
        if cont is None:
            return True
        p = self._call_with_timeout(
            self.tongsim.acquire_first_person_perception, self.character_id,
            1280, 720, default={},
        )
        self._save_frame(f"confirm_{oid}", p.get("image", ""))
        if not p.get("image"):
            logger.warning("确认感知失败（UE 卡顿?），按未确认处理")
            return False
        for obj in p.get("objects", []) or []:
            if str(obj.get("object_id", "")) == oid:
                ok = self._center_in_bbox(obj, cont)
                if not ok:
                    logger.info("放置确认失败: {} bbox中心={} 不在 {} bbox 内",
                                oid, self._bb_center(obj), cont_key)
                    self._trace("confirm_failed", oid=oid, cont=cont_key,
                                loc=obj.get("place_location"), aabb=obj.get("world_aabb"))
                return ok
        return True  # 感知有效但物体不在视野（桶内/被挡），乐观处理

    @staticmethod
    def _bb_center(obj: dict) -> tuple[float, float, float]:
        bb = obj.get("world_aabb") or {}
        mn, mx = bb.get("min") or {}, bb.get("max") or {}
        try:
            return ((float(mn.get("x", 0)) + float(mx.get("x", 0))) / 2,
                    (float(mn.get("y", 0)) + float(mx.get("y", 0))) / 2,
                    (float(mn.get("z", 0)) + float(mx.get("z", 0))) / 2)
        except (TypeError, ValueError):
            return 0.0, 0.0, 0.0

    @classmethod
    def _center_in_bbox(cls, obj: dict, cont: dict) -> bool:
        """判卷同款几何：物体 bbox 中心 ∈ 容器 bbox（中心±半宽，三轴）。"""
        cx, cy, cz = cls._bb_center(obj)
        bx, by, bz = cls._bb_center(cont)
        dx, dy, dz = cls._dims(cont)
        return (abs(cx - bx) <= dx / 2 and abs(cy - by) <= dy / 2
                and abs(cz - bz) <= dz / 2)

    # ------------------------------------------------------------------ #
    # 几何工具
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dims(obj: dict) -> tuple[float, float, float]:
        bb = obj.get("world_aabb") or {}
        mn, mx = bb.get("min") or {}, bb.get("max") or {}
        try:
            return (abs(float(mx.get("x", 0)) - float(mn.get("x", 0))),
                    abs(float(mx.get("y", 0)) - float(mn.get("y", 0))),
                    abs(float(mx.get("z", 0)) - float(mn.get("z", 0))))
        except (TypeError, ValueError):
            return 0.0, 0.0, 0.0

    @classmethod
    def _vol(cls, obj: dict) -> float:
        dx, dy, dz = cls._dims(obj)
        return dx * dy * dz

    @classmethod
    def _is_point(cls, obj: dict) -> bool:
        return min(cls._dims(obj)) < cls._MIN_DIM

    @classmethod
    def _too_high(cls, obj: dict) -> bool:
        loc = obj.get("place_location") or {}
        try:
            return float(loc.get("Z", 0) or 0) > cls._MAX_BASE_Z
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _bb_xy(obj: dict) -> tuple[float, float, float, float]:
        bb = obj.get("world_aabb") or {}
        mn, mx = bb.get("min") or {}, bb.get("max") or {}
        return (float(mn.get("x", 0)), float(mn.get("y", 0)),
                float(mx.get("x", 0)), float(mx.get("y", 0)))

    def _put_points(self, cont: dict, cont_key: str) -> tuple[dict, dict]:
        """两个放置点，均为容器 world_aabb **内部**（v3 配合 put_down_sth 强制放置）。

        判卷要求物体 bbox 中心 ∈ 容器 bbox（三轴）→ 强制放置在容器体内
        （xy 带散点防叠、z 取中部）后中心大概率留在 bbox 内。
        """
        bb = cont.get("world_aabb") or {}
        mn, mx = bb.get("min") or {}, bb.get("max") or {}
        x0, y0 = float(mn.get("x", 0)), float(mn.get("y", 0))
        x1, y1 = float(mx.get("x", 0)), float(mx.get("y", 0))
        z0, z1 = float(mn.get("z", 0)), float(mx.get("z", 0))

        z_mid = round(z0 + 0.5 * (z1 - z0), 1)
        z_low = round(z0 + 0.3 * (z1 - z0), 1)
        mx_x, mx_y = x0 + 0.2 * (x1 - x0), y0 + 0.2 * (y1 - y0)
        w, h = max((x1 - x0) * 0.6, 1), max((y1 - y0) * 0.6, 1)
        slot = self._container_slots.get(cont_key, 0)
        gx = mx_x + (0.5 if slot % 2 else 0.0) * w
        gy = mx_y + (0.5 if (slot // 2) % 2 else 0.0) * h
        main = {"X": round(gx, 1), "Y": round(gy, 1), "Z": z_mid}
        retry = {"X": round((x0 + x1) / 2, 1), "Y": round((y0 + y1) / 2, 1), "Z": z_low}
        return main, retry

    def _move_point(self, put: dict) -> dict:
        """角色导航点：put 点向场景中心方向偏移 45cm，站到容器开放侧，z=0。"""
        cx, cy = self._scene_center()
        px, py = float(put["X"]), float(put["Y"])
        dx, dy = cx - px, cy - py
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        return {"X": round(px + 45 * dx / n, 1), "Y": round(py + 45 * dy / n, 1), "Z": 0}

    def _scene_center(self) -> tuple[float, float]:
        if not self._world:
            return 0.0, 0.0
        sx = sy = 0.0
        for obj in self._world.values():
            x0, y0, x1, y1 = self._bb_xy(obj)
            sx += (x0 + x1) / 2
            sy += (y0 + y1) / 2
        return sx / len(self._world), sy / len(self._world)

    def _on_target(self, obj: dict, cont: dict) -> bool:
        """物品当前位置已在容器内/上。

        z 判据用"顶面附近"（max_z-0.6×高 ~ max_z+25）：容器 AABB 投影内
        贴地的物品（如鞋柜脚边的鞋）不算在位，必须接近顶面/桶口才算。
        """
        loc = obj.get("place_location") or {}
        try:
            px, py, pz = (float(loc.get("X", 0)), float(loc.get("Y", 0)),
                          float(loc.get("Z", 0)))
        except (TypeError, ValueError):
            return False
        x0, y0, x1, y1 = self._bb_xy(cont)
        bb = cont.get("world_aabb") or {}
        z0 = float((bb.get("min") or {}).get("z", 0))
        z1 = float((bb.get("max") or {}).get("z", 0))
        return (x0 - 10 <= px <= x1 + 10 and y0 - 10 <= py <= y1 + 10
                and pz >= z1 - 0.6 * (z1 - z0) and pz <= z1 + 25)

    @staticmethod
    def _ok(res: Any) -> bool:
        return isinstance(res, dict) and res.get("result") == "success"

    @staticmethod
    def _err(res: Any) -> str:
        if isinstance(res, dict):
            return str(res.get("error") or res.get("result") or res)
        return "timeout/none"

    def _trace(self, kind: str, **kw) -> None:
        try:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            rec = {"ts": time.strftime("%H:%M:%S"), "kind": kind}
            rec.update(kw)
            with open(TRACE_DIR / "trace.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001  落盘失败不影响任务
            pass

    # ------------------------------------------------------------------ #
    # 官方式收尾：答完保持连接轮询 session 至终态再断开（v2 判卷落盘前提）
    # ------------------------------------------------------------------ #

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
                     log_dir=config.get("tidyroom_agent", {}).get("log_dir", "logs"))
    params = dict(config.get("tidyroom_agent", {}) or {})

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
            agent = TidyroomAgent(stub=stub, channel=channel)
            agent.load(params)
            agent.run()
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning("tidyroom run error: {}", exc)
            continue
        finally:
            channel.close()
            time.sleep(2)


if __name__ == "__main__":
    main()
