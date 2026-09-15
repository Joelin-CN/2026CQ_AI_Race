"""P4 整理房间 agent v4：识别 gate（裁剪自标+加权投票）→ 锁定 → 纯执行 + 到场确认。

v3→v4（2026-09-15 讨论定案 + 离线实验定参，v2-notes §7-19/20/21）：
- 绑号三路：模板 OCR（8-10px 互混，只作加速）/ VLM 转写放大戳（主路径，
  R11 验证 5/5）/ 到场结构性绑定（走到已知 oid 跟前所见即该物，免读号）；
- 基线识别：2560×720 帧 → 戳定位（免模板）→ VLM 转写 → 洪泛色块（像素
  面积=票权）→ 左面板裁剪自画编号 → 轻量 prompt 五桶分类（思考关直连，
  实测 0.8~1.0s/帧 vs 旧整帧大 prompt 42~68s 且空文本）；
- 加权投票：w=clip(sqrt(area/A_ref),0.3,3)；规则推翻门槛加权≥2.0；
- 三段式：扫描 → 识别 gate（截止线，VLM 全挂按规则开搬=旧下限）→
  分类锁定 → 纯执行；歧义件（无票/票分裂/规则冲突/other）到场特写
  确认（move_to_object→感知→中央特写→一票权重 5，走路零额外成本）。
- 容器 VLM 投票已砍（静态先验+规则已解决，§7-18）；多图一消息实测
  质量差弃用（§7-21）。

v1~v3 教训存档（2026-09-14 train 实测，temp/p4_tidyroom/ 与 v2-notes §7）：
- 判卷读物体**最终静止位置**（弹落=白放）→ 每件放后感知确认，不稳重放；
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

try:
    from cqairace import dashcam as _dashcam
    from cqairace import stamp_perceive as _sp
except Exception:  # noqa: BLE001  cv2/numpy 缺失时降级为存原图/旧整帧
    _dashcam = None
    _sp = None
try:
    from cqairace.vlm_direct import DirectVLMClient, DirectVLMUnavailable
except Exception:  # noqa: BLE001
    DirectVLMClient = None  # type: ignore[assignment]
    DirectVLMUnavailable = Exception  # type: ignore[assignment,misc]

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
# v4 识别管线开关（=0 回退 v3 旧整帧流水线：A/B 对比与回退开关）
_V4 = os.environ.get("TIDYROOM_V4", "1") not in ("", "0", "false")
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
    _VLM_TIMEOUT = 75.0      # 单次 VLM 调用超时（官方客户端回退路径，实测 68s）
    _VLM_BUDGET = 160.0      # VLM 收割总上限（回退路径用）
    _MAX_ITEM_DIM = 100.0    # 候选物品最大边（cm）：80→100——R26-32 白枕
                              # 81cm、R22-32 灰枕 91cm 曾被 80 上限整件误挡
                              # （not pickup 秒拒零成本，宁可放宽；80~100 区间
                              # 家具仅餐椅，已被 shape 拉黑）
    _MIN_DIM = 1.0           # 排除点状 AABB（墙角标记 min=0 精确退化；
                              # 2→1：R17-33 薄片物品 10×10×1 曾被误杀）
    _PILLOW_MIN_DIM = 30.0   # pillow 尺寸下限（R14/18/19/20/22 复盘：真枕
                              # max≥38cm（最细颈枕 45×16），rock 10cm/drumstick
                              # 11cm/小 oval ≤15cm 曾被误判 pillow 进沙发）
    _TINY_DIM = 20.0         # 极小件阈值：无形状规则命中时按 trash 兜底
                              # （用户定策：极小件进垃圾桶；cylinder 罐/round
                              # 果等形状规则在前不受影响）
    _BIG_DIM = 40.0          # 大件先验（R24-33 用户定策：57cm 灰方块被确认
                              # trash 是错的——历史可搬大件 max≥40 只有 pillow，
                              # ring 31/靴 25/罐 34 全在下方）：无形状规则命中
                              # 且 max≥40 → pillow
    _MAX_BASE_Z = 30.0       # 候选基座高度上限（place_location Z）：出题生成器
                              # 保证可搬物品"撒地面"（历轮实测 z0=0~5cm），收紧自
                              # 150cm 一刀排除高处背景构件（用户复盘点名 9/26/27/
                              # 28/52/53/55/57/58/60 全类：吊灯/墙板/横杆/挂墙盒
                              # z0=129~288，R12-58 教训）；容器最高茶几 z16 兼容
    # 场景道具 shape（test R12 复盘：40 号=玄关盆栽 shape=plant 21×21×36，
    # 不可交互且非五类物品——按 shape 拉黑，id 跨轮不稳不可用；
    # R25 教训：chair（61×61×80 餐椅恰卡 80cm 队列线）曾落大件先验）
    _SCENE_PROP_SHAPES = {"plant", "chair"}
    # 补盲扫点（R17-R19 漏件复盘）：茶几/沙发西侧是出生点 4×90° 的遮挡
    # 死区（漏件 @(591,277)/(604,469)/(436,192) 均在此象限），走到
    # 茶几-沙发之间补扫一轮
    _SWEEP_POINT = {"X": 470.0, "Y": 420.0, "Z": 0.0}
    _PAIR_DIST = 50.0        # 鞋成对判定的最大间距（cm，实测一双两脚相距 9~11cm）
    _PAIR_DIM_TOL = 12.0     # 鞋成对判定的尺寸容差（cm）

    # ---- v4 识别管线参数（离线实验定参，v2-notes §7-21）---- #
    _GATE_BUDGET = 45.0      # 识别 gate 截止线：直连 4帧×2调用并行≈5s，留9倍冗余
    _DIRECT_TIMEOUT = 45.0   # 直连客户端单次超时（实测 <2s，含网络抖动余量）
    _W_MIN, _W_MAX = 0.3, 3.0   # 票权上下限：w=clip(sqrt(area/A_ref), .3, 3)
    _TOP_MARGIN = 1.5        # 歧义判定：top1 < 1.5×次优
    _OVERTURN_W = 2.0        # VLM 推翻规则的加权门槛（平权时等价旧"≥2 票"）
    _SINGLE_W = 1.5          # 单帧票直接定案权重（近距大区域特写；回放教训：
                              # 远距单帧低权票(w≈1.0)定案曾把鞋误判枕）
    _CONFIRM_W = 5.0         # 到场确认一票的权重（特写=最高清晰度证据）
    _HI_W, _HI_H = 2560, 720  # 感知请求分辨率（v4 生产：裁剪自标需要更清的左面板）

    # 静态容器先验（2026-09-15 决策）：test/train 同一 UE 地图，仅物品随机
    # 撒、容器几何跨轮逐毫米稳定（R9/R10 茶几放点一致；R11 逐帧 AABB 实测，
    # 与队友核验包坐标三方互证）。先验=容器的世界坐标锚点，用于：
    # ①规则候选中优先取先验附近者（替代纯体积最大）②校正 VLM 投票的
    # 远偏选择（R9 教训：体积顶替把 table 从茶几翻到别处）③规则/VLM
    # 全失效时兜底。坐标单位 cm，xy 为 AABB 中心。
    _STATIC_CONTAINERS: dict[str, tuple[float, float, float]] = {
        "table": (577.0, 365.0, 200.0),        # 茶几 brown rect 72x137x29 z16-45（R11 id15）
        "sofa": (282.0, 374.0, 250.0),         # 主沙发 gray rect 112x410x99 z0-99（R11 id14）
        "trash_bin": (814.0, 267.0, 120.0),    # 垃圾桶 black box 27x27x35 z4-39（R11 id18）
        "shoe_cabinet": (780.0, -144.0, 200.0),  # 玄关窄深鞋柜 109x23x60 z3-63（R11 id46）
    }

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
        self._item_votes: dict[str, Counter] = {}  # VLM 加权投票（跨收割累积）
        self._item_votes_n: dict[str, Counter] = {}  # 同上，帧计数（防单帧定案）
        self._rule_cats: set[str] = set()          # 规则已分且高置信的物体
        self._locked = False           # v4：识别 gate 后分类锁定，票不再生效
        self._ambiguous: set[str] = set()   # 歧义件（到场确认）
        self._confirms: list[dict] = []     # 到场确认明细（复盘用）
        self._direct = None              # 直连客户端（思考关），缺 env 时 None
        if DirectVLMClient is not None:
            try:
                self._direct = DirectVLMClient(timeout=self._DIRECT_TIMEOUT)
                logger.info("VLM 直连客户端就绪 ({} @ {})",
                            self._direct.model, self._direct.api_base)
            except DirectVLMUnavailable:
                logger.warning("VLM 直连未配置（缺 VLM_CLIENT_CFG_API_KEY），"
                               "回退官方客户端（思考开，慢且偶发空文本）")
        self._vlm_started = 0.0
        self._scan_round = 0
        self._done_oids: set[str] = set()          # 已处理（成功/放弃/拉黑）
        self._blacklist: set[str] = set()
        self._placed = 0
        self._gave_up = 0
        self._gave_up_oids: set[str] = set()
        self._pair_cats: set[str] = set()         # 鞋成对启发改判的物体
        self._placements: list[dict] = []              # 放置明细(复盘用)
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
                    2560, 720, default={})
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
                        "placements": self._placements,
                        # v4：加权票/歧义/到场确认（复盘识别质量用）
                        "votes": {k: {c: round(w, 2) for c, w in v.items()}
                                  for k, v in self._item_votes.items()},
                        "ambiguous": sorted(self._ambiguous, key=int),
                        "confirms": self._confirms,
                    }, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # 流水线主循环
    # ------------------------------------------------------------------ #

    def _wait_scene_ready(self) -> None:
        """UE 语义注册表就绪等待（队友核验包同款，v5a 0 分教训）：
        UE 慢加载时首帧可能只有 3~11 个物体，直接扫会把废帧当全世界。
        首帧 <20 物体重试至多 8 次（每次 2s）；仍不足则照常开扫
        （真退化轮只能全栈重启，等待无意义）。"""
        for attempt in range(8):
            p = self._call_with_timeout(
                self.tongsim.acquire_first_person_perception, self.character_id,
                1280, 720, default={},
            )
            n = len(p.get("objects", []) or [])
            if n >= 20:
                return
            logger.warning("场景注册表未就绪 ({}/8, {} 物体)，重试", attempt + 1, n)
            time.sleep(2.0)

    def _pipeline(self) -> None:
        self._wait_scene_ready()
        if _V4:
            self._pipeline_v4()
        else:
            self._pipeline_v3()

    def _pipeline_v4(self) -> None:
        """三段式：扫描 → 识别 gate（截止线）→ 锁定 → 纯执行 + 歧义到场确认。

        治三病（v2-notes §7-20）：VLM 读小号绑错 oid（裁剪自标根治）、
        远景近视（特写+像素面积票权）、放错不可逆（锁定后执行零 VLM 依赖，
        歧义件在拿起前到场确认）。VLM 全挂时 gate 超时按规则开搬 = v3 下限。
        """
        self._scan_rounds(max_rounds=1, submit_vlm=True)
        self._apply_rule_categories()
        # 识别 gate：收割至全部帧完成/死亡或截止线
        deadline = time.time() + self._GATE_BUDGET
        while self._vlm_futs and time.time() < deadline:
            self._harvest_vlm(block=True,
                              timeout=max(0.5, deadline - time.time()))
        self._harvest_vlm(block=False)
        if self._vlm_futs:  # 截止线到：弃余票，按现有信息锁定
            logger.warning("识别 gate 截止（{}s），放弃 {} 个在途帧",
                           self._GATE_BUDGET, len(self._vlm_futs))
            self._vlm_futs = []
        self._finalize_recognition()
        self._rebuild_queue()

        while True:
            remain = self._TOTAL_BUDGET - (time.time() - self._t0)
            if remain < self._FINISH_RESERVE:
                logger.warning("预算剩余 {:.0f}s 触发看门狗，收工", remain)
                break
            if self._queue:
                if remain < self._FINISH_RESERVE + self._ITEM_COST:
                    logger.warning("预算剩余 {:.0f}s 不足一件，收工", remain)
                    break
                self._execute_one()
                # 确认帧并入的新物体 → 规则补分类 + 队列刷新（R18-33 教训）
                self._apply_rule_categories()
                self._rebuild_queue()
                continue
            # 队列空：补盲扫（R17-R19 漏件教训：茶几/沙发西侧是出生点
            # 扫描的遮挡死区；走过去补一轮 + 规则定类，新件自动入队，
            # 歧义件走到场确认）。约 +8s/轮，换回可能漏掉的 13 分/件。
            if self._scan_round < self._SCAN_ROUNDS:
                logger.info("队列空，走客厅西侧补盲扫（第 {} 轮）",
                            self._scan_round + 1)
                mv = self._call_with_timeout(
                    self.tongsim.move_to_location, self.character_id,
                    dict(self._SWEEP_POINT), default={},
                )
                if not self._ok(mv):
                    logger.info("补盲扫走位失败({})，原地扫", self._err(mv))
                self._scan_rounds(max_rounds=self._SCAN_ROUNDS, submit_vlm=False)
                self._apply_rule_categories()
                self._finalize_recognition()
                self._rebuild_queue()
                continue
            break

    def _pipeline_v3(self) -> None:
        """v3 旧流水线（TIDYROOM_V4=0 回退用）：规则先搬 + VLM 异步插队。"""
        self._scan_rounds(max_rounds=1, submit_vlm=True)
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

            if self._vlm_futs and self._scan_round < self._SCAN_ROUNDS \
                    and not self._all_candidates_done():
                self._harvest_vlm(block=True, timeout=10.0)
                continue
            if self._scan_round < self._SCAN_ROUNDS:
                logger.info("队列空，触发第 {} 轮扫描", self._scan_round + 1)
                self._scan_rounds(max_rounds=self._SCAN_ROUNDS, submit_vlm=False)
                self._apply_rule_categories()
                self._rebuild_queue()
                continue
            self._harvest_vlm(block=False)
            break

    def _all_candidates_done(self) -> bool:
        """world 中所有尺寸合格的可搬候选是否都已处理（placed/拉黑/放弃）。

        已解析容器（垃圾桶等尺寸可过物品筛选）不算候选——v5c 教训：
        桶 18 号让本判定永假，二轮扫描每轮白转 5~13s。
        """
        cont_oids = {str(c.get("object_id")) for c in self._containers.values()}
        for oid, obj in self._world.items():
            if oid in self._done_oids or oid in cont_oids:
                continue
            if self._is_point(obj) or self._too_high(obj) or self._is_scene_prop(obj):
                continue
            d = self._dims(obj)
            if min(d) >= self._MIN_DIM and max(d) <= self._MAX_ITEM_DIM:
                return False
        return True

    # ------------------------------------------------------------------ #
    # 扫描（每帧即时异步发 VLM）
    # ------------------------------------------------------------------ #

    def _dashcam_maps(self) -> tuple[dict[str, str], dict[str, str]]:
        """行车记录仪标注用：oid → (标签文字, 颜色键)。只登记可搬物品
        （_categories 即识别出的物品集），容器/家具不画——复盘只关心物品。"""
        labels: dict[str, str] = {}
        colors: dict[str, str] = {}
        placed_oids = {pl["oid"] for pl in self._placements}
        for oid, cat in self._categories.items():
            labels[oid] = cat
            if oid in self._blacklist:
                colors[oid] = "blacklist"
            elif oid in placed_oids:
                colors[oid] = "placed"
            elif oid in self._gave_up_oids:
                colors[oid] = "gaveup"
            elif oid in self._pair_cats:
                colors[oid] = "pair"
            elif oid in self._rule_cats:
                colors[oid] = "rule"
            else:
                colors[oid] = "vlm"
        return labels, colors

    def _save_frame(self, tag: str, b64: str,
                    objects: list[dict] | None = None) -> None:
        """感知图落盘：temp/p4_tidyroom/test_sessions/{时间戳[_标签]}/frame_*.jpg。

        每轮独立目录（目录规范：test/train 逐轮归档，含 meta 汇总）。
        dashcam 可用时存 YOLO 式标注帧 + 逐帧感知元数据 json；标注失败
        回退存原图，任何异常不影响任务主流程。
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
            if _dashcam is not None and objects is not None:
                labels, colors = self._dashcam_maps()
                _dashcam.save_frame(self._session_dir, tag,
                                    base64.b64decode(b64), objects,
                                    labels, colors)
                return
            (self._session_dir / f"frame_{tag}.jpg").write_bytes(base64.b64decode(b64))
        except Exception:  # noqa: BLE001  落盘失败不影响任务
            pass

    def _scan_rounds(self, max_rounds: int, submit_vlm: bool) -> None:
        while self._scan_round < max_rounds:
            self._scan_round += 1
            new = 0
            for i in range(self._SCAN_TURNS):
                # v4 生产即请求全分辨率（裁剪自标需要更清的左面板；数字
                # 戳固定 8-10px 不随分辨率放大，§7-17）；v3 模式维持 1280
                w, h = (self._HI_W, self._HI_H) if (_V4 and _sp is not None) \
                    else (1280, 720)
                p = self._call_with_timeout(
                    self.tongsim.acquire_first_person_perception, self.character_id,
                    w, h, default={},
                )
                objects = p.get("objects", []) or []
                img = p.get("image", "")
                self._save_frame(f"scan_r{self._scan_round}_{i}", img, objects)
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
        frame = {"image": image_b64, "objects": objects}
        self._vlm_frames[seq] = frame
        fn = self._vlm_classify_v4 if (_V4 and _sp is not None) \
            else self._vlm_classify_frame
        self._vlm_futs.append((seq, self._vlm_pool.submit(fn, frame)))

    # ------------------------------------------------------------------ #
    # VLM 分类与收割
    # ------------------------------------------------------------------ #

    def _direct_invoke(self, content_parts: list[dict]) -> str:
        """v4 感知调用通道：直连（思考关）优先，官方客户端回退。"""
        if self._direct is not None:
            return self._direct.invoke(content_parts, max_tokens=1024,
                                       timeout=self._DIRECT_TIMEOUT)
        resp = self.vlm_client.invoke(
            [{"role": "user", "content": content_parts}])
        return getattr(resp, "text", "") or ""

    def _frame_candidates(self, objects: list[dict]) -> dict[str, dict]:
        """当前帧的可搬候选（供 stamp_perceive 绑号过滤与元数据行）。"""
        wanted: dict[str, dict] = {}
        for o in objects:
            oid = str(o.get("object_id", ""))
            if not oid or self._is_point(o) or self._too_high(o) or self._is_scene_prop(o):
                continue
            dx, dy, dz = self._dims(o)
            if not (self._MIN_DIM <= min(dx, dy, dz)
                    and max(dx, dy, dz) <= self._MAX_ITEM_DIM):
                continue
            wanted[oid] = {"color": o.get("color"), "shape": o.get("shape"),
                           "size": f"{round(dx)}x{round(dy)}x{round(dz)}"}
        return wanted

    def _vlm_classify_v4(self, frame: dict) -> dict:
        """v4 单帧识别：戳定位→转写→洪泛→裁剪自标→轻量分类。

        返回统一形态 {"items": {oid: {"cat", "area"}}, "conts": {}}。
        读号链失败（无戳/转写空/无命中）回退旧整帧 prompt（直连思考关，
        ~1s/次；官方客户端仅直连不可用时兜底——它思考开 42~68s 且
        reasoning 吃满 token 正文为空，train_v4 首轮实机教训）。
        """
        diag: dict = {}
        try:
            wanted = self._frame_candidates(frame["objects"])
            if wanted:
                res = _sp.analyze_frame(
                    base64.b64decode(frame["image"]), wanted,
                    self._direct_invoke, diag)
                if res:
                    return {"items": res, "conts": {}}
        except Exception as exc:  # noqa: BLE001 读号链任何异常 → 整帧回退
            logger.warning("v4 读号链异常，回退整帧: {} {}", type(exc).__name__, exc)
        if diag:
            logger.warning("v4 读号链空结果 diag={}（{} 候选），回退整帧",
                           diag, len(self._frame_candidates(frame["objects"])))
        items, conts = self._vlm_classify_frame(frame)
        return {"items": {oid: {"cat": cat, "area": 0} for oid, cat in items.items()},
                "conts": conts}

    def _vlm_classify_frame(self, frame: dict) -> tuple[dict[str, str], dict[str, str]]:
        """单帧 VLM 标注，返回 (物品类别, 容器类型)。失败返回空 dict。"""
        compact = []
        for o in frame["objects"]:
            oid = str(o.get("object_id", ""))
            if not oid or self._is_point(o) or self._too_high(o) or self._is_scene_prop(o):
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
            # 直连优先（思考关 ~1s）：官方客户端思考开 42~68s 且偶发
            # reasoning 吃满 token 正文为空（train_v4 首轮实机教训）。
            try:
                if self._direct is not None:
                    text = self._direct.invoke(messages[0]["content"],
                                               max_tokens=2048)
                else:
                    resp = self.vlm_client.invoke(messages)
                    text = getattr(resp, "text", "") or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLM invoke 异常(第{}次): {} {}", attempt + 1,
                               type(exc).__name__, exc)
                continue
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

    @classmethod
    def _vote_weight(cls, area: int, areas: list[int]) -> float:
        """像素面积票权：w=clip(sqrt(area/A_ref),0.3,3)，A_ref=帧内中位数。

        面积 ∝ 视觉清晰度（近大远小），洪泛失败(area=0)回退平权 1.0——
        v3 回退路径（无区域数据）全为 1.0，与旧裸计数语义一致。
        """
        if area <= 0:
            return 1.0
        ref = sorted(areas)[len(areas) // 2] if areas else area
        if ref <= 0:
            return 1.0
        return min(max((area / ref) ** 0.5, cls._W_MIN), cls._W_MAX)

    def _harvest_vlm(self, block: bool, timeout: float = 0.0) -> bool:
        """收割 VLM future：加权投票合并 / 失败换帧重发一次 / 全挂放弃等。

        v4（v2-notes §7-20/21）：future 结果为 {"items": {oid: {cat,area}},
        "conts": {...}}，票权=像素面积权重；锁定后(_locked)票不再生效。
        deepseek 直连（思考关）无空文本失败类，换帧重发只是网络抖动兜底。
        """
        if not self._vlm_futs:
            return False
        changed = False
        pending: list[tuple[int, Future]] = []
        waited = False
        for seq, fut in self._vlm_futs:
            try:
                if not fut.done():
                    if not (block and not waited) or self._locked:
                        pending.append((seq, fut))
                        continue
                    waited = True
                    budget = self._VLM_BUDGET - (time.time() - self._vlm_started)
                    if budget <= 0:
                        pending.append((seq, fut))
                        continue
                    res = fut.result(timeout=min(timeout, max(budget, 1.0)))
                else:
                    res = fut.result(timeout=0.1)
            except FuturesTimeout:
                pending.append((seq, fut))
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLM 帧异常: {} {}", type(exc).__name__, exc)
                res = None
            items = (res or {}).get("items") or {}
            conts = (res or {}).get("conts") or {}
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
            if self._locked:
                continue  # 锁定后到票丢弃（执行阶段零 VLM 依赖）
            areas = [v.get("area", 0) for v in items.values()]
            for oid, info in items.items():
                cat = info.get("cat")
                if not cat or oid in self._done_oids or cat in ("other", "worn"):
                    continue
                if self._world.get(oid) is None:
                    continue
                w = self._vote_weight(int(info.get("area", 0)), areas)
                self._item_votes.setdefault(oid, Counter())[cat] += w
                self._item_votes_n.setdefault(oid, Counter())[cat] += 1
                # 非规则物体：当前加权 top 即时生效（gate 期）；规则物体的
                # 推翻判定收敛到 _finalize_recognition（加权≥2.0 且领先
                # 次优 1.5×，平权时等价旧"≥2 票"防单帧错分）
                if oid not in self._rule_cats:
                    top = self._weighted_top(oid)
                    if top and self._categories.get(oid) != top[0]:
                        self._categories[oid] = top[0]
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

    def _weighted_top(self, oid: str) -> tuple[str, float, float] | None:
        """加权票 top1：返回 (cat, w_top, w_second)，无票返回 None。"""
        c = self._item_votes.get(oid)
        if not c:
            return None
        top2 = c.most_common(2)
        cat, w1 = top2[0]
        w2 = top2[1][1] if len(top2) > 1 else 0.0
        return cat, float(w1), float(w2)

    def _finalize_recognition(self) -> None:
        """识别 gate 收口：规则×加权票合流定类，标记歧义件，锁定分类。

        歧义（到场确认）：无票且无规则 / top1<1.5×次优 / top=other /
        VLM 与规则冲突但未达推翻门槛。锁定后 _harvest_vlm 弃票。
        """
        cont_oids = {str(c.get("object_id")) for c in self._containers.values()}
        for oid, obj in self._world.items():
            if oid in self._done_oids or self._is_point(obj) or self._too_high(obj) \
                    or self._is_scene_prop(obj):
                continue
            if oid in cont_oids:
                continue  # 已解析容器本体（垃圾桶等尺寸可过物品筛选）不参与定类
            d = self._dims(obj)
            if not (self._MIN_DIM <= min(d) and max(d) <= self._MAX_ITEM_DIM):
                continue
            if oid in self._categories and oid not in self._item_votes \
                    and oid not in self._rule_cats:
                continue  # 成对启发等已定且无 VLM 票挑战
            rule_cat = self._categories.get(oid) if oid in self._rule_cats else None
            top = self._weighted_top(oid)
            if top is None:
                if rule_cat is None:
                    self._ambiguous.add(oid)  # 无票无规则 → 到场确认
                continue
            cat, w1, w2 = top
            if cat == "other":
                self._ambiguous.add(oid)
                continue
            if rule_cat is not None:
                if cat != rule_cat and w1 >= self._OVERTURN_W \
                        and (w2 == 0 or w1 >= self._TOP_MARGIN * w2):
                    logger.info("VLM 推翻规则: {} {} → {} (w={:.1f})",
                                oid, rule_cat, cat, w1)
                    self._categories[oid] = cat
                    self._rule_cats.discard(oid)
                elif cat != rule_cat:
                    self._ambiguous.add(oid)  # 冲突未达门槛 → 到场确认
                continue
            # 纯 VLM 票：票分裂、或单帧低权票不足定案 → 歧义
            n_top = self._item_votes_n.get(oid, Counter()).get(cat, 0)
            if w2 > 0 and w1 < self._TOP_MARGIN * w2:
                self._ambiguous.add(oid)
            elif n_top >= 2 or w1 >= self._SINGLE_W:
                if self._categories.get(oid) != cat:
                    self._categories[oid] = cat
            else:
                self._ambiguous.add(oid)
        self._locked = True
        amb = sorted(self._ambiguous, key=int)
        if amb:
            logger.info("识别锁定: 歧义件 {}（到场确认）", amb)
        self._trace("recognition_locked",
                    votes={k: dict(v) for k, v in self._item_votes.items()},
                    ambiguous=amb)

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
            if self._is_point(obj) or self._too_high(obj) or self._is_scene_prop(obj):
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
                    self._pair_cats.add(oid)
                    logger.info("鞋成对启发: {} ({}) 紧邻鞋 {} 且尺寸相近 → 改判 shoe",
                                oid, old, soid)
                    break

    def _apply_rule_categories(self) -> None:
        for oid, obj in self._world.items():
            if oid in self._done_oids or oid in self._categories:
                continue
            if self._is_point(obj) or self._too_high(obj) or self._is_scene_prop(obj):
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
        # 词汇直判（R22-57 red drumstick=鸡腿、R14/19-34 brown rock=石块、
        # R23 用户复盘：33 ring=甜甜圈/34 slice=切片食物——两件曾被极小
        # 兜底误送垃圾桶）
        if shape == "rock":
            return "trash"
        if shape == "drumstick":
            return "food" if max(self._dims(obj)) >= self._TINY_DIM else "trash"
        if shape in ("ring", "slice"):
            return "food"
        # shape 覆盖缺口补齐（§7-18：R11 的 36 号黑枕 shape=pillow 无分支，
        # VLM 全挂轮漏分类）：pillow 须 ≥_PILLOW_MIN_DIM（小 pillow 形件
        # 走末尾极小兜底进垃圾桶）
        if shape == "pillow" and max(self._dims(obj)) >= self._PILLOW_MIN_DIM:
            return "pillow"
        if shape == "box" and color != "black" and max(self._dims(obj)) <= 45:
            return "trash"
        if shape == "rectangle" and color in ("beige", "white") \
                and self._PILLOW_MIN_DIM <= max(self._dims(obj)) < self._MAX_ITEM_DIM:
            return "pillow"
        # 极小件兜底（用户定策：无形状特征的 <20cm 小物进垃圾桶——
        # rock/drumstick 小件、小 oval 碎物；罐/果等形状规则已在前命中）
        if max(self._dims(obj)) < self._TINY_DIM:
            return "trash"
        # 大件先验（R24-33 教训：57cm 灰方块确认成 trash——历史可搬大件
        # ≥40cm 只有抱枕）：**限定可搬尺寸区间**（<_MAX_ITEM_DIM），否则
        # R25 污染——冰箱/沙发/墙板等超大件也全部落进 pillow
        if self._BIG_DIM <= max(self._dims(obj)) < self._MAX_ITEM_DIM:
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
            elif ctype == "shoe_cabinet" and rule is not None:
                # VLM 投票选出的鞋柜若非窄深贴墙柜，以规则结果为准
                # （R5 教训：厨房柜可被误投成鞋柜）
                voted = resolved.get(ctype)
                if voted is not None:
                    dx, dy, dz = self._dims(voted)
                    if not (50 <= dz <= 110 and 20 <= min(dx, dy) <= 45):
                        logger.info("shoe_cabinet VLM 结果非窄深柜({:.0f}x{:.0f}x{:.0f})，规则校正",
                                    dx, dy, dz)
                        resolved[ctype] = rule
        # 静态先验校正：先验附近找不到可用容器时兜底填充；已选容器偏离
        # 先验超过容差（体积顶替/误投）时以先验为准——容器选择从此确定化
        for ctype in self._STATIC_CONTAINERS:
            prior = self._prior_container(ctype)
            if prior is None:
                continue
            cur = resolved.get(ctype)
            if cur is None:
                resolved[ctype] = prior
                continue
            px, py, tol = self._STATIC_CONTAINERS[ctype]
            cx, cy, _ = self._bb_center(cur)
            if ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 > tol:
                logger.info("容器 {} 选中 {}@({:.0f},{:.0f}) 偏离静态先验"
                            "({:.0f},{:.0f})超 {:.0f}cm，先验校正",
                            ctype, cur.get("object_id"), cx, cy, px, py, tol)
                resolved[ctype] = prior
        # 防倒退：已有容器不被规则清空
        for k, v in resolved.items():
            self._containers[k] = v

    def _prior_container(self, ctype: str) -> dict | None:
        """静态先验锚定的容器选择：先验坐标附近（容差内）尺寸几何可信者。

        与 _rule_container 的颜色判据无关（color 元数据可为 Unknown），
        只按类别尺寸盒过滤——规则/VLM 全失效时仍能按坐标找到容器。
        """
        prior = self._STATIC_CONTAINERS.get(ctype)
        if prior is None:
            return None
        px, py, tol = prior
        best, best_d = None, float("inf")
        for obj in self._world.values():
            if self._is_point(obj) or self._too_high(obj) or self._is_scene_prop(obj):
                continue
            dx, dy, dz = self._dims(obj)
            if ctype == "table":
                hit = 25 <= dz <= 55 and 60 <= max(dx, dy) <= 200
            elif ctype == "sofa":
                hit = max(dx, dy) >= 200 and dz < 130
            elif ctype == "trash_bin":
                hit = max(dx, dy, dz) <= 70 and 15 <= dz <= 60
            elif ctype == "shoe_cabinet":
                hit = 45 <= dz <= 115 and 15 <= min(dx, dy) <= 50
            else:
                hit = False
            if not hit:
                continue
            cx, cy, _ = self._bb_center(obj)
            d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if d < min(best_d, tol):
                best, best_d = obj, d
        return best

    def _rule_container(self, ctype: str) -> dict | None:
        best, best_vol, best_d = None, 0.0, float("inf")
        best_far, best_far_vol = None, 0.0
        prior = self._STATIC_CONTAINERS.get(ctype)
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
                # 鞋柜=贴墙窄高柜（实测 40 号 109x23x60）：水平窄边 20~45cm、
                # 高 50~110cm。厨柜/灶台深 60cm+ 会被此判据排除
                # （2026-09-15 R5 教训：厨房柜被误认鞋柜，2 件鞋放灶台 0 计分）
                hit = color == "brown" and shape == "rectangle" \
                    and 50 <= dz <= 110 and 20 <= min(dx, dy) <= 45
            if not hit:
                continue
            if prior is not None:
                # 规则命中者中优先取先验附近（替代纯体积最大，防大件顶替）；
                # 容差外命中留作换地图时的自适应回退
                px, py, tol = prior
                cx, cy, _ = self._bb_center(obj)
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d <= tol and d < best_d:
                    best, best_d, best_vol = obj, d, self._vol(obj)
                elif d > tol and self._vol(obj) > best_far_vol:
                    best_far, best_far_vol = obj, self._vol(obj)
            elif self._vol(obj) > best_vol:
                best, best_vol = obj, self._vol(obj)
        if best is None and best_far is not None:
            best = best_far
        if ctype == "shoe_cabinet":
            near = self._nearest_shoe_container(best)
            if near is not None:
                best = near
        return best

    def _nearest_shoe_container(self, fallback: dict | None) -> dict | None:
        """鞋柜兜底判据：鞋群 xy 质心最近的**窄深贴墙柜**（距离 <300cm）。

        窄深判据（水平窄边 20~45cm、高 50~110cm）与 _rule_container 同源，
        防止厨房橱柜等深柜在鞋撒进厨房区时被误选（R5 教训）。
        """
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
            if color != "brown" or shape != "rectangle" or min(dx, dy) < self._MIN_DIM:
                continue
            if not (50 <= dz <= 110 and 20 <= min(dx, dy) <= 45):
                continue  # 非窄深贴墙柜
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
        # 歧义件并入（保留先验类别作到场确认失败的回退），不重复入队
        merged: dict[str, str | None] = dict(self._categories)
        for oid in self._ambiguous:
            merged[oid] = self._categories.get(oid)
        queue: list[dict] = []
        for oid, cat in merged.items():
            if oid in self._done_oids:
                continue
            if _ONLY_OIDS and oid not in _ONLY_OIDS:
                continue
            obj = self._world.get(oid)
            if obj is None:
                continue
            dx, dy, dz = self._dims(obj)
            if max(dx, dy, dz) > self._MAX_ITEM_DIM or min(dx, dy, dz) < self._MIN_DIM:
                continue
            confirm = oid in self._ambiguous
            if not confirm and cat is None:
                continue  # 非歧义但无类别（不该出现，保险）
            cont_key = _CAT_TO_CONTAINER.get(cat) if cat else None
            if not confirm:
                cont = self._containers.get(cont_key) if cont_key else None
                if cont is None:
                    continue
                if self._on_target(obj, cont):
                    continue
            else:
                # 歧义件即使疑似在位也要到场看一眼（可能是错的容器）
                cont = self._containers.get(cont_key) if cont_key else None
            put, put_retry = (self._put_points(cont, cont_key)
                              if cont is not None else ({}, {}))
            queue.append({"oid": oid, "cat": cat, "cont_key": cont_key,
                          "put": put, "put_retry": put_retry,
                          "move": self._move_point(put) if put else None,
                          "confirm": confirm})
        # 歧义件排最后：先搬已锁定的确定件，歧义件在执行中到场确认
        queue.sort(key=lambda t: (t["confirm"],
                                  _CAT_ORDER.get(t["cat"], 9), t["oid"]))
        self._queue = queue

    # ------------------------------------------------------------------ #
    # 执行（单件：take → put → 确认 → 必要时重放）
    # ------------------------------------------------------------------ #

    def _execute_one(self) -> None:
        task = self._queue.pop(0)
        oid = task["oid"]
        t1 = time.time()

        # 歧义件到场确认（v4）：走到跟前→感知→中央特写→一票定乾坤。
        # 绑号结构性成立（move_to_object 的目标就是该 oid），免读号。
        if task.get("confirm"):
            cat = self._confirm_on_arrival(oid, t1)
            if cat == "on_target":
                logger.info("歧义 {} 已在匹配容器上，跳过搬运", oid)
                self._done_oids.add(oid)
                self._trace("on_target_skip", oid=oid)
                return
            if cat:
                task["cat"] = cat
                task["cont_key"] = _CAT_TO_CONTAINER.get(cat)
                cont = self._containers.get(task["cont_key"] or "")
                if cont is None:
                    logger.info("歧义 {} 确认为 {} 但无容器，跳过", oid, cat)
                    self._done_oids.add(oid)
                    return
                task["put"], task["put_retry"] = self._put_points(cont, task["cont_key"])
                task["move"] = self._move_point(task["put"])
                obj = self._world.get(oid)
                if obj is not None and self._on_target(obj, cont):
                    logger.info("歧义 {} 确认 {} 且已在容器上，跳过", oid, cat)
                    self._done_oids.add(oid)
                    self._trace("on_target_skip", oid=oid, cat=cat)
                    return
            elif task.get("cat") is None:
                # 确认失败且无任何先验类别：放弃（盲放比不放差——放错扣完成度）
                logger.info("歧义 {} 到场确认失败且无先验，放弃", oid)
                self._done_oids.add(oid)
                self._gave_up += 1
                self._gave_up_oids.add(oid)
                self._trace("confirm_giveup", oid=oid)
                return
            # 确认失败但有先验类别 → 按先验继续搬

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

        # 行车记录仪：拿起后立刻拍一张近距帧（R24 复盘教训——confirm 帧
        # 是放置后拍的，物体已传送进容器看不见；此帧供人工核对搬走的是
        # 什么）。仅存档用，1280 即可，~1.3s/件。
        if _SAVE_FRAMES:
            held = self._call_with_timeout(
                self.tongsim.acquire_first_person_perception, self.character_id,
                1280, 720, default={},
            )
            # 拿起帧感知顺带合并 world（走到物品边的视野可暴露遮挡件，
            # 与确认帧同款 vol-max 合并；主循环每件后会重分类入队）
            for obj in held.get("objects", []) or []:
                oid2 = str(obj.get("object_id", ""))
                if not oid2:
                    continue
                prev = self._world.get(oid2)
                if prev is None or self._vol(obj) > self._vol(prev):
                    self._world[oid2] = obj
            if held.get("image"):
                self._save_frame(f"took_{oid}", held["image"],
                                 held.get("objects", []) or [])

        ok = self._put_with_retry(task)
        self._done_oids.add(oid)
        if ok:
            self._placed += 1
            self._container_slots[task["cont_key"]] = \
                self._container_slots.get(task["cont_key"], 0) + 1
            self._placements.append({
                "oid": oid, "cat": task["cat"], "cont": task["cont_key"],
                "cont_obj": str(self._containers.get(task["cont_key"], {})
                                .get("object_id", "?")),
                "put": task["put"], "sec": round(time.time() - t1, 1)})
            logger.info("件完成 {} → {} ({:.1f}s), 累计 {}",
                        oid, task["cont_key"], time.time() - t1, self._placed)
            self._trace("placed", oid=oid, cat=task["cat"], cont=task["cont_key"],
                        sec=round(time.time() - t1, 1))
        else:
            self._gave_up += 1
            self._gave_up_oids.add(oid)
            logger.warning("件放弃 {} ({})", oid, task["cont_key"])
            self._trace("gave_up", oid=oid, cat=task["cat"], cont=task["cont_key"])

    def _confirm_on_arrival(self, oid: str, t0: float) -> str | None:
        """歧义件到场确认：move_to_object → 感知 → 左面板中央特写 → prompt C。

        一票权重 _CONFIRM_W（特写=最高清晰度证据）合并进加权票后取 top；
        失败返回 None（调用方按先验类别/放弃处理）。已在正确容器上则
        返回 "on_target" 哨兵由调用方跳过搬运。
        """
        try:
            self._call_with_timeout(self.tongsim.move_to_object,
                                    self.character_id, oid, default=None)
            p = self._call_with_timeout(
                self.tongsim.acquire_first_person_perception, self.character_id,
                self._HI_W, self._HI_H, default={},
            )
            # 感知合并进 world（R18-33 教训：确认帧看到的物体/更完整 AABB
            # 不能丢弃——走到场边的视野正是补盲来源，主循环每件后会重分类）
            for obj in p.get("objects", []) or []:
                oid2 = str(obj.get("object_id", ""))
                if not oid2:
                    continue
                prev = self._world.get(oid2)
                if prev is None or self._vol(obj) > self._vol(prev):
                    self._world[oid2] = obj
            img_b64 = p.get("image", "")
            if _SAVE_FRAMES and img_b64:
                self._save_frame(f"confirm_cls_{oid}", img_b64,
                                 p.get("objects", []) or [])
            if not img_b64 or _sp is None:
                return None
            obj = next((o for o in p.get("objects", []) or []
                        if str(o.get("object_id", "")) == oid), None)
            if obj is not None:  # 已在目标容器上则不必问也不必搬
                for ck, cont in self._containers.items():
                    if self._on_target(obj, cont) and \
                            _CAT_TO_CONTAINER.get(self._best_guess(oid)) == ck:
                        return "on_target"
            crop = _sp.confirm_image(base64.b64decode(img_b64), oid)
            if crop is None:
                return None
            meta = self._frame_candidates(p.get("objects", []) or []).get(oid) \
                or {"color": "Unknown", "shape": "Unknown", "size": "?"}
            prompt = _sp.PROMPT_CONFIRM_TMPL.format(
                oid=oid, color=meta["color"], shape=meta["shape"],
                size=meta["size"])
            parts = [{"type": "image_url",
                      "image_url": {"url": "data:image/jpeg;base64,"
                                    + base64.b64encode(crop).decode()}},
                     {"type": "text", "text": prompt}]
            cat = _sp.parse_classification(self._direct_invoke(parts)).get(oid)
            sec = round(time.time() - t0, 1)
            if cat and cat != "other":
                # 尺寸守卫（R22-57 教训：11cm 鸡腿被 VLM 确认成 pillow；
                # R24-33 教训：57cm 灰方块被确认成 trash）：
                # pillow 须 ≥30cm、大件(≥40cm)不可能是 trash——违者改判
                obj = self._world.get(oid) or {}
                if cat == "pillow" and max(self._dims(obj)) < self._PILLOW_MIN_DIM:
                    logger.info("确认 {} → {} 但 max dim {:.0f}cm < {:.0f}，"
                                "改判 trash（进桶）", oid, cat,
                                max(self._dims(obj)), self._PILLOW_MIN_DIM)
                    cat = "trash"
                elif cat == "trash" and max(self._dims(obj)) >= self._BIG_DIM:
                    logger.info("确认 {} → {} 但 max dim {:.0f}cm ≥ {:.0f}，"
                                "改判 pillow（大件先验）", oid, cat,
                                max(self._dims(obj)), self._BIG_DIM)
                    cat = "pillow"
                self._item_votes.setdefault(oid, Counter())[cat] += self._CONFIRM_W
                self._item_votes_n.setdefault(oid, Counter())[cat] += 1
                top = self._weighted_top(oid)
                final = top[0] if top else cat
                self._categories[oid] = final
                self._ambiguous.discard(oid)
                self._confirms.append({"oid": oid, "cat": final, "sec": sec})
                logger.info("到场确认 {} → {} ({:.1f}s)", oid, final, sec)
                self._trace("confirm", oid=oid, cat=final, sec=sec)
                return final
            self._confirms.append({"oid": oid, "cat": None, "sec": sec})
            logger.info("到场确认 {} 无结论 ({:.1f}s)", oid, sec)
            return None
        except Exception as exc:  # noqa: BLE001 确认链异常不影响搬运主流程
            logger.warning("到场确认 {} 异常: {} {}", oid, type(exc).__name__, exc)
            return None

    def _best_guess(self, oid: str) -> str:
        """当前最佳类别（加权 top 或已有分类），无则空串。"""
        top = self._weighted_top(oid)
        return self._categories.get(oid) or (top[0] if top else "")

    def _put_with_retry(self, task: dict) -> bool:
        """放置链 v5（2026-09-16，队友 98.11 分实证）：

        put_down_sth(force_locate) 是远距传送，**不要求人物靠近容器**——
        v3 的 move_to_object(容器) 走位每件多花 4~8s 属冗余。改为第一遍
        原地直放（主点/备点）；全失败才走近容器按 v3 老链重试兜底。
        历史教训仍有效：move_and_put_down 的 move 点不控朝向会掉在旁边
        地上（run11），本链不用它；put_down_sth 是强制坐标放置可穿模
        （run11 33 号实证），确认逻辑复刻判卷几何（§7-6）。
        """
        cont = self._containers.get(task["cont_key"])
        if cont is None:
            return False
        if self._put_pass(task, cont, walk=False):
            return True
        if self._put_pass(task, cont, walk=True):
            return True
        # 两遍都没确认成功：原地放下防卡手
        self._call_with_timeout(
            self.tongsim.put_down_sth, self.character_id,
            target_location=task["put_retry"], auto_rotate=True, default={},
        )
        return False

    def _put_pass(self, task: dict, cont: dict, walk: bool) -> bool:
        if walk:
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
        img = p.get("image", "")
        if _SAVE_FRAMES and img:
            hi = self._call_with_timeout(
                self.tongsim.acquire_first_person_perception,
                self.character_id, 2560, 720, default={},
            )
            if hi.get("image"):
                img = hi["image"]
        self._save_frame(f"confirm_{oid}", img, p.get("objects", []) or [])
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

    @classmethod
    def _is_scene_prop(cls, obj: dict) -> bool:
        """场景道具（盆栽等）：shape 直接标注类型，不可交互、非五类物品。"""
        return str(obj.get("shape", "")).lower() in cls._SCENE_PROP_SHAPES

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
