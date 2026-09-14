"""P4 整理房间 agent：360° 扫描建图 + VLM 一次性分类规划 + 零 VLM 脚本化执行。

对应基线 16 分的失败解剖（temp/baseline_records/ + logs/ 实测）：
- 基线 61% 时间耗在逐步 VLM（6.5s/步 × 31 步）→ 本 agent 的 VLM 只在扫描后
  一次性出现，执行阶段纯元数据驱动；
- "can not take this object for not pickup" 是物体属性、与距离无关（人脚上的鞋、
  地毯等）→ take 失败一次永久拉黑，绝不重试；
- 放置点公式 = 目标容器 AABB 中心 xy + 顶面高度（基线成功案例反推）；
- 剩余 30s 主动 finish_task 锁住完成度分与时间分。

动作链只用两个复合指令（自带导航，实测成功）：
- move_and_take_object(object_id)：走到物体旁并抓起；
- move_and_put_down(move_target_location, put_target_location)：走到 move 点，
  把手中物放到 put 点。

运行（赛题系统 train tidyroom 已启动，仓库根目录）：
    uv run python -m cqairace.tidyroom_agent --run_times 1
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
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
    """整理房间三阶段：扫描 → 规划 → 执行，单次 run_step 内完成。"""

    _SCAN_TURNS = 4          # 首帧不转 + 3×90°，FOV 120° 全覆盖
    _TOTAL_BUDGET = 370.0    # 整题预算（秒），首 run_step 起算，留 30s 给判卷
    _FINISH_RESERVE = 30.0   # finish 看门狗预留
    _ITEM_COST = 12.0        # 单件 take+put 预估耗时（不够一件就收工）
    _OP_TIMEOUT = 40.0       # 单次外部调用超时
    _VLM_TIMEOUT = 75.0      # 单次 VLM 调用超时（实测大 prompt 推理 68s）
    _VLM_BUDGET = 115.0      # 阶段 B 的 VLM 总预算（4 帧并行 ≈ 单次耗时）
    _MAX_ITEM_DIM = 80.0     # 候选物品最大边（cm）：not pickup 秒拒零成本，放宽让抱枕类入队
    _MIN_DIM = 2.0           # 排除点状 AABB（墙角标记）
    _MAX_BASE_Z = 150.0      # 排除壁挂/吊灯（place_location Z 上限）

    _pool = ThreadPoolExecutor(max_workers=2)   # tongsim 调用超时兜底
    _vlm_pool = ThreadPoolExecutor(max_workers=4)  # 4 帧 VLM 并行分类

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
        self._world: dict[str, dict] = {}        # object_id → 物体元数据（全局清单）
        self._frames: list[dict] = []            # 扫描帧 [{"image", "objects"}]
        self._plan: list[dict] | None = None     # 待办队列
        self._containers: dict[str, dict] = {}   # 容器键 → 元数据
        self._container_slots: dict[str, int] = {}  # 容器键 → 已放置件数（散点用）
        self._blacklist: set[str] = set()
        self._placed = 0
        self._gave_up = 0

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #

    def run_step(self, subject, task_response: dict[str, Any]) -> dict[str, Any]:
        if self._t0 is None:
            self._t0 = time.time()
        try:
            if self._plan is None:
                self._phase_scan()
                self._phase_plan()
            self._phase_execute()
        except Exception as exc:  # noqa: BLE001 任何异常不外抛：保住已放件数
            logger.opt(exception=True).warning("tidyroom 阶段异常（保底收尾）: {}", exc)
        summary = (f"tidyroom: scanned={len(self._world)} placed={self._placed} "
                   f"gave_up={self._gave_up} blacklist={sorted(self._blacklist)}")
        logger.info(summary)
        self._trace("finish", summary=summary,
                    placed=self._placed, blacklist=sorted(self._blacklist))
        # 走官方 _handle_finish：置 subject_finished 并以官方格式提交
        return self._do_action({"action": "finish_task", "think": summary,
                                "output": self._placed})

    # ------------------------------------------------------------------ #
    # 阶段 A：扫描建图
    # ------------------------------------------------------------------ #

    def _phase_scan(self) -> None:
        for i in range(self._SCAN_TURNS):
            p = self._call_with_timeout(
                self.tongsim.acquire_first_person_perception, self.character_id,
                1280, 720, default={},
            )
            objects = p.get("objects", []) or []
            self._frames.append({"image": p.get("image", ""), "objects": objects})
            for obj in objects:
                oid = str(obj.get("object_id", ""))
                if not oid:
                    continue
                prev = self._world.get(oid)
                if prev is None or self._vol(obj) > self._vol(prev):
                    self._world[oid] = obj  # 保留包围盒更完整的一帧
            if i < self._SCAN_TURNS - 1:
                self._call_with_timeout(self.tongsim.turn_in_degree,
                                        self.character_id, 90, default=None)
                time.sleep(0.5)
        logger.info("阶段A 扫描完成: {} 帧, 全局清单 {} 物体", len(self._frames), len(self._world))

    # ------------------------------------------------------------------ #
    # 阶段 B：分类 + 放置规划
    # ------------------------------------------------------------------ #

    def _phase_plan(self) -> None:
        item_votes: dict[str, Counter] = {}
        container_votes: dict[str, Counter] = {}

        # 4 帧 VLM 分类并行发出（实测 deepseek-flash 大 prompt 单次 ~70s，
        # 串行会耗尽预算；并行后总耗时 ≈ 最慢单次）
        t_vlm = time.time()
        futs = [(self._vlm_pool.submit(self._vlm_classify_frame, frame), frame)
                for frame in self._frames]
        for fut, _frame in futs:
            try:
                remain = self._VLM_BUDGET - (time.time() - t_vlm)
                if remain <= 5:
                    logger.warning("VLM 总预算耗尽，停止收取剩余帧")
                    break
                items, conts = fut.result(timeout=remain)
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLM 帧分类失败: {} {}", type(exc).__name__, exc)
                items, conts = {}, {}
            for oid, cat in items.items():
                item_votes.setdefault(oid, Counter())[cat] += 1
            for oid, ctype in conts.items():
                container_votes.setdefault(oid, Counter())[ctype] += 1
        logger.info("VLM 并行分类完成: {:.1f}s, 物品票 {} 容器票 {}",
                    time.time() - t_vlm, len(item_votes), len(container_votes))

        categories = {oid: c.most_common(1)[0][0] for oid, c in item_votes.items() if c}
        containers_raw = {oid: c.most_common(1)[0][0] for oid, c in container_votes.items() if c}

        # 容器优先：同一 id 若被标为容器，从物品清单剔除（家具不该被搬）
        for oid in list(categories):
            if oid in containers_raw:
                del categories[oid]

        # 物品分类规则兜底：VLM 无结果的物体按 shape/color 启发式补齐
        vlm_miss = [oid for oid in self._world
                    if oid not in categories and oid not in containers_raw
                    and not self._is_point(self._world[oid])
                    and not self._too_high(self._world[oid])]
        for oid in vlm_miss:
            cat = self._rule_category(oid)
            if cat:
                categories[oid] = cat
        if vlm_miss:
            logger.info("物品分类规则兜底: VLM 未覆盖 {} 物体, 补齐 {}",
                        len(vlm_miss), sum(1 for o in vlm_miss if o in categories))

        self._containers = self._resolve_containers(containers_raw)
        self._plan = self._build_queue(categories)
        logger.info("阶段B 规划完成: 容器={} 队列={} 类别={}",
                    {k: v.get("object_id") for k, v in self._containers.items()},
                    [(t["oid"], t["cat"]) for t in self._plan], categories)

    def _vlm_classify_frame(self, frame: dict) -> tuple[dict[str, str], dict[str, str]]:
        """单帧 VLM 标注，返回 (物品类别, 容器类型)。失败返回空 dict（由规则兜底）。"""
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
            "1. items：散乱摆放、需要整理的小物件。类别只能是：\n"
            "   trash(垃圾碎屑)、cup(杯子/罐)、food(食物/水果)、shoe(散落在地上的鞋靴)、"
            "pillow(抱枕靠垫)。\n"
            "   特别注意：穿在人脚上的鞋（有人在里面）标为 worn，不要标 shoe。\n"
            "2. containers：可放置物品的目标家具。类型只能是：\n"
            "   trash_bin(垃圾桶)、table(茶几/餐桌)、sofa(沙发)、shoe_cabinet(鞋柜/边柜)。\n"
            "当前帧物体元数据（size_cm 是包围盒长宽高，loc 是世界坐标）：\n"
            f"{json.dumps(compact, ensure_ascii=False)}\n"
            "要求：家具本身、已摆放整齐的物品、墙上装饰都不要列入 items；"
            "垃圾桶虽然不大但它是 container 不是 trash。\n"
            "重要：不要思考过程，回答的第一个字符必须是 {{，格式如 "
            '{{"items": {{"36": "trash", "32": "cup"}}, "containers": {{"18": "trash_bin", "15": "table"}}}}；'
            "没有则用空字典。"
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + frame["image"]}},
                {"type": "text", "text": prompt},
            ],
        }]
        for attempt in range(2):  # 推理模型可能空正文/超时，重试一次
            resp = self._call_with_timeout(self.vlm_client.invoke, messages,
                                           timeout=self._VLM_TIMEOUT)
            if resp is None:
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
                conts = {k: v for k, v in conts.items() if v}
                return items, conts
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

    def _resolve_containers(self, voted: dict[str, str]) -> dict[str, dict]:
        """VLM 容器投票结果 → 每类容器选一个元数据对象；缺的/垃圾桶用规则校正。"""
        resolved: dict[str, dict] = {}
        for oid, ctype in voted.items():
            obj = self._world.get(oid)
            if obj is None:
                continue
            prev = resolved.get(ctype)
            if prev is None or self._vol(obj) > self._vol(prev):
                resolved[ctype] = obj  # 同类取包围盒更大的（主体而非部件）
        for ctype in ("trash_bin", "table", "sofa", "shoe_cabinet"):
            # 垃圾桶规则优先：黑小方箱特征极强，VLM 常把绿植等大件误标为垃圾桶
            rule = self._rule_container(ctype)
            if ctype == "trash_bin" and rule is not None:
                if ctype not in resolved or resolved[ctype].get("object_id") != rule.get("object_id"):
                    logger.info("trash_bin 规则校正: {} -> {}",
                                resolved.get(ctype, {}).get("object_id"), rule.get("object_id"))
                resolved[ctype] = rule
            elif ctype not in resolved and rule is not None:
                resolved[ctype] = rule
                logger.info("容器 {} 规则兜底命中 object_id={}", ctype, rule.get("object_id"))
        return resolved

    def _rule_container(self, ctype: str) -> dict | None:
        """容器识别的 shape/color 启发式（VLM 缺失时的兜底）。"""
        best, best_score = None, 0.0
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
                # 桌面矮胖（茶几 72×137×29），柜体细长（电视柜 43×240×37）
                hit = color == "brown" and shape == "rectangle" \
                    and 25 <= dz <= 55 and max(dx, dy) < 200
            elif ctype == "sofa":
                hit = shape == "rectangle" and max(dx, dy) > 250 and dz < 120
            elif ctype == "shoe_cabinet":
                hit = color == "brown" and shape == "rectangle" and dz > 55
            if hit and self._vol(obj) > best_score:
                best, best_score = obj, self._vol(obj)
        if ctype == "shoe_cabinet":
            near = self._nearest_shoe_container(best)
            if near is not None:
                best = near
        return best

    def _nearest_shoe_container(self, fallback: dict | None) -> dict | None:
        """鞋柜兜底判据：鞋类物体 xy 质心最近的 brown rectangle（距离 <300cm）。"""
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

    def _rule_category(self, oid: str) -> str | None:
        """物品分类的 shape 启发式（VLM 完全失败时兜底）。"""
        obj = self._world.get(oid)
        if obj is None:
            return None
        shape = str(obj.get("shape", "")).lower()
        color = str(obj.get("color", "")).lower()
        if shape in ("boot", "shoe"):
            return "shoe"
        if shape == "cylinder":
            return "cup"
        if shape == "irregular":
            return "trash"
        if shape == "round":
            return "food"
        if shape == "rectangle" and color in ("beige", "white") and max(self._dims(obj)) < 60:
            return "pillow"
        return None

    def _build_queue(self, categories: dict[str, str]) -> list[dict]:
        """分类结果 → 待办队列（可抓候选 + 有目标容器 + 未在位 + 分组排序）。"""
        queue: list[dict] = []
        for oid, cat in categories.items():
            obj = self._world.get(oid)
            if obj is None or cat in ("other", "worn"):
                continue
            if self._is_point(obj) or self._too_high(obj):
                continue  # 点状标记 / 壁挂吊灯
            cont_key = _CAT_TO_CONTAINER.get(cat)
            cont = self._containers.get(cont_key) if cont_key else None
            if cont is None:
                logger.info("物品 {} 类别 {} 无目标容器，跳过", oid, cat)
                continue
            dx, dy, dz = self._dims(obj)
            if max(dx, dy, dz) > self._MAX_ITEM_DIM or min(dx, dy, dz) < self._MIN_DIM:
                continue  # 家具/地毯/巨型物不搬
            if self._on_target(obj, cont):
                logger.info("物品 {} 已在目标 {} 上，跳过", oid, cont_key)
                continue
            put, put_retry = self._put_points(cont, cont_key)
            queue.append({"oid": oid, "cat": cat, "obj": obj, "cont_key": cont_key,
                          "put": put, "put_retry": put_retry,
                          "move": self._move_point(put)})
        # 分组（先易后难）+ 组内按 object_id 稳定排序
        queue.sort(key=lambda t: (_CAT_ORDER.get(t["cat"], 9), t["oid"]))
        return queue

    # ------------------------------------------------------------------ #
    # 阶段 C：执行（零 VLM）
    # ------------------------------------------------------------------ #

    def _phase_execute(self) -> None:
        assert self._plan is not None
        while self._plan:
            remain = self._TOTAL_BUDGET - (time.time() - self._t0)
            if remain < self._FINISH_RESERVE + self._ITEM_COST:
                logger.warning("预算剩余 {:.0f}s 不足一件，收工", remain)
                break
            task = self._plan.pop(0)
            oid = task["oid"]
            if oid in self._blacklist:
                continue

            t1 = time.time()
            take = self._call_with_timeout(
                self.tongsim.move_and_take_object, self.character_id, oid,
                which_hand=0, default={},
            )
            take_s = time.time() - t1
            if not self._ok(take):
                self._blacklist.add(oid)
                logger.info("take {} 失败({}) → 拉黑, 耗时 {:.1f}s",
                            oid, self._err(take), take_s)
                self._trace("take_failed", oid=oid, cat=task["cat"],
                            res=self._err(take), sec=round(take_s, 1))
                continue

            t2 = time.time()
            put = self._call_with_timeout(
                self.tongsim.move_and_put_down, self.character_id,
                move_target_location=task["move"], put_target_location=task["put"],
                which_hand=0, default={},
            )
            put_s = time.time() - t2
            if self._ok(put):
                self._placed += 1
                self._container_slots[task["cont_key"]] = \
                    self._container_slots.get(task["cont_key"], 0) + 1
                logger.info("put {} → {} 成功 ({:.1f}s+{:.1f}s), 累计 {} 件",
                            oid, task["cont_key"], take_s, put_s, self._placed)
                self._trace("placed", oid=oid, cat=task["cat"], cont=task["cont_key"],
                            put=task["put"], sec=round(take_s + put_s, 1))
                continue

            # 放置重试：容器中心顶面（保守点）
            put2 = self._call_with_timeout(
                self.tongsim.move_and_put_down, self.character_id,
                move_target_location=task["move"], put_target_location=task["put_retry"],
                which_hand=0, default={},
            )
            if self._ok(put2):
                self._placed += 1
                self._container_slots[task["cont_key"]] = \
                    self._container_slots.get(task["cont_key"], 0) + 1
                logger.info("put {} → {} 重试成功, 累计 {} 件", oid, task["cont_key"], self._placed)
                self._trace("placed_retry", oid=oid, cat=task["cat"], cont=task["cont_key"],
                            put=task["put_retry"], sec=round(time.time() - t2, 1))
                continue

            # 仍失败：确认手中是否有物，有则原地放下防卡手
            in_hand, _ = self._call_with_timeout(
                self.tongsim.has_object_in_hand, self.character_id, default=(False, None),
            )
            if in_hand:
                self._call_with_timeout(
                    self.tongsim.put_down_sth, self.character_id,
                    target_location=task["put"], auto_rotate=True, default={},
                )
            self._gave_up += 1
            logger.warning("put {} 两次失败({}/{})，放弃本件",
                           oid, self._err(put), self._err(put2))
            self._trace("put_failed", oid=oid, cat=task["cat"], cont=task["cont_key"],
                        err1=self._err(put), err2=self._err(put2))

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
        """主放置点（AABB 内散点）+ 重试点（中心顶面）。坐标系 {"X","Y","Z"} cm。

        公式来自基线成功案例：put = 容器 AABB 中心 xy + 顶面高度。
        垃圾桶例外：z 取桶身中上部（基线 30 = min_z+0.7×高 实测成功）。
        """
        bb = cont.get("world_aabb") or {}
        mn, mx = bb.get("min") or {}, bb.get("max") or {}
        x0, y0 = float(mn.get("x", 0)), float(mn.get("y", 0))
        x1, y1 = float(mx.get("x", 0)), float(mx.get("y", 0))
        z0, z1 = float(mn.get("z", 0)), float(mx.get("z", 0))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        if cont_key == "trash_bin":
            pz = z0 + 0.7 * (z1 - z0)
            return ({"X": cx, "Y": cy, "Z": pz},
                    {"X": cx, "Y": cy, "Z": z0 + 0.7 * (z1 - z0)})

        # 收缩 20% 网格散点（2×2 顺序取），同件数复用最后一个点
        mx_x, mx_y = x0 + 0.2 * (x1 - x0), y0 + 0.2 * (y1 - y0)
        w, h = max((x1 - x0) * 0.6, 1), max((y1 - y0) * 0.6, 1)
        slot = self._container_slots.get(cont_key, 0)
        gx = mx_x + (0.5 if slot % 2 else 0.0) * w
        gy = mx_y + (0.5 if (slot // 2) % 2 else 0.0) * h
        top = z1 + 2
        return ({"X": round(gx, 1), "Y": round(gy, 1), "Z": round(top, 1)},
                {"X": round(cx, 1), "Y": round(cy, 1), "Z": round(top, 1)})

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
        """物品当前位置已在容器内/上 → 无需搬运。

        z 判据用"顶面附近"（max_z - 0.6×高 ~ max_z+25）：容器 AABB 投影内
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
