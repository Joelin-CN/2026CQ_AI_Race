# -*- coding: utf-8 -*-
"""v4 识别管线单测：裁剪自标全链 / 像素面积票权 / 歧义判定与流转 / 队列合并。

合成帧数据自造（右面板=亮色块+黑字编号，洪泛可复现）；finalize 场景
对应 v2-notes §7-20 的歧义定义。
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from cqairace import stamp_perceive as sp
from cqairace.tidyroom_agent import TidyroomAgent

# ------------------------------------------------------------------ #
# stamp_perceive：合成复合帧全链
# ------------------------------------------------------------------ #


def make_composite(num="33", region=(90, 60, 70, 46)):
    """左 RGB + 右分割面板（亮色块 + 黑色编号），2560×720 的缩小版。"""
    h, w = 220, 480
    img = np.full((h, w, 3), (200, 220, 240), np.uint8)   # 米色底
    pw = w // 2
    x, y, rw, rh = region
    color = (80, 180, 250)                                # 亮橙（gray≈189 可作洪泛种子）
    img[y:y + rh, x:x + rw] = color                       # 左面板物体
    img[y:y + rh, pw + x:pw + x + rw] = color             # 右面板色块
    cx, cy = pw + x + rw // 2, y + rh // 2
    cv2 = sp.cv2
    cv2.putText(img, num, (cx - 11, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 0), 2, cv2.LINE_AA)          # ~10px 黑字编号（右面板）
    return img, pw


def test_locate_stamps_on_synthetic_panel():
    img, pw = make_composite()
    groups = sp.locate_stamp_groups(img[:, pw:])
    assert 1 <= len(groups) <= 3          # "33"（或笔画拆分后合组）
    g = groups[0]
    xs = [c["cx"] for c in g]
    assert 90 - 20 < min(xs) < 90 + 70 + 20   # 编号戳在色块中心附近


def test_analyze_frame_end_to_end_with_fake_vlm():
    img, pw = make_composite("33")
    ok, enc = sp.cv2.imencode(".jpg", img)
    wanted = {"33": {"color": "red", "shape": "cylinder", "size": "7x7x8"}}
    calls = {"n": 0}

    def fake_invoke(parts):
        text = next(p for p in parts if p["type"] == "text")["text"]
        if "蒙太奇" in text:
            return '{"1": 33}'
        calls["n"] += 1
        if calls["n"] == 1:      # 首次返回完全非 JSON，重试应恢复
            return "33 是个杯子"
        return '{"33": "cup"}'

    res = sp.analyze_frame(enc.tobytes(), wanted, fake_invoke)
    assert "33" in res
    assert res["33"]["cat"] == "cup"
    assert res["33"]["area"] > 0          # 洪泛区域面积即票权基础
    assert len(res["33"]["box"]) == 4
    assert calls["n"] == 2                # 首败重试生效


def test_parse_helpers():
    assert sp.parse_transcription('{"1": 36, "2": "8", "3": "X", "4": 1234}') \
        == {1: "36", 2: "8"}
    assert sp.parse_classification('{"36":"pillow","37":"桌子","38":"other"}') \
        == {"36": "pillow", "37": "other", "38": "other"}


def test_parse_tolerates_unquoted_bare_values():
    # 实机 train_v4b r1_1 教训：模型随机给 X 不加引号 → 整体 JSON 解析炸掉
    raw = '{"1":13,"2":4,"3":14,"17":X,"18":X,"23":37,"25":45}'
    assert sp.parse_transcription(raw) == {1: "13", 2: "4", 3: "14",
                                           23: "37", 25: "45"}
    assert sp.parse_classification('{"36":pillow,"37":cup}') \
        == {"36": "pillow", "37": "cup"}


def test_parse_tolerates_curly_quotes_and_trailing_comma():
    # 实机 train_v4c 教训：中文 prompt 诱发弯引号 → json.loads 不认
    raw = "{\u201c13\u201d:\u201cpillow\u201d,\u201c34\u201d:\u201ccup\u201d,}"
    assert sp.parse_classification(raw) == {"13": "pillow", "34": "cup"}


def test_confirm_image_center_crop_with_badge():
    img, pw = make_composite()
    ok, enc = sp.cv2.imencode(".jpg", img)
    out = sp.confirm_image(enc.tobytes(), "36")
    assert out
    dec = sp.cv2.imdecode(np.frombuffer(out, np.uint8), sp.cv2.IMREAD_COLOR)
    assert dec.shape[0] > 60 and dec.shape[1] > 60


# ------------------------------------------------------------------ #
# 票权公式
# ------------------------------------------------------------------ #


def test_vote_weight_median_and_clips():
    W = TidyroomAgent._vote_weight
    areas = [100, 400, 900]              # 中位数 400
    assert abs(W(100, areas) - 0.5) < 1e-9
    assert abs(W(400, areas) - 1.0) < 1e-9
    assert abs(W(900, areas) - 1.5) < 1e-9
    assert W(4, areas) == TidyroomAgent._W_MIN        # sqrt(0.01)=0.1 → 截 0.3
    assert W(40000, areas) == TidyroomAgent._W_MAX    # sqrt(100)=10 → 截 3.0
    assert W(0, areas) == 1.0                          # 洪泛失败平权回退


# ------------------------------------------------------------------ #
# finalize：规则×加权票合流与歧义判定
# ------------------------------------------------------------------ #


def obj(oid, shape="cylinder", x=500, y=400, dz=8, dx=7, dy=7):
    return {"object_id": oid, "color": "red", "shape": shape,
            "place_location": {"X": x, "Y": y, "Z": 2},
            "world_aabb": {"min": {"x": x - dx / 2, "y": y - dy / 2, "z": 0},
                           "max": {"x": x + dx / 2, "y": y + dy / 2, "z": dz}}}


def v4_agent(world, votes=None, cats=None, rule=()):
    a = TidyroomAgent.__new__(TidyroomAgent)
    a._world = {str(o["object_id"]): o for o in world}
    a._item_votes = votes or {}
    a._item_votes_n = {oid: Counter({c: 1 for c in cnt})
                       for oid, cnt in (votes or {}).items()}  # 默认每票 1 帧
    a._categories = cats or {}
    a._rule_cats = set(rule)
    a._ambiguous = set()
    a._done_oids = set()
    a._locked = False
    a._containers = {}
    a._container_votes = {}
    a._trace = lambda *k, **kw: None
    return a


def test_finalize_rule_stands_without_votes():
    a = v4_agent([obj("31")], cats={"31": "cup"}, rule={"31"})
    a._finalize_recognition()
    assert a._categories["31"] == "cup" and "31" not in a._ambiguous
    assert a._locked


def test_finalize_no_votes_no_rule_is_ambiguous():
    a = v4_agent([obj("31")])
    a._finalize_recognition()
    assert "31" in a._ambiguous


def test_finalize_split_votes_ambiguous():
    a = v4_agent([obj("31")],
                 votes={"31": Counter({"cup": 1.0, "trash": 1.0})})
    a._finalize_recognition()
    assert "31" in a._ambiguous          # top1 < 1.5×次优


def test_finalize_clear_top_sets_category():
    # 单帧但高权（w=2.0 ≥ _SINGLE_W=1.5，近距大区域）→ 直接定案
    a = v4_agent([obj("31")],
                 votes={"31": Counter({"cup": 2.0, "trash": 0.5})})
    a._finalize_recognition()
    assert a._categories["31"] == "cup" and "31" not in a._ambiguous


def test_finalize_single_low_vote_is_ambiguous():
    # 回放教训：远距单帧低权票（w≈1.0）不得定案（鞋曾被误判枕）
    a = v4_agent([obj("31")], votes={"31": Counter({"pillow": 1.0})})
    a._finalize_recognition()
    assert "31" in a._ambiguous and "31" not in a._categories


def test_finalize_two_frames_agree_sets_category():
    a = v4_agent([obj("31")], votes={"31": Counter({"shoe": 1.9})})
    a._item_votes_n["31"] = Counter({"shoe": 2})       # 两帧同票
    a._finalize_recognition()
    assert a._categories["31"] == "shoe" and "31" not in a._ambiguous


def test_finalize_overturns_rule_when_weighted_margin_met():
    a = v4_agent([obj("31", shape="cylinder", dz=45, dx=16, dy=16)],
                 votes={"31": Counter({"pillow": 2.5})},
                 cats={"31": "cup"}, rule={"31"})
    a._finalize_recognition()
    assert a._categories["31"] == "pillow" and "31" not in a._rule_cats


def test_finalize_rule_vlm_conflict_below_threshold_ambiguous():
    a = v4_agent([obj("31", shape="cylinder", dz=45, dx=16, dy=16)],
                 votes={"31": Counter({"pillow": 1.0})},
                 cats={"31": "cup"}, rule={"31"})
    a._finalize_recognition()
    assert "31" in a._ambiguous and a._categories["31"] == "cup"


def test_finalize_other_vote_is_ambiguous():
    # other 票不进投票箱（harvest 过滤），但若无票且无规则也歧义——
    # 这里验证有票但全被滤掉时的行为
    a = v4_agent([obj("31")], votes={"31": Counter()})
    a._finalize_recognition()
    assert "31" in a._ambiguous


def test_finalize_skips_point_objects():
    a = v4_agent([{"object_id": "9", "color": "Unknown", "shape": "Unknown",
                   "place_location": {"X": 1, "Y": 1, "Z": -10},
                   "world_aabb": {"min": {"x": 1, "y": 1, "z": -10},
                                   "max": {"x": 1, "y": 1, "z": -10}}}])
    a._finalize_recognition()
    assert "9" not in a._ambiguous


def test_finalize_skips_resolved_containers():
    # 垃圾桶尺寸可过物品筛选，但已是解析容器 → 不定类不到场确认
    binobj = obj("18", shape="box", dz=35, dx=27, dy=27)
    a = v4_agent([binobj])
    a._containers = {"trash_bin": binobj}
    a._finalize_recognition()
    assert "18" not in a._ambiguous and "18" not in a._categories


# ------------------------------------------------------------------ #
# 队列：歧义件合并（不重复入队、confirm 标记、保留先验类别）
# ------------------------------------------------------------------ #


def queue_agent():
    a = v4_agent([obj("31"), obj("32", shape="irregular", dz=6)])
    a._containers = {"trash_bin": obj("18", shape="box", x=814, y=267, dz=35, dx=27, dy=27),
                     "table": obj("15", shape="rectangle", x=577, y=365, dz=29, dx=72, dy=137)}
    a._container_slots = {}
    a._pair_cats = set()
    a._pair_shoes = lambda: None
    return a


def test_queue_marks_ambiguous_with_prior_cat():
    a = queue_agent()
    a._categories = {"31": "cup", "32": "trash"}
    a._ambiguous = {"31"}                # 31 歧义但保留 cup 先验
    a._rebuild_queue()
    t31 = next(t for t in a._queue if t["oid"] == "31")
    assert t31["confirm"] and t31["cat"] == "cup"
    t32 = next(t for t in a._queue if t["oid"] == "32")
    assert not t32["confirm"]
    assert len([t for t in a._queue if t["oid"] == "31"]) == 1  # 不重复
    assert a._queue[-1]["oid"] == "31"   # 歧义件排最后


def test_queue_ambiguous_without_cat_enters_without_container():
    a = queue_agent()
    a._categories = {}
    a._ambiguous = {"31"}
    a._rebuild_queue()
    assert len(a._queue) == 1 and a._queue[0]["oid"] == "31"
    assert a._queue[0]["cat"] is None and a._queue[0]["confirm"]


# ------------------------------------------------------------------ #
# 规则补缺（§7-18）：shape=pillow / box
# ------------------------------------------------------------------ #


def test_rule_pillow_shape():
    a = v4_agent([obj("36", shape="pillow", dz=33, dx=56, dy=28)])
    cat = a._rule_category("36")
    assert cat == "pillow"


def test_rule_box_small_nonblack_is_trash_but_black_bin_not():
    a = v4_agent([obj("40", shape="box", dz=30, dx=25, dy=25),
                  obj("18", shape="box", dz=35, dx=27, dy=27)])
    a._world["40"]["color"] = "gray"
    a._world["18"]["color"] = "black"
    assert a._rule_category("40") == "trash"
    assert a._rule_category("18") is None   # 黑 box=垃圾桶本体，不自搬自


def test_scene_prop_plant_excluded():
    # test R12 复盘：40 号=玄关盆栽 shape=plant 21×21×36，不可交互非五类
    plant = obj("40", shape="plant", dz=36, dx=21, dy=21)
    a = v4_agent([plant, obj("31")])
    assert a._is_scene_prop(plant) and not a._is_scene_prop(a._world["31"])
    a._apply_rule_categories()
    a._finalize_recognition()
    assert "40" not in a._categories and "40" not in a._ambiguous


def test_high_background_props_excluded_ground_items_pass():
    # 用户复盘点名的高处背景构件（吊灯/墙板/挂墙盒）须被基座高度挡住；
    # 贴地真物品与容器（茶几 z16）须通过
    wall_box = obj("58", shape="box", dz=32, dx=56, dy=44)   # R12-58 挂墙灰盒
    wall_box["place_location"]["Z"] = 128
    item = obj("32", shape="cylinder", dz=8)                  # 贴地真物品
    table = obj("15", shape="rectangle", dz=29, dx=72, dy=137)
    table["place_location"]["Z"] = 16                          # 容器：茶几
    a = v4_agent([wall_box, item, table])
    assert a._too_high(wall_box) and not a._too_high(item)
    assert not a._too_high(table)
    a._apply_rule_categories()
    a._finalize_recognition()
    assert "58" not in a._categories and "58" not in a._ambiguous
    assert "32" in a._categories or "32" in a._ambiguous


def test_flat_item_not_point_filtered():
    # R17-33 教训：薄片物品 10×10×1（垫子/杯垫类）不该被点状过滤
    # （墙角标记是 min=0 精确退化，阈值降到 1 后仍被排除）
    flat = obj("33", shape="cylinder", dz=1, dx=10, dy=10)
    point = {"object_id": "9", "color": "Unknown", "shape": "Unknown",
             "place_location": {"X": 1, "Y": 1, "Z": -10},
             "world_aabb": {"min": {"x": 1, "y": 1, "z": -10},
                             "max": {"x": 1, "y": 1, "z": -10}}}
    a = v4_agent([flat, point])
    assert not a._is_point(flat) and a._is_point(point)
    a._apply_rule_categories()
    assert "33" in a._categories or "33" in a._ambiguous  # 进入识别流程


def test_pillow_size_floor_and_tiny_fallback():
    # R22-57 red drumstick 11×10×6 / R14-33 rock 10×6×4 曾被误判 pillow
    drumstick = obj("57", shape="drumstick", dz=6, dx=11, dy=10)
    drumstick["color"] = "red"
    rock = obj("33", shape="rock", dz=4, dx=10, dy=6)
    rock["color"] = "brown"
    small_oval = obj("34", shape="oval", dz=7, dx=9, dy=9)
    small_oval["color"] = "brown"
    neck_pillow = obj("32", shape="cylinder", dz=15, dx=45, dy=16)  # 颈枕
    real_pillow = obj("36", shape="pillow", dz=35, dx=39, dy=44)
    a = v4_agent([drumstick, rock, small_oval, neck_pillow, real_pillow])
    a._apply_rule_categories()
    c = a._categories
    assert c["57"] == "trash"      # 极小 drumstick → 垃圾桶
    assert c["33"] == "trash"      # rock → 垃圾
    assert c["34"] == "trash"      # 极小 oval 兜底 → 垃圾
    assert c["32"] == "pillow"     # 45cm 颈枕不受影响
    assert c["36"] == "pillow"     # 39cm 真枕通过下限


def test_small_pillow_shape_not_pillow():
    # shape=pillow 但只有 15cm → 不是枕，走极小兜底 trash
    mini = obj("20", shape="pillow", dz=6, dx=15, dy=14)
    a = v4_agent([mini])
    a._apply_rule_categories()
    assert a._categories["20"] == "trash"


def test_can_rul_still_works_for_tiny_cylinder():
    # 极小兜底不得误伤：短圆柱罐（R9 33 号 7.6cm 红罐）仍是 cup
    can = obj("33", shape="cylinder", dz=8, dx=7, dy=7)
    can["color"] = "red"
    a = v4_agent([can])
    a._apply_rule_categories()
    assert a._categories["33"] == "cup"


def test_ring_and_slice_are_food():
    # R23 用户复盘：33 ring(甜甜圈 31×32×11)/34 slice(切片 16×3×16) 是
    # food→茶几，曾被极小兜底误送桶；slice 16cm 须先于极小兜底命中
    donut = obj("33", shape="ring", dz=11, dx=31, dy=32)
    donut["color"] = "brown"
    sl = obj("34", shape="slice", dz=16, dx=16, dy=3)
    sl["color"] = "brown"
    a = v4_agent([donut, sl])
    a._apply_rule_categories()
    assert a._categories["33"] == "food"
    assert a._categories["34"] == "food"


def test_big_item_prior_pillow():
    # R24-33 用户定策：57cm 灰方块是抱枕——历史可搬大件 max≥40 只有
    # pillow；ring 31cm/靴 25cm 不受影响
    square = obj("33", shape="square", dz=48, dx=57, dy=58)
    square["color"] = "gray"
    donut = obj("34", shape="ring", dz=11, dx=31, dy=32)
    donut["color"] = "brown"
    a = v4_agent([square, donut])
    a._apply_rule_categories()
    assert a._categories["33"] == "pillow"
    assert a._categories["34"] == "food"          # 31cm ring 不触发大件先验


def test_mid_size_no_rule_still_none():
    # 20~40cm 无形状规则件不落任何兜底（走确认）
    mid = obj("35", shape="oval", dz=22, dx=24, dy=20)
    mid["color"] = "gray"
    a = v4_agent([mid])
    a._apply_rule_categories()
    assert "35" not in a._categories


def test_big_prior_bounded_and_chair_excluded():
    # R25 教训：大件先验曾把冰箱/沙发/餐椅等 29 个家具全标 pillow
    chair = obj("22", shape="chair", dz=80, dx=61, dy=61)
    chair["color"] = "gray"
    fridge = obj("7", shape="Unknown", dz=160, dx=29, dy=51)
    fridge["color"] = "Unknown"
    big_square = obj("33", shape="square", dz=53, dx=65, dy=63)  # 可搬区间内
    big_square["color"] = "white"
    a = v4_agent([chair, fridge, big_square])
    a._apply_rule_categories()
    assert "22" not in a._categories       # chair 拉黑
    assert "7" not in a._categories        # >80cm 超出可搬区间不判
    assert a._categories["33"] == "pillow"  # 65cm 在 40~80 区间内


def test_max_dim_cap_100_admits_big_pillows():
    # R26-32 白枕 81cm / R22-32 灰枕 91cm 曾被 80 上限整件误挡
    p81 = obj("32", shape="rectangle", dz=37, dx=81, dy=55)
    p81["color"] = "white"
    p91 = obj("32", shape="pillow", dz=86, dx=91, dy=44)
    p91["color"] = "gray"
    big_unknown = obj("36", shape="square", dz=95, dx=95, dy=40)
    a = v4_agent([p81, p91, big_unknown])
    a._apply_rule_categories()
    assert a._categories["32"] == "pillow"
    assert a._categories["36"] == "pillow"          # 95cm 大件先验（<100）
    q = a
    q._containers = {"sofa": obj("14", shape="rectangle", dz=99, dx=112, dy=410)}
    q._container_slots = {}
    q._pair_cats = set()
    q._pair_shoes = lambda: None
    q._ambiguous = set()
    q._rebuild_queue()
    oids = [t["oid"] for t in q._queue]
    assert "32" in oids and "36" in oids           # 81/95cm 均入队


def test_furniture_thickness_collision_guard():
    # 用户提议：用椅子三维签名做碰撞——61×61×80 块状物即使 shape=Unknown
    # 也不进 pillow 判定/队列（历史物品 min≤44，椅 min=61）
    unk_chair = obj("21", shape="Unknown", dz=80, dx=61, dy=61)
    unk_chair["color"] = "gray"
    thick_pillow_shape = obj("22", shape="pillow", dz=80, dx=65, dy=65)  # 块状"枕"
    real_thick_pillow = obj("32", shape="pillow", dz=86, dx=91, dy=44)   # 44<50 ✓
    a = v4_agent([unk_chair, thick_pillow_shape, real_thick_pillow])
    a._apply_rule_categories()
    assert "21" not in a._categories       # Unknown 椅：大件先验被厚度守卫拦下
    assert "22" not in a._categories       # 块状 pillow 形：同拦
    assert a._categories["32"] == "pillow"  # 91×44×86 真厚枕通过


def test_sandal_is_shoe_and_big_cylinder_grounded():
    # R27 教训：29cm 凉鞋词汇缺失落空带；101cm 绿植 cylinder 被标 pillow
    sandal = obj("52", shape="sandal", dz=12, dx=29, dy=12)
    sandal["color"] = "blue"
    plant = obj("11", shape="cylinder", dz=101, dx=86, dy=79)
    plant["color"] = "green"
    neck = obj("33", shape="cylinder", dz=15, dx=45, dy=16)  # 颈枕仍 pillow
    a = v4_agent([sandal, plant, neck])
    a._apply_rule_categories()
    assert a._categories["52"] == "shoe"       # 凉鞋直判鞋柜
    assert "11" not in a._categories           # 101cm 绿植不判
    assert a._categories["33"] == "pillow"     # 45cm 颈枕不受影响


def test_paper_scrap_flat_object_to_trash():
    # R29 用户定案：59/61 号 [10,10,0] 零高度平面物=纸屑档，直判垃圾桶；
    # 全退化 [0,0,0] 点标记仍被排除
    scrap1 = obj("59", shape="irregular", dz=0, dx=10, dy=10)
    scrap1["color"] = "white"
    scrap2 = obj("61", shape="irregular", dz=0, dx=10, dy=10)
    scrap2["color"] = "white"
    marker = {"object_id": "9", "color": "Unknown", "shape": "Unknown",
              "place_location": {"X": 1, "Y": 1, "Z": -10},
              "world_aabb": {"min": {"x": 1, "y": 1, "z": -10},
                              "max": {"x": 1, "y": 1, "z": -10}}}
    a = v4_agent([scrap1, scrap2, marker])
    assert not a._is_point(scrap1) and a._is_point(marker)
    a._apply_rule_categories()
    assert a._categories["59"] == "trash" and a._categories["61"] == "trash"
    assert "9" not in a._categories
    a._containers = {"trash_bin": obj("18", shape="box", dz=35, dx=27, dy=27)}
    a._container_slots = {}
    a._pair_cats = set()
    a._pair_shoes = lambda: None
    a._ambiguous = set()
    a._rebuild_queue()
    assert {"59", "61"} <= {t["oid"] for t in a._queue}   # 纸屑入队进桶
