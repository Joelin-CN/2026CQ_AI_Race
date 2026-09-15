# -*- coding: utf-8 -*-
"""静态容器先验逻辑测试：先验锚定选择 / 远偏校正 / 兜底填充 / 自适应保留。

数据来自 R11 逐帧 AABB 实测（v2-system-notes.md §7-18 证据表）。
"""
from __future__ import annotations

from collections import Counter

from cqairace.tidyroom_agent import TidyroomAgent


def obj(oid, color, shape, x, y, z0, z1, dx, dy, dz):
    return {"object_id": oid, "color": color, "shape": shape,
            "place_location": {"X": x, "Y": y, "Z": z0},
            "world_aabb": {"min": {"x": x - dx / 2, "y": y - dy / 2, "z": z0},
                            "max": {"x": x + dx / 2, "y": y + dy / 2, "z": z1}}}


def agent_with(world):
    a = TidyroomAgent.__new__(TidyroomAgent)
    a._world = {str(o["object_id"]): o for o in world}
    a._container_votes = {}
    a._containers = {}
    return a


# R11 实测容器
COFFEE = obj(15, "brown", "rectangle", 577, 365, 16, 45, 72, 137, 29)     # 茶几
SOFA = obj(14, "gray", "rectangle", 282, 374, 0, 99, 112, 410, 99)        # 主沙发
BIN = obj(18, "black", "box", 814, 267, 4, 39, 27, 27, 35)                # 垃圾桶
CABINET = obj(46, "brown", "rectangle", 780, -144, 3, 63, 109, 23, 60)    # 鞋柜
SIDEBOARD = obj(8, "brown", "rectangle", -328, 243, 0, 83, 75, 207, 84)   # 餐边大柜
BIG_IMPOSTOR = obj(99, "brown", "rectangle", 900, -700, 0, 40, 180, 120, 40)  # 远处超大"桌"


def test_rule_prefers_prior_nearest_over_volume():
    a = agent_with([COFFEE, BIG_IMPOSTOR])
    got = a._rule_container("table")
    assert str(got["object_id"]) == "15"      # 先验附近者胜，而非体积最大者


def test_rule_far_fallback_when_no_prior_hit():
    a = agent_with([BIG_IMPOSTOR])            # 场景里没有先验附近的桌
    got = a._rule_container("table")
    assert str(got["object_id"]) == "99"      # 回退体积最大（自适应保留）


def test_resolve_corrects_far_vote():
    a = agent_with([COFFEE, SIDEBOARD])
    a._container_votes = {"8": Counter({"table": 3})}   # VLM 把餐边大柜投成 table
    a._resolve_containers()
    assert str(a._containers["table"]["object_id"]) == "15"


def test_resolve_prior_fills_when_rule_and_vlm_dead():
    # 茶几 color=Unknown（规则不命中）、无投票：先验按几何+坐标兜底
    unknown_table = obj(15, "Unknown", "rectangle", 577, 365, 16, 45, 72, 137, 29)
    a = agent_with([unknown_table, SOFA])
    a._resolve_containers()
    assert str(a._containers["table"]["object_id"]) == "15"
    assert str(a._containers["sofa"]["object_id"]) == "14"


def test_prior_container_size_gates():
    a = agent_with([COFFEE, SOFA, BIN, CABINET, SIDEBOARD])
    assert str(a._prior_container("table")["object_id"]) == "15"
    assert str(a._prior_container("sofa")["object_id"]) == "14"   # 餐边柜够大但离先验远
    assert str(a._prior_container("trash_bin")["object_id"]) == "18"
    assert str(a._prior_container("shoe_cabinet")["object_id"]) == "46"


def test_no_prior_object_keeps_dynamic_choice():
    # 先验附近空无一物（换地图）：动态选择不被清空
    a = agent_with([BIG_IMPOSTOR])
    a._container_votes = {"99": Counter({"table": 2})}
    a._resolve_containers()
    assert str(a._containers["table"]["object_id"]) == "99"
