# -*- coding: utf-8 -*-
"""dashcam 行车记录仪标注模块测试：合成复合图确定性往返。

不依赖 R9 引导模板（v0 覆盖数字有限），而是用 cv2 画的合成数字自建
模板，验证「提取→分组→识别→洪泛→画框」整条链路与降级路径。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from cqairace import dashcam as dc


def _synth_composite(labels: list[tuple[str, int, int, tuple[int, int, int]]]):
    """合成 1280x720 复合图。labels: (数字, 右面板x, 右面板y, 块色BGR)，
    在 (x,y) 处画平涂色块（120x90）并在其上写黑色数字。"""
    img = np.full((720, 1280, 3), 200, np.uint8)
    rng = np.random.default_rng(7)
    img[:, :640] = rng.integers(120, 180, (720, 640, 3), dtype=np.uint8)  # 左RGB
    for num, x, y, color in labels:
        cv2.rectangle(img, (640 + x, y), (640 + x + 120, y + 90), color, -1)
        cv2.putText(img, num, (640 + x + 30, y + 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 2, cv2.LINE_AA)
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    return enc.tobytes()


@pytest.fixture()
def synth_templates(monkeypatch):
    """用同一合成字体画 0-9 自建模板（大字号=全分辨率场景的代理）。"""
    temps = []
    for d in "0123456789":
        canvas = np.zeros((60, 60), np.uint8)
        cv2.putText(canvas, d, (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, 255, 2, cv2.LINE_AA)
        ys, xs = np.nonzero(canvas)
        crop = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        temps.append((d, dc._normalize(crop > 127)))
    monkeypatch.setattr(dc, "_TEMPLATES", temps)
    return temps


def test_roundtrip_found_numbers(synth_templates):
    raw = _synth_composite([
        ("33", 50, 100, (60, 160, 60)),
        ("7", 300, 300, (200, 120, 40)),
        ("12", 500, 500, (40, 80, 200)),
    ])
    ann = dc.annotate(raw, {"33": "cup"}, {"33": "rule"})
    assert ann is not None
    found = set(ann[1]["labels_found"])
    assert found == {"33", "7", "12"}


def test_box_on_left_panel_maps_region(synth_templates):
    raw = _synth_composite([("33", 50, 100, (60, 160, 60))])
    ann = dc.annotate(raw, {"33": "cup"}, {"33": "rule"})
    assert ann is not None
    img = cv2.imdecode(np.frombuffer(ann[0], np.uint8), cv2.IMREAD_COLOR)
    # 左半 640 内应有绿色框像素（placed=绿同款 rule=白；查非背景框线）
    left = img[:, :640]
    # 白色框线
    white = int(np.all(left > 240, axis=2).sum())
    assert white > 50, f"左面板未见白色标注框 (white={white})"


def test_garbage_bytes_returns_none(synth_templates):
    assert dc.annotate(b"not-an-image", None, None) is None
    assert dc.annotate(b"", None, None) is None


def test_save_frame_fallback_raw(synth_templates, tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "_TEMPLATES", [])
    raw = _synth_composite([("33", 50, 100, (60, 160, 60))])
    ok = dc.save_frame(tmp_path, "t1", raw, [{"object_id": 33}], {}, {})
    assert ok is False                       # 识别失败 → 存原图
    f = tmp_path / "frame_t1.jpg"
    assert f.exists() and f.read_bytes()
    meta = tmp_path / "frame_t1_meta.json"
    assert meta.exists()
    import json
    assert json.loads(meta.read_text())["annotated"] is False


def test_group_labels_no_cross_merge(synth_templates):
    glyphs = [
        {"x": 10, "y": 50, "w": 10, "h": 20, "cx": 15, "cy": 60, "digit": "1"},
        {"x": 22, "y": 50, "w": 10, "h": 20, "cx": 27, "cy": 60, "digit": "2"},
        {"x": 400, "y": 50, "w": 10, "h": 20, "cx": 405, "cy": 60, "digit": "7"},
    ]
    groups = dc._group_labels(glyphs)
    assert len(groups) == 2
    assert "".join(g["digit"] for g in groups[0]) == "12"
