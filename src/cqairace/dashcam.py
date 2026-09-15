# -*- coding: utf-8 -*-
"""行车记录仪 YOLO 式标注：解析感知复合图右面板编号 → 左面板画框+类别标签。

感知复合图（官方 simulation_interface_guide.md:43-45）：左半 RGB 原图 +
右半带数字 ID 标注的语义分割图，object_id 与数字一一对应。本模块把
右面板的黑色数字识别出来（模板匹配），从数字位置洪泛出所属色块区域，
映射回左半 RGB 画框并标注「oid:类别」，按判定来源/状态着色：

    rule=白  vlm=黄  pair=青  blacklist=红  placed=绿  gaveup=橙
    container=蓝  未知判定=灰

模板来源：temp/p4_tidyroom/dashcam_pin.py 引导（R9 存帧），当前 v0 覆盖
高置信数字 {0,1,3,4,7,9}；全分辨率帧（acquire 2560x720）字形更大，
识别更稳，模板待 scout 全分辨率帧重标后替换。识别失败的编号只画灰框。

设计约束：任何异常都不允许影响 agent 主流程——调用方 _save_frame 以
try/except 包裹，失败时回退存原图。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

TEMPLATES_PATH = Path(__file__).with_name("dashcam_templates.json")

# 画框颜色 (BGR)
COLORS: dict[str, tuple[int, int, int]] = {
    "rule": (255, 255, 255),
    "vlm": (0, 255, 255),
    "pair": (255, 255, 0),
    "blacklist": (0, 0, 255),
    "placed": (0, 255, 0),
    "gaveup": (0, 128, 255),
    "container": (255, 150, 0),
    "unknown": (160, 160, 160),
}

_GH, _GW = 24, 16       # 模板网格高x宽
_MATCH_TH = 0.34        # 最佳模板 XOR 上限
_MARGIN = 0.04          # 与次优不同数字的最小距离
_FLOOD_DIFF = 18.0      # 洪泛颜色容差
_MAX_REGION_FRAC = 0.22  # 洪泛区域占面板上限（超出视为串色到地板/墙，弃用）


def _load_templates() -> list[tuple[str, np.ndarray]]:
    try:
        data = json.loads(TEMPLATES_PATH.read_text())
    except Exception:
        return []
    out = []
    for t in data.get("templates", []):
        bits = np.frombuffer(base64.b64decode(t["bitmap"]), np.uint8)
        bm = np.unpackbits(bits)[: t["h"] * t["w"]].reshape(t["h"], t["w"])
        out.append((str(t["digit"]), bm.astype(np.uint8)))
    return out


_TEMPLATES = _load_templates()


def _normalize(mask: np.ndarray) -> np.ndarray:
    """字形位图 → 保持纵横比的 24x16 网格（居中补零）。"""
    h, w = mask.shape
    s = min(_GW / w, _GH / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(mask.astype(np.uint8) * 255, (nw, nh),
                   interpolation=cv2.INTER_AREA)
    out = np.zeros((_GH, _GW), np.uint8)
    y0, x0 = (_GH - nh) // 2, (_GW - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return (out > 127).astype(np.uint8)


def _match_digit(norm: np.ndarray) -> str | None:
    """模板匹配单字形；不满足阈值/边际条件返回 None。"""
    best_d, best_v, second = None, 1e9, 1e9
    for d, t in _TEMPLATES:
        v = float(np.mean(t != norm))
        if v < best_v:
            if d != best_d:
                second = best_v
            best_d, best_v = d, v
        elif d != best_d and v < second:
            second = v
    if best_d is None or best_v > _MATCH_TH or (second - best_v) < _MARGIN:
        return None
    return best_d


def _extract_glyphs(panel: np.ndarray) -> list[dict]:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    mask = (gray < 80).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (6 <= h <= 90 and 2 <= w <= 70 and area >= 18):
            continue
        if area / (w * h) < 0.33:
            continue
        comp = (lab[y:y + h, x:x + w] == i)
        digit = _match_digit(_normalize(comp))
        out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "cx": int(x + w / 2), "cy": int(y + h / 2),
                    "digit": digit})
    return out


def _group_labels(glyphs: list[dict]) -> list[list[dict]]:
    """相邻字形组成编号（同一标签的各位数字 y 对齐、x 紧邻、高度相近）。"""
    gs = sorted([g for g in glyphs if g["digit"]], key=lambda g: g["cx"])
    used = [False] * len(gs)
    groups = []
    for i, g in enumerate(gs):
        if used[i]:
            continue
        grp = [g]
        used[i] = True
        for j in range(i + 1, len(gs)):
            if used[j]:
                continue
            g2 = gs[j]
            last = grp[-1]
            if (abs(g2["cy"] - g["cy"]) <= max(6, g["h"] * 0.4)
                    and 0 <= g2["cx"] - last["cx"] <= last["w"] + g2["w"] + 8
                    and abs(g2["h"] - g["h"]) <= max(4, g["h"] * 0.5)):
                grp.append(g2)
                used[j] = True
        grp.sort(key=lambda g: g["cx"])
        groups.append(grp)
    return groups


def _label_region(panel: np.ndarray, grp: list[dict]) -> tuple[int, int, int, int]:
    """从编号位置洪泛出所属色块 bbox（右面板坐标）。失败退化为编号邻域框。"""
    h, w = panel.shape[:2]
    x0 = min(g["x"] for g in grp)
    x1 = max(g["x"] + g["w"] for g in grp)
    y0 = min(g["y"] for g in grp)
    y1 = max(g["y"] + g["h"] for g in grp)
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    best_box = None
    best_area = 0
    for sx, sy in ((x0 + x1) // 2, y0 - 3), ((x0 + x1) // 2, y1 + 3), \
                  (x0 - 3, (y0 + y1) // 2), (x1 + 3, (y0 + y1) // 2):
        if not (0 <= sx < w and 0 <= sy < h) or gray[sy, sx] < 80:
            continue
        flood = np.zeros((h + 2, w + 2), np.uint8)
        try:
            cv2.floodFill(panel.copy(), flood, (sx, sy), None,
                          loDiff=(_FLOOD_DIFF,) * 3, upDiff=(_FLOOD_DIFF,) * 3,
                          flags=cv2.FLOODFILL_MASK_ONLY | 8 | (255 << 8))
        except cv2.error:
            continue
        m = flood[1:-1, 1:-1]
        if m.sum() // 255 > h * w * _MAX_REGION_FRAC:
            continue
        ys, xs = np.nonzero(m)
        area = int(m.sum() // 255)
        if area > best_area:
            best_area = area
            best_box = (int(xs.min()), int(ys.min()),
                        int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    if best_box is None:
        best_box = (max(0, x0 - 30), max(0, y0 - 22), x1 - x0 + 60, y1 - y0 + 44)
    return best_box


def annotate(img_bytes: bytes,
             labels: dict[str, str] | None = None,
             colors: dict[str, str] | None = None) -> tuple[bytes, dict] | None:
    """标注一帧复合图。只画 labels 里登记的物体（agent 的可搬物品清单），
    编号未命中清单的不画——避免满图框住椅子/墙面/地板等噪声。
    返回 (标注后 jpg bytes, 识别元数据)；失败返回 None。"""
    if not img_bytes:
        return None
    buf = np.frombuffer(img_bytes, np.uint8)
    try:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if img is None:
        return None
    h, w = img.shape[:2]
    pw = w // 2
    if pw < 10:
        return None
    panel = img[:, pw:]
    labels = labels or {}
    colors = colors or {}
    scale = max(1.0, pw / 640.0)
    # oid → 最大区域（同一编号的多次戳章合并，取最完整的一块）
    best: dict[str, tuple[int, int, int, int]] = {}
    found_all: list[str] = []
    for grp in _group_labels(_extract_glyphs(panel)):
        num = "".join(g["digit"] for g in grp)
        if not num.isdigit() or len(num) > 3:
            continue
        found_all.append(num)
        if num not in labels:
            continue
        bx, by, bw, bh = _label_region(panel, grp)
        if bw * bh < 64 or bw > pw * 0.6:
            continue
        prev = best.get(num)
        if prev is None or bw * bh > prev[2] * prev[3]:
            best[num] = (bx, by, bw, bh)
    for oid, (bx, by, bw, bh) in best.items():
        text = f"{oid}:{labels[oid]}"
        color = COLORS.get(colors.get(oid, "unknown"), COLORS["unknown"])
        thick = max(2, round(2 * scale))
        lx, ly = max(0, bx - pw), by                      # 映射到左面板
        cv2.rectangle(img, (lx, ly), (min(w - 1, lx + bw), min(h - 1, ly + bh)),
                      color, thick)
        cv2.rectangle(img, (pw + bx, by), (min(w - 1, pw + bx + bw),
                                           min(h - 1, by + bh)), color, 1)
        fs = 0.55 * scale
        ty = max(round(16 * scale), ly - thick)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs,
                                    max(1, round(scale)))
        cv2.rectangle(img, (lx, ty - round(16 * scale)),
                      (lx + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (lx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, color, max(1, round(scale)), cv2.LINE_AA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return (enc.tobytes(), {"labels_found": found_all,
                            "drawn": sorted(best)}) if ok else None


def save_frame(directory: Path, tag: str, img_bytes: bytes,
               objects: list[dict] | None = None,
               labels: dict[str, str] | None = None,
               colors: dict[str, str] | None = None) -> bool:
    """落盘一帧：标注 jpg + 感知元数据 json。失败返回 False（调用方回退存原图）。"""
    ann = None
    if img_bytes:
        try:
            ann = annotate(img_bytes, labels, colors)
        except Exception:
            ann = None
    directory.mkdir(parents=True, exist_ok=True)
    ok = bool(ann and ann[1].get("labels_found"))
    if ok:
        (directory / f"frame_{tag}.jpg").write_bytes(ann[0])
    else:
        (directory / f"frame_{tag}.jpg").write_bytes(img_bytes or b"")
    if objects is not None:
        try:
            (directory / f"frame_{tag}_meta.json").write_text(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tag": tag,
                "annotated": ok,
                "objects": objects,
                "labels_found": (ann[1].get("labels_found", []) if ann else []),
            }, ensure_ascii=False))
        except Exception:
            pass
    return ok


def main(argv: list[str] | None = None) -> int:
    """离线再标注: python -m cqairace.dashcam <frame.jpg> [--meta m.json]"""
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--meta", help="帧元数据 json（含 objects）")
    ap.add_argument("--out", help="输出路径，默认 <image>_ann.jpg")
    args = ap.parse_args(argv)
    raw = Path(args.image).read_bytes()
    labels: dict[str, str] = {}
    if args.meta and Path(args.meta).exists():
        objs = json.loads(Path(args.meta).read_text()).get("objects", [])
        labels = {str(o.get("object_id", "")): str(o.get("shape", "?"))
                  for o in objs}
    ann = annotate(raw, labels)
    if ann is None:
        print("标注失败（模板缺失或图像异常）", file=sys.stderr)
        return 1
    out = Path(args.out or (str(args.image)[:-4] + "_ann.jpg"))
    out.write_bytes(ann[0])
    print(f"识别编号: {ann[1]['labels_found']}\n输出: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
