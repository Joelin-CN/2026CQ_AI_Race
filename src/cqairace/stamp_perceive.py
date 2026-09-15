# -*- coding: utf-8 -*-
"""生产感知：截定位 → VLM 转写 → 洪泛区域 → 裁剪自标（v4 识别管线核心）。

背景（v2-system-notes §7-19/20）：感知复合图右面板的数字编号是 8-10px
固定像素渲染（模板 OCR 有 6/8/9、1/4 互混），且色块是随机实例色（与
物体真实颜色无关）——object_id 绑定只能靠"读数字戳"或"到场结构性绑定"。

本模块实现读号路线（离线 R11 存档帧验证，temp/p4_tidyroom/v4_probe.py）：
  1. 阈值+连通域定位数字戳（免模板，dashcam._extract_glyphs 的松化版，
     不做数字识别只做字形检测）；
  2. 戳区 ×6 放大拼蒙太奇 → VLM 一次整组转写（格号→编号）；
  3. 转写结果**必须命中当前帧 objects 的 object_id 集合**才采（绑号
     过滤，final_r11_figure.py 同款规则）；未命中的戳直接丢弃；
  4. 从戳位洪泛出色块区域（复用 dashcam._label_region）——面积即投票
     权重的基础（∝ 视觉清晰度，远帧低权/近帧高权）；
  5. 区域映射到左面板 RGB 裁剪、放大、代码自画大号编号（VLM 不再读
     小号，绑号错误归零）→ 拼格一次分类。

VLM 调用不在本模块内：`analyze_frame(..., vlm_invoke)` 接收回调
`vlm_invoke(content_parts: list[dict]) -> str`（OpenAI content 数组，
含 image_url/text），由调用方（agent 线程池 / 离线脚本）提供客户端、
重试与超时。任何一步失败返回空 dict，调用方回退旧整帧路径。
"""
from __future__ import annotations

import base64
import re

import cv2
import numpy as np

from .dashcam import _label_region

# 戳字形检测参数（R11 帧 + v4_probe 干跑校准：25 组/帧，含家具戳）
_GLYPH_GRAY_TH = 90      # 暗字形阈值（dashcam 模板路径用 80，这里只定位不识别可放宽）
_GLYPH_H = (6, 22)       # 字形高范围（8-10px 编号的连通域）
_GLYPH_W = (2, 22)
_GLYPH_AREA = 12
_GLYPH_FILL = 0.35
_MAX_DIGITS = 3          # 编号最多 3 位

# 裁剪自标
_CROP_SCALE = 2.5        # 左面板裁剪放大倍数（INTER_CUBIC）
_CROP_PAD = 0.35         # 区域四周外扩比例
_BADGE_H = 34            # 自画编号徽标高度（px，画在裁剪图左上角）
_GRID_COLS = 3
_MAX_GRID_CELLS = 9      # 拼格上限（超出拆批次的余地留给调用方）

_CATS = ("trash", "cup", "food", "shoe", "pillow", "other")

# ------------------------------------------------------------------ #
# prompt（轻量版，2026-09-15 讨论定稿：五桶白名单+视觉锚点+负例+最短输出）
# ------------------------------------------------------------------ #

PROMPT_TRANSCRIBE = (
    "这是从分割图截取放大的数字蒙太奇，每格左上角红色数字是格号，"
    "格内黑色字符是物体编号(1-2位数字)。请逐格读出编号，看不清写X。"
    '只输出 JSON：{"1":36,"2":8,...}，全部格都要列出，第一个字符必须是{。'
)

PROMPT_CLASSIFY_TMPL = (
    "你是家居整理助手。场景：仿真客厅+玄关，地面散落待整理小物件，"
    "图中每格是一个物件的特写，编号已画在每格左上角。\n"
    "类别白名单（五选一）：trash=碎纸团/包装袋/烟头；cup=杯子/饮料瓶/易拉罐；"
    "food=水果/零食；shoe=鞋靴；pillow=抱枕/靠垫/颈枕。\n"
    "判据：≥30cm细长圆柱是颈枕(pillow)不是杯子；穿在人脚上的鞋不算；"
    "家具/装饰/已整齐摆放的不算。\n"
    "各编号元数据（size_cm 为包围盒长宽高）：\n{meta}\n"
    '只输出 JSON：{{"编号":"类别",...}}全部编号都要给出，认不出的给 other，'
    "第一个字符必须是{{。"
)

PROMPT_CONFIRM_TMPL = (
    "家居整理场景。图是第一人称全视野，编号 {oid} 的目标物体在图内某处"
    "（不一定居中）。目标特征——颜色:{color}，形状:{shape}，尺寸约 {size}cm。\n"
    "请先按特征在图中找到该物体，再分类。类别白名单：trash/cup/food/shoe/"
    "pillow（易拉罐算 cup；≥30cm细长圆柱是颈枕算 pillow；穿在人脚上的鞋"
    "不算；图中其他物体一律忽略，只对目标分类）。\n"
    "shape 词义提示（目标 shape 命中时优先按此判断）：boot/shoe/sandal/"
    "slipper/loafer=鞋靴；round/circle/ring/slice=食物(水果/甜甜圈/切片)；"
    "rock/irregular=垃圾碎屑；drumstick=鸡腿(食物)；square/box/rectangle="
    "方块类(≥40cm多为抱枕,<20cm多为杂物)；其他未知词结合外观与尺寸判断。\n"
    "尺寸硬约束：pillow 只能是 ≥30cm 的抱枕/靠垫/颈枕；<20cm 且不像杯罐/"
    "水果的小物优先 trash。\n"
    '只输出 {{"{oid}":"..."}}，找不到目标则给 other。'
)


# ------------------------------------------------------------------ #
# 戳定位（免模板）
# ------------------------------------------------------------------ #

def locate_stamp_groups(panel: np.ndarray) -> list[list[dict]]:
    """右面板数字戳分组：阈值→连通域→几何分组。返回字形组（x/y/w/h/cx/cy）。"""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    mask = (gray < _GLYPH_GRAY_TH).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (_GLYPH_H[0] <= h <= _GLYPH_H[1]
                and _GLYPH_W[0] <= w <= _GLYPH_W[1] and area >= _GLYPH_AREA):
            continue
        if area / (w * h) < _GLYPH_FILL:
            continue
        comps.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                      "cx": int(x + w / 2), "cy": int(y + h / 2)})
    comps.sort(key=lambda c: c["cx"])
    used = [False] * len(comps)
    groups: list[list[dict]] = []
    for i, c in enumerate(comps):
        if used[i]:
            continue
        grp = [c]
        used[i] = True
        for j in range(i + 1, len(comps)):
            if used[j]:
                continue
            c2 = comps[j]
            last = grp[-1]
            if (abs(c2["cy"] - c["cy"]) <= max(5, c["h"] * 0.5)
                    and 0 <= c2["cx"] - last["cx"] <= last["w"] + c2["w"] + 6
                    and abs(c2["h"] - c["h"]) <= max(3, c["h"] * 0.4)):
                grp.append(c2)
                used[j] = True
        if len(grp) <= _MAX_DIGITS:
            groups.append(grp)
    return groups


# ------------------------------------------------------------------ #
# 图构造
# ------------------------------------------------------------------ #

def _jpg(arr: np.ndarray, quality: int = 92) -> bytes:
    ok, enc = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return enc.tobytes() if ok else b""


def stamp_montage(panel: np.ndarray, groups: list[list[dict]],
                  scale: int = 6) -> bytes:
    """全部戳区放大拼白底蒙太奇（红色格号画在格内左上）。"""
    cells = []
    for k, g in enumerate(groups, 1):
        gx = max(0, min(c["x"] for c in g) - 12)
        gy = max(0, min(c["y"] for c in g) - 10)
        gx2 = min(panel.shape[1], max(c["x"] + c["w"] for c in g) + 12)
        gy2 = min(panel.shape[0], max(c["y"] + c["h"] for c in g) + 10)
        crop = panel[gy:gy2, gx:gx2]
        if crop.size == 0:
            crop = np.zeros((40, 40, 3), np.uint8)
        big = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, str(k), (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 255), 3)
        cells.append(big)
    if not cells:
        return b""
    ch = max(c.shape[0] for c in cells) + 14
    cw = max(c.shape[1] for c in cells) + 14
    cols = 4
    rows = (len(cells) + cols - 1) // cols
    canvas = np.full((rows * ch + 8, cols * cw + 8, 3), 255, np.uint8)
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        canvas[r * ch + 14:r * ch + 14 + c.shape[0],
               col * cw + 8:col * cw + 8 + c.shape[1]] = c
    return _jpg(canvas)


def crop_badged(img: np.ndarray, pw: int, box: tuple[int, int, int, int],
                oid: str) -> np.ndarray | None:
    """右面板区域 box → 左面板裁剪放大 + 自画编号徽标。失败返回 None。"""
    h, w = img.shape[:2]
    bx, by, bw, bh = box
    lx = max(0, bx - pw)
    pad_x, pad_y = int(bw * _CROP_PAD) + 10, int(bh * _CROP_PAD) + 10
    x0, y0 = max(0, lx - pad_x), max(0, by - pad_y)
    x1, y1 = min(pw, lx + bw + pad_x), min(h, by + bh + pad_y)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    crop = img[y0:y1, x0:x1]
    crop = cv2.resize(crop, (int(crop.shape[1] * _CROP_SCALE),
                             int(crop.shape[0] * _CROP_SCALE)),
                      interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(crop, (0, 0), (int(_BADGE_H * 1.6), _BADGE_H), (0, 0, 0), -1)
    cv2.putText(crop, oid, (5, _BADGE_H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                (0, 255, 255), 2)
    return crop


def build_grid(crops: dict[str, np.ndarray]) -> bytes:
    """oid→裁剪图 拼格（≤_MAX_GRID_CELLS 时一图，多余截断由调用方分批）。"""
    items = sorted(crops.items(), key=lambda kv: int(kv[0]))[:_MAX_GRID_CELLS]
    if not items:
        return b""
    cs = [c for _, c in items]
    ch = max(c.shape[0] for c in cs) + 10
    cw = max(c.shape[1] for c in cs) + 10
    rows = (len(cs) + _GRID_COLS - 1) // _GRID_COLS
    canvas = np.full((rows * ch + 8, _GRID_COLS * cw + 8, 3), 245, np.uint8)
    for i, c in enumerate(cs):
        r, col = divmod(i, _GRID_COLS)
        canvas[r * ch + 8:r * ch + 8 + c.shape[0],
               col * cw + 6:col * cw + 6 + c.shape[1]] = c
    return _jpg(canvas)


def confirm_image(img_bytes: bytes, oid: str) -> bytes | None:
    """到场确认图：**整个左面板** + 编号徽标（绑号结构性成立）。

    v2 修正（confirm_ab.py 实证）：旧版裁中央 20%~94%，但 move_to_object
    不保证朝向——R27-52 的确认图里根本没有凉鞋（拍的是沙发+落地灯，
    VLM 对视野中央的随机物体分类说 pillow，当晚全部确认错案同根因）。
    物体出现在感知 objects 里 = 必在 120° 视野内的左面板某处，全幅
    裁剪保证目标在图中，由 VLM 按元数据特征寻找。
    """
    buf = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    pw = w // 2
    if pw < 10:
        return None
    crop = img[0:h, 4:pw - 4]
    cv2.rectangle(crop, (0, 0), (int(_BADGE_H * 1.9), _BADGE_H), (0, 0, 0), -1)
    cv2.putText(crop, oid, (5, _BADGE_H - 9), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 255), 2)
    return _jpg(crop)


# ------------------------------------------------------------------ #
# 输出解析
# ------------------------------------------------------------------ #

def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    import json
    s = m.group(0)
    # 中文 prompt 诱发两类格式伤（实机 train_v4b/v4c 抓现行）：
    # ① 弯引号 “ ”（json.loads 不认）②未加引号的裸值（X 等，随机出现）
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        s = re.sub(r':\s*([A-Za-z_]\w*)\s*(?=[,}\]])', r': "\1"', s)
        s = re.sub(r",\s*([}\]])", r"\1", s)   # 尾逗号保险
        try:
            d = json.loads(s)
        except json.JSONDecodeError:
            return {}
    return d if isinstance(d, dict) else {}


def parse_transcription(text: str) -> dict[int, str]:
    """{'1': 36, '2': '8'} → {1: '36'}（无效值丢弃）。"""
    out: dict[int, str] = {}
    for k, v in _extract_json(text).items():
        try:
            ki = int(str(k))
        except ValueError:
            continue
        s = str(v).strip()
        if s.isdigit() and len(s) <= _MAX_DIGITS:
            out[ki] = s
    return out


def parse_classification(text: str) -> dict[str, str]:
    """{'36': 'pillow'} → 归一类别（未知名→other）。"""
    out: dict[str, str] = {}
    for k, v in _extract_json(text).items():
        cat = str(v).strip().lower()
        out[str(k)] = cat if cat in _CATS else "other"
    return out


# ------------------------------------------------------------------ #
# 高层编排
# ------------------------------------------------------------------ #

def _img_part(b: bytes) -> dict:
    return {"type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,"
                          + base64.b64encode(b).decode()}}


def _txt(t: str) -> dict:
    return {"type": "text", "text": t}


def analyze_frame(img_bytes: bytes, wanted: dict[str, dict],
                  vlm_invoke, diag: dict | None = None) -> dict[str, dict]:
    """单帧读号识别。wanted: oid → 元数据 {color, shape, size}（可搬候选）。

    返回 {oid: {"cat": 类别, "area": 区域像素面积(bbox), "box": 右面板 bbox}}。
    链路任何一步失败返回空 dict（调用方回退旧整帧路径）。
    vlm_invoke(content_parts) -> str；抛出的异常由调用方兜。
    diag: 可选，调用方传入 dict，本函数回填各环节计数供日志诊断。
    """
    if diag is not None:
        diag.update(groups=0, reads=0, pairs=0, crops=0, cats=0)
    buf = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return {}
    h, w = img.shape[:2]
    pw = w // 2
    if pw < 10:
        return {}
    panel = img[:, pw:]
    groups = locate_stamp_groups(panel)
    if diag is not None:
        diag["groups"] = len(groups)
    if not groups:
        return {}
    montage = stamp_montage(panel, groups)
    if not montage:
        return {}
    reads = parse_transcription(vlm_invoke([_img_part(montage), _txt(PROMPT_TRANSCRIBE)]))
    if diag is not None:
        diag["reads"] = len(reads)
    if not reads:
        return {}
    # 绑号过滤：转写编号必须命中候选清单
    pairs: list[tuple[str, list[dict], tuple[int, int, int, int], int]] = []
    for k, g in enumerate(groups, 1):
        oid = reads.get(k)
        if oid is None or oid not in wanted:
            continue
        bx, by, bw, bh = _label_region(panel, g)
        if bw * bh < 64 or bw > pw * 0.6:  # 同 dashcam.annotate 的合理性过滤
            continue
        pairs.append((oid, g, (bx, by, bw, bh), bw * bh))
    if diag is not None:
        diag["pairs"] = len(pairs)
    if not pairs:
        return {}
    # 同号多戳取最大区域（§7-15：大件会拆多块印号）
    best: dict[str, tuple] = {}
    for oid, g, box, area in pairs:
        cur = best.get(oid)
        if cur is None or area > cur[3]:
            best[oid] = (oid, g, box, area)
    crops: dict[str, np.ndarray] = {}
    boxes: dict[str, tuple] = {}
    for oid, _, box, area in best.values():
        c = crop_badged(img, pw, box, oid)
        if c is not None:
            crops[oid] = c
            boxes[oid] = box + (area,)
    if not crops:
        return {}
    grid = build_grid(crops)
    meta = "\n".join(f"{oid}: {wanted[oid]['color']}/{wanted[oid]['shape']}"
                     f"/{wanted[oid]['size']}"
                     for oid in sorted(crops, key=int))
    parts = [_img_part(grid), _txt(PROMPT_CLASSIFY_TMPL.format(meta=meta))]
    cats = parse_classification(vlm_invoke(parts))
    if not cats:
        # 有裁剪就该有输出；空=格式伤（裸值/弯引号等随机出现），重试一次
        cats = parse_classification(vlm_invoke(parts))
    if diag is not None:
        diag["crops"] = len(crops)
        diag["cats"] = len(cats)
    out: dict[str, dict] = {}
    for oid in crops:
        cat = cats.get(oid)
        if cat is None:
            continue
        out[oid] = {"cat": cat, "area": boxes[oid][4],
                    "box": boxes[oid][:4]}
    return out
