# -*- coding: utf-8 -*-
"""自研直连 VLM 客户端（思考关）：tidyroom v4 感知调用专用。

为什么不用官方 client（v2-notes §7-21）：官方 openai 客户端 `_invoke`
只传 model+messages，deepseek-flash 默认开思考（effort=high）——大
prompt 42~68s 且 8192 token 全被 reasoning 吃掉正文为空（生产
"空文本/解析失败/换帧重发"的根源）。直连 + thinking disabled 后实测
拼格 0.8~1.0s、单件 0.5~0.8s。

密钥只走环境变量（与官方同一套通用覆盖，不新增配置）：
  VLM_CLIENT_CFG_API_KEY / VLM_CLIENT_CFG_API_BASE / VLM_CLIENT_CFG_NAME
本地调试可用 DEEPSEEK_API_KEY。缺 key 时抛 DirectVLMUnavailable，
调用方（tidyroom_agent）回退官方 client。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class DirectVLMUnavailable(Exception):
    """环境变量未配置，无法直连。"""


class DirectVLMClient:
    """无状态线程安全；invoke 返回正文文本，失败抛异常。"""

    def __init__(self, timeout: float = 60.0, retries: int = 1):
        self.api_key = (os.environ.get("VLM_CLIENT_CFG_API_KEY")
                        or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        base = (os.environ.get("VLM_CLIENT_CFG_API_BASE") or "").strip()
        self.api_base = base or "https://api.deepseek.com/v1"
        self.model = (os.environ.get("VLM_CLIENT_CFG_NAME")
                      or "deepseek-flash").strip()
        self.timeout = timeout
        self.retries = retries
        if not self.api_key:
            raise DirectVLMUnavailable(
                "缺 VLM_CLIENT_CFG_API_KEY / DEEPSEEK_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def invoke(self, content_parts: list[dict], max_tokens: int = 1024,
               timeout: float | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},  # flash 默认 effort=high，必须关
            "temperature": 0,                  # 转写/分类跨运行稳定（回放实测方差大）
        }
        req = urllib.request.Request(
            self.api_base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key})
        last: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                with urllib.request.urlopen(
                        req, timeout=timeout or self.timeout) as r:
                    data = json.loads(r.read())
                return data["choices"][0]["message"].get("content", "") or ""
            except urllib.error.HTTPError as exc:
                # 4xx 不重试（key/参数问题），5xx/超时可重试
                body = exc.read()[:200] if exc.fp else b""
                if 400 <= exc.code < 500:
                    raise RuntimeError(
                        f"VLM HTTP {exc.code}: {body!r}") from exc
                last = exc
            except Exception as exc:  # noqa: BLE001 网络抖动/超时
                last = exc
            time.sleep(1.0)
        raise RuntimeError(f"VLM 直连失败: {type(last).__name__}: {last}")
