"""Proof-of-work token generation for ChatGPT API."""
from __future__ import annotations

import base64
import hashlib
import json
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple

CORES = [8, 16, 24, 32]
TIME_LAYOUT = "%a %b %d %Y %H:%M:%S"

# Cached values
_cached_scripts: List[str] = []
_cached_dpl: str = ""
_cached_time: int = 0

NAVIGATOR_KEYS = [
    "registerProtocolHandler−function registerProtocolHandler() { [native code] }",
    "storage−[object StorageManager]",
    "locks−[object LockManager]",
    "appCodeName−Mozilla",
    "permissions−[object Permissions]",
    "webdriver−false",
    "vendor−Google Inc.",
    "cookieEnabled−true",
    "product−Gecko",
    "hardwareConcurrency−32",
    "onLine−true",
    "pdfViewerEnabled−true",
    "language−en-US",
]

DOCUMENT_KEYS = ["_reactListeningo743lnnpvdg", "location"]

WINDOW_KEYS = [
    "0", "window", "self", "document", "name", "location",
    "customElements", "history", "navigation", "navigator",
    "origin", "screen", "innerWidth", "innerHeight",
    "performance", "crypto", "localStorage", "sessionStorage",
    "fetch", "setTimeout", "setInterval",
]


def get_parse_time() -> str:
    """Get formatted time string in EST."""
    now = datetime.now(timezone(timedelta(hours=-5)))
    return now.strftime(TIME_LAYOUT) + " GMT-0500 (Eastern Standard Time)"


def get_config(user_agent: str, req_token: Optional[str] = None) -> List[Any]:
    """Generate browser config for proof-of-work."""
    config = [
        random.choice([1920 + 1080, 2560 + 1440, 1920 + 1200, 2560 + 1600]),
        get_parse_time(),
        4294705152,
        0,
        user_agent,
        random.choice(_cached_scripts) if _cached_scripts else "https://chatgpt.com/backend-api/sentinel/sdk.js",
        _cached_dpl if _cached_dpl else "",
        "en-US",
        "en-US,en",
        0,
        random.choice(NAVIGATOR_KEYS),
        random.choice(DOCUMENT_KEYS),
        random.choice(WINDOW_KEYS),
        time.perf_counter() * 1000,
        str(uuid.uuid4()),
        "",
        random.choice(CORES),
        time.time() * 1000 - (time.perf_counter() * 1000),
    ]
    return config


def generate_answer(seed: str, diff: str, config: List[Any]) -> Tuple[str, bool]:
    """Generate proof-of-work answer by finding hash with required difficulty."""
    # Ensure diff is a valid hex string
    diff = str(diff).strip()
    if not diff:
        diff = "0fffff"  # Default easy difficulty

    # Pad to even length if needed
    if len(diff) % 2 != 0:
        diff = "0" + diff

    diff_len = len(diff) // 2  # hex string to bytes
    seed_encoded = seed.encode()

    static_config_part1 = (json.dumps(config[:3], separators=(',', ':'), ensure_ascii=False)[:-1] + ',').encode()
    static_config_part2 = (',' + json.dumps(config[4:9], separators=(',', ':'), ensure_ascii=False)[1:-1] + ',').encode()
    static_config_part3 = (',' + json.dumps(config[10:], separators=(',', ':'), ensure_ascii=False)[1:]).encode()

    target_diff = bytes.fromhex(diff)

    for i in range(500000):
        dynamic_json_i = str(i).encode()
        dynamic_json_j = str(i >> 1).encode()
        final_json_bytes = static_config_part1 + dynamic_json_i + static_config_part2 + dynamic_json_j + static_config_part3
        base_encode = base64.b64encode(final_json_bytes)
        hash_value = hashlib.sha3_512(seed_encoded + base_encode).digest()
        if hash_value[:diff_len] <= target_diff:
            return base_encode.decode(), True

    # Fallback if not solved
    return "wQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D" + base64.b64encode(f'"{seed}"'.encode()).decode(), False


def get_answer_token(seed: str, diff: str, config: List[Any]) -> Tuple[str, bool]:
    """Get proof-of-work answer token."""
    answer, solved = generate_answer(seed, diff, config)
    return "gAAAAAB" + answer, solved


def get_requirements_token(config: List[Any]) -> str:
    """Generate requirements token for chat-requirements endpoint."""
    require, _ = generate_answer(str(random.random()), "0fffff", config)
    return "gAAAAAC" + require
