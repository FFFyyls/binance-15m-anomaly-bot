import hashlib
import hmac
import logging
import os
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP

def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("bot")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

def utc_ms() -> int:
    return int(time.time() * 1000)

def sign_hmac_sha256(secret: str, query_string: str) -> str:
    return hmac.new(secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()

def round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    d = quantize_by_step(value, step, rounding=ROUND_DOWN)
    return float(d)

def round_step_up(value: float, step: float) -> float:
    if step <= 0:
        return value
    d = quantize_by_step(value, step, rounding=ROUND_UP)
    return float(d)

def round_tick(value: float, tick: float) -> float:
    if tick <= 0:
        return value
    return round((int(value / tick)) * tick, 12)

def quantize_by_step(value: float, step: float, rounding=ROUND_DOWN) -> Decimal:
    d_step = Decimal(str(step))
    if d_step <= 0:
        return Decimal(str(value))
    d_val = Decimal(str(value))
    factor = (d_val / d_step).to_integral_value(rounding=rounding)
    return factor * d_step

def format_by_step(value: float, step: float, rounding=ROUND_DOWN) -> str:
    d_step = Decimal(str(step))
    if d_step <= 0:
        return str(value)
    q = quantize_by_step(value, step, rounding=rounding)
    places = max(-d_step.normalize().as_tuple().exponent, 0)
    return f"{q:.{places}f}"

def get_proxy_url() -> str:
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        val = os.getenv(key, "").strip()
        if val:
            return val
    return ""
