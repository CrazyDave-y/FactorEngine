from __future__ import annotations

import numpy as np

from .data import Panel


def make_synthetic_panel(
    sessions: int = 10,
    securities: int = 12,
    time: int = 256,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0, 0.003, size=(sessions, securities, time)).astype(np.float32)
    close = np.exp(np.cumsum(ret.astype(np.float64), axis=-1)).astype(np.float32)
    volume = rng.lognormal(mean=8.0, sigma=0.45, size=ret.shape).astype(np.float32)
    volume_change = np.full_like(volume, np.nan)
    volume_change[..., 1:] = np.log(volume[..., 1:] / volume[..., :-1])
    target = np.full_like(ret, np.nan)
    noise = rng.normal(0.0, 0.0025, size=ret.shape).astype(np.float32)
    # 演示标签用当前特征加噪声构造，末尾留空。
    target[..., :-1] = 0.35 * ret[..., :-1] - 0.10 * volume_change[..., :-1] + noise[..., :-1]
    panel = Panel(
        {
            "close": close,
            "volume": volume,
            "return_1": ret,
            "volume_change": volume_change,
        }
    )
    return panel, target
