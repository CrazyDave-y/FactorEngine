import numpy as np
from factor_engine.data import Panel


class SyntheticLaneSource:
    """用哈希计数器按需生成 lane，读取顺序和分批大小不影响结果。

    这是工程测试数据，不模拟真实收益分布。
    """

    def __init__(self, sessions, securities, time, seed=0x135288F):
        if min(sessions, securities, time) <= 0:
            raise ValueError("source dimensions must be positive")
        self.sessions, self.securities, self.time = int(sessions), int(securities), int(time)
        self.seed = int(seed)
        self.lanes = self.sessions * self.securities
        if self.lanes * self.time >= 2**32:
            raise ValueError("source counter supports fewer than 2**32 cells")

    def _uniform(self, first, lanes, field):
        v = np.arange(first * self.time, (first + lanes) * self.time, dtype=np.uint32)
        v += np.uint32((self.seed + field * 0x9E3779B9) % 2**32)
        v ^= v >> np.uint32(16)
        v *= np.uint32(0x7FEB352D)
        v ^= v >> np.uint32(15)
        v *= np.uint32(0x846CA68B)
        v ^= v >> np.uint32(16)
        return (
            (v >> np.uint32(8)).astype(np.float32) / np.float32(2**24) - np.float32(0.5)
        ).reshape(1, lanes, self.time)

    def tile(self, first, lanes):
        ret = self._uniform(first, lanes, 1) * np.float32(0.01)
        volume = np.float32(1000) + self._uniform(first, lanes, 2) * np.float32(400)
        change = np.full_like(volume, np.nan)
        change[..., 1:] = np.log(volume[..., 1:] / volume[..., :-1])
        close = np.float32(100) + np.cumsum(ret, axis=-1, dtype=np.float32)
        target = (
            np.float32(0.35) * ret
            - np.float32(0.1) * change
            + np.float32(0.005) * self._uniform(first, lanes, 3)
        )
        target[..., -1] = np.nan
        return Panel(
            {"close": close, "volume": volume, "return_1": ret, "volume_change": change}
        ), target

    def batches(self, lanes):
        if lanes <= 0:
            raise ValueError("batch lanes must be positive")
        for first in range(0, self.lanes, lanes):
            yield self.tile(first, min(lanes, self.lanes - first))
