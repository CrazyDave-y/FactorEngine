from __future__ import annotations

from dataclasses import dataclass
from typing import cast
import numpy as np


@dataclass(frozen=True, slots=True)
class Panel:
    fields: dict[str, np.ndarray]
    valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("panel must contain at least one field")
        shapes = {np.asarray(v).shape for v in self.fields.values()}
        if len(shapes) != 1:
            raise ValueError("all panel fields must share one shape")
        shape = next(iter(shapes))
        if len(shape) != 3:
            raise ValueError("panel fields must have shape [sessions, securities, time]")

        converted = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in self.fields.items()}
        object.__setattr__(self, "fields", converted)

        if self.valid is None:
            valid = np.ones(shape, dtype=bool)
        else:
            valid = np.asarray(self.valid, dtype=bool)
            if valid.shape != shape:
                raise ValueError("valid mask shape must match panel fields")
        object.__setattr__(self, "valid", valid)

    @property
    def shape(self) -> tuple[int, int, int]:
        return next(iter(self.fields.values())).shape

    def subset_sessions(self, sessions: slice) -> "Panel":
        # 构造完成后 valid 必为数组；None 只用于调用方省略资格的情况。
        valid = cast(np.ndarray, self.valid)
        return Panel({k: v[sessions] for k, v in self.fields.items()}, valid[sessions])


@dataclass(frozen=True, slots=True)
class IndexedBatch:
    """在有序数据源中的起始 lane 及本批输入。"""

    first_lane: int
    panel: Panel
    target: np.ndarray
