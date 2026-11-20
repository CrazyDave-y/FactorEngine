from typing import TYPE_CHECKING

from .dsl import OPERATORS, OperatorSpec, ValueType
from .expression import *

if TYPE_CHECKING:
    from .engine import FactorEngine, SearchConfig, SearchResult

__all__ = ["OPERATORS", "OperatorSpec", "ValueType", "FactorEngine", "SearchConfig", "SearchResult"]


def __getattr__(name: str):
    # 后端和数学模块导入包时不加载搜索层；公开类仍是 engine 中的原类。
    if name in {"FactorEngine", "SearchConfig", "SearchResult"}:
        from . import engine

        value = getattr(engine, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
