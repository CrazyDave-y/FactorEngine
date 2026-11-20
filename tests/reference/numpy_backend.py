from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

from factor_engine.data import Panel
from factor_engine.ir import ExecutionPlan, Node, constant_param, terminal_param, window_param
from .numpy_ops import evaluate_op


PlanValue: TypeAlias = NDArray[np.float32] | np.float32


class NumpyPlanBackend:
    name = "dag_numpy"

    def score(self, plan, source):
        """参考后端也返回有序 Sum/Count；不参与生产后端选择。"""
        from tests.reference.metrics import rank_ic
        from factor_engine.metrics import LaneAccumulator

        accumulator = LaneAccumulator(len(plan.root_ids), source.lane_count)
        for batch in source.iter_batches(0, 32):
            values = self.evaluate(plan, batch.panel)
            target = batch.target.reshape(-1, source.time)
            lane_ic = np.asarray(
                [
                    [
                        rank_ic(factor, label)
                        for factor, label in zip(
                            value.reshape(-1, source.time), target, strict=True
                        )
                    ]
                    for value in values
                ]
            )
            accumulator.commit(batch.first_lane, lane_ic)
        return accumulator.finish()

    def evaluate(self, plan: ExecutionPlan, panel: Panel) -> list[np.ndarray]:
        values: list[PlanValue] = []
        for node in plan.nodes:
            values.append(self._node(node, values, panel))
        roots: list[np.ndarray] = []
        for root in plan.root_ids:
            value = values[root]
            if np.asarray(value).ndim == 0:
                value = np.full(panel.shape, np.float32(value), dtype=np.float32)
            roots.append(np.asarray(value, dtype=np.float32))
        return roots

    def _node(self, node: Node, values: list[PlanValue], panel: Panel) -> PlanValue:
        if node.op == "terminal":
            try:
                x = panel.fields[terminal_param(node)]
            except KeyError as exc:
                raise KeyError(f"panel has no terminal {node.param!r}") from exc
            return np.where(panel.valid & np.isfinite(x), x, np.nan).astype(np.float32, copy=False)
        if node.op == "const":
            return np.float32(constant_param(node))
        args = [values[index] for index in node.inputs]
        return evaluate_op(node.op, args, window_param(node) if node.param is not None else None)
