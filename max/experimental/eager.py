# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Eager Tensor for MAX."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional
import weakref

import numpy as np

from max import driver as md, engine as me, graph as mg, mlir


_CONTEXT = mg.graph._new_context()

_NamedAttribute = tuple[str, mlir.Attribute]
_Attributes = tuple[_NamedAttribute, ...]
_Operands = tuple[mlir.Value, ...]
_OpCache = dict[tuple[str, _Operands, _Attributes], mlir.OpView]


@dataclass
class Graph:
    sources: set[Tensor]
    leaves: weakref.WeakSet[Tensor]
    graph: mg.Graph

    async def compute(self):
        leaves = list(self.leaves)
        with self.graph as graph:
            graph.output(*(leaf._value for leaf in leaves))

        session = me.InferenceSession()
        model = session.load(graph)
        assert all(source._storage for source in self.sources)
        results = model(*(source._storage for source in self.sources))  # type: ignore

        for leaf, result in zip(leaves, results):
            assert isinstance(result, md.Tensor)
            leaf.__init__(result)  # type: ignore

    def merge(self, other: Graph):
        # So this is more complicated than it looks for even simple cases.
        # Say I write
        # >>> a, b, c, d = ...
        # >>> result = (a + b) + (c + d)
        # >>> print(await result)
        # then (a + b) and (c + d) each form a separate "graph". In order to
        # get eager errors from shape propagation, we need to be making graph
        # operations with the actual input shapes. However, the final result
        # is actually a graph which depends on the inputs from each of the
        # graphs for (a + b) and (c + d). If we didn't merge them in some way
        # we'd never get any benefit from fusion, we _want_ them to be a single
        # graph for execution. This means we need to reason about how to compose
        # the graphs.
        sources = self.sources | other.sources
        leaves = self.leaves | other.leaves
        arg_ids = dict((source, i) for i, source in enumerate(sources))
        value_map: dict[mlir.Value, mlir.Value] = {}
        op_cache: _OpCache = {}

        def clone_into(graph: mg.Graph, source: Graph):
            for i, arg in enumerate(source.sources):
                value_map[
                    source.graph._body.arguments[i]
                ] = graph._body.arguments[arg_ids[arg]]

            insertion_point = mlir.InsertionPoint(graph._body)

            # TODO: the first mo.chain.create() is implicit
            #     which means it's currently not put into the op cache
            # TODO: add __hash__ and __eq__ to PyNamedAttribute

            for op in source.graph._body:
                if op.name == "output":
                    continue

                operands = [value_map[operand] for operand in op.operands]  # type: ignore

                # The op cache allows us to recognize and deduplicate
                # shared subgraphs. This relies on the assumption that they
                # share _identical_ input nodes (which is the common case here).
                attrs_key = ((na.name, na.attr) for na in op.attributes)  # type: ignore
                key = (op.name, tuple(operands), tuple(attrs_key))
                if not (new_op := op_cache.get(key)):
                    op_cache[key] = new_op = op.clone(insertion_point)
                    assert new_op
                    for i, operand in enumerate(operands):
                        new_op.operands[i] = operand

                for new_result, result in zip(new_op.results, op.results):  # type: ignore
                    value_map[result] = new_result

        with mg.Graph(
            "_",
            input_types=[source._value.type for source in sources],
            context=_CONTEXT,
        ) as graph:
            clone_into(graph, self)
            clone_into(graph, other)

        new_graph = Graph(sources, leaves, graph)

        for node, value in zip(sources, graph.inputs):
            node._graph = new_graph
            node._value = value.tensor

        for node in leaves:
            node._graph = new_graph
            node._value = mg.Value(value_map[node._value._mlir_value]).tensor

        return new_graph


class Tensor:
    _graph: Graph
    _value: mg.TensorValue
    _storage: Optional[md.Tensor] = None

    def __init__(
        self,
        data: Optional[md.Tensor] = None,
        graph: Optional[Graph] = None,
        value: Optional[mg.TensorValue] = None,
    ):
        if data is None:
            assert graph
            assert value
            self._graph = graph
            self._value = value
        else:
            self._storage = data
            max_graph = mg.Graph(
                "_",
                input_types=[mg.TensorType(data.dtype, data.shape)],
                context=_CONTEXT,
            )
            self._value = max_graph.inputs[0].tensor
            self._graph = Graph({self}, weakref.WeakSet(), max_graph)

    @property
    def shape(self):
        return self._value.shape

    @property
    def dtype(self):
        return self._value.dtype

    @classmethod
    def from_numpy(cls, array: np.ndarray):
        return cls(md.Tensor.from_numpy(array))

    def to_numpy(self):
        if not self._storage:
            raise ValueError(
                "to_numpy: Must `await` this partial result first."
            )
        return self._storage.to_numpy()

    def __repr__(self):
        if not self._storage:
            return f"Tensor(<deferred>, dtype={self.dtype}, shape={self.shape})"
        return repr(self.to_numpy())

    def __await__(self):
        if not self._storage:
            yield from asyncio.create_task(self._graph.compute())
        assert self._storage
        return self

    def _binop(self, other: Tensor, op):
        # This mutates all relevant nodes to be part of the new graph.
        merged = self._graph.merge(other._graph)
        with merged.graph:
            new_node = Tensor(graph=merged, value=op(self._value, other._value))
        merged.leaves.add(new_node)
        return new_node

    def __add__(self, other: Tensor):
        return self._binop(other, mg.TensorValue.__add__)
