# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Subgraphs, declared at the call site with :func:`as_subgraph`.

A repeated block lowers to one shared body called once per layer, so the
compiler processes it once instead of once per repetition. Sharing is keyed on
the callable itself: one object called many times is staged once, and each
call's ``prefix`` is what resolves that one body's weights to the layer making
the call.

* core: one body, one ``mo.call`` per layer, each threading its own weights,
  with numerics matching the inlined reference;
* weights: the body names them relatively and declares them once, as
  placeholders the call prefix resolves;
* splitting: callables sharing a ``name`` are one body, and different names
  are different bodies;
* outputs: a nested (tuple/dict) return round-trips through the call;
* nesting: a subgraph called inside a body inlines, at any depth;
* modes: outside a capture the wrapper is transparent and just runs;
* distributed: a tensor-parallel block with sharded weights and an
  all-reduce;
* guardrails: ``allow_subgraphs=False`` opts out, and a body cannot read a
  value realized in its parent graph.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import (
    Module,
    ModuleList,
    as_subgraph,
    module_dataclass,
)
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    PlacementMapping,
    Replicated,
    Sharded,
    TensorLayout,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef

F32 = DType.float32
D = 4
H = 8


def default_type() -> TensorLayout:
    return TensorLayout(F32, ["batch", D], device=DeviceRef.CPU())


def zeros(*shape: int) -> Tensor:
    return Tensor.zeros(list(shape), dtype=F32, device=CPU())


def randn(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)


def count_graphs(mlir: str, name: str) -> int:
    return len(re.findall(rf"mo\.graph @{name}(?:_\d+)?\b", mlir))


def count_calls(mlir: str, name: str) -> int:
    return len(re.findall(rf"mo\.call @{name}(?:_\d+)?\b", mlir))


def externals(mlir: str) -> int:
    return mlir.count("mo.constant.external")


@module_dataclass
class Block(Module[[Tensor], Tensor]):
    """A residual MLP block."""

    w_in: Tensor  # [D, hidden]
    w_out: Tensor  # [hidden, D]

    def forward(self, x: Tensor) -> Tensor:
        return x + F.relu(x @ self.w_in) @ self.w_out


@module_dataclass
class Shared(Module[[Tensor], Tensor]):
    """Every layer through one body, each resolving its own weights."""

    layers: ModuleList

    def forward(self, x: Tensor) -> Tensor:
        # Each layer passed as itself and shared by name, so one body serves
        # them all and the prefix says whose weights each call gets. Passing
        # one stand-in layer would share too, but would then inline wrongly
        # under ``allow_subgraphs=False``.
        for layer in self.layers:
            x = as_subgraph(layer, name="Block")(x)
        return x


def _block() -> Block:
    return Block(zeros(D, H), zeros(H, D))


def _block_ref(
    x: np.ndarray, w_in: np.ndarray, w_out: np.ndarray
) -> np.ndarray:
    return x + np.maximum(x @ w_in, 0.0) @ w_out


def _stack_weights(
    rng: np.random.Generator, layers: int
) -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {}
    for i in range(layers):
        weights[f"layers.{i}.w_in"] = randn(rng, D, H)
        weights[f"layers.{i}.w_out"] = randn(rng, H, D)
    return weights


def _stack_ref(x: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    for i in range(len(weights) // 2):
        x = _block_ref(
            x, weights[f"layers.{i}.w_in"], weights[f"layers.{i}.w_out"]
        )
    return x


# ─── Core: one body, one call per layer ─────────────────────────────────────


def test_one_body_serves_every_layer() -> None:
    """The flagship case: declare the subgraph at the call site and stack the
    block in a plain loop. Three layers trace to one ``@Block`` body called
    three times, each threading its own weights via its prefix, and the result
    matches the inlined reference."""
    rng = np.random.default_rng(0)
    stack = Shared(layers=ModuleList([_block() for _ in range(3)]))
    weights = _stack_weights(rng, 3)

    mlir = str(stack.trace(default_type()))
    assert count_graphs(mlir, "Block") == 1
    assert count_calls(mlir, "Block") == 3

    compiled = stack.compile(default_type(), weights=weights)
    x = randn(rng, 2, D)
    np.testing.assert_allclose(
        compiled(Tensor(x, device=CPU())).to_numpy(),
        _stack_ref(x, weights),
        rtol=1e-4,
        atol=1e-4,
    )


def test_weights_are_declared_once_and_resolved_by_prefix() -> None:
    """The body names its weights relatively and declares them once, however
    deep the stack. Each call's prefix resolves those names to its own layer's
    weights at load time, so a per-layer full name is never emitted."""
    for n_layers in (3, 8):
        stack = Shared(layers=ModuleList([_block() for _ in range(n_layers)]))
        mlir = str(stack.trace(default_type()))
        assert count_calls(mlir, "Block") == n_layers
        # Two external constants (w_in, w_out) regardless of depth, not 2 x N.
        assert externals(mlir) == 2
        assert mlir.count("isPlaceholder = true") == 2
        assert 'name = "w_in"' in mlir and 'name = "w_out"' in mlir
        assert 'name = "layers.0.w_in"' not in mlir
        for i in range(n_layers):
            assert f'prefix = "layers.{i}."' in mlir


def test_distinct_callables_are_distinct_bodies() -> None:
    """Two names are two bodies: the dense-then-expert split a model makes by
    naming each group, with the layers inside a group sharing one body."""

    @module_dataclass
    class Split(Module[[Tensor], Tensor]):
        layers: ModuleList

        def forward(self, x: Tensor) -> Tensor:
            for i, layer in enumerate(self.layers):
                name = "dense" if i < 2 else "expert"
                x = as_subgraph(layer, name=name)(x)
            return x

    stack = Split(layers=ModuleList([_block() for _ in range(5)]))
    mlir = str(stack.trace(default_type()))
    assert (count_graphs(mlir, "dense"), count_calls(mlir, "dense")) == (1, 2)
    assert (count_graphs(mlir, "expert"), count_calls(mlir, "expert")) == (1, 3)


# ─── Outputs: a nested (tuple/dict) return round-trips through the call ──────


def test_structured_output_round_trips() -> None:
    """A nested return structure, here a tuple holding a dict, round-trips
    through the ``mo.call`` via the same pytree walk used for inputs: one shared
    body, one call per layer, and correct numerics."""

    @module_dataclass
    class NestBlock(Module[..., tuple[Tensor, dict[str, Tensor]]]):
        w: Tensor

        def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
            h = F.relu(x @ self.w)
            return h, {"residual": x + h}

    @module_dataclass
    class NestStack(Module[[Tensor], Tensor]):
        layers: ModuleList

        def forward(self, x: Tensor) -> Tensor:
            for layer in self.layers:
                a, d = as_subgraph(layer, name="NestBlock")(x)
                x = a + d["residual"]
            return x

    rng = np.random.default_rng(27)
    stack = NestStack(
        layers=ModuleList([NestBlock(zeros(D, D)) for _ in range(3)])
    )
    weights = {f"layers.{i}.w": randn(rng, D, D) for i in range(3)}

    mlir = str(stack.trace(default_type()))
    assert (
        count_graphs(mlir, "NestBlock"),
        count_calls(mlir, "NestBlock"),
    ) == (1, 3)

    compiled = stack.compile(default_type(), weights=weights)
    x = randn(rng, 2, D)
    expected = x
    for i in range(3):
        h = np.maximum(expected @ weights[f"layers.{i}.w"], 0.0)
        expected = h + (expected + h)
    np.testing.assert_allclose(
        compiled(Tensor(x, device=CPU())).to_numpy(),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


# ─── Nesting: a subgraph inside a body inlines (any depth) ──────────────────


def test_nested_subgraphs_inline() -> None:
    """Only the outermost call becomes a subgraph; a call nested inside a body
    inlines at every depth (here three levels, C -> B -> A). The graph compiler
    does not nest subgraphs."""

    @module_dataclass
    class A(Module[[Tensor], Tensor]):
        w: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return F.relu(x @ self.w)

    @module_dataclass
    class B(Module[[Tensor], Tensor]):
        a: A
        w: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + as_subgraph(self.a, name="A")(x) @ self.w

    @module_dataclass
    class C(Module[[Tensor], Tensor]):
        b: B
        w: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + as_subgraph(self.b, name="B")(x) @ self.w

    @module_dataclass
    class Outer(Module[[Tensor], Tensor]):
        layers: ModuleList

        def forward(self, x: Tensor) -> Tensor:
            for layer in self.layers:
                x = as_subgraph(layer, name="C")(x)
            return x

    stack = Outer(
        layers=ModuleList(
            [C(B(A(zeros(D, D)), zeros(D, D)), zeros(D, D)) for _ in range(2)]
        )
    )
    mlir = str(stack.trace(default_type()))
    assert (count_graphs(mlir, "C"), count_calls(mlir, "C")) == (1, 2)
    # The two inner levels inline rather than nesting their own subgraphs.
    assert "mo.graph @B" not in mlir and "mo.graph @A" not in mlir


# ─── Tensor parallelism: sharded weights + an all-reduce inside the body ────

# A two-way tensor-parallel mesh, simulated on CPU: the trace shape and numerics
# match a real two-GPU mesh, minus the hardware collectives.
MESH = DeviceMesh(devices=(CPU(), CPU()), mesh_shape=(2,), axis_names=("tp",))
REPLICATED = PlacementMapping(MESH, (Replicated(),))
COLUMN = PlacementMapping(MESH, (Sharded(1),))
ROW = PlacementMapping(MESH, (Sharded(0),))


@module_dataclass
class TPBlock(Module[[Tensor], Tensor]):
    """Column-parallel up-projection, row-parallel down-projection, all-reduced
    back to a replicated residual."""

    w_in: Tensor  # [D, H], column-parallel
    w_out: Tensor  # [H, D], row-parallel

    def forward(self, x: Tensor) -> Tensor:  # x replicated [batch, D]
        hidden = F.relu(x @ self.w_in)  # sharded on H
        # Row-parallel matmul leaves partial sums; transfer to replicated
        # performs the all-reduce.
        return x + F.transfer_to(hidden @ self.w_out, REPLICATED)


def _tp_block() -> TPBlock:
    return TPBlock(
        w_in=F.transfer_to(zeros(D, H), COLUMN),
        w_out=F.transfer_to(zeros(H, D), ROW),
    )


def test_tensor_parallel_block_shares_one_subgraph() -> None:
    """A real distributed block: sharded weights thread in per device and an
    all-reduce runs inside the body. Two layers share one ``@TPBlock``, each
    weight registers one external constant per shard, and numerics hold."""
    rng = np.random.default_rng(1)
    input_type = TensorLayout(
        F32, ["batch", D], DeviceMapping(MESH, (Replicated(),))
    )
    stack = Shared(layers=ModuleList([_tp_block() for _ in range(2)]))
    weights = _stack_weights(rng, 2)

    mlir = str(stack.trace(input_type))
    assert count_graphs(mlir, "Block") == 1
    assert count_calls(mlir, "Block") == 2
    # The shared body registers each sharded weight once under its *relative*
    # name (one external constant per device); each layer's ``mo.call`` carries
    # the per-layer prefix that resolves those names to its own weights.
    for weight in ("w_in", "w_out"):
        for shard in range(2):
            assert f'name = "{weight}._shard.{shard}"' in mlir
    for i in range(2):
        assert f'prefix = "layers.{i}."' in mlir

    compiled = stack.compile(input_type, weights=weights)
    x = randn(rng, 2, D)
    result = compiled(F.transfer_to(Tensor(x, device=CPU()), REPLICATED))
    assert result.placements == (Replicated(),)
    np.testing.assert_allclose(
        result.to_numpy(), _stack_ref(x, weights), rtol=1e-4, atol=1e-4
    )


# ─── Modes: outside a capture the wrapper is transparent ────────────────────


def test_calling_outside_a_capture_runs_eager() -> None:
    """Wrapping has no effect outside a capture: the call just runs ``forward``
    (no subgraph, no error) and matches the bare-forward result."""
    rng = np.random.default_rng(31)
    w_in, w_out = randn(rng, D, H), randn(rng, H, D)
    block = Block(Tensor(w_in, device=CPU()), Tensor(w_out, device=CPU()))
    x = randn(rng, 2, D)
    out = as_subgraph(block, name="Block")(Tensor(x, device=CPU())).to_numpy()
    np.testing.assert_allclose(
        out, _block_ref(x, w_in, w_out), rtol=1e-4, atol=1e-4
    )


# ─── Guardrails ─────────────────────────────────────────────────────────────


def test_allow_subgraphs_false_inlines_everything() -> None:
    """``allow_subgraphs=False`` inlines every call site instead of emitting
    shared bodies, so the model traces into one flat graph (no
    ``mo.graph``/``mo.call``) and the numerics are unchanged. This is the
    Module-level equivalent of the pipelines ``use_subgraphs`` flag."""
    rng = np.random.default_rng(30)
    stack = Shared(layers=ModuleList([_block() for _ in range(3)]))
    weights = _stack_weights(rng, 3)

    mlir = str(stack.trace(default_type(), allow_subgraphs=False))
    assert count_graphs(mlir, "Block") == 0
    assert count_calls(mlir, "Block") == 0

    compiled = stack.compile(
        default_type(), weights=weights, allow_subgraphs=False
    )
    x = randn(rng, 2, D)
    np.testing.assert_allclose(
        compiled(Tensor(x, device=CPU())).to_numpy(),
        _stack_ref(x, weights),
        rtol=1e-4,
        atol=1e-4,
    )


def test_subgraph_cannot_read_parent_graph_value() -> None:
    """KNOWN LIMITATION: a body may only read its own operands (its forward
    arguments and its parameters) plus eager constants it can re-materialize. A
    value *realized in the enclosing (parent) graph* lives in a different
    region and is out of scope, so reading one across the boundary fails.

    The fix is to thread that value in as a forward argument. This is exactly
    why gemma3 attention takes the rope ``freqs_cis`` table as an operand
    instead of reading the rope's cached tensor across the boundary -- the
    capture below is the reduced form of that bug.
    """

    @module_dataclass
    class CapturingStack(Module[[Tensor], Tensor]):
        w: Tensor

        def forward(self, x: Tensor) -> Tensor:
            # Realized in THIS (parent) graph; the body below closes over it
            # rather than taking it as an argument, so it crosses the boundary.
            parent_value = F.relu(x)

            @module_dataclass
            class Reader(Module[[Tensor], Tensor]):
                w: Tensor

                def forward(self, h: Tensor) -> Tensor:
                    # self.w threads in as an operand (fine); parent_value is
                    # captured from the parent graph (the unsupported case).
                    return h @ self.w + parent_value

            block = Reader(self.w)
            for _ in range(2):
                x = as_subgraph(block, name="Reader")(x)
            return x

    stack = CapturingStack(zeros(D, D))
    with pytest.raises(TypeError, match="Can't realize from a graph context"):
        stack.compile(default_type())
