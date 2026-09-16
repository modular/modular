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
"""Tests for max.experimental.compilation.

Three transforms share one boundary: :func:`stage` records a graph,
:func:`compile` records and runs one, and :func:`as_subgraph` records a body
that a graph calls many times. Each takes the callable, then the callable's own
arguments as specs, so every test spells the call twice, once for the
transform and once for the run, and the two must agree.

Each case is the most demanding one on its axis, because the simpler ones
cannot fail independently of it: a signature holding every parameter kind
subsumes the single-tensor case, and the model in :class:`TestModel` subsumes
each part it composes. What is tested separately is what a passing call cannot
show: a rejection, a retrace, or a body's identity.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, overload

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.compilation import (
    CompiledCallable,
    as_layout,
    as_subgraph,
    compile,
    stage,
)
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.sharding import (
    BufferLayout,
    DeviceMapping,
    DeviceMesh,
    PlacementMapping,
    Replicated,
    Sharded,
    TensorLayout,
)
from max.experimental.tensor import Tensor
from max.graph import BufferType, BufferValue, DeviceRef, TensorType
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import MHAKVCacheParams
from max.tree import flatten as tree_flatten
from max.tree import paths as tree_paths

_F32 = DType.float32


def _spec(*shape: int) -> TensorLayout:
    """One replicated slot on one device, every extent fixed."""
    mesh = DeviceMesh.single(CPU())
    return TensorLayout(
        _F32, list(shape), PlacementMapping(mesh, (Replicated(),))
    )


def _symbolic(*shape: int | str) -> TensorLayout:
    """One slot on one device, keeping any named extent symbolic."""
    return TensorLayout(_F32, list(shape), device=DeviceRef.CPU())


def _tensor(*values: float) -> Tensor:
    return Tensor.from_dlpack(np.array(values, dtype=np.float32))


def _ones(*shape: int) -> Tensor:
    return Tensor.from_dlpack(np.ones(shape, dtype=np.float32))


def _mesh2() -> DeviceMesh:
    return DeviceMesh(
        devices=(CPU(), CPU()), mesh_shape=(2,), axis_names=("x",)
    )


def _replicated_spec(*shape: int) -> TensorLayout:
    return TensorLayout(
        _F32, list(shape), PlacementMapping(_mesh2(), (Replicated(),))
    )


def _replicated(tensor: Tensor) -> Tensor:
    return Tensor._from_shards(
        (tensor.driver_tensor,) * 2, _mesh2(), (Replicated(),)
    )


def _sharded_spec(*shape: int) -> TensorLayout:
    return TensorLayout(
        _F32, list(shape), PlacementMapping(_mesh2(), (Sharded(0),))
    )


def _sharded(*per_device: Tensor) -> Tensor:
    return Tensor._from_shards(
        tuple(t.driver_tensor for t in per_device), _mesh2(), (Sharded(0),)
    )


def _calls(graph: object, name: str) -> int:
    """How many times a subgraph symbol is called from a graph body."""
    return len(re.findall(rf"mo\.call @{name}\b", str(graph)))


def _bodies(staged: object, name: str) -> int:
    """How many definitions of a symbol exist in the module, not the body."""
    return len(re.findall(rf"mo\.graph @{name}\b", str(staged)))


# ═════════════════════════════════════════════════════════════════════════
#  The boundary: every parameter kind Python has, at once
# ═════════════════════════════════════════════════════════════════════════


def _mixed_signature(x, /, pair, scale=2.0, *rest, bias, flag=True, **extras):  # noqa: ANN001, ANN202
    """Every parameter kind, with tensors and statics interleaved.

    Positional-only, positional-or-keyword holding a container, a defaulted
    positional that stays static, var-positional tensors, a required
    keyword-only tensor, a defaulted keyword-only static, and var-keyword
    tensors whose iteration order is observable.
    """
    out = x * scale + pair["lo"] - pair["hi"]
    for tensor in rest:
        out = out + tensor
    out = out + bias if flag else out - bias
    for name in sorted(extras):
        out = out * extras[name] + 1.0
    return {"out": out, "arity": len(rest), "tags": tuple(sorted(extras))}


def _mixed_specs(spec):  # noqa: ANN001, ANN202
    return (
        (spec(2), {"lo": spec(2), "hi": spec(1)}, 3.0, spec(2), spec(2)),
        {"bias": spec(2), "flag": False, "alpha": spec(2), "beta": spec(2)},
    )


def _mixed_args(make):  # noqa: ANN001, ANN202
    return (
        (
            make(1.0, 2.0),
            {"lo": make(10.0, 10.0), "hi": make(1.0)},
            3.0,
            make(100.0, 100.0),
            make(1000.0, 1000.0),
        ),
        {
            "bias": make(2.0, 2.0),
            "flag": False,
            "alpha": make(2.0, 2.0),
            "beta": make(0.5, 0.5),
        },
    )


_MIXED_RESULT = [1111.5, 1114.5]

# Flatten order is not call order: `pair`'s dict sorts "hi" before "lo", and the
# keyword block sorts alphabetically, so `bias` trails the var-keyword tensors.
_MIXED_ROUTES = [
    "0.0",
    "0.1.hi",
    "0.1.lo",
    "0.3",
    "0.4",
    "1.alpha",
    "1.beta",
    "1.bias",
]


class TestBoundary:
    """One graph input per tensor slot, wherever the slot sits."""

    def test_every_tensor_slot_is_an_input_addressed_by_its_route(self) -> None:
        args, kwargs = _mixed_specs(_spec)
        staged = stage(_mixed_signature)(*args, **kwargs)
        assert len(staged.graph.inputs) == len(_MIXED_ROUTES)
        assert list(staged.in_tree.leaf_paths) == _MIXED_ROUTES

    def test_it_round_trips_compiled(self) -> None:
        spec_args, spec_kwargs = _mixed_specs(_spec)
        run = compile(_mixed_signature)(*spec_args, **spec_kwargs)
        args, kwargs = _mixed_args(_tensor)
        out = run(*args, **kwargs)
        np.testing.assert_allclose(out["out"].to_numpy(), _MIXED_RESULT)
        assert out["arity"] == 2, "the var-positional tensors are inputs"
        assert out["tags"] == ("alpha", "beta"), "and so are the var-keyword"

    def test_a_default_that_is_never_passed_traces_as_a_static(self) -> None:
        spec_args, spec_kwargs = _mixed_specs(_spec)
        run = compile(_mixed_signature)(*spec_args[:2], **spec_kwargs)
        args, kwargs = _mixed_args(_tensor)
        out = run(*args[:2], **kwargs)
        # scale defaults to 2.0, and `rest` is empty.
        np.testing.assert_allclose(out["out"].to_numpy(), [10.5, 12.5])
        assert out["arity"] == 0

    def test_a_real_tensor_cannot_declare_the_boundary(self) -> None:
        """Only a layout registers a graph.

        A tensor's dims are whatever it currently holds, so taking a boundary
        from one fixes every dimension to that, the call-time inference a
        declared boundary exists to avoid. It is refused rather than walked as
        a pytree, which is what dropping it from the leaf set would do.
        """
        args, kwargs = _mixed_args(_tensor)
        with pytest.raises(TypeError, match="is a value, not a layout"):
            compile(_mixed_signature)(*args, **kwargs)

    def test_a_tensor_layout_declares_what_the_tensor_holds(self) -> None:
        """``tensor.layout`` is how to ask for those dims on purpose."""
        spec_args, spec_kwargs = _mixed_args(
            lambda *values: _tensor(*values).layout
        )
        run = compile(_mixed_signature)(*spec_args, **spec_kwargs)
        args, kwargs = _mixed_args(_tensor)
        np.testing.assert_allclose(
            run(*args, **kwargs)["out"].to_numpy(), _MIXED_RESULT
        )

    def test_a_symbolic_extent_accepts_any_size(self) -> None:
        run = compile(lambda x: x * 2)(_symbolic("n"))
        np.testing.assert_allclose(run(_tensor(1.0)).to_numpy(), [2.0])
        np.testing.assert_allclose(
            run(_tensor(1.0, 2.0, 3.0)).to_numpy(), [2.0, 4.0, 6.0]
        )

    def test_a_spec_that_describes_nothing_is_rejected(self) -> None:
        # ``TensorLayout`` rules this out statically; the guard answers untyped callers.
        not_a_spec: Any = object()
        with pytest.raises(TypeError, match="expected a TensorLayout"):
            as_layout(not_a_spec)


# ═════════════════════════════════════════════════════════════════════════
#  Rejections: what a passing call cannot show
# ═════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Projections:
    """A record of tensors, as every layer in ``nn/functional`` holds.

    A boundary is declared with layouts and called with tensors, so a record
    that crosses one holds either. The production KV cache says the same thing
    with a type parameter per leaf kind.
    """

    a: Tensor | TensorLayout
    b: Tensor | TensorLayout
    bias: Tensor | TensorLayout | None = None

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return {f.name: getattr(self, f.name) for f in fields(self)}, None

    @classmethod
    def __tree_unflatten__(
        cls, meta: None, children: Mapping[str, Any]
    ) -> Projections:
        del meta
        return cls(**children)


class TestRejections:
    def test_a_record_round_trips_and_a_bad_field_names_its_route(self) -> None:
        run = compile(lambda p: p.a * p.b)(Projections(a=_spec(2), b=_spec(2)))
        out = run(Projections(a=_tensor(2.0, 3.0), b=_tensor(4.0, 5.0)))
        np.testing.assert_allclose(out.to_numpy(), [8.0, 15.0])

        with pytest.raises(ValueError, match=r"0\.a"):
            run(Projections(a=_tensor(1.0), b=_tensor(4.0, 5.0)))

    def test_a_static_argument_is_part_of_the_signature(self) -> None:
        run = compile(lambda x, alpha: x * alpha)(_spec(2), 3.0)
        np.testing.assert_allclose(
            run(_tensor(1.0, 1.0), 3.0).to_numpy(), [3.0, 3.0]
        )
        with pytest.raises(ValueError, match=r"expected 3\.0, got 4\.0"):
            run(_tensor(1.0, 1.0), 4.0)

    def test_dropping_a_positional_shifts_the_rest_and_is_rejected(
        self,
    ) -> None:
        """Silent misalignment is the failure this guards against."""
        spec_args, spec_kwargs = _mixed_specs(_spec)
        run = compile(_mixed_signature)(*spec_args, **spec_kwargs)
        args, kwargs = _mixed_args(_tensor)
        with pytest.raises(ValueError):
            run(args[0], args[1], *args[3:], **kwargs)

    def test_a_tensor_is_required_where_a_tensor_was_staged(self) -> None:
        run = compile(lambda x: x * 2)(_spec(2))
        with pytest.raises(TypeError, match="expected a Tensor"):
            run(_spec(2))


# ═════════════════════════════════════════════════════════════════════════
#  Buffers at the boundary
# ═════════════════════════════════════════════════════════════════════════


def _writes_its_argument(x: Tensor) -> Tensor:
    x[0:1] = x[0:1] * 2
    return x


def _writes_a_local(x: Tensor) -> Tensor:
    y = x * 2
    y[0:1] = x[0:1]
    return y


class TestMutability:
    """A buffer input is how mutable state crosses a boundary.

    A layout describes where a value *sits*; a buffer exists so a kernel can
    write *through* it, which no layout says. What may be written is settled
    at every boundary, not just the outer one, so each rule is proven twice --
    once where a graph takes its arguments, and once where a subgraph body
    takes its operands, since a body has graph inputs of its own that the
    caller's do not cover. A body carries one rule more: a buffer it promoted
    cannot leave through the call.
    """

    def test_writing_a_tensor_staged_argument_is_refused(self) -> None:
        """Promotion would store into a graph-local buffer the caller never
        sees, so it is an error rather than a lost write."""
        with pytest.raises(TypeError, match="staged as a tensor"):
            stage(_writes_its_argument)(_spec(2))

    def test_writing_a_tensor_staged_operand_is_refused_in_a_body(self) -> None:
        body = as_subgraph(_writes_its_argument, name="block")
        with pytest.raises(TypeError, match="staged as a tensor"):
            stage(lambda x: body(x))(_spec(2))

    def test_a_body_produced_tensor_may_still_be_promoted(self) -> None:
        """Nothing outside the graph names it, so create-and-store is sound."""
        run = compile(_writes_a_local)(_spec(2))
        np.testing.assert_allclose(
            run(_tensor(1.0, 3.0)).to_numpy(), [1.0, 6.0]
        )

    def test_a_body_produced_tensor_may_be_promoted_in_a_body(self) -> None:
        def scratch(x: Tensor) -> Tensor:
            return _writes_a_local(x) * 2

        local = as_subgraph(scratch, name="block")
        run = compile(lambda x: local(x))(_spec(2))
        np.testing.assert_allclose(
            run(_tensor(1.0, 3.0)).to_numpy(), [2.0, 12.0]
        )

    def test_a_body_cannot_hand_back_the_buffer_it_promoted(self) -> None:
        """A graph output forwards a buffer; a call result would create one.

        So the promotion stays inside the body, and what leaves is a load.
        """
        escapes = as_subgraph(_writes_a_local, name="block")
        with pytest.raises(ValueError, match=r"create new mo\.buffer"):
            stage(lambda x: escapes(x))(_spec(2))

    def test_a_body_writes_through_a_buffer_the_caller_promoted(self) -> None:
        """An operand's type is the body's argument type, so a value a body
        writes has to be a buffer before the call, where a store of its own
        would be too late."""

        def block(scratch: Tensor, x: Tensor) -> Tensor:
            scratch[0:1] = x[0:1]
            return scratch * 2  # reading it loads implicitly

        layer = as_subgraph(block, name="block")

        def model(x: Tensor) -> Tensor:
            scratch = x * 0.0
            scratch[0:1] = scratch[0:1]  # promotes, before the call
            # Read back through the caller's handle, not the call result.
            # Reading a buffer-backed tensor loads implicitly.
            return layer(scratch, x) + scratch

        staged = stage(model)(_spec(2))

        body = str(staged).split("mo.graph @block")[1]
        assert "!mo.buffer" in body.split("{")[0], "the body takes a buffer"
        assert "mo.buffer.create" not in body, "so it allocates none of its own"

    def test_a_sharded_buffer_writes_each_shard_and_no_other(self) -> None:
        mesh = _mesh2()

        def fill(cache: Tensor, x: Tensor) -> Tensor:
            F.buffer_store(cache, x)
            return x * 2

        run = compile(fill)(
            BufferLayout(_F32, [4], DeviceMapping(mesh, (Sharded(0),))),
            _sharded_spec(4),
        )
        cache = _sharded(_tensor(0.0, 0.0), _tensor(0.0, 0.0))
        run(cache, _sharded(_tensor(1.0, 2.0), _tensor(3.0, 4.0)))

        shards = [s.to_numpy() for s in cache.local_shards]
        np.testing.assert_allclose(shards[0], [1.0, 2.0])
        np.testing.assert_allclose(shards[1], [3.0, 4.0])

    @pytest.mark.xfail(
        strict=True,
        reason="an index that touches a sharded axis reads the source in "
        "the mesh's global index space and writes the destination in each "
        "device's local one, so every device stores global row 0. Only "
        "this combination is affected: a whole-buffer store is correct, "
        "and so is a slice that leaves the sharded axes whole.",
    )
    def test_a_sharded_buffer_writes_each_shard_through_a_slice(self) -> None:
        mesh = _mesh2()

        def fill(cache: Tensor, x: Tensor) -> Tensor:
            cache[0:1] = x[0:1]
            return x * 2

        run = compile(fill)(
            BufferLayout(_F32, [4], DeviceMapping(mesh, (Sharded(0),))),
            _sharded_spec(4),
        )
        cache = _sharded(_tensor(0.0, 0.0), _tensor(0.0, 0.0))
        run(cache, _sharded(_tensor(1.0, 2.0), _tensor(3.0, 4.0)))

        shards = [s.to_numpy() for s in cache.local_shards]
        np.testing.assert_allclose(shards[0], [1.0, 0.0])
        np.testing.assert_allclose(shards[1], [0.0, 0.0], err_msg="row 2")


# ═════════════════════════════════════════════════════════════════════════
#  One body, many calls
# ═════════════════════════════════════════════════════════════════════════

#: Appended to once per stage of the bodies below, so a test can tell a body
#: that was staged from one that was answered from the cache.
_TRACED: list[str] = []


def _block(x: Tensor) -> Tensor:
    _TRACED.append("block")
    return x * 2.0


def _indexed_block(x: Tensor, idx: int) -> Tensor:
    """A body whose per-layer index is baked in, the ``layer_idx`` shape."""
    _TRACED.append(f"indexed:{idx}")
    return x * float(idx)


def _scaled_block(scale: float) -> Callable[[Tensor], Tensor]:
    """Returns a body that captures ``scale``, which no name can reveal."""

    def block(x: Tensor) -> Tensor:
        _TRACED.append(f"scaled:{scale}")
        return x * scale

    return block


class TestSubgraphs:
    """The callable is the body's identity, together with what it is passed.

    Deduplicating on the staged IR instead means every repeat is staged and
    thrown away, sixty-one times over for a transformer's layers.
    """

    def test_repeated_calls_share_one_body_staged_once(self) -> None:
        _TRACED.clear()
        shared = as_subgraph(_block)
        staged = stage(lambda x: shared(shared(shared(x))))(_spec(2))
        assert _bodies(staged, "block") == 1
        assert _calls(staged.graph, "block") == 3
        assert _TRACED == ["block"], "later calls must reuse, not retrace"

    def test_a_differing_static_argument_splits_the_bodies(self) -> None:
        """The ``layer_idx`` case: one callable, a different constant per layer.

        The index is an argument rather than a capture, so it reaches the key
        through the argument tree and the layers are told apart without
        staging them to find out. Sharing one body here would silently compute
        every layer with the first layer's index.
        """
        _TRACED.clear()

        def stack(x: Tensor) -> Tensor:
            for i in range(3):
                x = as_subgraph(_indexed_block, name="indexed")(x, i)
            return x

        staged = stage(stack)(_spec(2))

        assert _TRACED == ["indexed:0", "indexed:1", "indexed:2"]
        # Three bodies, so three symbols, each called once.
        assert _calls(staged.graph, "indexed") == 1
        assert _calls(staged.graph, "indexed_1") == 1
        assert _calls(staged.graph, "indexed_2") == 1

    def test_each_closure_is_its_own_body(self) -> None:
        """Two closures over different values, which no name tells apart.

        They share a qualname and a signature, and what they captured reaches
        the graph as a constant. Keying on the callable separates them without
        staging either to find out.
        """
        _TRACED.clear()
        two, three = (
            as_subgraph(_scaled_block(scale), name="scaled")
            for scale in (2.0, 3.0)
        )
        staged = stage(lambda x: two(three(two(x))))(_spec(2))
        assert _TRACED == ["scaled:2.0", "scaled:3.0"]
        assert _calls(staged.graph, "scaled") == 2, "the repeat shares a body"
        assert _calls(staged.graph, "scaled_1") == 1

    def test_a_symbolic_extent_reaches_the_body(self) -> None:
        """An operand carries its graph's extent, symbolic ones included."""
        shared = as_subgraph(_block)
        staged = stage(lambda x: shared(x))(_symbolic("n"))
        signature = str(staged).split("mo.graph @block")[1].split("{")[0]
        assert "n" in signature, "the body took the extent it was passed"

    def test_a_nested_call_inlines_into_its_parent_body(self) -> None:
        """The runtime cannot load a body that itself calls one."""
        inner = as_subgraph(_block, name="inner")
        outer = as_subgraph(lambda x: inner(inner(x)), name="outer")
        staged = stage(lambda x: outer(outer(x)))(_spec(2))
        assert _calls(staged.graph, "outer") == 2
        assert _bodies(staged, "inner") == 0, "the inner body was inlined"

    def test_it_inlines_when_subgraphs_are_off_and_when_eager(self) -> None:
        """The wrapper is transparent, so one source serves every mode."""
        shared = as_subgraph(_block)
        staged = stage(lambda x: shared(shared(x)), allow_subgraphs=False)(
            _spec(2)
        )
        assert _calls(staged.graph, "block") == 0
        np.testing.assert_allclose(
            shared(_tensor(1.0, 2.0)).to_numpy(), [2.0, 4.0]
        )


# ═════════════════════════════════════════════════════════════════════════
#  Weights a shared body declares
# ═════════════════════════════════════════════════════════════════════════


def _weighted_block(x: Tensor) -> Tensor:
    return x * F.constant_external(
        "w", TensorLayout(_F32, [1], DeviceRef.CPU()), is_placeholder=True
    )


class TestWeights:
    def test_a_prefix_resolves_the_body_s_weights_per_call_site(self) -> None:
        """One definition, and each call site reads its own entry out of it."""

        def model(x: Tensor) -> Tensor:
            for name in ("l0.", "l1."):
                x = as_subgraph(_weighted_block, name="block", prefix=name)(x)
            return x

        weights = {"l0.w": _tensor(2.0), "l1.w": _tensor(10.0)}
        staged = stage(model)(_spec(1))
        assert _bodies(staged, "block") == 1, "one body for both call sites"
        run = CompiledCallable(staged, weights)
        np.testing.assert_allclose(run(_tensor(1.0)).to_numpy(), [20.0])

    def test_declaring_in_full_inside_a_body_overrides_the_prefix(self) -> None:
        """``is_placeholder=False`` under a prefix: one shared name, not one each.

        The default follows the prefix, which is what a stack of layers wants.
        Overridden, every call site reads the same checkpoint entry, a tied
        weight, and the name in the body is already complete.
        """

        def body(x: Tensor) -> Tensor:
            return x * F.constant_external(
                "shared.w",
                TensorLayout(_F32, [1], DeviceRef.CPU()),
                is_placeholder=False,
            )

        def model(x: Tensor) -> Tensor:
            for name in ("l0.", "l1."):
                x = as_subgraph(body, name="block", prefix=name)(x)
            return x

        staged = stage(model)(_spec(1))
        whole = str(staged)
        assert _calls(staged.graph, "block") == 2, "still one body, two calls"
        assert 'name = "shared.w"' in whole
        assert "isPlaceholder = true" not in whole
        run = CompiledCallable(staged, {"shared.w": _tensor(3.0)})
        np.testing.assert_allclose(run(_tensor(1.0)).to_numpy(), [9.0])


# ═════════════════════════════════════════════════════════════════════════
#  A model: the shape a pipeline actually asks for
# ═════════════════════════════════════════════════════════════════════════

_MODEL = 4
_SLOTS = 3
_LAYERS = 2


@dataclass
class Cache:
    """What a layer sees: every slot is a tensor.

    Which slot is a buffer is decided once, at the boundary, by the spec it
    given, and nowhere else.
    """

    blocks: Tensor
    scale: Tensor

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return {f.name: getattr(self, f.name) for f in fields(self)}, None

    @classmethod
    def __tree_unflatten__(
        cls, meta: None, children: Mapping[str, Any]
    ) -> Cache:
        del meta
        return cls(**children)


@dataclass
class CacheSpec:
    """How a :class:`Cache` describes its boundary.

    One class per tree, so every field is exactly one type and neither lies.
    Unflattening rebuilds the data class, so a body is handed a ``Cache``.
    """

    blocks: BufferLayout
    scale: TensorLayout

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return {f.name: getattr(self, f.name) for f in fields(self)}, None

    @classmethod
    def __tree_unflatten__(
        cls, meta: None, children: Mapping[str, Any]
    ) -> Cache:
        del meta
        return Cache(**children)


def _weight() -> Tensor:
    return F.constant_external(
        "w",
        TensorLayout(_F32, [_MODEL, _MODEL], device=DeviceRef.CPU()),
        is_placeholder=True,
    )


def _layer_weights(*scales: float) -> dict[str, Tensor]:
    return {
        f"layers.{i}.w": _ones(_MODEL, _MODEL) * s for i, s in enumerate(scales)
    }


def _cache_spec() -> CacheSpec:
    """A fixed number of slots as a buffer, and a value read alongside them."""
    return CacheSpec(
        blocks=BufferLayout(_F32, [_SLOTS, _MODEL], DeviceRef.CPU()),
        scale=_spec(1),
    )


def _offset_spec() -> TensorLayout:
    """Where this step writes, known only at run time."""
    return TensorLayout(
        DType.int64,
        [],
        PlacementMapping(DeviceMesh.single(CPU()), (Replicated(),)),
    )


def _offset(value: int) -> Tensor:
    return Tensor.from_dlpack(np.array(value, dtype=np.int64))


def _decode_layer(x: Tensor, cache: Cache, pos: Tensor) -> Tensor:
    """Project, append this step at ``pos``, then read the slot back."""
    y = x @ _weight()
    cache.blocks[pos] = y[0]
    return y + cache.blocks[pos] * cache.scale


def _decode_stack() -> Any:
    """One body, one cache per layer, one shared write offset."""

    def model(x: Tensor, caches: list[Cache], pos: Tensor) -> Tensor:
        for i, cache in enumerate(caches):
            layer = as_subgraph(
                _decode_layer, name="layer", prefix=f"layers.{i}."
            )
            x = layer(x, cache, pos)
        return x

    return model


def _empty_caches() -> list[Cache]:
    return [
        Cache(
            blocks=Tensor.from_dlpack(
                np.zeros((_SLOTS, _MODEL), dtype=np.float32)
            ),
            scale=_ones(1),
        )
        for _ in range(_LAYERS)
    ]


@dataclass
class PagedSpec:
    """How a :class:`PagedCacheValues` describes its boundary.

    Mirrors every field, so the tree matches and unflattening hands the body
    the production class. Only ``kv_blocks`` is a buffer.
    """

    kv_blocks: BufferLayout
    cache_lengths: TensorLayout
    lookup_table: TensorLayout
    max_prompt_length: TensorLayout
    max_cache_length: TensorLayout
    page_stride: TensorLayout
    kv_scales: None = None
    attention_dispatch_metadata: TensorLayout | None = None
    mla_num_partitions: None = None

    def __tree_flatten__(self) -> tuple[tuple[Any, ...], tuple[str, ...]]:
        names = tuple(f.name for f in fields(self))
        return tuple(getattr(self, name) for name in names), names

    @classmethod
    def __tree_unflatten__(
        cls, aux: tuple[str, ...], children: Sequence[Any]
    ) -> PagedCacheValues:
        return PagedCacheValues(**dict(zip(aux, children, strict=True)))


@overload
def _spec_of(type: BufferType) -> BufferLayout: ...
@overload
def _spec_of(type: TensorType) -> TensorLayout: ...
@overload
def _spec_of(type: None) -> None: ...


def _spec_of(
    type: TensorType | BufferType | None,
) -> TensorLayout | None:
    """The spec a production KV cache's graph type describes.

    ``KVCacheParams.get_symbolic_inputs`` hands back ``max.graph`` types, so a
    pipeline that already has them converts once at the boundary.
    """
    if type is None:
        return None
    if isinstance(type, BufferType):
        return BufferLayout(type.dtype, type.shape, type.device)
    return TensorLayout(type.dtype, type.shape, type.device)


class TestModel:
    """A stack of layers over a cache, which is what all of this is for.

    The shape is a decode step: one body per layer group, a distinct cache
    per layer, a sequence extent that is symbolic because prefill and decode
    pass different ones, and a write offset that only the caller knows.
    """

    def test_each_layer_gets_its_own_buffer_out_of_one_body(self) -> None:
        x_spec = _symbolic("seq", _MODEL)
        staged = stage(_decode_stack())(
            x_spec,
            [_cache_spec() for _ in range(_LAYERS)],
            _offset_spec(),
        )
        assert _bodies(staged, "layer") == 1
        assert _calls(staged.graph, "layer") == _LAYERS
        buffers = [i for i in staged.graph.inputs if isinstance(i, BufferValue)]
        assert len(buffers) == _LAYERS, "one cache per layer, not one shared"

        body = str(staged).split("mo.graph @layer")[1]
        signature = body.split("{")[0]
        assert "!mo.buffer" in signature, "the body takes a buffer too"
        assert "seq" in signature, "and the extent its caller passed"
        assert "mo.buffer.create" not in body, "so nothing is copied in"

    def test_a_decode_loop_advances_the_cache_and_then_prefills(self) -> None:
        run = compile(
            _decode_stack(),
            weights=_layer_weights(2.0, 10.0),
        )(
            _symbolic("seq", _MODEL),
            [_cache_spec() for _ in range(_LAYERS)],
            _offset_spec(),
        )
        caches = _empty_caches()

        for step in range(2):
            out = run(_ones(1, _MODEL), caches, _offset(step))

        # x @ (2*ones) sums 4 ones -> 8, doubled by reading the slot back; then
        # 16 @ (10*ones) -> 640, likewise doubled.
        np.testing.assert_allclose(out.to_numpy(), np.full((1, _MODEL), 1280.0))
        first, second = (c.blocks.to_numpy() for c in caches)
        np.testing.assert_allclose(first[:2], np.full((2, _MODEL), 8.0))
        np.testing.assert_allclose(second[:2], np.full((2, _MODEL), 640.0))
        for slots in (first, second):
            np.testing.assert_allclose(slots[2], 0.0, err_msg="slot untouched")

        # The same compiled model serves a longer sequence.
        out = run(_ones(_SLOTS, _MODEL), caches, _offset(2))
        assert out.shape == [_SLOTS, _MODEL]
        np.testing.assert_allclose(caches[0].blocks.to_numpy()[2], 8.0)

    def test_over_a_mesh_each_device_owns_its_shard_of_the_cache(self) -> None:
        """A cache sharded across devices, written and read inside one body.

        Every device holds a different slice of the same cache, of a different
        length, and the body ends by gathering it, so the write has to land
        per device and the read has to cross them.
        """
        mesh = _mesh2()
        sharded = PlacementMapping(mesh, (Sharded(0),))
        cache_spec = CacheSpec(
            blocks=BufferLayout(
                _F32, [_SLOTS, _MODEL], DeviceMapping(mesh, (Sharded(0),))
            ),
            scale=_replicated_spec(1),
        )
        x_spec = _replicated_spec(_SLOTS, _MODEL)

        def layer(x: Tensor, cache: Cache) -> Tensor:
            y = F.transfer_to(x @ _weight(), sharded)
            cache.blocks[...] = y
            return F.allgather(cache.blocks) * cache.scale

        def model(x: Tensor, cache: Cache) -> Tensor:
            for i in range(_LAYERS):
                shared = as_subgraph(layer, name="layer", prefix=f"layers.{i}.")
                x = shared(x, cache)
            return x

        staged = stage(model)(x_spec, cache_spec)
        assert _bodies(staged, "layer") == 1
        assert _calls(staged.graph, "layer") == _LAYERS
        buffers = [i for i in staged.graph.inputs if isinstance(i, BufferValue)]
        assert len(buffers) == 2, "one cache shard per device"

        run = CompiledCallable(staged, _layer_weights(2.0, 10.0))
        cache = Cache(
            blocks=_sharded(_ones(2, _MODEL) * 0.0, _ones(1, _MODEL) * 0.0),
            scale=_replicated(_ones(1)),
        )
        out = run(_replicated(_ones(_SLOTS, _MODEL)), cache)

        # 8 replicated, split across devices, gathered back; then 8 @ (10*ones).
        for shard in out.local_shards:
            np.testing.assert_allclose(
                shard.to_numpy(), np.full((_SLOTS, _MODEL), 320.0)
            )
        shapes = [s.to_numpy().shape for s in cache.blocks.local_shards]
        assert shapes == [(2, _MODEL), (1, _MODEL)], "an uneven split"
        for shard in cache.blocks.local_shards:
            np.testing.assert_allclose(shard.to_numpy(), 320.0)

    def test_a_paged_transformer_shares_one_body(self) -> None:
        """The production paged cache, declared at a subgraph boundary.

        :meth:`KVCacheParams.get_symbolic_inputs` already types one device's
        boundary as a ``BufferLayout`` for the page pool and a ``TensorLayout``
        per piece of metadata, which is one spec per tensor argument, and
        :class:`PagedCacheValues` already carries the pytree protocol. So a
        real attention layer reaches this with nothing added to either.

        Staged rather than run: the page contents a kernel reads are the
        integration suite's business, while the boundary is this file's.
        """
        heads, head_dim = 4, 16
        hidden = heads * head_dim
        kv_params = MHAKVCacheParams(
            dtype=_F32,
            n_kv_heads=1,
            head_dim=head_dim,
            num_layers=_LAYERS,
            page_size=128,
            devices=[DeviceRef.CPU()],
        )
        declared = kv_params.get_symbolic_inputs()[0]
        cache_spec = PagedSpec(
            kv_blocks=_spec_of(declared.kv_blocks),
            cache_lengths=_spec_of(declared.cache_lengths),
            lookup_table=_spec_of(declared.lookup_table),
            max_prompt_length=_spec_of(declared.max_prompt_length),
            max_cache_length=_spec_of(declared.max_cache_length),
            page_stride=_spec_of(declared.page_stride),
            attention_dispatch_metadata=_spec_of(
                declared.attention_dispatch_metadata
            ),
        )
        x_spec = TensorLayout(
            _F32, ["total_seq_len", hidden], device=DeviceRef.CPU()
        )
        rows_spec = TensorLayout(DType.uint32, ["rows"], device=DeviceRef.CPU())

        def projection(name: str) -> Tensor:
            return F.constant_external(
                name,
                TensorLayout(_F32, [hidden, hidden], device=DeviceRef.CPU()),
                is_placeholder=True,
            )

        def block(
            x: Tensor,
            cache: PagedCacheValues,
            layer_idx: Tensor,
            input_row_offsets: Tensor,
        ) -> Tensor:
            q = (x @ projection("attn.qkv")).reshape((-1, heads, head_dim))
            attention = flash_attention_ragged(
                kv_params,
                input=q,
                kv_collection=cache,
                layer_idx=layer_idx,
                input_row_offsets=input_row_offsets,
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
                scale=math.sqrt(1.0 / head_dim),
            )
            return attention.reshape((-1, hidden)) @ projection("attn.o")

        def transformer(
            x: Tensor, cache: PagedCacheValues, input_row_offsets: Tensor
        ) -> Tensor:
            for i in range(_LAYERS):
                # The index crosses as an operand. Baked in as a constant,
                # the way a layer holding its own does, it would be one body
                # per layer.
                x = as_subgraph(block, name="block", prefix=f"layers.{i}.")(
                    x,
                    cache,
                    F.constant(i, DType.uint32, device=CPU()),
                    input_row_offsets,
                )
            return x

        staged = stage(transformer)(x_spec, cache_spec, rows_spec)
        assert _bodies(staged, "block") == 1
        assert _calls(staged.graph, "block") == _LAYERS

        buffers = [i for i in staged.graph.inputs if isinstance(i, BufferValue)]
        assert len(buffers) == 1, "one page pool, shared by every layer"

        body = str(staged).split("mo.graph @block")[1]
        signature = body.split("{")[0]
        assert "!mo.buffer" in signature, "the pool is a buffer in the body"
        assert "total_num_pages" in signature, "and stays symbolic"
        assert 'name = "attn.qkv"' in body, "weights are named relatively"
        for i in range(_LAYERS):
            assert f'prefix = "layers.{i}."' in str(staged.graph)


# ═════════════════════════════════════════════════════════════════════════
#  Trace, compile, export, execute
# ═════════════════════════════════════════════════════════════════════════


class TestLifecycle:
    def test_a_traced_graph_compiles_exports_and_executes(
        self, tmp_path: Path
    ) -> None:
        staged = stage(lambda x: x * 2, name="scale")(_spec(2))
        assert "mo.graph @scale" in str(staged), "the name is the transform's"

        run = CompiledCallable(staged)
        arg = _tensor(1.0, 2.0)
        np.testing.assert_allclose(run(arg).to_numpy(), [2.0, 4.0])

        raw = run.execute_raw(arg.driver_tensor)
        np.testing.assert_allclose(
            Tensor(storage=raw[0]).to_numpy(), [2.0, 4.0]
        )

        path = tmp_path / "scale.mef"
        run.export_mef(path)
        assert path.stat().st_size > 0

    def test_a_transform_option_may_also_be_a_parameter_name(self) -> None:
        """``name`` is the transform's; a body's own ``name`` is untouched."""
        staged = stage(lambda x, name: x * len(name), name="outer")(
            _spec(2), "ab"
        )
        assert "mo.graph @outer" in str(staged)


class TestATensorIsALeaf:
    """A tensor is one value, not a container of its per-device pieces.

    Only the graph boundary splits one into per-device values and puts it back
    together, so every other walk sees a tensor whole and needs no ``leaf=``
    to say so. This is the one test of that, end to end.
    """

    def test_a_leaf_everywhere_and_still_split_per_device_at_a_boundary(
        self,
    ) -> None:
        mapping = PlacementMapping(_mesh2(), (Sharded(0),))
        sharded = _tensor(1.0, 2.0, 3.0, 4.0).to(mapping)

        # A walk stops at a tensor, sharded or not, with no leaf= to ask for it.
        assert tree_flatten(sharded)[0] == [sharded]
        assert list(tree_paths({"w": [sharded]})) == ["w.0"]

        # The boundary still expands one argument into one input per device,
        # and a nested result comes back nested, with its shards rejoined.
        def split_and_pair(x: Tensor) -> dict[str, Any]:
            doubled = x * 2
            return {"doubled": doubled, "pair": [doubled, x]}

        spec = TensorLayout(_F32, [4], mapping)
        staged = stage(split_and_pair)(spec)
        assert len(staged.graph.inputs) == 2, "one input per device"

        out = CompiledCallable(staged)(sharded)

        assert list(out) == ["doubled", "pair"], "nested as returned"
        assert out["pair"][1].num_shards == 2, "a tensor, not its shards"
        np.testing.assert_allclose(
            out["doubled"].to_numpy(), [2.0, 4.0, 6.0, 8.0]
        )
        np.testing.assert_allclose(out["pair"][1].to_numpy(), [1, 2, 3, 4])
