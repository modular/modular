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
"""Tests for `max.experimental.custom.declare` and `CustomOp` staging, and
the interpreter binding cache (`custom_gc`) they feed."""
# MXF-353

import dataclasses
import inspect
import itertools
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from max import _core, _interpreter, engine
from max._core.dialects import builtin, mo
from max._interpreter_ops import custom_gc
from max._interpreter_ops import handlers as _handlers
from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental import custom as C
from max.experimental import executor
from max.experimental import functional as F
from max.experimental import realization_context as rc
from max.experimental.sharding import DeviceMesh, PlacementMapping, Sharded
from max.experimental.sharding.rules import unary_rule
from max.experimental.tensor import Tensor, realization_context
from max.experimental.testing import assert_all_close
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    DimLike,
    Graph,
    TensorType,
    TensorValue,
    default_custom_extensions_scope,
    ops,
)

KERNELS = Path(os.environ["MODULAR_CUSTOM_OP_DEF_KERNELS_PATH"])
KERNEL_VERIFICATION_OPS = Path(
    os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"]
)

# --- Small builders to keep the CustomOp-construction tests below dense.


def _f32(*dims: DimLike) -> TensorType:
    return TensorType(DType.float32, list(dims), DeviceRef.CPU())


_CustomOpResult = Tensor | list[Tensor] | TensorValue | list[TensorValue]


def _one_tensor(result: _CustomOpResult) -> Tensor:
    """Narrows an eager `CustomOp` call's single-output result to `Tensor`."""
    assert isinstance(result, Tensor)
    return result


def _one_value(result: _CustomOpResult) -> TensorValue:
    """Narrows a graph-staged `CustomOp` call's single-output result to
    `TensorValue`."""
    assert isinstance(result, TensorValue)
    return result


def _assert_values(expected: Any, actual: Tensor) -> None:
    """Asserts *actual* has the shape and values of the nested *expected*.

    `assert_all_close` reduces with `max(axis=-1)`, so it only lands on a
    scalar for a rank-1 tensor; comparing flattened copies keeps any rank on
    that path, and the shape is checked separately rather than implied.
    """
    want = Tensor(expected, dtype=actual.dtype, device=actual.device)
    assert [int(d) for d in want.shape] == [int(d) for d in actual.shape]
    count = actual.num_elements()
    assert_all_close(want.reshape([count]), actual.reshape([count]))


def _zeros(*shape: int) -> Tensor:
    return Tensor.zeros(list(shape), dtype=DType.float32, device=CPU())


def _full(value: float, *shape: int) -> Tensor:
    return Tensor.full(list(shape), value, dtype=DType.float32, device=CPU())


def make_downsample() -> C.CustomOp:
    n, w, c = C.Symbols("n", "w", "c")
    return C.declare(
        "mxf353_downsample",
        inputs={"x": C.TemplateType(DType.float32, [n, w, c])},
        outputs=[C.TemplateType(DType.float32, [n, w // 2, c])],
        custom_extensions=[KERNELS],
    )


def make_copy() -> C.CustomOp:
    """``mxf353_downsample`` with dim 1 undivided: an elementwise copy.

    The kernel copies elementwise over the OUTPUT's own dims, so an
    undivided declaration is as correct as a halved one and gives the
    composition tests a shape-preserving op.
    """
    n, w, c = C.Symbols("n", "w", "c")
    return C.declare(
        "mxf353_downsample",
        inputs={"x": C.TemplateType(DType.float32, [n, w, c])},
        outputs=[C.TemplateType(DType.float32, [n, w, c])],
        custom_extensions=[KERNELS],
    )


def make_q8_like(dtype_var: str = "A") -> C.CustomOp:
    """Two rank-2 inputs sharing their inner "k" dim: `x:[m,k]`, `y:[n,k] ->
    out:[m,n]`. Both inputs template on the SAME dtype variable, so a dtype
    mismatch between them is a definition-contract violation, not just a
    kernel-level one."""
    var = C.DTypeVar(dtype_var)
    m, k, n = C.Symbols("m", "k", "n")
    return C.declare(
        "mxf353_q8_like",
        inputs={
            "a": C.TemplateType(var, [m, k]),
            "b": C.TemplateType(var, [n, k]),
        },
        outputs=[C.TemplateType(var, [m, n])],
        custom_extensions=[KERNELS],
    )


def make_downsample_by(divisor: int) -> C.CustomOp:
    """``mxf353_downsample`` with dim 1 divided by *divisor* instead of 2.

    The kernel copies elementwise over the OUTPUT's own dims, so any divisor
    produces a correct result, letting two signatures share one kernel.
    """
    n, w, c = C.Symbols("n", "w", "c")
    return C.declare(
        "mxf353_downsample",
        inputs={"x": C.TemplateType(DType.float32, [n, w, c])},
        outputs=[C.TemplateType(DType.float32, [n, w // divisor, c])],
        custom_extensions=[KERNELS],
    )


def make_downsample_from(width: int) -> C.CustomOp:
    """``mxf353_downsample`` accepting only inputs whose dim 1 is *width*.

    Shares result types with ``make_downsample_by(width // 4)``, so the two
    differ in nothing but their INPUT template.
    """
    n, c = C.Symbols("n", "c")
    return C.declare(
        "mxf353_downsample",
        inputs={"x": C.TemplateType(DType.float32, [n, width, c])},
        outputs=[C.TemplateType(DType.float32, [n, width // 4, c])],
        custom_extensions=[KERNELS],
    )


@pytest.fixture(autouse=True)
def _clear_binding_cache() -> Iterator[None]:
    """Clears the process-global binding cache so a hit from an earlier
    test can't mask a real miss."""
    custom_gc._CACHE.clear()
    yield
    custom_gc._CACHE.clear()


@pytest.fixture
def force_interpreter_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forces every eager call through the interpreter (no compile
    fallback) by patching the module-global ``_DEFAULT_EXECUTOR`` directly:
    it's read once at import time, so ``monkeypatch.setenv`` has no effect."""
    monkeypatch.setattr(
        executor,
        "_DEFAULT_EXECUTOR",
        executor.InterpreterExecutor(max_ops=None),
    )


def _spy_on_compiles(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wraps ``custom_gc._compile`` to count real compiles -- a recompile onto
    the same key would still show a correct cache *size*, so ``len(_CACHE)``
    alone can't catch that."""
    counter = [0]
    original = custom_gc._compile

    def counting(*args: Any, **kwargs: Any) -> engine.Model:
        counter[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(custom_gc, "_compile", counting)
    return counter


def _call_scale(n: int) -> Tensor:
    """Stages+runs an undeclared ``mxf353_scale`` call at shape ``[n]``."""
    (y,) = F.custom(
        "mxf353_scale",
        device=CPU(),
        values=[_full(2.0, n)],
        out_types=[_f32(n)],
        parameters={"factor": 4},
        custom_extensions=KERNELS,
    )
    return y


def test_dual_context_eager_and_graph_staging() -> None:
    """A `CustomOp` behaves identically whether called eagerly (returns a
    realized `Tensor`) or staged inside an explicit `Graph` (returns a
    symbolic `TensorValue` with its output dim preserved algebraically)."""
    y = make_downsample()(_zeros(2, 8, 3))
    assert isinstance(y, Tensor)
    assert [int(d) for d in y.shape] == [2, 4, 3]

    in_t = TensorType(DType.float32, ["n", "w", "c"], DeviceRef.CPU())
    with Graph("g", input_types=[in_t], custom_extensions=[KERNELS]) as g:
        out = make_downsample()(g.inputs[0].tensor)
        g.output(_one_value(out))
    output_type = g.output_types[0]
    assert isinstance(output_type, TensorType)
    assert str(output_type.shape[1]) == "w // 2"


# --- custom_gc: binding cache for CustomOp-declared ops (MXF-353) --------


def test_binding_key_reflects_result_types() -> None:
    """`BindingKey` folds in the declared result types, not just the operand
    types: a collision regression, a kernel shared by signatures with
    different result types must never alias one binding. The key stays over
    SYMBOLIC input types too, so this can't smuggle shape back in."""
    cpu = CPU()
    half, third = make_downsample_by(2), make_downsample_by(3)
    in12 = [_f32(2, 12, 3)]
    assert custom_gc.make_key(half, in12, [KERNELS]) != custom_gc.make_key(
        third, in12, [KERNELS]
    )
    m_half = custom_gc.binding_for(half, cpu, in12, [KERNELS])
    m_third = custom_gc.binding_for(third, cpu, in12, [KERNELS])
    assert m_half is not m_third
    x12 = Buffer.zeros((2, 12, 3), DType.float32, cpu)
    assert tuple(m_half(x12)[0].shape) == (2, 6, 3)
    assert tuple(m_third(x12)[0].shape) == (2, 4, 3)

    defn = make_downsample()
    assert custom_gc.make_key(
        defn, [_f32(2, 8, 3)], [KERNELS]
    ) == custom_gc.make_key(defn, [_f32(5, 100, 7)], [KERNELS])


def test_binding_key_reflects_input_template() -> None:
    """The key is derived from the BINDING's signature, so a template that
    pins an input dim to a static can't alias the def that leaves the same
    dim symbolic: they compile different graphs (one static-shaped, one
    rank-polymorphic) from the same kernel, result types and extensions, so
    nothing else in the key would tell them apart."""
    cpu = CPU()
    plain, pinned = make_downsample_by(4), make_downsample_from(8)
    in8 = [_f32(2, 8, 3)]

    assert custom_gc.make_key(plain, in8, [KERNELS]) != custom_gc.make_key(
        pinned, in8, [KERNELS]
    )
    custom_gc.binding_for(plain, cpu, in8, [KERNELS])
    custom_gc.binding_for(pinned, cpu, in8, [KERNELS])
    assert len(custom_gc._CACHE) == 2


def test_binding_key_changes_on_kernel_source_edit(tmp_path: Path) -> None:
    """A rebuild that only touches kernel SOURCE bytes must change the
    binding key -- both for a single tracked file, and for an edit to one
    file inside a whole source directory (hashing only the directory's own
    mtime would miss that)."""
    defn = make_downsample()
    in_types = [_f32(2, 8, 3)]

    pkg_path = tmp_path / "fake.mojopkg"
    pkg_path.write_bytes(b"kernel bytes v1")
    key1 = custom_gc.make_key(defn, in_types, [pkg_path])
    pkg_path.write_bytes(pkg_path.read_bytes() + b"\x00")  # simulate rebuild
    assert custom_gc.make_key(defn, in_types, [pkg_path]) != key1

    src_dir = tmp_path / "kernels"
    src_dir.mkdir()
    (src_dir / "a.mojo").write_bytes(b"struct A: pass")
    (src_dir / "b.mojo").write_bytes(b"struct B: pass")
    key2 = custom_gc.make_key(defn, in_types, [src_dir])
    (src_dir / "b.mojo").write_bytes(b"struct B: pass  # edited")
    assert custom_gc.make_key(defn, in_types, [src_dir]) != key2


def test_binding_key_changes_on_size_preserving_edit(tmp_path: Path) -> None:
    """A same-size edit that leaves the recorded mtime untouched must still
    change the key: a bazel-written library can carry a pinned mtime, so a
    stat-keyed digest memo would hand back a stale binding."""
    defn = make_downsample()
    in_types = [_f32(2, 8, 3)]

    pkg_path = tmp_path / "fake.mojopkg"
    pkg_path.write_bytes(b"kernel bytes v1")
    stat = pkg_path.stat()
    key1 = custom_gc.make_key(defn, in_types, [pkg_path])
    pkg_path.write_bytes(b"kernel bytes v2")
    os.utime(pkg_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert pkg_path.stat().st_mtime_ns == stat.st_mtime_ns
    assert custom_gc.make_key(defn, in_types, [pkg_path]) != key1


def test_binding_key_reflects_overlay_extensions(tmp_path: Path) -> None:
    """The key covers the process-global overlay too, not just the def's own
    extensions: the compiled graph links against both (``Graph.__init__``),
    so a binding keyed on one half could outlive the kernel bytes it was
    built from, or keep serving after the overlay scope exits."""
    defn = make_downsample()
    in_types = [_f32(2, 8, 3)]

    overlay = tmp_path / "overlay.mojopkg"
    overlay.write_bytes(b"overlay kernel bytes")
    outside = custom_gc.make_key(defn, in_types, [KERNELS])
    with default_custom_extensions_scope(overlay):
        assert custom_gc.make_key(defn, in_types, [KERNELS]) != outside
    assert custom_gc.make_key(defn, in_types, [KERNELS]) == outside


def test_binding_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The binding cache stays within its bound. Each entry pins an
    ``engine.Model`` and its MEF buffer, so an unbounded cache leaks for a
    process that keeps declaring new signatures."""
    monkeypatch.setattr(custom_gc, "_CACHE_MAX_SIZE", 2)
    cpu, in12 = CPU(), [_f32(2, 12, 3)]
    for divisor in (2, 3, 4):
        custom_gc.binding_for(make_downsample_by(divisor), cpu, in12, [KERNELS])
        assert len(custom_gc._CACHE) <= 2
    # Exactly at the bound, so the three distinct keys really did evict one
    # rather than aliasing onto a single entry.
    assert len(custom_gc._CACHE) == 2


def test_binding_cache_evicts_least_recently_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hit refreshes recency, so eviction drops the least recently USED entry
    and not simply the oldest inserted. ``_cached``'s ``move_to_end`` is the
    only thing making this an LRU rather than an insertion-order FIFO, and
    deleting it leaves every other assertion in this suite passing."""
    monkeypatch.setattr(custom_gc, "_CACHE_MAX_SIZE", 2)
    cpu, in12 = CPU(), [_f32(2, 12, 3)]
    first, second, third = (make_downsample_by(d) for d in (2, 3, 4))
    keys = [
        custom_gc.make_key(defn, in12, [KERNELS])
        for defn in (first, second, third)
    ]
    custom_gc.binding_for(first, cpu, in12, [KERNELS])
    custom_gc.binding_for(second, cpu, in12, [KERNELS])
    custom_gc.binding_for(first, cpu, in12, [KERNELS])
    custom_gc.binding_for(third, cpu, in12, [KERNELS])
    assert keys[0] in custom_gc._CACHE
    assert keys[1] not in custom_gc._CACHE
    assert keys[2] in custom_gc._CACHE


def test_declared_tier_scenario(
    force_interpreter_only: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One declared-tier pass through the interpreter binding cache:
    rank-polymorphic reuse under exactly-one-compile discipline,
    redeclaration dedup, and the multi-output list branch."""
    compiles = _spy_on_compiles(monkeypatch)

    op = make_downsample()
    y1 = _one_tensor(op(_zeros(2, 8, 3)))
    y2 = _one_tensor(op(_zeros(5, 100, 7)))
    assert [int(d) for d in y1.shape] == [2, 4, 3]
    assert [int(d) for d in y2.shape] == [5, 50, 7]
    assert compiles[0] == 1  # one real compile serves both shapes of the rank
    assert len(custom_gc._CACHE) == 1

    # A separately-constructed but identical redeclaration (e.g. a re-run
    # notebook cell) dedups onto the same binding rather than recompiling.
    op_again = make_downsample()
    assert op_again is not op and op_again.token == op.token
    op_again(_zeros(3, 20, 1))
    assert compiles[0] == 1
    assert len(custom_gc._CACHE) == 1

    # A different declaration over the same kernel mints its OWN binding with
    # its own correct numerics.
    third = make_downsample_by(3)
    y3 = _one_tensor(third(_full(2.0, 1, 9, 1)))
    _assert_values([[[2.0], [2.0], [2.0]]], y3)
    assert compiles[0] == 2
    assert len(custom_gc._CACHE) == 2

    # `_single_or_list`'s list branch: a 2-output kernel's eager call returns
    # a `list[Tensor]`, not a bare `Tensor`. Shape [1]: the kernel writes
    # only index 0 of each output, so this is exactly checkable.
    (d,) = C.Symbols("d")
    multi_op = C.declare(
        "op_with_multiple_outputs",
        inputs={"x": C.TemplateType(DType.float32, [d])},
        outputs=[
            C.TemplateType(DType.float32, [d]),
            C.TemplateType(DType.float32, [d]),
        ],
        custom_extensions=[KERNEL_VERIFICATION_OPS],
    )
    outs = multi_op(_full(2.0, 1))
    assert isinstance(outs, list) and len(outs) == 2
    _assert_values([4.0], _one_tensor(outs[0]))
    _assert_values([8.0], _one_tensor(outs[1]))


def test_resolve_spec_refuses_a_disagreeing_def() -> None:
    """A token resolving to a def that contradicts the op it is stamped on is
    refused, since binding a kernel to the wrong declaration would run silently
    wrong results. ``None`` is what an op never staged through a `CustomOp`
    already returns, and the caller answers it by compiling."""
    op = make_downsample()
    with Graph(
        "g_resolve_spec",
        input_types=[_f32(2, 8, 3)],
        custom_extensions=[KERNELS],
    ) as g:
        out = _one_value(op(g.inputs[0].tensor))
        g.output(out)
    staged = out._mlir_value.owner
    assert isinstance(staged, mo.CustomOp)
    assert _handlers._resolve_spec(staged) is op

    # A token for a different kernel: what replaying a serialized graph in a
    # process whose registry holds different defs can produce.
    other_kernel = make_q8_like().token
    staged.discardable_attributes[C._SPEC_ATTR_KEY] = builtin.StringAttr(
        other_kernel
    )
    assert _handlers._resolve_spec(staged) is None

    # Right kernel, wrong arity: two declared results against one staged.
    n, w, c = C.Symbols("n", "w", "c")
    two_results = C.declare(
        "mxf353_downsample",
        inputs={"x": C.TemplateType(DType.float32, [n, w, c])},
        outputs=[
            C.TemplateType(DType.float32, [n, w // 2, c]),
            C.TemplateType(DType.float32, [n, w // 2, c]),
        ],
        custom_extensions=[KERNELS],
    )
    staged.discardable_attributes[C._SPEC_ATTR_KEY] = builtin.StringAttr(
        two_results.token
    )
    assert _handlers._resolve_spec(staged) is None


def test_can_execute_predicate() -> None:
    """``can_execute`` accepts a plain custom op staged via `CustomOp` but
    refuses an in-place one: ``ops.inplace_custom`` always adds a trailing
    chain operand/result, and ``ChainType``/``BufferType`` have no
    interpreter ``Buffer`` representation. An undeclared op is refused here
    too, statically, rather than part-way through execution."""
    in_t = TensorType(DType.float32, ["n", "w", "c"], DeviceRef.CPU())
    with Graph("g_plain", input_types=[in_t], custom_extensions=[KERNELS]) as g:
        out = make_downsample()(g.inputs[0].tensor)
        g.output(_one_value(out))
    assert _interpreter.can_execute(g) is True

    buffer_type = BufferType(DType.float32, [64], DeviceRef.CPU())
    graph = Graph("g_inplace", input_types=[buffer_type])
    with graph:
        graph._import_kernels([KERNEL_VERIFICATION_OPS])
        ops.inplace_custom(
            "mutable_input_tensor",
            device=DeviceRef.CPU(),
            values=[graph.inputs[0]],
        )
        graph.output()
    assert _interpreter.can_execute(graph) is False

    undeclared = Graph(
        "g_undeclared", input_types=[_f32(4)], custom_extensions=[KERNELS]
    )
    with undeclared:
        (y,) = ops.custom(
            "mxf353_scale",
            DeviceRef.CPU(),
            list(undeclared.inputs),
            out_types=[_f32(4)],
            parameters={"factor": 4},
        )
        undeclared.output(y)
    assert _interpreter.can_execute(undeclared) is False


def test_can_execute_refuses_mixed_device_operands() -> None:
    """Operands that disagree on device are refused even when the stamped
    token resolves: one binding runs on one device, so there is no device to
    compile it for. `CustomOp._op_device` refuses this at staging, so the
    op is hand-stamped here -- the guard exists for a graph that reached the
    interpreter by some other route (a replay, a rewrite)."""
    defn = make_q8_like()
    a_t = TensorType(DType.float32, ["m", "k"], DeviceRef.CPU(0))
    b_t = TensorType(DType.float32, ["n", "k"], DeviceRef.CPU(1))
    graph = Graph(
        "g_mixed_device",
        input_types=[a_t, b_t],
        custom_extensions=[KERNELS],
    )
    with graph:
        (out,) = ops.custom(
            "mxf353_q8_like",
            DeviceRef.CPU(0),
            list(graph.inputs),
            out_types=[TensorType(DType.float32, ["m", "n"], DeviceRef.CPU(0))],
        )
        graph.output(out)
    staged = out._mlir_value.owner
    assert isinstance(staged, mo.CustomOp)
    staged.discardable_attributes[C._SPEC_ATTR_KEY] = builtin.StringAttr(
        defn.token
    )
    # The def resolves; only the device disagreement refuses the graph.
    assert _handlers._resolve_spec(staged) is defn
    assert _interpreter.can_execute(graph) is False


def test_undeclared_custom_op_compiles_instead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom op with no ``CustomOp``, staged via ``F.custom`` directly,
    has no shape contract to build a binding from: the interpreter refuses it
    and the composite executor compiles it, still computing the right answer
    and never touching the binding cache. Forcing the interpreter surfaces
    that refusal as an ``UnsupportedGraphError`` rather than a wrong result."""
    _assert_values([8.0, 8.0, 8.0, 8.0], _call_scale(4))
    assert len(custom_gc._CACHE) == 0

    monkeypatch.setattr(
        executor,
        "_DEFAULT_EXECUTOR",
        executor.InterpreterExecutor(max_ops=None),
    )
    with pytest.raises(
        executor.UnsupportedGraphError,
        match=r"no custom\.declare declaration",
    ):
        _call_scale(4)


def test_undeclared_custom_op_never_replays_a_mutation() -> None:
    """A mutation staged ahead of an undeclared custom op must apply exactly
    once. Both ops land in one graph, and the compile fallback reruns the
    WHOLE graph -- so an interpreter that accepted this graph and only
    discovered the undeclared op part-way through would leave the buffer
    incremented twice. The refusal is static instead, and the compiled run
    is the only run."""
    buf = Buffer.zeros((4,), DType.float32, CPU())
    with rc.EagerRealizationContext() as ctx, realization_context(ctx):
        target = Tensor(storage=buf)
        F.buffer_store_slice(target, target[0:4] + 1.0, [slice(0, 4)])
        (scaled,) = F.custom(
            "mxf353_scale",
            device=CPU(),
            values=[_full(2.0, 4)],
            out_types=[_f32(4)],
            parameters={"factor": 4},
            custom_extensions=KERNELS,
        )

    _assert_values([1.0, 1.0, 1.0, 1.0], Tensor(storage=buf))
    _assert_values([8.0, 8.0, 8.0, 8.0], scaled)


# --- _check_realized_shape: rank/dtype/device/shape guard -------------------

_GRAPH_COUNTER = itertools.count()


def _staged_result_value(shape: list[int]) -> _core.Value[Any]:
    """Builds a lone graph input/output value of the given staged shape, for
    testing ``_check_realized_shape`` directly without staging a real custom
    op end to end."""
    g = Graph(
        f"g_shape_probe_{next(_GRAPH_COUNTER)}", input_types=[_f32(*shape)]
    )
    with g:
        g.output(g.inputs[0])
    return g.inputs[0]._mlir_value


_REALIZED_SHAPE_MISMATCHES: list[tuple[str, str, Callable[[], Buffer]]] = [
    (
        "rank_too_few_dims",  # old code: confusing IndexError (out.shape[i] OOB)
        "rank mismatch",
        lambda: Buffer.zeros((2, 4), DType.float32, CPU()),
    ),
    (
        "rank_too_many_dims",  # old code: passed silently (loop only walked staged dims)
        "rank mismatch",
        lambda: Buffer.zeros((2, 4, 3, 1), DType.float32, CPU()),
    ),
    (
        "dtype_mismatch",  # old code: dtype was never checked at all
        "dtype mismatch",
        lambda: Buffer.zeros((2, 4, 3), DType.int32, CPU()),
    ),
    (
        "dim_mismatch",  # pre-existing per-dim check, reached after rank/dtype agree
        "shape mismatch",
        lambda: Buffer.zeros((2, 5, 3), DType.float32, CPU()),
    ),
]


@pytest.mark.parametrize(
    "match,realized",
    [row[1:] for row in _REALIZED_SHAPE_MISMATCHES],
    ids=[row[0] for row in _REALIZED_SHAPE_MISMATCHES],
)
def test_check_realized_shape_raises(
    match: str, realized: Callable[[], Buffer]
) -> None:
    """A hard ``RuntimeError``, deliberately not an ``UnsupportedGraphError``:
    a kernel contradicting its own staged types is a contract violation, and
    a mid-execution refusal would let the compile fallback replay whatever
    the graph had already mutated."""
    staged = _staged_result_value([2, 4, 3])
    with pytest.raises(RuntimeError, match=match) as excinfo:
        _handlers._check_realized_shape(realized(), staged)
    assert not isinstance(excinfo.value, executor.UnsupportedGraphError)


# --- Signature grammar: Symbol/TemplateType, unification, dtype variables ---


def test_signature_symbols_are_namespaced_and_idempotent() -> None:
    """A signature symbol carries a reserved name so it can never be confused
    with a caller's dim, and re-wrapping it does not prefix twice.

    `Dim(x)` returns the same instance and Python re-runs `__init__` on it.
    """
    rows, k = C.Symbols("rows", "k")
    assert rows.name == "__co_rows"
    assert rows.symbol_name == "rows"
    rewrapped = Dim(rows)
    assert isinstance(rewrapped, C.Symbol)
    assert rewrapped.name == "__co_rows"
    assert C.Symbol(rows).name == "__co_rows"
    # Participates in the ordinary dim algebra.
    assert str(k // 2) == "__co_k // 2"
    # A caller's dim of the same spelling is a different symbol.
    assert rows != Dim("rows")


def test_template_type_takes_dtype_var_and_no_device() -> None:
    """`TemplateType` describes one side of a signature: a dtype that may be a
    variable, and a shape of signature dims. Device is a call-time property."""
    rows, k = C.Symbols("rows", "k")
    spec = C.TemplateType(C.DTypeVar("T"), [rows, k // 2])
    assert spec.dtype == C.DTypeVar("T")
    assert [str(d) for d in spec.shape] == ["__co_rows", "__co_k // 2"]
    concrete = C.TemplateType(DType.uint8, [4])
    assert concrete.dtype == DType.uint8
    not_a_dtype: Any = "T"  # bare strings are no longer dtype vars
    with pytest.raises(TypeError):
        C.TemplateType(not_a_dtype, [rows])


def test_template_type_is_immutable() -> None:
    """A signature is validated once at declaration, so an entry cannot be
    edited underneath the op afterwards: no attribute or shape mutation."""
    (rows,) = C.Symbols("rows")
    spec = C.TemplateType(DType.float32, [rows, 4])
    assert spec.shape == (rows, 4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.dtype = DType.int8  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.shape[1] = 2  # type: ignore[index]


def test_custom_op_is_immutable() -> None:
    """A declared op is a frozen value: nothing about it can change after
    `declare` validated it, and two identical declarations compare equal."""
    op = make_downsample()
    with pytest.raises(dataclasses.FrozenInstanceError):
        op.name = "other"  # type: ignore[misc]
    assert op == make_downsample()
    assert op != make_copy()


def test_source_extensions_compile_at_declaration() -> None:
    """A Mojo source package is compiled once when the op is declared, so a
    call only registers the binary instead of rerunning `mojo precompile`."""
    op = make_downsample()
    assert all(p.suffix == ".mojoc" for p in op.extensions)


def _quantize_signature() -> C.CustomOp:
    """A two-output def: a dtype change and two shape changes."""
    rows, k = C.Symbols("rows", "k")
    return C.declare(
        "mxf353_nvfp4_quantize",
        inputs={"x": C.TemplateType(DType.float32, [rows, k])},
        outputs=[
            C.TemplateType(DType.uint8, [rows, k // 2]),
            C.TemplateType(DType.float32, [rows, k // 16]),
        ],
        custom_extensions=[KERNELS],
    )


def test_declared_signature_is_stored_and_checked() -> None:
    """An op keeps its declared signature, and an output dim that is not a
    signature dim is refused at definition rather than at a call."""
    op = _quantize_signature()
    assert list(op.inputs) == ["x"]
    assert [str(d) for d in op.outputs[0].shape] == [
        "__co_rows",
        "__co_k // 2",
    ]

    rows, k = C.Symbols("rows", "k")
    with pytest.raises(TypeError, match="not a signature dim"):
        C.declare(
            "mxf353_nvfp4_quantize",
            inputs={"x": C.TemplateType(DType.float32, [rows, k])},
            # `Dim("cols")` belongs to no signature.
            outputs=[C.TemplateType(DType.uint8, [rows, Dim("cols")])],
            custom_extensions=[KERNELS],
        )


def test_repeated_symbol_is_equality_constraint() -> None:
    """A plain symbol repeated across two input templates ("k" in both) is a
    cross-input equality constraint: a mismatched call raises naming both
    values; a matched call runs the shared-dim contraction correctly."""
    op = make_q8_like()
    with pytest.raises(
        ValueError, match=r"input 'b'.*equal input 'a'.*expected 512, got 256"
    ):
        op(_zeros(4, 512), _zeros(8, 256))

    out = _one_tensor(op(_full(1.0, 2, 4), _full(2.0, 3, 4)))
    assert [int(d) for d in out.shape] == [2, 3]
    # out[i, j] = sum_l x[i, l] * y[j, l] = 4 * (1.0 * 2.0) = 8.0 everywhere.
    _assert_values([[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]], out)


def test_dtype_var_correlates() -> None:
    op = make_q8_like()
    x = _full(1.0, 2, 4)
    y_ok = Tensor.full([3, 4], 1.0, dtype=DType.float32, device=CPU())
    op(x, y_ok)  # f32 + f32: dtype variable "A" binds once, agrees, no raise.

    y_bad = Tensor.full([3, 4], 1.0, dtype=DType.float16, device=CPU())
    with pytest.raises(ValueError, match=r"input 'b'.*float32.*float16"):
        op(x, y_bad)


def test_unify_binds_symbols_and_enforces_agreement() -> None:
    """Unification binds each signature symbol to an actual dim, requires a
    repeated symbol to agree, and accepts an algebraic actual."""
    m, n, k = C.Symbols("m", "n", "k")
    T = C.DTypeVar("T")
    op = C.declare(
        "mxf353_q8_like",
        inputs={
            "a": C.TemplateType(T, [m, k]),
            "b": C.TemplateType(T, [n, k]),
        },
        outputs=[C.TemplateType(T, [m, n])],
        custom_extensions=[KERNELS],
    )
    bindings = op._unify([_f32(2, 8), _f32(3, 8)])
    assert str(bindings["__co_m"]) == "2"
    assert str(bindings["__co_k"]) == "8"

    # An algebraic actual binds like any other dim.
    algebraic = TensorType(DType.float32, [Dim("w") // 2, 8], DeviceRef.CPU())
    assert str(op._unify([algebraic, _f32(3, 8)])["__co_m"]) == "w // 2"

    with pytest.raises(ValueError, match="must equal"):
        op._unify([_f32(2, 8), _f32(3, 9)])
    with pytest.raises(ValueError, match="rank"):
        op._unify([_f32(2), _f32(3, 8)])
    with pytest.raises(ValueError, match="dtype"):
        op._unify(
            [
                _f32(2, 8),
                TensorType(DType.uint8, [3, 8], DeviceRef.CPU()),
            ]
        )


def test_static_input_dim_must_match() -> None:
    """A static in an input template is a hard constraint, not a hint."""
    (c,) = C.Symbols("c")
    op = C.declare(
        "mxf353_downsample",
        inputs={"x": C.TemplateType(DType.float32, [2, 8, c])},
        outputs=[C.TemplateType(DType.float32, [2, 4, c])],
        custom_extensions=[KERNELS],
    )
    _assert_values([[[0.0] * 3] * 4] * 2, _one_tensor(op(_zeros(2, 8, 3))))
    with pytest.raises(ValueError, match=r"dim 1 must equal static 8: got 6"):
        op(_zeros(2, 6, 3))


def test_output_dtype_var_must_be_bound_by_an_input() -> None:
    (m,) = C.Symbols("m")
    op = C.declare(
        "mxf353_downsample",
        inputs={"x": C.TemplateType(DType.float32, [m])},
        outputs=[C.TemplateType(C.DTypeVar("U"), [m])],
        custom_extensions=[KERNELS],
    )
    with pytest.raises(TypeError, match=r"'U' is bound by no input"):
        op(_zeros(4))


# --- Definition/call grammar errors -----------------------------------------

# Every entry here is a definition- or call-time refusal, so the shared
# `mxf353_downsample` inputs below never reach a kernel; the call-shaped rows
# (arity, mixing eager and staged values) raise there instead.
_DS_N, _DS_W, _DS_C = C.Symbols("n", "w", "c")
_DS_INPUTS = {"x": C.TemplateType(DType.float32, [_DS_N, _DS_W, _DS_C])}


def _ds(
    outputs: Sequence[C.TemplateType],
    inputs: Mapping[str, C.TemplateType] | None = None,
) -> C.CustomOp:
    return C.declare(
        "mxf353_downsample",
        inputs=_DS_INPUTS if inputs is None else inputs,
        outputs=outputs,
        custom_extensions=[KERNELS],
    )


def _mixed_args_call() -> object:
    """Calls a two-input def with one graph ``TensorValue`` and one eager
    ``Tensor``. The eager tensor is built outside the graph build, since that
    is the only place it could legitimately come from."""
    m, n = C.Symbols("m", "n")
    op = C.declare(
        "mxf353_q8_like",
        inputs={
            "a": C.TemplateType(DType.float32, [m]),
            "b": C.TemplateType(DType.float32, [n]),
        },
        outputs=[C.TemplateType(DType.float32, [m, n])],
        custom_extensions=[KERNELS],
    )
    eager = _zeros(4)
    g = Graph(
        "g_mixed_args", input_types=[_f32(4)], custom_extensions=[KERNELS]
    )
    with g:
        return op(g.inputs[0].tensor, eager)


def _algebraic_input_dim() -> C.CustomOp:
    """An input dim that is an expression over a real signature symbol
    (``m // 2``), not a foreign name: legal in an output, but `_unify` has
    no actual dim to solve ``m`` from, so this must be refused at
    definition rather than tripping an assertion at the first call."""
    (m,) = C.Symbols("m")
    return C.declare(
        "mxf353_downsample",
        inputs={"x": C.TemplateType(DType.float32, [m // 2])},
        outputs=[C.TemplateType(DType.float32, [m // 2])],
        custom_extensions=[KERNELS],
    )


_GRAMMAR_ERRORS: list[
    tuple[str, type[Exception], str, Callable[[], object]]
] = [
    (
        "call_arity_exceeds_declared_inputs",
        TypeError,
        r"'mxf353_downsample' expects 1 inputs, got 2",
        lambda: make_downsample()(_zeros(2, 8, 3), _zeros(2, 8, 3)),
    ),
    (
        "zero_inputs",
        ValueError,
        "at least one input",
        lambda: C.declare(
            "mxf353_downsample",
            inputs={},
            outputs=[C.TemplateType(DType.float32, [1])],
            custom_extensions=[KERNELS],
        ),
    ),
    (
        "zero_outputs",
        ValueError,
        "at least one output",
        lambda: C.declare(
            "mxf353_downsample",
            inputs=_DS_INPUTS,
            outputs=[],
            custom_extensions=[KERNELS],
        ),
    ),
    (
        "mixed_eager_and_graph_arguments",
        TypeError,
        "all eager Tensors or all graph TensorValues",
        _mixed_args_call,
    ),
    (
        "algebraic_dim_in_input_template",
        TypeError,
        "not a signature dim",
        lambda: C.declare(
            "mxf353_downsample",
            inputs={"x": C.TemplateType(DType.float32, [Dim("m") + 1])},
            outputs=[C.TemplateType(DType.float32, [Dim("m") + 1])],
            custom_extensions=[KERNELS],
        ),
    ),
    (
        "algebraic_input_dim_is_not_bindable",
        TypeError,
        "not directly bindable",
        _algebraic_input_dim,
    ),
    (
        # `Dim("cc")` occupies the position a signature symbol could legally
        # hold, but only the declaration mechanism, not the spelling, makes a
        # dim a signature dim.
        "foreign_dim_typo_at_definition",
        TypeError,
        r"cc.*not a signature dim",
        lambda: _ds([C.TemplateType(DType.float32, [Dim("cc"), _DS_C])]),
    ),
    (
        # `Dim("n")` spells the bound input symbol "n" yet is still foreign:
        # the guard matches by declaration, not by name.
        "foreign_dim_typo_matches_by_declaration_not_name",
        TypeError,
        r"n.*not a signature dim",
        lambda: _ds([C.TemplateType(DType.float32, [Dim("n"), _DS_W, _DS_C])]),
    ),
    (
        # The same refusal on an input dim rather than an output one.
        "foreign_dim_typo_in_input_at_definition",
        TypeError,
        r"n.*not a signature dim",
        lambda: _ds(
            [C.TemplateType(DType.float32, [Dim("n")])],
            inputs={"x": C.TemplateType(DType.float32, [Dim("n")])},
        ),
    ),
    (
        "foreign_symbol_inside_algebraic_out_dim",
        TypeError,
        r"cc.*not a signature dim",
        lambda: _ds([C.TemplateType(DType.float32, [Dim("cc") + 1, _DS_C])]),
    ),
]


@pytest.mark.parametrize(
    "exc,match,action",
    [row[1:] for row in _GRAMMAR_ERRORS],
    ids=[row[0] for row in _GRAMMAR_ERRORS],
)
def test_grammar_errors(
    exc: type[Exception], match: str, action: Callable[[], object]
) -> None:
    with pytest.raises(exc, match=match):
        action()


def test_grammar_accepts_valid_definitions() -> None:
    """Negative controls for the table above: constructs that must NOT
    raise, asserted directly so a stricter guard can't silently break them."""
    n, w, c = C.Symbols("n", "w", "c")
    C.declare(  # w // 2 over a bound signature symbol is algebraic but legitimate.
        "mxf353_downsample",
        inputs={"x": C.TemplateType(DType.float32, [n, w, c])},
        outputs=[C.TemplateType(DType.float32, [n, w // 2, c])],
        custom_extensions=[KERNELS],
    )
    # A static input dim alongside a symbolic one is equally legitimate.
    C.declare(
        "mxf353_downsample",
        inputs={"x": C.TemplateType(DType.float32, [2, w, c])},
        outputs=[C.TemplateType(DType.float32, [2, w // 2, c])],
        custom_extensions=[KERNELS],
    )
    x = Tensor.arange(24, dtype=DType.float32, device=CPU()).reshape([2, 4, 3])
    _assert_values(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
        ],
        _one_tensor(make_downsample()(x)),
    )


def test_incoming_dim_in_reserved_namespace_is_refused() -> None:
    """A caller may name a dim like a declared signature symbol; the
    ambiguity is refused at the boundary rather than resolved."""
    rows, cols = C.Symbols("rows", "cols")
    op = C.declare(
        "mxf353_downsample",
        inputs={"x": C.TemplateType(DType.float32, [rows, cols])},
        outputs=[C.TemplateType(DType.float32, [rows, cols])],
        custom_extensions=[KERNELS],
    )
    graph = Graph(
        "g_reserved",
        input_types=[_f32("__co_rows", 4)],
        custom_extensions=[KERNELS],
    )
    with pytest.raises(TypeError, match="reserved"):
        with graph:
            op(graph.inputs[0].tensor)


# --- Signature synthesis + F.functional composition -------------------------


def test_signature_and_functional_composition_scenario() -> None:
    """`inspect.signature` synthesis (the declared `inputs` keys, not
    `__call__`'s bare `*args`) and `F.functional` composition -- single-device
    parity plus SPMD dispatch across a CPU-simulated 2-device mesh -- against
    the same declared signatures."""
    downsample_sig = inspect.signature(make_downsample())
    assert list(downsample_sig.parameters) == ["x"]
    assert downsample_sig.parameters["x"].annotation == Tensor | TensorValue
    assert (
        downsample_sig.return_annotation
        == Tensor | list[Tensor] | TensorValue | list[TensorValue]
    )

    multi_arg_sig = inspect.signature(make_q8_like())
    assert list(multi_arg_sig.parameters) == ["a", "b"]
    assert all(
        p.annotation == Tensor | TensorValue
        for p in multi_arg_sig.parameters.values()
    )

    op = make_copy()
    wrapped = F.functional(op)
    x = _full(2.0, 2, 4, 3)
    _assert_values(_one_tensor(op(x)), _one_tensor(wrapped(x)))
    wrapped_sig = inspect.signature(wrapped)
    assert list(wrapped_sig.parameters) == ["x"]
    # `F.functional` passes parameter annotations through, but its wrapper
    # is eager-only, so it rewrites `TensorValue` out of the return.
    assert wrapped_sig.parameters["x"].annotation == Tensor | TensorValue
    assert wrapped_sig.return_annotation == Tensor | list[Tensor]

    # Whatever `__signature__` advertises, `__call__` must accept: these
    # parameters are keyword-capable, and `F.functional`'s distributed path
    # binds a keyword call through this same signature before dispatching per
    # shard, so a keyword-refusing `__call__` would make one call succeed or
    # fail purely on whether its arguments were sharded.
    q8 = make_q8_like()
    assert all(
        p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for p in inspect.signature(q8).parameters.values()
    )
    a, b = _full(1.0, 2, 4), _full(2.0, 3, 4)
    contracted = [[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]]
    _assert_values(contracted, _one_tensor(q8(a=a, b=b)))
    _assert_values(contracted, _one_tensor(q8(a, b=b)))
    # Keywords bind into declared operand order, not call order.
    _assert_values(contracted, _one_tensor(q8(b=b, a=a)))
    with pytest.raises(
        TypeError, match=r"mxf353_q8_like.*unexpected keyword argument 'c'"
    ):
        q8(a=a, b=b, c=b)
    _assert_values(_one_tensor(op(x)), _one_tensor(wrapped(x=x)))

    mesh = DeviceMesh(
        devices=(CPU(), CPU()), mesh_shape=(2,), axis_names=("tp",)
    )
    sharded_op = F.functional(op, rule=unary_rule)
    sharded_x = F.transfer_to(
        _full(2.0, 2, 4, 3), PlacementMapping(mesh, (Sharded(0),))
    )
    result = sharded_op(sharded_x)
    assert result.is_distributed
    assert result.placements == (Sharded(0),)
    # A sharded result has no single device for `assert_all_close` to
    # convert against, so each shard is checked on its own.
    assert len(result.local_shards) == 2
    for shard in result.local_shards:
        _assert_values([[[2.0] * 3] * 4], shard)
