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
"""Tests for `max.experimental.custom.declare` and `CustomOp` staging."""
# MXF-353

import dataclasses
import inspect
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental import custom as C
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh, PlacementMapping, Sharded
from max.experimental.sharding.rules import unary_rule
from max.experimental.tensor import Tensor
from max.experimental.testing import assert_all_close
from max.graph import (
    DeviceRef,
    Dim,
    DimLike,
    Graph,
    TensorType,
    TensorValue,
)

KERNELS = Path(os.environ["MODULAR_CUSTOM_OP_DEF_KERNELS_PATH"])

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
