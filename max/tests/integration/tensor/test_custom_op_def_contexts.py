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
"""Staging contexts and allocated-dim-name collisions for
`max.experimental.custom.declare`.

The sibling suite (``test_custom_op_def.py``) covers eager calls, one top-level
`Graph`, and the declaration grammar. This one covers where else a def can be
staged -- subgraphs, region-carrying ops, several graphs in one module -- and
what happens when an allocated data-dependent dim name collides with a name the
graph already holds.

Every collision case here is asserted on realized shapes and values after
execution, not on IR text: a wrong staged type that the runtime resizes anyway
is not a bug, and a wrong realized shape is one whether or not the IR looked
suspicious.
"""
# MXF-353

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from max import engine
from max._interpreter_ops import custom_gc
from max.driver import CPU
from max.dtype import DType
from max.experimental import custom as C
from max.experimental.tensor import Tensor
from max.experimental.testing import assert_all_close
from max.graph import DeviceRef, DimLike, Graph, TensorType, TensorValue, ops

KERNELS = Path(os.environ["MODULAR_CUSTOM_OP_DEF_KERNELS_PATH"])

#: Rank-2 fp32 input whose first column is positive in exactly two rows, so
#: `mxf353_filter` selects two.
TWO_OF_FOUR = [[1.0, 2.0], [-1.0, 5.0], [3.0, 4.0], [-2.0, 9.0]]
#: Same shape, three selected rows: enough to tell one call's answer from
#: another's when both carry the same dim name.
THREE_OF_FOUR = [[1.0, 2.0], [2.0, 5.0], [3.0, 4.0], [-2.0, 9.0]]


def _f32(*dims: DimLike) -> TensorType:
    return TensorType(DType.float32, list(dims), DeviceRef.CPU())


def _tensor(values: Any) -> Tensor:
    return Tensor(values, dtype=DType.float32, device=CPU())


def _one_value(result: Any) -> TensorValue:
    assert isinstance(result, TensorValue)
    return result


def _assert_values(expected: Any, actual: Tensor) -> None:
    """Asserts *actual* has the shape and values of the nested *expected*.

    Mirrors the sibling suite's helper: `assert_all_close` reduces with
    `max(axis=-1)`, so comparing flattened copies keeps any rank on that path
    and the shape is checked separately rather than implied.
    """
    want = _tensor(expected)
    assert [int(d) for d in want.shape] == [int(d) for d in actual.shape]
    count = actual.num_elements()
    assert_all_close(want.reshape([count]), actual.reshape([count]))


def _run(graph: Graph, *inputs: Tensor) -> list[Tensor]:
    model = engine.InferenceSession(devices=[CPU()]).load(graph)
    return [
        Tensor(storage=buffer)
        for buffer in model.execute(*[t.driver_tensor for t in inputs])
    ]


def _allocated(
    graph_name: str, kernel: str, symbol: str, ordinal: int = 0
) -> str:
    """Returns the dim name *kernel*'s *ordinal*-th staging allocates in a graph
    named *graph_name*, computed through the allocator itself.

    An allocated name carries a digest, so a test cannot spell one out.
    Allocating on a throwaway graph disturbs nothing: the ordinal sequence
    lives on the `Graph` object, so this object's sequence is its own.
    """
    (dim,) = C.Symbols(symbol)
    probe = Graph(graph_name, input_types=[_f32(1)])
    for _ in range(ordinal):
        C._allocate_unbound_dim(dim, {}, probe, kernel)
    return str(C._allocate_unbound_dim(dim, {}, probe, kernel))


def _scale(factor: int) -> C.CustomOp:
    """Shape-preserving rank-2 `x * factor`."""
    rows, cols = C.Symbols("rows", "cols")
    return C.declare(
        "mxf353_scale",
        inputs={"x": C.TemplateType(DType.float32, [rows, cols])},
        outputs=[C.TemplateType(DType.float32, [rows, cols])],
        parameters={"factor": factor},
        custom_extensions=[KERNELS],
    )


def _filter() -> C.CustomOp:
    """Data-dependent: keeps the rows whose first column is positive, so the
    output row count is known only by running the kernel's shape function."""
    rows, cols, kept = C.Symbols("rows", "cols", "kept")
    return C.declare(
        "mxf353_filter",
        inputs={"x": C.TemplateType(DType.float32, [rows, cols])},
        outputs=[C.TemplateType(DType.float32, [kept, cols])],
        custom_extensions=[KERNELS],
    )


@pytest.fixture(autouse=True)
def _clear_binding_cache() -> Iterator[None]:
    """Mirrors the sibling suites: `_CACHE` is process-global, so a hit left by
    an earlier test would hide a real miss."""
    custom_gc._CACHE.clear()
    yield
    custom_gc._CACHE.clear()


# --- Staging contexts -------------------------------------------------------


def test_subgraph_stages_and_executes() -> None:
    """A def inside a subgraph called from its parent computes the right
    values, not merely a graph that compiles."""
    op = _scale(3)
    graph = Graph(
        "g_subgraph_static",
        input_types=[_f32("r", "c")],
        custom_extensions=[KERNELS],
    )
    with graph:
        sub = graph.add_subgraph(
            "sub_static",
            input_types=[_f32("r", "c")],
            custom_extensions=[KERNELS],
        )
        with sub:
            sub.output(_one_value(op(sub.inputs[0].tensor)))
        (out,) = ops.call(sub, graph.inputs[0])
        graph.output(out)

    (got,) = _run(graph, _tensor([[1.0, 2.0], [3.0, 4.0]]))
    _assert_values([[3.0, 6.0], [9.0, 12.0]], got)


def test_subgraph_data_dependent_output_crosses_the_boundary() -> None:
    """A data-dependent result allocated inside a subgraph reaches the parent
    through `ops.call` and is sized by the kernel's shape function at run
    time.

    The allocated name carries the SUBGRAPH's name, since the ordinal and the
    embedded graph fragment both come from `Graph.current`, which inside a
    subgraph build is the subgraph.
    """
    op = _filter()
    graph = Graph(
        "g_subgraph_dynamic",
        input_types=[_f32("r", "c")],
        custom_extensions=[KERNELS],
    )
    with graph:
        sub = graph.add_subgraph(
            "sub_dynamic",
            input_types=[_f32("r", "c")],
            custom_extensions=[KERNELS],
        )
        with sub:
            inner = _one_value(op(sub.inputs[0].tensor))
            allocated = str(inner.shape[0])
            sub.output(inner)
        (out,) = ops.call(sub, graph.inputs[0])
        assert str(out.tensor.shape[0]) == allocated
        graph.output(out)

    assert allocated == _allocated("sub_dynamic", "mxf353_filter", "kept")
    (got,) = _run(graph, _tensor(TWO_OF_FOUR))
    _assert_values([[1.0, 2.0], [3.0, 4.0]], got)


def test_one_data_dependent_subgraph_called_twice_sizes_each_call() -> None:
    """Repeated calls are the whole point of a subgraph, and both call results
    carry the SAME allocated dim name -- the subgraph allocates once, at build
    time.

    Each call site must still be sized by its own operand, so the two results
    have different row counts despite the shared name. `mo.call` matches the
    callee signature by dim name, hence the statically-shaped input type.
    """
    op = _filter()
    graph = Graph(
        "g_subgraph_twice",
        input_types=[_f32(4, 2), _f32(4, 2)],
        custom_extensions=[KERNELS],
    )
    with graph:
        sub = graph.add_subgraph(
            "sub_twice", input_types=[_f32(4, 2)], custom_extensions=[KERNELS]
        )
        with sub:
            sub.output(_one_value(op(sub.inputs[0].tensor)))
        (first,) = ops.call(sub, graph.inputs[0])
        (second,) = ops.call(sub, graph.inputs[1])
        assert str(first.tensor.shape[0]) == str(second.tensor.shape[0])
        graph.output(first, second)

    got_first, got_second = _run(
        graph, _tensor(TWO_OF_FOUR), _tensor(THREE_OF_FOUR)
    )
    _assert_values([[1.0, 2.0], [3.0, 4.0]], got_first)
    _assert_values([[1.0, 2.0], [2.0, 5.0], [3.0, 4.0]], got_second)


def test_data_dependent_dim_flows_into_a_subgraph() -> None:
    """The other direction: an allocated dim produced in the parent is the
    declared input dim of a subgraph the parent then calls."""
    filter_op, scale_op = _filter(), _scale(2)
    graph = Graph(
        "g_into_subgraph",
        input_types=[_f32("r", "c")],
        custom_extensions=[KERNELS],
    )
    with graph:
        routed = _one_value(filter_op(graph.inputs[0].tensor))
        allocated = str(routed.shape[0])
        assert allocated.startswith(C._DYN_PREFIX)
        sub = graph.add_subgraph(
            "sub_consumer",
            input_types=[_f32(allocated, "c")],
            custom_extensions=[KERNELS],
        )
        with sub:
            sub.output(_one_value(scale_op(sub.inputs[0].tensor)))
        (out,) = ops.call(sub, routed)
        graph.output(out)

    (got,) = _run(graph, _tensor(TWO_OF_FOUR))
    _assert_values([[2.0, 4.0], [6.0, 8.0]], got)


def test_def_staged_in_both_parent_and_subgraph() -> None:
    """One def used on both sides of a subgraph boundary: the parent's result
    feeds the subgraph, which applies the same op again."""
    op = _scale(2)
    graph = Graph(
        "g_both_sides",
        input_types=[_f32("r", "c")],
        custom_extensions=[KERNELS],
    )
    with graph:
        outer = _one_value(op(graph.inputs[0].tensor))
        sub = graph.add_subgraph(
            "sub_both",
            input_types=[_f32("r", "c")],
            custom_extensions=[KERNELS],
        )
        with sub:
            sub.output(_one_value(op(sub.inputs[0].tensor)))
        (out,) = ops.call(sub, outer)
        graph.output(out)

    (got,) = _run(graph, _tensor([[1.0, 2.0], [3.0, 4.0]]))
    _assert_values([[4.0, 8.0], [12.0, 16.0]], got)


def test_def_in_a_cond_branch() -> None:
    """`ops.cond` builds its branches as blocks of the SAME graph, so a def
    staged in a branch shares the parent's dim namespace and ordinal
    sequence."""
    op = _scale(5)
    graph = Graph(
        "g_cond", input_types=[_f32(2, 2)], custom_extensions=[KERNELS]
    )
    with graph:
        x = graph.inputs[0].tensor
        pred = ops.constant(True, DType.bool, device=DeviceRef.CPU())
        graph.output(*ops.cond(pred, [_f32(2, 2)], lambda: op(x), lambda: x))

    (got,) = _run(graph, _tensor([[1.0, 2.0], [3.0, 4.0]]))
    _assert_values([[5.0, 10.0], [15.0, 20.0]], got)


def test_data_dependent_result_cannot_leave_a_cond_branch() -> None:
    """`ops.cond` fixes its result types before either branch runs, so an
    allocated dim has nowhere to go. Refused with both types printed, which is
    enough for a user to see the allocated name and reach for `ops.rebind`.
    """
    op = _filter()
    graph = Graph(
        "g_cond_dynamic", input_types=[_f32(4, 2)], custom_extensions=[KERNELS]
    )
    with pytest.raises(TypeError, match=r"__dyn_g_cond_dynamic_mxf353_filter"):
        with graph:
            x = graph.inputs[0].tensor
            pred = ops.constant(True, DType.bool, device=DeviceRef.CPU())
            ops.cond(
                pred,
                [_f32("declared_rows", 2)],
                lambda: op(x),
                lambda: op(x),
            )


def test_data_dependent_result_leaves_a_cond_branch_via_rebind() -> None:
    """The workaround the refusal above points at: assert the count the branch
    will produce, and the allocated dim is gone by the time `cond` checks
    types."""
    op = _filter()
    graph = Graph(
        "g_cond_rebind", input_types=[_f32(4, 2)], custom_extensions=[KERNELS]
    )
    with graph:
        x = graph.inputs[0].tensor
        pred = ops.constant(True, DType.bool, device=DeviceRef.CPU())
        graph.output(
            *ops.cond(
                pred,
                [_f32(2, 2)],
                lambda: ops.rebind(_one_value(op(x)), [2, 2]),
                lambda: ops.rebind(_one_value(op(x)), [2, 2]),
            )
        )

    (got,) = _run(graph, _tensor(TWO_OF_FOUR))
    _assert_values([[1.0, 2.0], [3.0, 4.0]], got)


def test_def_in_a_while_loop_body() -> None:
    """A shape-preserving def in a `while_loop` body runs once per iteration."""
    op = _scale(2)
    graph = Graph(
        "g_while", input_types=[_f32(2, 2)], custom_extensions=[KERNELS]
    )
    with graph:
        x = graph.inputs[0].tensor
        counter = ops.constant(0, DType.int32, device=DeviceRef.CPU())
        results = ops.while_loop(
            (counter, x),
            lambda i, v: i < 3,
            lambda i, v: (i + 1, _one_value(op(v))),
        )
        graph.output(results[1])

    (got,) = _run(graph, _tensor([[1.0, 1.0], [1.0, 1.0]]))
    _assert_values([[8.0, 8.0], [8.0, 8.0]], got)


def test_same_def_in_two_graphs_of_one_module() -> None:
    """Several graphs commonly share one module. Each allocates under its own
    graph name, and `load_all` compiles both.

    `custom_extensions` has to be repeated at `load_all`: the module carries the
    ops but the kernel package is a per-load argument.
    """
    op = _filter()
    first = Graph(
        "g_module_first",
        input_types=[_f32("r", "c")],
        custom_extensions=[KERNELS],
    )
    with first:
        a = _one_value(op(first.inputs[0].tensor))
        first.output(a)
    second = Graph(
        "g_module_second",
        input_types=[_f32("r", "c")],
        custom_extensions=[KERNELS],
        module=first.module,
    )
    with second:
        b = _one_value(op(second.inputs[0].tensor))
        second.output(b)

    assert str(a.shape[0]) == _allocated(
        "g_module_first", "mxf353_filter", "kept"
    )
    assert str(b.shape[0]) == _allocated(
        "g_module_second", "mxf353_filter", "kept"
    )

    models = engine.InferenceSession(devices=[CPU()]).load_all(
        first.module, custom_extensions=[KERNELS]
    )
    got_a = Tensor(
        storage=models["g_module_first"].execute(
            _tensor(TWO_OF_FOUR).driver_tensor
        )[0]
    )
    got_b = Tensor(
        storage=models["g_module_second"].execute(
            _tensor(THREE_OF_FOUR).driver_tensor
        )[0]
    )
    _assert_values([[1.0, 2.0], [3.0, 4.0]], got_a)
    _assert_values([[1.0, 2.0], [2.0, 5.0], [3.0, 4.0]], got_b)


def test_chained_data_dependent_defs_mint_distinct_dims() -> None:
    """A data-dependent result feeding a second data-dependent op: the second
    staging takes the next ordinal rather than inheriting the first's dim."""
    op = _filter()
    graph = Graph(
        "g_chain", input_types=[_f32(4, 2)], custom_extensions=[KERNELS]
    )
    with graph:
        first = _one_value(op(graph.inputs[0].tensor))
        second = _one_value(op(first))
        assert str(first.shape[0]) != str(second.shape[0])
        graph.output(second)

    (got,) = _run(graph, _tensor(THREE_OF_FOUR))
    _assert_values([[1.0, 2.0], [2.0, 5.0], [3.0, 4.0]], got)


# --- Graph.copy ------------------------------------------------------------


def test_copy_refuses_a_graph_with_a_data_dependent_dim() -> None:
    """A copy keeps the graph's name and contents but would restart the
    data-dependent dim numbering, so an op staged into it could reuse a dim
    the copy already holds. Refused at `copy()`, naming the cause."""
    op = _filter()
    graph = Graph(
        "g_copy_dynamic",
        input_types=[_f32("r", "c")],
        custom_extensions=[KERNELS],
    )
    with graph:
        graph.output(_one_value(op(graph.inputs[0].tensor)))

    with pytest.raises(ValueError, match=r"cannot be copied.*data-dependent"):
        graph.copy()


def test_copy_allows_a_graph_whose_custom_ops_are_all_static() -> None:
    """Only data-dependent dims are numbered per graph, so a custom op whose
    output shape the inputs determine leaves `copy()` available."""
    op = _scale(3)
    graph = Graph(
        "g_copy_static",
        input_types=[_f32("r", "c")],
        custom_extensions=[KERNELS],
    )
    with graph:
        graph.output(_one_value(op(graph.inputs[0].tensor)))

    copied = graph.copy()
    assert copied.name == graph.name
    assert copied._module is not graph._module


# --- Allocated-name collisions -------------------------------------------------


def test_dim_name_fragment_is_lossy() -> None:
    """`_dim_name_fragment` maps every non-word character to `_`, so graph or
    kernel names differing only in those characters share one fragment. That is
    why the allocator hashes the RAW names and not the fragments."""
    assert C._dim_name_fragment("g.x") == C._dim_name_fragment("g_x")
    assert C._dim_name_fragment("ep.dispatch.fp8") == "ep_dispatch_fp8"


def test_allocated_names_survive_a_shared_fragment() -> None:
    """Two graph names that sanitize alike allocate different dims, because the
    digest covers the raw name. Asserted on the naming function because only
    one fixture kernel registers a shape function."""
    (symbol,) = C.Symbols("k")
    dotted = Graph("g.frag", input_types=[_f32(1)])
    underscored = Graph("g_frag", input_types=[_f32(1)])
    assert str(C._allocate_unbound_dim(symbol, {}, dotted, "k")) != str(
        C._allocate_unbound_dim(symbol, {}, underscored, "k")
    )


def test_allocated_name_fields_are_unambiguous() -> None:
    """A differently-split (graph, kernel) pair allocates different dims: the
    digest's fields are length-prefixed, so the `_` joining them in the
    readable part carries no meaning it has to escape."""
    (symbol,) = C.Symbols("k")
    graph_ab = Graph("a_b", input_types=[_f32(1)])
    graph_a = Graph("a", input_types=[_f32(1)])
    assert str(C._allocate_unbound_dim(symbol, {}, graph_ab, "c")) != str(
        C._allocate_unbound_dim(symbol, {}, graph_a, "b_c")
    )


def test_allocated_name_is_stable_across_processes() -> None:
    """The exact string a known input allocates, pinned. `hashlib`, not the
    builtin `hash()`: a per-process salt would make these names vary run to run,
    and `Graph._allocate_data_dependent_ordinal` documents why stable names
    are load-bearing for IR cache hits. A self-comparison inside one process cannot catch that."""
    (symbol,) = C.Symbols("kept")
    graph = Graph("g.frag", input_types=[_f32(1)])
    allocated = str(C._allocate_unbound_dim(symbol, {}, graph, "mxf353_filter"))
    assert allocated == "__dyn_g_frag_mxf353_filter_0_kept_65b13b9cb2e8152f"


def test_subgraph_named_like_its_parent_is_refused() -> None:
    """A subgraph may not reuse its parent's name: the call resolves the name to
    the parent graph, which is not a subgraph.

    This is what forecloses the identical-name half of the allocated-name
    collision. The diagnostic names the symptom, not the duplicate name.
    """
    op = _filter()
    graph = Graph(
        "g_selfnamed", input_types=[_f32(4, 2)], custom_extensions=[KERNELS]
    )
    with pytest.raises(ValueError, match="Only subgraphs can be called"):
        with graph:
            sub = graph.add_subgraph(
                "g_selfnamed",
                input_types=[_f32(4, 2)],
                custom_extensions=[KERNELS],
            )
            with sub:
                sub.output(_one_value(op(sub.inputs[0].tensor)))
            ops.call(sub, graph.inputs[0])


def test_parent_and_subgraph_with_one_fragment_execute_correctly() -> None:
    """A subgraph whose name only differs from its parent's in characters
    `_dim_name_fragment` erases still allocates its own dim name.

    `add_subgraph` unions the parent's params into the subgraph, so a shared
    name would leave `_set_output_param_decls` declaring nothing for the
    subgraph's own custom op and the graph would reach the compiler with a
    parameter reference nothing declares (an internal MEF-conversion complaint
    a user could not act on). The digest in the allocated name is what keeps the
    two apart.
    """
    op = _filter()
    graph = Graph(
        "g.frag",
        input_types=[_f32(4, 2), _f32(4, 2)],
        custom_extensions=[KERNELS],
    )
    with graph:
        parent_out = _one_value(op(graph.inputs[0].tensor))
        sub = graph.add_subgraph(
            "g_frag", input_types=[_f32(4, 2)], custom_extensions=[KERNELS]
        )
        with sub:
            inner = _one_value(op(sub.inputs[0].tensor))
            assert str(inner.shape[0]) != str(parent_out.shape[0])
            sub.output(inner)
        (called,) = ops.call(sub, graph.inputs[1])
        graph.output(parent_out, called)

    got_parent, got_called = _run(
        graph, _tensor(TWO_OF_FOUR), _tensor(THREE_OF_FOUR)
    )
    _assert_values([[1.0, 2.0], [3.0, 4.0]], got_parent)
    _assert_values([[1.0, 2.0], [2.0, 5.0], [3.0, 4.0]], got_called)


def test_sibling_subgraphs_with_one_fragment_execute_correctly() -> None:
    """Two SIBLING subgraphs whose names share a fragment: each allocates its
    own name, and each call is sized by its own operand.

    Sibling subgraphs each declare their own params, so this case executed
    correctly even while the names were shared; it stays here as the control
    for the parent/subgraph case above.
    """
    op = _filter()
    graph = Graph(
        "g_siblings",
        input_types=[_f32(4, 2), _f32(4, 2)],
        custom_extensions=[KERNELS],
    )
    with graph:
        first = graph.add_subgraph(
            "s.1", input_types=[_f32(4, 2)], custom_extensions=[KERNELS]
        )
        with first:
            a = _one_value(op(first.inputs[0].tensor))
            first.output(a)
        second = graph.add_subgraph(
            "s_1", input_types=[_f32(4, 2)], custom_extensions=[KERNELS]
        )
        with second:
            b = _one_value(op(second.inputs[0].tensor))
            second.output(b)
        assert str(a.shape[0]) != str(b.shape[0])
        (call_a,) = ops.call(first, graph.inputs[0])
        (call_b,) = ops.call(second, graph.inputs[1])
        graph.output(call_a, call_b)

    got_a, got_b = _run(graph, _tensor(TWO_OF_FOUR), _tensor(THREE_OF_FOUR))
    _assert_values([[1.0, 2.0], [3.0, 4.0]], got_a)
    _assert_values([[1.0, 2.0], [2.0, 5.0], [3.0, 4.0]], got_b)


def test_operand_dim_in_the_param_namespace_is_inert() -> None:
    """`__param_` is the third reserved namespace, and an operand dim spelled
    that way is harmless: a valued `Param` is folded to a static at definition
    and an unvalued one makes the def uncallable, so nothing ever resolves a
    caller's dim to a parameter."""
    op = _scale(3)
    graph = Graph(
        "g_param_ns",
        input_types=[_f32("__param_factor", 2)],
        custom_extensions=[KERNELS],
    )
    with graph:
        graph.output(_one_value(op(graph.inputs[0].tensor)))

    (got,) = _run(graph, _tensor([[1.0, 2.0], [3.0, 4.0]]))
    _assert_values([[3.0, 6.0], [9.0, 12.0]], got)


def test_operand_dim_in_the_dynamic_namespace_that_matches_nothing_is_inert() -> (
    None
):
    """Control for the aliasing tests below: a `__dyn_` operand dim that is not
    what this graph allocates changes nothing, so those tests are about the
    match, not about the namespace."""
    op = _filter()
    graph = Graph(
        "g_dyn_ns_control",
        input_types=[
            _f32(4, 2),
            _f32("__dyn_some_other_graph_mxf353_filter_0_kept"),
        ],
        custom_extensions=[KERNELS],
    )
    with graph:
        out = _one_value(op(graph.inputs[0].tensor))
        graph.output(out, graph.inputs[1])

    got, _ = _run(graph, _tensor(TWO_OF_FOUR), _tensor([1.0, 2.0, 3.0]))
    _assert_values([[1.0, 2.0], [3.0, 4.0]], got)


# --- Aliased allocated names -------------------------------------------------


def test_aliased_allocated_dim_is_refused() -> None:
    """A caller-authored dim spelled exactly like the dim a staging is about to
    allocate would replace the shape function's answer:
    `_set_output_param_decls` emits no declaration and the result is sized by
    the caller's dim, writing past the buffer whenever that dim is the
    smaller one. The allocator refuses it instead.
    """
    op = _filter()
    aliased = _allocated("g_alias", "mxf353_filter", "kept")
    graph = Graph(
        "g_alias",
        input_types=[_f32(4, 2), _f32(aliased)],
        custom_extensions=[KERNELS],
    )
    with pytest.raises(TypeError, match="already holds a dim"):
        with graph:
            op(graph.inputs[0].tensor)


def test_aliased_allocated_dim_at_a_later_ordinal_is_refused() -> None:
    """The alias need not hit the first staging: the guard considers every
    ordinal, not just zero, so the first staging succeeds and the second is the
    one refused."""
    op = _filter()
    aliased = _allocated("g_alias_ordinal", "mxf353_filter", "kept", ordinal=1)
    graph = Graph(
        "g_alias_ordinal",
        input_types=[_f32(4, 2), _f32(4, 2), _f32(aliased)],
        custom_extensions=[KERNELS],
    )
    with pytest.raises(TypeError, match="already holds a dim"):
        with graph:
            op(graph.inputs[0].tensor)
            op(graph.inputs[1].tensor)


def test_rebind_pins_a_data_dependent_result() -> None:
    """`ops.rebind` asserts at run time a count the compiler cannot infer, and
    stays the way to pin a data-dependent result to an expected one."""
    op = _filter()
    graph = Graph(
        "g_rebind_control",
        input_types=[_f32(4, 2)],
        custom_extensions=[KERNELS],
    )
    with graph:
        graph.output(ops.rebind(_one_value(op(graph.inputs[0].tensor)), [2, 2]))

    (got,) = _run(graph, _tensor(TWO_OF_FOUR))
    _assert_values([[1.0, 2.0], [3.0, 4.0]], got)
