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
"""The fused W4A8 MoE FFN op's graph contract, without compiling or running.

Stages the op's graph at MiMo-V2.6-Flash's expert slice and asserts what the
graph says: the op is registered under its name, it takes the operands in the
declared order with the counter workspace and its chain last, it carries the
tuning parameters and the band that picks its regime, and it returns the three
tensors the two-launch chain produces, with the shapes and dtypes a routed MoE
layer feeds and consumes. The token rows may be a dynamic dimension, which is
what the model's routed batch has.

Nothing here compiles: no `session.load`, no `Module.compile`. The op's numerics
are gated bit for bit in Mojo
(`Kernels/test/gpu/Moka/test/test_megaffn_mimo_port.mojo` against the component,
`Kernels/test/gpu/linalg/test_mega_ffn_mimo_port.mojo` against the chain), and
its execution through a graph in `test_megaffn_mimo_gpu.py`.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from _megaffn_mimo_graphs import (
    BAND_ROWS,
    E4M3,
    FLOOR_ROWS,
    FUSED_ACTIVATION,
    K1,
    K2,
    N1,
    N2,
    build_port_graph,
    input_specs,
)
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    StaticDim,
    TensorType,
    TensorValue,
    Type,
)
from max.nn._graph_ops import (
    MEGAFFN_MIMO_COUNTER_WORDS,
    MEGAFFN_MIMO_SLOTS,
    megaffn_mimo_w4a8,
)

NUM_EXPERTS = 256
"""MiMo's per-rank expert count, the walk's full expert-scan width."""
ROWS = 64
GROUPS = 64


def _op_text() -> str:
    """Returns the printed form of the one custom op the graph holds."""
    specs, _ = input_specs(
        NUM_EXPERTS, ROWS, GROUPS, DeviceRef.GPU(0), DeviceRef.CPU()
    )
    text = str(build_port_graph(specs))
    assert "megaffn.mimo.w4a8" in text, text
    assert text.count("mo.custom") == 1, text
    start = text.index("mo.custom")
    return text[start : text.index("\n", start)]


def _operand_types(op_text: str) -> list[str]:
    """Returns one printed custom op's operand types, in order."""
    operands = op_text[op_text.index(" : (") + 4 : op_text.index(") -> (")]
    return ["!mo." + part for part in operands.split("!mo.")[1:]]


def test_op_takes_the_declared_operands_in_order() -> None:
    types = _operand_types(_op_text())
    # Eleven tensors from the routing and the weights, the counter workspace the
    # op mutates in place, and the chain that mutation rides on.
    assert len(types) == 13, types
    assert types[-1] == "!mo.chain", types
    assert types[-2].startswith("!mo.buffer"), types

    def _element_type(spec: str) -> str:
        if spec.startswith("!mo.chain"):
            return "chain"
        if "tensor<" not in spec:
            return "buffer"
        return spec.split("], ")[1].split(",")[0]

    assert [_element_type(spec) for spec in types] == [
        "f8e4m3fn",  # hidden_states
        "ui8",  # w13
        "ui8",  # w2
        "f8e8m0fnu",  # a_scales
        "f8e8m0fnu",  # w13_scales
        "f8e8m0fnu",  # w2_scales
        "ui32",  # expert_start_indices
        "ui32",  # a_scale_offsets
        "si32",  # expert_ids
        "f32",  # expert_scales_l1
        "f32",  # expert_scales_l2
        "buffer",  # arrival_count
        "chain",  # the in-place mutation's ordering
    ], types
    assert f"[{ROWS}, {K1}]" in types[0], types[0]
    assert f"[{NUM_EXPERTS}, {N1}, {K1 // 2}]" in types[1], types[1]
    assert f"[{NUM_EXPERTS}, {N2}, {K2 // 2}]" in types[2], types[2]
    assert f"[{NUM_EXPERTS + 1}]" in types[6], types[6]


def test_op_carries_the_tuning_parameters() -> None:
    op = _op_text()
    for key, value in (
        ("num_experts", NUM_EXPERTS),
        ("ctas", 148),
        ("stages", 6),
        ("tasks", 16),
    ):
        assert f"{key} = {value}" in op, (key, op)


def test_op_carries_the_band_and_the_activation() -> None:
    """The regime is the op's, so its bounds and its activation ride on it.

    Nothing in the graph compares the routed rows: the op picks the arm from
    its own operand count, which is what keeps a `mo.if` out of every MoE
    call. That makes these three parameters the whole of the policy, and none
    of them may fall back to a default here.
    """
    op = _op_text()
    assert f"band_rows = {BAND_ROWS}" in op, op
    assert f"floor_rows = {FLOOR_ROWS}" in op, op
    # Plain silu, the f32 result rounded through bfloat16 before the requant.
    assert 'activation = "silu_bf16"' in op, op


def test_op_returns_the_chain_s_tensors() -> None:
    specs, _ = input_specs(
        NUM_EXPERTS, ROWS, GROUPS, DeviceRef.GPU(0), DeviceRef.CPU()
    )
    graph = build_port_graph(specs)
    device = DeviceRef.GPU(0)

    # The two-launch chain's own output types: the bf16 result, the packed
    # intermediate and its 5-D scale tile.
    assert list(graph.output_types) == [
        TensorType(DType.bfloat16, [ROWS, N2], device=device),
        TensorType(DType.float8_e4m3fn, [ROWS, K2], device=device),
        TensorType(
            DType.float8_e8m0fnu, [GROUPS, K2 // 128, 32, 4, 4], device=device
        ),
    ]


def test_op_takes_the_token_rows_as_a_dimension() -> None:
    """The row count is the model's, so the op may not force it static."""
    specs, _ = input_specs(
        NUM_EXPERTS, "rows", "scale_rows", DeviceRef.GPU(0), DeviceRef.CPU()
    )
    graph = build_port_graph(specs)
    assert "megaffn.mimo.w4a8" in str(graph)
    result = graph.output_types[0]
    assert isinstance(result, TensorType), result
    assert result.dtype == DType.bfloat16
    assert not isinstance(result.shape[0], StaticDim), result.shape


def _operands(graph: Graph) -> dict[str, TensorValue | BufferValue]:
    """Returns the staged graph's operands by name, as the wrapper takes them."""
    specs, _ = input_specs(
        NUM_EXPERTS, ROWS, GROUPS, DeviceRef.GPU(0), DeviceRef.CPU()
    )
    operands: dict[str, TensorValue | BufferValue] = {}
    for (name, _), value in zip(specs, graph.inputs, strict=True):
        if not isinstance(value, TensorValue | BufferValue):
            raise TypeError(f"{name} is neither a tensor nor a buffer")
        operands[name] = value
    return operands


def _invoke(
    args: dict[str, TensorValue | BufferValue], **tuning: int
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Calls the op with `args`, in its declared order.

    The band and the activation are the call's, not the op's: every call here
    passes them, as a model does. `tuning` is the launch's own numbers, for the
    cases that leave them out of range.
    """
    return megaffn_mimo_w4a8(
        args["hidden_states"].tensor,
        args["w13"].tensor,
        args["w2"].tensor,
        args["a_scales"].tensor,
        args["w13_scales"].tensor,
        args["w2_scales"].tensor,
        args["expert_start_indices"].tensor,
        args["a_scale_offsets"].tensor,
        args["expert_ids"].tensor,
        args["expert_scales_l1"].tensor,
        args["expert_scales_l2"].tensor,
        args["arrival_count"].buffer,
        band_rows=BAND_ROWS,
        floor_rows=FLOOR_ROWS,
        activation=FUSED_ACTIVATION,
        **tuning,
    )


def _type_overrides(case: str) -> dict[str, Type[Any]]:
    """Returns the one input type a negative case stages wrongly."""
    device = DeviceRef.GPU(0)
    cases: dict[str, dict[str, Type[Any]]] = {
        "w13-dtype": {
            "w13": TensorType(E4M3, (NUM_EXPERTS, N1, K1 // 2), device=device)
        },
        "hidden-rank": {
            "hidden_states": TensorType(E4M3, (ROWS,), device=device)
        },
        "a_scales-dtype": {
            "a_scales": TensorType(
                E4M3, (GROUPS, K1 // 128, 32, 4, 4), device=device
            )
        },
        "expert_ids-dtype": {
            "expert_ids": TensorType(
                DType.uint32, (NUM_EXPERTS,), device=device
            )
        },
        "counters-short": {
            "arrival_count": BufferType(DType.uint32, (32,), device=device)
        },
        "counters-rank": {
            "arrival_count": BufferType(
                DType.uint32, (MEGAFFN_MIMO_SLOTS, 8), device=device
            )
        },
        "counters-dtype": {
            "arrival_count": BufferType(
                DType.int32, (MEGAFFN_MIMO_COUNTER_WORDS,), device=device
            )
        },
    }
    return cases.get(case, {})


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("w13-dtype", "dtype uint8"),
        ("hidden-rank", "rank 2"),
        ("a_scales-dtype", "float8_e8m0fnu"),
        ("expert_ids-dtype", "int32"),
        ("counters-type", "counters as a buffer"),
        ("counters-short", "counters hold"),
        ("counters-rank", "rank 1"),
        ("counters-dtype", "dtype uint32"),
        ("ctas-zero", "one CTA per SM"),
        ("ctas-past-one-a-sm", "one CTA per SM"),
        ("tasks-zero", "task ring"),
        ("tasks-past-the-reserved-bytes", "task ring"),
        ("stages-too-deep", "K-group ring"),
    ],
)
def test_op_refuses_a_contract_violation(case: str, match: str) -> None:
    specs, _ = input_specs(
        NUM_EXPERTS, ROWS, GROUPS, DeviceRef.GPU(0), DeviceRef.CPU()
    )
    overrides = _type_overrides(case)
    types = [
        (name, overrides.get(name, spec_type)) for name, spec_type in specs
    ]
    tuning = {
        "ctas-zero": {"ctas": 0},
        "ctas-past-one-a-sm": {"ctas": 4096},
        "tasks-zero": {"tasks": 1},
        "tasks-past-the-reserved-bytes": {"tasks": 64},
        "stages-too-deep": {"stages": 9},
    }.get(case, {})
    with Graph(
        "megaffn_mimo_negative", input_types=[t for _, t in types]
    ) as graph:
        args = _operands(graph)
        if case == "counters-type":
            # A tensor where the persistent workspace buffer belongs.
            call_args: list[Any] = [
                args[name].tensor
                for name in (
                    "hidden_states",
                    "w13",
                    "w2",
                    "a_scales",
                    "w13_scales",
                    "w2_scales",
                    "expert_start_indices",
                    "a_scale_offsets",
                    "expert_ids",
                    "expert_scales_l1",
                    "expert_scales_l2",
                )
            ]
            call_args.append(args["expert_scales_l1"].tensor)
            with pytest.raises(TypeError, match=match):
                megaffn_mimo_w4a8(
                    *call_args,
                    band_rows=BAND_ROWS,
                    floor_rows=FLOOR_ROWS,
                    activation=FUSED_ACTIVATION,
                )
            return
        with pytest.raises((ValueError, TypeError), match=match):
            _invoke(args, **tuning)


@pytest.mark.parametrize(
    ("rows", "groups"),
    [((MEGAFFN_MIMO_SLOTS + 8) * 8, GROUPS), ("rows", "scale_rows")],
)
def test_op_refuses_a_ceiling_past_its_pools(
    rows: int | str, groups: int | str
) -> None:
    """A band ceiling the counters cannot hold is refused at graph build.

    The pools are the one launch's, so the widest step the band admits bounds
    them: the bound holds whether the graph's row count is fixed or, as in the
    model, symbolic, so an oversized ceiling fails here instead of as a failed
    launch while serving.
    """
    ceiling = (MEGAFFN_MIMO_SLOTS + 8) * 8
    specs, _ = input_specs(
        NUM_EXPERTS, rows, groups, DeviceRef.GPU(0), DeviceRef.CPU()
    )
    with pytest.raises(ValueError, match="pools the counters hold"):
        build_port_graph(specs, band_rows=ceiling, floor_rows=0)


def test_op_accepts_the_same_rows_above_its_band() -> None:
    """Past the ceiling the step is the chain's, which needs no pools.

    The same row count that the band refuses stages fine above it: the op runs
    the two-launch chain there, and that chain has no counter workspace at all.
    Before the op picked its own regime this had to be refused, because a graph
    conditional compiled both arms and the one launch's counters would have
    bounded every step of the graph.
    """
    rows = (MEGAFFN_MIMO_SLOTS + 8) * 8
    specs, _ = input_specs(
        NUM_EXPERTS, rows, GROUPS, DeviceRef.GPU(0), DeviceRef.CPU()
    )
    text = str(build_port_graph(specs, band_rows=rows // 2, floor_rows=0))
    assert "megaffn.mimo.w4a8" in text, text


def test_op_refuses_negative_bounds() -> None:
    """Negative bounds are refused: neither is a routed-row count."""
    specs, _ = input_specs(
        NUM_EXPERTS, ROWS, GROUPS, DeviceRef.GPU(0), DeviceRef.CPU()
    )
    with pytest.raises(ValueError, match="may be negative"):
        build_port_graph(specs, floor_rows=-1)
    with pytest.raises(ValueError, match="may be negative"):
        build_port_graph(specs, band_rows=-1)


def test_op_requires_the_band_and_the_activation() -> None:
    """The policy has no default: a call that omits it is refused, not guessed.

    A defaulted ceiling would leave every step on the chain without saying so,
    and a defaulted activation would run plain ``silu`` where the model's chain
    rounds through bfloat16. Both are the caller's to name, and neither may be
    passed positionally either, where the twelve operands would hide it.
    """
    parameters = inspect.signature(megaffn_mimo_w4a8).parameters
    for name in ("band_rows", "floor_rows", "activation"):
        parameter = parameters[name]
        assert parameter.default is inspect.Parameter.empty, (
            f"{name} has a default, so a caller can leave the policy out"
        )
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} may be passed positionally: {parameter.kind}"
        )
