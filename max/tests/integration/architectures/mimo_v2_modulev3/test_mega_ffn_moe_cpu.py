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
"""The routed experts' one-FFN op, staged on a virtual B200, uncompiled.

The default TP4 MoE stages ``megaffn.mimo.w4a8`` with its band, ``floor < rows
<= ceiling``, so the op itself runs one launch inside the band -- a c2-and-up
decode step -- and the two-launch chain outside it, which is a c1 step and
every prefill step. Nothing outside the op branches: the graph the layer stages
holds no ``mo.if`` around the MoE at all, which is what keeps the engine's
eager prefill path from recording and polling events at 47 conditionals a
step. A split the kernel is not locked to (TP2, and so every other device
count) keeps the chain too.

The bounds are the layer's own constants, so the tests below move them with
``monkeypatch.setattr`` on the module rather than through the environment: the
call site reads them when it stages, and a served checkpoint is a precompiled
graph in which a late environment change would be silently inert.

Restoring the four-launch chain -- the two expert GEMMs and the two quantizes
between them, which is what an unfused TP4 MoE stages -- takes
``MAX_DISABLE_FUSED_SWIGLU_NVFP4`` alone: it drops the chain's fused epilogue
*and* keeps the one launch out of the graph, because the op would otherwise run
that same fused epilogue out of band and the escape hatch would not mean what
it says.

Nothing here compiles: no ``session.load``, no ``Module.compile``. What the
launches compute is gated elsewhere: the staged weight layout in
``test_mega_ffn_moe_gate_up_cpu.py``, the kernel's read order against the chain
on a GPU in ``graph/test_megaffn_mimo_gpu.py``, and the model's numerics in the
MoE's own tests. The op comes from the closed-source kernel packages, so the
tests that read it skip in a source-only build; every test that holds the chain
runs there too.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest
from max import functional as F
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import Graph, SymbolicDim
from max.nn._graph_ops import MEGAFFN_MIMO_MAX_EXPERTS, MEGAFFN_MIMO_SLOTS
from max.nn.activation import GatedActivation
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.mimo_v2_modulev3.layers import moe as mimo_moe
from max.pipelines.architectures.mimo_v2_modulev3.layers.moe import (
    MEGAFFN_MIMO_BAND_MIN_ROWS,
    MEGAFFN_MIMO_BAND_ROWS,
    MiMoV2MoE,
)
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh, TensorLayout

MEGAFFN = "megaffn.mimo.w4a8"
CHAINED_GATE_UP = "grouped_matmul_swiglu_nvfp4"
"""The chain's first launch, which the MegaFFN replaces. It is not in the
layer's graph at all any more: the op runs it itself, out of band."""

CHAINED_GATE_UP_PLAIN = "grouped_matmul_block_scaled"
"""The chain's first launch once its fused epilogue is off, so the activation
is a launch of its own."""

CHAINED_ACTIVATION = "ep.fused_silu"
"""The chain's activation launch, which only the unfused chain has."""

FUSED_SWIGLU_SWITCH = "MAX_DISABLE_FUSED_SWIGLU_NVFP4"
"""The chain's escape hatch, set from the helper below. It also keeps the op out
of the graph: the op runs the fused gate/up chain out of band, so a layer that
kept it under this switch would still run the epilogue it was told to drop."""


HIDDEN = 4096
MOE_DIM = 2048
"""MiMo-V2.6-Flash's expert geometry: at TP4 that is the op's 1024 gate-up rows
a device over K1 4096, and 4096 down rows over K2 512."""

EXPERTS = 2
"""Two expert slots keep this test's weights near 25 MiB; the op's expert count
is a compile-time parameter, so the walk's width is not what is under test."""

TOP_K = 8
DEVICES = 4
TOKENS = "tokens"


def _mesh(devices: int = DEVICES) -> DeviceMesh:
    """A virtual B200 mesh, the shape the MegaFFN kernel is built for."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(devices)
    return DeviceMesh(
        tuple(Accelerator(i) for i in range(devices)),
        (devices,),
        ("tp",),
    )


def _new_moe(mesh: DeviceMesh) -> MiMoV2MoE:
    """One routed-expert layer on ``mesh``, as the tests below build it."""
    with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
        return MiMoV2MoE(
            hidden_dim=HIDDEN,
            num_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
            moe_dim=MOE_DIM,
            norm_topk_prob=True,
            quant_config=QuantConfig.mxfp4(
                mlp_quantized_layers={1},
                embedding_output_dtype=DType.bfloat16,
            ),
            route_on_every_device=True,
        )


def _switches(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fused_swiglu: bool = True,
) -> None:
    """Sets the chain's escape hatch, so no arm can come from the caller.

    ``fused_swiglu`` false unsets the fused epilogue the chain's first launch
    applies, and keeps the op out of the graph with it.
    """
    if fused_swiglu:
        monkeypatch.delenv(FUSED_SWIGLU_SWITCH, raising=False)
    else:
        monkeypatch.setenv(FUSED_SWIGLU_SWITCH, "1")


def _staged_moe(
    monkeypatch: pytest.MonkeyPatch,
    devices: int,
    *,
    fused_swiglu: bool = True,
    prepare: Callable[[MiMoV2MoE], None] | None = None,
) -> str:
    """Stages one MoE layer's graph on a virtual mesh and returns its text.

    ``prepare`` edits the layer after its constructor and before it stages, for
    the cases that swap a piece the constructor builds.
    """
    _switches(monkeypatch, fused_swiglu=fused_swiglu)
    mesh = _mesh(devices)
    with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
        moe = _new_moe(mesh)
        if prepare is not None:
            prepare(moe)
        staged, _registry = moe.stage(
            TensorLayout(DType.bfloat16, [SymbolicDim(TOKENS), HIDDEN], mesh)
        )
    return str(staged)


def _band(text: str) -> tuple[int, int]:
    """The (floor, ceiling) the staged op carries.

    One op a device, so the first is the shape the others repeat. Both bounds
    ride as the op's parameters, which is the whole of the regime policy: any
    graph with them absent, or with a third arm, would make this refuse rather
    than guess.
    """
    floor = re.search(r"floor_rows = (\d+)", text)
    ceiling = re.search(r"band_rows = (\d+)", text)
    assert floor is not None and ceiling is not None, text
    return int(floor.group(1)), int(ceiling.group(1))


C1_ROWS = 64
"""A single request's routed rows at c1: 8 verify rows, top-8."""

C2_ROWS = 128
"""Two requests' at c2, the shortest step the default band lets in."""


def _skip_without_the_one_launch() -> None:
    """Skips the caller unless this build registers the experts' one launch.

    ``megaffn.mimo.w4a8`` comes from the closed-source kernel packages, so a
    source-only build has no such op and the layer stages the chain instead:
    the tests that read the op's band or its counters have nothing to read
    there. The check is the model's own, asked of an empty graph, which carries
    the same kernel library the staging does.
    """
    with Graph("mega_ffn_probe"):
        registered = mimo_moe._mega_ffn_mimo_is_registered()
    if not registered:
        pytest.skip(
            "megaffn.mimo.w4a8 is registered only by the closed-source kernel "
            "packages; this build stages the chain instead"
        )


def test_tp4_decode_stages_the_op_with_its_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One op a device, carrying the band, and no branch anywhere.

    The band is a host-side decision inside the op: a step whose rows fall in
    it takes the one launch, and a step outside it runs the chain the op holds
    itself. Neither arm is a region of the graph, so the layer stages one op
    per device and nothing else to choose between them.
    """
    _skip_without_the_one_launch()
    text = _staged_moe(monkeypatch, DEVICES)
    assert text.count(MEGAFFN) == DEVICES, text
    # The chain lives inside the op, so it is not in the graph to be read off.
    assert CHAINED_GATE_UP not in text, text
    assert _band(text) == (
        MEGAFFN_MIMO_BAND_MIN_ROWS,
        MEGAFFN_MIMO_BAND_ROWS,
    ), text
    # One op a device allocates the counters, so a device's allocations do not
    # multiply.
    assert text.count("mo.buffer.create") == DEVICES, text


def test_no_graph_conditional_around_the_moe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The graph holds no `mo.if` at all, which is the point of the change.

    A `mo.if` a MoE call is a region the engine's eager dispatcher cannot launch
    straight through: it recorded and polled an event per region at 47 regions
    a prefill step, worth +10-13% TTFT on a 542-token prefill with no GPU work
    behind it. The regime therefore has to be decided where the rows are, on
    the host, inside the op -- and a batch whose rows are symbolic in the graph
    is exactly why it cannot be decided when the graph is built.
    """
    text = _staged_moe(monkeypatch, DEVICES)
    assert "mo.if" not in text, text
    assert "mo.cond" not in text, text


def test_the_floor_keeps_c1_on_the_chain() -> None:
    """The default floor excludes a c1 step and admits a c2 one.

    Rows are what the condition compares, so the default has to sit between the
    two: above a single request's 64 and at or below two requests' 128. Anything
    else either loses the c2 win or keeps the c1 loss.
    """
    assert C1_ROWS <= MEGAFFN_MIMO_BAND_MIN_ROWS < C2_ROWS
    assert C2_ROWS <= MEGAFFN_MIMO_BAND_ROWS


def test_a_floor_above_the_band_leaves_every_step_on_the_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Floor and ceiling invert cleanly: no step's rows satisfy both.

    Both bounds reach the op and the op still stages, because the regime is a
    run-time comparison it makes -- and one made where a step's rows are known,
    which is not when the graph is built.
    """
    _skip_without_the_one_launch()
    monkeypatch.setattr(mimo_moe, "MEGAFFN_MIMO_BAND_MIN_ROWS", 1024)
    text = _staged_moe(monkeypatch, DEVICES)
    assert _band(text) == (1024, MEGAFFN_MIMO_BAND_ROWS), text
    assert MEGAFFN in text, text


def test_a_zero_band_leaves_every_step_on_the_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ceiling of 0 is the way to turn the one launch off.

    No step with rows falls at or below it, and the op runs the chain inside
    itself for those, so what is left is the op's own bookkeeping: the op and the
    counter workspace it stages stay in the graph, which is the difference from
    a zero band on a graph conditional, which also kept the conditional's host
    cost.
    """
    _skip_without_the_one_launch()
    monkeypatch.setattr(mimo_moe, "MEGAFFN_MIMO_BAND_ROWS", 0)
    text = _staged_moe(monkeypatch, DEVICES)
    assert _band(text) == (MEGAFFN_MIMO_BAND_MIN_ROWS, 0), text
    assert text.count("mo.buffer.create") == DEVICES, text
    assert "mo.if" not in text, text


def test_the_band_ceiling_fits_the_pool_counters() -> None:
    """The band's ceiling can never ask for more pools than the op holds.

    An expert walks one pool per 8 routed rows, so the widest step the band
    admits over the most experts the walk scans still fits the counters. That
    is why the layer's call can drop the op's own pool check: no band it
    builds with these constants can make the op raise.
    """
    assert (
        MEGAFFN_MIMO_BAND_ROWS // 8 + MEGAFFN_MIMO_MAX_EXPERTS
        <= MEGAFFN_MIMO_SLOTS
    )


def test_the_counters_are_staged_once_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One counter workspace a device, outside any arm of the regime.

    Every MoE layer of a model calls one compiled body, so all of them reach
    this one workspace, and a body staged twice would show here as a second
    `buffer.create` per device.
    """
    _skip_without_the_one_launch()
    text = _staged_moe(monkeypatch, DEVICES)
    assert text.count("mo.buffer.create") == DEVICES, text


@pytest.mark.parametrize(
    "activation",
    [
        # A clamp is a function the kernel does not compute.
        GatedActivation(
            fused_activation="silu", limit=30.0, round_to_bf16=True
        ),
        # So is a silu that does not round through bfloat16, which is what
        # leaves the downs' re-quantize reading other values.
        GatedActivation(fused_activation="silu", round_to_bf16=False),
        # And so is another gated activation entirely.
        GatedActivation(fused_activation="swigluoai", alpha=1.7, limit=7.0),
    ],
    ids=["clamped-silu", "unrounded-silu", "swigluoai"],
)
def test_another_activation_keeps_the_chain(
    monkeypatch: pytest.MonkeyPatch, activation: GatedActivation
) -> None:
    """Only plain bf16-rounded silu goes down the one launch."""

    def swap(moe: MiMoV2MoE) -> None:
        moe.fused_activation = activation

    text = _staged_moe(monkeypatch, DEVICES, prepare=swap)
    assert MEGAFFN not in text, text
    assert CHAINED_GATE_UP in text, text


def test_the_unfused_switch_alone_stages_the_four_launch_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MAX_DISABLE_FUSED_SWIGLU_NVFP4`` alone means what it says.

    The op's out-of-band arm runs the fused gate/up chain whatever the switch
    says, so a layer that kept the op under this switch would still run the
    epilogue it was told to drop. The switch therefore keeps the op out of the
    graph as well, and the four launches come back on their own.
    """
    text = _staged_moe(monkeypatch, DEVICES, fused_swiglu=False)
    assert MEGAFFN not in text, text
    assert "mo.buffer.create" not in text, text
    assert CHAINED_GATE_UP not in text, text
    assert CHAINED_GATE_UP_PLAIN in text, text
    assert CHAINED_ACTIVATION in text, text
    assert "band_rows" not in text, text
    assert "mo.if" not in text, text


def test_two_way_split_keeps_the_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TP2 halves the gate-up rows the op is locked to, so it does not apply."""
    text = _staged_moe(monkeypatch, 2)
    assert MEGAFFN not in text, text
    assert CHAINED_GATE_UP in text, text


def test_a_build_without_the_kernel_keeps_the_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A build whose kernel library lacks the op stages the chain instead.

    ``megaffn.mimo.w4a8`` is registered only by the closed-source kernel
    packages, so a source-only build has no such op and emitting one would
    fail the graph on an unknown op. The build those users ran before the op
    existed is the two-launch chain, so that is what they keep.
    """
    monkeypatch.setattr(mimo_moe, "_mega_ffn_mimo_is_registered", lambda: False)
    text = _staged_moe(monkeypatch, DEVICES)
    assert MEGAFFN not in text, text
    assert CHAINED_GATE_UP in text, text
    assert "mo.buffer.create" not in text, text


def test_an_expert_count_past_the_walk_keeps_the_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More experts than the walk scans is a config the chain carries.

    The walk's width is a static parameter of the op, and past it the op raises
    when the graph is built. The real 256 would take a 257-expert layer -- some
    3 GiB of declared weights a CPU test will not stage -- so the limit moves
    and the comparison against it stays what is under test.
    """
    monkeypatch.setattr(mimo_moe, "MEGAFFN_MIMO_MAX_EXPERTS", EXPERTS - 1)
    text = _staged_moe(monkeypatch, DEVICES)
    assert MEGAFFN not in text, text
    assert CHAINED_GATE_UP in text, text
    assert "mo.buffer.create" not in text, text
