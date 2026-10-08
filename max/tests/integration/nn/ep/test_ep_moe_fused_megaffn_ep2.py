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
"""EP2 runtime gate for the one-launch fused EP MoE (``mega_ffn.ep_fused``) under
CUDA-graph replay.

Arm A is the shipping expert-parallel MoE (``ep.dispatch_async`` -> side-stream
shared expert -> ``ep.dispatch_wait`` -> FFN -> combine; at NVFP4 the default
``mega_ffn.ep_combine_send`` role). Arm B is ``mega_ffn.ep_fused``: dispatch,
MegaFFN and the weighted combine in ONE launch per rank, the shared expert
inline on the default stream. Same weights, same routing, same inputs, same two
devices, both formats (NVFP4 and MXFP8), top-k 8. The MXFP8 comparisons are
routed-only (no shared expert in either arm).

What this exercises that the kernel harness cannot:

* production allocation and lowering: EP init sets the workspace up through
  ``mega_ffn.ep_fused_init`` (``MODULAR_EP_FUSED_MOE=1``) and records it in the
  global cache; the graph op, the emitter and the selector;
* CUDA-graph capture of the fused arm, instantiated ONCE, then replayed across
  changing generations (inputs and routes change in place, both bank parities
  are used, no recapture, no host-side parity or protocol reset);
* two chained MoE layers per replay on one shared workspace;
* two independent workspaces (two fused models) alternating on the same
  devices;
* a rank with zero source tokens and a ragged split;
* a NaN canary on the captured output before every replay, so an unwritten
  token row is caught;
* the fail-closed host checks: a batch past the capacity, a launch whose EP
  init set no workspace up, or a workspace of another layout raises before any
  launch;
* the backend's refusal of a configuration it does not serve, which keeps the
  shipping chain;
* a fused and a shipping model alternating on ONE EP instance, whose counter
  buffers hold the fused counters in their reserved words.

The optional pilot (``EP2_FUSED_PILOT=1``) times captured SHIP vs captured FULL
replays, paired and interleaved, on two cells per format. It is a small
current-mode pilot, not a campaign.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import statistics
import time
from collections.abc import Callable, Iterator
from unittest import mock

import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, Shape, ShardingStrategy, TensorType
from max.graph.weights import WeightData
from max.nn.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_manager import _plan_fused_moe_workspace
from max.nn.layer import LayerList, Module
from max.nn.moe import MoEQuantized
from max.nn.moe.expert_parallel import forward_moe_sharded_layers
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from test_common.graph_utils import is_b100_b200

HIDDEN_DIM = 6144
MOE_DIM = 2048
NUM_EXPERTS = 32  # 16 local experts per rank at EP2
TOP_K = 8
N_DEVICES = 2
MAX_TOKENS_PER_RANK = 24
N_LAYERS = int(
    os.environ.get("EP2_LAYERS", "2")
)  # diagnosis knob; the gate runs 2
N_GENERATIONS = 4  # >= 3 changing generations, both parities

# Source tokens per rank. ``bs1``: one rank holds no tokens at all.
CELLS = {
    "dense24": [24, 24],
    "bs1": [1, 0],
    "ragged": [24, 5],
}

# The dense specialization's capacity (`FUSED_EP_DENSE_MAX_TPR` in
# `mega_ffn_ep_workspace.mojo`) and its cells: the dense point it exists for, a
# small batch it also serves, and the ragged guardrail.
DENSE_MAX_TOKENS_PER_RANK = 32
DENSE_CELLS = {
    "dense32": [32, 32],
    "bs8": [8, 8],
    "ragged": [24, 5],
}

FORMATS = ("nvfp4", "mxfp8")

T975_11 = 2.201  # t(0.975, 11)


# ----------------------------------------------------------------------------
# Synthetic weights
# ----------------------------------------------------------------------------
def _shared_bf16(w: dict[str, torch.Tensor], device: str) -> None:
    for name, shape in (
        ("gate_proj", (MOE_DIM, HIDDEN_DIM)),
        ("up_proj", (MOE_DIM, HIDDEN_DIM)),
        ("down_proj", (HIDDEN_DIM, MOE_DIM)),
    ):
        w[f"shared_experts.{name}.weight"] = (
            torch.randn(*shape, dtype=torch.bfloat16, device=device) * 1e-2
        )


def _nvfp4_weights() -> dict[str, torch.Tensor]:
    """Random NVFP4 experts (E2M1 pairs, E4M3 scales per 16, per-tensor
    ``weight_scale_2`` and ``input_scale``) plus a BF16 shared expert."""
    torch.manual_seed(20260929)
    device = "cuda:0"

    def _proj(
        w: dict[str, torch.Tensor],
        prefix: str,
        out_dim: int,
        in_dim: int,
        weight_scale_2: torch.Tensor | None = None,
        input_scale: float = 1.0,
    ) -> torch.Tensor:
        w[f"{prefix}.weight"] = torch.randint(
            0, 256, (out_dim, in_dim // 2), dtype=torch.uint8, device=device
        )
        w[f"{prefix}.weight_scale"] = (
            torch.rand(out_dim, in_dim // 16, device=device) * 100.0 + 50.0
        ).to(torch.float8_e4m3fn)
        if weight_scale_2 is None:
            weight_scale_2 = torch.rand((), device=device) * 1e-4
        w[f"{prefix}.weight_scale_2"] = weight_scale_2
        w[f"{prefix}.input_scale"] = torch.full(
            (), input_scale, dtype=torch.float32, device=device
        )
        return weight_scale_2

    w: dict[str, torch.Tensor] = {
        "gate.gate_score.weight": torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )
        * 1e-3
    }
    for e in range(NUM_EXPERTS):
        s2 = _proj(w, f"experts.{e}.gate_proj", MOE_DIM, HIDDEN_DIM)
        _proj(w, f"experts.{e}.up_proj", MOE_DIM, HIDDEN_DIM, weight_scale_2=s2)
        # Non-unit, per-expert down input scales: the fused op must invert
        # them exactly as the shipping chain does.
        _proj(
            w,
            f"experts.{e}.down_proj",
            HIDDEN_DIM,
            MOE_DIM,
            input_scale=0.75 + 0.5 * (e % 4) / 3,
        )
    _shared_bf16(w, device)
    return w


def _mxfp8_weights() -> dict[str, torch.Tensor]:
    """Random MXFP8 experts and shared expert (E4M3 with E8M0 scales per 32;
    the ``conftest`` recipe)."""
    torch.manual_seed(20260929)
    device = "cuda:0"
    fp8_dtype = torch.float8_e4m3fn
    scale_dtype = torch.float8_e8m0fnu
    fp8_max = torch.finfo(fp8_dtype).max
    fp8_min = torch.finfo(fp8_dtype).min
    gen = torch.Generator(device="cpu")
    gen.manual_seed(2026)

    def _proj(
        w: dict[str, torch.Tensor], prefix: str, out_dim: int, in_dim: int
    ) -> None:
        weight = (
            torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device=device)
            * 100
        ).clamp(fp8_min, fp8_max)
        scale_bits = torch.randint(
            109,
            117,
            (out_dim, in_dim // 32),
            dtype=torch.uint8,
            device="cpu",
            generator=gen,
        )
        w[f"{prefix}.weight"] = weight.to(fp8_dtype)
        w[f"{prefix}.weight_scale"] = scale_bits.view(scale_dtype).to(device)

    w: dict[str, torch.Tensor] = {
        "gate.gate_score.weight": torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )
        * 1e-3
    }
    for e in range(NUM_EXPERTS):
        _proj(w, f"experts.{e}.gate_proj", MOE_DIM, HIDDEN_DIM)
        _proj(w, f"experts.{e}.up_proj", MOE_DIM, HIDDEN_DIM)
        _proj(w, f"experts.{e}.down_proj", HIDDEN_DIM, MOE_DIM)
    # The cuda MXFP8 EP role keeps the shared expert's rows in the dispatch,
    # which stacks its weights with the routed experts': same format.
    _proj(w, "shared_experts.gate_proj", MOE_DIM, HIDDEN_DIM)
    _proj(w, "shared_experts.up_proj", MOE_DIM, HIDDEN_DIM)
    _proj(w, "shared_experts.down_proj", HIDDEN_DIM, MOE_DIM)
    return w


def _wrap(
    weights: dict[str, torch.Tensor],
) -> dict[str, WeightData | torch.Tensor]:
    out: dict[str, WeightData | torch.Tensor] = {}
    for key, value in weights.items():
        value = value.cpu()
        if value.dtype == torch.float8_e4m3fn:
            max_dtype = DType.float8_e4m3fn
        elif value.dtype == torch.float8_e8m0fnu:
            max_dtype = DType.float8_e8m0fnu
        else:
            out[key] = value
            continue
        out[key] = WeightData(
            Buffer.from_dlpack(value.view(torch.uint8)).view(max_dtype),
            key,
            max_dtype,
            Shape(value.shape),
        )
    return out


def _quant_config(fmt: str) -> QuantConfig:
    if fmt == "nvfp4":
        return QuantConfig(
            input_scale=InputScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                origin=ScaleOrigin.STATIC,
                dtype=DType.float32,
                block_size=(1, 16),
            ),
            weight_scale=WeightScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                dtype=DType.float8_e4m3fn,
                block_size=(1, 16),
            ),
            mlp_quantized_layers=set(),
            attn_quantized_layers=set(),
            embedding_output_dtype=None,
            format=QuantFormat.NVFP4,
            can_use_fused_swiglu=True,
        )
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 32),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, 32),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.MXFP8,
        can_use_fused_swiglu=True,
    )


def _weights(fmt: str) -> dict[str, WeightData | torch.Tensor]:
    return _wrap(_nvfp4_weights() if fmt == "nvfp4" else _mxfp8_weights())


# ----------------------------------------------------------------------------
# Arms
# ----------------------------------------------------------------------------
def _ship_combine_send(fmt: str) -> bool:
    """Whether the shipping arm fuses the combine send into the MegaFFN
    epilogue. Diagnosis only: EP2_SHIP_COMBINE_SEND=0 runs the NVFP4 shipping
    chain with a separate combine (the `decode_only` role)."""
    return (
        fmt == "nvfp4" and os.environ.get("EP2_SHIP_COMBINE_SEND", "1") == "1"
    )


@contextlib.contextmanager
def _fused_moe_request(on: bool) -> Iterator[None]:
    """Sets the internal developer switch EP init reads, or leaves it unset
    (the default: no opt-in), for the duration of the block."""
    with mock.patch.dict(os.environ):
        os.environ.pop("MODULAR_EP_FUSED_MOE", None)
        if on:
            os.environ["MODULAR_EP_FUSED_MOE"] = "1"
        yield


def _build_arm(
    fmt: str,
    fused: bool,
    wrapped: dict[str, WeightData | torch.Tensor],
    session: InferenceSession,
    devices_ref: list[DeviceRef],
    name: str,
    n_layers: int = N_LAYERS,
    shared: bool = True,
    max_tpr: int = MAX_TOKENS_PER_RANK,
    expect_ready: bool | None = None,
    after_init: Callable[[EPConfig], None] | None = None,
    ep_init_from: EPCommInitializer | None = None,
) -> tuple[Model, EPCommInitializer, str]:
    """Builds one arm. ``fused`` requests the fused path at EP init, which
    must then be ready unless ``expect_ready`` says otherwise.
    ``shared=False`` drops the shared expert from both the module and the
    weights (MXFP8 fused-arm comparisons are routed-only: the cuda MXFP8
    shipping chain computes a shared expert only fused into the EP dispatch,
    which the fused path does not do). ``after_init`` edits the config after
    EP init, before the graph is built. ``ep_init_from`` builds the model on
    an existing EP instance (its buffers and workspace) instead of a new one,
    selecting the fused path only if ``fused``."""
    quant = _quant_config(fmt)
    dtype = DType.uint8 if fmt == "nvfp4" else DType.float8_e4m3fn
    ep_config = EPConfig(
        dispatch_dtype=dtype,
        combine_dtype=DType.bfloat16,
        hidden_size=HIDDEN_DIM,
        top_k=TOP_K,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tpr,
        n_gpus_per_node=N_DEVICES,
        n_nodes=1,
        dispatch_quant_config=quant,
        # The shipping roles per format. MXFP8 on cuda keeps the shared
        # expert's rows in the dispatch (`fused_shared_expert`; the plain
        # block-scaled `dispatch_wait` asserts NVFP4 packing, so that is the
        # only cuda MXFP8 chain). NVFP4 serving roles fuse the combine send
        # into the MegaFFN epilogue (`ep_fuse_ffn_combine_send_for_pipeline`),
        # an NVFP4-only op. The fused MoE is qualified with an unfused shared
        # expert only (`_can_fuse_megaffn_ep`).
        fused_shared_expert=(fmt == "mxfp8" and not fused and shared),
        fuse_ffn_combine_send=_ship_combine_send(fmt),
        moe_dim=MOE_DIM,
    )
    if ep_init_from is None:
        ep_comm_init = EPCommInitializer(ep_config)
    else:
        ep_comm_init = ep_init_from
        ep_config = dataclasses.replace(
            ep_init_from.config,
            fused_moe_ready=fused and ep_init_from.config.fused_moe_ready,
        )
    ep_batch_manager = EPBatchManager(ep_config)

    # One module per layer, as a model has: two `forward_moe_sharded_layers`
    # calls on ONE module would reuse the shared expert's memoized weight
    # concat across the shipping path's side-stream regions (an MLIR
    # dominance error). Every layer loads the same weights.
    class _Stack(Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = LayerList(
                [
                    MoEQuantized(
                        devices=[DeviceRef.CPU()] + devices_ref,
                        hidden_dim=HIDDEN_DIM,
                        num_experts=NUM_EXPERTS,
                        num_experts_per_token=TOP_K,
                        moe_dim=MOE_DIM,
                        has_shared_experts=shared,
                        shared_experts_dim=MOE_DIM if shared else 0,
                        ep_size=N_DEVICES,
                        dtype=dtype,
                        ep_batch_manager=ep_batch_manager,
                        quant_config=quant,
                        # NVFP4: BF16 shared expert (as the shipping NVFP4
                        # test). MXFP8: the quantized shared expert both
                        # arms share, fused into the dispatch (SHIP) or a
                        # plain MXFP8 MLP (fused MoE).
                        shared_experts_dtype=(
                            DType.bfloat16 if fmt == "nvfp4" else None
                        ),
                    )
                    for _ in range(n_layers)
                ]
            )

        def __call__(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError("use the sharded layers directly")

    stack = _Stack()
    layer_shards = []
    for layer in stack.layers:
        assert isinstance(layer, MoEQuantized)
        layer.sharding_strategy = ShardingStrategy.expert_parallel(N_DEVICES)
        layer_shards.append(list(layer.shard(devices_ref)))
    stack.load_state_dict(
        {
            f"layers.{l}.{k}": v
            for l in range(n_layers)
            for k, v in wrapped.items()
            if shared or not k.startswith("shared_experts.")
        }
    )
    if ep_init_from is None:
        with _fused_moe_request(fused):
            ep_comm_init.ep_init(session)
        ready = fused if expect_ready is None else expect_ready
        assert ep_config.fused_moe_ready == ready, (name, ready)
        assert (ep_comm_init.fused_moe_workspace is not None) == ready
    if after_init is not None:
        after_init(ep_config)

    with Graph(
        name,
        input_types=[
            *(
                TensorType(
                    DType.bfloat16,
                    (f"input_len_{i}", HIDDEN_DIM),
                    DeviceRef.GPU(i),
                )
                for i in range(N_DEVICES)
            ),
            *ep_batch_manager.input_types(),
        ],
    ) as graph:
        xs = [x.tensor for x in graph.inputs[:N_DEVICES]]
        ep_batch_manager.fetch_buffers(graph.inputs[N_DEVICES:])
        # Two chained MoE layers (same weights): consecutive fused launches
        # on one shared workspace, as every layer of a model sees them.
        for shards in layer_shards:
            xs = forward_moe_sharded_layers(shards, xs)
        graph.output(*xs)

    ir = str(graph)
    compiled = session.load(graph, weights_registry=stack.state_dict())
    return compiled, ep_comm_init, ir


def _assert_structure(
    fused_ir: str,
    ship_ir: str,
    ship_init: EPCommInitializer,
    ship_has_side_stream: bool,
    ship_combine_send: bool,
) -> None:
    assert (
        fused_ir.count('symbol = "mega_ffn.ep_fused"') == N_LAYERS * N_DEVICES
    )
    for op in (
        "ep.dispatch_async",
        "ep.dispatch_wait",
        "ep.dispatch",
        "ep.combine",
    ):
        assert f'symbol = "{op}' not in fused_ir, op
    assert 'symbol = "mega_ffn.ep_combine_send"' not in fused_ir
    # The fused branch never uses a side stream (exclusive-launch contract).
    assert "mo.sequence" not in fused_ir
    # The arm EP init did not set the fused path up for: no fused op and no
    # fused workspace, and the shipping chain and its FFN + combine-send
    # selector are unchanged.
    assert 'symbol = "mega_ffn.ep_fused"' not in ship_ir
    assert not ship_init.config.fused_moe_ready
    assert ship_init.fused_moe_workspace is None
    # EP dispatch and combine stay: custom ops on the split path, otherwise
    # the graph's own distributed EP ops.
    for stage in ("dispatch", "combine"):
        assert (
            f'symbol = "ep.{stage}' in ship_ir
            or f"mo.distributed.ep.{stage}" in ship_ir
        ), stage
    assert (
        'symbol = "mega_ffn.ep_combine_send"' in ship_ir
    ) == ship_combine_send
    assert ("mo.sequence" in ship_ir) == ship_has_side_stream


def _inputs(lengths: list[int], gen: int, tag: str) -> list[torch.Tensor]:
    torch.manual_seed(7919 * gen + 31 * len(tag) + sum(lengths))
    return [torch.randn(n, HIDDEN_DIM, dtype=torch.bfloat16) for n in lengths]


def _snapshot(bufs: list[Buffer]) -> list[torch.Tensor]:
    return [torch.from_dlpack(b).cpu().float().clone() for b in bufs]


def _ship_reference(
    ship: Model,
    ship_init: EPCommInitializer,
    bufs: list[Buffer],
    n_layers: int = N_LAYERS,
) -> list[torch.Tensor]:
    """The shipping chain's output for ``n_layers`` chained layers.

    Executes a ONE-layer shipping graph layer by layer, feeding its outputs
    back in, instead of one multi-layer graph: the NVFP4 two-layer shipping
    graph produces non-finite output on its second execution on main itself
    (with the shared expert on its side stream; overlap off clears it); one-layer
    graphs are clean under
    repeated execution with changing inputs.
    """
    cur: list[Buffer] = list(bufs)
    for _ in range(n_layers):
        cur = ship.execute(*cur, *ship_init.model_inputs())
    return _snapshot(cur)


def _assert_finite(
    out: list[torch.Tensor], lengths: list[int], where: str
) -> None:
    for i, o in enumerate(out):
        if lengths[i] == 0:
            continue
        bad = (~torch.isfinite(o)).any(dim=-1).nonzero().flatten().tolist()
        assert not bad, (
            f"{where} rank {i}: non-finite rows {bad[:16]}"
            f"{'...' if len(bad) > 16 else ''} of {lengths[i]}"
        )


def _compare(
    ref: list[torch.Tensor],
    out: list[torch.Tensor],
    lengths: list[int],
    where: str,
) -> int:
    exact = 0
    for i, (r, o) in enumerate(zip(ref, out, strict=True)):
        assert r.shape == o.shape == (lengths[i], HIDDEN_DIM), where
        if lengths[i] == 0:
            continue
        assert torch.isfinite(o).all(), f"{where} rank {i}: non-finite output"
        assert o.abs().max() > 0, f"{where} rank {i}: all-zero output"
        cos = torch.nn.functional.cosine_similarity(r, o, dim=-1)
        assert cos.min() > 0.9999, (
            f"{where} rank {i}: cosine {cos.min().item():.6f}"
        )
        exact += int(torch.equal(r, o))
    return exact


_needs_ep2 = [
    pytest.mark.skipif(
        accelerator_api() == "hip", reason="NVIDIA-only kernels"
    ),
    pytest.mark.skipif(not is_b100_b200(), reason="requires B100/B200 (SM100)"),
    pytest.mark.skipif(
        accelerator_count() < N_DEVICES, reason="requires 2 GPUs"
    ),
]


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize(
    ("fmt", "layers"),
    [
        ("nvfp4", 1),
        ("mxfp8", 1),
        ("mxfp8", 2),
        ("nvfp4", 2),
    ],
)
def test_ep2_ship_reference_repeats(fmt: str, layers: int) -> None:
    """The shipping arm alone, executed eagerly across changing generations.

    The reference the replay test compares against must itself be healthy
    under repeated execution with changing routes (the shipping tests execute
    once). Each output is read twice, straight after ``execute`` and after a
    device sync, so a stale read would show as a difference; the same inputs
    are then executed again, so a stale protocol state would show as a
    non-repeatable result.
    """
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)
    wrapped = _weights(fmt)
    ship, ship_init, _ = _build_arm(
        fmt, False, wrapped, session, devices_ref, "SHIP_R", n_layers=layers
    )
    for cell, lengths in CELLS.items():
        bufs = [
            Buffer.from_dlpack(
                torch.zeros(n, HIDDEN_DIM, dtype=torch.bfloat16)
            ).to(devices[i])
            for i, n in enumerate(lengths)
        ]
        for gen in range(N_GENERATIONS):
            xs = _inputs(lengths, gen, f"{fmt}{cell}")
            for i, x in enumerate(xs):
                bufs[i].inplace_copy_from(Buffer.from_dlpack(x).to(devices[i]))
            outs = ship.execute(*bufs, *ship_init.model_inputs())
            fast = _snapshot(outs)
            for d in devices:
                d.synchronize()
            synced = _snapshot(outs)
            where = f"{fmt} {cell} gen {gen} shipping"
            _assert_finite(synced, lengths, where + " (read after sync)")
            _assert_finite(fast, lengths, where + " (read after execute)")
            for i in range(N_DEVICES):
                assert torch.equal(fast[i], synced[i]), (
                    f"{where} rank {i}: the read after execute differs from the read after a device sync"
                )
            again = _snapshot(ship.execute(*bufs, *ship_init.model_inputs()))
            _assert_finite(again, lengths, where + " (second execution)")
            for i in range(N_DEVICES):
                assert torch.equal(synced[i], again[i]), (
                    f"{where} rank {i}: two executions of the same inputs differ"
                )
        print(
            f"{fmt} {cell} ({layers} layer(s) per graph): shipping arm finite and repeatable over"
            f" {N_GENERATIONS} generations",
            flush=True,
        )


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize("mode", ("eager", "graph"))
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_fused_mode_vs_ship(fmt: str, mode: str) -> None:
    """One fused arm against the shipping arm, eager or captured.

    A bisect aid: the shipping reference is checked for finiteness at every
    generation next to exactly one fused arm executed either eagerly or as
    one captured graph replayed, so a reference defect can be attributed to
    fused eager execution or to capture/replay.
    """
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)
    wrapped = _weights(fmt)
    shared = fmt == "nvfp4"
    ship, ship_init, _ = _build_arm(
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP_M",
        n_layers=1,
        shared=shared,
    )
    fused, init, _ = _build_arm(
        fmt, True, wrapped, session, devices_ref, "FULL_M", shared=shared
    )
    key = 500
    for cell, lengths in CELLS.items():
        bufs = [
            Buffer.from_dlpack(
                torch.zeros(n, HIDDEN_DIM, dtype=torch.bfloat16)
            ).to(devices[i])
            for i, n in enumerate(lengths)
        ]
        outs: list[Buffer] | None = None
        key += 1
        for gen in range(N_GENERATIONS):
            xs = _inputs(lengths, gen, f"{fmt}{cell}")
            for i, x in enumerate(xs):
                bufs[i].inplace_copy_from(Buffer.from_dlpack(x).to(devices[i]))
            where = f"{fmt} {cell} gen {gen} {mode}"
            ref = _ship_reference(ship, ship_init, bufs)
            _assert_finite(ref, lengths, where + " shipping reference")
            if mode == "eager":
                got = _snapshot(fused.execute(*bufs, *init.model_inputs()))
            else:
                if outs is None:
                    outs = fused.capture(key, *bufs, *init.model_inputs())
                fused.replay(key, *bufs, *init.model_inputs())
                got = _snapshot(outs)
            _assert_finite(got, lengths, where + " fused")
            _compare(ref, got, lengths, where)
            # The reference again, after the fused arm ran on this generation.
            ref2 = _ship_reference(ship, ship_init, bufs)
            _assert_finite(
                ref2, lengths, where + " shipping reference after the fused arm"
            )
            for i in range(N_DEVICES):
                assert torch.equal(ref[i], ref2[i]), (
                    f"{where} rank {i}: the shipping arm's output changed after the fused arm ran"
                )
        print(
            f"{fmt} {cell} {mode}: {N_GENERATIONS} generations, fused == shipping",
            flush=True,
        )


def _graph_replay_body(
    fmt: str, max_tpr: int, cells: dict[str, list[int]]
) -> int:
    """One capture per workspace replayed across generations and both parities
    on ``cells`` at capacity ``max_tpr``, bit-exact against the eager
    one-layer shipping reference; then the capacity refusal at ``max_tpr + 1``.
    Returns the token block EP init reported for the workspace."""
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)
    wrapped = _weights(fmt)

    shared = fmt == "nvfp4"
    ship, ship_init, ship_ir = _build_arm(
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP",
        n_layers=1,
        shared=shared,
        max_tpr=max_tpr,
    )
    fused_a, init_a, fused_ir = _build_arm(
        fmt,
        True,
        wrapped,
        session,
        devices_ref,
        "FULL_A",
        shared=shared,
        max_tpr=max_tpr,
    )
    fused_b, init_b, _ = _build_arm(
        fmt,
        True,
        wrapped,
        session,
        devices_ref,
        "FULL_B",
        shared=shared,
        max_tpr=max_tpr,
    )
    _assert_structure(
        fused_ir,
        ship_ir,
        ship_init,
        ship_has_side_stream=shared,
        ship_combine_send=_ship_combine_send(fmt),
    )
    # Two EP instances: two cache keys (their ep.init receive-count
    # buffers), so two workspaces, each of the size the host plan query
    # reports for the configuration.
    keys_a = init_a.recv_count_ptrs[0].to_numpy()
    keys_b = init_b.recv_count_ptrs[0].to_numpy()
    assert not set(keys_a.tolist()) & set(keys_b.tolist())
    assert init_a.fused_moe_workspace == init_b.fused_moe_workspace
    assert init_a.fused_moe_workspace is not None
    workspace_bytes, token_block = init_a.fused_moe_workspace
    assert _plan_fused_moe_workspace(init_a.config) == (
        workspace_bytes,
        token_block,
        "",
    )

    exact = 0
    total = 0
    key = 100
    for cell, lengths in cells.items():
        # Fixed device buffers per cell: a captured graph binds them, and the
        # generations below rewrite them in place.
        bufs = [
            Buffer.from_dlpack(
                torch.zeros(n, HIDDEN_DIM, dtype=torch.bfloat16)
            ).to(devices[i])
            for i, n in enumerate(lengths)
        ]
        nan_bufs = [
            Buffer.from_dlpack(
                torch.full((n, HIDDEN_DIM), float("nan"), dtype=torch.bfloat16)
            ).to(devices[i])
            for i, n in enumerate(lengths)
        ]
        arms = [("A", fused_a, init_a, key), ("B", fused_b, init_b, key + 1)]
        key += 2
        outs: dict[str, list[Buffer]] = {}
        prev: dict[str, list[torch.Tensor]] = {}
        for gen in range(N_GENERATIONS):
            xs = _inputs(lengths, gen, f"{fmt}{cell}")
            for i, x in enumerate(xs):
                bufs[i].inplace_copy_from(Buffer.from_dlpack(x).to(devices[i]))
            ref = _ship_reference(ship, ship_init, bufs)
            _assert_finite(
                ref, lengths, f"{fmt} {cell} gen {gen} shipping reference"
            )
            for arm, model, init, k in arms:
                if gen == 0:
                    # Instantiate ONCE; every later generation is a replay of
                    # this graph with the inputs rewritten in place.
                    outs[arm] = model.capture(k, *bufs, *init.model_inputs())
                # Canary: an unwritten token row stays NaN.
                for i, o in enumerate(outs[arm]):
                    if lengths[i] > 0:
                        o.inplace_copy_from(nan_bufs[i])
                model.replay(k, *bufs, *init.model_inputs())
                got = _snapshot(outs[arm])
                where = f"{fmt} {cell} gen {gen} workspace {arm}"
                exact += _compare(ref, got, lengths, where)
                total += sum(1 for n in lengths if n > 0)
                if arm in prev:
                    # Fresh output: the inputs changed, so must the result.
                    changed = any(
                        lengths[i] > 0 and not torch.equal(prev[arm][i], got[i])
                        for i in range(N_DEVICES)
                    )
                    assert changed, (
                        f"{where}: output identical to the previous generation"
                    )
                prev[arm] = got
        print(
            f"{fmt} {cell}: {N_GENERATIONS} generations x 2 workspaces replayed,"
            f" all matched the eager shipping arm",
            flush=True,
        )
    print(
        f"{fmt}: fused (captured, replayed) == shipping (eager) bit-exactly on"
        f" {exact}/{total} (cell, generation, workspace, rank) checks",
        flush=True,
    )

    # Expected failure, last: a batch past the fused per-rank capacity must be
    # refused by the binding's host check on every rank BEFORE any launch, so
    # nothing is poisoned and the process continues.
    over = [
        Buffer.from_dlpack(
            torch.randn(max_tpr + 1, HIDDEN_DIM, dtype=torch.bfloat16)
        ).to(devices[i])
        for i in range(N_DEVICES)
    ]
    with pytest.raises(Exception, match="max_token_per_rank"):
        fused_a.execute(*over, *init_a.model_inputs())
    print(f"{fmt}: capacity overflow refused before launch", flush=True)
    # The workspaces are still healthy after the refusal.
    first = next(iter(cells.values()))
    xs = _inputs(first, 99, fmt)
    ok = [Buffer.from_dlpack(x).to(devices[i]) for i, x in enumerate(xs)]
    ref = _ship_reference(ship, ship_init, ok)
    got = _snapshot(fused_a.execute(*ok, *init_a.model_inputs()))
    _compare(ref, got, first, f"{fmt} post-refusal")
    return token_block


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_fused_graph_replay(fmt: str) -> None:
    # Capacity 24 keeps the decode specialization (token block 8).
    assert _graph_replay_body(fmt, MAX_TOKENS_PER_RANK, CELLS) == 8


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_fused_dense_cap32_graph_replay(fmt: str) -> None:
    """At EP2 the backend picks the dense specialization (token block 32)
    for capacity 32, and it serves [32,32] (plus a small and a ragged batch)
    through the production allocation and graph replay."""
    token_block = _graph_replay_body(
        fmt, DENSE_MAX_TOKENS_PER_RANK, DENSE_CELLS
    )
    assert token_block == 32, token_block
    print(f"{fmt}: dense specialization selected (token block 32)", flush=True)


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_fused_replay_trace_matches(fmt: str) -> None:
    """The captured graph's launch sequence equals eager execution's."""
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)
    wrapped = _weights(fmt)
    fused, init, _ = _build_arm(
        fmt,
        True,
        wrapped,
        session,
        devices_ref,
        "FULL_T",
        shared=(fmt == "nvfp4"),
    )
    lengths = CELLS["dense24"]
    bufs = [
        Buffer.from_dlpack(x).to(devices[i])
        for i, x in enumerate(_inputs(lengths, 0, "trace"))
    ]
    fused.capture(7, *bufs, *init.model_inputs())
    fused.replay(7, *bufs, *init.model_inputs())
    fused.debug_verify_replay(7, *bufs, *init.model_inputs())
    print(
        f"{fmt}: captured launch trace verified against eager execution",
        flush=True,
    )


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_fused_refused_config_keeps_shipping(fmt: str) -> None:
    """With the switch on, a capacity the backend does not serve (above 32
    tokens per rank) gets no workspace, and the model keeps the shipping
    chain. The backend decided; Python holds no rule for it."""
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)
    _, init, ir = _build_arm(
        fmt,
        True,
        _weights(fmt),
        session,
        devices_ref,
        "REFUSED",
        n_layers=1,
        shared=(fmt == "nvfp4"),
        max_tpr=40,
        expect_ready=False,
    )
    assert _plan_fused_moe_workspace(init.config)[:2] == (0, 0)
    assert 'symbol = "mega_ffn.ep_fused"' not in ir
    for stage in ("dispatch", "combine"):
        assert (
            f'symbol = "ep.{stage}' in ir or f"mo.distributed.ep.{stage}" in ir
        ), stage
    print(f"{fmt}: capacity 40 refused at EP init, shipping chain kept")


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
def test_ep2_fused_refused_under_allreduce() -> None:
    """With the switch on, the allreduce backend gets no fused workspace: its
    EP ops see one rank, which the backend's rank rule refuses before it
    allocates, so readiness stays off and the layers keep the allreduce
    chain."""
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    session = InferenceSession(devices=devices)
    config = EPConfig(
        dispatch_dtype=DType.uint8,
        combine_dtype=DType.bfloat16,
        hidden_size=HIDDEN_DIM,
        top_k=TOP_K,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=MAX_TOKENS_PER_RANK,
        n_gpus_per_node=N_DEVICES,
        n_nodes=1,
        dispatch_quant_config=_quant_config("nvfp4"),
        moe_dim=MOE_DIM,
        use_allreduce=True,
    )
    init = EPCommInitializer(config)
    with _fused_moe_request(True):
        init.ep_init(session)
    assert not config.fused_moe_ready
    assert init.fused_moe_workspace is None
    assert "allreduce" in _plan_fused_moe_workspace(config)[2]


def _claim_ready(config: EPConfig) -> None:
    config.fused_moe_ready = True


def _shrink_capacity(config: EPConfig) -> None:
    config.max_tokens_per_rank = 16


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("no_workspace", "no fused EP MoE workspace"),
        ("other_layout", "set up for another configuration"),
    ],
)
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_fused_launch_fails_closed(fmt: str, case: str, match: str) -> None:
    """A fused launch whose EP instance has no workspace (EP init was not
    asked for one), or a workspace of another layout (capacity 24 at init,
    16 in the graph), raises on the host before any rank launches; the
    devices stay usable."""
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)
    wrapped = _weights(fmt)
    shared = fmt == "nvfp4"
    fused, init, ir = _build_arm(
        fmt,
        case == "other_layout",
        wrapped,
        session,
        devices_ref,
        f"BAD_{case}",
        n_layers=1,
        shared=shared,
        after_init=(
            _claim_ready if case == "no_workspace" else _shrink_capacity
        ),
    )
    assert ir.count('symbol = "mega_ffn.ep_fused"') == N_DEVICES
    lengths = [8, 8]
    bufs = [
        Buffer.from_dlpack(x).to(devices[i])
        for i, x in enumerate(_inputs(lengths, 0, case))
    ]
    with pytest.raises(Exception, match=match):
        fused.execute(*bufs, *init.model_inputs())
    ship, ship_init, _ = _build_arm(
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        f"SHIP_{case}",
        n_layers=1,
        shared=shared,
    )
    _assert_finite(
        _ship_reference(ship, ship_init, bufs, n_layers=1),
        lengths,
        f"{fmt} {case}: shipping arm after the refused launch",
    )
    print(f"{fmt} {case}: refused on the host before any launch")


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
def test_ep2_fused_and_shipping_share_an_ep_instance() -> None:
    """NVFP4: a fused and a shipping model on ONE EP instance, executed
    alternately across generations. The fused counters sit in the reserved
    words of the counter buffers the shipping kernels use; the shipping model
    on the shared instance must match one on its own instance bit for bit,
    and the fused model must match it as everywhere else."""
    fmt = "nvfp4"
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)
    wrapped = _weights(fmt)
    fused, init, _ = _build_arm(
        fmt, True, wrapped, session, devices_ref, "FULL_S"
    )
    ship_s, _, ship_s_ir = _build_arm(
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP_S",
        n_layers=1,
        ep_init_from=init,
    )
    assert 'symbol = "mega_ffn.ep_fused"' not in ship_s_ir
    ship, ship_init, _ = _build_arm(
        fmt, False, wrapped, session, devices_ref, "SHIP_OWN", n_layers=1
    )
    for cell, lengths in CELLS.items():
        bufs = [
            Buffer.from_dlpack(
                torch.zeros(n, HIDDEN_DIM, dtype=torch.bfloat16)
            ).to(devices[i])
            for i, n in enumerate(lengths)
        ]
        for gen in range(N_GENERATIONS):
            xs = _inputs(lengths, gen, f"shared{cell}")
            for i, x in enumerate(xs):
                bufs[i].inplace_copy_from(Buffer.from_dlpack(x).to(devices[i]))
            where = f"{cell} gen {gen}"
            ref = _ship_reference(ship, ship_init, bufs)
            got = _snapshot(fused.execute(*bufs, *init.model_inputs()))
            _compare(ref, got, lengths, where + " fused on the shared instance")
            shared_ref = _ship_reference(ship_s, init, bufs)
            for i in range(N_DEVICES):
                assert torch.equal(ref[i], shared_ref[i]), (
                    f"{where} rank {i}: shipping on the shared instance differs"
                )
    print("nvfp4: fused and shipping alternate on one EP instance", flush=True)


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.skipif(
    os.environ.get("EP2_FUSED_NSYS", "0") != "1",
    reason="nsys subject: set EP2_FUSED_NSYS=1",
)
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_fused_nsys_subject(fmt: str) -> None:
    """Profile subject: a few replays of the captured two-layer FULL graph, then
    of the one-layer shipping graph, at dense24. Run under
    ``nsys profile --cuda-graph-trace=node`` to list the graph nodes and kernel
    identities per replay; the outputs are checked so the replays are real.
    """
    n_rep = int(os.environ.get("EP2_FUSED_NSYS_REPLAYS", "3"))
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)
    wrapped = _weights(fmt)
    shared = fmt == "nvfp4"
    ship1, ship1_init, _ = _build_arm(
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP1",
        n_layers=1,
        shared=shared,
    )
    fused, init, _ = _build_arm(
        fmt, True, wrapped, session, devices_ref, "FULL", shared=shared
    )
    lengths = CELLS["dense24"]
    bufs = [
        Buffer.from_dlpack(x).to(devices[i])
        for i, x in enumerate(_inputs(lengths, 0, "nsys"))
    ]
    ref = _ship_reference(ship1, ship1_init, bufs)
    outs = fused.capture(11, *bufs, *init.model_inputs())
    for d in devices:
        d.synchronize()
    for _ in range(n_rep):
        fused.replay(11, *bufs, *init.model_inputs())
    for d in devices:
        d.synchronize()
    _compare(ref, _snapshot(outs), lengths, f"{fmt} nsys FULL replay")
    outs1 = ship1.capture(12, *bufs, *ship1_init.model_inputs())
    for _ in range(n_rep):
        ship1.replay(12, *bufs, *ship1_init.model_inputs())
    for d in devices:
        d.synchronize()
    _assert_finite(_snapshot(outs1), lengths, f"{fmt} nsys SHIP1 replay")
    print(
        f"NSYS SUBJECT {fmt}: {n_rep} FULL(2-layer) replays then {n_rep} SHIP(1-layer) replays done",
        flush=True,
    )


def _paired(d: list[float]) -> tuple[float, float, float, int]:
    m = statistics.fmean(d)
    h = (
        T975_11 * statistics.stdev(d) / len(d) ** 0.5
        if len(d) > 1
        else float("nan")
    )
    return m, m - h, m + h, sum(1 for x in d if x > 0)


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.skipif(
    os.environ.get("EP2_FUSED_PILOT", "0") != "1",
    reason="pilot: set EP2_FUSED_PILOT=1",
)
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_fused_pilot(fmt: str) -> None:
    """Small current-mode pilot: captured SHIP vs captured FULL, replayed.

    Complete-output scope (token-indexed routed output plus the shared expert at
    NVFP4; MXFP8 is routed-only,
    two chained layers), the same execution mode for both arms (one graph
    replay per iteration), paired ABBA within each of 12 pairs per cell, host
    wall over N back-to-back replays plus a device sync (so it includes the
    replay enqueue; no host-bound guard beyond that). FULL eager executes are
    timed too, as a separate labelled row (the runtime cost of the graph
    path vs eager, not a fusion gain).
    """
    n_pairs = int(os.environ.get("EP2_FUSED_PILOT_PAIRS", "12"))
    n_iter = int(os.environ.get("EP2_FUSED_PILOT_ITERS", "200"))
    n_warm = int(os.environ.get("EP2_FUSED_PILOT_WARM", "20"))
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)
    wrapped = _weights(fmt)
    shared = fmt == "nvfp4"
    ship, ship_init, _ = _build_arm(
        fmt, False, wrapped, session, devices_ref, "SHIP", shared=shared
    )
    ship1, ship1_init, _ = _build_arm(
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP1",
        n_layers=1,
        shared=shared,
    )
    fused, init, _ = _build_arm(
        fmt, True, wrapped, session, devices_ref, "FULL", shared=shared
    )
    # The engine cannot capture a graph whose shared expert runs on the
    # side stream (unjoined at capture). Where that is the shipping
    # configuration (NVFP4), a second shipping arm is built with the shipping
    # knob MODULAR_OVERLAP_SHARED_EXPERT=0 (shared expert inline; nothing else
    # changes) so a captured, mode-matched baseline exists. Built lazily.
    ship_no: tuple[Model, EPCommInitializer, str] | None = None

    def ship_nooverlap() -> tuple[Model, EPCommInitializer, str]:
        nonlocal ship_no
        if ship_no is None:
            prev = os.environ.get("MODULAR_OVERLAP_SHARED_EXPERT")
            os.environ["MODULAR_OVERLAP_SHARED_EXPERT"] = "0"
            try:
                ship_no = _build_arm(
                    fmt,
                    False,
                    wrapped,
                    session,
                    devices_ref,
                    "SHIP_NOOVL",
                    shared=shared,
                )
            finally:
                if prev is None:
                    del os.environ["MODULAR_OVERLAP_SHARED_EXPERT"]
                else:
                    os.environ["MODULAR_OVERLAP_SHARED_EXPERT"] = prev
        return ship_no

    def sync() -> None:
        for d in devices:
            d.synchronize()

    def timed(fn: Callable[[], None], n: int) -> float:
        fn()  # one enqueue outside the window keeps the first launch's setup out
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        return (time.perf_counter() - t0) / n * 1e6

    def qualified(
        outs: list[Buffer],
        ref: list[torch.Tensor],
        lengths: list[int],
        where: str,
    ) -> str | None:
        """None when finite and matching the reference, else the reason."""
        got = _snapshot(outs)
        for i in range(N_DEVICES):
            if lengths[i] == 0:
                continue
            if not torch.isfinite(got[i]).all():
                return f"{where} rank {i}: non-finite output"
            cos = (
                torch.nn.functional.cosine_similarity(ref[i], got[i], dim=-1)
                .min()
                .item()
            )
            if cos <= 0.9999:
                return f"{where} rank {i}: cosine {cos:.6f}"
        return None

    results: dict[str, dict[str, object]] = {}

    def run_cell(cell: str, lengths: list[int]) -> dict[str, object]:
        bufs = [
            Buffer.from_dlpack(x).to(devices[i])
            for i, x in enumerate(_inputs(lengths, 0, "pilot"))
        ]
        ref = _ship_reference(ship1, ship1_init, bufs)
        nan_bufs = [
            Buffer.from_dlpack(
                torch.full((n, HIDDEN_DIM), float("nan"), dtype=torch.bfloat16)
            ).to(devices[i])
            for i, n in enumerate(lengths)
        ]

        def poison(outs: list[Buffer]) -> None:
            for i, o in enumerate(outs):
                if lengths[i] > 0:
                    o.inplace_copy_from(nan_bufs[i])

        # Captured arms. Each is qualified on its SECOND replay (poisoned
        # before it): finite and matching the layer-by-layer reference. The
        # fused arm MUST capture; a shipping graph that the engine refuses to
        # capture (best-effort capture; the NVFP4 shipping graph carries side
        # streams) is recorded as not qualified, with the reason.
        quals: dict[str, str | None] = {}
        replays: dict[str, tuple[Callable[[], None], list[Buffer]]] = {}
        outs_full = fused.capture(2, *bufs, *init.model_inputs())

        def replay_full() -> None:
            fused.replay(2, *bufs, *init.model_inputs())

        replays["FULL_graph"] = (replay_full, outs_full)
        try:
            outs_ship = ship.capture(1, *bufs, *ship_init.model_inputs())

            def replay_ship() -> None:
                ship.replay(1, *bufs, *ship_init.model_inputs())

            replays["SHIP_graph"] = (replay_ship, outs_ship)
        except Exception as e:  # recorded, not hidden
            quals["SHIP_graph"] = (
                f"capture refused: {type(e).__name__}: {str(e)[:200]}"
            )
        captured: list[tuple[Model, int]] = [(fused, 2)]
        if "SHIP_graph" in replays:
            captured.append((ship, 1))

        def qualify(name: str) -> None:
            fn, outs = replays[name]
            fn()
            poison(outs)
            fn()
            sync()
            quals[name] = qualified(outs, ref, lengths, f"{fmt} {cell} {name}")

        for name in list(replays):
            qualify(name)
        assert quals["FULL_graph"] is None, quals["FULL_graph"]
        if quals.get("SHIP_graph") is not None:
            # Fallback 1 (same configuration, two graph launches): the
            # one-layer shipping graph twice.
            try:
                outs_s1a = ship1.capture(3, *bufs, *ship1_init.model_inputs())
                outs_s1b = ship1.capture(
                    4, *outs_s1a, *ship1_init.model_inputs()
                )
                captured.extend([(ship1, 3), (ship1, 4)])

                def replay_s1x2() -> None:
                    ship1.replay(3, *bufs, *ship1_init.model_inputs())
                    ship1.replay(4, *outs_s1a, *ship1_init.model_inputs())

                replays["SHIP1x2_graph"] = (replay_s1x2, outs_s1b)
                qualify("SHIP1x2_graph")
            except Exception as e:
                quals["SHIP1x2_graph"] = (
                    f"capture refused: {type(e).__name__}: {str(e)[:200]}"
                )
        if (
            quals.get("SHIP_graph") is not None
            and quals.get("SHIP1x2_graph") is not None
        ):
            # Fallback 2 (one deviation: shared expert inline instead of on
            # the side stream), one captured two-layer graph.
            try:
                sn, sn_init, _ = ship_nooverlap()
                outs_sn = sn.capture(5, *bufs, *sn_init.model_inputs())
                captured.append((sn, 5))

                def replay_sn() -> None:
                    sn.replay(5, *bufs, *sn_init.model_inputs())

                replays["SHIP_nooverlap_graph"] = (replay_sn, outs_sn)
                qualify("SHIP_nooverlap_graph")
            except Exception as e:
                quals["SHIP_nooverlap_graph"] = (
                    f"capture refused: {type(e).__name__}: {str(e)[:200]}"
                )
        eager_q = qualified(
            fused.execute(*bufs, *init.model_inputs()),
            ref,
            lengths,
            f"{fmt} {cell} FULL_eager",
        )
        assert eager_q is None, eager_q

        def eager_s1x2() -> None:
            cur: list[Buffer] = list(bufs)
            for _ in range(N_LAYERS):
                cur = ship1.execute(*cur, *ship1_init.model_inputs())

        # The eager two-layer shipping chain (layer by layer) is always
        # qualified by construction (it IS the reference) and always timed.
        quals["SHIP1x2_eager"] = None
        for name, q in quals.items():
            print(
                f"PILOT {fmt} {cell}: {name} "
                + ("QUALIFIED" if q is None else f"NOT qualified ({q})"),
                flush=True,
            )

        arms: dict[str, Callable[[], None]] = {"FULL_graph": replay_full}
        for name in ("SHIP_graph", "SHIP1x2_graph", "SHIP_nooverlap_graph"):
            if name in replays and quals.get(name) is None:
                arms[name] = replays[name][0]
        arms["SHIP1x2_eager"] = eager_s1x2

        def eager_full() -> None:
            fused.execute(*bufs, *init.model_inputs())

        arms["FULL_eager"] = eager_full
        for fn in arms.values():
            timed(fn, n_warm)
        rows: dict[str, list[float]] = {k: [] for k in arms}
        for p in range(n_pairs):
            order = list(arms) if p % 2 == 0 else list(reversed(arms))
            for name in order:
                rows[name].append(timed(arms[name], n_iter))
        # Primary baseline: the same execution mode (one captured graph) when
        # it qualified, else two captured one-layer graphs, else the eager
        # layer-by-layer chain (a MODE mismatch, recorded as such).
        primary = next(
            n
            for n in (
                "SHIP_graph",
                "SHIP1x2_graph",
                "SHIP_nooverlap_graph",
                "SHIP1x2_eager",
            )
            if n in arms
        )
        d = [
            b - f
            for b, f in zip(rows[primary], rows["FULL_graph"], strict=True)
        ]
        d2 = [
            b - f
            for b, f in zip(
                rows["SHIP1x2_eager"], rows["FULL_graph"], strict=True
            )
        ]
        eg = [
            g - f
            for g, f in zip(rows["FULL_eager"], rows["FULL_graph"], strict=True)
        ]
        m, lo, hi, pos = _paired(d)
        m2, lo2, hi2, pos2 = _paired(d2)
        me, loe, hie, pose = _paired(eg)
        result: dict[str, object] = {
            "lengths": lengths,
            "n_pairs": n_pairs,
            "n_iter": n_iter,
            "qualification": quals,
            "primary_baseline": primary,
            "means_us": {k: statistics.fmean(v) for k, v in rows.items()},
            "samples_us": rows,
            "primary_minus_FULL_graph": {
                "mean": m,
                "lo": lo,
                "hi": hi,
                "pos": pos,
            },
            "SHIP1x2_eager_minus_FULL_graph": {
                "mean": m2,
                "lo": lo2,
                "hi": hi2,
                "pos": pos2,
            },
            "mode_matched": primary != "SHIP1x2_eager",
            "primary_deviation": (
                "shared expert inline (MODULAR_OVERLAP_SHARED_EXPERT=0) instead of the shipping side stream"
                if primary == "SHIP_nooverlap_graph"
                else (
                    "eager layer-by-layer chain vs one captured graph"
                    if primary == "SHIP1x2_eager"
                    else None
                )
            ),
            "FULL_eager_minus_FULL_graph": {
                "mean": me,
                "lo": loe,
                "hi": hie,
                "pos": pose,
            },
        }
        print(
            f"PILOT {fmt} {cell}: "
            + ", ".join(
                f"{k} {statistics.fmean(v):.2f} us" for k, v in rows.items()
            )
            + f"; {primary}-FULL_graph {m:+.2f} [{lo:+.2f}, {hi:+.2f}] {pos}/{n_pairs}"
            + f"; SHIP1x2_eager-FULL_graph {m2:+.2f} [{lo2:+.2f}, {hi2:+.2f}] {pos2}/{n_pairs}"
            + f"; eager-graph (FULL) {me:+.2f} [{loe:+.2f}, {hie:+.2f}]",
            flush=True,
        )
        # Release this cell's captured graphs (their working memory is pinned
        # while captured; the memory manager's cache is finite).
        for model, k in captured:
            model.release_captured_graph(k)
        return result

    for cell in ("dense24", "ragged"):
        results[cell] = run_cell(cell, CELLS[cell])
    out = os.environ.get("EP2_FUSED_PILOT_OUT")
    if out:
        with open(os.path.join(out, f"pilot_{fmt}.json"), "w") as f:
            json.dump(results, f, indent=1)
