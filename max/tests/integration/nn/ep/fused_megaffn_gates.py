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
"""Runtime tests of the one-launch fused EP MoE (``mega_ffn.ep_fused``) at any
EP size, shared by the EP8, EP4 and EP2 64- and 128-local tests.

Arm A is the shipping expert-parallel MoE (``ep.dispatch_async`` -> side-stream
shared expert -> ``ep.dispatch_wait`` -> FFN -> combine; at NVFP4 the default
``mega_ffn.ep_combine_send`` role). Arm B is ``mega_ffn.ep_fused``: dispatch,
MegaFFN and the weighted combine in ONE launch per rank, the shared expert
inline on the default stream. Both arms see the same weights, routing, inputs
and devices; the MXFP8 comparisons are routed-only (no shared expert in either
arm). Every arm goes through production EP init (``mega_ffn.ep_fused_init``
with ``MODULAR_EP_FUSED_MOE=1``), the graph op and the selector.

A test describes its deployment with a `Geometry` and uses the devices
``0 .. n_devices - 1`` explicitly. The EP2 test keeps its own copy of these
gates.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from unittest import mock

import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    Shape,
    ShardingStrategy,
    TensorType,
    ops,
)
from max.graph.weights import WeightData
from max.nn.comm.ep import (
    EPBatchManager,
    EPCommInitializer,
    EPConfig,
    ep_manager,
)
from max.nn.comm.ep.ep_manager import (
    _bound_fused_moe_workspace,
    _plan_fused_moe_workspace,
)
from max.nn.comm.ep.ep_manager import logger as ep_init_logger
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

FORMATS = ("nvfp4", "mxfp8")

FUSED_OP = 'symbol = "mega_ffn.ep_fused"'

# Matches the binding's host check that refuses a batch past the capacity.
OVERFLOW_MATCH = "max_token_per_rank"


@dataclasses.dataclass(frozen=True)
class Geometry:
    """One EP deployment of the fused MoE."""

    n_devices: int
    num_experts: int
    max_tokens_per_rank: int
    hidden_dim: int = 6144
    moe_dim: int = 2048
    top_k: int = 8
    n_layers: int = 2
    """Chained MoE layers per graph: consecutive fused launches on one
    workspace, as every layer of a model sees them."""
    n_generations: int = 4
    """Changing generations per cell: >= 3, so both bank parities recur."""


def devices(geo: Geometry) -> tuple[list[Accelerator], list[DeviceRef]]:
    """The devices ``0 .. n_devices - 1``, whatever else is visible."""
    accs = [Accelerator(i) for i in range(geo.n_devices)]
    return accs, [DeviceRef(d.label, d.id) for d in accs]


# ----------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------
def _shared_bf16(
    geo: Geometry, w: dict[str, torch.Tensor], device: str
) -> None:
    for name, shape in (
        ("gate_proj", (geo.moe_dim, geo.hidden_dim)),
        ("up_proj", (geo.moe_dim, geo.hidden_dim)),
        ("down_proj", (geo.hidden_dim, geo.moe_dim)),
    ):
        w[f"shared_experts.{name}.weight"] = (
            torch.randn(*shape, dtype=torch.bfloat16, device=device) * 1e-2
        )


def _router(geo: Geometry, device: str) -> torch.Tensor:
    return (
        torch.randn(
            geo.num_experts, geo.hidden_dim, dtype=torch.bfloat16, device=device
        )
        * 1e-3
    )


def _nvfp4_weights(geo: Geometry) -> dict[str, torch.Tensor]:
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
        "gate.gate_score.weight": _router(geo, device)
    }
    for e in range(geo.num_experts):
        s2 = _proj(w, f"experts.{e}.gate_proj", geo.moe_dim, geo.hidden_dim)
        _proj(
            w,
            f"experts.{e}.up_proj",
            geo.moe_dim,
            geo.hidden_dim,
            weight_scale_2=s2,
        )
        # Non-unit, per-expert down input scales: the fused op must invert
        # them exactly as the shipping chain does.
        _proj(
            w,
            f"experts.{e}.down_proj",
            geo.hidden_dim,
            geo.moe_dim,
            input_scale=0.75 + 0.5 * (e % 4) / 3,
        )
    _shared_bf16(geo, w, device)
    return w


def _mxfp8_weights(geo: Geometry) -> dict[str, torch.Tensor]:
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
        "gate.gate_score.weight": _router(geo, device)
    }
    for e in range(geo.num_experts):
        _proj(w, f"experts.{e}.gate_proj", geo.moe_dim, geo.hidden_dim)
        _proj(w, f"experts.{e}.up_proj", geo.moe_dim, geo.hidden_dim)
        _proj(w, f"experts.{e}.down_proj", geo.hidden_dim, geo.moe_dim)
    # The cuda MXFP8 EP role keeps the shared expert's rows in the dispatch,
    # which stacks its weights with the routed experts': same format.
    _proj(w, "shared_experts.gate_proj", geo.moe_dim, geo.hidden_dim)
    _proj(w, "shared_experts.up_proj", geo.moe_dim, geo.hidden_dim)
    _proj(w, "shared_experts.down_proj", geo.hidden_dim, geo.moe_dim)
    return w


def wrap(
    weights: Mapping[str, torch.Tensor],
) -> dict[str, WeightData | torch.Tensor]:
    """Moves weights to the host, as `WeightData` where MAX needs the
    dtype."""
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


def weights(geo: Geometry, fmt: str) -> dict[str, WeightData | torch.Tensor]:
    """Synthetic weights of ``fmt``, generated on ``cuda:0`` and kept on the
    host; the generation's device memory is returned to the driver."""
    wrapped = wrap(
        _nvfp4_weights(geo) if fmt == "nvfp4" else _mxfp8_weights(geo)
    )
    torch.cuda.empty_cache()
    return wrapped


def quant_config(fmt: str) -> QuantConfig:
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


# ----------------------------------------------------------------------------
# Arms
# ----------------------------------------------------------------------------
def ship_combine_send(fmt: str) -> bool:
    """Whether the shipping arm fuses the combine send into the MegaFFN
    epilogue: the NVFP4 serving roles do, and it is NVFP4-only."""
    return fmt == "nvfp4"


@contextlib.contextmanager
def fused_moe_request(on: bool) -> Iterator[None]:
    """Sets the internal developer switch EP init reads, or leaves it unset
    (the default: no opt-in), for the duration of the block."""
    saved = os.environ.pop("MODULAR_EP_FUSED_MOE", None)
    try:
        if on:
            os.environ["MODULAR_EP_FUSED_MOE"] = "1"
        yield
    finally:
        os.environ.pop("MODULAR_EP_FUSED_MOE", None)
        if saved is not None:
            os.environ["MODULAR_EP_FUSED_MOE"] = saved


class _WarningRecords(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def ep_init_warnings() -> Iterator[list[str]]:
    """Collects the warnings EP init logs inside the block, whatever the
    logger's propagation and level elsewhere."""
    handler = _WarningRecords()
    level = ep_init_logger.level
    ep_init_logger.addHandler(handler)
    if ep_init_logger.getEffectiveLevel() > logging.WARNING:
        ep_init_logger.setLevel(logging.WARNING)
    try:
        yield handler.messages
    finally:
        ep_init_logger.removeHandler(handler)
        ep_init_logger.setLevel(level)


def ep_config(
    geo: Geometry,
    fmt: str,
    fused: bool,
    shared: bool = True,
    max_tpr: int | None = None,
    num_experts: int | None = None,
    use_allreduce: bool = False,
) -> EPConfig:
    """The EP configuration of one arm.

    The shipping roles per format: MXFP8 on cuda keeps the shared expert's
    rows in the dispatch (``fused_shared_expert``; the plain block-scaled
    ``dispatch_wait`` asserts NVFP4 packing, so that is the only cuda MXFP8
    chain with a shared expert). NVFP4 serving roles fuse the combine send
    into the MegaFFN epilogue, an NVFP4-only op. The fused MoE is qualified
    with an unfused shared expert only.
    """
    return EPConfig(
        dispatch_dtype=DType.uint8 if fmt == "nvfp4" else DType.float8_e4m3fn,
        combine_dtype=DType.bfloat16,
        hidden_size=geo.hidden_dim,
        top_k=geo.top_k,
        n_experts=geo.num_experts if num_experts is None else num_experts,
        max_tokens_per_rank=(
            geo.max_tokens_per_rank if max_tpr is None else max_tpr
        ),
        n_gpus_per_node=geo.n_devices,
        n_nodes=1,
        dispatch_quant_config=quant_config(fmt),
        fused_shared_expert=(fmt == "mxfp8" and not fused and shared),
        # The send writes into peer buffers, which the allreduce backend has
        # none of.
        fuse_ffn_combine_send=ship_combine_send(fmt) and not use_allreduce,
        moe_dim=geo.moe_dim,
        use_allreduce=use_allreduce,
    )


def build_arm(
    geo: Geometry,
    fmt: str,
    fused: bool,
    wrapped: Mapping[str, WeightData | torch.Tensor],
    session: InferenceSession,
    devices_ref: list[DeviceRef],
    name: str,
    n_layers: int | None = None,
    shared: bool = True,
    max_tpr: int | None = None,
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
    layers = geo.n_layers if n_layers is None else n_layers
    quant = quant_config(fmt)
    dtype = DType.uint8 if fmt == "nvfp4" else DType.float8_e4m3fn
    config = ep_config(geo, fmt, fused, shared=shared, max_tpr=max_tpr)
    if ep_init_from is None:
        ep_comm_init = EPCommInitializer(config)
    else:
        ep_comm_init = ep_init_from
        config = dataclasses.replace(
            ep_init_from.config,
            fused_moe_ready=fused and ep_init_from.config.fused_moe_ready,
        )
    ep_batch_manager = EPBatchManager(config)

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
                        hidden_dim=geo.hidden_dim,
                        num_experts=geo.num_experts,
                        num_experts_per_token=geo.top_k,
                        moe_dim=geo.moe_dim,
                        has_shared_experts=shared,
                        shared_experts_dim=geo.moe_dim if shared else 0,
                        ep_size=geo.n_devices,
                        dtype=dtype,
                        ep_batch_manager=ep_batch_manager,
                        quant_config=quant,
                        # NVFP4: BF16 shared expert (as GLM-5.3 ships it).
                        # MXFP8: the quantized shared expert both arms share,
                        # fused into the dispatch (shipping) or a plain MXFP8
                        # MLP (fused MoE).
                        shared_experts_dtype=(
                            DType.bfloat16 if fmt == "nvfp4" else None
                        ),
                    )
                    for _ in range(layers)
                ]
            )

        def __call__(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError("use the sharded layers directly")

    stack = _Stack()
    layer_shards = []
    for layer in stack.layers:
        assert isinstance(layer, MoEQuantized)
        layer.sharding_strategy = ShardingStrategy.expert_parallel(
            geo.n_devices
        )
        layer_shards.append(list(layer.shard(devices_ref)))
    stack.load_state_dict(
        {
            f"layers.{l}.{k}": v
            for l in range(layers)
            for k, v in wrapped.items()
            if shared or not k.startswith("shared_experts.")
        }
    )
    if ep_init_from is None:
        with fused_moe_request(fused):
            ep_comm_init.ep_init(session)
        ready = fused if expect_ready is None else expect_ready
        assert config.fused_moe_ready == ready, (name, ready)
        assert (ep_comm_init.fused_moe_workspace is not None) == ready
    if after_init is not None:
        after_init(config)

    with Graph(
        name,
        input_types=[
            *(
                TensorType(
                    DType.bfloat16,
                    (f"input_len_{i}", geo.hidden_dim),
                    DeviceRef.GPU(i),
                )
                for i in range(geo.n_devices)
            ),
            *ep_batch_manager.input_types(),
        ],
    ) as graph:
        xs = [x.tensor for x in graph.inputs[: geo.n_devices]]
        ep_batch_manager.fetch_buffers(graph.inputs[geo.n_devices :])
        for shards in layer_shards:
            xs = forward_moe_sharded_layers(shards, xs)
        graph.output(*xs)

    ir = str(graph)
    compiled = session.load(graph, weights_registry=stack.state_dict())
    return compiled, ep_comm_init, ir


def assert_keeps_shipping(ir: str) -> None:
    """No fused op; EP dispatch and combine stay (custom ops on the split
    path, otherwise the graph's own distributed EP ops)."""
    assert FUSED_OP not in ir
    for stage in ("dispatch", "combine"):
        assert (
            f'symbol = "ep.{stage}' in ir or f"mo.distributed.ep.{stage}" in ir
        ), stage


def assert_structure(
    geo: Geometry,
    fused_ir: str,
    ship_ir: str,
    ship_init: EPCommInitializer,
    ship_has_side_stream: bool,
    ship_combine_send: bool,
) -> None:
    # One fused launch per (layer, device), and none of the ops it replaces.
    assert fused_ir.count(FUSED_OP) == geo.n_layers * geo.n_devices
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
    # The arm EP init did not set the fused path up for: no fused workspace,
    # and the shipping chain and its FFN + combine-send selector unchanged.
    assert_keeps_shipping(ship_ir)
    assert not ship_init.config.fused_moe_ready
    assert ship_init.fused_moe_workspace is None
    assert (
        'symbol = "mega_ffn.ep_combine_send"' in ship_ir
    ) == ship_combine_send
    assert ("mo.sequence" in ship_ir) == ship_has_side_stream


def inputs(
    geo: Geometry, lengths: list[int], gen: int, tag: str
) -> list[torch.Tensor]:
    torch.manual_seed(7919 * gen + 31 * len(tag) + sum(lengths))
    return [
        torch.randn(n, geo.hidden_dim, dtype=torch.bfloat16) for n in lengths
    ]


def snapshot(bufs: list[Buffer]) -> list[torch.Tensor]:
    return [torch.from_dlpack(b).cpu().float().clone() for b in bufs]


def ship_reference(
    ship: Model,
    ship_init: EPCommInitializer,
    bufs: list[Buffer],
    n_layers: int,
) -> list[torch.Tensor]:
    """The shipping chain's output for ``n_layers`` chained layers, from a
    ONE-layer shipping graph executed layer by layer (its outputs fed back
    in), the form that stays finite under repeated execution with changing
    inputs."""
    cur: list[Buffer] = list(bufs)
    for _ in range(n_layers):
        cur = ship.execute(*cur, *ship_init.model_inputs())
    return snapshot(cur)


def assert_finite(
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


def compare(
    geo: Geometry,
    ref: list[torch.Tensor],
    out: list[torch.Tensor],
    lengths: list[int],
    where: str,
) -> int:
    """Fused vs shipping: shapes, finite, non-zero and row cosine > 0.9999
    on every rank with tokens; returns how many ranks matched bit for bit."""
    exact = 0
    for i, (r, o) in enumerate(zip(ref, out, strict=True)):
        assert r.shape == o.shape == (lengths[i], geo.hidden_dim), where
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


# ----------------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------------
def graph_replay(
    geo: Geometry,
    fmt: str,
    cells: Mapping[str, list[int]],
    wrapped: Mapping[str, WeightData | torch.Tensor] | None = None,
) -> int:
    """Two fused models on two independent EP instances, each captured ONCE
    and replayed across changing generations and both parities on
    ``cells``, against the eager one-layer shipping reference executed layer
    by layer; then a batch past the capacity, which the binding must refuse
    on the host before any launch, and the workspaces' health after it.
    Returns the token block EP init reported."""
    accs, devices_ref = devices(geo)
    session = InferenceSession(devices=accs)
    if wrapped is None:
        wrapped = weights(geo, fmt)
    shared = fmt == "nvfp4"
    ship, ship_init, ship_ir = build_arm(
        geo,
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP",
        n_layers=1,
        shared=shared,
    )
    fused_a, init_a, fused_ir = build_arm(
        geo, fmt, True, wrapped, session, devices_ref, "FULL_A", shared=shared
    )
    fused_b, init_b, _ = build_arm(
        geo, fmt, True, wrapped, session, devices_ref, "FULL_B", shared=shared
    )
    assert_structure(
        geo,
        fused_ir,
        ship_ir,
        ship_init,
        ship_has_side_stream=shared,
        ship_combine_send=ship_combine_send(fmt),
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
        assert len(lengths) == geo.n_devices, cell
        assert max(lengths) <= geo.max_tokens_per_rank, cell
        # Fixed device buffers per cell: a captured graph binds them, and the
        # generations below rewrite them in place.
        bufs = [
            Buffer.from_dlpack(
                torch.zeros(n, geo.hidden_dim, dtype=torch.bfloat16)
            ).to(accs[i])
            for i, n in enumerate(lengths)
        ]
        nan_bufs = [
            Buffer.from_dlpack(
                torch.full(
                    (n, geo.hidden_dim), float("nan"), dtype=torch.bfloat16
                )
            ).to(accs[i])
            for i, n in enumerate(lengths)
        ]
        arms = [("A", fused_a, init_a, key), ("B", fused_b, init_b, key + 1)]
        key += 2
        outs: dict[str, list[Buffer]] = {}
        prev: dict[str, list[torch.Tensor]] = {}
        for gen in range(geo.n_generations):
            xs = inputs(geo, lengths, gen, f"{fmt}{cell}")
            for i, x in enumerate(xs):
                bufs[i].inplace_copy_from(Buffer.from_dlpack(x).to(accs[i]))
            ref = ship_reference(ship, ship_init, bufs, geo.n_layers)
            assert_finite(
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
                got = snapshot(outs[arm])
                where = f"{fmt} {cell} gen {gen} workspace {arm}"
                exact += compare(geo, ref, got, lengths, where)
                total += sum(1 for n in lengths if n > 0)
                if arm in prev:
                    # Fresh output: the inputs changed, so must the result.
                    changed = any(
                        lengths[i] > 0 and not torch.equal(prev[arm][i], got[i])
                        for i in range(geo.n_devices)
                    )
                    assert changed, (
                        f"{where}: output identical to the previous generation"
                    )
                prev[arm] = got
        for model, k in ((fused_a, key - 2), (fused_b, key - 1)):
            model.release_captured_graph(k)
        print(
            f"{fmt} {cell} {lengths}: {geo.n_generations} generations x 2"
            " workspaces replayed, all within tolerance of the eager shipping"
            " arm",
            flush=True,
        )
    print(
        f"{fmt}: fused (captured, replayed) == shipping (eager) bit-exactly on"
        f" {exact}/{total} (cell, generation, workspace, rank) checks",
        flush=True,
    )

    # A batch past the fused per-rank capacity: the binding's host check
    # refuses it on every rank before any launch, so nothing is poisoned and
    # nothing falls back to another chain.
    over_lengths = [geo.max_tokens_per_rank + 1] * geo.n_devices
    over = [
        Buffer.from_dlpack(x).to(accs[i])
        for i, x in enumerate(inputs(geo, over_lengths, 98, fmt))
    ]
    with pytest.raises(Exception, match=OVERFLOW_MATCH):
        fused_a.execute(*over, *init_a.model_inputs())
    print(f"{fmt}: capacity overflow {over_lengths} refused before launch")
    first = next(iter(cells.values()))
    # The workspaces are still healthy after the refusal.
    xs = inputs(geo, first, 99, fmt)
    ok = [Buffer.from_dlpack(x).to(accs[i]) for i, x in enumerate(xs)]
    ref = ship_reference(ship, ship_init, ok, geo.n_layers)
    for model, init in ((fused_a, init_a), (fused_b, init_b)):
        got = snapshot(model.execute(*ok, *init.model_inputs()))
        compare(geo, ref, got, first, f"{fmt} post-refusal")
    return token_block


def ship_reference_repeats(
    geo: Geometry, fmt: str, layers: int, cells: Mapping[str, list[int]]
) -> None:
    """The shipping arm alone (``layers`` chained layers, one module each, the
    shared expert on its side stream), executed eagerly across changing
    generations.

    The reference the fused gates compare against must itself be healthy
    under repeated execution with changing routes. Each output is read twice,
    straight after ``execute`` and after a device sync, so a stale read would
    show as a difference; the same inputs are then executed again, so a stale
    protocol state would show as a non-repeatable result.
    """
    accs, devices_ref = devices(geo)
    session = InferenceSession(devices=accs)
    ship, ship_init, _ = build_arm(
        geo,
        fmt,
        False,
        weights(geo, fmt),
        session,
        devices_ref,
        "SHIP_R",
        n_layers=layers,
    )
    for cell, lengths in cells.items():
        bufs = [
            Buffer.from_dlpack(
                torch.zeros(n, geo.hidden_dim, dtype=torch.bfloat16)
            ).to(accs[i])
            for i, n in enumerate(lengths)
        ]
        for gen in range(geo.n_generations):
            xs = inputs(geo, lengths, gen, f"{fmt}{cell}")
            for i, x in enumerate(xs):
                bufs[i].inplace_copy_from(Buffer.from_dlpack(x).to(accs[i]))
            outs = ship.execute(*bufs, *ship_init.model_inputs())
            fast = snapshot(outs)
            for d in accs:
                d.synchronize()
            synced = snapshot(outs)
            where = f"{fmt} {cell} gen {gen} shipping"
            assert_finite(synced, lengths, where + " (read after sync)")
            assert_finite(fast, lengths, where + " (read after execute)")
            for i in range(geo.n_devices):
                assert torch.equal(fast[i], synced[i]), (
                    f"{where} rank {i}: the read after execute differs from"
                    " the read after a device sync"
                )
            again = snapshot(ship.execute(*bufs, *ship_init.model_inputs()))
            assert_finite(again, lengths, where + " (second execution)")
            for i in range(geo.n_devices):
                assert torch.equal(synced[i], again[i]), (
                    f"{where} rank {i}: two executions of the same inputs"
                    " differ"
                )
        print(
            f"{fmt} {cell} ({layers} layer(s) per graph): shipping arm finite"
            f" and repeatable over {geo.n_generations} generations",
            flush=True,
        )


class _ArrivalCountResize:
    """``ep_manager``'s ``ops`` with every ``arrival_count`` the combine send
    mints resized to ``words``, so the send's host check sees that size."""

    def __init__(self, words: int) -> None:
        self.words = words
        self.resized = 0

    def __getattr__(self, name: str) -> object:
        return getattr(ops, name)

    def buffer_create(
        self, type: BufferType, init_value: float | bool | None = None
    ) -> BufferValue:
        assert type.dtype == DType.uint32, type
        self.resized += 1
        return ops.buffer_create(
            BufferType(DType.uint32, shape=[self.words], device=type.device),
            init_value=init_value,
        )


def ship_arrival_count_refused(
    geo: Geometry, words: int, lengths: list[int]
) -> None:
    """The NVFP4 shipping arm, one layer, with its combine send handed an
    ``arrival_count`` of ``words`` words: execution must fail on the send's
    host check, which names the size, before any launch reads the buffer.
    """
    resize = _ArrivalCountResize(words)
    accs, devices_ref = devices(geo)
    session = InferenceSession(devices=accs)
    with mock.patch.object(ep_manager, "ops", resize):
        ship, ship_init, _ = build_arm(
            geo,
            "nvfp4",
            False,
            weights(geo, "nvfp4"),
            session,
            devices_ref,
            "SHIP_ARRIVAL",
            n_layers=1,
        )
    # One combine send per device and layer.
    assert resize.resized == geo.n_devices, resize.resized
    bufs = [
        Buffer.from_dlpack(x).to(accs[i])
        for i, x in enumerate(inputs(geo, lengths, 0, "arrival"))
    ]
    with pytest.raises(ValueError, match=f"arrival_count has {words} words"):
        ship.execute(*bufs, *ship_init.model_inputs())
    print(
        f"nvfp4: an arrival_count of {words} words refused by the combine"
        " send's host check",
        flush=True,
    )


def mode_vs_ship(
    geo: Geometry, fmt: str, mode: str, cells: Mapping[str, list[int]]
) -> None:
    """One fused arm against the shipping arm, eager or captured.

    A bisect aid: the shipping reference is checked for finiteness at every
    generation next to exactly one fused arm executed either eagerly or as
    one captured graph replayed, so a reference defect can be attributed to
    fused eager execution or to capture/replay. The reference is taken again
    after the fused arm ran and must not have changed.
    """
    accs, devices_ref = devices(geo)
    session = InferenceSession(devices=accs)
    wrapped = weights(geo, fmt)
    shared = fmt == "nvfp4"
    ship, ship_init, _ = build_arm(
        geo,
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP_M",
        n_layers=1,
        shared=shared,
    )
    fused, init, _ = build_arm(
        geo, fmt, True, wrapped, session, devices_ref, "FULL_M", shared=shared
    )
    key = 500
    for cell, lengths in cells.items():
        bufs = [
            Buffer.from_dlpack(
                torch.zeros(n, geo.hidden_dim, dtype=torch.bfloat16)
            ).to(accs[i])
            for i, n in enumerate(lengths)
        ]
        outs: list[Buffer] | None = None
        key += 1
        exact = 0
        checked = 0
        for gen in range(geo.n_generations):
            xs = inputs(geo, lengths, gen, f"{fmt}{cell}")
            for i, x in enumerate(xs):
                bufs[i].inplace_copy_from(Buffer.from_dlpack(x).to(accs[i]))
            where = f"{fmt} {cell} gen {gen} {mode}"
            ref = ship_reference(ship, ship_init, bufs, geo.n_layers)
            assert_finite(ref, lengths, where + " shipping reference")
            if mode == "eager":
                got = snapshot(fused.execute(*bufs, *init.model_inputs()))
            else:
                if outs is None:
                    outs = fused.capture(key, *bufs, *init.model_inputs())
                fused.replay(key, *bufs, *init.model_inputs())
                got = snapshot(outs)
            assert_finite(got, lengths, where + " fused")
            exact += compare(geo, ref, got, lengths, where)
            checked += sum(1 for n in lengths if n > 0)
            ref2 = ship_reference(ship, ship_init, bufs, geo.n_layers)
            assert_finite(
                ref2, lengths, where + " shipping reference after the fused arm"
            )
            for i in range(geo.n_devices):
                assert torch.equal(ref[i], ref2[i]), (
                    f"{where} rank {i}: the shipping arm's output changed"
                    " after the fused arm ran"
                )
        if outs is not None:
            del outs
            fused.release_captured_graph(key)
        print(
            f"{fmt} {cell} {mode}: {geo.n_generations} generations, fused"
            f" within tolerance of shipping, {exact}/{checked} rank outputs"
            " bit-exact",
            flush=True,
        )


def replay_trace_matches(geo: Geometry, fmt: str, lengths: list[int]) -> None:
    """The captured graph's launch sequence equals eager execution's."""
    accs, devices_ref = devices(geo)
    session = InferenceSession(devices=accs)
    fused, init, _ = build_arm(
        geo,
        fmt,
        True,
        weights(geo, fmt),
        session,
        devices_ref,
        "FULL_T",
        shared=(fmt == "nvfp4"),
    )
    bufs = [
        Buffer.from_dlpack(x).to(accs[i])
        for i, x in enumerate(inputs(geo, lengths, 0, "trace"))
    ]
    fused.capture(7, *bufs, *init.model_inputs())
    fused.replay(7, *bufs, *init.model_inputs())
    fused.debug_verify_replay(7, *bufs, *init.model_inputs())
    print(f"{fmt}: captured launch trace verified against eager execution")


def refused_capacity(
    geo: Geometry, fmt: str, max_tpr: int, reason_fragment: str
) -> None:
    """With the switch on, a capacity the backend does not serve gets no
    workspace and a logged refusal that names the rule, and the model keeps
    the shipping chain: never a silent fallback, never a fused launch out of
    its range."""
    accs, devices_ref = devices(geo)
    session = InferenceSession(devices=accs)
    with ep_init_warnings() as warnings:
        _, init, ir = build_arm(
            geo,
            fmt,
            True,
            weights(geo, fmt),
            session,
            devices_ref,
            "REFUSED",
            n_layers=1,
            shared=(fmt == "nvfp4"),
            max_tpr=max_tpr,
            expect_ready=False,
        )
    size, token_block, reason = _plan_fused_moe_workspace(init.config)
    assert (size, token_block) == (0, 0)
    assert reason_fragment in reason, reason
    assert any(reason in m for m in warnings), warnings
    assert_keeps_shipping(ir)
    print(f"{fmt}: capacity {max_tpr} refused at EP init ({reason})")


def refused_at_ep_init(
    geo: Geometry,
    fmt: str,
    reason_fragment: str,
    num_experts: int | None = None,
    use_allreduce: bool = False,
) -> str:
    """EP init alone, with the switch on, for a configuration the backend
    refuses: no workspace, readiness off, the plan query agrees, and the
    refusal is logged. Returns the reason."""
    accs, _ = devices(geo)
    session = InferenceSession(devices=accs)
    config = ep_config(
        geo,
        fmt,
        True,
        shared=False,
        num_experts=num_experts,
        use_allreduce=use_allreduce,
    )
    init = EPCommInitializer(config)
    with ep_init_warnings() as warnings, fused_moe_request(True):
        init.ep_init(session)
    assert not config.fused_moe_ready
    assert init.fused_moe_workspace is None
    size, token_block, reason = _plan_fused_moe_workspace(config)
    assert (size, token_block) == (0, 0)
    assert reason_fragment in reason, reason
    assert any(reason in m for m in warnings), warnings
    return reason


def planned_as_refused(
    geo: Geometry,
    fmt: str,
    reason_fragment: str,
    num_experts: int | None = None,
) -> str:
    """The host plan query, exact and as the memory planner's bound, budgets
    no fused workspace for a configuration the backend refuses, and both name
    the same rule. Returns the reason."""
    config = ep_config(geo, fmt, True, shared=False, num_experts=num_experts)
    size, token_block, reason = _plan_fused_moe_workspace(config)
    assert (size, token_block) == (0, 0)
    assert reason_fragment in reason, reason
    assert _bound_fused_moe_workspace(config, config.dispatch_dtype) == (
        0,
        reason,
    )
    return reason


def _claim_ready(config: EPConfig) -> None:
    config.fused_moe_ready = True


def launch_fails_closed(
    geo: Geometry,
    fmt: str,
    case: str,
    match: str,
    smaller_capacity: int,
) -> None:
    """A fused launch whose EP instance has no workspace (EP init was not
    asked for one), or a workspace of another layout (the graph built for
    ``smaller_capacity``), raises on the host before any rank launches; the
    devices stay usable."""
    accs, devices_ref = devices(geo)
    session = InferenceSession(devices=accs)
    wrapped = weights(geo, fmt)
    shared = fmt == "nvfp4"

    def _shrink_capacity(config: EPConfig) -> None:
        config.max_tokens_per_rank = smaller_capacity

    fused, init, ir = build_arm(
        geo,
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
    assert ir.count(FUSED_OP) == geo.n_devices
    lengths = [min(8, smaller_capacity)] * geo.n_devices
    bufs = [
        Buffer.from_dlpack(x).to(accs[i])
        for i, x in enumerate(inputs(geo, lengths, 0, case))
    ]
    with pytest.raises(Exception, match=match):
        fused.execute(*bufs, *init.model_inputs())
    ship, ship_init, _ = build_arm(
        geo,
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        f"SHIP_{case}",
        n_layers=1,
        shared=shared,
    )
    assert_finite(
        ship_reference(ship, ship_init, bufs, n_layers=1),
        lengths,
        f"{fmt} {case}: shipping arm after the refused launch",
    )
    print(f"{fmt} {case}: refused on the host before any launch")


def shared_instance(
    geo: Geometry, fmt: str, cells: Mapping[str, list[int]]
) -> None:
    """A fused and a shipping model on ONE EP instance, executed alternately
    across generations. The fused counters sit in the reserved words of the
    counter buffers the shipping kernels use; the shipping model on the
    shared instance must match one on its own instance bit for bit, and the
    fused model must stay within the same tolerance of it as everywhere
    else."""
    accs, devices_ref = devices(geo)
    session = InferenceSession(devices=accs)
    wrapped = weights(geo, fmt)
    shared = fmt == "nvfp4"
    fused, init, _ = build_arm(
        geo, fmt, True, wrapped, session, devices_ref, "FULL_S", shared=shared
    )
    ship_s, _, ship_s_ir = build_arm(
        geo,
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP_S",
        n_layers=1,
        shared=shared,
        ep_init_from=init,
    )
    assert FUSED_OP not in ship_s_ir
    ship, ship_init, _ = build_arm(
        geo,
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP_OWN",
        n_layers=1,
        shared=shared,
    )
    for cell, lengths in cells.items():
        bufs = [
            Buffer.from_dlpack(
                torch.zeros(n, geo.hidden_dim, dtype=torch.bfloat16)
            ).to(accs[i])
            for i, n in enumerate(lengths)
        ]
        for gen in range(geo.n_generations):
            xs = inputs(geo, lengths, gen, f"shared{cell}")
            for i, x in enumerate(xs):
                bufs[i].inplace_copy_from(Buffer.from_dlpack(x).to(accs[i]))
            where = f"{fmt} {cell} gen {gen}"
            ref = ship_reference(ship, ship_init, bufs, geo.n_layers)
            got = snapshot(fused.execute(*bufs, *init.model_inputs()))
            compare(
                geo, ref, got, lengths, where + " fused on the shared instance"
            )
            shared_ref = ship_reference(ship_s, init, bufs, geo.n_layers)
            for i in range(geo.n_devices):
                assert torch.equal(ref[i], shared_ref[i]), (
                    f"{where} rank {i}: shipping on the shared instance differs"
                )
    print(f"{fmt}: fused and shipping alternate on one EP instance", flush=True)


def nsys_subject(
    geo: Geometry, fmt: str, lengths: list[int], n_rep: int
) -> None:
    """Profile subject: ``n_rep`` replays of the captured fused graph, then of
    the one-layer shipping graph. Under ``nsys profile
    --cuda-graph-trace=node`` each fused replay must show exactly
    ``n_layers`` fused kernels per device and no EP dispatch or combine
    kernel; the outputs are checked so the replays are real."""
    accs, devices_ref = devices(geo)
    session = InferenceSession(devices=accs)
    wrapped = weights(geo, fmt)
    shared = fmt == "nvfp4"
    ship1, ship1_init, _ = build_arm(
        geo,
        fmt,
        False,
        wrapped,
        session,
        devices_ref,
        "SHIP1",
        n_layers=1,
        shared=shared,
    )
    fused, init, _ = build_arm(
        geo, fmt, True, wrapped, session, devices_ref, "FULL", shared=shared
    )
    bufs = [
        Buffer.from_dlpack(x).to(accs[i])
        for i, x in enumerate(inputs(geo, lengths, 0, "nsys"))
    ]
    ref = ship_reference(ship1, ship1_init, bufs, geo.n_layers)
    outs = fused.capture(11, *bufs, *init.model_inputs())
    for d in accs:
        d.synchronize()
    for _ in range(n_rep):
        fused.replay(11, *bufs, *init.model_inputs())
    for d in accs:
        d.synchronize()
    compare(geo, ref, snapshot(outs), lengths, f"{fmt} nsys FULL replay")
    outs1 = ship1.capture(12, *bufs, *ship1_init.model_inputs())
    for _ in range(n_rep):
        ship1.replay(12, *bufs, *ship1_init.model_inputs())
    for d in accs:
        d.synchronize()
    assert_finite(snapshot(outs1), lengths, f"{fmt} nsys SHIP1 replay")
    print(
        f"NSYS SUBJECT {fmt}: {n_rep} FULL ({geo.n_layers}-layer) replays then"
        f" {n_rep} SHIP (1-layer) replays on {geo.n_devices} devices",
        flush=True,
    )
