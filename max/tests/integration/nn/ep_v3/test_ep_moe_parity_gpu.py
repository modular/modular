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
"""Two-GPU expert-parallel ``ExpertParallelMoE`` against one device.

The EP side runs the host path a ModuleV3 pipeline runs: ``start`` sets up
the buffers and ``model_inputs()`` feeds them in the order of
``payload_types``, which the layer reads through ``payload_buffers``. So the
test catches host-side bugs that a graph IR comparison cannot see, such as
buffers passed in the wrong order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from max import functional as F
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.nn import Module
from max.nn.common_layers.quant_moe import ExpertParallelMoE, QuantizedMoE
from max.nn.ep.init import (
    init_graph,
    payload_buffers,
    payload_types,
    start,
)
from max.nn.ep_config import EPConfig
from max.nn.module import InputType
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
    TensorLayout,
)

N_DEVICES = 2
HIDDEN = 256
MOE_DIM = 256
NUM_EXPERTS = 8
TOP_K = 2
MAX_TOKENS_PER_RANK = 64
# Odd, so the two devices get 19 and 18 tokens.
NUM_TOKENS = 37
NUM_MOE_LAYERS = 1

# The smallest top-2 versus third-score gap a token may have. bf16 rounding
# of the router scores cannot flip a pick above it, so both sides route each
# token to the same experts.
_MIN_ROUTER_GAP = 0.1

# One static input scale for every NVFP4 projection. The EP dispatch
# quantizes with the largest gate input scale and the single-device path with
# each expert's own; equal scales make the two the same.
_NVFP4_INPUT_SCALE = 16.0 / (448.0 * 6.0)
# Root mean square of the E2M1 code values, which uniform random bytes pick
# with equal probability.
_FP4_CODE_RMS = float(np.sqrt(np.mean(np.square([0, 0.5, 1, 1.5, 2, 3, 4, 6]))))


@dataclass(frozen=True)
class DispatchFormat:
    """The weight format and the matching EP dispatch format."""

    dispatch_dtype: DType
    quant_config: QuantConfig | None


_FORMATS: dict[str, DispatchFormat] = {
    "bf16": DispatchFormat(DType.bfloat16, None),
    "fp8": DispatchFormat(
        DType.float8_e4m3fn,
        QuantConfig.blockscaled_fp8(
            mlp_quantized_layers={0}, attn_quantized_layers=()
        ),
    ),
    "nvfp4": DispatchFormat(
        DType.uint8,
        QuantConfig.nvfp4(mlp_quantized_layers={0}, attn_quantized_layers=()),
    ),
}


def _assert_allclose(tol: float) -> Callable[[np.ndarray, np.ndarray], None]:
    def check(got: np.ndarray, want: np.ndarray) -> None:
        np.testing.assert_allclose(got, want, rtol=tol, atol=tol)

    return check


def _assert_quantized_parity(
    cosine_bound: float, relative_bound: float
) -> Callable[[np.ndarray, np.ndarray], None]:
    """Bounds each token's direction and size against the reference.

    Cosine similarity alone ignores magnitude, so it would pass a uniform
    scale error, such as router weights applied twice.
    """

    def check(got: np.ndarray, want: np.ndarray) -> None:
        got_norm = np.linalg.norm(got, axis=-1)
        want_norm = np.linalg.norm(want, axis=-1)
        cos = np.sum(got * want, axis=-1) / (got_norm * want_norm)
        relative = np.linalg.norm(got - want, axis=-1) / want_norm
        assert cos.min() > cosine_bound, (
            f"token {int(cos.argmin())}: cosine similarity"
            f" {cos.min():.6f} <= {cosine_bound}"
        )
        assert relative.max() < relative_bound, (
            f"token {int(relative.argmax())}: relative error"
            f" {relative.max():.6f} >= {relative_bound}"
        )

    return check


# The quantized cosine bound is the per-token cosine similarity
# test_ep_moe_fp4.py asserts for NVFP4. The elementwise rtol = atol = 8e-2 of
# test_ep_moe_fp8.py does not fit fp8 here: that test's outputs are far below
# unit scale, and at unit scale the fp8 activation rounding moves single
# elements of either side by up to 0.34 from the exact result.
_CHECKS: dict[str, Callable[[np.ndarray, np.ndarray], None]] = {
    "bf16": _assert_allclose(2e-2),
    "fp8": _assert_quantized_parity(0.995, 0.1),
    "nvfp4": _assert_quantized_parity(0.995, 0.1),
}


@dataclass(frozen=True)
class Case:
    """One parametrization of the parity test."""

    dispatch: str
    eplb: bool
    """Turns on EPLB: an identity ``log2phy``/``logcnt`` placement and the
    per-layer expert-load tally, which adds the layer index input."""

    @property
    def format(self) -> DispatchFormat:
        return _FORMATS[self.dispatch]


@dataclass
class EPRun:
    """An EP layer, compiled and initialized."""

    execute: Callable[[Sequence[Buffer]], list[Buffer]]
    """Runs the layer on one token shard per device; returns one output shard
    per device."""

    expert_load: Callable[[], list[np.ndarray]]
    """Each device's ``[num_moe_layers, n_local_experts]`` tally, or nothing
    when the case does not count expert load."""


def _ep_config(case: Case) -> EPConfig:
    fmt = case.format
    return EPConfig(
        dispatch_dtype=fmt.dispatch_dtype,
        dispatch_quant_config=fmt.quant_config,
        combine_dtype=DType.bfloat16,
        hidden_size=HIDDEN,
        top_k=TOP_K,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=MAX_TOKENS_PER_RANK,
        n_gpus_per_node=N_DEVICES,
        n_nodes=1,
        eplb_enabled=case.eplb,
        eplb_count_expert_load=case.eplb,
        num_moe_layers=NUM_MOE_LAYERS,
    )


def _token_layout(mesh: DeviceMesh) -> TensorLayout:
    return TensorLayout(
        DType.bfloat16,
        ["tokens", HIDDEN],
        DeviceMapping(mesh, (Sharded(axis=0),)),
    )


class _EPHarness(Module[..., Tensor]):
    """Feeds the EP graph inputs to the layer as ``max.nn.ep`` buffers."""

    def __init__(self, moe: ExpertParallelMoE) -> None:
        super().__init__()
        self.moe = moe

    def forward(
        self,
        x: Tensor,
        payload: Mapping[str, Sequence[Tensor]],
        layer_idx: tuple[Tensor, ...],
    ) -> Tensor:
        """``layer_idx`` holds the MoE layer index when expert load counts."""
        buffers = payload_buffers(payload, x.mesh)
        return self.moe(x, layer_idx=next(iter(layer_idx), None), ep=buffers)


def _identity_placement(config: EPConfig) -> tuple[np.ndarray, np.ndarray]:
    shape = (config.num_moe_layers, config.num_logical_experts)
    log2phy = np.broadcast_to(
        np.arange(config.num_logical_experts, dtype=np.int32)[None, :, None],
        (*shape, config.max_replicas),
    )
    return np.ascontiguousarray(log2phy), np.ones(shape, np.int32)


def _layer_ep(
    case: Case,
    weights: Mapping[str, Tensor],
    session: InferenceSession,
    devices: Sequence[Accelerator],
) -> EPRun:
    config = _ep_config(case)
    mesh = DeviceMesh(tuple(devices), (N_DEVICES,), ("ep",))
    with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
        harness = _EPHarness(
            ExpertParallelMoE(
                hidden_dim=HIDDEN,
                num_experts=NUM_EXPERTS,
                num_experts_per_token=TOP_K,
                moe_dim=MOE_DIM,
                quant_config=case.format.quant_config,
                ep_config=config,
            )
        )
    layer_idx_types: tuple[InputType, ...] = tuple(
        TensorLayout(DType.int32, [1], DeviceMapping(mesh, (Replicated(),)))
        for _ in range(config.eplb_count_expert_load)
    )
    # Every graph compiles before EP init touches a device, so a
    # precompile record run captures all of them.
    run = harness.compile(
        _token_layout(mesh),
        payload_types(config),
        layer_idx_types,
        weights={f"moe.{name}": value for name, value in weights.items()},
        session=session,
    )

    init = init_graph(config)
    log2phy, logcnt = (
        [Buffer.from_numpy(table).to(d) for d in devices]
        for table in _identity_placement(config)
    )
    exchange = start(
        config,
        session.load(init.trace(*init.input_types())),
        eplb=(log2phy, logcnt) if config.eplb_enabled else None,
    )
    layer_idx = [
        Buffer.from_numpy(np.zeros(1, np.int32)).to(d)
        for d in devices
        if config.eplb_count_expert_load
    ]

    def execute(x_shards: Sequence[Buffer]) -> list[Buffer]:
        return run.execute_raw(*x_shards, *exchange.model_inputs(), *layer_idx)

    def expert_load() -> list[np.ndarray]:
        # The tallies are the last inputs, one per device.
        n_tallies = N_DEVICES * config.eplb_count_expert_load
        tallies = exchange.buffers[len(exchange.buffers) - n_tallies :]
        return [b.to(CPU()).to_numpy() for b in tallies]

    return EPRun(execute, expert_load)


def _bits(values: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    """Rounds float32 ``values`` to ``dtype`` and returns the raw bytes."""
    rounded = torch.from_numpy(values.astype(np.float32)).to(dtype)
    return rounded.view(torch.uint8).numpy()


def _host(array: np.ndarray, dtype: DType) -> Buffer:
    """Host bytes viewed as ``dtype``; numpy has no bf16 or fp8 dtype."""
    return Buffer.from_numpy(array).view(dtype)


def _in_dim(params: Mapping[str, Tensor], name: str) -> int:
    """The unpacked input width of the projection ``name`` belongs to."""
    weight = params[name.rpartition(".")[0] + ".weight"]
    packing = 2 if weight.dtype == DType.uint8 else 1
    return int(weight.shape[-1]) * packing


def _normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)


def _shape(param: Tensor) -> tuple[int, ...]:
    return tuple(int(d) for d in param.shape)


# Each generator takes the RNG, every parameter, and the name of the one to
# fill. In every format the effective weight has a standard deviation of
# 1 / sqrt(in_dim), so each projection's output stays near unit scale.
_WeightGen = Callable[[np.random.Generator, Mapping[str, Tensor], str], Buffer]


def _bf16_weight(
    rng: np.random.Generator, params: Mapping[str, Tensor], name: str
) -> Buffer:
    values = _normal(rng, _shape(params[name])) / np.sqrt(_in_dim(params, name))
    return _host(_bits(values, torch.bfloat16), DType.bfloat16)


def _fp8_weight(
    rng: np.random.Generator, params: Mapping[str, Tensor], name: str
) -> Buffer:
    values = _normal(rng, _shape(params[name]))
    return _host(_bits(values, torch.float8_e4m3fn), DType.float8_e4m3fn)


def _fp4_weight(
    rng: np.random.Generator, params: Mapping[str, Tensor], name: str
) -> Buffer:
    codes = rng.integers(0, 256, _shape(params[name]), dtype=np.uint8)
    return _host(codes, DType.uint8)


def _fp8_block_scale(
    rng: np.random.Generator, params: Mapping[str, Tensor], name: str
) -> Buffer:
    scale = 1 / np.sqrt(_in_dim(params, name))
    return _host(
        np.full(_shape(params[name]), scale, np.float32), DType.float32
    )


def _fp4_block_scale(
    rng: np.random.Generator, params: Mapping[str, Tensor], name: str
) -> Buffer:
    ones = np.ones(_shape(params[name]), np.float32)
    return _host(_bits(ones, torch.float8_e4m3fn), DType.float8_e4m3fn)


def _fp4_tensor_scale(
    rng: np.random.Generator, params: Mapping[str, Tensor], name: str
) -> Buffer:
    scale = 1 / (_FP4_CODE_RMS * np.sqrt(_in_dim(params, name)))
    return _host(np.array(scale, np.float32), DType.float32)


def _fp4_input_scale(
    rng: np.random.Generator, params: Mapping[str, Tensor], name: str
) -> Buffer:
    return _host(np.array(_NVFP4_INPUT_SCALE, np.float32), DType.float32)


_WEIGHTS_BY_DTYPE: dict[DType, _WeightGen] = {
    DType.bfloat16: _bf16_weight,
    DType.float8_e4m3fn: _fp8_weight,
    DType.uint8: _fp4_weight,
}

_SCALES: dict[str, _WeightGen] = {
    "weight_scale_inv": _fp8_block_scale,
    "weight_scale": _fp4_block_scale,
    "weight_scale_2": _fp4_tensor_scale,
    "input_scale": _fp4_input_scale,
}


def _random_weights(
    params: Mapping[str, Tensor], rng: np.random.Generator
) -> dict[str, Buffer]:
    """Random host weights for every parameter of the reference layer."""
    return {
        name: (
            _SCALES.get(name.rpartition(".")[2]) or _WEIGHTS_BY_DTYPE[p.dtype]
        )(rng, params, name)
        for name, p in params.items()
    }


def _routed_tokens(
    rng: np.random.Generator, gate_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Draws bf16-exact tokens whose top-k router picks are unambiguous.

    Returns the tokens and each token's top-k expert ids.
    """
    rows: list[np.ndarray] = []
    picks: list[np.ndarray] = []
    while len(rows) < NUM_TOKENS:
        row = _bf16_round(_normal(rng, (HIDDEN,)))
        scores = gate_weight @ row
        order = np.argsort(-scores)
        if scores[order[TOP_K - 1]] - scores[order[TOP_K]] >= _MIN_ROUTER_GAP:
            rows.append(row)
            picks.append(order[:TOP_K])
    return np.stack(rows), np.stack(picks)


def _bf16_round(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).to(torch.float32).numpy()


def _to_device_bf16(values: np.ndarray, device: Accelerator) -> Buffer:
    return (
        Buffer.from_numpy(_bits(values, torch.bfloat16))
        .view(DType.bfloat16, list(values.shape))
        .to(device)
    )


def _to_float32(buffer: Buffer) -> np.ndarray:
    host = buffer.to(CPU())
    bits = host.view(DType.int16, list(host.shape)).to_numpy()
    return torch.from_numpy(bits).view(torch.bfloat16).to(torch.float32).numpy()


_CASES = [
    pytest.param(
        Case(dispatch, eplb),
        id=f"{dispatch}-{'eplb' if eplb else 'no_eplb'}",
    )
    for dispatch in ("bf16", "fp8", "nvfp4")
    for eplb in (False, True)
]


@pytest.mark.parametrize("case", _CASES)
def test_ep_matches_single_device(case: Case) -> None:
    count = accelerator_count()
    assert count >= N_DEVICES, f"needs {N_DEVICES} GPUs; found {count}"
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    session = InferenceSession(devices=[CPU(), *devices])
    rng = np.random.default_rng(20261112)

    with F.lazy(), default_device(devices[0]), default_dtype(DType.bfloat16):
        reference = QuantizedMoE(
            hidden_dim=HIDDEN,
            num_experts=NUM_EXPERTS,
            num_experts_per_token=TOP_K,
            moe_dim=MOE_DIM,
            quant_config=case.format.quant_config,
        )
    host_weights = _random_weights(dict(reference.parameters), rng)
    weights = {
        name: Tensor(storage=buffer) for name, buffer in host_weights.items()
    }
    ref_run = reference.compile(
        TensorLayout(DType.bfloat16, ["tokens", HIDDEN], devices[0]),
        weights=weights,
        session=session,
    )
    ep = _layer_ep(case, weights, session, devices)

    gate_weight = _to_float32(host_weights["gate.gate_score.weight"])
    tokens, picks = _routed_tokens(rng, gate_weight)
    (ref_out,) = ref_run.execute_raw(_to_device_bf16(tokens, devices[0]))
    shards = np.array_split(tokens, N_DEVICES)
    ep_out = ep.execute(
        [_to_device_bf16(s, d) for s, d in zip(shards, devices, strict=True)]
    )

    ep_result = np.concatenate([_to_float32(b) for b in ep_out])
    ref_result = _to_float32(ref_out)
    assert np.isfinite(ep_result).all() and np.isfinite(ref_result).all()
    _CHECKS[case.dispatch](ep_result, ref_result)

    counts = np.bincount(picks.ravel(), minlength=NUM_EXPERTS)
    n_local = NUM_EXPERTS // N_DEVICES
    want_load = [
        counts[d * n_local : (d + 1) * n_local][None].astype(np.uint64)
        for d in range(N_DEVICES)
    ]
    got_load = ep.expert_load()
    assert len(got_load) == N_DEVICES * case.eplb
    for got, want in zip(got_load, want_load, strict=False):
        np.testing.assert_array_equal(got, want)
