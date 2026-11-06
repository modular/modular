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
"""GPU coverage for DSpark stage forward and attention projections."""

from __future__ import annotations

import dataclasses

import ml_dtypes
import numpy as np
import pytest
import torch
from dspark_test_support import _full_hf, _small_config, _text
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    Device,
    accelerator_api,
    accelerator_count,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import KVCacheInputsPerDevice, KVCacheParams
from max.pipelines.architectures.deepseekV41_modulev3.dspark_drafter import (
    drafter_layer_infos,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
    build_kv_params,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.dspark import (
    DSparkAttention,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.dspark_moe import (
    DSparkMoE,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.rope import (
    RopeParams,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor import realization_context as rc
from max.tensor.sharding import DeviceMesh
from max.tensor.tensor import Tensor, realization_context
from test_common.simple_kv_cache import paged_kv_cache_inputs

_WIDTH = 512
_ROW_BYTES = 528
"""Bytes of one packed MXFP8 record row: 512 E4M3 codes then 16 raw
UE8M0 scale bytes, the window record the trunk's own store writes."""
_GROUP = 32
_WINDOW = 8
_SENTINEL = 0xA5
_ROPE = 64

_CUDA_OR_HIP = accelerator_api() in ("cuda", "hip")
_requires_gpu = pytest.mark.skipif(
    accelerator_count() == 0 or not _CUDA_OR_HIP,
    reason="the packed store asserts a GPU target (the leased box)",
)


def _config() -> DeepseekV41Config:
    """Return a compact geometry supported by the packed attention kernels."""
    return dataclasses.replace(
        _small_config(),
        head_dim=_WIDTH,
        sliding_window=_WINDOW,
        num_attention_heads=16,
        q_lora_rank=256,
        o_groups=4,
        o_lora_rank=128,
    )


def _ceil_log2(x: np.ndarray) -> np.ndarray:
    """The smallest integer ``k`` with ``2**k >= x``, for ``x > 0``.

    Exact where ``np.log2`` is not: the two correction steps are the same
    fix the production quantizer applies at power-of-two boundaries.
    """
    k = np.floor(np.log2(x)).astype(np.int64)
    k = np.where(np.power(2.0, k) < x, k + 1, k)
    k = np.where(np.power(2.0, k - 1) >= x, k - 1, k)
    return k


def _bits(t: Tensor) -> np.ndarray:
    """A realized bf16 tensor on the host, as uint16 bit patterns."""
    host = torch.from_dlpack(t.to(CPU()))
    assert host.dtype == torch.bfloat16
    return host.view(torch.uint16).numpy()


def _bf16_tensor(x: np.ndarray, device: Device) -> Tensor:
    """A realized bf16 tensor from float host values."""
    bits = (
        np.ascontiguousarray(x.astype(np.float32))
        .astype(ml_dtypes.bfloat16)
        .view(np.uint16)
    )
    return Tensor(
        storage=Buffer.from_numpy(bits).view(DType.bfloat16).to(device)
    )


def _int_tensor(x: np.ndarray, dtype: DType, device: Device) -> Tensor:
    return Tensor(storage=Buffer.from_numpy(np.ascontiguousarray(x)).to(device))


def _fp32_tensor(x: np.ndarray, device: Device) -> Tensor:
    return Tensor(storage=Buffer.from_numpy(np.ascontiguousarray(x)).to(device))


def _swa_leaf_params(device_ref: DeviceRef) -> KVCacheParams:
    """The production packed swa leaf: 43 slots of 528-byte record rows."""
    params = build_kv_params(
        V41Config.from_hf(_full_hf()),
        KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
        [device_ref],
    )
    leaf = params.children["swa"]
    assert isinstance(leaf, KVCacheParams)
    # The trunk's window leaf holds opaque uint8 record rows: 528 bytes
    # per token, the packed MXFP8 record the store writes whole.
    assert leaf.dtype == DType.uint8
    assert leaf.head_dim == _ROW_BYTES
    return leaf


class _Leaf:
    """One device's packed swa leaf with reversed page ids and sentinel bytes.

    ``paged_kv_cache_inputs`` hands pages out sequentially from 0; the
    constructor reverses the used page ids, so a two-or-more-page leaf
    (the ragged round trip, the 205-token poison fixture) forces every
    store and gather through the lookup table rather than a position
    arithmetic shortcut, and a one-page leaf (the 44-token fault
    fixture) keeps the identity -- the only permutation of one page.
    The main round-trip test asserts the nonidentity explicitly.
    """

    def __init__(self, params: KVCacheParams, starts: list[int]) -> None:
        device = params.devices[0].to_device()
        # Size the table for the final lengths: one "prompt" of the whole
        # prefix keeps both length scalars at bounds every later store and
        # read respects.
        inputs = paged_kv_cache_inputs(params, starts)
        lut = inputs.lookup_table.to(CPU()).to_numpy()
        null_page = int(lut.max())
        used = sorted({int(p) for p in lut.flat if int(p) != null_page})
        remap = {p: used[-1 - i] for i, p in enumerate(used)}
        lut = np.vectorize(lambda p: remap.get(int(p), int(p)))(lut)
        self.lut = lut.astype(np.uint32)
        blocks_np = np.full(tuple(inputs.kv_blocks.shape), _SENTINEL, np.uint8)
        self.blocks = Buffer.from_numpy(blocks_np).to(device)
        self.inputs = dataclasses.replace(
            inputs,
            kv_blocks=self.blocks,
            lookup_table=Buffer.from_numpy(np.ascontiguousarray(self.lut)).to(
                device
            ),
        )
        self.device = device
        self.n_slots = int(inputs.kv_blocks.shape[2])
        self.slots_per_page = int(inputs.kv_blocks.shape[3])

    def raw(
        self, cache_lengths: list[int]
    ) -> KVCacheInputsPerDevice[Buffer, Buffer]:
        """The leaf's buffers with the rows beginning at ``cache_lengths``."""
        return dataclasses.replace(
            self.inputs,
            cache_lengths=Buffer.from_numpy(
                np.array(cache_lengths, np.uint32)
            ).to(self.device),
        )

    def cache(self, cache_lengths: list[int]) -> PagedCacheValues:
        """The leaf as one device's cache; rows begin at ``cache_lengths``."""
        lengths = self.raw(cache_lengths)
        return PagedCacheValues(
            kv_blocks=Tensor(storage=lengths.kv_blocks),
            cache_lengths=Tensor(storage=lengths.cache_lengths),
            lookup_table=Tensor(storage=lengths.lookup_table),
            max_prompt_length=Tensor(storage=lengths.max_prompt_length),
            max_cache_length=Tensor(storage=lengths.max_cache_length),
            page_stride=Tensor(storage=lengths.page_stride),
            kv_scales=(
                Tensor(storage=lengths.kv_scales)
                if lengths.kv_scales is not None
                else None
            ),
            scales_page_stride=(
                Tensor(storage=lengths.scales_page_stride)
                if lengths.scales_page_stride is not None
                else None
            ),
            scales_lookup_table=(
                Tensor(storage=lengths.scales_lookup_table)
                if lengths.scales_lookup_table is not None
                else None
            ),
            attention_dispatch_metadata=(
                Tensor(storage=lengths.attention_dispatch_metadata)
                if lengths.attention_dispatch_metadata is not None
                else None
            ),
            mla_num_partitions=(
                Tensor(storage=lengths.mla_num_partitions)
                if lengths.mla_num_partitions is not None
                else None
            ),
        )


def _stage(
    config: DeepseekV41Config, index: int, mesh: DeviceMesh
) -> DSparkAttention:
    info = drafter_layer_infos(config)[index]
    assert info.layer_id == 40 + index, info.layer_id
    return DSparkAttention(
        config,
        info,
        mesh,
        rope=RopeParams.from_text_config(_text()),
        max_position=config.max_seq_len,
    )


def _mxfp8_weight(
    w: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return checkpoint-format MXFP8 codes, scales, and decoded values."""
    n, k = w.shape
    exponents = rng.integers(-2, 2, size=(n // _GROUP, k // _GROUP))
    expanded = np.repeat(
        np.repeat(np.power(2.0, exponents).astype(np.float32), _GROUP, 0),
        _GROUP,
        1,
    )
    codes = (w / expanded).astype(ml_dtypes.float8_e4m3fn)
    return (
        codes.view(np.uint8),
        (127 + exponents).astype(np.uint8),
        codes.astype(np.float32) * expanded,
    )


def _random_weights(
    config: DeepseekV41Config, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Build synthetic stage weights and decoded reference values."""
    heads, d = config.num_attention_heads, config.head_dim
    hidden = config.hidden_size
    rank, groups = config.o_lora_rank, config.o_groups
    w = {
        "wq_a": rng.standard_normal((config.q_lora_rank, hidden)) * 0.1,
        "wq_b": rng.standard_normal((heads * d, config.q_lora_rank)) * 0.1,
        "wkv": rng.standard_normal((d, hidden)) * 0.5,
        "wo_a": rng.standard_normal((groups * rank, heads * d // groups)) * 0.1,
    }
    out: dict[str, np.ndarray] = {}
    for name, values in w.items():
        out[f"{name}_codes"], out[f"{name}_scale"], out[name] = _mxfp8_weight(
            values.astype(np.float32), rng
        )
    return out | {
        "q_norm": 1.0 + 0.1 * rng.standard_normal(config.q_lora_rank),
        "kv_norm": 1.0 + 0.1 * rng.standard_normal(d),
        "wo_b": rng.standard_normal((hidden, groups * rank)) * 0.05,
        "wo_b_scale": np.full(
            (hidden // 32, groups * rank // 32), 127, np.uint8
        ),
        "attn_sink": 0.1 * rng.standard_normal(heads),
    }


def _fp8_tensor(codes: np.ndarray, dtype: DType, device: Device) -> Tensor:
    return Tensor(
        storage=Buffer.from_numpy(np.ascontiguousarray(codes))
        .view(dtype)
        .to(device)
    )


def _install_weights(
    attn: DSparkAttention,
    w: dict[str, np.ndarray],
    device: Device,
) -> None:
    """Places the synthetic weights on the device."""
    for name in ("q_norm", "kv_norm"):
        getattr(attn, name).weight = _bf16_tensor(w[name], device)
    attn.wqa_wkv.weight = _fp8_tensor(
        np.concatenate([w["wq_a_codes"], w["wkv_codes"]]),
        DType.float8_e4m3fn,
        device,
    )
    attn.wqa_wkv.scale = _fp8_tensor(
        np.concatenate([w["wq_a_scale"], w["wkv_scale"]]),
        DType.float8_e8m0fnu,
        device,
    )
    for name in ("wq_b", "wo_a"):
        getattr(attn, name).weight = _fp8_tensor(
            w[f"{name}_codes"], DType.float8_e4m3fn, device
        )
        getattr(attn, name).scale = _fp8_tensor(
            w[f"{name}_scale"], DType.float8_e8m0fnu, device
        )
    wo_b_bits = (
        np.ascontiguousarray(w["wo_b"])
        .astype(ml_dtypes.float8_e4m3fn)
        .view(np.uint8)
    )
    attn.wo_b.weight = _fp8_tensor(wo_b_bits, DType.float8_e4m3fn, device)
    attn.wo_b.scale = _fp8_tensor(w["wo_b_scale"], DType.float8_e8m0fnu, device)
    attn.attn_sink = _fp32_tensor(w["attn_sink"].astype(np.float32), device)


def _ragged(
    per_request: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Stacks per-request row blocks request-major, with the row offsets."""
    row_offsets = np.zeros(len(per_request) + 1, dtype=np.uint32)
    row_offsets[1:] = np.cumsum([len(r) for r in per_request])
    return np.concatenate(per_request), row_offsets


def _positions(starts: list[int], lengths: list[int]) -> np.ndarray:
    """Absolute positions of each request's ``lengths`` rows."""
    return np.concatenate(
        [
            np.arange(s, s + n, dtype=np.int32)
            for s, n in zip(starts, lengths, strict=True)
        ]
    )


# The independent reference


def _r(x: np.ndarray) -> np.ndarray:
    """Rounds to bfloat16 and back, as every production tensor does."""
    return x.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def _rms(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """The reference RMSNorm: fp32 math, weight before the bf16 cast."""
    x = x.astype(np.float32)
    inv = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-20)
    return _r(_r(weight) * x * inv)


def _fake_quant(x: np.ndarray) -> np.ndarray:
    """The MXFP8 1x32 round trip: UE8M0 scale per group, E4M3 codes."""
    grouped = x.astype(np.float32).reshape(*x.shape[:-1], -1, _GROUP)
    amax = np.maximum(np.abs(grouped).max(axis=-1, keepdims=True), 1e-4)
    scale = np.power(2.0, _ceil_log2(amax * (1.0 / 448.0))).astype(np.float32)
    codes = np.clip(grouped / scale, -448.0, 448.0)
    codes = codes.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    return _r((codes * scale).reshape(x.shape))


def _rope_tail(
    x: np.ndarray, positions: np.ndarray, inv_freqs: np.ndarray, inverse: bool
) -> np.ndarray:
    """Rotates the last 64 channels pairwise; ``x`` is ``[t, ..., d]``."""
    angles = positions.astype(np.float32)[:, None] * inv_freqs[None, :]
    expand = (slice(None),) + (None,) * (x.ndim - 2) + (slice(None),)
    cos = np.cos(angles)[expand]
    sin = np.sin(angles)[expand] * (-1.0 if inverse else 1.0)
    head, tail = x[..., :-_ROPE], x[..., -_ROPE:]
    pairs = tail.reshape(*tail.shape[:-1], _ROPE // 2, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = np.stack([real * cos - imag * sin, real * sin + imag * cos], -1)
    return np.concatenate(
        [head, _r(rotated.reshape(tail.shape))], axis=-1
    ).astype(np.float32)


def _reference_stage(
    config: DeepseekV41Config,
    w: dict[str, np.ndarray],
    inv_freqs: np.ndarray,
    context_x: list[np.ndarray],
    block_x: np.ndarray,
    starts: list[int],
) -> np.ndarray:
    """The stage's output over the blocks, from the weights and raw rows."""
    heads, d, k = (
        config.num_attention_heads,
        config.head_dim,
        len(block_x) // len(starts),
    )
    groups, rank = config.o_groups, config.o_lora_rank
    block_pos = np.concatenate(
        [np.arange(s, s + k, dtype=np.int32) for s in starts]
    )

    def kv_records(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        kv = _rms(_r(_fake_quant(_r(x)) @ w["wkv"].T), w["kv_norm"])
        return _fake_quant(_rope_tail(kv, positions, inv_freqs, False))

    qr = _rms(_r(_fake_quant(_r(block_x)) @ w["wq_a"].T), w["q_norm"])
    q = _r(_fake_quant(qr) @ w["wq_b"].T).reshape(len(block_x), heads, d)
    q = _rope_tail(q, block_pos, inv_freqs, False)
    block_kv = kv_records(block_x, block_pos)
    sink = w["attn_sink"].astype(np.float32)

    out = np.zeros_like(q)
    for r, start in enumerate(starts):
        lo = max(0, start - config.sliding_window)
        context = kv_records(context_x[r][lo:start], np.arange(lo, start))
        keys = np.concatenate([context, block_kv[r * k : (r + 1) * k]])
        for row in range(r * k, (r + 1) * k):
            scores = (q[row] @ keys.T) * d**-0.5  # [heads, keys]
            top = np.maximum(scores.max(-1, keepdims=True), sink[:, None])
            p = np.exp(scores - top)
            denominator = p.sum(-1, keepdims=True) + np.exp(sink[:, None] - top)
            out[row] = (p @ keys) / denominator
    out = _rope_tail(_r(out), block_pos, inv_freqs, True)

    group_width = heads * d // groups
    wo_a = w["wo_a"].reshape(groups, rank, group_width)
    projected = np.einsum(
        "tgw,grw->tgr",
        _fake_quant(out.reshape(len(block_x), heads * d)).reshape(
            len(block_x), groups, group_width
        ),
        wo_a,
    )
    flat = _fake_quant(_r(projected).reshape(len(block_x), groups * rank))
    wo_b = w["wo_b"].astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    return _r(flat @ wo_b.T)


def _host(t: Tensor) -> np.ndarray:
    return torch.from_dlpack(t.to(CPU())).float().numpy()


# The executed forward


@_requires_gpu
def test_stage_forward_matches_the_reference() -> None:
    """Store, read the window, run the block: the output is the oracle's.

    Three requests with distinct committed lengths: a full window, a
    partial one and a context shorter than the window, so the ``-1`` list
    entries below position zero are exercised. The block is a whole
    ``dspark_block_size`` of rows per request at positions ``L .. L + k - 1``.
    """
    config = _config()
    rng = np.random.default_rng(7)
    k = config.dspark_block_size
    starts = [20, 9, 3]
    b = len(starts)
    weights = _random_weights(config, rng)
    context_x = [
        rng.standard_normal((s, config.hidden_size)).astype(np.float32)
        for s in starts
    ]
    block_x = rng.standard_normal((b * k, config.hidden_size)).astype(
        np.float32
    )
    block_pos = _positions(starts, [k] * b)
    context_main, context_offsets = _ragged(context_x)
    context_pos = _positions([0] * b, starts)

    with rc.EagerRealizationContext() as ctx, realization_context(ctx):
        device = Accelerator()
        mesh = DeviceMesh((device,), (1,), ("tp",))
        attn = _stage(config, 2, mesh)
        _install_weights(attn, weights, device)
        # Pages for the block's rows too: the stage stores them.
        leaf = _Leaf(_swa_leaf_params(DeviceRef.GPU()), [s + k for s in starts])

        attn.store_context(
            leaf.cache([0] * b),
            _bf16_tensor(context_main, device),
            _int_tensor(context_offsets, DType.uint32, device),
            _int_tensor(context_pos, DType.int32, device),
        )
        window = attn.read_window(
            leaf.cache(starts),
            _int_tensor(np.array(starts, np.int32), DType.int32, device),
        )
        out = attn(
            _bf16_tensor(block_x, device),
            window,
            _int_tensor(block_pos, DType.int32, device),
        )
        inv_freqs = attn.rope.inv_freqs()

    want = _reference_stage(
        config, weights, inv_freqs, context_x, block_x, starts
    )
    assert float(np.abs(want).max()) > 0.0
    got = _host(out)
    assert got.shape == want.shape
    rms_err = np.sqrt(((got - want) ** 2).mean())
    assert rms_err < 0.05 * np.sqrt((want**2).mean()), rms_err
    np.testing.assert_allclose(
        got, want, rtol=0.1, atol=0.15 * float(np.abs(want).max())
    )


_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)


def _decode_mxfp4(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Decode packed E2M1 values with E8M0 scales."""
    experts, n, half = codes.shape
    k = half * 2
    flat = codes.reshape(experts * n, half)
    nibbles = np.stack([flat & 0xF, flat >> 4], axis=-1).reshape(experts * n, k)
    magnitude = _E2M1[(nibbles & 0x7).astype(np.int64)]
    signed = np.where(nibbles >= 8, -magnitude, magnitude)
    decoded = signed * np.repeat(
        np.power(
            2.0, scale.reshape(experts * n, -1).astype(np.float32) - 127.0
        ),
        32,
        axis=-1,
    )
    return decoded.reshape(experts, n, k)


def _mxfp4_weight(
    experts: int, n: int, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build synthetic MXFP4 weights and their decoded values."""
    codes = rng.integers(0, 256, (experts, n, k // 2)).astype(np.uint8)
    scale = rng.integers(123, 126, (experts, n, k // 32)).astype(np.uint8)
    return codes, scale, _decode_mxfp4(codes, scale)


def _moe_weights(
    config: DeepseekV41Config, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Build synthetic weights and decoded values for the MoE reference."""
    hidden, inter = config.hidden_size, config.moe_intermediate_size
    experts = config.dspark_n_routed_experts
    w: dict[str, np.ndarray] = {}
    for name, n, k in (
        ("gate_up", 2 * inter, hidden),
        ("down", hidden, inter),
    ):
        codes, scale, decoded = _mxfp4_weight(experts, n, k, rng)
        w[f"{name}_codes"], w[f"{name}_scale"] = codes, scale
        w[name] = decoded
    w["gate"] = _r(rng.standard_normal((experts, hidden)) * 0.1)
    w["gate_bias"] = (rng.standard_normal(experts) * 0.1).astype(np.float32)
    for name, n, k in (
        ("w1", inter, hidden),
        ("w3", inter, hidden),
        ("w2", hidden, inter),
    ):
        values = (rng.standard_normal((n, k)) * 0.1).astype(np.float32)
        w[name] = values.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
        w[f"{name}_scale"] = np.full((n // 32, k // 32), 127, np.uint8)
    return w


def _install_moe_weights(
    moe: DSparkMoE, w: dict[str, np.ndarray], device: Device
) -> None:
    """Load synthetic MoE weights into the device module."""
    moe.gate.weight = _bf16_tensor(w["gate"], device)
    moe.gate.bias = _fp32_tensor(w["gate_bias"], device)
    for proj, prefix in (
        (moe.experts.gate_up_proj, "gate_up"),
        (moe.experts.down_proj, "down"),
    ):
        proj.weight = _int_tensor(w[f"{prefix}_codes"], DType.uint8, device)
        proj.weight_scale = _fp8_tensor(
            w[f"{prefix}_scale"], DType.float8_e8m0fnu, device
        )
    for name in ("w1", "w3", "w2"):
        linear = getattr(moe.shared_experts, name)
        linear.weight = _fp8_tensor(
            np.ascontiguousarray(w[name])
            .astype(ml_dtypes.float8_e4m3fn)
            .view(np.uint8),
            DType.float8_e4m3fn,
            device,
        )
        linear.scale = _fp8_tensor(
            w[f"{name}_scale"], DType.float8_e8m0fnu, device
        )


def _reference_moe(
    config: DeepseekV41Config,
    w: dict[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """NumPy reference for the drafter's gated and expert projections."""
    tokens = x.shape[0]
    hidden, inter = config.hidden_size, config.moe_intermediate_size
    k = config.dspark_num_experts_per_tok
    limit = float(config.swiglu_limit)

    logits = x @ w["gate"].T
    softplus = np.maximum(logits, 0.0) + np.log(np.exp(-np.abs(logits)) + 1.0)
    scores = np.sqrt(softplus)
    top = np.argsort(-(scores + w["gate_bias"]), axis=-1, kind="stable")[:, :k]
    picked = np.take_along_axis(scores, top, axis=1)
    total = picked.sum(-1, keepdims=True)
    weights = picked / total * config.routed_scaling_factor

    def silu(v: np.ndarray) -> np.ndarray:
        return v / (1.0 + np.exp(-v))

    routed = np.zeros((tokens, hidden), np.float32)
    for t in range(tokens):
        row = _fake_quant(x[t : t + 1])[0]
        for j in range(k):
            expert = top[t, j]
            gate_up = _r(row @ w["gate_up"][expert].T).astype(np.float32)
            gate, up = gate_up[:inter], gate_up[inter:]
            gate = np.minimum(gate, limit)
            up = np.clip(up, -limit, limit)
            h = silu(gate) * up * weights[t, j]
            routed[t] += _r(
                _fake_quant(_r(h)[None])[0] @ w["down"][expert].T
            ).astype(np.float32)

    a1 = _r(_fake_quant(x) @ w["w1"].T).astype(np.float32)
    a3 = _r(_fake_quant(x) @ w["w3"].T).astype(np.float32)
    gate = np.minimum(a1, limit)
    up = np.clip(a3, -limit, limit)
    activated = _r(silu(gate) * up)
    shared = _r(_fake_quant(activated) @ w["w2"].T).astype(np.float32)
    return _r(shared + routed)


@_requires_gpu
def test_moe_forward_matches_the_reference() -> None:
    config = dataclasses.replace(_small_config(), moe_intermediate_size=128)
    rng = np.random.default_rng(11)
    tokens = 7
    x = rng.standard_normal((tokens, config.hidden_size)).astype(np.float32)
    weights = _moe_weights(config, rng)

    with rc.EagerRealizationContext() as ctx, realization_context(ctx):
        device = Accelerator()
        mesh = DeviceMesh((device,), (1,), ("tp",))
        moe = DSparkMoE(config, mesh)
        _install_moe_weights(moe, weights, device)
        out = moe(_bf16_tensor(x, device))

    want = _reference_moe(config, weights, _r(x))
    assert float(np.abs(want).max()) > 0.0
    got = _host(out)
    assert got.shape == want.shape
    rms_err = np.sqrt(((got - want) ** 2).mean())
    assert rms_err < 0.05 * np.sqrt((want**2).mean()), rms_err
    np.testing.assert_allclose(
        got, want, rtol=0.1, atol=0.15 * float(np.abs(want).max())
    )
