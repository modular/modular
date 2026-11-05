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
"""NULL-mask parity for the MLA attention arms at the DSpark drafter geometry.

The DSpark drafter's block call (SERVOPT-1757) feeds ``S = 7`` query rows per
request through ``mla_prefill_decode_graph`` with
``MHAMaskVariant.NULL_MASK``: block diffusion drafts all seven rows in
parallel, so every row must attend to every cached key in its window --
including the keys the OTHER six query rows store this step. The target's
verify feeds the same op ``CAUSAL_MASK`` (``test_spec_mla_verify8.py``), and
no MLA test had ever asked for the null mask, so both AMD arms the drafter
lands on -- the FP8 decode fold (TP8: 8 heads per rank) and the one-shot
decompression prefill (a bf16 latent cache, and the TP1/TP2 head counts
64/32) -- were plumbed for ``NullMask`` but untested.

Head counts and cache dtypes pick the arm the way the drafter's TP split
does: 8 heads over an FP8 cache routes S=7 through the AMD decode fold
(``mla.mojo``'s gate: FP8, ``num_heads <= 16``, ``S <= 8``,
``num_heads * S <= 128``); a bf16 cache or a head count above 16 routes S=7
through the AMD prefill branch. Each param asserts that routing host-side
against a mirror of ``mla_decode_max_seq_len``, so a kernel change that
silently re-routes an arm fails the test instead of just renaming it.

Every request's committed context is filled with random nonzero latents in
the cache's exact storage format -- RMSNormed, roped at the row's absolute
position, and quantized -- written straight into the pages before the call,
byte-identical in the arm and its reference. Zero-filled contexts (what a
fresh cache gives) put nothing in the context tiles for a null-path defect
to misread; real latents make every context key a distinct witness. The
fill bypasses the store op on purpose: an output-less in-place op is only
schedulable through a compiled graph, and prefilling keeps the context
bytes identical across every call by construction rather than by trusting
a second kernel. The bytes are the same recipe the store op's own test
validates (``test_mla_store_latent_gpu.py``).

The arrangement oracle never runs the null mask. K3 is NoPE, so every call
ropes q and the stored key slice by the identity table
``kimi_k3_nope_freqs_cis`` builds -- nothing is position-dependent -- and
softmax is invariant to the order keys sit in the cache. Row ``r``'s null
output therefore equals the LAST row of a CAUSAL call whose per-request
block is arranged with row ``r`` moved to the end: that row stores at
``cache_length + S - 1``, the causal boundary a block's last row sees, so
it attends to the whole window -- the context plus every block row's key,
each stored exactly once. The arrangement is the only difference between
the arm under test and its reference, so the comparison isolates the mask.

Shared machinery -- the store position, the rope, the projections, the
attention arithmetic -- cancels between the two arms by construction; a
defect there belongs to the causal tests' stepwise oracle, not here. Each
request is normalized by its own max output magnitude, so a long-context
request's error cannot hide behind a short request's larger scale.

RED controls, all of which must FAIL the parity check rather than pass it:

* passing ``CAUSAL_MASK`` where null is expected must diverge on every
  request (it hides the other rows' keys from every row but the last) --
  proving the mask string is honored, in both directions: if the kernel
  ignored it, this control would see no divergence while the parity above
  would fail. Checked on every arm at no extra cost: the last arrangement
  is the identity, so its full output IS the causal arm.
* a reference with the LAST query row's key dropped from every other
  row's window must diverge per request -- proving the tolerance catches
  a single missing key in each request, not just the short one. Run on the
  production fold arm; see the tolerance note for the per-request margins.

The drafter's real rope, not the target's identity table, is covered
separately: the arrangement oracle is INVALID under a positional rope
(moving row r last changes its rope position), so the YaRN case gates three
properties with the comparison each can survive. A float32 numpy reference
of the kernel's math (projections, RMSNorm, interleaved rope at
``cache_length + row``, absorbed attention over the full window, e4m3 cache
rows) gates the semantics at a loose bound -- the fold's internal FP8 q
cascade cannot be modeled from source, and that comparison's floor would
hide a one-position rope defect, which the float32 model of this draw puts
at only 0.4-0.5% per request at the natural rope/nope balance. The
position defect is therefore gated kernel against kernel: feeding the same
NULL_MASK call a rope table shifted one row is exactly "rope at position +
1" for every row the kernel owns, and on a rope-amplified weight draw the
measured divergence sits at 0.32-0.62 per request -- 13x the
kernel-vs-kernel bound at the weakest request. Under any rope a null
call's last row must equal a causal call's last row.

One accepted gap: every request always feeds exactly ``S = 7`` rows, the
fixed block DSpark drafts -- the fold's variable-per-request query-length
path stays covered by its own kernel tests. And the YaRN case runs at
drafter-scale positions (contexts up to 250), where the yarn correction
has not yet engaged; long-context rope behavior is the drafter lane's
end-to-end job (``dspark/test_torch_reference.py``).
"""

from __future__ import annotations

import os

os.environ.setdefault("MODULAR_DEBUG", "source-tracebacks")

from collections.abc import Sequence

import numpy as np
import pytest
from _paged_cache_test_utils import block_ids_for_batch, build_paged_cache
from max import functional as F
from max.driver import CPU, Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kernels import mla_prefill_decode_graph
from max.nn.kv_cache import MLAKVCacheParams
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.pipelines.architectures.kimi_k3_modulev3.dspark.config import (
    K3DSparkDraftConfig,
)
from max.pipelines.architectures.kimi_k3_modulev3.dspark.dspark import (
    k3_dspark_yarn_freqs_cis,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
    kimi_k3_mla_prefill_plan,
    kimi_k3_nope_freqs_cis,
)
from max.pipelines.architectures.kimi_k3_modulev3.spec_mla import (
    kimi_k3_mla_buffer_size,
)
from max.support.math import ceildiv
from max.tensor import Tensor, default_device

_PAGE_SIZE = 128
# The DSpark block width: seven draft rows, no anchor inside the block call.
_BLOCK_ROWS = 7
# A ragged batch: different committed contexts, two of whose windows cross a
# 128-token page boundary (125 + 7 = 132 and 250 + 7 = 257). The short
# context keeps a single dropped key at ~1/10 of the softmax mass; the long
# ones are what the per-request metrics are for.
_CACHE_LENGTHS = (125, 61, 3, 250)
# The longest committed context plus its block rows: this test "model"'s
# context limit.
_MAX_SEQ_LEN = 250 + _BLOCK_ROWS
# The batch's page-aligned block windows, the way the scheduler budgets them.
_TRUE_PADDED_WINDOW = sum(
    ceildiv(cache + _BLOCK_ROWS, _PAGE_SIZE) * _PAGE_SIZE
    for cache in _CACHE_LENGTHS
)

# Production head dims -- the latent width must be exactly 576 for the fused
# MLA kernels -- over a shrunken hidden size, as in test_spec_mla_verify8.py.
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 128
_Q_LORA_RANK = 1536
_KV_LORA_RANK = 512
_HIDDEN = 512

_SEED = 1757
_WEIGHT_SCALE = 0.05

# The arms under test: (query heads per rank, latent cache dtype, the AMD
# arm the pair must route S=7 to).
_ARMS = [
    pytest.param(8, DType.float8_e4m3fn, "decode_fold", id="fold_fp8_h8"),
    pytest.param(8, DType.bfloat16, "prefill", id="prefill_bf16_h8"),
    pytest.param(64, DType.float8_e4m3fn, "prefill", id="prefill_fp8_h64"),
    pytest.param(32, DType.float8_e4m3fn, "prefill", id="prefill_fp8_h32"),
]

_FLOAT8_DTYPES = frozenset({DType.float8_e4m3fn, DType.float8_e5m2})


def _plan_cap(heads_per_device: int, cache_dtype: DType) -> int:
    """The plan cap the shipped sizing helper computes for this test's model.

    The helper counts ``1 + num_draft_tokens`` verify rows; passing
    ``_BLOCK_ROWS - 1`` makes that this block's width of seven. The cap
    follows the arm the kernel routes this pair to, the way the shipped
    callers size it, so the plan covers the windows the arm actually
    opens on whatever device the suite runs.
    """
    return kimi_k3_mla_buffer_size(
        max_batch_context_length=_TRUE_PADDED_WINDOW,
        max_seq_len=_MAX_SEQ_LEN,
        page_size=_PAGE_SIZE,
        num_draft_tokens=_BLOCK_ROWS - 1,
        max_batch_size=len(_CACHE_LENGTHS),
        num_q_heads_per_device=heads_per_device,
        cache_dtype=cache_dtype,
        device_api=accelerator_api(),
    )


# Per-arm parity bounds. The arm and its arrangement reference run the SAME
# kernel arm over the same weight draw and the same context bytes, so the
# bound covers that arm's cross-call noise (reduction order over permuted
# key positions), not a cross-branch difference. Measured per request on
# MI355 over a nonzero filled context -- worst request per arm: fold fp8
# H=8 1.14e-2 (the 250-key page-crossing request), prefill fp8 H=64 9.0e-3,
# H=32 5.6e-3 -- so 2.5e-2 clears the loudest arm by 2.2x while the
# causal-where-null RED sits >= 7.5x above it everywhere. The bf16 arm
# quantizes nothing on the cache path and measures <= 1.4e-3 on MI355, but
# B200's bf16 prefill arm drifts to 5.5e-3 on its worst request (the fp8
# arms of the same geometry sit at 7.5e-3 there), so the bound must cover
# both platforms' cross-call noise; the causal RED stays >= 19x above it.
_TOL_PARITY_FP8 = 2.5e-2
_TOL_PARITY_BF16 = 1e-2
# The YaRN case's two comparisons and their floors. The float32 reference
# can only be modeled from source up to the fold's internal FP8 q cascade
# (measured 5-8% per request at the natural draw), so its bound gates the
# O(1) semantic errors, not subtle ones. The rope-position RED is kernel
# against kernel -- the +1 defect injected through the runtime table
# input, where the q cascade runs identically in both arms -- so its
# floor is the kernel-vs-kernel noise and the amplified signal lands at
# 0.32-0.62 per request, 13x the bound at its weakest.
_TOL_YARN_F32 = 2e-1
_TOL_YARN_POSITION = 2.5e-2


def _rel_error(got: np.ndarray, want: np.ndarray) -> float:
    """One relative error over a whole comparison, as verify8's tests use."""
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-12))


def _request_rel_errors(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Per-request relative error: each request by its own max ``|want|``.

    A long-context request's output scale must not be set by a short
    request's, or a defect in the request the lane cares about -- the
    page-crossing ones -- hides behind the batch's loudest request.
    """
    denom = np.maximum(np.max(np.abs(want), axis=(1, 2)), 1e-12)
    return np.max(np.abs(got - want), axis=(1, 2)) / denom


def _report(label: str, got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Prints per-request relative errors; returns them for asserting."""
    errs = _request_rel_errors(got, want)
    print(f"{label}: per-request rel={np.array2string(errs, precision=4)}")
    return errs


# ===-----------------------------------------------------------------------===#
# Host-side mirrors of the kernel's quantization and rope
# ===-----------------------------------------------------------------------===#


def _bf16(x: np.ndarray) -> np.ndarray:
    """Rounds float32 to bfloat16 values, exactly as a bf16 cast does."""
    bits = np.asarray(x, np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def _e4m3_bytes(x: np.ndarray) -> np.ndarray:
    """float32 to e4m3fn bit patterns, round-to-nearest-even, saturating.

    Mirrors ``cast_saturating[float8_e4m3fn]``: values past the finite max
    448 clip to it rather than going NaN (torch's plain cast NaNs there, so
    it is not a reference). Verified bit-exact against torch below 448,
    which is far past anything a normalized latent or query reaches.
    """

    def _rne(frac: np.ndarray, floor: np.ndarray) -> np.ndarray:
        return (frac > 0.5) | ((frac == 0.5) & (np.mod(floor, 2) == 1))

    x = np.asarray(x, np.float32)
    sign = np.signbit(x).astype(np.uint8)
    a = np.abs(x)
    out = np.zeros(a.shape, np.uint8)
    sat = a >= 448.0
    with np.errstate(all="ignore"):
        e = np.floor(np.log2(a)).astype(np.int32)
    normal = (a >= 2.0**-6) & ~sat
    frac_m = a[normal] / np.exp2(e[normal].astype(np.float64)) - 1.0
    t = frac_m * 8.0
    floor = np.floor(t)
    mant = (floor + _rne(t - floor, floor)).astype(np.int32)
    exp = e[normal] + (mant == 8)
    mant = np.where(mant == 8, 0, mant)
    over = (exp > 8) | ((exp == 8) & (mant > 6))
    bits = np.where(over, 0x7E, ((exp + 7) << 3) | mant)
    staged = np.zeros(a.shape, np.int32)
    staged[normal] = bits
    out = np.where(normal, staged.astype(np.uint8), out)
    # Subnormals: value = mantissa * 2^-9; rounding up to 8 promotes to the
    # normal 2^-6 (exponent field 1, zero mantissa).
    sub = (a > 0) & (a < 2.0**-6)
    t = a[sub] / 2.0**-9
    floor = np.floor(t)
    mant = (floor + _rne(t - floor, floor)).astype(np.int32)
    staged = np.zeros(a.shape, np.int32)
    staged[sub] = np.where(mant == 8, 1 << 3, mant)
    out = np.where(sub, staged.astype(np.uint8), out)
    out = np.where(sat, np.uint8(0x7E), out)
    return out | (sign << np.uint8(7))


def _quantize_rows(rows: np.ndarray, dtype: DType) -> np.ndarray:
    """Float32 rows to the cache dtype's little-endian byte patterns."""
    if dtype == DType.float8_e4m3fn:
        return _e4m3_bytes(rows)
    assert dtype == DType.bfloat16
    return (_bf16(rows).view(np.uint32) >> 16).astype("<u2").view(np.uint8)


def _dequantize_rows(bytes_: np.ndarray, dtype: DType) -> np.ndarray:
    """Cache-dtype byte patterns back to their exact float32 values."""
    if dtype == DType.float8_e4m3fn:
        b = bytes_.astype(np.uint8)
        sign = np.where(b >= 128, -1.0, 1.0)
        b = b & np.uint8(0x7F)
        exp = (b >> 3).astype(np.int32) - 7
        mant = (b & 7).astype(np.float32)
        val = np.where(
            exp > -7,
            (1.0 + mant / 8.0) * np.exp2(exp.astype(np.float32)),
            mant * 2.0**-9,
        )
        return (sign * val).astype(np.float32)
    assert dtype == DType.bfloat16
    return (bytes_.view("<u2").astype(np.uint32) << 16).view(np.float32)


def _rms_norm(x: np.ndarray, eps: float) -> np.ndarray:
    """The cache's in-place latent norm: ``x / rms(x)`` (gamma is one)."""
    mean_sq = np.mean(np.square(x.astype(np.float64)), axis=-1, keepdims=True)
    return (x / np.sqrt(mean_sq + eps)).astype(np.float32)


def _rope_interleaved(vecs: np.ndarray, table: np.ndarray) -> np.ndarray:
    """The fused rope kernel's rotation, on ``[..., rope_dim]`` rows.

    ``table`` holds each position's interleaved ``(cos, sin)`` pairs; even
    elements are the real part, odd the imaginary, matching
    ``fused_qk_rope.rope_value``'s complex multiply.
    """
    a = vecs[..., 0::2].astype(np.float32)
    b = vecs[..., 1::2].astype(np.float32)
    cos = table[..., 0::2].astype(np.float32)
    sin = table[..., 1::2].astype(np.float32)
    re = a * cos - b * sin
    im = a * sin + b * cos
    out = np.empty(vecs.shape, np.float32)
    out[..., 0::2] = re
    out[..., 1::2] = im
    return out


def _stored_row_bytes(
    raw_latents: np.ndarray,
    positions: np.ndarray,
    table: np.ndarray,
    cache_dtype: DType,
    eps: float,
) -> np.ndarray:
    """Rows in the cache's storage format, mirroring the fused store kernel.

    RMSNorm over the ``kv_lora_rank``-wide head, interleaved rope over the
    tail at each row's absolute position, then the cast to the cache dtype
    -- the recipe ``test_mla_store_latent_gpu.py`` validates against the
    kernel itself.
    """
    normed = _rms_norm(raw_latents[:, :_KV_LORA_RANK], eps)
    roped = _bf16(
        _rope_interleaved(raw_latents[:, _KV_LORA_RANK:], table[positions])
    )
    return _quantize_rows(np.concatenate([normed, roped], axis=1), cache_dtype)


# ===-----------------------------------------------------------------------===#
# Cache construction with a filled context
# ===-----------------------------------------------------------------------===#


def _context_latents(rng: np.random.Generator) -> list[np.ndarray]:
    """One random nonzero latent draw per request's committed context.

    Drawn once per test so every call -- the arm and every reference
    arrangement -- reads byte-identical context rows.
    """
    return [
        (
            rng.standard_normal((cache, _KV_LORA_RANK + _QK_ROPE_HEAD_DIM))
            * 0.05
        ).astype(np.float32)
        for cache in _CACHE_LENGTHS
    ]


def _context_byte_rows(
    latents: list[np.ndarray], table: np.ndarray, cache_dtype: DType, eps: float
) -> list[np.ndarray]:
    """The context rows as cache bytes, roped at their absolute positions."""
    return [
        _stored_row_bytes(rows, np.arange(len(rows)), table, cache_dtype, eps)
        for rows in latents
    ]


def _blocks_buffer(
    params: MLAKVCacheParams,
    prompt_lens: Sequence[int],
    context_rows: list[np.ndarray],
    device: Accelerator,
) -> Tensor:
    """A blocks buffer whose context slots hold the given bytes, rest zero.

    Mirrors ``build_paged_cache``'s page assignment (sequential ids from
    ``block_ids_for_batch``, one trailing null page) so the rows land in the
    same pages its lookup table resolves.
    """
    assert params.page_size is not None
    seq_lens = [
        cache + rows
        for cache, rows in zip(_CACHE_LENGTHS, prompt_lens, strict=False)
    ]
    blocks = block_ids_for_batch(seq_lens, params.page_size)
    shape = (sum(len(ids) for ids in blocks) + 1, *params.shape_per_block)
    raw = np.zeros(
        (*shape[:-1], shape[-1] * params.dtype.size_in_bytes), np.uint8
    )
    for _request, (page_ids, rows) in enumerate(
        zip(blocks, context_rows, strict=True)
    ):
        for pos, row in enumerate(rows):
            raw[
                page_ids[pos // params.page_size],
                0,
                0,
                pos % params.page_size,
                0,
                :,
            ] = row
    buf = (
        Buffer.from_dlpack(raw).to(device).view(dtype=params.dtype, shape=shape)
    )
    return Tensor(storage=buf)


def _routes_to_decode_fold(heads: int, cache_dtype: DType) -> bool:
    """Host mirror of the arm router: does S=7 fold or take the prefill arm?

    Mirrors ``mla_decode_max_seq_len`` (``mla.mojo``): AMD folds only an
    FP8 cache at ``num_heads <= 16``, everything else routes S>1 to the
    prefill branch; other GPUs fold up to ``MLA_DECODE_MAX_SEQ_LEN``. Dense
    MLA only -- the sparse router is not under test here.
    """
    if accelerator_api() == "hip" and (
        cache_dtype not in _FLOAT8_DTYPES or heads > 16
    ):
        max_seq_len = 1
    else:
        max_seq_len = 8
    return _BLOCK_ROWS <= max_seq_len


# ===-----------------------------------------------------------------------===#
# The layer, the input draw and one kernel call
# ===-----------------------------------------------------------------------===#


def _kv_params(
    heads: int, cache_dtype: DType, device: Accelerator
) -> MLAKVCacheParams:
    return MLAKVCacheParams(
        dtype=cache_dtype,
        head_dim=_KV_LORA_RANK + _QK_ROPE_HEAD_DIM,
        num_layers=1,
        devices=[DeviceRef.from_device(device)],
        page_size=_PAGE_SIZE,
        num_q_heads=heads,
    )


def _draw_bf16(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """A random float32 draw already rounded to the bf16 grid.

    The layer's weights and input are bf16 casts of the same draw, so the
    kernel and every reference see bit-identical operand values.
    """
    return _bf16(rng.standard_normal(shape) * _WEIGHT_SCALE)


def _make_layer(
    heads: int,
    params: MLAKVCacheParams,
    rng: np.random.Generator,
) -> tuple[KimiK3MLAAttention, dict[str, np.ndarray]]:
    """Builds a random-weight MLA layer with production head geometry.

    Returns the layer and its bf16-valued weight draws, for the numpy
    reference. Only the weights the attention core reads are drawn; the
    output gate and ``o_proj`` never run, because the comparison stops at
    the attention output.
    """
    weights = {
        "q_a_proj": _draw_bf16(rng, (_Q_LORA_RANK, _HIDDEN)),
        "q_b_proj": _draw_bf16(
            rng,
            (heads * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM), _Q_LORA_RANK),
        ),
        "kv_a_proj_with_mqa": _draw_bf16(
            rng, (_KV_LORA_RANK + _QK_ROPE_HEAD_DIM, _HIDDEN)
        ),
        "kv_b_proj": _draw_bf16(
            rng,
            (heads * (_QK_NOPE_HEAD_DIM + _V_HEAD_DIM), _KV_LORA_RANK),
        ),
    }
    layer = KimiK3MLAAttention(
        hidden_size=_HIDDEN,
        num_heads=heads,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        q_lora_rank=_Q_LORA_RANK,
        kv_lora_rank=_KV_LORA_RANK,
        kv_params=params,
        layer_idx=0,
    )
    for name, draw in weights.items():
        weight = getattr(layer, name).weight
        getattr(layer, name).weight = Tensor(draw, dtype=DType.float32).cast(
            weight.dtype
        )
    layer.q_a_layernorm.weight = Tensor(
        np.ones(_Q_LORA_RANK, np.float32), dtype=DType.float32
    ).cast(layer.q_a_layernorm.weight.dtype)
    layer.kv_a_layernorm.weight = Tensor(
        np.ones(_KV_LORA_RANK, np.float32), dtype=DType.float32
    ).cast(layer.kv_a_layernorm.weight.dtype)
    return layer, weights


def _block_input(rng: np.random.Generator) -> np.ndarray:
    """The block call's hidden states, bf16-valued: ``[requests, rows,
    hidden]``."""
    return _bf16(
        rng.standard_normal((len(_CACHE_LENGTHS), _BLOCK_ROWS, _HIDDEN))
        * _WEIGHT_SCALE
    )


def _mla_attention(
    layer: KimiK3MLAAttention,
    hidden_block: np.ndarray,
    mask_variant: MHAMaskVariant,
    device: Accelerator,
    buffer_size: int,
    context_rows: list[np.ndarray],
    freqs_cis: Tensor | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """One eager ``mla_prefill_decode_graph`` call over a per-request block.

    Mirrors ``KimiK3MLAAttention.forward_latent_ragged``'s attention core
    (``mla_attention.py``) with the mask variant, rope table and scale
    exposed -- the layer hardcodes ``CAUSAL_MASK``, the identity table and
    ``q_head_dim ** -0.5`` -- and stops before the output gate and
    ``o_proj``, so the comparison sees the attention output itself.

    ``hidden_block`` is request-major: ``[requests, rows, hidden]``, in the
    call's row order. Returns ``[requests, rows, heads * v_head_dim]`` as
    float32.
    """
    num_requests, rows, _ = hidden_block.shape
    total_rows = num_requests * rows
    assert isinstance(layer.kv_params, MLAKVCacheParams)
    x = Tensor(hidden_block.reshape(total_rows, _HIDDEN), dtype=DType.float32)
    input_row_offsets = Tensor(
        np.arange(0, total_rows + 1, rows, dtype=np.uint32),
        dtype=DType.uint32,
    )
    # A fresh cache per call -- context prefilled with the same bytes each
    # time, block slots zero -- so no call depends on an earlier one's
    # writes and every call sees an identical context.
    kv_collection = build_paged_cache(
        layer.kv_params,
        [rows] * num_requests,
        device,
        cache_lengths=_CACHE_LENGTHS,
    )
    kv_collection.kv_blocks = _blocks_buffer(
        layer.kv_params, [rows] * num_requests, context_rows, device
    )
    plan = kimi_k3_mla_prefill_plan(
        kv_params=layer.kv_params,
        kv_collection=kv_collection,
        input_row_offsets=input_row_offsets,
        buffer_size=buffer_size,
        # The host plan extent that production threads through the proposer:
        # each request's window page-aligned and summed, the value
        # ``spec_block_extent_bound`` bounds for a real batch.
        # The kernel reads the extent as a host scalar, so the tensor must
        # not ride the caller's default device.
        buffer_length=Tensor(
            np.array(
                [
                    sum(
                        ceildiv(cache + rows, _PAGE_SIZE) * _PAGE_SIZE
                        for cache in _CACHE_LENGTHS
                    )
                ],
                np.int32,
            ),
            dtype=DType.int32,
            device=CPU(),
        ),
    )
    if freqs_cis is None:
        freqs_cis = kimi_k3_nope_freqs_cis(
            max_seq_len=buffer_size,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
            dtype=layer.kv_params.dtype,
            device=device,
        )

    q_lat = layer.q_a_layernorm(layer.q_a_proj(x))
    query = F.reshape(
        layer.q_b_proj(q_lat),
        [total_rows, layer.num_heads, layer.q_head_dim],
    )
    # UNNORMALIZED: the kernel normalizes the latent half in the cache after
    # storing it, so normalizing here would apply the gain twice.
    kv = layer.kv_a_proj_with_mqa(x)
    kv_b = F.reshape(
        layer.kv_b_proj.weight.T,
        [
            layer.kv_lora_rank,
            layer.num_heads,
            layer.qk_nope_head_dim + layer.v_head_dim,
        ],
    )
    w_uk_base = kv_b[:, :, : layer.qk_nope_head_dim]

    scalar_args = kv_collection.attention_dispatch_metadata
    num_partitions_scalar = kv_collection.mla_num_partitions
    if scalar_args is None or num_partitions_scalar is None:
        raise ValueError("an MLA cache must provide the MLA dispatch scalars")

    attn = mla_prefill_decode_graph(
        q=query,
        kv=kv,
        input_row_offsets=input_row_offsets,
        freqs_cis=F.cast(freqs_cis, query.dtype),
        kv_norm_gamma=layer.kv_a_layernorm.weight,
        buffer_row_offsets=plan.buffer_row_offsets,
        cache_offsets=plan.cache_offsets,
        buffer_length=plan.buffer_lengths,
        # The same K up-projection twice: flat for prefill's plain matmul,
        # per-head batched for decode's fold.
        w_k=F.reshape(
            F.permute(w_uk_base, [1, 2, 0]), [-1, layer.kv_lora_rank]
        ),
        w_uk=F.transpose(w_uk_base, 0, 1),
        w_uv=F.permute(kv_b[:, :, layer.qk_nope_head_dim :], [1, 2, 0]),
        kv_params=layer.kv_params,
        kv_collection=kv_collection,
        layer_idx=F.constant(layer.layer_idx, DType.uint32, device=CPU()),
        mask_variant=mask_variant,
        scale=layer.q_head_dim**-0.5 if scale is None else scale,
        epsilon=layer.kv_a_layernorm.eps,
        v_head_dim=layer.v_head_dim,
        scalar_args=scalar_args,
        num_partitions_scalar=num_partitions_scalar,
    )
    return attn.cast(DType.float32).to_numpy().reshape(num_requests, rows, -1)


# ===-----------------------------------------------------------------------===#
# Oracles
# ===-----------------------------------------------------------------------===#


def _full_window_reference(
    layer: KimiK3MLAAttention,
    hidden: np.ndarray,
    context_rows: list[np.ndarray],
    device: Accelerator,
    buffer_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Null semantics built from causal calls, one arrangement per row.

    For each block row ``r``, a causal call arranges the request's rows with
    ``r`` moved last. The stored keys are position-independent (identity
    rope), and a block's last row is the only one whose causal boundary is
    the whole window, so its output attends to the context plus every block
    row's key -- exactly what the null mask must give row ``r``.

    Returns the assembled reference ``[requests, block_rows, flat]`` and the
    natural-order causal run's full output (the last arrangement is the
    identity, so its whole tensor is the causal arm).
    """
    per_row = []
    causal_natural = None
    for r in range(_BLOCK_ROWS):
        order = [j for j in range(_BLOCK_ROWS) if j != r] + [r]
        out = _mla_attention(
            layer,
            hidden[:, order, :],
            MHAMaskVariant.CAUSAL_MASK,
            device,
            buffer_size,
            context_rows,
        )
        per_row.append(out[:, -1, :])
        if r == _BLOCK_ROWS - 1:
            causal_natural = out
    assert causal_natural is not None
    return np.stack(per_row, axis=1), causal_natural


def _dropped_last_key_reference(
    layer: KimiK3MLAAttention,
    hidden: np.ndarray,
    context_rows: list[np.ndarray],
    device: Accelerator,
    buffer_size: int,
) -> np.ndarray:
    """The reference minus the last query row's key, for the earlier rows.

    Row ``r``'s window omits the last block row's key: a causal call over
    the first ``S - 1`` rows with ``r`` moved last attends to the context
    plus exactly those rows' keys. The last row itself keeps the true
    full-window reference -- a causal call always stores and then sees a
    row's own key, so its own window cannot lose it. The sabotage is
    therefore surgical: it perturbs rows ``0..S-2`` and touches nothing else.
    """
    per_row = []
    for r in range(_BLOCK_ROWS - 1):
        order = [j for j in range(_BLOCK_ROWS - 1) if j != r] + [r]
        out = _mla_attention(
            layer,
            hidden[:, order, :],
            MHAMaskVariant.CAUSAL_MASK,
            device,
            buffer_size,
            context_rows,
        )
        per_row.append(out[:, -1, :])
    natural = _mla_attention(
        layer,
        hidden,
        MHAMaskVariant.CAUSAL_MASK,
        device,
        buffer_size,
        context_rows,
    )
    per_row.append(natural[:, -1, :])
    return np.stack(per_row, axis=1)


# ===-----------------------------------------------------------------------===#
# The YaRN case's float32 reference
# ===-----------------------------------------------------------------------===#


# The yarn case's weight draw amplifies the rope slices -- every head's
# rope rows of ``q_b_proj`` and the latent's rope tail -- so the
# position-sensitive part of the score carries mass. In the float32 model
# of the comparison, a +1-position defect moves the output by only 0.4-0.5%
# per request at the natural draw's rope/nope balance (the nope dot, which
# no position can touch, dominates the score), far below the
# kernel-vs-kernel bound; this gain lifts the measured signal to 0.32-0.62
# per request on MI355, 13x the bound at the weakest. Bigger is not
# better: in the float32 model a gain of 32 peaks the softmax enough that
# one request's signal collapses to 1.2x the bound.
_YARN_ROPE_GAIN = 12.0


def _amplify_yarn_rope(
    weights: dict[str, np.ndarray],
    layer: KimiK3MLAAttention,
    gain: float = _YARN_ROPE_GAIN,
) -> None:
    """Scales the yarn draw's rope slices, on the draw and on the layer."""
    heads = layer.num_heads
    q_b = weights["q_b_proj"].copy()
    for head in range(heads):
        base = head * layer.q_head_dim
        q_b[base + layer.qk_nope_head_dim : base + layer.q_head_dim, :] *= gain
    weights["q_b_proj"] = _bf16(q_b)
    kv = weights["kv_a_proj_with_mqa"].copy()
    kv[_KV_LORA_RANK:, :] *= gain
    weights["kv_a_proj_with_mqa"] = _bf16(kv)
    layer.q_b_proj.weight = Tensor(
        weights["q_b_proj"], dtype=DType.float32
    ).cast(layer.q_b_proj.weight.dtype)
    layer.kv_a_proj_with_mqa.weight = Tensor(
        weights["kv_a_proj_with_mqa"], dtype=DType.float32
    ).cast(layer.kv_a_proj_with_mqa.weight.dtype)


def _yarn_drafter_config() -> K3DSparkDraftConfig:
    """The published drafter's config (Inferact/Kimi-K3-DSpark), rope fields
    exact: yarn factor 32, theta 5e4, original context 32768, mscale
    1.0/1.0 -- so the table rotates for real and the scale carries the
    yarn mscale squared."""
    return K3DSparkDraftConfig(
        hidden_size=7168,
        intermediate_size=14336,
        num_hidden_layers=5,
        num_attention_heads=64,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        vocab_size=163840,
        mask_token_id=163837,
        target_layer_ids=(2, 23, 47, 71, 89),
        num_target_layers=93,
        markov_rank=256,
        rms_norm_eps=1e-5,
        rope_theta=50000.0,
        rope_factor=32.0,
        rope_original_max_position_embeddings=32768,
        rope_beta_fast=32.0,
        rope_beta_slow=1.0,
        rope_mscale=1.0,
        rope_mscale_all_dim=1.0,
    )


def _yarn_f32_reference(
    weights: dict[str, np.ndarray],
    hidden: np.ndarray,
    context_rows: list[np.ndarray],
    table: np.ndarray,
    scale: float,
    q_eps: float,
    kv_eps: float,
    heads: int,
) -> np.ndarray:
    """The fold's null-mask math in float32, as far as source can model it.

    Mirrors ``mla_decode_branch_bf16``: bf16 projections and norms, the
    absorbed ``w_uk`` q projection, e4m3 cache rows for the block stores
    the kernel writes (the store's bytes are verified against this model),
    softmax over the request's FULL window (null semantics), the
    bf16-rounded attention accumulator, and the ``w_uv`` output
    projection.

    The bound this reference can support is loose, and deliberately so:
    the fold runs its q side through an internal FP8 MMA cascade whose
    quantization points a source-level model cannot reproduce (guessing an
    e4m3 staging of q would mis-model the arm twice over -- the kernel's q
    quantizes somewhere the source does not say -- so q stays in float32
    here and the bound absorbs the difference, measured at 5-8% relative
    per request). What this check gates is the semantics an O(1) error
    breaks -- the window composition, the null key set, the position
    formula, the absorbed layout. A one-position rope defect is 0.4-0.5%
    here at the natural draw and is gated by the shifted-table RED instead,
    kernel against kernel where the q cascade runs identically in both
    arms.
    """
    eps_q = np.float32(q_eps)
    eps_kv = np.float32(kv_eps)
    out = []
    for request, cache in enumerate(_CACHE_LENGTHS):
        x = hidden[request]  # [S, hidden], bf16-valued
        rows = _BLOCK_ROWS
        q_a = _bf16(x @ weights["q_a_proj"].T)
        q_lat = _bf16(
            q_a / np.sqrt(np.mean(q_a**2, axis=-1, keepdims=True) + eps_q)
        )
        q_full = _bf16(q_lat @ weights["q_b_proj"].T)
        q_full = q_full.reshape(
            rows, heads, _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM
        )
        q_nope = q_full[:, :, :_QK_NOPE_HEAD_DIM]
        q_rope = q_full[:, :, _QK_NOPE_HEAD_DIM:]

        kv_b = weights["kv_b_proj"].T.reshape(
            _KV_LORA_RANK, heads, _QK_NOPE_HEAD_DIM + _V_HEAD_DIM
        )
        q_positions = cache + np.arange(rows)
        q_latent = np.einsum(
            "lhs,rhs->hrl", kv_b[:, :, :_QK_NOPE_HEAD_DIM], q_nope
        )
        q_roped = _bf16(
            _rope_interleaved(q_rope, _bf16(table[q_positions])[:, None, :])
        ).transpose(1, 0, 2)

        # The block rows the kernel stores this call: norm, rope, quantize
        # -- then dequantized as the MMA reads them.
        kv_raw = _bf16(x @ weights["kv_a_proj_with_mqa"].T)
        block_positions = cache + np.arange(rows)
        normed = _rms_norm(kv_raw[:, :_KV_LORA_RANK], float(eps_kv))
        roped = _bf16(
            _rope_interleaved(
                kv_raw[:, _KV_LORA_RANK:], _bf16(table[block_positions])
            )
        )
        block_rows = _dequantize_rows(
            _quantize_rows(
                np.concatenate([normed, roped], axis=1), DType.float8_e4m3fn
            ),
            DType.float8_e4m3fn,
        )
        keys = np.concatenate([context_rows[request], block_rows], axis=0)

        # Null semantics: every row attends to every key in its window.
        scores = np.einsum(
            "hrd,kd->hrk", q_latent, keys[:, :_KV_LORA_RANK]
        ) + np.einsum("hrd,kd->hrk", q_roped, keys[:, _KV_LORA_RANK:])
        scores = scores.astype(np.float64) * scale
        weights_p = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights_p /= weights_p.sum(axis=-1, keepdims=True)
        accumulated = np.einsum(
            "hrk,kl->hrl", weights_p, keys[:, :_KV_LORA_RANK]
        )
        accumulated = _bf16(accumulated)
        attended = np.einsum(
            "lhv,hrl->hrv",
            kv_b[:, :, _QK_NOPE_HEAD_DIM:],
            accumulated,
        )
        out.append(
            _bf16(attended)
            .transpose(1, 0, 2)
            .reshape(rows, heads * _V_HEAD_DIM)
        )
    return np.stack(out, axis=0)


# ===-----------------------------------------------------------------------===#
# Tests
# ===-----------------------------------------------------------------------===#


@pytest.mark.compile_on_gpu(
    "the fused MLA prefill/decode graph op is not interpreter-bindable"
)
@pytest.mark.parametrize("heads,cache_dtype,arm", _ARMS)
def test_null_mask_matches_full_window_reference(
    heads: int, cache_dtype: DType, arm: str
) -> None:
    """Seven rows under NULL_MASK attend to the whole window, per arm.

    Each arm pairs the null-mask block call against the causal-arrangement
    reference over a nonzero filled context; the only difference is the
    mask.
    """
    if accelerator_api() == "hip":
        assert (
            "decode_fold"
            if _routes_to_decode_fold(heads, cache_dtype)
            else "prefill"
        ) == arm, (
            f"heads={heads}, cache={cache_dtype} no longer routes S=7 to the"
            f" {arm} arm this test was built for; the kernel router moved"
        )

    device = Accelerator()
    with default_device(device):
        params = _kv_params(heads, cache_dtype, device)
        layer, _ = _make_layer(heads, params, np.random.default_rng(_SEED))
        hidden = _block_input(np.random.default_rng(_SEED + 1))
        context = _context_byte_rows(
            _context_latents(np.random.default_rng(_SEED + 2)),
            kimi_k3_nope_freqs_cis(
                max_seq_len=_plan_cap(heads, cache_dtype),
                qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
                dtype=DType.float32,
                device=CPU(),
            ).to_numpy(),
            cache_dtype,
            layer.kv_a_layernorm.eps,
        )
        buffer_size = _plan_cap(heads, cache_dtype)

        null_out = _mla_attention(
            layer,
            hidden,
            MHAMaskVariant.NULL_MASK,
            device,
            buffer_size,
            context,
        )
        reference, causal_out = _full_window_reference(
            layer, hidden, context, device, buffer_size
        )

    tol = _TOL_PARITY_BF16 if cache_dtype == DType.bfloat16 else _TOL_PARITY_FP8
    errs = _report(
        f"null mask heads={heads} cache={cache_dtype} vs full-window reference",
        null_out,
        reference,
    )
    assert np.all(errs < tol), (
        f"NULL_MASK block call (heads={heads}, cache={cache_dtype}) diverges"
        f" from the full-window reference per request: {errs}"
    )

    # RED control: causal where null is expected must NOT pass the check
    # above, in any request. Causal hides the other rows' keys from every
    # row but the last, so this must diverge; if it did not, the parity
    # check could not tell the masks apart and a kernel that ignored the
    # mask string would slip through it.
    red = _request_rel_errors(causal_out, reference)
    print(f"RED causal-where-null (heads={heads}): per-request rel={red}")
    assert np.all(red > tol), (
        "a CAUSAL_MASK call passed the null-mask parity check for requests"
        f" {np.nonzero(red <= tol)[0]} (heads={heads}, rel={red}); the test"
        " cannot tell the masks apart there"
    )


@pytest.mark.compile_on_gpu(
    "the fused MLA prefill/decode graph op is not interpreter-bindable"
)
def test_red_control_dropped_last_key() -> None:
    """A reference missing the last row's key must fail the parity check.

    Models the defect the mask machinery could plausibly hide -- one key
    dropped from every row's window (an off-by-one in the fold's
    ``num_keys``, or the OOB cutoff masking one row too many). Rows
    ``0..S-2``'s references lose the last block row's key, so the null arm
    must diverge from them in EVERY request -- the page-crossing requests
    included, which a batch-normalized error would hide behind the short
    request's scale -- while the last row's reference keeps the true window
    and must still match, proving the sabotage perturbs exactly what it
    claims to. Runs on the production fold arm; the measured signal sits
    at >= 3x the parity bound in every request (0.075 at its weakest, the
    cache=125 page-crosser), so the other arms inherit its verdict.
    """
    heads = 8
    cache_dtype = DType.float8_e4m3fn
    device = Accelerator()
    with default_device(device):
        params = _kv_params(heads, cache_dtype, device)
        layer, _ = _make_layer(heads, params, np.random.default_rng(_SEED))
        hidden = _block_input(np.random.default_rng(_SEED + 1))
        context = _context_byte_rows(
            _context_latents(np.random.default_rng(_SEED + 2)),
            kimi_k3_nope_freqs_cis(
                max_seq_len=_plan_cap(heads, cache_dtype),
                qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
                dtype=DType.float32,
                device=CPU(),
            ).to_numpy(),
            cache_dtype,
            layer.kv_a_layernorm.eps,
        )
        buffer_size = _plan_cap(heads, cache_dtype)

        null_out = _mla_attention(
            layer,
            hidden,
            MHAMaskVariant.NULL_MASK,
            device,
            buffer_size,
            context,
        )
        dropped = _dropped_last_key_reference(
            layer, hidden, context, device, buffer_size
        )

    err = _report(
        f"RED dropped-last-key (fold arm) rows 0..{_BLOCK_ROWS - 2}",
        null_out[:, :-1, :],
        dropped[:, :-1, :],
    )
    assert np.all(err > _TOL_PARITY_FP8), (
        "a reference missing the last query row's key passed the null-mask"
        f" parity check for requests {np.nonzero(err <= _TOL_PARITY_FP8)[0]}"
        f" (rel={err}); the tolerance cannot catch a dropped key there"
    )
    last = _rel_error(null_out[:, -1, :], dropped[:, -1, :])
    print(f"RED dropped-last-key surgical last row (fold arm): {last}")
    assert last < _TOL_PARITY_FP8, (
        "the dropped-key sabotage changed the last row's reference"
        f" (rel={last}); it is not surgical"
    )


@pytest.mark.compile_on_gpu(
    "the fused MLA prefill/decode graph op is not interpreter-bindable"
)
def test_yarn_null_mask_matches_f32_reference() -> None:
    """The fold's null mask under the drafter's REAL yarn table.

    The arrangement oracle is invalid here -- a positional rope makes a
    row's key depend on where it is stored -- so this test gates three
    properties, each with the comparison its signal can survive:

    * SEMANTICS, against a float32 numpy reference that mirrors the
      kernel's math from source (bf16 projections, absorbed w_uk layout,
      e4m3 cache rows -- the store model is verified byte-for-byte -- and
      null attention over the whole window). The bound is loose because
      the fold runs q through an internal FP8 MMA cascade a source-level
      model cannot reproduce; what the check catches is any O(1) error --
      a wrong window, a wrong key set, a wrong position formula, a wrong
      absorption layout.
    * POSITIONS, kernel against kernel: the same NULL_MASK call fed a rope
      table shifted one row -- which ropes every row the kernel owns one
      position late, the off-by-one a position bug would be -- must
      diverge per request well past the kernel-vs-kernel noise floor.
      Injected through the runtime table input rather than the reference
      because the kernel-vs-float32 comparison's floor (the fold's FP8 q
      cascade, measured 5-8% per request) exceeds the natural draw's
      0.4-0.5% position signal; kernel against kernel the cascade runs
      identically in both arms. The rope-amplified draw
      (``_YARN_ROPE_GAIN``) is what lifts the signal: at the natural
      rope/nope balance the position-insensitive nope dot dominates the
      score, and the amplified signal measures 13x the bound at its
      weakest request.
    * MASK ARITHMETIC, kernel against kernel: under any rope, a null
      call's last row must equal a causal call's last row -- both see the
      whole window at the same positions, so only the mask differs.
    """
    heads = 8
    cache_dtype = DType.float8_e4m3fn
    device = Accelerator()
    with default_device(device):
        params = _kv_params(heads, cache_dtype, device)

        # The natural-draw semantics and mask-arithmetic comparisons.
        layer, weights = _make_layer(
            heads, params, np.random.default_rng(_SEED)
        )
        hidden = _block_input(np.random.default_rng(_SEED + 1))
        config = _yarn_drafter_config()
        buffer_size = _plan_cap(heads, cache_dtype)
        table = k3_dspark_yarn_freqs_cis(config, positions=buffer_size)
        context_rows = _context_byte_rows(
            _context_latents(np.random.default_rng(_SEED + 2)),
            table,
            cache_dtype,
            layer.kv_a_layernorm.eps,
        )
        context_f32 = [
            _dequantize_rows(rows, cache_dtype) for rows in context_rows
        ]
        scale = config.attention_scale

        freqs_cis = Tensor(table, dtype=DType.float32).to(device)
        null_out = _mla_attention(
            layer,
            hidden,
            MHAMaskVariant.NULL_MASK,
            device,
            buffer_size,
            context_rows,
            freqs_cis=freqs_cis,
            scale=scale,
        )
        causal_out = _mla_attention(
            layer,
            hidden,
            MHAMaskVariant.CAUSAL_MASK,
            device,
            buffer_size,
            context_rows,
            freqs_cis=freqs_cis,
            scale=scale,
        )
        reference = _yarn_f32_reference(
            weights,
            hidden,
            context_f32,
            _bf16(table),
            scale,
            layer.q_a_layernorm.eps,
            layer.kv_a_layernorm.eps,
            heads,
        )

        # The position RED: one draw with the rope slices amplified so the
        # position-sensitive score carries mass, called twice -- once with
        # the true table, once with a table shifted one row, which is
        # "rope at position + 1" for every row the kernel ropes.
        red_layer, red_weights = _make_layer(
            heads, params, np.random.default_rng(_SEED)
        )
        _amplify_yarn_rope(red_weights, red_layer)
        red_hidden = _block_input(np.random.default_rng(_SEED + 1))
        red_context = _context_byte_rows(
            _context_latents(np.random.default_rng(_SEED + 2)),
            table,
            cache_dtype,
            red_layer.kv_a_layernorm.eps,
        )
        true_table = _mla_attention(
            red_layer,
            red_hidden,
            MHAMaskVariant.NULL_MASK,
            device,
            buffer_size,
            red_context,
            freqs_cis=freqs_cis,
            scale=scale,
        )
        shifted_table = k3_dspark_yarn_freqs_cis(
            config, positions=buffer_size + 1
        )[1:]
        late_table = _mla_attention(
            red_layer,
            red_hidden,
            MHAMaskVariant.NULL_MASK,
            device,
            buffer_size,
            red_context,
            freqs_cis=Tensor(shifted_table, dtype=DType.float32).to(device),
            scale=scale,
        )

    errs = _report("yarn null vs float32 reference", null_out, reference)
    assert np.all(errs < _TOL_YARN_F32), (
        f"the fold's NULL_MASK call under the drafter's yarn table diverges"
        f" from the float32 reference per request: {errs}"
    )

    # RED: the same call roped one position late must move every request
    # past the kernel-vs-kernel noise floor by a wide margin.
    red = _request_rel_errors(late_table, true_table)
    print(f"RED yarn rope-one-position-late: per-request rel={red}")
    assert np.all(red > _TOL_YARN_POSITION), (
        "a rope table shifted one position produced output the comparison"
        f" cannot tell from the true table's (rel={red}); the test cannot"
        " catch a rope-position off-by-one"
    )
    # Under any rope, the last block row sees the whole window under both
    # masks at the same positions, so the two kernels must agree on it.
    last = _rel_error(null_out[:, -1, :], causal_out[:, -1, :])
    print(f"yarn null vs causal last row (fold arm): {last}")
    assert last < _TOL_PARITY_FP8, (
        "under the yarn table the null and causal calls' last rows differ"
        f" (rel={last}); one of them is not attending the full window"
    )
