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
"""Op-level tests for ``mo.mla.store.ragged.paged``.

The op stores raw MLA latent rows into one layer of a paged cache: RMSNorm
over the ``kv_lora_rank``-wide head of each row, interleaved RoPE over the
rope tail at the row's absolute ``cache_length + token_idx`` position, and a
cast to the cache dtype. The oracle is the same math in float32; the
tolerance is sized to the cache dtype's own quantization step plus the
kernel's internal bfloat16 rounding, so it stays far below any real wiring
error.

Both cache dtypes the K3 drafter needs are covered: ``float8_e4m3fn`` (the
production latent) and ``bfloat16``. Real geometry throughout -- a 576-wide
latent (512 latent + 64 rope), a ragged batch with nonzero cache lengths,
requests whose stored ranges cross page boundaries, and a two-layer cache
written at a nonzero layer index.

Every oracle carries its RED controls, run through the same compiled graph
so they cost no extra compile: a RoPE table shifted one position (a rope
phase offset), a zero gamma (drops the norm), cache lengths bumped by one
(stores at ``cache_length + row + 1``), and a wrong layer index (stores into
layer 0) must each make the comparison fail.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max import tree
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.kernels import store_mla_latent_ragged
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    MLAKVCacheParams,
)
from test_common.simple_kv_cache import paged_kv_cache_inputs

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0,
    reason="mo.mla.store.ragged.paged is GPU-only",
)

KV_NORM_DIM = 512  # kv_lora_rank, the head of each row the RMSNorm covers
ROPE_DIM = 64  # qk_rope_head_dim, the tail of each row the RoPE covers
HEAD_DIM = KV_NORM_DIM + ROPE_DIM  # the real latent: 576
NUM_Q_HEADS = 8  # the drafter's per-rank heads at TP8
# The drafter stores into a nonzero layer of a multi-layer cache (five in
# production), so pin the layer addressing: LAYER_IDX holds the stores and
# every other layer must stay untouched.
NUM_LAYERS = 2
LAYER_IDX = 1
PAGE_SIZE = 16
MAX_POS = 64
THETA = 5e5
EPSILON = 1e-6
# Request 1's stored range [10, 18) crosses page 0 at slot 16; request 2's
# [29, 34) starts mid-page and crosses page 1 at slot 32.
CACHE_LENS = [10, 29]
PROMPT_LENS = [8, 5]
TOTAL_ROWS = sum(PROMPT_LENS)

# 384.0 in e4m3 (0x7C), and a huge finite bfloat16 (0x7F7F): unreachable
# magnitudes for the stored rows, so a poisoned slot still holding its fill
# value is provably untouched.
_POISON_BYTES = {DType.float8_e4m3fn: 0x7C, DType.bfloat16: 0x7F7F}
_UINT_DTYPES = {
    DType.float8_e4m3fn: DType.uint8,
    DType.bfloat16: DType.uint16,
}

# The comparison tolerance per cache dtype: the kernel rounds its bfloat16
# math before the final cast, so allow one destination-format ulp plus that
# rounding chain. Any real wiring error (a dropped gamma, a mis-normed row, a
# mis-roped position) is O(1) relative and sails past these.
_RTOL = {DType.float8_e4m3fn: 2.0**-3, DType.bfloat16: 2.0**-6}
_ATOL = 2.0**-9


@pytest.fixture(scope="module")
def gpu() -> tuple[InferenceSession, Accelerator]:
    device = Accelerator(0)
    return InferenceSession(devices=[device]), device


def _rope_table() -> np.ndarray:
    """Interleaved ``[cos, sin]`` RoPE table, shape ``[MAX_POS, ROPE_DIM]``."""
    inv_freq = 1.0 / (THETA ** (np.arange(0, ROPE_DIM, 2) / ROPE_DIM))
    ang = np.outer(np.arange(MAX_POS), inv_freq)
    return (
        np.stack([np.cos(ang), np.sin(ang)], axis=-1)
        .reshape(MAX_POS, ROPE_DIM)
        .astype(np.float32)
    )


def _kv_params(dtype: DType) -> MLAKVCacheParams:
    return MLAKVCacheParams(
        dtype=dtype,
        head_dim=HEAD_DIM,
        num_layers=NUM_LAYERS,
        page_size=PAGE_SIZE,
        num_q_heads=NUM_Q_HEADS,
        devices=[DeviceRef.GPU()],
    )


def _store_coords(
    lookup_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per stored row: its page id, its slot in the page, and its position.

    The page ids are read off the actual lookup table the kernel used, so
    they cannot drift from the pages the test helper assigned.
    """
    pages, slots, positions = [], [], []
    for batch, (cache_len, prompt_len) in enumerate(
        zip(CACHE_LENS, PROMPT_LENS, strict=True)
    ):
        for token_idx in range(prompt_len):
            pos = cache_len + token_idx
            pages.append(lookup_table[batch, pos // PAGE_SIZE])
            slots.append(pos % PAGE_SIZE)
            positions.append(pos)
    return np.asarray(pages), np.asarray(slots), np.asarray(positions)


def _expected_rows(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    freqs: torch.Tensor,
    lookup_table: np.ndarray,
) -> torch.Tensor:
    """The stored rows in float32, mirroring the kernel's math.

    RMSNorm over the first ``KV_NORM_DIM`` elements with ``gamma``, interleaved
    RoPE over the tail at each row's ``cache_length + token_idx`` position.
    Row 0 is all zeros and row 1 is scaled down, covering the epsilon path and
    a tiny row.
    """
    _, _, positions = _store_coords(lookup_table)
    x = kv[:, :KV_NORM_DIM].float()
    rstd = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + EPSILON)
    normed = x * rstd * gamma.float()

    v = kv[:, KV_NORM_DIM:].float()
    f = freqs[torch.as_tensor(positions)]
    a, b = v[:, 0::2], v[:, 1::2]
    f_re, f_im = f[:, 0::2], f[:, 1::2]
    re = a * f_re - b * f_im
    im = a * f_im + b * f_re
    roped = torch.stack([re, im], dim=-1).reshape(-1, ROPE_DIM)
    return torch.cat([normed, roped], dim=1)


def _build_graph(dtype: DType) -> Graph:
    kv_params = _kv_params(dtype)

    def forward(
        kv: TensorValue,
        input_row_offsets: TensorValue,
        freqs_cis: TensorValue,
        kv_norm_gamma: TensorValue,
        layer_idx: TensorValue,
        *cache_inputs: TensorValue,
    ) -> None:
        kv_collection = kv_params.unflatten_kv_inputs(iter(cache_inputs))[0]
        store_mla_latent_ragged(
            kv,
            input_row_offsets,
            freqs_cis,
            kv_norm_gamma,
            kv_params,
            kv_collection,
            layer_idx[0],
            EPSILON,
        )

    return Graph(
        "store_mla_latent_ragged",
        forward=forward,
        input_types=[
            TensorType(
                DType.bfloat16,
                [TOTAL_ROWS, HEAD_DIM],
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.uint32, [len(PROMPT_LENS) + 1], device=DeviceRef.GPU()
            ),
            TensorType(
                DType.float32, [MAX_POS, ROPE_DIM], device=DeviceRef.GPU()
            ),
            TensorType(DType.bfloat16, [KV_NORM_DIM], device=DeviceRef.GPU()),
            # A runtime scalar so the wrong-layer RED reuses this graph.
            TensorType(DType.uint32, [1], device=DeviceRef.CPU()),
            *kv_params.flattened_kv_inputs(),
        ],
    )


def _to_bf16_buffer(t: torch.Tensor, device: Accelerator) -> Buffer:
    bits = t.view(torch.uint16).numpy()
    return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)


def _run(
    model: Model,
    device: Accelerator,
    layer_idx: int,
    kv: torch.Tensor,
    gamma: torch.Tensor,
    freqs: np.ndarray,
    blocks: Buffer,
    cache_inputs: KVCacheInputsPerDevice[Buffer, Buffer],
) -> np.ndarray:
    """One execution over the given operands; returns the raw cache image."""
    cache_inputs.kv_blocks = blocks
    input_row_offsets = np.concatenate([[0], np.cumsum(PROMPT_LENS)])
    model(
        _to_bf16_buffer(kv, device),
        Buffer.from_numpy(input_row_offsets.astype(np.uint32)).to(device),
        Buffer.from_numpy(freqs).to(device),
        _to_bf16_buffer(gamma, device),
        Buffer.from_numpy(np.array([layer_idx], dtype=np.uint32)),
        *tree.leaves(cache_inputs),
    )
    host = cache_inputs.kv_blocks.copy(device=CPU())
    return host.view(_UINT_DTYPES[cache_inputs.kv_blocks.dtype]).to_numpy()


def _stored_rows(
    raw: np.ndarray, dtype: DType, lookup_table: np.ndarray, layer: int
) -> torch.Tensor:
    """Dequantized stored rows, gathered at their page/slot coordinates."""
    pages, slots, _ = _store_coords(lookup_table)
    image = torch.from_numpy(raw).view(
        torch.float8_e4m3fn if dtype == DType.float8_e4m3fn else torch.bfloat16
    )
    rows = image[pages, 0, layer, slots][:, 0, :]
    return rows.float()


def _assert_stored_matches(
    raw: np.ndarray,
    dtype: DType,
    expected: torch.Tensor,
    lookup_table: np.ndarray,
    layer: int = LAYER_IDX,
) -> None:
    torch.testing.assert_close(
        _stored_rows(raw, dtype, lookup_table, layer),
        expected,
        rtol=_RTOL[dtype],
        atol=_ATOL,
    )


def _assert_untouched(
    raw: np.ndarray, dtype: DType, lookup_table: np.ndarray
) -> None:
    """Every slot the op does not own still holds its poison fill value.

    That includes every layer but ``LAYER_IDX``, so a layer-addressing bug
    cannot hide behind a single-layer cache.
    """
    pages, slots, _ = _store_coords(lookup_table)
    masked = raw.copy()
    masked[pages, 0, LAYER_IDX, slots, 0, :] = _POISON_BYTES[dtype]
    assert np.all(masked == _POISON_BYTES[dtype]), (
        "the store touched slots it does not own"
    )


@pytest.mark.parametrize(
    "dtype", [DType.float8_e4m3fn, DType.bfloat16], ids=["fp8", "bf16"]
)
def test_store_mla_latent_ragged(
    gpu: tuple[InferenceSession, Accelerator], dtype: DType
) -> None:
    session, device = gpu
    model = session.load(_build_graph(dtype))

    torch.manual_seed(1757)
    kv = torch.randn(TOTAL_ROWS, HEAD_DIM).to(torch.bfloat16)
    kv[0] = 0.0
    kv[1] = 1e-3 * torch.randn(HEAD_DIM, dtype=torch.float32)
    gamma = torch.randn(KV_NORM_DIM).to(torch.bfloat16)
    freqs = _rope_table()

    cache_inputs = paged_kv_cache_inputs(
        _kv_params(dtype),
        PROMPT_LENS,
        cache_lengths=CACHE_LENS,
    )
    lookup_table = cache_inputs.lookup_table.copy(device=CPU()).to_numpy()

    def fresh_pool() -> Buffer:
        poison = np.full(
            cache_inputs.kv_blocks.shape,
            _POISON_BYTES[dtype],
            dtype=np.uint8 if dtype == DType.float8_e4m3fn else np.uint16,
        )
        return Buffer.from_numpy(poison).view(dtype).to(device)

    expected = _expected_rows(kv, gamma, torch.from_numpy(freqs), lookup_table)

    # The green run: stored rows match the reference and nothing else moves
    # -- in any layer.
    raw = _run(
        model, device, LAYER_IDX, kv, gamma, freqs, fresh_pool(), cache_inputs
    )
    _assert_stored_matches(raw, dtype, expected, lookup_table)
    _assert_untouched(raw, dtype, lookup_table)

    # RED (wrong rope phase): the same table shifted one row makes every
    # stored row rope at ``cache_length + row + 1`` while still landing in
    # its own slot, which must fail the comparison above. The store-offset
    # RED below moves the stored slot itself.
    shifted = np.concatenate([freqs[1:], freqs[-1:]], axis=0)
    raw = _run(
        model, device, LAYER_IDX, kv, gamma, shifted, fresh_pool(), cache_inputs
    )
    with pytest.raises(AssertionError):
        _assert_stored_matches(raw, dtype, expected, lookup_table)

    # RED (dropped norm): a zero gamma stores all-zero norm heads, which must
    # fail the comparison above. The rope tail stays correct, so this is the
    # control for the RMSNorm half of the oracle.
    raw = _run(
        model,
        device,
        LAYER_IDX,
        kv,
        torch.zeros_like(gamma),
        freqs,
        fresh_pool(),
        cache_inputs,
    )
    with pytest.raises(AssertionError):
        _assert_stored_matches(raw, dtype, expected, lookup_table)

    # RED (wrong store offset): bumping the cache lengths by one makes every
    # row land at ``cache_length + row + 1`` -- still inside the request's
    # assigned pages, so this is purely the offset bug -- and must fail the
    # comparison above.
    original_cache_lengths = cache_inputs.cache_lengths
    cache_inputs.cache_lengths = Buffer.from_numpy(
        (np.asarray(CACHE_LENS) + 1).astype(np.uint32)
    ).to(device)
    raw = _run(
        model, device, LAYER_IDX, kv, gamma, freqs, fresh_pool(), cache_inputs
    )
    cache_inputs.cache_lengths = original_cache_lengths
    with pytest.raises(AssertionError):
        _assert_stored_matches(raw, dtype, expected, lookup_table)

    # RED (wrong layer): storing with layer index 0 lands the rows in layer
    # 0 instead of ``LAYER_IDX``. The rows must be provably present in layer
    # 0, and the ``LAYER_IDX`` comparison must fail -- a store that ignored
    # the layer index, or wrote both layers, cannot satisfy both.
    raw = _run(model, device, 0, kv, gamma, freqs, fresh_pool(), cache_inputs)
    _assert_stored_matches(raw, dtype, expected, lookup_table, layer=0)
    with pytest.raises(AssertionError):
        _assert_stored_matches(raw, dtype, expected, lookup_table)
