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
"""Input-prologue fusion for ``mo.fused_qk_rms_norm_rope.ragged.paged``.

The kernel reads Q through a ``FusedInputTensor`` load lambda. When Q is a view
over a wider producer -- e.g. a ``slice`` + ``reshape`` that carves Q out of a
combined ``[Q | IndexQ]`` matmul output -- the graph compiler folds that
prologue into the Q read, so no materialized copy of the combined buffer is
needed.

These tests exercise the fusion end to end. The oracle is the un-fused path:
feeding Q as a plain rank-3 projection must produce the exact same normalized +
RoPE'd Q as feeding a ``slice`` + ``reshape`` view that resolves to the same
values. The underlying norm/RoPE math is validated numerically against a
two-step reference by the Mojo kernel test
(``max/kernels/test/gpu/nn/test_fused_qk_rms_norm_rope.mojo``); here we prove the
graph-compiler fusion is transparent.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import fused_qk_rms_norm_rope_ragged
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0,
    reason="fused_qk_rms_norm_rope.ragged.paged is GPU-only",
)

HEAD_DIM = 128
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
NUM_LAYERS = 1
PAGE_SIZE = 128
LAYER_IDX = 0
MAX_POS = 128
THETA = 5e5
EPSILON = 1e-6
PROMPT_LENS = [3, 5]
TOTAL_SEQ_LEN = sum(PROMPT_LENS)
Q_WIDTH = NUM_Q_HEADS * HEAD_DIM

DTYPES = [DType.float32, DType.bfloat16]
DTYPE_IDS = ["f32", "bf16"]


@pytest.fixture(scope="module")
def gpu() -> tuple[InferenceSession, Accelerator]:
    device = Accelerator(0)
    return InferenceSession(devices=[device]), device


def _rope_freqs(rope_dim: int) -> np.ndarray:
    """Interleaved ``[cos, sin]`` RoPE table, shape ``[MAX_POS, rope_dim]``."""
    inv_freq = 1.0 / (THETA ** (np.arange(0, rope_dim, 2) / rope_dim))
    ang = np.outer(np.arange(MAX_POS), inv_freq)
    return np.stack([np.cos(ang), np.sin(ang)], axis=-1).reshape(
        MAX_POS, rope_dim
    )


def _to_device(arr: np.ndarray, dtype: DType, device: Accelerator) -> Buffer:
    if dtype == DType.bfloat16:
        bits = np.ascontiguousarray(
            arr.astype(ml_dtypes.bfloat16).view(np.uint16)
        )
        return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)
    return Buffer.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).to(
        device
    )


def _from_device(buf: Buffer, dtype: DType) -> np.ndarray:
    out = buf.copy(device=CPU())
    if dtype == DType.bfloat16:
        return (
            out.view(DType.uint16)
            .to_numpy()
            .view(ml_dtypes.bfloat16)
            .astype(np.float32)
        )
    return out.to_numpy()


def _row_offsets() -> np.ndarray:
    return np.concatenate([[0], np.cumsum(PROMPT_LENS)]).astype(np.uint32)


def _build_graph(
    dtype: DType,
    *,
    interleaved: bool,
    rope_dim: int,
    combined_width: int | None,
) -> Graph:
    """Builds a graph returning normalized + RoPE'd Q.

    ``combined_width`` selects the Q source: ``None`` feeds Q as a rank-3
    projection (un-fused); an integer feeds a rank-2 buffer of that width and
    carves Q out with a ``slice`` + ``reshape`` view (fused prologue). Q is
    always embedded at column ``HEAD_DIM`` of the combined buffer so the slice
    has a non-zero offset.
    """
    if combined_width is None:
        in_type = TensorType(
            dtype,
            ["total_seq_len", NUM_Q_HEADS, HEAD_DIM],
            device=DeviceRef.GPU(),
        )
    else:
        in_type = TensorType(
            dtype, ["total_seq_len", combined_width], device=DeviceRef.GPU()
        )

    kv_params = _kv_params(dtype)
    input_types = [
        in_type,
        TensorType(
            DType.uint32, ["input_row_offsets_len"], device=DeviceRef.GPU()
        ),
        TensorType(dtype, [MAX_POS, rope_dim], device=DeviceRef.GPU()),
        TensorType(dtype, [HEAD_DIM], device=DeviceRef.GPU()),  # q_gamma
        TensorType(dtype, [HEAD_DIM], device=DeviceRef.GPU()),  # k_gamma
        *kv_params.flattened_kv_inputs(),
    ]

    with Graph("fused_qk_rms_norm_rope_fusion", input_types=input_types) as g:
        inp = g.inputs[0].tensor
        row_offsets = g.inputs[1].tensor
        freqs_cis = g.inputs[2].tensor
        q_gamma = g.inputs[3].tensor
        k_gamma = g.inputs[4].tensor
        (
            blocks,
            cache_lengths,
            lookup_table,
            max_prompt_length,
            max_cache_length,
            _dispatch_metadata,
        ) = g.inputs[5:]

        if combined_width is not None:
            # slice + reshape carve Q out of the combined buffer; both fold into
            # the kernel's Q read lambda via input-prologue fusion.
            inp = inp[:, HEAD_DIM : HEAD_DIM + Q_WIDTH]
            inp = inp.reshape([inp.shape[0], NUM_Q_HEADS, HEAD_DIM])

        kv_collection = PagedCacheValues(
            blocks.buffer,
            cache_lengths.tensor,
            lookup_table.tensor,
            max_prompt_length.tensor,
            max_cache_length.tensor,
        )
        g.output(
            fused_qk_rms_norm_rope_ragged(
                kv_params,
                inp,
                row_offsets,
                kv_collection,
                q_gamma=q_gamma,
                k_gamma=k_gamma,
                freqs_cis=freqs_cis,
                epsilon=EPSILON,
                layer_idx=ops.constant(
                    LAYER_IDX, DType.uint32, DeviceRef.CPU()
                ),
                weight_offset=0.0,
                interleaved=interleaved,
            )
        )
    return g


def _kv_params(dtype: DType) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=NUM_LAYERS,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
    )


def _run(
    session: InferenceSession,
    device: Accelerator,
    dtype: DType,
    *,
    interleaved: bool,
    rope_dim: int,
    q_np: np.ndarray,
    gamma: np.ndarray,
    combined_width: int | None,
) -> np.ndarray:
    graph = _build_graph(
        dtype,
        interleaved=interleaved,
        rope_dim=rope_dim,
        combined_width=combined_width,
    )
    model = session.load(graph)

    # Allocate the paged cache for the batch. Its contents are irrelevant to Q,
    # which is computed purely from the projection.
    kv_manager = PagedKVCacheManager(
        _kv_params(dtype),
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )
    batch = [
        create_text_context(np.empty(n, dtype=np.int64)) for n in PROMPT_LENS
    ]
    for ctx in batch:
        kv_manager.claim(ctx.request_id, replica_idx=0)
        kv_manager.alloc(ctx, replica_idx=0)
    kv_rt = kv_manager.runtime_inputs_for_leaf([batch]).inputs[0]
    assert kv_rt.attention_dispatch_metadata is not None

    if combined_width is None:
        in_np = q_np
    else:
        in_np = np.zeros((TOTAL_SEQ_LEN, combined_width), dtype=np.float32)
        in_np[:, HEAD_DIM : HEAD_DIM + Q_WIDTH] = q_np.reshape(
            TOTAL_SEQ_LEN, Q_WIDTH
        )

    inputs = [
        _to_device(in_np, dtype, device),
        Buffer.from_numpy(_row_offsets()).to(device),
        _to_device(_rope_freqs(rope_dim), dtype, device),
        _to_device(gamma, dtype, device),  # q_gamma
        _to_device(gamma, dtype, device),  # k_gamma
        kv_rt.kv_blocks,
        kv_rt.cache_lengths,
        kv_rt.lookup_table,
        kv_rt.max_prompt_length,
        kv_rt.max_cache_length,
        kv_rt.attention_dispatch_metadata,
    ]
    (result,) = model.execute(*inputs)
    assert isinstance(result, Buffer)
    return _from_device(result, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize(
    "interleaved", [True, False], ids=["interleaved", "split"]
)
@pytest.mark.parametrize("rope_frac", [1, 2], ids=["full_rope", "partial_rope"])
def test_fused_matches_unfused(
    gpu: tuple[InferenceSession, Accelerator],
    dtype: DType,
    interleaved: bool,
    rope_frac: int,
) -> None:
    """A sliced + reshaped Q view must produce the same Q as the direct path.

    The two runs are identical except for how Q reaches the kernel: a plain
    rank-3 projection vs. a ``slice`` + ``reshape`` of a wider combined buffer
    that the graph compiler fuses into the Q load. The results must be
    bit-exact -- the fusion cannot change the values it feeds the kernel.
    """
    session, device = gpu
    rope_dim = HEAD_DIM // rope_frac

    rng = np.random.default_rng(0)
    q_np = rng.standard_normal((TOTAL_SEQ_LEN, NUM_Q_HEADS, HEAD_DIM)).astype(
        np.float32
    )
    # Non-uniform per-channel gamma catches any channel mis-indexing in the
    # fused Q read that a constant gamma would mask.
    gamma = rng.standard_normal(HEAD_DIM).astype(np.float32)

    # Combined buffer is wider than Q on both sides (offset HEAD_DIM before,
    # HEAD_DIM after) to mimic a real fused `[IndexQ | Q | ...]` layout.
    combined_width = Q_WIDTH + 2 * HEAD_DIM

    direct = _run(
        session,
        device,
        dtype,
        interleaved=interleaved,
        rope_dim=rope_dim,
        q_np=q_np,
        gamma=gamma,
        combined_width=None,
    )
    fused = _run(
        session,
        device,
        dtype,
        interleaved=interleaved,
        rope_dim=rope_dim,
        q_np=q_np,
        gamma=gamma,
        combined_width=combined_width,
    )

    assert direct.shape == (TOTAL_SEQ_LEN, NUM_Q_HEADS, HEAD_DIM)
    assert not np.any(np.isnan(fused))
    np.testing.assert_array_equal(fused, direct)
