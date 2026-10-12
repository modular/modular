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
"""Tests the padded (non-ragged) paged KV cache graph ops on GPU."""

from __future__ import annotations

import numpy as np
from max import tree
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_padded_kv_cache,
    fused_qkv_padded_matmul,
)
from max.nn.kv_cache import MHAKVCacheParams
from test_common.simple_kv_cache import paged_kv_cache_inputs

TOTAL_NUM_PAGES = 16
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 64
HIDDEN_DIM = 128
PAGE_SIZE = 128
VALID_LENGTHS = [5, 140, 1]

# The float32 GEMM runs on TF32 tensor cores, so the projections carry about
# three significant digits; the attention reference inherits that error.
QKV_TOL = 1e-2
ATTN_TOL = 2e-2


def _kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
    )


def _reference_qkv(
    x: np.ndarray, wqkv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits ``x @ wqkv.T`` into Q, K and V with per-head trailing dims."""
    batch_size, seq_len, _ = x.shape
    qkv = x @ wqkv.T
    q_dim = N_HEADS * HEAD_DIM
    kv_dim = N_KV_HEADS * HEAD_DIM
    q = qkv[..., :q_dim].reshape(batch_size, seq_len, N_HEADS, HEAD_DIM)
    k = qkv[..., q_dim : q_dim + kv_dim].reshape(
        batch_size, seq_len, N_KV_HEADS, HEAD_DIM
    )
    v = qkv[..., q_dim + kv_dim :].reshape(
        batch_size, seq_len, N_KV_HEADS, HEAD_DIM
    )
    return q, k, v


def _reference_causal_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, valid_length: int
) -> np.ndarray:
    """Causal attention for one sequence's first ``valid_length`` rows."""
    group = N_HEADS // N_KV_HEADS
    out = np.zeros((valid_length, N_HEADS, HEAD_DIM), dtype=np.float32)
    for h in range(N_HEADS):
        kv_h = h // group
        scores = q[:valid_length, h] @ k[:valid_length, kv_h].T
        scores = scores / np.sqrt(HEAD_DIM)
        scores = np.where(
            np.tri(valid_length, dtype=bool), scores, -np.inf
        ).astype(np.float32)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        out[:, h] = probs @ v[:valid_length, kv_h]
    return out


def _build_graph(*, with_attention: bool) -> Graph:
    kv_params = _kv_params()
    batch_size = len(VALID_LENGTHS)
    padded_seq_len = max(VALID_LENGTHS)
    qkv_dim = (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM

    x_type = TensorType(
        DType.float32,
        [batch_size, padded_seq_len, HIDDEN_DIM],
        device=DeviceRef.GPU(),
    )
    wqkv_type = TensorType(
        DType.float32, [qkv_dim, HIDDEN_DIM], device=DeviceRef.GPU()
    )
    valid_lengths_type = TensorType(
        DType.uint32, [batch_size], device=DeviceRef.GPU()
    )

    with Graph(
        "kv_cache_padded",
        input_types=[
            x_type,
            wqkv_type,
            valid_lengths_type,
            *kv_params.flattened_kv_inputs(),
        ],
    ) as graph:
        x_in, wqkv_in, valid_lengths_in, *_kv_rest = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(graph.inputs[3:]))[0]
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        q = fused_qkv_padded_matmul(
            kv_params,
            x_in.tensor,
            wqkv_in.tensor,
            kv_collection,
            layer_idx,
            valid_lengths_in.tensor,
            N_HEADS,
        )
        if with_attention:
            q = ops.reshape(q, [batch_size, padded_seq_len, N_HEADS, HEAD_DIM])
            q = flash_attention_padded_kv_cache(
                kv_params,
                q,
                kv_collection,
                layer_idx,
                valid_lengths_in.tensor,
                MHAMaskVariant.CAUSAL_MASK,
                scale=1.0 / np.sqrt(HEAD_DIM),
            )
        graph.output(q)
    return graph


def _run(graph: Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs ``graph`` and returns ``(x, wqkv, output, kv_blocks)`` as numpy."""
    device = Accelerator()
    kv_params = _kv_params()
    session = InferenceSession(devices=[device])
    model = session.load(graph)
    runtime_inputs = paged_kv_cache_inputs(
        kv_params, VALID_LENGTHS, total_num_pages=TOTAL_NUM_PAGES
    )

    rng = np.random.default_rng(7)
    x_np = rng.standard_normal(
        (len(VALID_LENGTHS), max(VALID_LENGTHS), HIDDEN_DIM), dtype=np.float32
    )
    wqkv_np = rng.standard_normal(
        ((N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM, HIDDEN_DIM), dtype=np.float32
    ) / np.sqrt(HIDDEN_DIM)
    wqkv_np = wqkv_np.astype(np.float32)
    lengths = np.array(VALID_LENGTHS, dtype=np.uint32)

    (output,) = model(
        Buffer.from_numpy(x_np).to(device),
        Buffer.from_numpy(wqkv_np).to(device),
        Buffer.from_numpy(lengths).to(device),
        *tree.leaves(runtime_inputs),
    )
    assert isinstance(output, Buffer)
    return (
        x_np,
        wqkv_np,
        output.to_numpy(),
        runtime_inputs.kv_blocks.to_numpy(),
    )


def _lookup_table() -> np.ndarray:
    kv_params = _kv_params()
    runtime_inputs = paged_kv_cache_inputs(
        kv_params, VALID_LENGTHS, total_num_pages=TOTAL_NUM_PAGES
    )
    return runtime_inputs.lookup_table.to_numpy()


def test_fused_qkv_padded_matmul_matches_reference() -> None:
    x_np, wqkv_np, q_out, kv_blocks = _run(_build_graph(with_attention=False))
    q_ref, k_ref, v_ref = _reference_qkv(x_np, wqkv_np)
    lookup_table = _lookup_table()

    np.testing.assert_allclose(
        q_out.reshape(q_ref.shape), q_ref, rtol=QKV_TOL, atol=QKV_TOL
    )

    # kv_blocks: [page, kv_idx, layer, token, kv_head, head_dim].
    for b, valid_length in enumerate(VALID_LENGTHS):
        for t in range(max(VALID_LENGTHS)):
            page = lookup_table[b, t // PAGE_SIZE]
            slot = t % PAGE_SIZE
            if t < valid_length:
                np.testing.assert_allclose(
                    kv_blocks[page, 0, 0, slot],
                    k_ref[b, t],
                    rtol=QKV_TOL,
                    atol=QKV_TOL,
                )
                np.testing.assert_allclose(
                    kv_blocks[page, 1, 0, slot],
                    v_ref[b, t],
                    rtol=QKV_TOL,
                    atol=QKV_TOL,
                )
            elif page < TOTAL_NUM_PAGES:
                # Padded positions inside an allocated page stay untouched.
                assert not kv_blocks[page, :, 0, slot].any()

    # The null page backing the lookup table's padding is never written.
    assert not kv_blocks[TOTAL_NUM_PAGES].any()


def test_flash_attention_padded_kv_cache_matches_reference() -> None:
    x_np, wqkv_np, attn_out, _ = _run(_build_graph(with_attention=True))
    q_ref, k_ref, v_ref = _reference_qkv(x_np, wqkv_np)

    for b, valid_length in enumerate(VALID_LENGTHS):
        expected = _reference_causal_attention(
            q_ref[b], k_ref[b], v_ref[b], valid_length
        )
        np.testing.assert_allclose(
            attn_out[b, :valid_length], expected, rtol=ATTN_TOL, atol=ATTN_TOL
        )


def test_print_kv_cache_paged_gpu_executes() -> None:
    device = Accelerator()
    kv_params = _kv_params()
    batch_size = len(VALID_LENGTHS)

    with Graph(
        "print_kv_cache_gpu",
        input_types=[
            TensorType(DType.uint32, [batch_size], device=DeviceRef.GPU()),
            *kv_params.flattened_kv_inputs(),
        ],
    ) as graph:
        valid_lengths_in, *_kv_rest = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(graph.inputs[1:]))[0]
        ops.inplace_custom(
            "mo.print_kv_cache.paged",
            device=DeviceRef.GPU(),
            values=[
                valid_lengths_in.tensor,
                *kv_collection.flatten_without_attention_dispatch_metadata(),
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                ops.constant(True, DType.bool, device=DeviceRef.CPU()),
            ],
        )
        graph.output()

    model = InferenceSession(devices=[device]).load(graph)
    runtime_inputs = paged_kv_cache_inputs(
        kv_params, VALID_LENGTHS, total_num_pages=TOTAL_NUM_PAGES
    )
    model(
        Buffer.from_numpy(np.array(VALID_LENGTHS, dtype=np.uint32)).to(device),
        *tree.leaves(runtime_inputs),
    )
