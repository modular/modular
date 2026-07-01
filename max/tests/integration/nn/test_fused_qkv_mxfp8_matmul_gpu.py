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
"""Numeric test for the fused MXFP8 QKV-matmul-with-KV-write kernel on SM100.

Exercises the ``mo.fused_qkv_matmul.ragged.paged.scale.mxfp8`` path that
``quantized_fused_qkv_matmul`` uses for ``QuantFormat.MXFP8``: the activation
and the concatenated QKV weight are quantized to ``float8_e4m3fn`` with E8M0
block scales, the fused matmul runs, the Q projection is returned, and K/V are
written in place into a paged KV cache.

It checks two things:

1. The returned Q projection against an fp32 reference of the un-quantized
   ``a @ wqkv.T``.
2. The K/V cache contents against a bf16 reference path. Both paths write
   through the same paged-cache store epilogue, so the raw ``kv_blocks``
   buffers are directly comparable element-wise without decoding the paged
   layout. This is what confirms K and V land in the right slots.

Tolerances absorb the MXFP8 round-trip but are tight enough to catch a wrong
layout or wrong kernel.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.kernels import (
    _fused_qkv_ragged_matmul_scaled_mxfp8,
    fused_qkv_ragged_matmul,
    quantize_dynamic_block_scaled,
)
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParams,
    MHAKVCacheParams,
    PagedCacheValues,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from test_common.graph_utils import is_b100_b200


def _skip_if_not_supported() -> None:
    if accelerator_count() == 0:
        pytest.skip("No GPU available for MXFP8 fused-QKV test")
    if accelerator_api() == "hip":
        pytest.skip("MXFP8 block-scaled MMA only supports NVIDIA GPUs")
    if not is_b100_b200():
        pytest.skip("MXFP8 block-scaled MMA requires B100 or B200 (SM100)")


def _cosine_and_rel_l2(out: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    out_flat = out.reshape(-1).astype(np.float32)
    ref_flat = ref.reshape(-1).astype(np.float32)
    cos = float(
        np.dot(out_flat, ref_flat)
        / (np.linalg.norm(out_flat) * np.linalg.norm(ref_flat) + 1e-12)
    )
    rel = float(
        np.linalg.norm(out_flat - ref_flat) / (np.linalg.norm(ref_flat) + 1e-12)
    )
    return cos, rel


def _make_cache(
    kv_params: KVCacheParams,
    session: InferenceSession,
    seq_len: int,
) -> KVCacheInputsPerDevice[Buffer, Buffer]:
    """Allocate a single request's KV cache and return its runtime inputs."""
    manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=8,
    )
    context = create_text_context(np.empty(seq_len))
    manager.claim(context.request_id, replica_idx=0)
    manager.alloc(context, replica_idx=0)
    return manager.runtime_inputs_for_leaf([[context]]).inputs[0]


def _build_qkv_value(
    *,
    is_mxfp8: bool,
    a: TensorValue,
    wqkv: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    kv_params: KVCacheParams,
    num_heads: int,
) -> TensorValue:
    """Q projection for either the fused MXFP8 path or the bf16 reference."""
    if not is_mxfp8:
        return fused_qkv_ragged_matmul(
            kv_params,
            input=a,
            input_row_offsets=input_row_offsets,
            wqkv=wqkv,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=num_heads,
        )

    a_q, a_scales = quantize_dynamic_block_scaled(
        a,
        sf_vector_size=32,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )
    w_q, w_scales = quantize_dynamic_block_scaled(
        wqkv,
        sf_vector_size=32,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )
    return _fused_qkv_ragged_matmul_scaled_mxfp8(
        kv_params,
        input=a_q,
        input_row_offsets=input_row_offsets,
        wqkv=w_q,
        kv_collection=kv_collection,
        layer_idx=layer_idx,
        n_heads=num_heads,
        input_scale=a_scales,
        weight_scale=w_scales,
    )


def _run_path(
    *,
    is_mxfp8: bool,
    a_np: np.ndarray,
    wqkv_np: np.ndarray,
    seq_len: int,
    num_heads: int,
    kv_params: KVCacheParams,
    device: Accelerator,
    device_ref: DeviceRef,
    session: InferenceSession,
) -> tuple[np.ndarray, np.ndarray]:
    """Build, run one QKV path; return (Q output, KV cache blocks)."""
    hidden = a_np.shape[1]
    qkv_dim = wqkv_np.shape[0]
    kv_symbolic = kv_params.get_symbolic_inputs().inputs[0]

    with Graph(
        f"qkv_{'mxfp8' if is_mxfp8 else 'bf16'}",
        input_types=[
            TensorType(
                DType.bfloat16, shape=(seq_len, hidden), device=device_ref
            ),
            TensorType(DType.uint32, shape=(2,), device=device_ref),
            TensorType(
                DType.bfloat16, shape=(qkv_dim, hidden), device=device_ref
            ),
            *kv_symbolic.flatten(),
        ],
    ) as graph:
        layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())
        (
            a,
            input_row_offsets,
            wqkv,
            blocks,
            cache_lengths,
            lookup_table,
            max_prompt_length,
            max_cache_length,
            *_rest,
        ) = graph.inputs
        kv_collection = PagedCacheValues(
            blocks.buffer,
            cache_lengths.tensor,
            lookup_table.tensor,
            max_prompt_length.tensor,
            max_cache_length.tensor,
        )
        q_out = _build_qkv_value(
            is_mxfp8=is_mxfp8,
            a=a.tensor,
            wqkv=wqkv.tensor,
            input_row_offsets=input_row_offsets.tensor,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            kv_params=kv_params,
            num_heads=num_heads,
        )
        graph.output(q_out)

    model = session.load(graph)
    kv_runtime = _make_cache(kv_params, session, seq_len)

    a_buf = Buffer.from_dlpack(torch.from_numpy(a_np).to(torch.bfloat16)).to(
        device
    )
    wqkv_buf = Buffer.from_dlpack(
        torch.from_numpy(wqkv_np).to(torch.bfloat16)
    ).to(device)
    row_offsets_buf = Buffer.from_dlpack(
        torch.tensor([0, seq_len], dtype=torch.uint32)
    ).to(device)

    (out_buf,) = model.execute(
        a_buf, row_offsets_buf, wqkv_buf, *kv_runtime.flatten()
    )
    q_out_np = torch.from_dlpack(out_buf).to(torch.float32).cpu().numpy()
    # The cache is bf16, which numpy can't represent, so read it through torch.
    kv_blocks_np = (
        torch.from_dlpack(kv_runtime.kv_blocks).to(torch.float32).cpu().numpy()
    )
    return q_out_np, kv_blocks_np


# MiniMax-M3-shaped GQA with the head count scaled down to keep the test light.
# K (hidden) must stay a multiple of 128, the rank-5 SF K-group size.
@pytest.mark.parametrize(
    "label,seq_len,num_heads,num_kv_heads,head_dim,hidden",
    [
        ("prefill", 96, 16, 4, 128, 768),
        ("decode", 1, 16, 4, 128, 768),
    ],
)
def test_fused_qkv_mxfp8_matmul(
    label: str,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden: int,
) -> None:
    _skip_if_not_supported()

    qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim

    rng = np.random.default_rng(0)
    # Small-magnitude inputs keep values inside the E4M3 dynamic range so block
    # scaling is well-conditioned.
    a_np = (rng.standard_normal((seq_len, hidden)) * 0.1).astype(np.float32)
    wqkv_np = (rng.standard_normal((qkv_dim, hidden)) * 0.1).astype(np.float32)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        page_size=128,
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        devices=[device_ref],
    )

    q_mxfp8, kv_mxfp8 = _run_path(
        is_mxfp8=True,
        a_np=a_np,
        wqkv_np=wqkv_np,
        seq_len=seq_len,
        num_heads=num_heads,
        kv_params=kv_params,
        device=device,
        device_ref=device_ref,
        session=session,
    )
    _q_ref, kv_ref = _run_path(
        is_mxfp8=False,
        a_np=a_np,
        wqkv_np=wqkv_np,
        seq_len=seq_len,
        num_heads=num_heads,
        kv_params=kv_params,
        device=device,
        device_ref=device_ref,
        session=session,
    )

    q_dim = num_heads * head_dim
    q_host_ref = (a_np @ wqkv_np.T)[:, :q_dim]
    q_cos, q_rel = _cosine_and_rel_l2(q_mxfp8, q_host_ref)
    # Unwritten cache slots are zero in both buffers, so they do not distort
    # the cosine.
    kv_cos, kv_rel = _cosine_and_rel_l2(kv_mxfp8, kv_ref)

    print(
        f"\n=== fused_qkv_mxfp8 {label} "
        f"(S={seq_len}, H={num_heads}, KV={num_kv_heads}, D={head_dim}, "
        f"K={hidden}) ===\n"
        f"  Q   cosine / rel-L2 : {q_cos:.5f} / {q_rel:.5f}\n"
        f"  K/V cosine / rel-L2 : {kv_cos:.5f} / {kv_rel:.5f}",
        flush=True,
    )

    assert q_mxfp8.shape == (seq_len, q_dim)
    assert q_cos > 0.99, f"{label}: Q cosine {q_cos:.5f} too low"
    assert q_rel < 0.1, f"{label}: Q rel-L2 {q_rel:.5f} too high"
    assert np.any(kv_mxfp8 != 0.0), f"{label}: MXFP8 KV cache is all zeros"
    assert kv_cos > 0.99, f"{label}: K/V cosine {kv_cos:.5f} too low"
    assert kv_rel < 0.1, f"{label}: K/V rel-L2 {kv_rel:.5f} too high"
