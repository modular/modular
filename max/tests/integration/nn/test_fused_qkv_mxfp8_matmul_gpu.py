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
    _fused_qkv_index_ragged_matmul_scaled_mxfp8,
    _fused_qkv_ragged_matmul_scaled_mxfp8,
    fused_qkv_ragged_matmul,
    quantize_dynamic_block_scaled,
)
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParams,
    MHAKVCacheParams,
    MLAKVCacheParams,
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


def test_fused_qkv_mxfp8_matmul_fp8_cache() -> None:
    """The MXFP8 fused-QKV store writes into a scale-free FP8 cache.

    This is the dense-layer write op for a block-sparse MSA FP8 KV. Dense MXFP8 layers
    route through ``quantized_fused_qkv_matmul`` ->
    ``mo.fused_qkv_matmul.ragged.paged.scale.mxfp8``. The GEMM outputs the BF16
    Q scratch and the epilogue saturating-casts K/V into the FP8 cache (the
    kernel passes ``output_dtype`` (bf16) to the GEMM).
    Confirms the stored K/V match a BF16-cache run within FP8 rounding.
    """
    _skip_if_not_supported()

    seq_len, num_heads, num_kv_heads, head_dim, hidden = 96, 16, 4, 128, 768
    qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim

    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal((seq_len, hidden)) * 0.1).astype(np.float32)
    wqkv_np = (rng.standard_normal((qkv_dim, hidden)) * 0.1).astype(np.float32)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])

    def _params(dtype: DType) -> MHAKVCacheParams:
        return MHAKVCacheParams(
            dtype=dtype,
            page_size=128,
            n_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=1,
            devices=[device_ref],
        )

    fp8_params = _params(DType.float8_e4m3fn)
    # Scale-free FP8: the MSA/M3 config never attaches a kvcache_quant_config.
    assert fp8_params.is_fp8_kv_dtype
    assert not fp8_params.quantized_kv_cache

    def _kv(kv_params: MHAKVCacheParams) -> np.ndarray:
        return _run_path(
            is_mxfp8=True,
            a_np=a_np,
            wqkv_np=wqkv_np,
            seq_len=seq_len,
            num_heads=num_heads,
            kv_params=kv_params,
            device=device,
            device_ref=device_ref,
            session=session,
        )[1]

    kv_fp8 = _kv(fp8_params)
    kv_bf16 = _kv(_params(DType.bfloat16))

    kv_cos, kv_rel = _cosine_and_rel_l2(kv_fp8, kv_bf16)
    print(
        f"\n=== fused_qkv_mxfp8 fp8-cache store ===\n"
        f"  K/V cosine / rel-L2 : {kv_cos:.5f} / {kv_rel:.5f}",
        flush=True,
    )

    assert np.all(np.isfinite(kv_fp8)), "FP8 KV cache has non-finite values"
    assert np.any(kv_fp8 != 0.0), "FP8 KV cache is all zeros"
    assert kv_cos > 0.99, f"FP8 vs BF16 K/V cosine {kv_cos:.5f} too low"


def test_fused_qkv_index_mxfp8_matmul_fp8_main_cache() -> None:
    """The 5-way sparse (QKV + index-QK) store into an FP8 main cache.

    Exercises the dual-cache fused op the sparse block-sparse layers use. The
    GEMM outputs the BF16 Q/IndexQ scratch and the epilogue saturating-casts
    K/V into the main cache (index cache stays BF16). Runs the op
    with a BF16 and an FP8 main cache and asserts the FP8 K/V match the BF16
    reference within FP8 tolerance, while the Q/IndexQ output (unaffected by the
    cache dtype) is bit-identical.
    """
    _skip_if_not_supported()

    seq_len = 96
    n_heads, n_kv_heads, head_dim = 16, 4, 128
    # num_index_heads must be an MLA-dispatch-supported value (8/16/32/64/128);
    # this store-only test doesn't run MLA decode, but the index cache's
    # runtime inputs bind dispatch metadata eagerly.
    num_index_heads, idx_head_dim = 16, 128
    hidden = 768
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    iq_dim = num_index_heads * idx_head_dim
    ik_dim = idx_head_dim
    n_total = q_dim + 2 * kv_dim + iq_dim + ik_dim

    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal((seq_len, hidden)) * 0.1).astype(np.float32)
    wqkv_np = (rng.standard_normal((n_total, hidden)) * 0.1).astype(np.float32)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])

    def _run(
        main_dtype: DType,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the 5-way op with a main cache of ``main_dtype`` (index BF16).

        Returns ``(Q, IndexQ, main-cache K/V blocks)`` as fp32.
        """
        main_params = MHAKVCacheParams(
            dtype=main_dtype,
            page_size=128,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_layers=1,
            devices=[device_ref],
        )
        index_params = MLAKVCacheParams(
            dtype=DType.bfloat16,
            page_size=128,
            head_dim=idx_head_dim,
            num_layers=1,
            devices=[device_ref],
            num_q_heads=num_index_heads,
        )

        main_sym = main_params.get_symbolic_inputs().inputs[0]
        index_sym = index_params.get_symbolic_inputs().inputs[0]
        n_main = len(main_sym.flatten())

        with Graph(
            f"qkv_index_mxfp8_{main_dtype}_main_cache",
            input_types=[
                TensorType(
                    DType.bfloat16, (seq_len, hidden), device=device_ref
                ),
                TensorType(DType.uint32, (2,), device=device_ref),
                TensorType(
                    DType.bfloat16, (n_total, hidden), device=device_ref
                ),
                *main_sym.flatten(),
                *index_sym.flatten(),
            ],
        ) as graph:
            a, iro, wqkv, *rest = graph.inputs
            main_in, index_in = rest[:n_main], rest[n_main:]
            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())
            a_q, a_scales = quantize_dynamic_block_scaled(
                a.tensor,
                sf_vector_size=32,
                scales_type=DType.float8_e8m0fnu,
                out_type=DType.float8_e4m3fn,
            )
            w_q, w_scales = quantize_dynamic_block_scaled(
                wqkv.tensor,
                sf_vector_size=32,
                scales_type=DType.float8_e8m0fnu,
                out_type=DType.float8_e4m3fn,
            )
            main_kv = PagedCacheValues(
                main_in[0].buffer,
                main_in[1].tensor,
                main_in[2].tensor,
                main_in[3].tensor,
                main_in[4].tensor,
            )
            index_kv = PagedCacheValues(
                index_in[0].buffer,
                index_in[1].tensor,
                index_in[2].tensor,
                index_in[3].tensor,
                index_in[4].tensor,
            )
            q, index_q = _fused_qkv_index_ragged_matmul_scaled_mxfp8(
                main_params,
                index_params,
                input=a_q,
                input_row_offsets=iro.tensor,
                wqkv=w_q,
                kv_collection=main_kv,
                index_kv_collection=index_kv,
                layer_idx=layer_idx,
                n_heads=n_heads,
                num_index_heads=num_index_heads,
                idx_head_dim=idx_head_dim,
                input_scale=a_scales,
                weight_scale=w_scales,
            )
            graph.output(q, index_q)

        model = session.load(graph)
        main_rt = _make_cache(main_params, session, seq_len)
        index_rt = _make_cache(index_params, session, seq_len)

        a_buf = Buffer.from_dlpack(
            torch.from_numpy(a_np).to(torch.bfloat16)
        ).to(device)
        wqkv_buf = Buffer.from_dlpack(
            torch.from_numpy(wqkv_np).to(torch.bfloat16)
        ).to(device)
        iro_buf = Buffer.from_dlpack(
            torch.tensor([0, seq_len], dtype=torch.uint32)
        ).to(device)

        q_buf, iq_buf = model.execute(
            a_buf, iro_buf, wqkv_buf, *main_rt.flatten(), *index_rt.flatten()
        )
        q_np = torch.from_dlpack(q_buf).to(torch.float32).cpu().numpy()
        iq_np = torch.from_dlpack(iq_buf).to(torch.float32).cpu().numpy()
        main_kv_np = (
            torch.from_dlpack(main_rt.kv_blocks).to(torch.float32).cpu().numpy()
        )
        return q_np, iq_np, main_kv_np

    q_fp8, iq_fp8, kv_fp8 = _run(DType.float8_e4m3fn)
    q_bf16, iq_bf16, kv_bf16 = _run(DType.bfloat16)

    kv_cos, kv_rel = _cosine_and_rel_l2(kv_fp8, kv_bf16)
    print(
        f"\n=== fused_qkv_index_mxfp8 fp8-main-cache store ===\n"
        f"  K/V cosine / rel-L2 : {kv_cos:.5f} / {kv_rel:.5f}",
        flush=True,
    )

    assert np.all(np.isfinite(kv_fp8)), "FP8 main cache has non-finite values"
    assert np.any(kv_fp8 != 0.0), "FP8 main cache was not written"
    # Q and IndexQ come from the BF16 scratch, unaffected by the cache dtype.
    np.testing.assert_array_equal(q_fp8, q_bf16)
    np.testing.assert_array_equal(iq_fp8, iq_bf16)
    # K/V differ only by the FP8 store rounding.
    assert kv_cos > 0.99, f"FP8 vs BF16 K/V cosine {kv_cos:.5f} too low"
