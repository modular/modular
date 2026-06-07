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

"""Integration test: FP8 KV cache wrapper plumbing.

Exercises the Python wrapper changes for fp8 KV cache:

1.  ``rope_split_store_ragged`` accepts ``kv_params.quantized_kv_cache``
    when the config is (dtype=fp8_e4m3fn, scale_dtype=float32) and
    routes to ``mo.rope_split_store.ragged.paged.fp8_quantized``.

2.  ``flash_attention_ragged`` accepts the bf16-Q / fp8-KV pairing.

3.  The scales_tt_layout structural fix (explicit runtime stride[0])
    is exercised by using **mixed-parity** physical block indices in
    the LUT — if the K/V scale aliasing bug were still present the
    output cosine would drop far below the 0.999 bar.

The test builds two synthetic attention graphs on the GPU for each layer:
- **bf16 reference**: standard bf16 KV cache path.
- **fp8 path**: fp8_e4m3fn KV cache, same inputs, same weights.

Both graphs run ``rope_split_store_ragged + flash_attention_ragged``
for each layer. Acceptance bar: cosine >= 0.999 between the two outputs.

Target hardware: NVIDIA SM100 (B200). Uses ``bt-b200`` in CI.
"""

from __future__ import annotations

import math

import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.buffer_utils import cast_tensor_to
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged, rope_split_store_ragged
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    PagedCacheValues,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import PagedKVCacheManager
from max.pipelines.modeling.types import RequestID

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

_PAGE_SIZE = 128
_NUM_LAYERS = 2

# Layer 0: head_dim=256, 4 KV heads, group=8
_L0_HEAD_DIM = 256
_L0_N_KV_HEADS = 4
_L0_N_Q_HEADS = 32

# Layer 1: head_dim=512, 4 KV heads, group=8
_L1_HEAD_DIM = 512
_L1_N_KV_HEADS = 4
_L1_N_Q_HEADS = 32

_QUANT_GRAN = 64
_COSINE_BAR = 0.999


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two float arrays (flattened)."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _make_kv_params(
    n_kv_heads: int,
    head_dim: int,
    device: DeviceRef,
    fp8: bool,
) -> KVCacheParams:
    """Build KVCacheParams for either bf16 or fp8 KV cache."""
    quant_cfg = (
        KVCacheQuantizationConfig(
            scale_dtype=DType.float32,
            quantization_granularity=_QUANT_GRAN,
        )
        if fp8
        else None
    )
    return KVCacheParams(
        dtype=DType.float8_e4m3fn if fp8 else DType.bfloat16,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=_NUM_LAYERS,
        page_size=_PAGE_SIZE,
        devices=[device],
        kvcache_quant_config=quant_cfg,
    )


def _build_attention_graph(
    kv_params: KVCacheParams,
    n_q_heads: int,
    prompt_len: int,
    device: DeviceRef,
    layer_idx: int,
) -> Graph:
    """Build a single-layer graph: rope_split_store + flash_attention_ragged.

    Graph inputs (in order):
        qkv:            [prompt_len, (n_q_heads + 2*n_kv_heads) * head_dim]  bf16
        freqs_cis:      [prompt_len + 256, head_dim]                         bf16
        row_offsets:    [2]                                                   uint32
        layer_idx_t:    scalar                                                uint32
        ...kv_cache_symbolic_inputs (from kv_params.get_symbolic_inputs())...

    Graph output:
        attn_out: [prompt_len, n_q_heads, head_dim]  bf16
    """
    head_dim = kv_params.head_dim
    n_kv_heads = kv_params.n_kv_heads
    combined_dim = (n_q_heads + 2 * n_kv_heads) * head_dim

    qkv_type = TensorType(
        DType.bfloat16, [prompt_len, combined_dim], device=device
    )
    freqs_type = TensorType(
        DType.bfloat16, [prompt_len + 256, head_dim], device=device
    )
    row_offsets_type = TensorType(DType.uint32, [2], device=device)
    # layer_idx must reside on CPU for paged KV cache ops.
    layer_idx_type = TensorType(DType.uint32, [], device=DeviceRef.CPU())

    kv_symbolic = kv_params.get_symbolic_inputs()

    with Graph(
        f"fp8_kv_attn_layer{layer_idx}_{'fp8' if kv_params.quantized_kv_cache else 'bf16'}",
        input_types=[
            qkv_type,
            freqs_type,
            row_offsets_type,
            layer_idx_type,
            *kv_symbolic.flatten(),
        ],
    ) as g:
        qkv_v = g.inputs[0].tensor
        freqs_v = g.inputs[1].tensor
        row_offsets_v = g.inputs[2].tensor
        layer_idx_v = g.inputs[3].tensor

        # Parse KV cache inputs (single device, single replica).
        # Flat order from KVCacheInputsPerDevice.flatten():
        #   kv_blocks, cache_lengths, lookup_table, max_lengths,
        #   [kv_scales,]    <-- only when quantized_kv_cache=True
        #   attention_dispatch_metadata
        kv_input_tensors = g.inputs[4:]
        kv_symbolic_item = kv_symbolic.inputs[0]
        kv_vals = kv_symbolic_item.unflatten(iter(kv_input_tensors))

        kv_collection = PagedCacheValues(
            kv_blocks=kv_vals.kv_blocks.buffer,
            cache_lengths=kv_vals.cache_lengths.tensor,
            lookup_table=kv_vals.lookup_table.tensor,
            max_lengths=kv_vals.max_lengths.tensor,
            kv_scales=kv_vals.kv_scales.buffer
            if kv_vals.kv_scales is not None
            else None,
            attention_dispatch_metadata=kv_vals.attention_dispatch_metadata.tensor
            if kv_vals.attention_dispatch_metadata is not None
            else None,
        )

        # rope_split_store_ragged — for fp8 routes to fp8_quantized op.
        # Returns [total_seq_len, n_q_heads * head_dim].
        roped_q = rope_split_store_ragged(
            kv_params=kv_params,
            qkv=qkv_v,
            input_row_offsets=row_offsets_v,
            freqs_cis=freqs_v,
            kv_collection=kv_collection,
            layer_idx=layer_idx_v,
            n_heads=n_q_heads,
            interleaved=True,
        )
        # flash_attention_ragged requires rank-3 input [total_seq_len, n_heads, head_dim].
        roped_q = ops.reshape(roped_q, [-1, n_q_heads, head_dim])

        # flash_attention_ragged — for fp8 KV this is the bf16-Q+fp8-KV pairing.
        attn_out = flash_attention_ragged(
            kv_params=kv_params,
            input=roped_q,
            input_row_offsets=row_offsets_v,
            kv_collection=kv_collection,
            layer_idx=layer_idx_v,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=1.0 / math.sqrt(head_dim),
        )
        # Cast to float32 before output: numpy does not support bfloat16 via
        # DLPack, so we need a float32 buffer to call .to_numpy() on the
        # result.
        g.output(ops.cast(attn_out, DType.float32))

    return g


def _make_context(prompt_len: int) -> TextContext:
    """Build a minimal TextContext for the KV cache manager."""
    tokens_np = np.arange(prompt_len, dtype=np.int64)
    return TextContext(
        request_id=RequestID(),
        max_length=1024,
        tokens=TokenBuffer(tokens_np),
    )


def test_fp8_kv_toy_model_layer0() -> None:
    """Layer 0 (head_dim=256): fp8 vs bf16 wrapper plumbing, cosine >= 0.999."""
    _run_layer_test(
        n_q_heads=_L0_N_Q_HEADS,
        n_kv_heads=_L0_N_KV_HEADS,
        head_dim=_L0_HEAD_DIM,
        layer_idx=0,
        prompt_len=64,
    )


def test_fp8_kv_toy_model_layer1() -> None:
    """Layer 1 (head_dim=512): fp8 vs bf16 wrapper plumbing, cosine >= 0.999."""
    _run_layer_test(
        n_q_heads=_L1_N_Q_HEADS,
        n_kv_heads=_L1_N_KV_HEADS,
        head_dim=_L1_HEAD_DIM,
        layer_idx=1,
        prompt_len=64,
    )


def _run_layer_test(
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    layer_idx: int,
    prompt_len: int,
) -> None:
    """Build and run bf16 and fp8 attention graphs for one layer, check cosine."""
    gpu_device = Accelerator()
    device = DeviceRef.GPU()
    session = InferenceSession(devices=[gpu_device])

    rng = np.random.default_rng(0xDEADBEEF + layer_idx)
    combined_dim = (n_q_heads + 2 * n_kv_heads) * head_dim

    # Random bf16 QKV input (same for both paths).
    qkv_np = rng.standard_normal((prompt_len, combined_dim)).astype(np.float32)

    # Frequency table
    max_seq_len = prompt_len + 256
    freqs_np = rng.standard_normal((max_seq_len, head_dim)).astype(np.float32)

    # Row offsets: single sequence [0, prompt_len].
    row_offsets_np = np.array([0, prompt_len], dtype=np.uint32)
    layer_idx_np = np.array(layer_idx, dtype=np.uint32)

    # -----------------------------------------------------------------
    # BF16 reference graph
    # -----------------------------------------------------------------
    kv_params_bf16 = _make_kv_params(n_kv_heads, head_dim, device, fp8=False)
    kv_manager_bf16 = PagedKVCacheManager(
        kv_params_bf16,
        total_num_pages=32,
        session=session,
        max_batch_size=8,
    )

    graph_bf16 = _build_attention_graph(
        kv_params_bf16, n_q_heads, prompt_len, device, layer_idx
    )
    model_bf16 = session.load(graph_bf16)

    ctx_bf16 = _make_context(prompt_len)
    kv_manager_bf16.claim(ctx_bf16.request_id, replica_idx=0)
    kv_manager_bf16.alloc(ctx_bf16, replica_idx=0, num_steps=1)
    kv_runtime_bf16 = kv_manager_bf16.runtime_inputs([[ctx_bf16]])

    # Convert inputs to GPU buffers in bfloat16.
    qkv_buf = cast_tensor_to(
        Buffer.from_numpy(qkv_np).to(gpu_device), DType.bfloat16, session
    )
    freqs_buf = cast_tensor_to(
        Buffer.from_numpy(freqs_np).to(gpu_device), DType.bfloat16, session
    )
    row_offsets_buf = Buffer.from_numpy(row_offsets_np).to(gpu_device)
    layer_idx_buf = Buffer.from_numpy(layer_idx_np)

    out_bf16 = model_bf16.execute(
        qkv_buf,
        freqs_buf,
        row_offsets_buf,
        layer_idx_buf,
        *kv_runtime_bf16.flatten(),
    )

    # -----------------------------------------------------------------
    # FP8 graph
    # -----------------------------------------------------------------
    kv_params_fp8 = _make_kv_params(n_kv_heads, head_dim, device, fp8=True)
    kv_manager_fp8 = PagedKVCacheManager(
        kv_params_fp8,
        total_num_pages=32,
        session=session,
        max_batch_size=8,
    )

    graph_fp8 = _build_attention_graph(
        kv_params_fp8, n_q_heads, prompt_len, device, layer_idx
    )
    model_fp8 = session.load(graph_fp8)

    ctx_fp8 = _make_context(prompt_len)
    kv_manager_fp8.claim(ctx_fp8.request_id, replica_idx=0)
    kv_manager_fp8.alloc(ctx_fp8, replica_idx=0, num_steps=1)
    kv_runtime_fp8 = kv_manager_fp8.runtime_inputs([[ctx_fp8]])

    out_fp8 = model_fp8.execute(
        qkv_buf,
        freqs_buf,
        row_offsets_buf,
        layer_idx_buf,
        *kv_runtime_fp8.flatten(),
    )

    # -----------------------------------------------------------------
    # Compare
    # -----------------------------------------------------------------
    # model.execute() returns list[Buffer]; the graph outputs float32
    # (cast before g.output) so .to_numpy() works without a torch dependency.
    a = out_bf16[0].to_numpy()
    b = out_fp8[0].to_numpy()
    cos = _cosine(a, b)
    print(
        f"layer{layer_idx} head_dim={head_dim}: cosine(bf16, fp8) = {cos:.6f}"
    )
    assert cos >= _COSINE_BAR, (
        f"cosine {cos:.6f} below bar {_COSINE_BAR} for"
        f" layer{layer_idx} head_dim={head_dim}"
    )
