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
"""The quantized dense MLA op compiles with the split record as the last operand.

No Python builder emits ``attention.mla.graph.prefill.decode.paged.quantized``,
so no graph in the tree compiles its registration; the test builds one graph
with the operand list the registration declares and compiles it for a virtual
B200, as a CPU worker precompiling a B200 graph does. Virtual-device mode
latches process-wide, so this file needs a target of its own.
"""

from __future__ import annotations

from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)

set_virtual_device_api("cuda")
set_virtual_device_target_arch("sm_100a")
set_virtual_device_count(1)

from max import tree
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.attention_mask import AttentionMaskVariant
from max.nn.kv_cache import MIXED_STEP_SPLIT_SIZE, MLAKVCacheParams
from max.nn.legacy.attention.multi_latent_attention import (
    LatentAttentionWithRope,
)
from max.nn.legacy.rotary_embedding import DeepseekYarnRotaryEmbedding
from max.nn.rope_scaling import DeepseekYarnRopeScalingParams

OP_NAME = "attention.mla.graph.prefill.decode.paged.quantized"
NUM_HEADS = 8
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
KV_LORA_RANK = 512
HIDDEN_SIZE = 1024
Q_LORA_RANK = 256
BUFFER_SIZE = 4096
PAGE_SIZE = 128


def _layer() -> LatentAttentionWithRope:
    """Builds the MLA layer whose weights and cache the op takes."""
    rope = DeepseekYarnRotaryEmbedding(
        dim=QK_ROPE_HEAD_DIM,
        n_heads=NUM_HEADS,
        theta=10000.0,
        max_seq_len=2048,
        scaling_params=DeepseekYarnRopeScalingParams(
            scaling_factor=40.0,
            original_max_position_embeddings=4096,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
            mscale_all_dim=1.0,
        ),
    )
    kv_params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=NUM_HEADS,
    )
    return LatentAttentionWithRope(
        rope=rope,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=1,
        hidden_size=HIDDEN_SIZE,
        kv_params=kv_params,
        dtype=DType.bfloat16,
        devices=[DeviceRef.GPU()],
        graph_mode="auto",
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        buffer_size=BUFFER_SIZE,
    )


def _quantized_op_graph() -> Graph:
    """Builds the op's declared operand list, the split record last."""
    attn = _layer()
    kv_params = attn.kv_params
    gpu = DeviceRef.GPU()
    cache_types = tree.leaves(kv_params.get_symbolic_inputs()[0])
    input_types = [
        TensorType(
            DType.bfloat16,
            ["total_tokens", NUM_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM],
            gpu,
        ),
        TensorType(
            DType.bfloat16,
            ["total_tokens", KV_LORA_RANK + QK_ROPE_HEAD_DIM],
            gpu,
        ),
        TensorType(DType.uint32, ["row_offsets_len"], gpu),
        *cache_types,
        # The scales page the op declares; the op does not read it.
        BufferType(DType.float8_e4m3fn, [1, 1, 1, 1, 1, 1], gpu),
        TensorType(DType.uint32, [MIXED_STEP_SPLIT_SIZE], gpu),
    ]
    with Graph(OP_NAME, input_types=input_types) as graph:
        q, kv, input_row_offsets = (x.tensor for x in graph.inputs[:3])
        n = len(cache_types)
        kv_collection = kv_params.unflatten_kv_inputs(
            iter(graph.inputs[3 : 3 + n])
        )[0]
        kv_scales = graph.inputs[3 + n].buffer
        record = graph.inputs[4 + n].tensor
        assert kv_collection.attention_dispatch_metadata is not None
        assert kv_collection.mla_num_partitions is not None
        metadata = attn.create_mla_prefill_metadata(
            input_row_offsets, kv_collection
        )
        output = ops.inplace_custom(
            OP_NAME,
            device=gpu,
            values=[
                q,
                kv,
                input_row_offsets,
                ops.cast(attn.rope.freqs_cis, q.dtype).to(q.device),
                attn.kv_a_proj_layernorm,
                metadata.buffer_row_offsets[0],
                metadata.cache_offsets[0],
                metadata.buffer_lengths.to(DeviceRef.CPU())[0],
                attn.w_k,
                attn.w_uk,
                attn.w_uv,
                *kv_collection.flatten_without_attention_dispatch_metadata(),
                kv_scales,
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                ops.constant(attn.scale, DType.float32, device=DeviceRef.CPU()),
                ops.constant(1e-6, DType.float32, device=DeviceRef.CPU()),
                kv_collection.attention_dispatch_metadata,
                kv_collection.mla_num_partitions,
                record,
            ],
            out_types=[
                TensorType(
                    DType.bfloat16, ["total_tokens", NUM_HEADS, V_HEAD_DIM], gpu
                )
            ],
            parameters={"mask_str": AttentionMaskVariant.CAUSAL_MASK.value},
        )[0].tensor
        graph.output(output)
    return graph


def test_the_quantized_op_compiles_with_the_record() -> None:
    graph = _quantized_op_graph()
    InferenceSession(devices=[Accelerator()]).compile(graph)
