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
"""Test pipelines MLA attention layer."""

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kernels import flare_mla_prefill_ragged
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from test_common.context_utils import create_text_context


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="MLA kernel only supports Nvidia GPUs"
)
def test_kv_cache_paged_mla_prefill(gpu_session: InferenceSession) -> None:
    cuda = Accelerator()
    session = gpu_session
    num_q_heads = 32
    q_head_dim = 192
    k_head_dim = 128
    num_layers = 1
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=576,
        num_layers=num_layers,
        cache_strategy="paged",
        page_size=128,
        devices=[DeviceRef.GPU()],
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)
    input_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", num_q_heads, q_head_dim],
        DeviceRef.GPU(),
    )
    k_buffer_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", num_q_heads, k_head_dim],
        DeviceRef.GPU(),
    )
    v_buffer_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", num_q_heads, k_head_dim],
        DeviceRef.GPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], DeviceRef.GPU()
    )
    kv_manager = PagedKVCacheManager(
        kv_params,
        total_num_pages=8,
        session=session,
    )

    blocks_type, cache_lengths_type, lookup_table_type, is_cache_empty_type = (
        kv_params.get_symbolic_inputs()[0]
    )

    def construct() -> Graph:
        with Graph(
            "call_mla_prefill",
            input_types=[
                input_type,
                input_row_offsets_type,
                k_buffer_type,
                v_buffer_type,
                blocks_type,
                cache_lengths_type,
                lookup_table_type,
                is_cache_empty_type,
            ],
        ) as g:
            (
                input,
                input_row_offsets,
                k_buffer,
                v_buffer,
                blocks,
                cache_lengths,
                lookup_table,
                is_cache_empty,
            ) = g.inputs

            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

            kv_collection = PagedCacheValues(
                blocks.buffer,
                cache_lengths.tensor,
                lookup_table.tensor,
                is_cache_empty.tensor,
            )
            result = flare_mla_prefill_ragged(
                kv_params,
                input.tensor,
                k_buffer.tensor,
                v_buffer.tensor,
                input_row_offsets.tensor,
                input_row_offsets.tensor,  # actually buffer_row_offsets
                cache_lengths.tensor,
                kv_collection,
                layer_idx,
                MHAMaskVariant.CAUSAL_MASK,
                1,  # scale
            )
            g.output(result.cast(DType.float32))
        return g

    g = construct()
    # Create contexts
    batch = []
    for i in range(batch_size):
        context = create_text_context(np.empty(prompt_lens[i]))
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(context, replica_idx=0, num_steps=1)
        batch.append(context)

    input_row_offsets = Buffer(
        DType.uint32,
        [batch_size + 1],
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum
    input_row_offsets = input_row_offsets.to(cuda)

    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.get_runtime_inputs([batch])[0]
    )
    model = session.load(g)

    input_tensor = Buffer.zeros(
        (total_seq_len, num_q_heads, q_head_dim), dtype=DType.bfloat16
    )
    k_buffer_tensor = Buffer.zeros(
        (total_seq_len, num_q_heads, k_head_dim), dtype=DType.bfloat16
    )
    v_buffer_tensor = Buffer.zeros(
        (total_seq_len, num_q_heads, k_head_dim), dtype=DType.bfloat16
    )

    result = model.execute(
        input_tensor.to(cuda),
        input_row_offsets.to(cuda),
        k_buffer_tensor.to(cuda),
        v_buffer_tensor.to(cuda),
        blocks.to(cuda),
        cache_lengths.to(cuda),
        lookup_table_tensor.to(cuda),
        is_cache_empty_buf,
    )[0]
    assert isinstance(result, Buffer)

    host = CPU(0)
    assert np.all(np.isfinite(result.to(host).to_numpy()))
