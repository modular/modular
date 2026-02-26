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

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.kv_cache import PagedKVCacheManager
from max.nn.attention.attention_with_rope import (
    AttentionWithRope,
    AttentionWithRopeNoOpaque,
    Module,
    PagedKVCacheTensorsNoOpaque,
)
from max.nn.kv_cache import (
    KVCacheParams,
    NestedIterableDataclass,
    PagedCacheValues,
)
from max.nn.rotary_embedding import RotaryEmbedding
from test_common.context_utils import create_text_context

AttentionFn = Callable[
    [
        TensorValue,
        TensorValue,
        BufferValue,
        TensorValue,
        TensorValue,
        TensorValue,
        TensorValue,
    ],
    TensorValue,
]


def build_and_execute_graph(
    session: InferenceSession,
    input: npt.NDArray[np.floating[Any]],
    input_row_offsets: npt.NDArray[np.integer[Any]],
    device: Device,
    attention_fn: AttentionFn,
    model: Module,
    kv_inputs: Sequence[Buffer],
    kv_input_symbols: NestedIterableDataclass,
) -> npt.NDArray[np.floating[Any]]:
    device_ref = DeviceRef.from_device(device)
    hidden_state_type = TensorType(
        DType.float32,
        [Dim("total_seq_len"), input.shape[-1]],
        device=device_ref,
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        ["input_row_offsets_len"],
        device=device_ref,
    )

    with Graph(
        "reference_graph",
        input_types=[
            hidden_state_type,
            input_row_offsets_type,
            *kv_input_symbols,
        ],
    ) as g:
        (
            hidden_state,
            input_row_offsets_symbol,
            blocks,
            cache_lengths,
            lookup_table,
            max_lengths,
        ) = g.inputs
        layer_idx = ops.constant(0, dtype=DType.uint32, device=device_ref)

        output = attention_fn(
            hidden_state.tensor,
            input_row_offsets_symbol.tensor,
            blocks.buffer,
            cache_lengths.tensor,
            lookup_table.tensor,
            max_lengths.tensor,
            layer_idx.tensor,
        )

        g.output(output)

    compiled_model = session.load(g, weights_registry=model.state_dict())

    return cast(
        Buffer, compiled_model(input, input_row_offsets, *kv_inputs)[0]
    ).to_numpy()


def test_compare_attention_with_rope_no_opaque() -> None:
    dim = 1024
    n_heads = 16
    n_kv_heads = 16
    head_dim = dim // n_heads
    max_seq_len = 1024
    device = CPU()
    page_size = 128
    max_batch_size = 8
    batch_size = 2
    session = InferenceSession(devices=[device])
    kv_params = KVCacheParams(
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=DType.float32,
        num_layers=1,
        page_size=page_size,
        cache_strategy="paged",
        devices=[DeviceRef.from_device(device)],
    )

    rope_ref = RotaryEmbedding(dim, n_heads, 10000.0, max_seq_len)
    rope_no_opaque = RotaryEmbedding(dim, n_heads, 10000.0, max_seq_len)

    no_opaque_attention = AttentionWithRopeNoOpaque(
        rope=rope_no_opaque,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        hidden_size=dim,
        kv_params=kv_params,
        scale=0.125,
    )
    reference_attention = AttentionWithRope(
        rope=rope_ref,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        hidden_size=dim,
        kv_params=kv_params,
        stacked_qkv=True,
        has_bias=False,
        scale=0.125,
    )

    qkv_shape = tuple(
        int(dim) for dim in no_opaque_attention.qkv_proj.weight.shape
    )
    o_shape = tuple(int(dim) for dim in no_opaque_attention.o_proj.weight.shape)

    attention_weights = {
        "qkv_proj.weight": torch.randn(qkv_shape),
        "o_proj.weight": torch.randn(o_shape),
    }

    no_opaque_attention.load_state_dict(attention_weights)
    reference_attention.load_state_dict(attention_weights)

    kv_manager = PagedKVCacheManager(
        kv_params,
        total_num_pages=32,
        session=session,
        max_batch_size=128,
    )

    # Create contexts and claim seq_ids in cache.
    batch = []
    for _ in range(batch_size):
        context = create_text_context(
            np.empty(max_seq_len), max_length=max_seq_len
        )
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(context, replica_idx=0, num_steps=1)
        batch.append(context)

    kv_inputs = list(kv_manager.runtime_inputs([batch])[0])
    kv_input_symbols = kv_params.get_symbolic_inputs()[0]

    def reference_attention_fn(
        hidden_state: TensorValue,
        input_row_offsets: TensorValue,
        blocks: BufferValue,
        cache_lengths: TensorValue,
        lookup_table: TensorValue,
        max_lengths: TensorValue,
        layer_idx: TensorValue,
    ) -> TensorValue:
        kv_collection = PagedCacheValues(
            blocks.buffer,
            cache_lengths.tensor,
            lookup_table.tensor,
            max_lengths.tensor,
        )

        return reference_attention(
            layer_idx,
            hidden_state.tensor,
            kv_collection,
            reference_attention.rope.freqs_cis,
            input_row_offsets.tensor,
        )

    def no_opaque_attention_fn(
        hidden_state: TensorValue,
        input_row_offsets: TensorValue,
        blocks: BufferValue,
        cache_lengths: TensorValue,
        lookup_table: TensorValue,
        max_lengths: TensorValue,
        layer_idx: TensorValue,
    ) -> TensorValue:
        kv_cache_tensors = PagedKVCacheTensorsNoOpaque(
            blocks=blocks,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_lengths=max_lengths,
        )

        return no_opaque_attention(
            layer_idx,
            hidden_state.tensor,
            kv_cache_tensors,
            input_row_offsets.tensor,
        )

    hidden_state = np.random.randn(max_seq_len * batch_size, dim).astype(
        np.float32
    )
    input_row_offsets = np.array(
        [0, max_seq_len, max_seq_len * 2], dtype=np.uint32
    )

    reference_output = build_and_execute_graph(
        session,
        hidden_state,
        input_row_offsets,
        device,
        reference_attention_fn,
        kv_inputs=kv_inputs,
        model=reference_attention,
        kv_input_symbols=kv_input_symbols,
    )
    with pytest.raises(NotImplementedError) as e:
        _no_opaque_output = build_and_execute_graph(
            session,
            hidden_state,
            input_row_offsets,
            device,
            no_opaque_attention_fn,
            kv_inputs=kv_inputs,
            model=no_opaque_attention,
            kv_input_symbols=kv_input_symbols,
        )
        assert "flash_attention_ragged_no_opaque not implemented" in str(
            e.value
        )
    # assert torch.allclose(no_opaque_output, reference_output)
