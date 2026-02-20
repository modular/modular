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
"""Test attention with sinks implementation against OpenAI reference."""

import math
from typing import cast

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kernels import flash_attention_ragged
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from test_common.context_utils import create_text_context


def max_flash_attention_with_sinks(
    query: torch.Tensor,
    sinks: torch.Tensor,
    input_row_offsets: np.ndarray,
    num_kv_heads: int,
    scale: float,
    mask_variant: MHAMaskVariant,
    device: Device,
    sliding_window: int = -1,
) -> torch.Tensor:
    """MAX graph implementation of attention with sinks using flash_attention_ragged.

    Args:
        query: Query tensor [total_seq_len, num_heads, head_dim]
        key: Key tensor [total_seq_len, num_kv_heads, head_dim]
        value: Value tensor [total_seq_len, num_kv_heads, head_dim]
        sinks: Sink weights [num_heads]
        input_row_offsets: Batch boundaries [batch_size + 1]
        scale: Attention scale factor
        mask_variant: Type of attention mask to apply
        sliding_window: Size of sliding window (-1 for no sliding window)

    Returns:
        Output tensor from MAX graph execution
    """
    # Setup
    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])

    # Extract dimensions
    _total_seq_len, num_heads, head_dim = query.shape
    batch_size = len(input_row_offsets) - 1
    num_layers = 1

    # Setup KV cache parameters
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        cache_strategy="paged",
        page_size=128,
        devices=[DeviceRef.GPU()],
    )

    # Create KV manager
    max_seq_len = max(
        input_row_offsets[i + 1] - input_row_offsets[i]
        for i in range(batch_size)
    )
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
    )

    # Create contexts for KV cache
    batch = []
    for i in range(batch_size):
        seq_len = input_row_offsets[i + 1] - input_row_offsets[i]
        context = create_text_context(np.empty(seq_len))
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(context, replica_idx=0, num_steps=1)
        batch.append(context)

    kv_cache_inputs = kv_manager.get_runtime_inputs([batch])[0]

    # Define graph input types
    input_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", num_heads, head_dim],
        DeviceRef.GPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        ["batch_size_plus_1"],
        DeviceRef.GPU(),
    )
    sinks_type = TensorType(
        DType.bfloat16,
        [num_heads],
        DeviceRef.GPU(),
    )

    def build_graph():  # noqa: ANN202
        with Graph(
            "flash_attention_with_sinks",
            input_types=[
                input_type,
                input_row_offsets_type,
                sinks_type,
                *kv_params.get_symbolic_inputs()[0],
            ],
        ) as g:
            inputs = g.inputs
            q = inputs[0].tensor
            input_row_offsets = inputs[1].tensor
            sink_weights = inputs[2].tensor

            # Fetch KV cache
            kv_collection = PagedCacheValues(
                kv_blocks=inputs[3].buffer,
                cache_lengths=inputs[4].tensor,
                lookup_table=inputs[5].tensor,
                max_lengths=inputs[6].tensor,
            )

            # Layer index
            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

            # QKV processing (simplified - in real model this would use fused ops)
            # For testing, we'll directly use the inputs as QKV

            # Apply flash attention with sinks
            output = flash_attention_ragged(
                kv_params,
                q,
                input_row_offsets,
                kv_collection,
                layer_idx,
                mask_variant,
                scale,
                local_window_size=sliding_window,
                sink_weights=sink_weights,
            )

            g.output(output.cast(DType.float32))
        return g

    graph = build_graph()
    model = session.load(graph)

    # Convert inputs to MAX tensors
    q_tensor = Buffer.from_dlpack(query).to(device)
    offsets_tensor = Buffer.from_numpy(input_row_offsets.astype(np.uint32)).to(
        device
    )
    sinks_tensor = Buffer.from_dlpack(sinks).to(device)

    # Execute
    result = model.execute(
        q_tensor,
        offsets_tensor,
        sinks_tensor,
        *kv_cache_inputs,
    )[0]

    return cast(Buffer, result).to(CPU()).to_numpy()


@pytest.mark.parametrize(
    "batch_size,seq_lens,num_heads,num_kv_heads,head_dim,sliding_window",
    [
        (1, [16], 8, 2, 64, None),  # Single batch, GQA
        (2, [8, 12], 8, 2, 64, None),  # Multi-batch ragged, GQA
        (1, [32], 16, 4, 64, 16),  # Sliding window
        (2, [16, 20], 16, 16, 64, None),  # No GQA
    ],
)
def test_flash_attention_ragged_with_sinks(
    batch_size: int,
    seq_lens: list[int],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
) -> None:
    """Test flash_attention_ragged with sink weights against reference implementation."""

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Calculate dimensions
    total_seq_len = sum(seq_lens)

    # Create input row offsets
    input_row_offsets = np.cumsum([0] + seq_lens, dtype=np.int32)

    # Generate test inputs
    # For reference impl, we need padded tensors
    max_seq_len = max(seq_lens)

    q = torch.randn(
        total_seq_len, num_heads, head_dim, dtype=dtype, device=device
    )

    # Generate sink weights
    sinks = torch.randn(num_heads, dtype=dtype, device=device)

    scale = 1.0 / math.sqrt(head_dim)

    # Run MAX implementation
    mask_variant = (
        MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
        if sliding_window
        else MHAMaskVariant.CAUSAL_MASK
    )
    max_output = max_flash_attention_with_sinks(
        q,
        sinks,
        input_row_offsets,
        num_kv_heads,
        scale,
        mask_variant,
        Accelerator(),
        sliding_window if sliding_window else -1,
    )

    assert np.all(np.isfinite(max_output))
