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
"""Test AttentionWithRope layer with FP8 quantization on AMD GPUs."""

import numpy as np
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType, ops
from max.graph.weights import WeightData
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy import (
    Float8Config,
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
)
from max.nn.legacy.attention.attention_with_rope import AttentionWithRope
from max.nn.legacy.float8_config import Float8WeightScaleSpec
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.legacy.rotary_embedding import RotaryEmbedding
from test_common.context_utils import create_text_context


def _create_fp8_weights(
    num_heads: int,
    num_kv_heads: int,
    hidden_size: int,
    head_dim: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create FP8 weights with negative zeros injected at every second position."""
    torch.manual_seed(seed)

    # Generate random weights
    q_weight = torch.randn(
        num_heads * head_dim, hidden_size, dtype=torch.float32
    )
    k_weight = torch.randn(
        num_kv_heads * head_dim, hidden_size, dtype=torch.float32
    )
    v_weight = torch.randn(
        num_kv_heads * head_dim, hidden_size, dtype=torch.float32
    )
    o_weight = torch.randn(
        hidden_size, num_heads * head_dim, dtype=torch.float32
    )

    # Inject negative zero values - set every second value to negative zero
    negative_zero = torch.tensor(-0.0, dtype=torch.float32)

    for weight in [q_weight, k_weight, v_weight, o_weight]:
        weight_flat = weight.view(-1)
        weight_flat[1::2] = negative_zero

    # Convert to FP8
    return (
        q_weight.to(torch.float8_e4m3fn),
        k_weight.to(torch.float8_e4m3fn),
        v_weight.to(torch.float8_e4m3fn),
        o_weight.to(torch.float8_e4m3fn),
    )


def _create_kv_manager(
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    device: Accelerator,
    gpu_session: InferenceSession,
) -> tuple[PagedKVCacheManager, KVCacheParams]:
    """Create and configure the KV cache manager."""
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        page_size=128,
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        cache_strategy="paged",
        devices=[DeviceRef.GPU()],
    )

    manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=gpu_session,
        max_batch_size=128,
    )

    return manager, kv_params


def _create_attention_state_dict(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    float8_config: Float8Config,
    num_heads: int,
    num_kv_heads: int,
    hidden_size: int,
    head_dim: int,
) -> dict[str, WeightData]:
    """Create state dict for AttentionWithRope with FP8 weights and scales."""
    state_dict: dict[str, WeightData] = {}

    # Common weight entries for all projections
    weights_info = [
        ("q_proj", q_weight, num_heads * head_dim),
        ("k_proj", k_weight, num_kv_heads * head_dim),
        ("v_proj", v_weight, num_kv_heads * head_dim),
        ("o_proj", o_weight, hidden_size),
    ]

    for proj_name, weight, out_dim in weights_info:
        # Add weight
        state_dict[f"{proj_name}.weight"] = WeightData(
            Buffer.from_dlpack(weight.view(torch.uint8)).view(
                DType.float8_e4m3fn
            ),
            f"{proj_name}.weight",
            DType.float8_e4m3fn,
            Shape(weight.shape),
        )

        # Add weight scale based on granularity
        if (
            float8_config.weight_scale.granularity
            == Float8ScaleGranularity.TENSOR
        ):
            # Static scaling - single scale value
            scale_tensor = torch.tensor([1.0], dtype=torch.float32)
            scale_shape = Shape([1])
        else:  # ROWWISE
            # Dynamic scaling - per-row scales
            scale_tensor = torch.ones(out_dim, 1, dtype=torch.float32)
            scale_shape = Shape(scale_tensor.shape)

        state_dict[f"{proj_name}.weight_scale"] = WeightData(
            Buffer.from_dlpack(scale_tensor),
            f"{proj_name}.weight_scale",
            DType.float32,
            scale_shape,
        )

        # Add input scale only for static scaling
        if float8_config.input_scale.origin == Float8ScaleOrigin.STATIC:
            input_scale_tensor = torch.tensor([1.0], dtype=torch.float32)
            state_dict[f"{proj_name}.input_scale"] = WeightData(
                Buffer.from_dlpack(input_scale_tensor),
                f"{proj_name}.input_scale",
                DType.float32,
                Shape([1]),
            )

    return state_dict


def _build_and_execute_attention_graph(
    attention: AttentionWithRope,
    rope: RotaryEmbedding,
    kv_manager: PagedKVCacheManager,
    kv_params: KVCacheParams,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    device: Accelerator,
    gpu_session: InferenceSession,
    graph_name: str,
) -> torch.Tensor:
    """Build graph, execute model, and return results."""
    blocks_type, cache_lengths_type, lookup_table_type, max_lengths_type = (
        kv_params.get_symbolic_inputs()[0]
    )

    # Prepare input data
    np.random.seed(42)
    input_data = np.random.randn(batch_size * seq_len, hidden_size).astype(
        np.float32
    )

    if batch_size == 1:
        input_row_offsets_data = np.array([0, seq_len], dtype=np.uint32)
    else:
        input_row_offsets_data = np.array(
            [0, seq_len, seq_len * 2], dtype=np.uint32
        )

    with Graph(
        graph_name,
        input_types=[
            TensorType(
                DType.bfloat16,
                shape=("seq_len", hidden_size),
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.uint32,
                shape=["row_offsets_length"],
                device=DeviceRef.GPU(),
            ),
            blocks_type,
            cache_lengths_type,
            lookup_table_type,
            max_lengths_type,
        ],
    ) as graph:
        freqs_cis = rope.freqs_cis
        layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

        (
            x,
            input_row_offsets,
            blocks,
            cache_lengths,
            lookup_table,
            max_lengths,
        ) = graph.inputs

        kv_collection = PagedCacheValues(
            blocks.buffer,
            cache_lengths.tensor,
            lookup_table.tensor,
            max_lengths.tensor,
        )
        output = attention(
            layer_idx=layer_idx.tensor,
            x=x.tensor,
            kv_collection=kv_collection,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets.tensor,
        )

        graph.output(output)

    model = gpu_session.load(graph, weights_registry=attention.state_dict())

    # Prepare tensors for execution
    input_tensor = Buffer.from_dlpack(
        torch.from_numpy(input_data).to(torch.bfloat16)
    ).to(device)
    input_row_offsets_tensor = Buffer.from_dlpack(
        torch.from_numpy(input_row_offsets_data)
    ).to(device)

    # Set up KV cache batch
    batch = [create_text_context(np.empty(seq_len)) for _ in range(batch_size)]

    for context in batch:
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(context, replica_idx=0, num_steps=1)

    fetch_result = kv_manager.get_runtime_inputs([batch])[0]
    blocks_tensor = fetch_result[0]
    cache_lengths_tensor = fetch_result[1]
    lookup_table_tensor = fetch_result[2]
    max_lengths_tensor = fetch_result[3]

    result = model.execute(
        input_tensor,
        input_row_offsets_tensor,
        blocks_tensor,
        cache_lengths_tensor,
        lookup_table_tensor,
        max_lengths_tensor,
    )[0]

    return torch.from_dlpack(result)


def test_attention_with_rope_fp8_amd_static(
    gpu_session: InferenceSession,
) -> None:
    """Test AttentionWithRope applies AMD FP8 conversion with static scaling."""

    # Configuration for static scaling
    float8_config = Float8Config(
        input_scale=Float8InputScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.TENSOR,
            origin=Float8ScaleOrigin.STATIC,
        ),
        weight_scale=Float8WeightScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.TENSOR,
        ),
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
    )

    # Test parameters
    batch_size = 1
    seq_len = 4
    hidden_size = 128
    num_heads = 4
    num_kv_heads = 4
    head_dim = hidden_size // num_heads

    device = Accelerator(0)
    head_dim = hidden_size // num_heads

    # Set up KV cache and rope
    rope = RotaryEmbedding(
        dim=hidden_size,
        n_heads=num_heads,
        theta=10000.0,
        max_seq_len=seq_len * 2,
    )

    kv_manager, kv_params = _create_kv_manager(
        batch_size, seq_len, num_kv_heads, head_dim, device, gpu_session
    )

    # Create AttentionWithRope layer with float8_config
    attention = AttentionWithRope(
        rope=rope,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
        kv_params=kv_params,
        devices=[DeviceRef.GPU()],
        dtype=DType.float8_e4m3fn,
        float8_config=float8_config,
    )

    # Create weights with negative zeros
    q_weight, k_weight, v_weight, o_weight = _create_fp8_weights(
        num_heads, num_kv_heads, hidden_size, head_dim
    )

    # Create and load state dict using helper function
    state_dict = _create_attention_state_dict(
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        float8_config,
        num_heads,
        num_kv_heads,
        hidden_size,
        head_dim,
    )
    attention.load_state_dict(state_dict)

    # Execute the test
    result_torch = _build_and_execute_attention_graph(
        attention,
        rope,
        kv_manager,
        kv_params,
        batch_size,
        seq_len,
        hidden_size,
        device,
        gpu_session,
        "test_attention_fp8_amd_static",
    )

    assert torch.isfinite(result_torch).all(), (
        "Output should be finite (no NaN or Inf from FP8 conversion)"
    )


def test_attention_with_rope_fp8_amd_dynamic(
    gpu_session: InferenceSession,
) -> None:
    """Test AttentionWithRope applies AMD FP8 conversion with dynamic scaling."""

    # Configuration for dynamic scaling
    float8_config = Float8Config(
        input_scale=Float8InputScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.COLWISE,
            origin=Float8ScaleOrigin.DYNAMIC,
        ),
        weight_scale=Float8WeightScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.ROWWISE,
        ),
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
    )

    # Test parameters
    batch_size = 2
    seq_len = 8
    hidden_size = 256
    num_heads = 8
    num_kv_heads = 8

    device = Accelerator(0)
    head_dim = hidden_size // num_heads

    # Set up KV cache and rope
    rope = RotaryEmbedding(
        dim=hidden_size,
        n_heads=num_heads,
        theta=10000.0,
        max_seq_len=seq_len * 2,
    )

    kv_manager, kv_params = _create_kv_manager(
        batch_size, seq_len, num_kv_heads, head_dim, device, gpu_session
    )

    # Create AttentionWithRope layer with dynamic float8_config
    attention = AttentionWithRope(
        rope=rope,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
        kv_params=kv_params,
        devices=[DeviceRef.GPU()],
        dtype=DType.float8_e4m3fn,
        float8_config=float8_config,
    )

    # Create weights with negative zeros
    q_weight, k_weight, v_weight, o_weight = _create_fp8_weights(
        num_heads, num_kv_heads, hidden_size, head_dim
    )

    # Create and load state dict using helper function
    state_dict = _create_attention_state_dict(
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        float8_config,
        num_heads,
        num_kv_heads,
        hidden_size,
        head_dim,
    )
    attention.load_state_dict(state_dict)

    # Execute the test
    result_torch = _build_and_execute_attention_graph(
        attention,
        rope,
        kv_manager,
        kv_params,
        batch_size,
        seq_len,
        hidden_size,
        device,
        gpu_session,
        "test_attention_fp8_amd_dynamic",
    )

    assert torch.isfinite(result_torch).all(), (
        "Output should be finite (no NaN or Inf from FP8 conversion)"
    )
