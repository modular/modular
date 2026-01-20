# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Tests for Qwen3VL text decoder rotary embedding."""

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.qwen3vl_moe.nn.data_processing import (
    get_rope_index,
)
from max.pipelines.architectures.qwen3vl_moe.nn.text_rotary import (
    Qwen3VLTextRotaryEmbedding,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextRotaryEmbedding as HFQwen3VLMoeTextRotaryEmbedding,
)
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader

RTOL = 8e-3
ATOL = 8e-3


@torch.no_grad()
def generate_torch_outputs(
    position_ids: torch.Tensor,
    input_tensor: torch.Tensor,
    hf_text_config: dict,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate reference outputs using the HuggingFace implementation."""
    # Create the reference model
    ref_model = HFQwen3VLMoeTextRotaryEmbedding(
        config=Qwen3VLMoeTextConfig(**hf_text_config)
    )
    ref_model.to(dtype=dtype).to(device=position_ids.device)
    ref_model.eval()

    # Forward pass
    with torch.no_grad():
        cos, sin = ref_model(input_tensor, position_ids)
        cos = cos[..., : cos.shape[-1] // 2]
        sin = sin[..., : sin.shape[-1] // 2]

    return cos, sin


def generate_max_outputs(
    position_ids: torch.Tensor,
    qwen3vl_config: dict,
    dtype: DType,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate outputs using MAX Qwen3VLTextRotaryEmbedding implementation."""

    text_config = qwen3vl_config["text_config"]

    # Extract mrope_section from rope_scaling if available
    rope_scaling = text_config.get("rope_scaling", {})
    mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

    # Create Qwen3VLTextRotaryEmbedding
    rotary_embedding = Qwen3VLTextRotaryEmbedding(
        dim=text_config["hidden_size"],
        n_heads=text_config["num_attention_heads"],
        theta=text_config["rope_theta"],
        max_seq_len=text_config["max_position_embeddings"],
        dtype=DType.bfloat16,
        mrope_section=mrope_section,
        head_dim=text_config["head_dim"],
        interleaved=True,
        scaling_params=None,
    )

    session = InferenceSession(devices=[CPU()])

    # Define input types
    position_ids_type = TensorType(
        DType.int64, shape=position_ids.shape, device=DeviceRef.CPU()
    )

    with Graph(
        "Qwen3VLTextRotaryEmbedding",
        input_types=(position_ids_type,),
    ) as graph:
        pos_ids = graph.inputs[0]
        # Compute freqs_cis from position_ids
        # freqs_cis returns (total_seq_len, head_dim) after reshaping
        freqs_cis = rotary_embedding.freqs_cis_base_position_ids(pos_ids.tensor)
        graph.output(freqs_cis)

    compiled = session.load(graph, weights_registry={})

    # Execute the model
    result = compiled.execute(
        Buffer.from_dlpack(position_ids),
    )
    freqs_cis_torch = from_dlpack(result[0])

    # Extract cos and sin: shape (total_seq_len, half_dim)
    cos_half = freqs_cis_torch[..., 0]  # (total_seq_len, half_dim)
    sin_half = freqs_cis_torch[..., 1]  # (total_seq_len, half_dim)

    # Add batch dimension to match HuggingFace output: (1, total_seq_len, half_dim)
    cos_half = cos_half.unsqueeze(0)
    sin_half = sin_half.unsqueeze(0)

    return cos_half, sin_half


@pytest.mark.parametrize(
    "height,width",
    # 14x20 patches, 28x14 patches, 146x98 patches (patch_size=16)
    [(224, 320), (448, 224), (2336, 1568)],
)
def test_decoder_rotary_embedding_with_grid_thw(
    height: int, width: int
) -> None:
    """Test rotary embedding using grid_thw and get_rope_index."""
    torch.manual_seed(42)

    # Load config
    loader = get_config_loader()
    hf_full_config = loader.load_config(ConfigNames.QWEN3VL_30B)
    hf_text_config = hf_full_config["text_config"]
    hf_vision_config = hf_full_config["vision_config"]
    qwen3vl_config = loader.create_qwen3vl_config(ConfigNames.QWEN3VL_30B)

    # Create test inputs
    hidden_size = hf_text_config["hidden_size"]
    num_heads = hf_text_config["num_attention_heads"]
    head_dim = hf_text_config.get("head_dim")
    patch_size = hf_vision_config["patch_size"]
    spatial_merge_size = hf_vision_config["spatial_merge_size"]

    # Calculate number of patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    seq_len = num_patches_h * num_patches_w

    # Generate grid_thw: (num_images, 3) where 3 is (temporal, height, width)
    # For images, temporal is 1
    grid_thw = np.array([[1, num_patches_h, num_patches_w]], dtype=np.int64)

    # Create input_ids with vision tokens
    # Format: [text_tokens..., vision_start_token, image_token, vision_tokens..., text_tokens...]
    batch_size = 1
    image_token_id = qwen3vl_config["image_token_id"]
    vision_start_token_id = qwen3vl_config["vision_start_token_id"]

    # Create a simple input_ids sequence with vision tokens
    # Start with some text tokens, then vision block, then more text
    num_text_before = 10
    num_text_after = 5
    input_ids_list = (
        [0] * num_text_before  # Text tokens (using 0 as placeholder)
        + [vision_start_token_id, image_token_id]  # Vision start + image token
        + [0] * seq_len  # Vision tokens (using 0 as placeholder)
        + [0] * num_text_after  # More text tokens
    )
    total_seq_len = len(input_ids_list)
    input_ids = np.array([input_ids_list], dtype=np.int64)

    # Get position_ids using get_rope_index
    position_ids_np, _rope_delta = get_rope_index(
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
        video_token_id=qwen3vl_config["video_token_id"],
        vision_start_token_id=vision_start_token_id,
        input_ids=input_ids,
        image_grid_thw=grid_thw,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=None,
    )
    # Convert to torch tensor: position_ids_np shape is (3, batch_size, seq_len)
    position_ids = torch.from_numpy(position_ids_np)

    # Create random hidden states for torch model: (batch_size, seq_len, num_heads, head_dim)
    input_tensor = torch.randn(
        batch_size, total_seq_len, num_heads, head_dim, dtype=torch.bfloat16
    )

    # Generate reference output
    torch_cos, torch_sin = generate_torch_outputs(
        position_ids=position_ids.to("cuda"),
        input_tensor=input_tensor.to("cuda"),
        hf_text_config=hf_text_config,
        dtype=torch.bfloat16,
    )
    torch_cos = torch_cos.to("cpu")
    torch_sin = torch_sin.to("cpu")

    # Generate MAX output
    position_ids = position_ids.squeeze(1)
    max_cos, max_sin = generate_max_outputs(
        position_ids=position_ids,
        qwen3vl_config=qwen3vl_config,
        dtype=DType.bfloat16,
    )

    # Verify output shapes
    expected_shape = (batch_size, total_seq_len, head_dim // 2)
    assert torch_cos.shape == expected_shape, (
        f"Expected cos shape {expected_shape}, got Torch: {torch_cos.shape}, Max: {max_cos.shape}"
    )
    assert max_cos.shape == expected_shape, (
        f"Expected cos shape {expected_shape}, got Torch: {torch_cos.shape}, Max: {max_cos.shape}"
    )
    assert torch_sin.shape == expected_shape, (
        f"Expected sin shape {expected_shape}, got Torch: {torch_sin.shape}, Max: {max_sin.shape}"
    )
    assert max_sin.shape == expected_shape, (
        f"Expected sin shape {expected_shape}, got Torch: {torch_sin.shape}, Max: {max_sin.shape}"
    )

    # Compare outputs
    assert_tensors_close(
        torch_cos,
        max_cos,
        rtol=RTOL,
        atol=ATOL,
        message=f"Rotary embedding cos outputs do not match for grid_thw {grid_thw.tolist()}",
    )
    assert_tensors_close(
        torch_sin,
        max_sin,
        rtol=RTOL,
        atol=ATOL,
        message=f"Rotary embedding sin outputs do not match for grid_thw {grid_thw.tolist()}",
    )
