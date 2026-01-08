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

"""Compare SDK Qwen3VLTokenizer outputs to Transformers AutoProcessor outputs."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from max.interfaces import (
    RequestID,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.pipelines.architectures.qwen3vl_moe.tokenizer import Qwen3VLTokenizer
from transformers import AutoProcessor
from utils.config_loader import ConfigNames, get_config_loader


def _build_messages(image_url: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]


def test_qwen3vl_tokenizer() -> None:
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

    # Transformers reference outputs
    hf_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-30B-A3B-Instruct", trust_remote_code=True
    )
    hf_inputs = hf_processor.apply_chat_template(
        _build_messages(image_url),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Convert tensors to numpy for comparison
    hf_input_ids = hf_inputs["input_ids"].cpu().numpy()
    hf_attention_mask = hf_inputs["attention_mask"].cpu().numpy()
    hf_pixel_values = hf_inputs["pixel_values"].cpu().numpy()
    hf_image_grid_thw = hf_inputs["image_grid_thw"].cpu().numpy()

    # SDK tokenizer under test
    # Configure required vision token ids and settings from bundled config
    loader = get_config_loader()
    cfg = loader.load_config(ConfigNames.QWEN3VL_30B)

    tokenizer = Qwen3VLTokenizer(
        model_path="Qwen/Qwen3-VL-30B-A3B-Instruct", trust_remote_code=True
    )
    tokenizer.image_token_id = cfg["image_token_id"]
    tokenizer.video_token_id = cfg["video_token_id"]
    tokenizer.vision_start_token_id = cfg["vision_start_token_id"]
    tokenizer.vision_end_token_id = cfg["vision_end_token_id"]
    tokenizer.spatial_merge_size = cfg["vision_config"]["spatial_merge_size"]
    tokenizer.num_position_embeddings = cfg["vision_config"][
        "num_position_embeddings"
    ]

    # Build MAX request mirroring the HF messages
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": "Describe this image."},
                ],
            )
        ],
        request_id=RequestID("test-id"),
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
    )

    context = asyncio.run(tokenizer.new_context(request))

    # Extract SDK outputs
    # Use all_tokens (tokens[:end_idx]) instead of full tokens array to exclude resize padding
    # The context resizes tokens to CHUNK_SIZE boundary, so we should only compare the actual tokens
    sdk_input_ids = (
        context.tokens.all
    )  # This is tokens[:end_idx], excluding resize padding
    assert isinstance(sdk_input_ids, np.ndarray)

    assert context.vision_data is not None, (
        "Expected vision_data for image input"
    )
    sdk_pixel_values = context.vision_data.concatenated_pixel_values
    sdk_image_grid_thw = context.vision_data.image_grid_thw

    # Compare input_ids length and infer attention mask expectations
    # hf_input_ids shape is (B, seq_len). SDK input_ids shape is a ragged tensor of shape(total_seq_len,).

    assert sdk_input_ids.shape == (hf_input_ids.shape[1],)
    assert np.array_equal(sdk_input_ids, hf_input_ids.flatten())

    # HF mask should be all ones for single-sample, no padding
    assert hf_attention_mask.shape == (1, hf_input_ids.shape[1])
    assert int(np.sum(hf_attention_mask)) == hf_input_ids.shape[1]

    # Compare vision tensors
    assert sdk_pixel_values.shape == hf_pixel_values.shape
    assert np.array_equal(sdk_image_grid_thw, hf_image_grid_thw)

    # Numerical closeness on pixel values
    assert np.allclose(sdk_pixel_values, hf_pixel_values, rtol=1e-2, atol=1e-2)


def test_qwen3vl_tokenizer_no_images() -> None:
    """Test Qwen3VL tokenizer with text-only input (no images)."""
    # Build text-only messages
    text_only_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the capital of France?"}
            ],
        }
    ]

    # Transformers reference outputs
    hf_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-30B-A3B-Instruct", trust_remote_code=True
    )
    hf_inputs = hf_processor.apply_chat_template(
        text_only_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Convert tensors to numpy for comparison
    hf_input_ids = hf_inputs["input_ids"].cpu().numpy()
    hf_attention_mask = hf_inputs["attention_mask"].cpu().numpy()

    # SDK tokenizer under test
    # Configure required vision token ids and settings from bundled config
    loader = get_config_loader()
    cfg = loader.load_config(ConfigNames.QWEN3VL_30B)

    tokenizer = Qwen3VLTokenizer(
        model_path="Qwen/Qwen3-VL-30B-A3B-Instruct", trust_remote_code=True
    )
    tokenizer.image_token_id = cfg["image_token_id"]
    tokenizer.video_token_id = cfg["video_token_id"]
    tokenizer.vision_start_token_id = cfg["vision_start_token_id"]
    tokenizer.vision_end_token_id = cfg["vision_end_token_id"]
    tokenizer.spatial_merge_size = cfg["vision_config"]["spatial_merge_size"]
    tokenizer.num_position_embeddings = cfg["vision_config"][
        "num_position_embeddings"
    ]

    # Build MAX request mirroring the HF messages
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    {"type": "text", "text": "What is the capital of France?"}
                ],
            )
        ],
        request_id=RequestID("test-id"),
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
    )

    context = asyncio.run(tokenizer.new_context(request))

    # Extract SDK outputs
    # Use all_tokens (tokens[:end_idx]) instead of full tokens array to exclude resize padding
    # The context resizes tokens to CHUNK_SIZE boundary, so we should only compare the actual tokens
    sdk_input_ids = (
        context.tokens.all
    )  # This is tokens[:end_idx], excluding resize padding
    assert isinstance(sdk_input_ids, np.ndarray)

    # For text-only input, vision_data should be None
    assert context.vision_data is None, (
        "Expected vision_data to be None for text-only input"
    )

    # Compare input_ids length and infer attention mask expectations
    # hf_input_ids shape is (B, seq_len). SDK input_ids shape is a ragged tensor of shape(total_seq_len,).

    assert sdk_input_ids.shape == (hf_input_ids.shape[1],)
    assert np.array_equal(sdk_input_ids, hf_input_ids.flatten())

    # HF mask should be all ones for single-sample, no padding
    assert hf_attention_mask.shape == (1, hf_input_ids.shape[1])
    assert int(np.sum(hf_attention_mask)) == hf_input_ids.shape[1]

    # For text-only input, image_token_indices should be empty
    assert context.image_token_indices.shape == (0,), (
        "Expected empty image_token_indices for text-only input"
    )
