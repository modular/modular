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

"""Tests for Idefics3 tokenizer."""

import io

import pytest
from max.interfaces import (
    RequestID,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.pipelines.architectures.idefics3.tokenizer import Idefics3Tokenizer
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from PIL import Image


class MockKVCacheConfig(KVCacheConfig):
    def __init__(self):
        self.enable_prefix_caching = True


class MockModelConfig(MAXModelConfig):
    def __init__(self):
        self._kv_cache = MockKVCacheConfig()


class MockPipelineConfig(PipelineConfig):
    def __init__(self):
        self._model_config = MockModelConfig()


@pytest.mark.asyncio
async def test_idefics3_tokenizer_image_token_indices() -> None:
    """Test that the tokenizer correctly computes image token indices."""

    pipeline_config = MockPipelineConfig()
    tokenizer = Idefics3Tokenizer(
        "HuggingFaceM4/Idefics3-8B-Llama3", pipeline_config=pipeline_config
    )
    assert tokenizer.vision_token_ids == [128257]
    assert tokenizer.enable_prefix_caching is True

    # Create a real image for the test using config dimensions
    img_buffer = io.BytesIO()
    image_size = 448
    Image.new("RGB", (image_size, image_size), color="red").save(
        img_buffer, format="PNG"
    )
    test_image = img_buffer.getvalue()

    # Create request with image
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    {"type": "text", "text": "test"},
                    {"type": "image", "content": test_image},
                ],
            )
        ],
        images=[test_image],
        request_id=RequestID("test-id"),
        model_name="HuggingFaceM4/Idefics3-8B-Llama3",
    )

    context = await tokenizer.new_context(request)
    assert context is not None
    # Idefics3 tokenizer turns this single image into 17 patch groups
    assert len(context.images) == 17
