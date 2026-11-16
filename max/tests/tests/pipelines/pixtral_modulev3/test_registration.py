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
"""Checks registered Pixtral patch packing and role weight isolation."""

import asyncio
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights
from max.pipelines.architectures.pixtral.tokenizer import PixtralTokenizer
from max.pipelines.architectures.pixtral_modulev3.batch_processor import (
    PixtralModuleV3BatchProcessor,
)
from max.pipelines.architectures.pixtral_modulev3.context import (
    PixtralContext,
)
from max.pipelines.architectures.pixtral_modulev3.model_config import (
    PixtralConfig,
)
from max.pipelines.architectures.pixtral_modulev3.modules import PixtralDecoder
from max.pipelines.architectures.pixtral_modulev3.tokenizer import (
    PixtralModuleV3Tokenizer,
)
from max.pipelines.architectures.pixtral_modulev3.weight_adapters import (
    convert_registered_state_dict,
)
from max.pipelines.context import (
    ImageMetadata,
    TextAndVisionContext,
    TokenBuffer,
)
from max.pipelines.modeling.types import TextGenerationRequest
from max.support.image import hash_image

_IMG, _IMG_BREAK, _IMG_END = 10, 12, 13


def test_selected_images_keep_patch_order_and_separate_attention_blocks() -> (
    None
):
    config = PixtralConfig.__new__(PixtralConfig)
    config.patch_size = 2
    config.image_size = 4
    processor = PixtralModuleV3BatchProcessor.__new__(
        PixtralModuleV3BatchProcessor
    )
    processor.config = config
    first = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    second = np.full((1, 2, 2), 20, dtype=np.float32)
    images = [
        NonCallableMock(spec=ImageMetadata, pixel_values=first),
        NonCallableMock(spec=ImageMetadata, pixel_values=second),
    ]
    patches, mask, positions = processor.pack_vision_inputs(
        [(NonCallableMock(spec=TextAndVisionContext), images)], [CPU()]
    )
    np.testing.assert_array_equal(
        patches.to_numpy(),
        [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15],
            [20] * 4,
        ],
    )
    np.testing.assert_array_equal(positions.to_numpy(), [0, 1, 2, 3, 0])
    attention = mask.to_numpy()[0, 0]
    np.testing.assert_array_equal(attention[:4, :4], np.zeros((4, 4)))
    np.testing.assert_array_equal(attention[:4, 4], np.full(4, -10000))
    assert attention[4, 4] == 0


def test_registered_weights_keep_decoder_and_vision_names_independent() -> None:
    weight = NonCallableMock(data=lambda: NonCallableMock())
    weights: dict[str, Weights] = {
        "language_model.model.embed_tokens.weight": weight,
        "vision_tower.patch_conv.weight": weight,
        "multi_modal_projector.linear_1.weight": weight,
    }
    assert set(convert_registered_state_dict(weights)) == {
        "decoder.language_model.embed_tokens.weight",
        "vision.patch_conv",
        "vision.multi_modal_projector.linear_1.weight",
    }


def test_decoder_vision_inputs_stay_single_device_tensors() -> None:
    config = PixtralConfig.__new__(PixtralConfig)
    config.dtype = DType.bfloat16
    config.hidden_size = 16
    config.devices = [DeviceRef.CPU()]
    types = PixtralDecoder.vision_input_types(config, NonCallableMock())
    assert isinstance(types.embeddings, TensorType)
    assert isinstance(types.scatter_indices, TensorType)
    assert types.embeddings.shape == ["total_image_tokens", 16]
    assert types.scatter_indices.shape == ["num_image_token_indices"]


def test_images_take_rows_only_at_patch_tokens_and_carry_cache_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 2x2 patch image: a break closes the first row, an end the second.
    tokens = [1, _IMG, _IMG, _IMG_BREAK, _IMG, _IMG, _IMG_END, 2]
    base = TextAndVisionContext(
        max_length=16,
        tokens=TokenBuffer(np.asarray(tokens, dtype=np.int64)),
        vision_token_ids=[_IMG, _IMG_BREAK],
        images=[
            ImageMetadata(
                start_idx=1,
                end_idx=6,
                pixel_values=np.zeros((3, 4, 4), dtype=np.float32),
            )
        ],
    )

    async def base_context(
        self: PixtralTokenizer, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        del self, request
        return base

    monkeypatch.setattr(PixtralTokenizer, "new_context", base_context)
    tokenizer = PixtralModuleV3Tokenizer.__new__(PixtralModuleV3Tokenizer)
    tokenizer.vision_token_ids = [_IMG, _IMG_BREAK]
    tokenizer._image_size = 1024
    request = NonCallableMock(spec=TextGenerationRequest, images=[b"image"])

    context = asyncio.run(tokenizer.new_context(request))

    assert isinstance(context, PixtralContext)
    (image,) = context.images
    assert image.embedding_rows == 4
    assert image.image_hash == hash_image(b"image", 1024)
    np.testing.assert_array_equal(context.image_token_indices, [1, 2, 4, 5])
