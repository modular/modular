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
"""Checks Muse Glimmer's role contracts and optional image tower."""

from unittest.mock import MagicMock, NonCallableMock

from max.graph.weights import Weights
from max.pipelines.architectures.muse_glimmer.model_config import (
    MuseGlimmerConfig,
)
from max.pipelines.architectures.muse_glimmer.modules import (
    MuseGlimmerDecoder,
    MuseGlimmerVision,
)
from max.pipelines.architectures.muse_glimmer.weight_adapters import (
    convert_registered_state_dict,
)
from max.pipelines.lib.role_contracts import DecoderInputs, VisionInputs


def test_a_text_checkpoint_constructs_no_vision_tower() -> None:
    config = MuseGlimmerConfig.__new__(MuseGlimmerConfig)
    config.vision_config = None
    assert not MuseGlimmerVision.is_enabled(config)
    config.vision_config = NonCallableMock()
    assert MuseGlimmerVision.is_enabled(config)


def test_decoder_uses_semantic_fields_in_the_existing_tower_order() -> None:
    tokens, offsets, count, embeddings, indices, cache = (
        NonCallableMock() for _ in range(6)
    )
    logits = NonCallableMock()
    model = MagicMock(return_value=(logits,))
    decoder = NonCallableMock(spec=MuseGlimmerDecoder, model=model)
    outputs = MuseGlimmerDecoder.forward(
        decoder,
        DecoderInputs(
            tokens=tokens,
            input_row_offsets=offsets,
            return_n_logits=count,
            vision=VisionInputs(embeddings=embeddings, scatter_indices=indices),
            kv=(cache,),
        ),
    )
    model.assert_called_once_with(
        tokens, count, offsets, embeddings, indices, cache
    )
    assert outputs.next_token_logits is logits
    assert outputs.logits is None
    assert outputs.logit_offsets is None


def test_registered_weights_adapt_both_roles_without_cross_contamination() -> (
    None
):
    language = NonCallableMock(data=lambda: "language")
    vision = NonCallableMock(data=lambda: "vision")
    weights: dict[str, Weights] = {
        "model.language_model.embed_tokens.weight": language,
        "model.vision_tower.patch_embedder.position_embedding_table.weight": vision,
    }
    assert convert_registered_state_dict(weights) == {
        "decoder.language_model.embed_tokens.weight": "language",
        "vision.vision_tower.patch_embedder.position_embedding_table": "vision",
    }
