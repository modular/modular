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
"""Checks Muse Spark's role contracts."""

from unittest.mock import MagicMock, NonCallableMock

import pytest
from max.graph.weights import Weights
from max.pipelines.architectures.muse_spark.arch import muse_spark_arch
from max.pipelines.architectures.muse_spark.model_config import MuseSparkConfig
from max.pipelines.architectures.muse_spark.modules import MuseSparkDecoder
from max.pipelines.architectures.muse_spark.weight_adapters import (
    convert_registered_state_dict,
)
from max.pipelines.lib.role_contracts import DecoderInputs, VisionInputs
from max.pipelines.speculative import spec_adapters


@pytest.mark.parametrize("ep_buffers", [None, ("counters", "pointers")])
def test_decoder_feeds_the_text_tower_in_its_graph_input_order(
    ep_buffers: tuple[str, ...] | None,
) -> None:
    tokens, offsets, count, embeddings, indices, full, sliding = (
        NonCallableMock() for _ in range(7)
    )
    logits = NonCallableMock()
    model = MagicMock(return_value=(logits,))
    decoder = NonCallableMock(spec=MuseSparkDecoder, model=model)
    outputs = MuseSparkDecoder.forward(
        decoder,
        DecoderInputs(
            tokens=tokens,
            input_row_offsets=offsets,
            return_n_logits=count,
            vision=VisionInputs(embeddings=embeddings, scatter_indices=indices),
            kv=(full, sliding),
            ep=ep_buffers,
        ),
    )
    model.assert_called_once_with(
        tokens,
        count,
        offsets,
        embeddings,
        indices,
        full,
        sliding,
        *(ep_buffers or ()),
    )
    assert outputs.next_token_logits is logits
    assert outputs.logits is None


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


def test_spec_decoding_holds_a_call_for_the_registered_target() -> None:
    # A registered decoder verifies drafts only under its exact arch name.
    assert spec_adapters.has_adapter(muse_spark_arch.name)


def test_exports_carry_the_ep_init_graph() -> None:
    """A virtual-device export never runs EP init, so only the specs carry it."""
    assert MuseSparkConfig.export_ep_init
