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
"""Regression tests for CENG-684: the gemma4 MTP path must carry vision.

The unified MTP (speculative-decoding) path for Gemma 4 was originally
text-only: the arch used a text-only context/modalities and the fused graph
hardcoded empty image embeddings into the target LM, so image tokens were
ingested by the tokenizer but the vision encoder output never reached the
model -- the served model was effectively blind on image prompts.

These CPU-only structural tests guard the two wiring points that made the
model blind, so a revert to the text-only path fails fast without needing a
full GPU serve (the end-to-end "produces non-blind output" check is the
served smoke on the real checkpoint):

* the arch advertises image/video and uses the multimodal context, so the
  tokenizer injects image tokens and a vision-aware batch processor runs; and
* the fused MTP graph declares the per-device image-embedding + scatter-index
  inputs (``enable_vision``), so the vision encoder output has somewhere to
  bind and reach the target's ``merge_multimodal_embeddings``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.gemma4.context import Gemma4Context
from max.pipelines.architectures.unified_mtp_gemma4 import (
    unified_mtp_gemma4_arch,
)
from max.pipelines.architectures.unified_mtp_gemma4.batch_processor import (
    UnifiedMTPGemma4BatchProcessor,
)
from max.pipelines.architectures.unified_mtp_gemma4.model import (
    UnifiedMTPGemma4Model,
)
from max.pipelines.architectures.unified_mtp_gemma4.unified_mtp_gemma4 import (
    UnifiedMTPGemma4,
)
from max.pipelines.modeling.types import InputModality

_HIDDEN_SIZE = 128


def test_unified_mtp_gemma4_arch_is_multimodal() -> None:
    """The MTP arch must be multimodal, not served text-only.

    A text-only ``context_type``/``input_modalities`` is exactly what made
    the model blind: without image/video modalities the request path never
    injects image tokens or runs a vision-aware batch processor.
    """
    assert unified_mtp_gemma4_arch.context_type is Gemma4Context
    assert InputModality.IMAGE in unified_mtp_gemma4_arch.input_modalities
    assert InputModality.VIDEO in unified_mtp_gemma4_arch.input_modalities
    assert unified_mtp_gemma4_arch.batching is UnifiedMTPGemma4BatchProcessor
    assert unified_mtp_gemma4_arch.pipeline_model is UnifiedMTPGemma4Model


def test_mtp_graph_declares_per_device_vision_inputs() -> None:
    """The fused MTP graph signature must carry vision inputs.

    ``input_types`` must enable the vision inputs so the per-device image
    embeddings + scatter indices immediately follow ``tokens`` and reach the
    target LM. If ``enable_vision`` regresses, the tensor right after
    ``tokens`` is the (uint32, 1-D) row-offsets input and these assertions
    fail.
    """
    n_devices = 2
    devices = [DeviceRef("gpu", i) for i in range(n_devices)]

    # Duck-typed ``self`` exposing only what ``input_types`` reads; avoids
    # constructing the full target + draft modules. Cast for the type checker
    # since we deliberately pass a stand-in to the unbound method.
    fake_self = cast(
        UnifiedMTPGemma4,
        SimpleNamespace(
            config=SimpleNamespace(
                devices=devices,
                text_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE),
            ),
            enable_structured_output=False,
        ),
    )
    kv_params = MagicMock()
    kv_params.flattened_kv_inputs.return_value = []

    input_types = UnifiedMTPGemma4.input_types(fake_self, kv_params)

    # tokens, then per-device image embeddings, then per-device scatter indices.
    image_embeddings = input_types[1 : 1 + n_devices]
    image_indices = input_types[1 + n_devices : 1 + 2 * n_devices]

    for embed_type in image_embeddings:
        assert embed_type.dtype == DType.bfloat16
        assert int(embed_type.shape[-1]) == _HIDDEN_SIZE

    for index_type in image_indices:
        assert index_type.dtype == DType.int32
