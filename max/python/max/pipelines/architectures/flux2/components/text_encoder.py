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

"""Graph 1: Mistral3 text encoder component for Flux2Executor."""

from __future__ import annotations

from typing import Any

import numpy as np
from max.driver import Buffer, load_devices
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import Weights, load_weights
from max.pipelines.lib.compiled_component import CompiledComponent
from max.pipelines.lib.model_manifest import ModelManifest
from max.profiler import traced

# ModuleV2 imports only.
from ...mistral3.text_encoder.mistral3 import Mistral3TextEncoderTransformer
from ...mistral3.text_encoder.model_config import Mistral3TextEncoderConfig


class TextEncoder(CompiledComponent):
    """Graph 1: Mistral3 text encoder -> prompt_embeds + text_ids.

    Encapsulates the full lifecycle: config extraction from the manifest,
    weight loading and key adaptation, Module construction, graph
    compilation, and runtime execution.

    Output shapes:
        - ``prompt_embeds``: ``(1, S, L*D)`` where L = number of hidden
          state layers, D = hidden_size
        - ``text_ids``: ``(1, S, 4)`` int64 with T=0, H=0, W=0,
          L=arange(S)
    """

    _model: Model

    @traced(message="TextEncoder.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
    ) -> None:
        super().__init__(manifest, session)

        config = manifest["text_encoder"]
        config_dict = config.huggingface_config.to_dict()
        encoding = config.quantization_encoding or "bfloat16"
        devices = load_devices(config.device_specs)

        mistral_config = Mistral3TextEncoderConfig.initialize_from_config(
            config_dict, encoding, devices
        )

        paths = config.resolved_weight_paths()
        weights = load_weights(paths)
        state_dict = self._adapt_state_dict(weights)

        module = Mistral3TextEncoderTransformer(mistral_config)
        module.load_state_dict(state_dict, weight_alignment=1, strict=True)

        with Graph("text_encode", input_types=module.input_types()) as graph:
            outputs = module(*(v.tensor for v in graph.inputs))
            graph.output(outputs)

        self._model = self._session.load(
            graph, weights_registry=module.state_dict()
        )

    @traced(message="TextEncoder.__call__")
    def __call__(self, tokens: Buffer) -> tuple[Buffer, Buffer]:
        """Encode token IDs into prompt embeddings and text position IDs.

        Args:
            tokens: Token IDs, shape ``(S,)`` int64.

        Returns:
            A tuple of ``(prompt_embeds, text_ids)`` where:
            - ``prompt_embeds`` has shape ``(1, S, L*D)``
            - ``text_ids`` has shape ``(1, S, 4)`` int64
        """
        result = self._model.execute(tokens)
        prompt_embeds = (
            result[0] if isinstance(result, (list, tuple)) else result
        )

        batch_size = prompt_embeds.shape[0]
        seq_len = prompt_embeds.shape[1]
        text_ids = self._build_text_ids(batch_size, seq_len)

        return prompt_embeds, text_ids

    @staticmethod
    def _build_text_ids(batch_size: int, seq_len: int) -> Buffer:
        """Create 4D text position IDs in (T, H, W, L) format.

        For text tokens: T=0, H=0, W=0, L=arange(S).

        Returns:
            Buffer of shape ``(batch_size, seq_len, 4)`` int64.
        """
        coords = np.stack(
            [
                np.zeros(seq_len, dtype=np.int64),
                np.zeros(seq_len, dtype=np.int64),
                np.zeros(seq_len, dtype=np.int64),
                np.arange(seq_len, dtype=np.int64),
            ],
            axis=-1,
        )  # (seq_len, 4)
        text_ids = np.tile(coords[np.newaxis, :, :], (batch_size, 1, 1))
        return Buffer.from_dlpack(np.ascontiguousarray(text_ids))

    @staticmethod
    def _adapt_state_dict(weights: Weights) -> dict[str, Any]:
        """Strip HuggingFace prefixes to match the Module hierarchy.

        Adapts keys like ``language_model.model.layers.0.…`` or
        ``model.layers.0.…`` to ``layers.0.…``, keeping only
        ``embed_tokens.*`` and ``layers.*`` entries.
        """
        state_dict: dict[str, Any] = {}
        for key, value in weights.items():
            if key.startswith("language_model.model."):
                adapted = key.removeprefix("language_model.model.")
            elif key.startswith("model."):
                adapted = key.removeprefix("model.")
            elif key.startswith(("embed_tokens.", "layers.")):
                adapted = key
            else:
                continue
            if not adapted.startswith(("embed_tokens.", "layers.")):
                continue
            state_dict[adapted] = value.data()
        return state_dict
