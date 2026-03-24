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

"""Mistral3 text encoder ComponentModel wrapper.

This module provides a graph-API ComponentModel wrapper for the Mistral3 text
encoder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import traced

from .mistral3 import Mistral3TextEncoderTransformer
from .model_config import Mistral3TextEncoderConfig


class Mistral3TextEncoderModel(ComponentModel):
    """Mistral3 text encoder Module V2."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.session = session
        self.config = Mistral3TextEncoderConfig.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.load_model()

    @traced(message="Mistral3TextEncoderModel.load_model")
    def load_model(self) -> Callable[..., Any]:
        """Load and compile the Mistral3 text encoder."""
        state_dict = {}
        for key, value in self.weights.items():
            if key.startswith("language_model.model."):
                adapted_key = key.removeprefix("language_model.model.")
            elif key.startswith("model."):
                adapted_key = key.removeprefix("model.")
            elif key.startswith(("embed_tokens.", "layers.")):
                adapted_key = key
            else:
                continue
            if not adapted_key.startswith(("embed_tokens.", "layers.")):
                continue
            state_dict[adapted_key] = value.data()

        nn_model = Mistral3TextEncoderTransformer(self.config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)
        self.state_dict = nn_model.state_dict()

        with Graph(
            "mistral3_text_encoder",
            input_types=nn_model.input_types(),
        ) as graph:
            outputs = nn_model(*(value.tensor for value in graph.inputs))
            graph.output(outputs)

        self.model: Model = self.session.load(
            graph,
            weights_registry=self.state_dict,
        )
        return self.model.execute

    def __call__(self, tokens: Buffer) -> Buffer:
        """Run the compiled text encoder."""
        outputs = self.model.execute(tokens)
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs
