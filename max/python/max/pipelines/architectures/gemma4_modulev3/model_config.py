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
"""Finalized checkpoint configuration for registered Gemma4 roles."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from max.graph.weights import WeightData
from max.nn.outputs import ReturnLogits
from max.pipelines.lib.graph_specs import GraphRole
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.registration_runtime import (
    DeploymentConfig,
    ModuleRuntime,
)
from max.pipelines.lib.weight_transform import WeightTransform
from transformers import AutoConfig
from typing_extensions import Self

from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..gemma4.weight_adapters import GEMMA4_VISION_PREFIXES, GEMMA4_WEIGHT_RULES


@dataclass(kw_only=True)
class Gemma4Config(Gemma4ForConditionalGenerationConfig, DeploymentConfig):
    """Keeps raw keys for quantization, then assigns weights to their roles."""

    preserve_checkpoint_keys: ClassVar[bool] = True
    weight_prefixes: ClassVar[Mapping[str, str]] = {
        GraphRole.DECODER: "decoder.",
        GraphRole.VISION: "vision.",
    }
    runtime: ModuleRuntime | None = None

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, Any],
        return_logits: ReturnLogits | ModelOutputConfig,
    ) -> None:
        """Preserves raw checkpoint names while parsing quantization."""
        outputs = (
            return_logits
            if isinstance(return_logits, ModelOutputConfig)
            else ModelOutputConfig(return_logits=return_logits)
        )
        # Finalizes a copy so the planning config stays unchanged. Not
        # dataclasses.replace: it reads every field, and Gemma4TextConfig
        # overrides the rope_theta field with a property that raises.
        self.text_config = copy.copy(self.text_config)
        super().finalize(
            huggingface_config, dict(state_dict), outputs.return_logits
        )
        self.text_config.return_hidden_states = outputs.return_hidden_states
        n_devices = len(self.devices)
        if n_devices > 1 and (
            self.text_config.num_key_value_heads % n_devices
            or self.text_config.num_global_key_value_heads % n_devices
        ):
            raise ValueError(
                "Gemma4 ModuleV3 tensor parallelism requires the device count "
                "to divide both local and global KV head counts"
            )

    def apply_runtime(self, runtime: ModuleRuntime) -> Self:
        """Binds the deployment mesh used to construct the decoder."""
        return replace(self, runtime=runtime)

    def prepare_weights(
        self, state_dict: Mapping[str, WeightData]
    ) -> dict[str, WeightData]:
        """Applies the architecture's declared rules to each role separately."""
        transform = WeightTransform(GEMMA4_WEIGHT_RULES)
        language = {
            name: value
            for name, value in state_dict.items()
            if not name.startswith(GEMMA4_VISION_PREFIXES)
        }
        vision = {
            name: value
            for name, value in state_dict.items()
            if name.startswith(GEMMA4_VISION_PREFIXES)
        }
        return {
            **{
                "decoder.language_model." + name: value
                for name, value in transform.apply_data(language).items()
            },
            **{
                "vision." + name: value
                for name, value in transform.apply_data(vision).items()
            },
        }
