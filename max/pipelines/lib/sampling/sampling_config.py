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
"""MAX sampling configuration."""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping
from dataclasses import dataclass

from max.dtype import DType
from max.interfaces import SamplingParamsGenerationConfigDefaults

from ..max_config import MAXConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class SamplingConfig(MAXConfig):
    in_dtype: DType = DType.float32
    """The data type of the input tokens."""

    out_dtype: DType = DType.float32
    """The data type of the output logits."""

    enable_structured_output: bool = False
    """Enable structured generation/guided decoding for the server. This allows the user to pass a json
    schema in the response_format field, which the LLM will adhere to."""

    enable_variable_logits: bool = False
    """Enable the sampling graph to accept a ragged tensor of different sequences as inputs, along with
    their associated logit_offsets. This is needed to produce additional logits for echo and speculative
    decoding purposes."""

    do_penalties: bool = False
    """Whether to apply frequency and presence penalties to the model's output."""

    enable_min_tokens: bool = False
    """Whether to enable min_tokens, which blocks the model from generating
    stopping tokens before the min_tokens count is reached. This defaults to
    false.
    """

    _config_file_section_name: str = "sampling_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    @classmethod
    def from_generation_config_sampling_defaults(
        cls,
        sampling_params_defaults: SamplingParamsGenerationConfigDefaults,
        **kwargs,
    ) -> SamplingConfig:
        """
        Create a SamplingConfig instance from SamplingParamsGenerationConfigDefaults and additional keyword arguments.

        This method inspects the provided SamplingParamsGenerationConfigDefaults to determine if penalty-related
        or min-tokens-related fields are set to non-default values. If so, it enables the corresponding flags
        ('do_penalties' and 'enable_min_tokens') in the resulting SamplingConfig unless they are already set
        in kwargs.

        Args:
            sampling_params_defaults (SamplingParamsGenerationConfigDefaults): The generation config defaults
                containing explicit values for sampling parameters.
            **kwargs: Additional keyword arguments to override or supplement the config.

        Returns:
            SamplingConfig: A new SamplingConfig instance with the appropriate fields set.
        """
        config_kwargs = kwargs.copy()

        gen_config_explicit = sampling_params_defaults.values_to_update
        if config_kwargs.get("do_penalties", False) is False:
            has_penalties = any(
                field in gen_config_explicit
                and gen_config_explicit[field] not in (None, 0, 1.0)
                for field in [
                    "frequency_penalty",
                    "presence_penalty",
                    "repetition_penalty",
                ]
            )
            if has_penalties:
                config_kwargs["do_penalties"] = True

        if config_kwargs.get("enable_min_tokens", False) is False:
            has_min_tokens = any(
                field in gen_config_explicit
                and gen_config_explicit[field] not in (None, 0)
                for field in ["min_tokens", "min_new_tokens"]
            )
            if has_min_tokens:
                config_kwargs["enable_min_tokens"] = True

        return cls(**config_kwargs)

    @classmethod
    def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
        """Get the enum mapping for SamplingConfig."""
        return {
            "DType": DType,
        }

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "enable_structured_output": "Whether to enable constrained decoding in the text generation pipeline. This defaults to false.",
            "enable_variable_logits": "Whether to enable the sampling graph to accept a ragged tensor of different sequences as inputs, along with their associated logit_offsets. This is needed to produce additional logits for echo and speculative decoding purposes. This defaults to false.",
            "enable_min_tokens": "Whether to enable min_tokens, which blocks the model from generating stopping tokens before the min_tokens count is reached. This defaults to false.",
            "do_penalties": "Whether to apply frequency and presence penalties to the model's output. This defaults to false.",
            "in_dtype": "The data type of the input tokens. This defaults to float32.",
            "out_dtype": "The data type of the output logits. This defaults to float32.",
        }
