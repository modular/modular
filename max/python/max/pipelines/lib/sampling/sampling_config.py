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
"""MAX sampling configuration."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from max.config import ConfigFileModel
from max.dtype import DType
from max.interfaces import SamplingParamsGenerationConfigDefaults
from pydantic import BeforeValidator, Field, PrivateAttr

logger = logging.getLogger("max.pipelines")


def _coerce_dtype(value: Any) -> DType | Any:
    """Coerce string values to DType enum members.

    DType is a C++ enum with integer values, so Pydantic cannot natively
    coerce YAML/CLI strings like ``"float32"`` into :class:`DType` members.
    This validator handles case-insensitive name matching.
    """
    if isinstance(value, DType):
        return value
    if isinstance(value, str):
        value_cf = value.casefold()
        for member in DType:
            if member.name.casefold() == value_cf:
                return member
    return value


CoercedDType = Annotated[DType, BeforeValidator(_coerce_dtype)]


class SamplingConfig(ConfigFileModel):
    in_dtype: CoercedDType = Field(
        default=DType.float32,
        description="The data type of the input tokens.",
    )

    out_dtype: CoercedDType = Field(
        default=DType.float32,
        description="The data type of the output logits.",
    )

    enable_structured_output: bool = Field(
        default=False,
        description=(
            "Enable structured generation/guided decoding for the server. This "
            "allows the user to pass a json schema in the response_format "
            "field, which the LLM will adhere to."
        ),
    )

    enable_variable_logits: bool = Field(
        default=False,
        description=(
            "Enable the sampling graph to accept a ragged tensor of different "
            "sequences as inputs, along with their associated logit_offsets. "
            "This is needed to produce additional logits for echo and "
            "speculative decoding purposes."
        ),
    )

    enable_penalties: bool = Field(
        default=False,
        description=(
            "Whether to apply frequency and presence penalties to the model's "
            "output."
        ),
    )

    enable_min_tokens: bool = Field(
        default=False,
        description=(
            "Whether to enable min_tokens, which blocks the model from "
            "generating stopping tokens before the min_tokens count is reached."
        ),
    )

    _config_file_section_name: str = PrivateAttr(default="sampling_config")
    """The section name to use when loading this config from a ConfigFileModel file.
    This is used to differentiate between different config sections in a single
    ConfigFileModel file."""

    @classmethod
    def from_generation_config_sampling_defaults(
        cls,
        sampling_params_defaults: SamplingParamsGenerationConfigDefaults,
        **kwargs,
    ) -> SamplingConfig:
        """Creates a SamplingConfig from generation config defaults and kwargs.

        Inspects the provided defaults to determine if penalty-related or
        min-tokens-related fields are set to non-default values; if so,
        enables the corresponding flags in the result unless already set in
        kwargs.

        Args:
            sampling_params_defaults: The generation config defaults
                containing explicit values for sampling parameters.
            **kwargs: Additional keyword arguments to override or supplement
                the config.

        Returns:
            A new SamplingConfig instance with the appropriate fields set.
        """
        config_kwargs = kwargs.copy()

        gen_config_explicit = sampling_params_defaults.values_to_update
        if config_kwargs.get("enable_penalties", False) is False:
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
                config_kwargs["enable_penalties"] = True

        if config_kwargs.get("enable_min_tokens", False) is False:
            has_min_tokens = any(
                field in gen_config_explicit
                and gen_config_explicit[field] not in (None, 0)
                for field in ["min_tokens", "min_new_tokens"]
            )
            if has_min_tokens:
                config_kwargs["enable_min_tokens"] = True

        return cls(**config_kwargs)
