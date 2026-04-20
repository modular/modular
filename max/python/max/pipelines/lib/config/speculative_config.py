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
"""MAX Speculative Decoding configuration."""

from __future__ import annotations

import logging
from typing import Literal

from max.config import ConfigFileModel
from pydantic import Field

logger = logging.getLogger("max.pipelines")

SpeculativeMethod = Literal["standalone", "eagle", "mtp"]
"""The supported methods for speculative decoding."""

RejectionSamplingStrategy = Literal[
    "greedy", "residual", "typical-acceptance", "logit-comparison"
]


class SpeculativeConfig(ConfigFileModel):
    """Configuration for speculative decoding."""

    speculative_method: SpeculativeMethod | None = Field(
        default=None, description="The speculative decoding method to use."
    )
    """The speculative decoding method to use."""

    num_speculative_tokens: int = Field(
        default=2, description="The number of speculative tokens."
    )
    """The number of speculative tokens to generate per step."""

    rejection_sampling_strategy: RejectionSamplingStrategy | None = Field(
        default=None,
        description=(
            "Rejection sampling strategy for verifying draft tokens."
            " Defaults to 'typical-acceptance' for eagle/mtp,"
            " 'residual' for standalone."
        ),
    )

    _config_file_section_name: str = "speculative_config"

    def is_eagle(self) -> bool:
        """Returns whether the speculative method is EAGLE (shared embedding/lm_head)."""
        return self.speculative_method == "eagle"

    def is_standalone(self) -> bool:
        """Returns whether the speculative method is a standalone model."""
        return self.speculative_method == "standalone"

    def is_mtp(self) -> bool:
        """Returns whether the speculative method is MTP."""
        return self.speculative_method == "mtp"

    def uses_greedy_rejection(self) -> bool:
        """Returns whether the greedy rejection sampling strategy is used."""
        return self.rejection_sampling_strategy == "greedy"

    def uses_typical_acceptance(self) -> bool:
        """Returns whether the typical-acceptance sampling strategy is used."""
        return self.rejection_sampling_strategy == "typical-acceptance"

    def uses_logit_comparison(self) -> bool:
        """Returns whether the logit-comparison sampling strategy is used."""
        return self.rejection_sampling_strategy == "logit-comparison"
