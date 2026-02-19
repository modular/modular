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


class SpeculativeConfig(ConfigFileModel):
    """Configuration for speculative decoding."""

    speculative_method: SpeculativeMethod | None = Field(
        default=None, description="The speculative decoding method to use."
    )

    num_speculative_tokens: int = Field(
        default=5, description="The number of speculative tokens."
    )

    _config_file_section_name: str = "speculative_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    def is_eagle(self) -> bool:
        """Returns whether the speculative method is EAGLE (shared embedding/lm_head)."""
        return self.speculative_method == "eagle"

    def is_standalone(self) -> bool:
        """Returns whether the speculative method is a standalone model."""
        return self.speculative_method == "standalone"

    def is_mtp(self) -> bool:
        """Returns whether the speculative method is MTP."""
        return self.speculative_method == "mtp"
