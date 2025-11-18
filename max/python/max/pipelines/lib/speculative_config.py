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
"""MAX Speculative Decoding configuration."""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from max.config import MAXConfig

logger = logging.getLogger("max.pipelines")


class SpeculativeMethod(str, Enum):
    """The supported methods for speculative decoding."""

    STANDALONE = "standalone"


@dataclass
class SpeculativeConfig(MAXConfig):
    """Configuration for speculative decoding."""

    speculative_method: SpeculativeMethod | None = None
    """The speculative decoding method to use."""

    num_speculative_tokens: int = 5
    """The number of speculative tokens."""

    _config_file_section_name: str = "speculative_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    def is_standalone(self) -> bool:
        """Whether the speculative method is a standalone model"""
        return self.speculative_method == SpeculativeMethod.STANDALONE

    @classmethod
    def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
        """Get the enum mapping for SpeculativeConfig."""
        return {
            "speculative_method": SpeculativeMethod,
        }

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "speculative_method": "The speculative decoding method to use (standalone, eagle, deepseek_mtp).",
            "num_speculative_tokens": "The number of speculative tokens. Defaults to the number in the draft model config if present.",
            "model": "The name of the draft model, eagle head, or additional weights.",
        }
