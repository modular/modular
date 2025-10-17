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
"""MAX LoRA configuration."""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from .max_config import MAXConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class LoRAConfig(MAXConfig):
    enable_lora: bool = False
    """Enables LoRA on the server"""

    lora_paths: list[str] = field(default_factory=list)
    """List of statically defined LoRA paths"""

    max_lora_rank: int = 16
    """Maximum rank of all possible LoRAs"""

    max_num_loras: int = 1
    """The maximum number of active LoRAs in a batch"""

    _config_file_section_name: str = "lora_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    @classmethod
    def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
        """Get the enum mapping for LoRAConfig."""
        # LoRAConfig doesn't use any enums currently
        return {}

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "enable_lora": "Enables LoRA on the server",
            "lora_paths": "List of paths to the LoRAs.",
            "max_lora_rank": "The maximum rank of all possible LoRAs. Typically 8 or 16. Default is 16.",
            "max_num_loras": "The maximum number of active LoRAs in a batch. Default is 1.",
        }
