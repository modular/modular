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
"""MAX profiling configuration."""

from __future__ import annotations

import enum
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

from max.engine import GPUProfilingMode

from .max_config import MAXConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class ProfilingConfig(MAXConfig):
    gpu_profiling: GPUProfilingMode = GPUProfilingMode.OFF
    """Whether to enable GPU profiling of the model."""

    _config_file_section_name: str = "profiling_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    @classmethod
    def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
        """Get the enum mapping for ProfilingConfig."""
        return {
            "GPUProfilingMode": GPUProfilingMode,
        }

    def __post_init__(self):
        gpu_profiling_env = os.environ.get("MODULAR_ENABLE_PROFILING", "off")

        if self.gpu_profiling == GPUProfilingMode.OFF:
            try:
                self.gpu_profiling = GPUProfilingMode(gpu_profiling_env)
            except ValueError:
                valid_values = [mode.value for mode in GPUProfilingMode]
                raise ValueError(  # noqa: B904
                    "gpu_profiling must be one of: " + ", ".join(valid_values)
                )

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "gpu_profiling": "Whether to turn on GPU profiling for the model. This defaults to 'off'.",
        }
