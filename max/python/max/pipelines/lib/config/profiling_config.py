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
"""MAX profiling configuration."""

from __future__ import annotations

import logging
import os
from typing import cast, get_args

from max.config import ConfigFileModel
from max.engine import GPUProfilingMode
from pydantic import Field, PrivateAttr, model_validator
from typing_extensions import Self

logger = logging.getLogger("max.pipelines")


class ProfilingConfig(ConfigFileModel):
    gpu_profiling: GPUProfilingMode = Field(
        default="off",
        description="Whether to enable GPU profiling of the model.",
    )

    _config_file_section_name: str = PrivateAttr(default="profiling_config")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    @model_validator(mode="after")
    def _normalize_gpu_profiling(self) -> Self:
        """Normalize gpu_profiling field after validation."""
        gpu_profiling_env = os.environ.get("MODULAR_ENABLE_PROFILING", "off")

        if self.gpu_profiling == "off":
            valid_values = list(get_args(GPUProfilingMode))
            if gpu_profiling_env in valid_values:
                self.gpu_profiling = cast(GPUProfilingMode, gpu_profiling_env)
            else:
                raise ValueError(
                    "gpu_profiling must be one of: " + ", ".join(valid_values)
                )
        return self
