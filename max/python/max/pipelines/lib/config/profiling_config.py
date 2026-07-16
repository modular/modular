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
    """Configuration for the GPU (NVTX/Nsight) and libkineto/HTA profilers."""

    gpu_profiling: GPUProfilingMode = Field(
        default="off",
        description="Whether to enable GPU profiling of the model.",
    )
    """Whether to enable GPU profiling of the model."""

    # --- libkineto / HTA / Dynolog profiler ---------------------------- #
    # Orthogonal to gpu_profiling above (NVTX/Nsight). When enabled, the
    # session emits Chrome-trace JSON via libkineto, importable by HTA and
    # triggerable on demand from Dynolog.
    #
    # Field names are prefixed with profiling_ so they line up 1:1 with the
    # session.debug.profiling_* setter mirrors defined alongside
    # InferenceSession.profiling.

    profiling_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for the libkineto-backed HTA/Dynolog profiler. "
            "Defaults to False; can also be toggled via the "
            "MODULAR_MAX_DEBUG_PROFILING_ENABLED environment variable "
            "(1/true/yes/on enable, 0/false/no/off disable). An explicitly "
            "provided value takes precedence over the environment variable."
        ),
    )
    """Master switch for the libkineto-backed profiler."""

    profiling_output_path: str | None = Field(
        default=None,
        description=(
            "Where to write the Chrome-trace JSON. Accepts a file path "
            "(supports {rank} and {pid} template variables) or a directory "
            "(traces are named per-PID per-timestamp inside it). Template "
            "expansion happens at trace-write time, not at config validation."
        ),
    )
    """Trace output path; file or directory."""

    profiling_dynolog_enabled: bool = Field(
        default=True,
        description=(
            "Whether to listen for Dynolog IPC on-demand-profile requests. "
            "Defaults to True so fleet-wide collection works out of the box."
        ),
    )
    """Whether the Dynolog IPC listener is active. On by default."""

    profiling_warmup_steps: int = Field(
        default=0,
        ge=0,
        description=(
            "Reserved for a future step-windowed capture mode (MXTOOLS-190): "
            "the number of Model::execute() iterations to skip after start() "
            "before recording would begin. Not yet consumed; the profiler "
            "currently records continuously from start() until stop()."
        ),
    )
    """Reserved; step-windowed capture is not yet implemented."""

    profiling_active_steps: int = Field(
        default=10,
        ge=1,
        description=(
            "Reserved for a future step-windowed capture mode (MXTOOLS-190): "
            "the number of Model::execute() iterations to record once the "
            "warmup window completes. Not yet consumed; the profiler currently "
            "records continuously from start() until stop(). Must be at "
            "least 1."
        ),
    )
    """Reserved; step-windowed capture is not yet implemented."""

    profiling_periodic_flush_seconds: int = Field(
        default=60,
        ge=1,
        description=(
            "Periodically flush in-flight trace chunks to disk every N "
            "seconds. Makes long-running serving captures crash-safe."
        ),
    )
    """Crash-safe flush cadence."""

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

    @model_validator(mode="after")
    def _normalize_kineto_enabled(self) -> Self:
        """Promote MODULAR_MAX_DEBUG_PROFILING_ENABLED into the enabled field.

        The env var fills the default only; an explicit caller value always
        wins — including ``False`` against a truthy env var::

            # MODULAR_MAX_DEBUG_PROFILING_ENABLED=1
            ProfilingConfig(profiling_enabled=False).profiling_enabled  # False

        Unlike ``_normalize_gpu_profiling``, which cannot be suppressed by
        passing ``gpu_profiling="off"`` when the env var is set, this uses
        ``model_fields_set`` so a caller can always opt out.
        """
        if "profiling_enabled" in self.model_fields_set:
            return self
        env = os.environ.get("MODULAR_MAX_DEBUG_PROFILING_ENABLED", "")
        # Tolerate surrounding whitespace/newlines that some shells and process
        # orchestrators append, so " 1 " enables rather than warns.
        normalized = env.strip().lower()
        if normalized == "":
            return self
        if normalized in ("1", "true", "yes", "on"):
            self.profiling_enabled = True
        elif normalized in ("0", "false", "no", "off"):
            self.profiling_enabled = False
        else:
            logger.warning(
                (
                    "MODULAR_MAX_DEBUG_PROFILING_ENABLED=%r ignored; expected"
                    " one of 1/true/yes/on or 0/false/no/off"
                ),
                env,
            )
        return self
