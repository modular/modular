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

import os
from typing import get_args

from max.config import ConfigFileModel
from max.engine import GPUProfilingMode
from pydantic import ConfigDict, Field, PrivateAttr, field_validator


class ProfilingConfig(ConfigFileModel):
    """Configuration for the GPU (NVTX/Nsight) profiler and kernel tracing.

    ``max serve`` has no command-line flags for the ``kernel_trace_*``
    limits; set them in the ``profiling`` section of a ``--config-file``.
    """

    model_config = ConfigDict(frozen=True)

    # validate_default so the MODULAR_ENABLE_PROFILING fallback below also
    # applies when the field is not provided at all.
    gpu_profiling: GPUProfilingMode = Field(
        default="off",
        validate_default=True,
        description="Whether to enable GPU profiling of the model.",
    )
    """Whether to enable GPU profiling of the model."""

    # Read by the model worker. The CLI skips them, so only a config file
    # sets them.
    # TODO(MXTOOLS-651): The replay's span and capture-size limits set the
    # default pass cap; raise it once the replay works in chunks.
    kernel_trace_max_passes: int = Field(
        default=64,
        ge=1,
        description=(
            "The most forward passes a per-request kernel capture records "
            "before it stops."
        ),
    )
    """The most forward passes that one per-request kernel capture records.

    When tracing exports spans and ``MAX_SERVE_KERNEL_TRACE_HEADERS`` is on,
    ``max serve`` captures the GPU kernels of the forward passes that run a
    request whose ``x-max-trace-level`` header asks for them. A capture
    stops after this many passes, traced or not, and its requests are
    traced no further."""

    kernel_trace_max_batch_links: int = Field(
        default=128,
        ge=1,
        description="The most request links a max.batch span carries.",
    )
    """The most request links that one ``max.batch`` span carries.

    When tracing exports spans, ``max serve`` emits a ``max.batch`` span for
    each forward pass at ``MAX_SERVE_KERNEL_TRACE_LEVEL=batch`` or higher,
    and for each pass that runs a request traced with the
    ``x-max-trace-level`` header, linked to the spans of the requests it
    ran. The default is the OpenTelemetry SDK's per-span link limit, and a
    higher value raises that limit too."""

    kernel_trace_max_spans: int = Field(
        default=20_000,
        ge=1,
        description="The most spans a kernel capture's replay exports.",
    )
    """The most spans that replaying one per-request kernel capture exports.

    Once a kernel capture armed by a request's ``x-max-trace-level`` header
    stops, ``max serve`` replays its GPU activity into spans. Every traced
    pass's ``max.batch.gpu`` span comes first, and spans past this many are
    dropped, with a warning."""

    # Parsing holds the GIL, roughly 12 ms per MiB, so the default bounds the
    # stall to about 0.8 s; it peaks at about 6x the file size in memory.
    # TODO(MXTOOLS-651): at about 1.2 KB per kernel, 64 MiB is about 55k
    # kernels across all the worker's GPUs. A full 64-pass capture of an
    # 8B-class model (~500 kernels a pass) fits, but a large MoE's (~2k a
    # pass) does not; replay in chunks or parse off the GIL if those need
    # spans.
    kernel_trace_max_capture_bytes: int = Field(
        default=64 * 1024 * 1024,
        ge=1,
        description="The largest kernel capture file the replay reads.",
    )
    """The largest per-request kernel capture file, in bytes, that the
    replay reads.

    ``max serve`` replays a kernel capture armed by a request's
    ``x-max-trace-level`` header into spans once it stops. A larger capture
    file stays on disk but exports no spans, and the server warns."""

    _config_file_section_name: str = PrivateAttr(default="profiling_config")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    @field_validator("gpu_profiling", mode="before")
    @classmethod
    def _normalize_gpu_profiling(cls, value: object) -> object:
        """Applies MODULAR_ENABLE_PROFILING when the value is "off"."""
        if value == "off":
            gpu_profiling_env = os.environ.get(
                "MODULAR_ENABLE_PROFILING", "off"
            )
            valid_values = list(get_args(GPUProfilingMode))
            if gpu_profiling_env not in valid_values:
                raise ValueError(
                    "gpu_profiling must be one of: " + ", ".join(valid_values)
                )
            return gpu_profiling_env
        return value
