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

"""Interface for accessing GPU statistics of all supported GPU types."""

from __future__ import annotations

from contextlib import ExitStack
from types import TracebackType
from typing import Any

from ._nvml import NVMLContext
from ._rsmi import RSMIContext
from ._types import GPUStats


class GPUDiagContext:
    """Context for accessing GPU statistics of all supported GPU types."""

    def __init__(self) -> None:
        self._nvml: NVMLContext | None = None
        self._rsmi: RSMIContext | None = None
        self._stack: ExitStack | None = None
        # Errors are currently tucked away in here, never to be exposed, but
        # maybe we can expose this somehow in the future.  Still useful from
        # within PDB when debugging, sometimes.
        self._errors: list[tuple[str, Exception]] = []

    def __enter__(self) -> GPUDiagContext:
        if self._stack is not None:
            raise AssertionError("Can't enter an already-entered context")
        with ExitStack() as stack:
            try:
                self._nvml = stack.enter_context(NVMLContext())
            except Exception as e:
                self._errors.append(("nv", e))
            try:
                self._rsmi = stack.enter_context(RSMIContext())
            except Exception as e:
                self._errors.append(("amd", e))
            self._stack = stack.pop_all()
        return self

    def __exit__(
        self,
        exc_type: type[Any] | None,
        exc_value: Any,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._stack is not None
        self._nvml = None
        self._rsmi = None
        self._stack.close()
        self._stack = None

    def get_stats(self) -> dict[str, GPUStats]:
        """Get GPU statistics for all GPUs."""
        stats: dict[str, GPUStats] = {}
        if self._nvml is not None:
            nvml_stats = self._nvml.get_stats()
            for gpu_id, gpu_stats in nvml_stats.items():
                stats[f"nv{gpu_id}"] = gpu_stats
        if self._rsmi is not None:
            rsmi_stats = self._rsmi.get_stats()
            for gpu_id, gpu_stats in rsmi_stats.items():
                stats[f"amd{gpu_id}"] = gpu_stats
        return stats
