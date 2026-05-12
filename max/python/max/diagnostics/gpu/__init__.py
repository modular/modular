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

"""Deprecated: use max.profiler.gpu instead."""

import warnings

warnings.warn(
    "max.diagnostics.gpu is deprecated and will be removed in a future release."
    " Use max.profiler.gpu instead.",
    DeprecationWarning,
    stacklevel=2,
)

from max.profiler.gpu import (
    HARDWARE_THROTTLE_REASONS as HARDWARE_THROTTLE_REASONS,
)
from max.profiler.gpu import BackgroundRecorder as BackgroundRecorder
from max.profiler.gpu import ClockStats as ClockStats
from max.profiler.gpu import GPUDiagContext as GPUDiagContext
from max.profiler.gpu import GPUStats as GPUStats
from max.profiler.gpu import MemoryStats as MemoryStats
from max.profiler.gpu import ThrottleReason as ThrottleReason
from max.profiler.gpu import UtilizationStats as UtilizationStats
