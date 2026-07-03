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

"""Deprecated: use max.profiler.cpu instead."""

import warnings

warnings.warn(
    "max.diagnostics.cpu is deprecated and will be removed in a future release."
    " Use max.profiler.cpu instead.",
    DeprecationWarning,
    stacklevel=2,
)

from max.profiler.cpu import CPUMetrics as CPUMetrics
from max.profiler.cpu import CPUMetricsCollector as CPUMetricsCollector
from max.profiler.cpu import collect_pids_for_port as collect_pids_for_port
