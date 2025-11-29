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

"""Performance profiling and tracing utilities for MAX.

This module provides tools for profiling and tracing MAX operations to analyze
performance characteristics. Profiling captures timing information for code
execution, which helps identify bottlenecks and optimize your models.

To enable profiling, set the ``MODULAR_ENABLE_PROFILING=1`` environment
variable before running your code. Without this variable, profiling calls will
be no-ops with minimal overhead.

The profiler supports three usage patterns:

1. **Context manager**: Use :class:`Tracer` as a context manager to profile a
   code block.
2. **Decorator**: Use :func:`@traced <traced>` to profile entire functions.
3. **Manual stack**: Use :class:`Tracer` methods to explicitly control profiling
   spans.
"""

from max._core.profiler import is_profiling_enabled, set_gpu_profiling_state
from max.profiler.tracing import Tracer, traced

__all__ = [
    "Tracer",
    "is_profiling_enabled",
    "set_gpu_profiling_state",
    "traced",
]
