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
"""This module implements the low level concurrency library."""

from std.gpu.host import DeviceContext

from std.runtime.asyncrt import parallelism_level as std_parallelism_level


def parallelism_level(ctx: Optional[DeviceContext]) -> Int:
    """Gets the parallelism level from a DeviceContext.

    For CPU contexts this returns the number of worker threads in the
    runtime associated with that context. Falls back to the global
    parallelism level if the context is None or the query fails.

    Args:
        ctx: The device context to query.

    Returns:
        The parallelism level of the context.
    """
    from std.gpu.host import DeviceAttribute

    if ctx:
        try:
            return ctx.value().get_attribute(DeviceAttribute.PARALLELISM_LEVEL)
        except:
            pass
    return std_parallelism_level()
