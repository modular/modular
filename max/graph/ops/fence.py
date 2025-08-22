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
"""Fence: identity with ordering semantics (Python-level stub).

This function expresses a readiness fence for one or more values.
"""

from __future__ import annotations

from typing import Any

from max.mlir.dialects import mo

from ..graph import Graph
from ..value import Value


def fence(*values: Value[Any]) -> list[Value[Any]]:
    """Returns the input value(s) unchanged, serving as a scheduling barrier.

    This operation is a pure identity on values but prevents the asynchronous
    runtime from reordering operations across it. This is critical when dealing
    with operations that have implicit cross-device synchronization
    requirements.

    The primary use case is preventing deadlocks with distributed operations
    that are implemented as per-device ops with implicit synchronization:

    .. code-block:: python

        # Example: Per-device collective operations (illustrative).
        # NOTE: ops.custom("allreduce", ...) is a hypothetical per-device
        # user-defined allreduce op -- not a public API.
        with Graph("distributed_example", input_types=input_types) as g:
            x0, x1, x2 = g.inputs  # Inputs on GPU(0), GPU(1), GPU(2)

            # Each device runs its own allreduce operation instance.
            # These ops implicitly synchronize with each other.
            result_gpu0 = ops.custom("allreduce", device=DeviceRef.GPU(0),
                                    inputs=[x0, x1, x2])
            result_gpu1 = ops.custom("allreduce", device=DeviceRef.GPU(1),
                                    inputs=[x0, x1, x2])
            result_gpu2 = ops.custom("allreduce", device=DeviceRef.GPU(2),
                                    inputs=[x0, x1, x2])

            # DANGER: Without fence, the async runtime might reorder the
            # transfer to happen before all allreduce ops are enqueue'd.
            # This causes deadlock:
            # - GPU0's transfer waits for result_gpu1.
            # - GPU1's allreduce waits for GPU0 to reach its allreduce.
            # - But GPU0 can't reach its allreduce because it's blocked on
            #   transfer.

            # Solution: fence all results to prevent any reordering.
            [result_gpu0, result_gpu1, result_gpu2] = ops.fence(
                result_gpu0, result_gpu1, result_gpu2
            )

            # Now safe - transfers cannot be reordered before the fence.
            # NOTE: The following are each enqueue'd on GPU0's stream.
            y0 = ops.transfer_to(result_gpu1, DeviceRef.GPU(0))
            y1 = ops.transfer_to(result_gpu2, DeviceRef.GPU(0))
            g.output(result_gpu0, y0, y1)

    Args:
        values: One or more values to fence as variadic arguments.

    Returns:
        A list containing the same input value(s) unchanged, preserving order.

    Raises:
        ValueError: If no input values are provided.
    """
    if not values:
        raise ValueError("fence() requires at least one input")

    return Graph.current._add_op(mo.fence, list(values))
