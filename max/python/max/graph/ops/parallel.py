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
"""Op implementation for mo.parallel."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from max._core import Value as _Value
from max.mlir.dialects import mo

from ..graph import Graph
from ..value import TensorValue, TensorValueLike, Value


def parallel(
    inputs: Iterable[TensorValueLike],
    body_fn: Callable[[TensorValue], TensorValue | Iterable[TensorValue]],
) -> list[TensorValue]:
    """Execute a function in parallel for each input via ``mo.parallel``.

    The body function receives a single representative ``TensorValue`` (typed
    like the first input) and should return one ``TensorValue``.  The runtime
    dispatches the body once per input, substituting the actual per-launch
    tensor.

    Example:

    .. code-block:: python

        results = ops.parallel([gpu0_tensor, gpu1_tensor], lambda x: ops.relu(x))

    Args:
        inputs: Tensors to dispatch over.  All must share the same shape and
            dtype; device labels must match (IDs may differ).
        body_fn: Callable that takes one ``TensorValue`` (the body block
            argument) and returns one ``TensorValue`` result.

    Returns:
        One result tensor per input, in the same order.
    """
    tensor_inputs: list[TensorValue] = [TensorValue(v) for v in inputs]
    if not tensor_inputs:
        raise ValueError("parallel requires at least one input")

    graph = Graph.current
    result_types = [inp.type.to_mlir() for inp in tensor_inputs]

    with graph._pause_verification():
        results, parallel_op = graph._add_op_get_op_with_results(
            mo.parallel_, result_types, tensor_inputs
        )

        body_block = parallel_op.bodyRegion.blocks[0]
        with graph._block(body_block):
            block_arg = Value.from_mlir(
                _Value._from_cmlir(body_block.arguments[0])
            )
            assert isinstance(block_arg, TensorValue)

            body_result = body_fn(block_arg)

            if isinstance(body_result, TensorValue):
                body_result = [body_result]
            else:
                body_result = list(body_result)

            if len(body_result) != 1:
                raise ValueError(
                    f"parallel body must return exactly 1 tensor, "
                    f"got {len(body_result)}"
                )

            graph._add_op(mo.YieldOp, body_result)

    graph._verify_op(parallel_op)

    return [r.tensor for r in results]
