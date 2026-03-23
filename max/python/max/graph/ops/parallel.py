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
from ..value import (
    BufferValue,
    BufferValueLike,
    TensorValue,
    TensorValueLike,
    Value,
    _ChainValue,
)


def parallel(
    inputs: Iterable[TensorValueLike],
    body_fn: Callable[..., TensorValue | Iterable[TensorValue]],
    *,
    extra_inputs: Iterable[BufferValueLike] | None = None,
    chain: _ChainValue | None = None,
) -> list[TensorValue] | tuple[list[TensorValue], _ChainValue]:
    """Execute a function in parallel for each input via ``mo.parallel``.

    The body function receives a representative ``TensorValue`` (typed like
    the first input) and should return one ``TensorValue``.  The runtime
    dispatches the body once per input, substituting the actual per-launch
    tensor.

    When ``extra_inputs`` are provided (e.g. signal buffers for bundled
    collectives), the parallel op uses tupled syntax and the body function
    receives an additional ``BufferValue`` argument per extra-input group.

    When ``chain`` is provided, the parallel region is sequenced relative
    to prior ops and the returned ``out_chain`` represents completion of
    all parallel launches. A chain is required when ``extra_inputs``
    contain buffers that need ordering guarantees.

    Examples:

    .. code-block:: python

        # Simple elementwise (no chain needed):
        results = ops.parallel([gpu0, gpu1], lambda x: ops.relu(x))

        # Bundled allreduce with chain:
        results, out_chain = ops.parallel(
            tensors, body_fn, extra_inputs=signal_bufs, chain=in_chain
        )

    Args:
        inputs: Tensors to dispatch over.  All must share the same shape and
            dtype; device labels must match (IDs may differ).
        body_fn: Callable that takes one ``TensorValue`` (and optionally one
            ``BufferValue`` per extra-input group) and returns one
            ``TensorValue`` result.
        extra_inputs: Optional per-device buffer values (e.g. signal buffers).
            When provided, must have the same length as ``inputs``.
        chain: Optional chain value for sequencing. Required when
            extra_inputs contain buffers. Typically obtained from
            ``graph._merge_chains(...)``.

    Returns:
        When ``chain`` is provided: ``(tensor_results, out_chain)``.
        When ``chain`` is omitted: ``tensor_results``.
    """
    tensor_inputs: list[TensorValue] = [TensorValue(v) for v in inputs]
    if not tensor_inputs:
        raise ValueError("parallel requires at least one input")

    buffer_inputs: list[BufferValue] | None = None
    if extra_inputs is not None:
        buffer_inputs = [BufferValue(v) for v in extra_inputs]
        if len(buffer_inputs) != len(tensor_inputs):
            raise ValueError(
                f"extra_inputs length ({len(buffer_inputs)}) must match "
                f"inputs length ({len(tensor_inputs)})"
            )

    graph = Graph.current
    result_types = [inp.type.to_mlir() for inp in tensor_inputs]

    op_args: list[object] = [result_types, tensor_inputs]
    if buffer_inputs is not None:
        op_args.append(buffer_inputs)

    op_kwargs: dict[str, object] = {}
    if chain is not None:
        op_kwargs["in_chain"] = chain

    with graph._pause_verification():
        results, parallel_op = graph._add_op_get_op_with_results(
            mo.parallel_, *op_args, **op_kwargs
        )

        body_block = parallel_op.bodyRegion.blocks[0]
        with graph._block(body_block):
            block_args = [
                Value.from_mlir(_Value._from_cmlir(arg))
                for arg in body_block.arguments
            ]

            if buffer_inputs is not None:
                body_result = body_fn(*block_args)
            else:
                assert len(block_args) == 1
                body_result = body_fn(block_args[0])

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

    num_tensors = len(tensor_inputs)
    tensor_results = [r.tensor for r in results[:num_tensors]]

    if chain is not None:
        out_chain = _ChainValue(results[num_tensors])
        return tensor_results, out_chain

    return tensor_results
