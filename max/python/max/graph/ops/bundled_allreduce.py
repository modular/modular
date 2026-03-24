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
"""Op implementation for bundled allreduce using mo.parallel."""

from __future__ import annotations

from collections.abc import Iterable

from max.mlir.dialects import mo

from ..graph import Graph
from ..value import BufferValue, BufferValueLike, TensorValue, TensorValueLike
from .parallel import parallel
from .utils import _buffer_values, _tensor_values


def sum_body(tensor: TensorValue, signal_buffer: BufferValue) -> TensorValue:
    """Per-device bundled allreduce sum, for use inside an ``ops.parallel`` body.

    This emits a single ``mo.bundled.allreduce.sum`` op using the body
    block arguments and the graph's current chain for ordering. It is
    designed to be composed with other ops in the ``body_fn`` passed to
    :func:`~max.graph.ops.parallel`.

    The caller must ensure ``graph._current_chain`` is set to the merged
    chain (including all device-chain dependencies) before entering the
    parallel region. The :func:`sum` wrapper handles this automatically.

    Example:

    .. code-block:: python

        results, out_chain = ops.parallel(
            tensors,
            lambda t, sig: ops.relu(ops.bundled_allreduce.sum_body(t, sig)),
            extra_inputs=signal_buffers,
            chain=in_chain,
        )

    Args:
        tensor: The per-device tensor block argument.
        signal_buffer: The per-device signal buffer block argument.

    Returns:
        The reduced tensor.
    """
    graph = Graph.current
    ar_out, _chain = graph._add_op(
        mo.bundled_allreduce_sum,
        tensor,
        signal_buffer,
        graph._current_chain,
    )
    return ar_out.tensor


def sum(
    inputs: Iterable[TensorValueLike], signal_buffers: Iterable[BufferValueLike]
) -> list[TensorValue]:
    """Bundled allreduce sum using mo.parallel with per-device dispatch.

    Unlike :func:`max.graph.ops.allreduce.sum`, this stages a single
    ``mo.parallel`` region containing ``mo.bundled.allreduce.sum``, so the
    compiler can lower each device launch independently (async dispatch).

    Args:
        inputs: The input tensors to reduce, one per device.
        signal_buffers: Device buffer values used for synchronization,
            one per device (must match length of ``inputs``).

    Returns:
        A list of reduced tensors, one per device.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    if len(inputs) != len(signal_buffers):
        raise ValueError(
            f"expected number of inputs ({len(inputs)}) and number of "
            f"signal buffers ({len(signal_buffers)}) to match"
        )

    devices = [inp.device for inp in inputs]
    graph = Graph.current

    in_chain = graph._merge_chains(
        [graph._current_chain, *(graph.device_chains[d] for d in devices)]
    )

    # Set the current chain so that sum_body (and any composed ops) emit IR
    # referencing the merged chain rather than a stale pre-merge chain.
    graph._current_chain = in_chain

    result = parallel(
        inputs, sum_body, extra_inputs=signal_buffers, chain=in_chain
    )
    assert isinstance(result, tuple)
    results, out_chain = result

    graph._update_chain(out_chain)
    for d in devices:
        graph.device_chains[d] = out_chain

    return results
