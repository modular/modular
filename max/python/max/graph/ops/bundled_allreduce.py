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

from max import mlir
from max._core import Value as _CValue
from max.mlir.dialects import mo

from ..graph import Graph, _location
from ..value import BufferValue, BufferValueLike, TensorValue, TensorValueLike
from ..value import Value as _GValue
from .parallel import _graph_type_to_mlir, _graph_val_to_mlir, parallel
from .utils import _buffer_values, _tensor_values


def sum(
    inputs: Iterable[TensorValueLike], signal_buffers: Iterable[BufferValueLike]
) -> list[TensorValue]:
    """Bundled allreduce sum using mo.parallel with per-device dispatch.

    Stages an ``mo.parallel`` region containing ``mo.bundled.expand`` and
    ``mo.bundled.allreduce.sum``, so the compiler can lower each device
    launch independently (async dispatch).

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

    graph._current_chain = in_chain

    input_types = [_graph_type_to_mlir(inp) for inp in inputs]
    sig_mlir_vals = [_graph_val_to_mlir(sb) for sb in signal_buffers]

    def body_fn(tensor: TensorValue, signal_buffer: BufferValue) -> TensorValue:
        tensor_mlir = _graph_val_to_mlir(tensor)
        chain_mlir = _graph_val_to_mlir(graph._current_chain)
        ip = mlir.InsertionPoint(graph._current_block)
        with ip, _location():
            expand_op = mo.BundledExpandOp(
                results_=input_types, input=tensor_mlir
            )
            peers = list(expand_op.results)
            chain_type = mlir.Type.parse("!mo.chain")
            ar_op = mlir.Operation.create(
                "mo.bundled.allreduce.sum",
                results=[input_types[0], chain_type],
                operands=[*peers, *sig_mlir_vals, chain_mlir],
            )
        graph._current_chain = _GValue.from_mlir(
            _CValue._from_cmlir(ar_op.results[1])
        )
        return _GValue.from_mlir(_CValue._from_cmlir(ar_op.results[0])).tensor

    result = parallel(
        inputs, body_fn, extra_inputs=signal_buffers, chain=in_chain
    )
    assert isinstance(result, tuple)
    results, out_chain = result

    graph._update_chain(out_chain)
    for d in devices:
        graph.device_chains[d] = out_chain

    return results
