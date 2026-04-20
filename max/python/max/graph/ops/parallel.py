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

from max import mlir
from max._core import Value as _Value
from max.mlir.dialects import mo

from ..graph import Graph, _location
from ..value import (
    BufferValue,
    BufferValueLike,
    TensorValue,
    TensorValueLike,
    Value,
    _ChainValue,
)


def _graph_val_to_mlir(v: Value) -> mlir.Value:  # type: ignore[type-arg]
    """Convert a graph Value to a raw MLIR Value."""
    return mlir.Value._CAPICreate(v._mlir_value._CAPIPtr)  # type: ignore[attr-defined]


def _graph_type_to_mlir(v: Value) -> mlir.Type:  # type: ignore[type-arg]
    """Get the MLIR type of a graph Value."""
    return mlir.Type._CAPICreate(v.type.to_mlir()._CAPIPtr)  # type: ignore


def _build_bundle_type(element_types: list[mlir.Type]) -> mlir.Type:
    """Construct ``!mo.bundle<[t0, t1, ...]>`` from individual MLIR types."""
    types_str = ", ".join(str(t) for t in element_types)
    return mlir.Type.parse(f"!mo.bundle<[{types_str}]>")


def _create_tensor_bundle(
    values: list[TensorValue],
) -> tuple[mlir.Value, mlir.Type, list[mlir.Type]]:  # type: ignore[type-arg]
    """Create an ``mo.tensor.bundle`` op and return (bundle_val, bundle_type, elem_types)."""
    mlir_vals = [_graph_val_to_mlir(v) for v in values]
    mlir_types = [_graph_type_to_mlir(v) for v in values]
    bundle_type = _build_bundle_type(mlir_types)
    bundle_val = mo.tensor_bundle(bundle_type, mlir_vals)
    return bundle_val, bundle_type, mlir_types


def parallel(
    inputs: Iterable[TensorValueLike],
    body_fn: Callable[..., TensorValue | Iterable[TensorValue]],
    *,
    extra_inputs: Iterable[BufferValueLike] | None = None,
    chain: _ChainValue | None = None,
    result_types: list[object] | None = None,
) -> list[TensorValue] | tuple[list[TensorValue], _ChainValue]:
    """Execute a function in parallel for each input via ``mo.parallel``.

    The body function receives a representative ``TensorValue`` (typed like
    the first input) and should return one or more ``TensorValue`` results.
    The runtime dispatches the body once per input, substituting the actual
    per-launch tensor.

    When ``extra_inputs`` are provided (e.g. signal buffers for bundled
    collectives), the body function receives an additional ``BufferValue``
    argument per extra-input group.

    When ``chain`` is provided, the parallel region is sequenced relative
    to prior ops and the returned ``out_chain`` represents completion of
    all parallel launches.

    Args:
        inputs: Tensors to dispatch over.  All must share the same shape and
            dtype; device labels must match (IDs may differ).
        body_fn: Callable receiving one ``TensorValue`` (and optionally one
            ``BufferValue`` per extra-input group) and returning one or more
            ``TensorValue`` results.
        extra_inputs: Optional per-device buffer values (e.g. signal buffers).
            When provided, must have the same length as ``inputs``.
        chain: Optional chain value for sequencing.
        result_types: Optional explicit per-device result types for each yield
            operand.  When omitted, result types are inferred from inputs.

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
        if chain is None:
            raise ValueError(
                "chain is required when extra_inputs (buffers) are provided"
            )

    if chain is not None and buffer_inputs is None:
        raise ValueError(
            "extra_inputs (buffers) are required when chain is provided"
        )

    graph = Graph.current

    with graph._pause_verification():
        ip = mlir.InsertionPoint(graph._body)
        with ip, _location():
            tensor_bundle, tensor_bundle_type, tensor_mlir_types = (
                _create_tensor_bundle(tensor_inputs)
            )

            bundle_operands = [tensor_bundle]
            block_arg_types = [tensor_mlir_types[0]]

            buffer_mlir_vals: list[mlir.Value] | None = None  # type: ignore[type-arg]
            if buffer_inputs is not None:
                buffer_mlir_vals = [
                    _graph_val_to_mlir(v) for v in buffer_inputs
                ]
                buffer_mlir_types = [
                    _graph_type_to_mlir(v) for v in buffer_inputs
                ]
                block_arg_types.append(buffer_mlir_types[0])

            parallel_result_types: list[mlir.Type] = [tensor_bundle_type]
            chain_mlir = (
                _graph_val_to_mlir(chain) if chain is not None else None
            )

            parallel_op = mo.ParallelOp(  # type: ignore[call-arg]
                results_=parallel_result_types,
                inputs=bundle_operands,
                buffers=buffer_mlir_vals,  # type: ignore[arg-type]
                in_chain=chain_mlir,
                block_arg_types=block_arg_types,
            )

        # Populate the body block.
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

            if len(body_result) < 1:
                raise ValueError("parallel body must return at least 1 tensor")

            graph._add_op(mo.YieldOp, body_result)

    graph._verify_op(parallel_op)

    # Unbundle the result tensor bundle.
    result_bundle = parallel_op.results[0]
    with mlir.InsertionPoint(graph._body), _location():
        unbundled = mo.TensorUnbundleOp(
            outputs=tensor_mlir_types, input=result_bundle
        )

    tensor_results = [
        Value.from_mlir(_Value._from_cmlir(v)).tensor for v in unbundled.outputs
    ]

    if chain is not None:
        chain_result = parallel_op.results[-1]
        out_chain = _ChainValue(_Value._from_cmlir(chain_result))  # type: ignore[arg-type]
        return tensor_results, out_chain

    return tensor_results
