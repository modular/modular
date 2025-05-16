# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from max import mlir
from max.driver import Accelerator, Tensor, accelerator_count
from max.dtype import torch_to_max_type
from max.engine import Model
from max.engine.api import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    KernelLibrary,
    Shape,
    TensorType,
    ops,
)
from max.mlir import Context

try:
    import torch
except ImportError:
    raise ImportError(
        "torch not found - install `max[torch]` (if using pip/uv) or max-conda (if using magic/conda)"
    )


class CustomOpLibrary:
    _context: Context
    _kernel_library: KernelLibrary
    _session: InferenceSession

    def __init__(self, kernel_library: Path | KernelLibrary):
        devices = [Accelerator(i) for i in range(accelerator_count())]

        self._context = Context()
        self._kernel_library = (
            kernel_library
            if isinstance(kernel_library, KernelLibrary)
            else KernelLibrary(self._context, [kernel_library])
        )
        self._session = InferenceSession(devices=devices)

    def __getattr__(self, attr: str):
        if attr.startswith("_"):
            return super().__getattribute__(attr)

        return CustomOp(self, attr)


@dataclass
class CustomOp:
    library: CustomOpLibrary
    name: str

    @property
    def context(self) -> mlir.Context:
        return self.library._context

    @property
    def kernel_library(self) -> KernelLibrary:
        return self.library._kernel_library

    @property
    def kernel(self) -> mlir.Operation:
        analysis = self.kernel_library._analysis
        return analysis.kernel(self.name)


###############################################################################
# Convert torch.Tensor to a TensorType
###############################################################################


def convert_shape(shape: torch.Size) -> Shape:
    return Shape([int(dim) for dim in shape])


def convert_device(device: torch.device) -> DeviceRef:
    type = device.type
    index = device.index or 0
    if type == "cpu":
        return DeviceRef.CPU(index)
    elif type == "cuda":
        return DeviceRef.GPU(index)
    else:
        raise TypeError(f"Unable to convert {type} to a MAX device type.")


def torch_tensor_to_type(tensor: torch.Tensor) -> TensorType:
    dtype = torch_to_max_type(tensor.dtype)
    shape = convert_shape(tensor.shape)
    device = convert_device(tensor.device)
    return TensorType(dtype, shape, device=device)


###############################################################################
# Tensor Conversions
###############################################################################


def to_torch_tensors(tensor: Sequence[Tensor]) -> TorchTensors:
    """Convert a MAX tensor to a torch.Tensor.

    This function handles the special case of most Torch operations which
    return a single tensor value.
    """
    torch_tensors = [torch.from_dlpack(t) for t in tensor]
    return torch_tensors[0] if len(torch_tensors) == 1 else torch_tensors


def custom_op_graph(
    op: CustomOp,
    input_types: Iterable[TensorType],
    result_types: Iterable[TensorType],
) -> Graph:
    """Construct the Graph API graph representing a call to a Mojo CustomOp.

    This function builds a graph which invokes the given Mojo CustomOp through
    a Max custom operation. The generated custom op is in destination passing
    style, where the outputs are mutable buffers passed to the graph.o

    Args:
        op: The custom operation to be called.
        result_types: The tensors which serve as a specification of the output
            types of the graph.

    Returns:
        Graph: The MAX graph calling the provided custom op.
    """

    output_types = [t.as_buffer() for t in result_types]
    graph_types = [*output_types, *input_types]

    with Graph(
        op.name,
        input_types=graph_types,
        context=op.context,
        kernel_library=op.kernel_library,
    ) as graph:
        results = ops.custom(
            op.name,
            list(graph.inputs[len(output_types) :]),
            out_types=list(result_types),
        )
        for input, result in zip(graph.inputs, results):
            input.buffer[...] = result.tensor

        graph.output()

    return graph


def get_strings(attr: mlir.Attribute) -> list[str]:
    as_array = mlir.ArrayAttr(attr)
    return [mlir.StringAttr(s).value for s in as_array]


def op_signature(op: mlir.Operation) -> inspect.Signature:
    """Compute the Python-level signature of the provided custom op.

    The computed signature is derived from the KGEN-level annotations on the
    given MLIR operation. These annotations are attached to the KGEN function
    at the MOGGPreElab stage of the compilation pipeline.

    This function currently only supports tensor inputs and outputs. Computed
    signature will have one torch.Tensor input/result for each DPS input/result
    of the custom operation.

    Args:
        op: The MLIR operation representing the custom op (kgen.func op).

    Returns:
        inspect.Signature: The Python-level signature for the custom op.
    """

    # TODO(GEX-2219): support non-dps outputs
    num_dps_outputs = mlir.IntegerAttr(
        op.attributes["mogg.num_dps_outputs"]
    ).value

    # TODO(GEX-2223): Expose more of MojoLibraryAnalysis so we don't need to
    # hard code MLIR attributes.
    io_specs = get_strings(op.attributes["mogg.args_io_specs"])
    arg_names = get_strings(op.attributes["mogg.arg_src_names"])
    input_specs = io_specs[num_dps_outputs:]
    nargs = len(input_specs)
    arg_names = arg_names[num_dps_outputs : num_dps_outputs + nargs]
    args = [
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=torch.Tensor,
        )
        for name in arg_names
    ]
    result_type = (
        torch.Tensor
        if num_dps_outputs == 1
        else tuple([torch.Tensor] * num_dps_outputs)
    )
    return inspect.Signature(args, return_annotation=result_type)


TorchTensors = Union[torch.Tensor, Sequence[torch.Tensor]]

CompiledModelKey = tuple[tuple[mlir.Type, ...], tuple[mlir.Type, ...]]
ModelSignature = tuple[tuple[TensorType, ...], tuple[TensorType, ...]]


def model_signature(
    args: Iterable[torch.Tensor],
    results: TorchTensors,
) -> ModelSignature:
    results = (results,) if isinstance(results, torch.Tensor) else results

    input_types = tuple(torch_tensor_to_type(arg) for arg in args)
    result_types = tuple(torch_tensor_to_type(result) for result in results)
    return (input_types, result_types)


def model_key(
    context: mlir.Context,
    args: Iterable[torch.Tensor],
    results: TorchTensors,
) -> CompiledModelKey:
    sig_args, sig_results = model_signature(args, results)

    with context:
        input_types = tuple(arg.to_mlir() for arg in sig_args)
        result_types = tuple(result.to_mlir() for result in sig_results)
    return (input_types, result_types)


def register_custom_op(op: CustomOp, name: Optional[str] = None):
    # This will hold the compiled model once the registered fake tensor function
    # is invoked for the first time.
    # model_cache = op.operation_cache
    model_cache: dict[CompiledModelKey, Model] = {}

    # model: Optional[CompiledOperation] = None
    registered_fake: Optional[Callable[..., TorchTensors]] = None

    signature: inspect.Signature = op_signature(op.kernel)

    # Compile the model if it has not been compiled already.
    def compile_model(*args: torch.Tensor) -> TorchTensors:
        assert registered_fake is not None, (
            "Must register_fake for pytorch custom op before compiling"
        )
        results = registered_fake(*args)

        key = model_key(op.context, args, results)
        if key not in model_cache:
            sig = model_signature(args, results)
            graph = custom_op_graph(op, *sig)
            model = op.library._session.load(graph)
            model_cache[key] = model

        return results

    def custom_op(*args: torch.Tensor) -> TorchTensors:
        # In eager mode, the fake_tensor function will not be called,
        # so we call it here.
        assert registered_fake is not None, (
            "Must register_fake for pytorch custom op before invoking a "
            "custom op"
        )

        # registered_fake with real inputs will create buffers for the outputs
        outputs = compile_model(*args)

        model = model_cache[model_key(op.context, args, outputs)]

        dps = outputs if isinstance(outputs, tuple) else (outputs,)
        converted = map(Tensor.from_dlpack, (*dps, *args))
        model(*converted)
        return outputs

    custom_op.__signature__ = signature  # type: ignore
    name = name or f"max::torch.{op.name}"
    custom_op = torch.library.custom_op(name, custom_op, mutates_args=())

    @custom_op.register_fake  # type: ignore
    def fake_fn(*args: torch.Tensor):
        assert registered_fake is not None, (
            "Must register_fake for pytorch custom op before compiling"
        )
        return compile_model(*args)

    def register_fake(fn):
        nonlocal registered_fake
        registered_fake = fn
        return fn

    custom_op.register_fake = register_fake  # type: ignore
    return custom_op
