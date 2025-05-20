# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional, Union

from max import mlir
from max._core import Type as _Type
from max.driver import Accelerator, Tensor, accelerator_count
from max.dtype import DType, torch_to_max_type
from max.engine import Model
from max.engine.api import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    KernelLibrary,
    Shape,
    TensorType,
    Type,
    ops,
)
from max.mlir import Context

try:
    import torch
    from torch._library.custom_ops import CustomOpDef
except ImportError:
    raise ImportError(
        "torch not found - install `max[torch]` (if using pip/uv) or max-conda (if using magic/conda)"
    )


class CustomOpLibrary:
    """A PyTorch interface to custom operations implemented in Mojo.

    This API allows for easy passing of PyTorch data as
    ``torch.Tensor`` values to the corresponding custom op. ``CustomOpLibrary``
    handles the compilation of the Mojo custom ops and marshalling of data between
    PyTorch and the executable Mojo code.

    For example, consider a grayscale operation implemented in Mojo:

    .. code-block:: mojo
       :caption: my_library/grayscale.mojo

        @register("grayscale")
        struct Grayscale:
            @staticmethod
            fn execute[
                # The kind of device this is running on: "cpu" or "gpu"
                target: StaticString,
            ](
                img_out: OutputTensor[type = DType.uint8, rank=2],
                img_in: InputTensor[type = DType.uint8, rank=3],
                ctx: DeviceContextPtr,
            ) raises:
                ...

    You can then use ``CustomOpLibrary`` to invoke the Mojo operation like so:

    .. code-block:: python

        import torch
        from max.torch import CustomOpLibrary

        op_library = CustomOpLibrary("my_library")
        grayscale_op = op_library.grayscale

        def grayscale(pic: torch.Tensor) -> torch.Tensor:
            result = pic.new_empty(pic.shape[:-1])
            grayscale_op(result, pic)
            return result

        img = (torch.rand(64, 64, 3) * 255).to(torch.uint8)
        result = grayscale(img)

    The custom operation produced by ``op_library.<opname>`` will have the
    same interface as the backing Mojo operation. Each ``InputTensor`` or
    ``OutputTensor`` argument corresponds to a
    :code_link:`https://docs.pytorch.org/docs/stable/tensors.html#tensor-class-reference|torch.Tensor`
    value in Python. Each argument corresponding to an ``OutputTensor`` in the
    Mojo operation will be modified in-place.
    """

    _context: Context
    _kernel_library: KernelLibrary
    _session: InferenceSession
    _ops: dict[str, CustomOp]

    def __init__(self, kernel_library: Path | KernelLibrary):
        """
        Args:
            kernel_library: The path to a ``.mojo`` file or a ``.mojopkg`` with
              your custom op kernels, or the corresponding library object.
        """
        devices = [Accelerator(i) for i in range(accelerator_count())]

        if isinstance(kernel_library, KernelLibrary):
            self._context = kernel_library._context
            self._kernel_library = kernel_library
        else:
            self._context = mlir.Context()
            self._kernel_library = KernelLibrary(self._context)
            self._kernel_library.load_paths(self._context, [kernel_library])

        self._session = InferenceSession(devices=devices)
        self._ops = {}

    def __getattr__(self, attr: str) -> CustomOp:
        compiled = self._ops
        if not (result := compiled.get(attr)):

            @torch.compiler.disable
            def update_cache():
                nonlocal result
                result = CustomOp(self, attr)
                compiled[attr] = result

            update_cache()

        assert result is not None
        return result


ParametersDict = dict[str, Union[bool, int, str, DType]]
ParameterKey = tuple[str, Union[bool, int, str, DType]]


class CustomOp:
    library: CustomOpLibrary
    name: str
    parameters: Optional[ParametersDict]

    _custom_op_def: CustomOpDef
    _parameter_specializations: dict[tuple[ParameterKey, ...], CustomOp]

    def __init__(
        self,
        library: CustomOpLibrary,
        name: str,
        parameters: Optional[ParametersDict] = None,
    ):
        self.library = library
        self.name = name
        self.parameters = parameters
        self._parameter_specializations = {}
        self._custom_op_def = compile_custom_op(self)

    def __getitem__(self, parameters: ParametersDict) -> CustomOp:
        assert self.parameters is None, (
            "Only one set of parameters may be specified"
        )
        normalized = tuple(sorted(parameters.items()))

        if not (result := self._parameter_specializations.get(normalized)):
            result = CustomOp(self.library, self.name, parameters)
            self._parameter_specializations[normalized] = result

        return result

    def __call__(self, *args, **kwargs):
        return self._custom_op_def(*args, **kwargs)

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
    context: mlir.Context,
    op: CustomOp,
    input_types: Iterable[TensorType],
    result_types: Iterable[TensorType],
    parameters: ParametersDict,
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
    graph_types: list[Type] = [*output_types, *input_types]

    with Graph(
        op.name,
        input_types=graph_types,
        context=context,
        kernel_library=op.kernel_library,
    ) as graph:
        results = ops.custom(
            op.name,
            list(graph.inputs[len(output_types) :]),
            out_types=list(result_types),
            parameters=parameters,
        )
        for input, result in zip(graph.inputs, results):
            input.buffer[...] = result.tensor

        graph.output()

    return graph


def get_strings(attr: mlir.Attribute) -> list[str]:
    as_array = mlir.ArrayAttr(attr)
    return [mlir.StringAttr(s).value for s in as_array]


def num_outputs(op: mlir.Operation) -> int:
    return mlir.IntegerAttr(op.attributes["mogg.num_dps_outputs"]).value


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
    num_dps_outputs = num_outputs(op)

    # TODO(GEX-2223): Expose more of MojoLibraryAnalysis so we don't need to
    # hard code MLIR attributes.
    io_specs = get_strings(op.attributes["mogg.args_io_specs"])
    arg_names = get_strings(op.attributes["mogg.arg_src_names"])
    input_specs = io_specs[num_dps_outputs:]
    nargs = len(input_specs)
    args = [
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=torch.Tensor,
        )
        for name in arg_names[: nargs + num_dps_outputs]
    ]
    return inspect.Signature(args, return_annotation=None)


TorchTensors = Union[torch.Tensor, Sequence[torch.Tensor]]

CompiledModelKey = tuple[_Type, ...]
ModelSignature = tuple[tuple[TensorType, ...], tuple[TensorType, ...]]


def model_signature(
    args: Iterable[torch.Tensor],
    results: TorchTensors,
) -> ModelSignature:
    results = (results,) if isinstance(results, torch.Tensor) else results

    input_types = tuple(torch_tensor_to_type(arg) for arg in args)
    result_types = tuple(torch_tensor_to_type(result) for result in results)
    return (input_types, result_types)


def model_key(args: Iterable[torch.Tensor]) -> CompiledModelKey:
    return tuple(torch_tensor_to_type(arg).to_mlir() for arg in args)


def compile_custom_op(op: CustomOp) -> CustomOpDef:
    # This will hold the compiled model once the registered fake tensor function
    # is invoked for the first time.
    model_cache: dict[CompiledModelKey, Model] = {}

    signature: inspect.Signature = op_signature(op.kernel)

    num_dps_outputs = num_outputs(op.kernel)
    mutated_args = list(signature.parameters.keys())[:num_dps_outputs]

    parameters = op.parameters or {}

    # Compile the model if it has not been compiled already.
    def compile_model(*args: torch.Tensor) -> Model:
        key = model_key(args)

        if not (model := model_cache.get(key)):
            sig = model_signature(
                args[num_dps_outputs:], args[:num_dps_outputs]
            )
            graph = custom_op_graph(
                mlir.Context(), op, sig[0], sig[1], parameters
            )
            model = op.library._session.load(graph)
            model_cache[key] = model

        return model

    def callable(*args: torch.Tensor):
        # In eager mode, the fake_tensor function will not be called,
        # so we call it here.
        # registered_fake with real inputs will create buffers for the outputs
        model = compile_model(*args)
        converted = map(Tensor.from_dlpack, args)
        model(*converted)
        return None

    name = f"max::torch.{op.name}"
    callable.__signature__ = signature  # type: ignore
    custom_op = torch.library.custom_op(
        name, callable, mutates_args=mutated_args
    )

    return custom_op
