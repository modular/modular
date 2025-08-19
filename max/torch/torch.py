# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import inspect
import itertools
import threading
from collections.abc import Hashable, Iterable, Mapping, Sequence
from concurrent import futures
from functools import partial
from pathlib import Path
from typing import Callable, Union, overload

from max import mlir
from max._core import Attribute
from max._core.dialects import builtin, kgen
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine.api import InferenceSession, Model
from max.graph import (
    DeviceRef,
    Graph,
    KernelLibrary,
    Shape,
    TensorType,
    TensorValue,
    Type,
    Value,
    ops,
)

try:
    import torch  # type: ignore
    from torch._library.custom_ops import CustomOpDef  # type: ignore
except ImportError:
    raise ImportError(  # noqa: B904
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
                img_out: OutputTensor[dtype = DType.uint8, rank=2],
                img_in: InputTensor[dtype = DType.uint8, rank=3],
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

    For more information, see the [custom ops for PyTorch](/max/tutorials/custom-kernels-pytorch) tutorial.
    """

    _kernel_library: KernelLibrary
    _session: InferenceSession
    _ops: dict[str, CustomOp]
    _ops_lock: threading.Lock

    def __init__(self, kernel_library: Path | KernelLibrary) -> None:
        """
        Args:
            kernel_library: The path to a ``.mojo`` file or a ``.mojopkg`` with
              your custom op kernels, or the corresponding library object.
        """
        devices = [Accelerator(i) for i in range(accelerator_count())]

        if isinstance(kernel_library, KernelLibrary):
            self._kernel_library = kernel_library
        else:
            context = mlir.Context()
            self._kernel_library = KernelLibrary(context)
            self._kernel_library.load_paths(context, [kernel_library])

        self._session = InferenceSession(devices=devices)
        self._ops = {}
        self._ops_lock = threading.Lock()

    @property
    def _context(self) -> mlir.Context:
        return self._kernel_library._context

    def __getattr__(self, attr: str) -> CustomOp:
        """Get a custom op from the library registered with the given name."""
        compiled = self._ops
        with self._ops_lock:
            if not (result := compiled.get(attr)):

                @torch.compiler.disable
                def update_cache() -> None:
                    nonlocal result
                    if attr not in self._kernel_library:
                        raise AttributeError(
                            f"custom library does not register operation {attr}"
                        )

                    result = CustomOp(self, attr)
                    compiled[attr] = result

                update_cache()

        assert result is not None
        return result


ParametersDict = Mapping[str, Union[bool, int, str, DType]]
ParameterKey = tuple[str, Union[bool, int, str, DType]]

CompiledModelKey = tuple[Hashable, ...]


def _model_key(args: Iterable[TensorType]) -> CompiledModelKey:
    return tuple(arg.to_mlir() for arg in args)


class CustomOp:
    library: CustomOpLibrary
    name: str
    parameters: ParametersDict | None

    _custom_op_def: CustomOpDef
    _parameter_specializations: dict[tuple[ParameterKey, ...], CustomOp]

    def __init__(
        self,
        library: CustomOpLibrary,
        name: str,
        parameters: ParametersDict | None = None,
    ) -> None:
        self.library = library
        self.name = name
        self.parameters = parameters

        if parameters:
            suffix = "".join(
                f"{key}_{value}" for key, value in sorted(parameters.items())
            )
            name = f"{name}_{suffix}"
        op = MaxOp(self, name, self.library, num_outputs=self.num_outputs)
        self._custom_op_def = op.custom_op_def()

        self._parameter_specializations = {}

    def __getitem__(self, parameters: ParametersDict) -> CustomOp:
        """Parameterizes the custom op Mojo function with the given parameters."""
        if self.parameters is not None:
            raise TypeError("Parameters already specified")

        normalized = tuple(sorted(parameters.items()))

        if not (result := self._parameter_specializations.get(normalized)):
            result = CustomOp(self.library, self.name, parameters)
            self._parameter_specializations[normalized] = result

        return result

    def __call__(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> None:
        """Calls the PyTorch custom op with the given arguments."""
        return self._custom_op_def(*args, **kwargs)

    @property
    def kernel(self) -> kgen.GeneratorOp:
        """Retrieves the op definition MLIR from the custom op library."""
        analysis = self.library._kernel_library._analysis
        return analysis.kernel(self.name)

    @property
    def num_outputs(self) -> int:
        """Returns the number of outputs of the custom op."""
        # TODO(GEX-2219): support non-dps outputs
        attrs = self.kernel.discardable_attributes
        num_dps_outputs = attrs["mogg.num_dps_outputs"]
        assert isinstance(num_dps_outputs, builtin.IntegerAttr)
        return num_dps_outputs.value

    def op(self, *args: TensorValue, result_types: Sequence[TensorType]):
        # Infer custom op device from inputs
        device = next(
            itertools.chain(
                (arg.type.device for arg in args),
                (type.device for type in result_types),
                (DeviceRef.CPU(0),),
            )
        )
        return ops.custom(
            self.name,
            device,
            args,
            out_types=result_types,
            parameters=self.parameters,
        )

    @property
    def torch_signature(self) -> inspect.Signature:
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

        op = self.kernel
        num_dps_outputs = self.num_outputs

        # TODO(GEX-2223): Expose more of MojoLibraryAnalysis so we don't need to
        # hard code MLIR attributes.
        arg_type_names = op.discardable_attributes["mogg.arg_type_names"]
        assert isinstance(arg_type_names, builtin.ArrayAttr)
        io_specs = _mogg_annotations(arg_type_names)

        # Validate all argument types are supported
        _validate_op_arg_types(io_specs, self.name)

        # Filter to keep only tensor-like types
        tensor_like_types = {
            "tensor_internal::ManagedTensorSlice",
            "stdlib::SIMD",
        }
        io_specs = [spec for spec in io_specs if spec in tensor_like_types]

        arg_src_names = op.discardable_attributes["mogg.arg_src_names"]
        assert isinstance(arg_src_names, builtin.ArrayAttr)
        arg_names = _mogg_annotations(arg_src_names)

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


class MaxOp:
    fn: Callable[..., Iterable[Value] | Value | None] | CustomOp
    name: str
    library: CustomOpLibrary
    # Specify explicit input types to eg. compile with symbolic dims
    input_types: Sequence[TensorType] | None = None
    output_types: Sequence[TensorType] | None = None
    num_outputs: int

    def __init__(
        self,
        fn: Callable[..., Iterable[Value] | Value | None] | CustomOp,
        name: str,
        library: CustomOpLibrary,
        # Specify explicit input types to eg. compile with symbolic dims
        input_types: Sequence[TensorType] | None = None,
        output_types: Sequence[TensorType] | None = None,
        num_outputs: int | None = None,
    ) -> None:
        if num_outputs is None:
            self.num_outputs = 1 if output_types is None else len(output_types)
        elif output_types is not None and len(output_types) != num_outputs:
            raise ValueError(f"{num_outputs=} does not match {output_types=}")
        else:
            self.num_outputs = num_outputs

        self.fn = fn
        self.name = name
        self.library = library
        self.input_types = input_types
        self.output_types = output_types

    @property
    def _context(self) -> mlir.Context:
        return self.library._context

    @property
    def torch_signature(self) -> inspect.Signature:
        """Computes the signature of the generated PyTorch custom op."""
        if isinstance(self.fn, CustomOp):
            return self.fn.torch_signature

        base_signature = inspect.signature(self.fn)
        dps_args = [
            inspect.Parameter(
                f"__out{i}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
            )
            for i in range(self.num_outputs)
        ]
        args = [
            param.replace(annotation=torch.Tensor)
            for param in base_signature.parameters.values()
        ]
        return inspect.Signature((*dps_args, *args), return_annotation=None)

    def graph(
        self,
        input_types: Sequence[TensorType],
        result_types: Sequence[TensorType],
    ) -> Graph:
        """Builds a MAX graph with the given input and output types.

        The created graph will be destination-passing-style; the output
        types will be converted to buffer types so that inputs may be
        passed as mutable arguments to the compiled model.

        Args:
            input_types: The input types of the graph.
            result_types: The output types of the graph.
                Required for destination-passing-style.

        Returns:
            A MAX graph that executes the op.
        """
        output_types = [t.as_buffer() for t in result_types]
        graph_types: list[Type] = [*output_types, *input_types]

        with Graph(
            self.name,
            input_types=graph_types,
            context=self._context,
            kernel_library=self.library._kernel_library,
        ) as graph:
            # Awkward design, there's probably a better way here
            fn: Callable[..., Iterable[Value] | Value | None] = (
                partial(self.fn.op, result_types=result_types)
                if isinstance(self.fn, CustomOp)
                else self.fn
            )
            args = (input.tensor for input in graph.inputs[self.num_outputs :])
            results = fn(*args)
            iterable_results: Iterable[Value]
            if isinstance(results, Value):
                iterable_results = [results]
            elif results is None:
                iterable_results = []
            else:
                iterable_results = results
            for input, result in zip(graph.inputs, iterable_results):
                input.buffer[...] = result.tensor

            graph.output()

        return graph

    def custom_op_def(self) -> CustomOpDef:
        """Builds and registers a PyTorch custom operation with this MAX graph."""
        # This will hold the compiled model once the registered fake tensor function
        # is invoked for the first time.
        model_cache: dict[CompiledModelKey, futures.Future[Model]] = {}
        model_cache_lock = threading.Lock()
        executor = futures.ThreadPoolExecutor()

        signature: inspect.Signature = self.torch_signature
        mutated_args = list(signature.parameters.keys())[: self.num_outputs]

        # Compile the model if it has not been compiled already.
        def compiled_model(*args: torch.Tensor) -> Model:
            arg_types = tuple(max_tensor_type(arg) for arg in args)
            # args are destination-passing-style
            output_types = self.output_types or arg_types[: self.num_outputs]
            input_types = self.input_types or arg_types[self.num_outputs :]

            # Creating types requires an MLIR context
            with self._context:
                key = _model_key((*input_types, *output_types))
            # Only one thread can schedule model compilation for a given key
            with model_cache_lock:
                if not (model_future := model_cache.get(key)):
                    graph = self.graph(input_types, output_types)
                    model_cache[key] = model_future = executor.submit(
                        self.library._session.load, graph
                    )
            return model_future.result()

        # ops always have no return type! they assign their results to mutable buffers
        def callable(*args: torch.Tensor) -> None:
            # In eager mode, the fake_tensor function will not be called,
            # so we call it here.
            # registered_fake with real inputs will create buffers for the outputs
            model = compiled_model(*args)

            # Torch `__dlpack__(stream=...)` has substantial overhead.
            # - Manually retrieving and syncing the stream drops dlpack marshalling
            #   from ~60us per tensor to ~15us per tensor.
            # - Further optimizations are possible. Moving more of this behavior
            #   into a single C++ ffi call can drop overhead to ~2us.
            # - Generally users shouldn't be putting this marshalling into their
            #   inner loop. Gains are much more substantial for larger graphs
            #   which can take advantage of MAX's automatic kernel fusion.
            def fast_from_dlpack(t: torch.Tensor):
                if t.device.type == "cuda":
                    stream = torch.cuda.current_stream(t.device).cuda_stream
                    device = max_device(t.device)
                    data = t.__dlpack__()
                    try:
                        return Tensor._from_dlpack(data, device, stream)
                    except Exception:
                        # This approach fails when passing the tensor across threads.
                        # Fall back to letting torch slowly sync streams.
                        return Tensor.from_dlpack(t)
                return Tensor.from_dlpack(t)

            converted = map(fast_from_dlpack, args)
            model(*converted)

        name = f"max::torch.{self.name}"
        callable.__signature__ = signature  # type: ignore
        return torch.library.custom_op(
            name, callable, mutates_args=mutated_args
        )


@overload
def graph_op(
    fn: Callable[..., Value | None],
    name: str | None = None,
    kernel_library: Path | KernelLibrary | None = None,
    input_types: Sequence[TensorType] | None = None,
    output_types: Sequence[TensorType] | None = None,
    num_outputs: int | None = None,
) -> CustomOpDef: ...


@overload
def graph_op(
    fn: None = None,
    name: str | None = None,
    kernel_library: Path | KernelLibrary | None = None,
    input_types: Sequence[TensorType] | None = None,
    output_types: Sequence[TensorType] | None = None,
    num_outputs: int | None = None,
) -> Callable[[Callable[..., Iterable[Value] | Value | None]], CustomOpDef]: ...


def graph_op(
    fn: Callable[..., Iterable[Value] | Value | None] | None = None,
    name: str | None = None,
    kernel_library: Path | KernelLibrary | None = None,
    input_types: Sequence[TensorType] | None = None,
    output_types: Sequence[TensorType] | None = None,
    num_outputs: int | None = None,
) -> (
    CustomOpDef
    | Callable[[Callable[..., Iterable[Value] | Value | None]], CustomOpDef]
):
    """A decorator to create PyTorch custom operations using MAX graph operations.

    This decorator allows you to define larger graphs using [MAX graph
    ops](/max/api/python/graph/ops) or the MAX :obj:`~max.nn` modules and
    call them with PyTorch tensors, or integrate them into PyTorch modules.
    These custom ops can be called eagerly, and support compilation with
    ``torch.compile`` and the Inductor backend.

    The resulting custom operation uses destination-passing style, where output
    tensors are passed as the first arguments and modified in-place. This
    allows PyTorch to manage the memory and streams of the output tensors.
    Tensors internal to the computation are managed via MAX's graph compiler
    and memory planning.

    The default behavior is to JIT-compile for the specific input and output
    shapes needed. If you are passing variable-sized inputs, for instance a
    batch size or sequence length which may take on many different values
    between calls, you should specify this dimension as a symbolic dimension
    through :obj:`input_types` and :obj:`output_types`. Otherwise you will
    end up compiling specialized graphs for each possible variation of
    inputs, which may use a lot of memory.

    If neither `output_types` nor `num_outputs` is specified, default to 1
    output.

    For example to create a functional-style PyTorch op backed by MAX:

    .. code-block:: python
        :caption: grayscale.py

        import torch
        import numpy as np
        import max.torch
        from max.dtype import DType
        from max.graph import ops

        @max.torch.graph_op
        def max_grayscale(pic: max.graph.TensorValue):
            scaled = pic.cast(DType.float32) * np.array([0.21, 0.71, 0.07])
            grayscaled = ops.sum(scaled, axis=-1).cast(pic.dtype)
            # max reductions don't remove the dimension, need to squeeze
            return ops.squeeze(grayscaled, axis=-1)

        @torch.compile
        def grayscale(pic: torch.Tensor):
            output = pic.new_empty(pic.shape[:-1])  # Remove color channel dimension
            max_grayscale(output, pic)  # Call as destination-passing style
            return output

        device = "cuda" if torch.cuda.is_available() else "cpu"
        img = (torch.rand(64, 64, 3, device=device) * 255).to(torch.uint8)
        result = grayscale(img)
        print(f"Input shape: {img.shape}")
        print(f"Output shape: {result.shape}")
        print("Grayscale conversion completed successfully!")

    Args:
        fn: The function to decorate. If None, returns a decorator.
        name: Optional name for the custom operation. Defaults to the function name.
        kernel_library: Optional kernel library to use for compilation. Useful
            for creating graphs with custom Mojo ops.
        input_types: Optional sequence of input tensor types for compilation.
            If None, types are inferred from runtime arguments.
        output_types: Optional sequence of output tensor types for compilation.
            If None, types are inferred from runtime arguments.
        num_outputs: The number of outputs of the graph. We need to know this ahead
            of time to register with PyTorch before we've compiled the final kernels.

    Returns:
        A PyTorch custom operation that can be called with torch.Tensor arguments.
    """

    def decorator(
        fn: Callable[..., Iterable[Value] | Value | None],
    ) -> CustomOpDef:
        library = kernel_library or KernelLibrary(mlir.Context())
        op = MaxOp(
            fn,
            name or fn.__name__,
            CustomOpLibrary(library),
            input_types=input_types,
            output_types=output_types,
            num_outputs=num_outputs,
        )
        return op.custom_op_def()

    return decorator if fn is None else decorator(fn)


###############################################################################

# Convert torch.Tensor to a TensorType

###############################################################################


def max_shape(shape: torch.Size) -> Shape:
    """Returns the equivalent MAX shape for a PyTorch shape."""
    return Shape([int(dim) for dim in shape])


def max_device_ref(device: torch.device) -> DeviceRef:
    """Returns the equivalent MAX graph device for a PyTorch device."""
    type = device.type
    index = device.index or 0
    if type == "cpu":
        return DeviceRef.CPU(index)
    elif type == "cuda":
        return DeviceRef.GPU(index)
    else:
        raise TypeError(f"Unable to convert {type} to a MAX device type.")


def max_device(device: torch.device) -> Device:
    """Returns the equivalent MAX device for a PyTorch device."""
    DeviceType = {"cuda": Accelerator, "cpu": CPU}[device.type]
    return DeviceType(device.index)


def max_tensor_type(tensor: torch.Tensor) -> TensorType:
    """Returns the equivalent MAX tensor type for a PyTorch tensor."""
    dtype = DType.from_torch(tensor.dtype)
    shape = max_shape(tensor.shape)
    device = max_device_ref(tensor.device)
    return TensorType(dtype, shape, device=device)


###############################################################################

# Tensor Conversions

###############################################################################


def _mogg_annotations(array_attr: builtin.ArrayAttr) -> list[str]:
    def string_value(x: Attribute) -> str:
        if isinstance(x, builtin.StringAttr):
            return x.value
        assert isinstance(x, builtin.UnitAttr)
        return ""

    return [string_value(s) for s in array_attr.value]


def _validate_op_arg_types(io_specs: list[str], op_name: str) -> None:
    """Validate that all argument types in a custom op are supported.

    Args:
        io_specs: List of type specifications from mogg.arg_type_names
        op_name: Name of the custom operation for error reporting

    Raises:
        ValueError: If any type is not supported (not a tensor type or
            implicitly handled type)
    """
    # The set of types that are legal tensor inputs.
    tensor_like_types = {
        "tensor_internal::ManagedTensorSlice",
        "stdlib::SIMD",
    }

    # Types that are ignored or implicitly supplied by the graph compiler.
    types_to_ignore = {
        "stdlib::DeviceContextPtr",
        "stdlib::DeviceContextPtrList",
        "stdlib::Error",
    }

    # Validate that all types are either tensor-like or can be ignored
    allowed_types = tensor_like_types | types_to_ignore
    for spec in io_specs:
        if spec and spec not in allowed_types:
            raise ValueError(
                f"Unsupported argument type '{spec}' in custom op '{op_name}'."
            )
