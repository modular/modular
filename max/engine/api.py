# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import json
from inspect import Parameter, Signature
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import numpy as np
from max._driver import Tensor as _Tensor
from max._engine import FrameworkFormat as _FrameworkFormat
from max._engine import InferenceSession as _InferenceSession
from max._engine import Model as _Model
from max._engine import MojoValue, PrintStyle
from max._engine import TensorData as _TensorData
from max._engine import TensorSpec as _TensorSpec
from max._engine import TorchInputSpec as _TorchInputSpec
from max.driver import CPU, Device, DLPackArray, Tensor
from max.dtype import DType

InputShape = Optional[List[Union[int, str, None]]]
CustomExtensionType = Union[str, Path, Any]
CustomExtensionsType = Union[List[CustomExtensionType], CustomExtensionType]
TensorOrMojoType = Union[Tensor, MojoValue]
ExecResultType = Union[
    Dict[str, Union[np.ndarray, dict, list, tuple]], List[TensorOrMojoType]
]
# Need to use tuple instead of Union to ensure that Python 3.9 support works.
ScalarType = (int, float, bool, np.generic)
InputType = Union[Tensor, MojoValue, int, float, bool, np.generic]


def _map_execute_kwarg(
    input_value: Any, expected_dtype: DType, keep_referenced: dict[int, Any]
) -> Any:
    def _wrap_tensor(value: Any) -> _TensorData:
        # NOTE: this only works if the tensor/array is contiguous.
        if _is_torch_tensor(value):
            keep_referenced[value.data_ptr()] = value
            return _TensorData(
                value.data_ptr(),
                list(value.shape),
                DType[str(value.dtype).removeprefix("torch.")]._to(),
            )
        if isinstance(value, np.ndarray):
            keep_referenced[value.ctypes.data] = value
            return _TensorData(
                value.ctypes.data,
                list(value.shape),
                DType[str(value.dtype)]._to(),
            )
        # Just pass the value through if it's not a tensor/array.
        return value

    if expected_dtype == DType._unknown:
        # This currently indicates that the value expected by the model
        # internally is not a `M::Tensor`. We recursively try to wrap torch
        # tensors and np arrays, and pass other values as-is, since no metadata
        # is available to check the runtime values against.
        # TODO(MSDK-43): Introduce input specs for non-tensor inputs.

        def wrap_nested(value: Any) -> Any:
            """Traverse a potentially nested python data structure (e.g. lists,
            dictionaries, and tuples) containing `torch.tensor` and
            `numpy.ndarray`s leaf nodes and wrap them.
            """
            if isinstance(value, list):
                return [wrap_nested(v) for v in value]
            if isinstance(value, dict):
                return {
                    wrap_nested(k): wrap_nested(v) for k, v in value.items()
                }
            if isinstance(value, tuple):
                return tuple(wrap_nested(v) for v in value)
            return _wrap_tensor(value)

        return wrap_nested(input_value)

    if not isinstance(input_value, np.ndarray) and not _is_torch_tensor(
        input_value
    ):
        # Indicates that the model expects an ndarray (internally `M::Tensor`),
        # but if the input isn't already an ndarray, then we can attempt to
        # interpret it as a scalar primitive that needs to be converted to an
        # ndarray.
        return _wrap_tensor(np.array(input_value))
    return _wrap_tensor(input_value)


class Model:
    """A loaded model that you can execute.

    Do not instantiate this class directly. Instead, create it with
    :obj:`InferenceSession`.
    """

    _impl: _Model

    @classmethod
    def _init(cls, _core_model):
        model = cls()
        model._impl = _core_model
        return model

    def _export_mef(self, path):
        """Exports the compiled model as a mef to a file.

        Args:
          path: The filename where the mef is exported to.
        """
        self._impl._export_mef(path)

    def execute(
        self,
        *args: InputType,
        copy_inputs_to_device: bool = True,
        output_device: Device | None = None,
    ) -> list[TensorOrMojoType]:
        """Executes the model with the provided input and returns the outputs.

        For example, if the model has one input tensor:

        .. code-block:: python

            input_tensor = np.random.rand(1, 224, 224, 3)
            model.execute(input_tensor)

        Args:
            args:
              A list of input tensors. We currently support :obj:`np.ndarray`,
              :obj:`torch.Tensor`, and :obj:`max.driver.Tensor` inputs. All
              inputs will be copied to the device that the model is resident on
              prior to executing.

            copy_inputs_to_device:
              Whether to copy all input tensors to the model's device. Defaults
              to :obj:`True`. If set to :obj:`False`, input tensors will remain
              on whatever device they're currently on, which the model must be
              prepared for.

            output_device:
              The device to copy output tensors to. Defaults to :obj:`None`, in
              which case the tensors will remain resident on the same device as
              the model.

        Returns:
            A list of output tensors and Mojo values. The output tensors will be
            resident on the execution device by default.

        Raises:
            RuntimeError: If the given input tensors' shape don't match what
              the model expects.zzz

            TypeError: If the given input tensors' dtype cannot be cast to what
              the model expects.

            ValueError: If positional inputs are not one of the supported
              types, i.e. :obj:`np.ndarray`, :obj:`torch.Tensor`, and
              :obj:`max.driver.Tensor`.
        """
        input_impls: list[Union[_Tensor, MojoValue]] = []

        input_idx = 0
        for idx, arg in enumerate(args):
            # Validate that input is one of supported types and convert if
            # necessary.
            if isinstance(arg, MojoValue):
                input_impls.append(arg)
                continue

            if isinstance(arg, Tensor):
                if not arg.is_contiguous:
                    raise ValueError(
                        "Max does not currently support executing"
                        " non-contiguous tensors. Before executing these"
                        " tensors, please make a contiguous copy of them"
                        " using `.contiguous` before feeding them into the"
                        " `execute` API."
                    )
                tensor = arg
            elif isinstance(arg, DLPackArray):
                tensor = Tensor.from_dlpack(arg)
            elif isinstance(arg, ScalarType):
                spec = self.input_metadata[idx]
                tensor = Tensor.scalar(arg, spec.dtype, self.devices[0])
            else:
                raise ValueError(
                    "All positional arguments must be of the type"
                    " `max.driver.Tensor`, `MojoValue`, or a tensor type"
                    " implementing the dlpack protocol. We do not"
                    f" currently support inputs of the type {type(arg)}."
                )
            if copy_inputs_to_device:
                input_devices = self.input_devices
                if input_idx >= len(input_devices):
                    raise ValueError(
                        "Number of inputs does not match expected number ("
                        f"{len(input_devices)}) for model"
                    )
                tensor = tensor.to(input_devices[input_idx])
                input_idx = input_idx + 1
            input_impls.append(tensor._impl)
        results = self._impl.execute_device_tensors(input_impls)

        processed_results = []
        for idx, result in enumerate(results):
            # If the output is a MojoValue, we return it directly.
            if not isinstance(result, _Tensor):
                processed_results.append(result)
                continue
            wrapped_tensor = Tensor._from_impl(result)
            # If an output device is provided and it is different from the
            # device the tensor is already present on, we should copy to
            # that device.
            if output_device and output_device != self.output_devices[idx]:
                wrapped_tensor = wrapped_tensor.copy(output_device)
            processed_results.append(wrapped_tensor)
        return processed_results

    def __call__(
        self, *args: InputType, **kwargs: InputType
    ) -> list[TensorOrMojoType]:
        """Executes the model with the provided input and returns the outputs.

        Models can be called with any mixture of named and positional inputs:

        .. code-block:: python

            model(a, b, d=d, c=c)

        This function assumes that positional inputs cannot collide with any
        named inputs that would be present in the same position. If we have a
        model that takes named inputs `a`, `b`, `c`, and `d` (in that order),
        the following is invalid.

        .. code-block:: python
            model(a, d, b=b, c=c)

        The function will assume that input `d` will map to the same position as
        input `b`.

        Args:
            args: A list of input tensors. We currently support the following
              input types:
                * Any tensors implementing the dlpack protocol, such as
                  :obj:`np.ndarray`, :obj:`torch.Tensor`
                * Max Driver tensors, i.e. :obj:`max.driver.Tensor`
                * Scalar inputs, i.e. :obj:`bool`, :obj:`float`, :obj:`int`,
                  :obj:`np.generic`
                * Mojo value inputs, i.e. :obj:`MojoValue`

            kwargs: Named inputs. We can support the same types supported
              in :obj:`args`.

        Returns:
            A list of output tensors and Mojo values. The output tensors will be
            resident on the execution device by default.

        Raises:
            RuntimeError: If the given input tensors' shape don't match what
              the model expects.

            TypeError: If the given input tensors' dtype cannot be cast to
              what the model expects.

            ValueError: If positional inputs are not one of the supported
              types, i.e. :obj:`np.ndarray`, :obj:`torch.Tensor`, and
              :obj:`max.driver.Tensor`.

            ValueError: If an input name does not correspond to what the model
              expects.

            ValueError: If any positional and named inputs collide.

            ValueError: If the number of inputs is less than what the model
              expects.
        """
        bound = self.signature.bind(*args, **kwargs)
        return self.execute(*bound.arguments.values())

    def execute_legacy(
        self,
        **kwargs: Any,
    ) -> Dict[str, Union[np.ndarray, dict, list, tuple]]:
        """Executes the model with a set of named tensors. This API is maintained
        primarily to support frameworks that require named inputs (i.e. ONNX).

        NOTICE: This API does not support GPU inputs and is slated for
        deprecation.

        For example, if the model has one input tensor named `input0`:

        .. code-block:: python

            input_tensor = np.random.rand(1, 224, 224, 3)
            model.execute_legacy(input0=input_tensor)

        Args:
            kwargs: The input tensors, each specified with the appropriate
              tensor name as a keyword and its value as an :obj:`np.ndarray`.
              You can find the tensor names to use as keywords from
              :obj:`~Model.input_metadata`.

        Returns:
            A dictionary of output values, each as an :obj:`np.ndarray`,
            :obj:`Dict`, :obj:`List`, or :obj:`Tuple` identified by its output
            name.

        Raises:
            RuntimeError: If the given input tensors' name and shape don't
              match what the model expects.

            TypeError: If the given input tensors' dtype cannot be cast to what
              the model expects.

        """
        # Wrapping the tensors happens by recording their addresses, which does
        # not increase reference count, so we need to ensure the garbage
        # collector does not free them. Since numpy arrays are not hashable, we
        # do this with a dictionary with pointer keys.
        keep_referenced = dict()  # type: ignore
        dtype_map = {spec.name: spec.dtype for spec in self.input_metadata}
        for input_name, input_value in kwargs.items():
            kwargs[input_name] = _map_execute_kwarg(
                input_value, dtype_map[input_name], keep_referenced
            )
        return self._impl.execute(**kwargs)

    def __repr__(self) -> str:
        return f"Model(inputs={self.input_metadata})"

    @property
    def input_metadata(self) -> list[TensorSpec]:
        """
        Metadata about the model's input tensors, as a list of
        :obj:`TensorSpec` objects.

        For example, you can print the input tensor names, shapes, and dtypes:

        .. code-block:: python

            for tensor in model.input_metadata:
                print(f'name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')
        """
        return [TensorSpec._init(spec) for spec in self._impl.input_metadata]

    @property
    def output_metadata(self) -> list[TensorSpec]:
        """
        Metadata about the model's output tensors, as a list of
        :obj:`TensorSpec` objects.

        For example, you can print the output tensor names, shapes, and dtypes:

        .. code-block:: python

            for tensor in model.ouput_metadata:
                print(f'name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')
        """
        return [TensorSpec._init(spec) for spec in self._impl.output_metadata]

    @property
    def signature(self) -> Signature:
        """Get input signature for model."""
        parameters = [
            Parameter(input.name, Parameter.POSITIONAL_OR_KEYWORD)
            for input in self.input_metadata
        ]
        return Signature(parameters=parameters)

    @property
    def devices(self) -> list[Device]:
        """
        Returns the device objects used in the Model.
        """
        return [Device(device) for device in self._impl.devices]

    @property
    def input_devices(self) -> List[Device]:
        """
        Device of the model's input tensors, as a list of
        :obj:`Device` objects.
        """
        return [Device(device) for device in self._impl.input_devices]

    @property
    def output_devices(self) -> List[Device]:
        """
        Device of the model's output tensors, as a list of
        :obj:`Device` objects.
        """
        return [Device(device) for device in self._impl.output_devices]


class TensorSpec:
    """
    Defines the properties of a tensor, including its name, shape and data type.

    For usage examples, see :obj:`Model.input_metadata`.
    """

    _impl: _TensorSpec

    def __init__(self, shape: InputShape, dtype: DType, name: str):
        self._impl = _TensorSpec(shape, dtype._to(), name)

    @classmethod
    def _init(cls, _core_tensor_spec):
        tensor_spec = cls([], DType.bool, "")
        tensor_spec._impl = _core_tensor_spec
        return tensor_spec

    def __repr__(self) -> str:
        return (
            f"TensorSpec(shape={self.shape}, dtype={self.dtype},"
            f" name={self.name})"
        )

    def __str__(self) -> str:
        if self.shape is not None:
            mlir_shape = [
                str(dim) if dim is not None else "-1" for dim in self.shape
            ]
            shape_str = "x".join(mlir_shape)
            return f"{shape_str}x{self.dtype.name}"
        else:
            return f"None x {self.dtype.name}"

    @property
    def shape(self) -> list[int] | None:
        """The shape of the tensor as a list of integers.

        If a dimension size is unknown/dynamic (such as the batch size), its
        value is ``None``."""
        return self._impl.shape

    @property
    def dtype(self) -> DType:
        """A tensor data type."""
        return DType._from(self._impl.dtype)

    @property
    def name(self) -> str:
        """A tensor name."""
        return self._impl.name


class TorchInputSpec:
    """
    Specifies valid input specification for a TorchScript model.

    Before you load a TorchScript model, you must create an instance of this class
    for each input tensor, and pass them to the `input_specs` argument of
    :meth:`InferenceSession.load`.

    For example code, see :meth:`InferenceSession.load`.
    """

    _impl: _TorchInputSpec

    def __init__(self, shape: InputShape, dtype: DType):
        self._impl = _TorchInputSpec(shape, dtype._to())

    @classmethod
    def _init(cls, _core_torch_load_spec):
        torch_load_spec = cls([], DType.bool)
        torch_load_spec._impl = _core_torch_load_spec
        return torch_load_spec

    def __repr__(self) -> str:
        return f"TorchInputSpec(shape={self.shape}, dtype={self.dtype})"

    def __str__(self) -> str:
        if self.shape is not None:
            mlir_shape = [
                str(dim) if isinstance(dim, int) else "-1" for dim in self.shape
            ]
            shape_str = "x".join(mlir_shape)
            return f"{shape_str}x{self.dtype.name}"
        else:
            return f"None x {self.dtype.name}"

    @property
    def shape(self) -> InputShape:
        """The shape of the torch input tensor as a list of integers.

        If a dimension size is unknown/dynamic (such as the batch size), the
        `shape` should be ``None``."""
        return self._impl.shape

    @property
    def dtype(self) -> DType:
        """A torch input tensor data type."""
        return DType._from(self._impl.dtype)


def _unwrap_pybind_objects(value: Any) -> Any:
    """Unwraps pybind objects from python class wrappers."""
    if isinstance(value, list):
        return [_unwrap_pybind_objects(v) for v in value]
    if isinstance(value, dict):
        return {
            _unwrap_pybind_objects(k): _unwrap_pybind_objects(v)
            for k, v in value.items()
        }
    if isinstance(value, TensorSpec):
        # Unwrap TensorSpec to _TensorSpec.
        return value._impl
    if isinstance(value, TorchInputSpec):
        # Unwrap TorchInputSpec to _TorchInputSpec.
        return value._impl
    return value


def _is_torch_tensor(obj: Any) -> bool:
    """Checks if an object is a `torch.Tensor`."""
    t = type(obj)
    return t.__module__ == "torch" and t.__name__ == "Tensor"


def _is_torch_metadata_module(obj: Any) -> bool:
    """Checks if an object is an `TorchMetadata`."""
    return type(obj).__name__ == "TorchMetadata"


def _is_torchscript_module(obj: Any) -> bool:
    """Checks if an object is a `torch.jit.script.ScriptModule` or a compatible
    (sub)class thereof.
    """
    t = type(obj)
    return t.__name__ in [
        "ScriptModule",
        "RecursiveScriptModule",
    ] and t.__module__ in ["torch.jit.script", "torch.jit._script"]


def _is_torchscript_function(obj: Any) -> bool:
    """Checks if an object is a `torch.jit.ScriptFunction`."""
    t = type(obj)
    return t.__name__ == "ScriptFunction" and t.__module__ == "torch.jit"


def _is_torch_mlir_module(obj: Any) -> bool:
    """Checks if an object is a `modular.torch_mlir.Module`."""
    # Only check last submodule since the higher level modules in the hierarchy
    # may differ depending on where the mlir module is built
    return type(obj).__name__ == "Module" and type(obj).__module__.startswith(
        "modular.torch_mlir"
    )


def _is_max_graph(obj: Any) -> bool:
    """Checks if an object is `max.graph.Graph`."""

    # TODO(MSDK-677): We should use isinstance here once max.graph
    # is available in nightlies.
    object_kind = type(obj)
    return (
        object_kind.__name__ == "Graph"
        and object_kind.__module__.startswith("max.graph")
    )


def _remove_static_info_from_torch_jit_graph(graph: Any):
    """Removes any static tensor type information from a torch.jit graph."""
    import torch  # type: ignore

    def _remove_static_info_from_value(value):
        if value.type().isSubtypeOf(torch._C.TensorType.get()):
            value.setType(torch._C.TensorType.get())

    def _remove_static_info_from_node(node):
        for input in node.inputs():
            _remove_static_info_from_value(input)

        for output in node.outputs():
            _remove_static_info_from_value(output)

    # Apply this function to all nodes in the graph
    for node in graph.nodes():
        _remove_static_info_from_node(node)


def _process_custom_extensions_object(
    custom_extension: CustomExtensionType,
) -> CustomExtensionType:
    if isinstance(custom_extension, Path) or isinstance(custom_extension, str):
        return custom_extension
    if _is_torch_metadata_module(custom_extension):
        return custom_extension._get_jit_functions()._c
    if _is_torchscript_module(custom_extension):
        return custom_extension._c
    raise TypeError("Unsupported type for custom ops libraries.")


def _process_custom_extensions_objects(
    custom_extensions: CustomExtensionsType,
) -> CustomExtensionsType:
    if not isinstance(custom_extensions, Iterable) or isinstance(
        custom_extensions, str
    ):
        custom_extensions = [custom_extensions]
    return [
        _process_custom_extensions_object(custom_extension)
        for custom_extension in custom_extensions
    ]


class InferenceSession:
    """Manages an inference session in which you can load and run models.

    You need an instance of this to load a model as a :obj:`Model` object.
    For example:

    .. code-block:: python

        session = engine.InferenceSession()
        model_path = Path('bert-base-uncased')
        model = session.load(model_path)

    Args:
        num_threads: Number of threads to use for the inference session. This
          defaults to the number of physical cores on your machine.
    """

    _impl: _InferenceSession

    def __init__(
        self,
        num_threads: int | None = None,
        devices: list[Device] = [CPU()],
        *,
        custom_extensions: CustomExtensionsType | None = None,
    ):
        config: dict[str, Any] = {}
        self.num_threads = num_threads
        if num_threads:
            config["num_threads"] = num_threads
        config["devices"] = [device._device for device in devices]
        if custom_extensions is not None:
            config["custom_extensions"] = _process_custom_extensions_objects(
                custom_extensions
            )
        self._impl = _InferenceSession(config)

    def __repr__(self) -> str:
        if self.num_threads:
            return (
                "<modular engine"
                f" InferenceSession(num_threads={self.num_threads})>"
            )
        else:
            return "<modular engine InferenceSession>"

    def load(
        self,
        model: Union[str, Path, Any],
        *,
        custom_extensions: CustomExtensionsType | None = None,
        custom_ops_path: str | None = None,
        input_specs: list[TorchInputSpec] | None = None,
        weights_registry: dict[str, DLPackArray] | None = None,
    ) -> Model:
        """Loads a trained model and compiles it for inference.

        Note: PyTorch models must be in TorchScript format.

        Args:
            model: Path to a model, or a TorchScript model instance.
              May be a TorchScript model or an ONNX model.

            custom_extensions: The extensions to load for the model.
              Supports paths to `.mojopkg` custom ops, `.so` custom op libraries
              for PyTorch and `.pt` torchscript files for torch metadata
              libraries. Supports :obj:`TorchMetadata` and
              :obj:`torch.jit.ScriptModule` objects for
              torch metadata libraries without serialization.

            custom_ops_path: The path to your custom ops Mojo package.
              Deprecated, use ``custom_extensions`` instead.

            input_specs: The tensor specifications (shape and data type) for
              each of the model inputs. This is required when loading serialized
              TorchScript models because they do not include type and shape
              annotations.

              For example:

              .. code-block:: python

                  session = engine.InferenceSession()
                  model = session.load(
                      "clip-vit.torchscript",
                      input_specs = [
                          engine.TorchInputSpec(
                              shape=[1, 16], dtype=DType.int32
                          ),
                          engine.TorchInputSpec(
                              shape=[1, 3, 224, 224], dtype=DType.float32
                          ),
                          engine.TorchInputSpec(
                              shape=[1, 16], dtype=DType.int32
                          ),
                      ],
                  )

              If the model supports an input with dynamic shapes, use ``None``
              as the dimension size in ``shape``.

            weights_registry: A mapping from names of model weights' names to
              their values. The values are currently expected to be dlpack
              arrays. If an array is a read-only numpy array, the user must
              ensure that its lifetime extends beyond the lifetime of the model.

        Returns:
            The loaded model, compiled and ready to execute.

        Raises:
            RuntimeError: If the path provided is invalid.
        """

        options_dict: dict[str, Any] = {}

        if custom_extensions is not None:
            options_dict["custom_extensions"] = (
                _process_custom_extensions_objects(custom_extensions)
            )
        if custom_ops_path is not None:
            if "custom_extensions" not in options_dict:
                options_dict["custom_extensions"] = list()
            options_dict["custom_extensions"].extend(
                _process_custom_extensions_objects(custom_ops_path)
            )
        if input_specs is not None:
            options_dict["input_specs"] = _unwrap_pybind_objects(input_specs)

        if isinstance(model, (str, bytes)):
            model = Path(str(model))

        if isinstance(model, Path) or isinstance(model, str):
            model_path = Path(str(model))
            _model = self._impl.compile_from_path(model_path, options_dict)
        elif _is_max_graph(model):
            options_dict["pipeline_name"] = model.name
            _model = self._impl.compile_from_object(
                model._module._CAPIPtr, _FrameworkFormat.max_graph, options_dict
            )
        else:
            if _is_torchscript_module(model):
                _remove_static_info_from_torch_jit_graph(model.graph)
                _model = self._impl.compile_from_object(
                    model._c, _FrameworkFormat.torchscript_module, options_dict
                )
            elif _is_torchscript_function(model):
                _remove_static_info_from_torch_jit_graph(model.graph)
                _model = self._impl.compile_from_object(
                    model, _FrameworkFormat.torchscript_function, options_dict
                )
            elif _is_torch_mlir_module(model):
                options_dict["_mlir_module_capsule_name"] = (
                    "modular.torch_mlir.ir.Module._CAPIPtr"
                )
                _model = self._impl.compile_from_object(
                    model, _FrameworkFormat.torch_mlir, options_dict
                )

            else:
                raise RuntimeError(
                    "The model is not a valid string path, Path object, "
                    " torch.jit.ScriptModule or MlirModule."
                )

        _model.load(weights_registry if weights_registry else {})
        return Model._init(_model)

    def _get_torch_custom_op_schemas(self):
        return self._impl._get_torch_custom_op_schemas()

    def set_debug_print_options(
        self,
        style: Union[str, PrintStyle] = PrintStyle.COMPACT,
        precision: int = 6,
        output_directory: str = "",
    ):
        """Sets the debug print options.

        See `Value.print`.

        This affects debug printing across all model execution using the same
        InferenceSession.

        Tensors saved with `BINARY` can be loaded using
        `max.driver.MemmapTensor()`, but you will have to provide the expected
        dtype and shape.

        Tensors saved with `BINARY_MAX_CHECKPOINT` are saved with the shape and
        dtype information, and can be loaded with
        `max.driver.tensor.load_max_tensor()`.

        Warning: Even with style set to `NONE`, debug print ops in the graph can
        stop optimizations. If you see performance issues, try fully removing
        debug print ops.

        Args:
            style: How the values will be printed. Can be `COMPACT`, `FULL`,
                `BINARY`, `BINARY_MAX_CHECKPOINT` or `NONE`.
            precision: If the style is `FULL`, the digits of precision in the
                output.
            output_directory: If the style is `BINARY`, the directory to store
                output tensors.
        """
        if isinstance(style, str):
            style = getattr(PrintStyle, style, None)  # type: ignore
        if not isinstance(style, PrintStyle):
            raise TypeError(
                "Invalid debug print style. Please use one of 'COMPACT',"
                " 'FULL', 'BINARY', 'BINARY_MAX_CHECKPOINT', or 'NONE'."
            )
        if style == PrintStyle.FULL and not isinstance(precision, int):
            raise TypeError("Debug print precision must be an int.")
        if style in (PrintStyle.BINARY, PrintStyle.BINARY_MAX_CHECKPOINT):
            if isinstance(output_directory, str):
                pass
            elif isinstance(output_directory, Path):
                output_directory = str(output_directory)
            else:
                raise TypeError("Debug print output directory must be a str.")

            if not output_directory:
                raise ValueError(
                    "Debug print output directory cannot be empty."
                )
        self._impl.set_debug_print_options(style, precision, output_directory)

    @property
    def stats_report(self) -> Dict[str, Any]:
        """
        Metadata about model compilation (PyTorch only).

        Prints a list of "fallback ops", which are ops that could not be lowered
        to our internal dialect MO. Fallback ops have to be executed using the
        original framework (i.e. PyTorch), which makes the model much slower.
        This function is a good starting point for debugging model performance.
        """
        return json.loads(self._impl.stats_report)


def remove_annotations(cls: Type) -> Type:
    del cls.__annotations__
    return cls


remove_annotations(Model)
remove_annotations(InferenceSession)
remove_annotations(TensorSpec)
remove_annotations(TorchInputSpec)
