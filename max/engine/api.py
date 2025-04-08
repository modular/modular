# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""MAX Engine APIs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from enum import Enum, IntEnum, auto
from inspect import Parameter, Signature
from pathlib import Path
from typing import (
    Any,
    Optional,
    Union,
    cast,
)

import numpy as np
import numpy.typing as npt
from max._core.engine import FrameworkFormat as _FrameworkFormat
from max._core.engine import InferenceSession as _InferenceSession
from max._core.engine import Model as Model
from max._core.engine import MojoValue, PrintStyle
from max._core.engine import TensorData as _TensorData
from max._core.engine import TensorSpec as TensorSpec
from max._core.engine import TorchInputSpec as TorchInputSpec
from max._core.profiler import set_gpu_profiling_state
from max._core_types.driver import DLPackArray
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.profiler import Tracer, traced
from max.support.paths import (
    _build_mojo_source_package,
    is_mojo_source_package_path,
)

# Manually define dlpack compatible types since MyPy isn't aware that ndarray
# implements the protocol.
DLPackCompatible = Union[DLPackArray, npt.NDArray]
InputShape = Optional[list[Union[int, str, None]]]
CustomExtensionType = Union[str, Path, Any]
CustomExtensionsType = Union[list[CustomExtensionType], CustomExtensionType]
# Need to use tuple instead of Union to ensure that Python 3.9 support works.
ScalarType = (int, float, bool, np.generic)
InputType = Union[
    DLPackCompatible, Tensor, MojoValue, int, float, bool, np.generic
]


class GPUProfilingMode(str, Enum):
    """The supported modes for GPU profiling."""

    OFF = "off"
    ON = "on"
    DETAILED = "detailed"


def _raise_if_not_contiguous(x: InputType) -> None:
    should_raise = False
    if isinstance(x, bool):
        return
    elif _is_torch_tensor(x):
        # This code does not import torch, so we ignore the type checker here
        if not x.is_contiguous():  # type: ignore
            should_raise = True
    elif isinstance(x, np.ndarray) and not x.flags.c_contiguous:
        should_raise = True
    elif isinstance(x, Tensor) and not x.is_contiguous:
        should_raise = True
    if should_raise:
        raise ValueError(
            "Max does not currently support executing"
            " non-contiguous tensors. Before passing these"
            " tensors to Max, please make a contiguous copy of them"
            " using `.contiguous()` before feeding them into the"
            " `execute` or `load` APIs."
        )


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
                DType[str(value.dtype).removeprefix("torch.")],
            )
        if isinstance(value, np.ndarray):
            keep_referenced[value.ctypes.data] = value
            return _TensorData(
                value.ctypes.data,
                list(value.shape),
                DType[str(value.dtype)],
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


@traced
def _Model_execute(
    self: Model,
    *args: InputType,
    copy_inputs_to_device: bool = True,
) -> list[Tensor | MojoValue]:
    tracer = Tracer()
    input_impls: list[Union[Tensor, MojoValue]] = []

    input_idx = 0
    for idx, arg in enumerate(args):
        _raise_if_not_contiguous(arg)

        # Validate that input is one of supported types and convert if
        # necessary.
        if isinstance(arg, MojoValue):
            input_impls.append(arg)
            continue

        if isinstance(arg, Tensor):
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
            tracer.push(f"copy_inputs_to_device_{input_idx}")
            input_devices = self.input_devices
            if input_idx >= len(input_devices):
                raise ValueError(
                    "Number of inputs does not match expected number ("
                    f"{len(input_devices)}) for model"
                )
            tensor = tensor.to(input_devices[input_idx])
            input_idx = input_idx + 1
            tracer.pop()
        input_impls.append(tensor)
    results = self._execute_device_tensors(input_impls)

    processed_results: list[Tensor | MojoValue] = []
    for idx, result in enumerate(results):
        tracer.push(f"process_result_{idx}")
        # If the output is a MojoValue, we return it directly.
        if not isinstance(result, Tensor):
            processed_results.append(result)
            tracer.pop()
            continue
        processed_results.append(result)
        tracer.pop()
    return processed_results


def _Model_call(
    self: Model, *args: InputType, **kwargs: InputType
) -> list[Tensor | MojoValue]:
    bound = self.signature.bind(*args, **kwargs)
    return self.execute(*bound.arguments.values())


def _Model_execute_legacy(
    self: Model,
    **kwargs: Any,
) -> dict[str, Union[np.ndarray, dict, list, tuple]]:
    # Wrapping the tensors happens by recording their addresses, which does
    # not increase reference count, so we need to ensure the garbage
    # collector does not free them. Since numpy arrays are not hashable, we
    # do this with a dictionary with pointer keys.
    keep_referenced: dict[int, Any] = {}
    dtype_map = {spec.name: spec.dtype for spec in self.input_metadata}
    for input_name, input_value in kwargs.items():
        kwargs[input_name] = _map_execute_kwarg(
            input_value, dtype_map[input_name], keep_referenced
        )
    return self._execute(**kwargs)


def _Model_repr(self: Model) -> str:
    return f"Model(inputs={self.input_metadata})"


def _Model_signature(self: Model) -> Signature:
    """Get input signature for model."""
    parameters = [
        Parameter(input.name, Parameter.POSITIONAL_OR_KEYWORD)
        for input in self.input_metadata
    ]
    return Signature(parameters=parameters)


Model.execute = _Model_execute  # type: ignore[method-assign]
Model.__call__ = _Model_call  # type: ignore[method-assign]
Model.execute_legacy = _Model_execute_legacy  # type: ignore[method-assign]
Model.__repr__ = _Model_repr  # type: ignore[method-assign]
Model.signature = property(_Model_signature)  # type: ignore[assignment]


def _TensorSpec_str(self) -> str:
    if self.shape is not None:
        mlir_shape = [
            str(dim) if dim is not None else "-1" for dim in self.shape
        ]
        shape_str = "x".join(mlir_shape)
        return f"{shape_str}x{self.dtype.name}"
    else:
        return f"None x {self.dtype.name}"


def _TensorSpec_repr(self) -> str:
    return (
        f"TensorSpec(shape={self.shape}, dtype={self.dtype}, name={self.name})"
    )


TensorSpec.__str__ = _TensorSpec_str  # type: ignore[method-assign]
TensorSpec.__repr__ = _TensorSpec_repr  # type: ignore[method-assign]


def _TorchInputSpec_str(self) -> str:
    device_str = "" if len(self.device) == 0 else f" {self.device}"
    if self.shape is not None:
        mlir_shape = [
            str(dim) if isinstance(dim, int) else "-1" for dim in self.shape
        ]
        shape_str = "x".join(mlir_shape)
        return f"{shape_str}x{self.dtype.name}{device_str}"
    else:
        return f"None x {self.dtype.name}{device_str}"


def _TorchInputSpec_repr(self) -> str:
    return f"TorchInputSpec(shape={self.shape}, dtype={self.dtype}, device={self.device!r})"


TorchInputSpec.__str__ = _TorchInputSpec_str  # type: ignore[method-assign]
TorchInputSpec.__repr__ = _TorchInputSpec_repr  # type: ignore[method-assign]


def _is_torch_tensor(obj: Any) -> bool:
    """Checks if an object is a `torch.Tensor`."""
    t = type(obj)
    return t.__module__ == "torch" and t.__name__ == "Tensor"


def _is_torch_metadata_module(obj: Any) -> bool:
    """Checks if an object is an `TorchMetadata`."""
    return type(obj).__name__ == "TorchMetadata"


def _is_torchscript_module(obj: Any) -> bool:
    """Checks if an object is a `torch.jit.script.ScriptModule` or a compatible
    (sub)class thereof."""
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
    """Checks if an object is a `max._torch_mlir.Module`."""
    # Only check last submodule since the higher level modules in the hierarchy
    # may differ depending on where the mlir module is built
    return type(obj).__name__ == "Module" and type(obj).__module__.startswith(
        "max._torch_mlir"
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
    """Removes any static tensor type information from a torch.jit graph.
    Preserve device annotations."""
    import torch  # type: ignore

    def _remove_static_info_from_value(value):
        if (
            value.type().isSubtypeOf(torch._C.TensorType.get())
            and value.type().device()
        ):
            value.setType(
                torch._C.TensorType.get().with_device(value.type().device())
            )

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
        if is_mojo_source_package_path(Path(custom_extension)):
            # Builds the source directory into a .mojopkg file.
            return _build_mojo_source_package(Path(custom_extension))

        # Pass the path through as is.
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


class SplitKReductionPrecision(IntEnum):
    """Internal use."""

    ACCUM = auto()
    OUTPUT = auto()


class AssertLevel(str, Enum):
    """Internal use."""

    NONE = "none"
    WARN = "warn"
    SAFE = "safe"
    ALL = "all"


class LogLevel(str, Enum):
    """Internal use."""

    NOTSET = "notset"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class InferenceSession:
    """Manages an inference session in which you can load and run models.

    You need an instance of this to load a model as a :obj:`Model` object.
    For example:

    .. code-block:: python

        session = engine.InferenceSession()
        model_path = Path('bert-base-uncased')
        model = session.load(model_path)
    """

    _impl: _InferenceSession

    def __init__(
        self,
        num_threads: int | None = None,
        devices: Iterable[Device] = [CPU()],
        *,
        custom_extensions: CustomExtensionsType | None = None,
    ):
        """
        Args:
            num_threads: Number of threads to use for the inference session.
              This defaults to the number of physical cores on your machine.
            devices: A list of devices on which to run inference. Default is
              the host CPU only.
            custom_extensions: The extensions to load for the model.
              Supports paths to `.mojopkg` custom ops, `.so` custom op libraries
              for PyTorch and `.pt` torchscript files for torch metadata
              libraries. Supports :obj:`TorchMetadata` and
              :obj:`torch.jit.ScriptModule` objects for
              torch metadata libraries without serialization.
        """
        config: dict[str, Any] = {}
        self.num_threads = num_threads
        if num_threads:
            config["num_threads"] = num_threads
        config["devices"] = devices
        if custom_extensions is not None:
            config["custom_extensions"] = _process_custom_extensions_objects(
                custom_extensions
            )
        self._impl = _InferenceSession(config)

    def __repr__(self) -> str:
        if self.num_threads:
            return f"<modular engine InferenceSession(num_threads={self.num_threads})>"
        else:
            return "<modular engine InferenceSession>"

    def load(
        self,
        model: Union[str, Path, Any],
        *,
        custom_extensions: CustomExtensionsType | None = None,
        custom_ops_path: str | None = None,
        input_specs: list[TorchInputSpec] | None = None,
        weights_registry: Mapping[str, DLPackCompatible] | None = None,
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
        weights_registry_real: Mapping[str, DLPackCompatible] = (
            weights_registry or {}
        )

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
        if _is_max_graph(model):
            if "custom_extensions" not in options_dict:
                options_dict["custom_extensions"] = list()
            options_dict["custom_extensions"].extend(
                _process_custom_extensions_objects(model.kernel_libraries_paths)  # type: ignore
            )

        if input_specs is not None:
            options_dict["input_specs"] = input_specs

        if isinstance(model, (str, bytes)):
            model = Path(str(model))

        if isinstance(model, Path):
            _model = self._impl.compile_from_path(model, options_dict)
        elif _is_max_graph(model):
            options_dict["pipeline_name"] = model.name

            # TODO: if the model has been loaded from a serialized MLIR file, we don't have
            # the _weights attribute available to us
            if hasattr(model, "_weights"):
                for weight_name, weight in model._weights.items():
                    if weight_name not in weights_registry_real:
                        raise ValueError(
                            f"Weight '{weight_name}' is not in the weights registry."
                        )

                    registered_weight = weights_registry_real[weight_name]
                    expected_device = weight.value.device
                    if (
                        expected_device is None
                        or expected_device.device_type.value == "cpu"
                    ) != (
                        # 1 is the value of DLDeviceType::kDLCPU
                        registered_weight.__dlpack_device__()[0] == 1
                    ):
                        raise ValueError(
                            f"Mismatch in device type for weight '{weight_name}'."
                        )

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
                    "max._torch_mlir.ir.Module._CAPIPtr"
                )
                _model = self._impl.compile_from_object(
                    model, _FrameworkFormat.torch_mlir, options_dict
                )

            else:
                raise RuntimeError(
                    "The model is not a valid string path, Path object, "
                    " torch.jit.ScriptModule or MlirModule."
                )

        for weight_name, weight in weights_registry_real.items():
            try:
                _raise_if_not_contiguous(weight)
            except ValueError as e:
                raise ValueError(
                    f"Weight '{weight_name}' is not contiguous: {str(e)}"
                ) from e

        _model._load(weights_registry_real)
        return _model

    def _get_torch_custom_op_schemas(self):
        return self._impl._get_torch_custom_op_schemas()  # type: ignore

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
            style = cast(
                Union[str, PrintStyle], getattr(PrintStyle, style, style)
            )
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

    def set_split_k_reduction_precision(
        self, precision: str | SplitKReductionPrecision
    ):
        """Sets the accumulation precision for split k reductions in large matmuls."""
        if not isinstance(precision, SplitKReductionPrecision):
            try:
                precision = SplitKReductionPrecision[precision]
            except:
                msg = f"Invalid precision ({precision}). Please use one of: {[x.name for x in SplitKReductionPrecision]}"
                raise TypeError(msg)

        self._set_mojo_define("SPLITK_REDUCTION_SCHEME", precision)

    def set_mojo_log_level(self, level: str | LogLevel):
        """Sets the verbosity of mojo logging in the compiled model."""
        if not isinstance(level, LogLevel):
            try:
                level = LogLevel[level]
            except:
                msg = f"Invalid log level ({level}). Please use one of: {[x.name for x in LogLevel]}"
                raise TypeError(msg)

        self._set_mojo_define("LOGGING_LEVEL", level)

    def set_mojo_assert_level(self, level: str | AssertLevel):
        """Sets which mojo asserts are kept in the compiled model."""
        if not isinstance(level, AssertLevel):
            try:
                level = AssertLevel[level]
            except:
                msg = f"Invalid assert level ({level}). Please use one of: {[x.name for x in AssertLevel]}"
                raise TypeError(msg)

        self._set_mojo_define("ASSERT", level)

    def gpu_profiling(self, mode: GPUProfilingMode):
        """Enables end to end gpu profiling configuration."""
        if mode == GPUProfilingMode.OFF:
            return

        self._set_mojo_define("MODULAR_ENABLE_PROFILING", 1)
        self._set_mojo_define("MODULAR_ENABLE_GPU_PROFILING", 1)
        if mode == GPUProfilingMode.DETAILED:
            self._set_mojo_define("MODULAR_ENABLE_GPU_PROFILING_DETAILED", 1)

        set_gpu_profiling_state(mode.value)

    def _use_experimental_kernels(self, mode: str):
        """Enables experimental kernels."""
        if mode.lower() in ("false", "off", "no", "0"):
            return

        self._set_mojo_define("USE_EXPERIMENTAL_KERNELS", 1)

    def _dump_gpu_asm(self, option: bool | str | Path = True):
        """Enables dumping of gpu asm.

        Specifying a True would print the kernel output to screen, specifying a
        string or Path would write the kernel output to the specified path. If
        a path contains '%' it is replaced with a unique identifier for the
        kernel.
        """
        self._set_mojo_define("DUMP_GPU_ASM", str(option))

    def _dump_gpu_llvm(self, option: bool | str | Path = True):
        """Enables dumping of gpu llvm.

        Specifying a True would print the kernel output to screen, specifying a
        string or Path would write the kernel output to the specified path. If
        a path contains '%' it is replaced with a unique identifier for the
        kernel.
        """
        self._set_mojo_define("DUMP_GPU_LLVM", str(option))

    def _set_mojo_define(self, key: str, value: bool | int | str):
        """Enables overwriting of any mojo config directly."""
        self._impl.set_mojo_define(key, value)

    @property
    def stats_report(self) -> dict[str, Any]:
        """Metadata about model compilation (PyTorch only).

        Prints a list of "fallback ops", which are ops that could not be lowered
        to our internal dialect MO. Fallback ops have to be executed using the
        original framework (i.e. PyTorch), which makes the model much slower.
        This function is a good starting point for debugging model performance.
        """
        return json.loads(self._impl.stats_report)

    def reset_stats_report(self) -> None:
        """Clears all entries in `stats_report`."""
        self._impl.reset_stats_report()

    @property
    def devices(self) -> list[Device]:
        """A list of available devices."""
        return self._impl.devices
