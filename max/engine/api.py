# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""MAX Engine APIs."""

from __future__ import annotations

import faulthandler
import os
import signal
import sys
import threading
from collections.abc import Iterable, Mapping
from enum import Enum, IntEnum, auto
from inspect import Parameter, Signature
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
from max._core.engine import InferenceSession as _InferenceSession
from max._core.engine import Model as Model
from max._core.engine import MojoValue, PrintStyle
from max._core.engine import TensorSpec as TensorSpec
from max._core.profiler import set_gpu_profiling_state
from max.driver import Device, DLPackArray, Tensor
from max.profiler import traced
from mojo.paths import _build_mojo_source_package, is_mojo_source_package_path

# Manually define dlpack compatible types since MyPy isn't aware that ndarray

# implements the protocol

InputShape = Optional[list[Union[int, str, None]]]
CustomExtensionType = Union[str, Path, Any]
CustomExtensionsType = Union[list[CustomExtensionType], CustomExtensionType]

# Need to use tuple instead of Union to ensure that Python 3.9 support works

ScalarType = (int, float, bool, np.generic)
InputType = Union[DLPackArray, Tensor, MojoValue, int, float, bool, np.generic]


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


@traced
def _Model_execute(self: Model, *args: InputType) -> list[Tensor | MojoValue]:
    # Original tensor-only execution path
    input_impls: list[Union[Tensor, MojoValue]] = []

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
            tensor = Tensor.scalar(arg, spec.dtype, self.input_devices[idx])
        else:
            raise ValueError(
                "All positional arguments must be of the type"
                " `max.driver.Tensor`, `MojoValue`, or a tensor type"
                " implementing the dlpack protocol. We do not"
                f" currently support inputs of the type {type(arg)}."
            )

        input_impls.append(tensor)
    return self._execute_device_tensors(input_impls)


def _Model_call(
    self: Model, *args: InputType, **kwargs: InputType
) -> list[Tensor | MojoValue]:
    bound = self.signature.bind(*args, **kwargs)
    return self.execute(*bound.arguments.values())


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
Model.__repr__ = _Model_repr  # type: ignore[method-assign]
Model.signature = property(_Model_signature)  # type: ignore[assignment]


def _TensorSpec_str(self: TensorSpec) -> str:
    if self.shape is not None:
        mlir_shape = [
            str(dim) if dim is not None else "-1" for dim in self.shape
        ]
        shape_str = "x".join(mlir_shape)
        return f"{shape_str}x{self.dtype.name}"
    else:
        return f"None x {self.dtype.name}"


def _TensorSpec_repr(self: TensorSpec) -> str:
    return (
        f"TensorSpec(shape={self.shape}, dtype={self.dtype}, name={self.name})"
    )


TensorSpec.__str__ = _TensorSpec_str  # type: ignore[method-assign]
TensorSpec.__repr__ = _TensorSpec_repr  # type: ignore[method-assign]


def _is_torch_tensor(obj: Any) -> bool:
    """Checks if an object is a `torch.Tensor`."""
    t = type(obj)
    return t.__module__ == "torch" and t.__name__ == "Tensor"


def _is_torch_metadata_module(obj: Any) -> bool:
    """Checks if an object is an `TorchMetadata`."""
    return type(obj).__name__ == "TorchMetadata"


def _is_max_graph(obj: Any) -> bool:
    """Checks if an object is `max.graph.Graph`."""
    # TODO(MSDK-677): We should use isinstance here once max.graph
    # is available in nightlies.
    object_kind = type(obj)
    return (
        object_kind.__name__ == "Graph"
        and object_kind.__module__.startswith("max.graph")
    )


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


class PdlLevel(IntEnum):
    """Internal use."""

    # No PDL
    OFF = auto()

    # Start subsequent kernel at the end
    # Apply to elementwise and reduction kernels
    OVERLAP_AT_END = auto()

    # Start subsequent kernel at the beginning
    # Apply to elementwise and reduction kernels
    OVERLAP_AT_BEGINNING = auto()


class AssertLevel(str, Enum):
    """The AssertLevel specifies the assert level used by the Mojo Ops."""

    NONE = "none"
    WARN = "warn"
    SAFE = "safe"
    ALL = "all"


class LogLevel(str, Enum):
    """The LogLevel specifies the log level used by the Mojo Ops."""

    NOTSET = "notset"
    TRACE = "trace"
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

        session = engine.InferenceSession(devices=[CPU()])
        model_path = Path('bert-base-uncased')
        model = session.load(model_path)
    """

    _impl: _InferenceSession
    # This is shared across sessions. Compilation is currently not thread safe.
    _compilation_lock = threading.Lock()

    def __init__(
        self,
        devices: Iterable[Device],
        num_threads: int | None = None,
        *,
        custom_extensions: CustomExtensionsType | None = None,
    ) -> None:
        """Construct an inference session.

        Args:
            num_threads: Number of threads to use for the inference session.
              This defaults to the number of physical cores on your machine.
            devices: A list of devices on which to run inference. Default is
              the host CPU only.
            custom_extensions: The extensions to load for the model.
              Supports paths to a `.mojopkg` custom ops library or a `.mojo`
              source file.
        """
        config: dict[str, Any] = {}
        self.num_threads = num_threads
        if num_threads:
            config["num_threads"] = num_threads

        # Process the provided iterable `devices`.
        final_devices: list[Device] = []
        seen_devices: set[Device] = set()
        for device in devices:
            if device not in seen_devices:
                final_devices.append(device)
                seen_devices.add(device)
        # If the user provided an empty iterable, final_devices remains empty.

        # Assign the ordered, unique list to the config.
        config["devices"] = final_devices

        if custom_extensions is not None:
            config["custom_extensions"] = _process_custom_extensions_objects(
                custom_extensions
            )
        self._impl = _InferenceSession(config)

        # Register async-safe Python stack trace handler
        # This enables Python stack traces in crash reports without GIL deadlocks
        try:
            faulthandler.register(
                signal.SIGUSR2, file=sys.stderr, all_threads=True, chain=False
            )
        except (OSError, RuntimeError):
            # Ignore errors if SIGUSR2 is already registered or unavailable
            pass

        if env_val := os.getenv("MOJO_LOGGING_LEVEL"):
            self.set_mojo_log_level(env_val)

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
        weights_registry: Mapping[str, DLPackArray] | None = None,
    ) -> Model:
        """Loads a trained model and compiles it for inference.

        Args:
            model: Path to a model.

            custom_extensions: The extensions to load for the model.
              Supports paths to `.mojopkg` custom ops.

            custom_ops_path: The path to your custom ops Mojo package.
              Deprecated, use ``custom_extensions`` instead.

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
        weights_registry_real: Mapping[str, DLPackArray] = (
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
                            f"Mismatch in device type for weight '{weight_name}'. Expected {expected_device} but weight is {registered_weight}"
                        )

            with self._compilation_lock:
                _model = self._impl.compile_from_object(
                    model._module._CAPIPtr,
                    options_dict,
                )
        else:
            raise RuntimeError("The model is not a valid path or module.")

        for weight_name, weight in weights_registry_real.items():
            try:
                _raise_if_not_contiguous(weight)
            except ValueError as e:
                raise ValueError(
                    f"Weight '{weight_name}' is not contiguous: {str(e)}"
                ) from e

        _model._load(weights_registry_real)
        return _model

    def set_debug_print_options(
        self,
        style: Union[str, PrintStyle] = PrintStyle.COMPACT,
        precision: int = 6,
        output_directory: str | Path | None = None,
    ) -> None:
        """Sets the debug print options.

        See `Value.print`.

        This affects debug printing across all model execution using the same
        InferenceSession.

        Tensors saved with `BINARY` can be loaded using
        `max.driver.Tensor.mmap()`, but you will have to provide the expected
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
            if output_directory is None:
                output_directory = ""
            elif isinstance(output_directory, str):
                pass
            elif isinstance(output_directory, Path):
                output_directory = str(output_directory)
            else:
                raise TypeError(
                    "Debug print output directory must be a str or Path."
                )

            if not output_directory:
                raise ValueError(
                    "Debug print output directory cannot be empty."
                )
        else:
            output_directory = ""
        self._impl.set_debug_print_options(style, precision, output_directory)

    def set_split_k_reduction_precision(
        self, precision: str | SplitKReductionPrecision
    ) -> None:
        """Sets the accumulation precision for split k reductions in large matmuls."""
        if not isinstance(precision, SplitKReductionPrecision):
            try:
                precision = SplitKReductionPrecision[precision]
            except:
                msg = f"Invalid precision ({precision}). Please use one of: {[x.name for x in SplitKReductionPrecision]}"
                raise TypeError(msg)  # noqa: B904

        self._set_mojo_define("SPLITK_REDUCTION_SCHEME", precision)

    def set_mojo_log_level(self, level: str | LogLevel) -> None:
        """Sets the verbosity of mojo logging in the compiled model."""
        if not isinstance(level, LogLevel):
            try:
                level = LogLevel[level]
            except:
                msg = f"Invalid log level ({level}). Please use one of: {[x.name for x in LogLevel]}"
                raise TypeError(msg)  # noqa: B904

        self._set_mojo_define("LOGGING_LEVEL", level)

    def set_mojo_assert_level(self, level: str | AssertLevel) -> None:
        """Sets which mojo asserts are kept in the compiled model."""
        if not isinstance(level, AssertLevel):
            try:
                level = AssertLevel[level]
            except:
                msg = f"Invalid assert level ({level}). Please use one of: {[x.name for x in AssertLevel]}"
                raise TypeError(msg)  # noqa: B904

        self._set_mojo_define("ASSERT", level)

    def gpu_profiling(self, mode: GPUProfilingMode) -> None:
        """Enables end to end gpu profiling configuration."""
        if mode == GPUProfilingMode.OFF:
            return

        self._set_mojo_define("MODULAR_ENABLE_PROFILING", 1)
        self._set_mojo_define("MODULAR_ENABLE_GPU_PROFILING", 1)
        if mode == GPUProfilingMode.DETAILED:
            self._set_mojo_define("MODULAR_ENABLE_GPU_PROFILING_DETAILED", 1)

        set_gpu_profiling_state(mode.value)

    def _use_experimental_kernels(self, mode: str) -> None:
        """Enables experimental kernels."""
        if mode.lower() in ("false", "off", "no", "0"):
            return

        self._set_mojo_define("USE_EXPERIMENTAL_KERNELS", 1)

    def _pdl_level(self, level: str | PdlLevel) -> None:
        """Level of overlap of kernel launch."""
        if not isinstance(level, PdlLevel):
            if level not in {"0", "1", "2"}:
                msg = f"Invalid pdl level ({level}). Please use one of: {[0, 1, 2]} corresponding to {[x.name for x in PdlLevel]}"
                raise TypeError(msg)

        self._set_mojo_define("PDL_LEVEL", int(level))

    def _dump_gpu_asm(self, option: bool | str | Path = True) -> None:
        """Enables dumping of gpu asm.

        Specifying a True would print the kernel output to screen, specifying a
        string or Path would write the kernel output to the specified path. If
        a path contains '%' it is replaced with a unique identifier for the
        kernel.
        """
        self._set_mojo_define("DUMP_GPU_ASM", str(option))

    def _dump_gpu_llvm(self, option: bool | str | Path = True) -> None:
        """Enables dumping of gpu llvm.

        Specifying a True would print the kernel output to screen, specifying a
        string or Path would write the kernel output to the specified path. If
        a path contains '%' it is replaced with a unique identifier for the
        kernel.
        """
        self._set_mojo_define("DUMP_GPU_LLVM", str(option))

    def _set_mojo_define(self, key: str, value: bool | int | str) -> None:
        """Enables overwriting of any mojo config directly."""
        self._impl.set_mojo_define(key, value)

    @property
    def devices(self) -> list[Device]:
        """A list of available devices."""
        return self._impl.devices
