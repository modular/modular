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
"""MAX Engine APIs."""

from __future__ import annotations

import faulthandler
import os
import signal
import sys
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum, IntEnum, auto
from inspect import Parameter, Signature
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast
from unittest import mock

import numpy as np
from max._core.engine import CompiledModels as _CompiledModels
from max._core.engine import DebugConfig as DebugConfig
from max._core.engine import InferenceSession as _InferenceSession
from max._core.engine import Model as Model
from max._core.engine import ModelMetadata as ModelMetadata
from max._core.engine import PrintStyle
from max._core.engine import TensorSpec as TensorSpec
from max._core.mlrt import AsyncValue as _AsyncValue
from max._core.profiler import (
    kineto_disable as _kineto_disable,
)
from max._core.profiler import (
    kineto_enable as _kineto_enable,
)
from max._core.profiler import (
    kineto_is_enabled as _kineto_is_enabled,
)
from max._core.profiler import (
    kineto_state as _kineto_state,
)
from max._core.profiler import (
    kineto_wait_for_trace as _kineto_wait_for_trace,
)
from max._core.profiler import (
    set_gpu_profiling_state,
)
from max.driver import CPU, Buffer, Device, DLPackArray, is_virtual_device_mode
from max.engine._compilation_stats import _record_phase
from max.graph import Graph, Module
from max.profiler import traced
from mojo.paths import _build_mojo_source_package, is_mojo_source_package_path

# Manually define dlpack compatible types since MyPy isn't aware that ndarray

# implements the protocol

InputShape = list[int | str | None] | None
CustomExtensionType = str | Path
CustomExtensionsType = Sequence[CustomExtensionType] | CustomExtensionType
"""Specifies one or more custom extension libraries to load with an
:class:`InferenceSession`.

It may be a single path or a sequence of paths, where each path is a ``str``
or :class:`~pathlib.Path` pointing to a compiled ``.mojoc``/``.mojopkg`` custom ops
library or a ``.mojo`` source file. When a ``.mojo`` source path is provided,
it's automatically compiled into a package before loading.
"""

# Need to use tuple instead of Union to ensure that Python 3.9 support works

ScalarType = (int, float, bool, np.generic)
InputType = DLPackArray | Buffer | int | float | bool | np.generic


GPUProfilingMode = Literal["off", "on", "detailed"]
"""The supported modes for GPU profiling.

GPU profiling modes control the level of instrumentation when profiling
MAX applications with NVIDIA Nsight Systems or Nsight Compute. Higher
levels provide more detail but may introduce additional overhead.

- ``off``: Disable GPU profiling instrumentation. This is the default
  mode and incurs no profiling overhead.
- ``on``: Enable basic GPU profiling. Adds CUDA driver calls and NVTX
  markers for correlating kernel executions with host-side code.
- ``detailed``: Enable detailed GPU profiling with additional NVTX
  markers from Python code. This mode provides the most visibility into
  which Python operations correspond to which GPU kernels, but has the
  highest overhead.

See Also:
    :meth:`InferenceSession.gpu_profiling`: Method to set the profiling mode.
"""


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
    elif isinstance(x, Buffer) and not x.is_contiguous:
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
def _Model_execute(self: Model, *args: InputType) -> list[Buffer]:
    # Original tensor-only execution path
    input_impls: list[Buffer] = []

    for idx, arg in enumerate(args):
        _raise_if_not_contiguous(arg)

        # Validate that input is one of supported types and convert if
        # necessary.
        if isinstance(arg, Buffer):
            buffer = arg
        elif isinstance(arg, DLPackArray):
            buffer = Buffer.from_dlpack(arg)
        elif isinstance(arg, ScalarType):
            spec = self.input_metadata[idx]
            buffer = Buffer.scalar(arg, spec.dtype, self.input_devices[idx])
        else:
            raise ValueError(
                "All positional arguments must be of the type"
                " `max.driver.Buffer` or a tensor type"
                " implementing the dlpack protocol. We do not"
                f" currently support inputs of the type {type(arg)}."
            )

        input_impls.append(buffer)
    return self._execute_device_tensors(input_impls)


def _Model_call(
    self: Model, *args: InputType, **kwargs: InputType
) -> list[Buffer]:
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


def _normalize_graph_key(graph_key: int) -> int:
    if isinstance(graph_key, bool) or not isinstance(graph_key, int):
        raise TypeError("graph_key must be an int.")
    if graph_key < 0 or graph_key > 2**64 - 1:
        raise ValueError("graph_key must be in range [0, 2^64 - 1].")
    return graph_key


def _normalize_graph_keys(graph_keys: int | Sequence[int]) -> list[int]:
    if isinstance(graph_keys, bool):
        raise TypeError("graph_keys must be an int or sequence of ints.")
    if isinstance(graph_keys, int):
        return [_normalize_graph_key(graph_keys)]
    if isinstance(graph_keys, (str, bytes)):
        raise TypeError("graph_keys must be a sequence of ints.")
    normalized: list[int] = []
    for graph_key in graph_keys:
        normalized.append(_normalize_graph_key(graph_key))
    if not normalized:
        raise ValueError("graph_keys must not be empty.")
    return normalized


def _Model_capture(
    self: Model, graph_keys: int | Sequence[int], *inputs: Buffer
) -> list[Buffer]:
    """Capture execution into a device graph for caller-provided key.

    Capture is best-effort and model-dependent. If the model issues
    capture-unsafe operations (for example, host-device synchronization),
    graph capture may fail. Callers should choose capture-safe execution paths.
    """
    if not inputs:
        raise ValueError("Model.capture requires input buffers.")
    normalized_keys = _normalize_graph_keys(graph_keys)
    return self._capture(normalized_keys, list(inputs))


def _Model_replay(
    self: Model, graph_keys: int | Sequence[int], *inputs: Buffer
) -> None:
    """Replay the captured device graph for a caller-provided key."""
    if not inputs:
        raise ValueError("Model.replay requires input buffers.")
    normalized_keys = _normalize_graph_keys(graph_keys)
    self._replay(normalized_keys, list(inputs))


def _Model_debug_verify_replay(
    self: Model, graph_keys: int | Sequence[int], *inputs: Buffer
) -> None:
    """Execute eagerly and verify the launch trace matches the captured graph.

    This method validates that graph capture correctly represents eager
    execution by running the model and comparing kernel launch sequences
    against a previously captured device graph.

    Args:
        self: The model to debug/verify
        graph_keys: Caller-provided graph key or per-device keys identifying
            captured graphs.
        inputs: Input buffers matching the captured input signature (same
            shapes and dtypes used during capture).

    Raises:
        TypeError: If ``graph_keys`` is neither an int nor a sequence of ints.
        ValueError: If any key in ``graph_keys`` is out of uint64 range.
        ValueError: If no input buffers are provided.
        RuntimeError: If no graph has been captured for ``graph_keys``.
        RuntimeError: If the eager execution trace doesn't match the captured graph.

    Example:
        >>> model.capture([1, 1], input_tensor)
        >>> model.debug_verify_replay([1, 1], input_tensor)  # Validates capture
        >>> model.replay([1, 1], input_tensor)  # Safe to use optimized replay
    """
    if not inputs:
        raise ValueError("Model.debug_verify_replay requires input buffers.")
    normalized_keys = _normalize_graph_keys(graph_keys)
    self._debug_verify_replay(normalized_keys, list(inputs))


def _Model_release_captured_graph(
    self: Model, graph_keys: int | Sequence[int]
) -> None:
    """Releases a previously captured device graph and its working memory.

    Drops the runtime-side reference for the given key(s); the underlying
    device graph and its captured-time scratch buffers are freed once any
    in-flight replay completes. Releasing a key that was never captured is
    a no-op.

    Note that the caller is still responsible for dropping any output
    :class:`Buffer` handles returned by the corresponding
    :meth:`Model.capture` call. Those buffers reference device memory that
    the runtime cannot reclaim while Python references remain.

    Args:
        self: The model whose captured graph should be released.
        graph_keys: Caller-provided graph key or per-device keys identifying
            captured graphs to release.

    Raises:
        TypeError: If ``graph_keys`` is neither an int nor a sequence of ints.
        ValueError: If any key in ``graph_keys`` is out of uint64 range.

    Example:
        >>> outputs = model.capture(42, input_tensor)
        >>> model.replay(42, input_tensor)
        >>> del outputs  # Drop Python-side handles first.
        >>> model.release_captured_graph(42)
    """
    normalized_keys = _normalize_graph_keys(graph_keys)
    self._release_captured_graph(normalized_keys)


Model.execute = _Model_execute  # type: ignore[method-assign]
Model.__call__ = _Model_call  # type: ignore[method-assign]
Model.__repr__ = _Model_repr  # type: ignore[method-assign]
Model.signature = property(_Model_signature)  # type: ignore[assignment]
Model.capture = _Model_capture  # type: ignore[method-assign]
Model.replay = _Model_replay  # type: ignore[method-assign]
Model.debug_verify_replay = _Model_debug_verify_replay  # type: ignore[method-assign]
Model.release_captured_graph = _Model_release_captured_graph  # type: ignore[method-assign]


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


def _process_custom_extensions_object(
    custom_extension: CustomExtensionType,
) -> CustomExtensionType:
    if is_mojo_source_package_path(Path(custom_extension)):
        # Builds the source directory into a .mojoc file.
        return _build_mojo_source_package(Path(custom_extension))

    # Pass the path through as is.
    return custom_extension


def _process_custom_extensions_objects(
    custom_extensions: CustomExtensionsType,
) -> list[CustomExtensionType]:
    if not isinstance(custom_extensions, Iterable) or isinstance(
        custom_extensions, str
    ):
        custom_extensions = [custom_extensions]
    return [
        _process_custom_extensions_object(custom_extension)
        for custom_extension in custom_extensions
    ]


def _derive_pipeline_name(module: Module) -> str:
    """Concatenate the sym_names of every non-subgraph `mo.graph` in `module`.

    Used as the diagnostic ``pipelineName`` passed to
    ``compile_from_object``; replaces the previous reliance on ``Graph.name``
    so the public :class:`Module` overload doesn't need a separate name kwarg.
    """
    return "+".join(module.top_level_graph_names())


class SplitKReductionPrecision(IntEnum):
    """Internal use."""

    ACCUM = auto()
    OUTPUT = auto()


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


class CompiledModel:
    """A compiled model artifact, ready for initialization with weights.

    Returned by :meth:`InferenceSession.compile`. Pass it to
    :meth:`InferenceSession.init` or :meth:`InferenceSession.init_all` to
    produce executable :class:`Model` instances.

    A :class:`CompiledModel` is not directly executable: compilation is
    independent of the device-memory allocations performed during
    initialization, so a single artifact can be initialized more than once.
    """

    _compiled: _AsyncValue[_CompiledModels]
    _expected_weights: dict[str, Any] | None
    # Top-level graph names captured from the source MLIR module when known.
    # Empty for path-compiled artifacts (no module to inspect). Used by the
    # virtual-device-mode short-circuit in ``init_all``; the non-virtual path
    # derives keys from each runtime ``Model.name`` instead.
    _graph_names: tuple[str, ...]

    def __init__(
        self,
        compiled: _AsyncValue[_CompiledModels],
        expected_weights: dict[str, Any] | None,
    ) -> None:
        # Internal constructor; users obtain instances from
        # :meth:`InferenceSession.compile`.
        self._compiled = compiled
        self._expected_weights = expected_weights
        self._graph_names = ()

    def __repr__(self) -> str:
        return "CompiledModel()"

    def export_mef(self, path: str | Path) -> None:
        """Exports this compiled artifact to a MEF file.

        Writes the serialized model straight from the compiled artifact, so
        it does not require the model to be initialized on a device. This
        makes it usable in cross-compilation / virtual-device scenarios where
        the target device may not be attached.

        Args:
            path: Filesystem path to write the MEF to.
        """
        self._compiled.wait()
        if (exc := self._compiled.exception()) is not None:
            raise exc
        self._compiled.result().export_mef(str(path))


class _ProfilingNamespace:
    """Runtime control surface for the libkineto-backed MAX profiler.

    Exposes the on-demand profiling lifecycle (start, stop, wait, state)
    that produces HTA-compatible Chrome trace JSON. Configuration lives on
    the ``ProfilingConfig`` model in
    ``max.pipelines.lib.config.profiling_config`` (e.g.
    ``profiling_output_path``).

    .. note::

       libkineto's profiler state is **process-global**. Calling
       :meth:`start` on one ``InferenceSession`` enables the profiler for the
       whole process, including any other live sessions. Only one MAX
       process per host should drive ``start()`` / ``stop()`` at a time —
       for multi-rank captures, an orchestrator must broadcast the enable
       command to every rank process.

       For the same reason, ``with session.profiling:`` blocks **must not be
       nested**: an inner ``__exit__`` will call :meth:`stop` and disable the
       profiler for any enclosing scope.

    This namespace is created automatically as ``session.profiling`` and
    should not be instantiated by user code.

    Example:

    .. code-block:: python

        session = InferenceSession(devices=[Accelerator()])
        model = session.load(my_graph)
        session.profiling.start()
        model.execute(input_data)
        session.profiling.stop()
        session.profiling.wait_for_trace()
    """

    def start(self) -> None:
        """Enable libkineto and begin recording.

        Subscribes to CUPTI activity callbacks. Tracy and libkineto are
        mutually exclusive at build time, so in ``--config=tracy`` builds
        (which do not link libkineto) this is a no-op. Idempotent — calling
        :meth:`start` while already enabled is a no-op.

        On builds without libkineto (today: macOS and Linux aarch64) or
        hosts without a live CUDA primary context, this is a safe no-op:
        ``state`` will still report ``"warmup"`` but no trace file is
        written by the matching :meth:`stop`.
        """
        _kineto_enable()

    def stop(self) -> None:
        """Disable libkineto and flush the trace.

        Unregisters CUPTI callbacks and finalizes the trace file; use
        :meth:`wait_for_trace` if you need to ensure serialization is
        complete before reading it.
        """
        _kineto_disable()

    def wait_for_trace(self) -> None:
        """Block until the most recent :meth:`stop` finishes serializing."""
        _kineto_wait_for_trace()

    @property
    def state(self) -> Literal["idle", "warmup", "active", "flushing"]:
        """Current profiler state.

        Returns:
            One of ``"idle"``, ``"warmup"``, ``"active"``, or ``"flushing"``.
        """
        return cast(
            Literal["idle", "warmup", "active", "flushing"], _kineto_state()
        )

    @property
    def is_enabled(self) -> bool:
        """``True`` between :meth:`start` and :meth:`stop`.

        Equivalent to ``state in {"warmup", "active"}``; ``False`` in
        ``"idle"`` and ``"flushing"``.

        Cheap relative to constructing a trace name you would otherwise skip,
        but still crosses the Python/C++ FFI boundary on every call — cache
        the result if you need it inside a tight loop.
        """
        return _kineto_is_enabled()

    def __enter__(self) -> _ProfilingNamespace:
        """Enter a profiling context: equivalent to calling :meth:`start`.

        Lets callers write::

            with session.profiling:
                model.execute(input_data)

        and have :meth:`stop` invoked automatically on scope exit, even if
        the body raises.  The returned object is the namespace itself, so
        ``state`` / ``is_enabled`` remain accessible from inside the block.
        """
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the profiling context: calls :meth:`stop`.

        Does NOT wait for the trace to serialize — callers that need the
        file on disk before reading it should call :meth:`wait_for_trace`
        after the ``with`` block.  This matches the rule that ``stop()``
        and ``wait_for_trace()`` are split so callers can interleave other
        cleanup between them.
        """
        self.stop()


class InferenceSession:
    """Manages an inference session in which you can load and run models.

    You need an ``InferenceSession`` instance to load a model as a
    :class:`~max.engine.Model` object. For example:

    .. code-block:: python

        session = engine.InferenceSession(devices=[CPU()])
        model = session.load(model_path)

    For workflows that need to separate compilation from weight binding,
    use :meth:`compile` followed by :meth:`init` or :meth:`init_all`.
    For example:

    .. code-block:: python

        session = engine.InferenceSession(devices=[CPU()])
        compiled = session.compile(model_path)
        model = session.init(compiled)

    Args:
        devices: A list of devices on which to run inference. The host CPU
            is always included automatically.
        num_threads: The number of execution threads. Defaults to ``None``,
            which lets the runtime choose automatically.
        custom_extensions: The extensions to load for the model. Supports
            paths to a ``.mojoc``/``.mojopkg`` custom ops library or a
            ``.mojo`` source file.
    """

    _impl: _InferenceSession
    # DebugConfig is a process-wide singleton. Assigning it as a class
    # attribute at import time means both ``InferenceSession.debug`` and
    # ``session.debug`` return the same underlying object, and any
    # ``MODULAR_DEBUG`` env-var parsing happens exactly once (at import).
    debug: DebugConfig = _InferenceSession.debug
    # libkineto's profiler state is process-global (see _ProfilingNamespace
    # docstring), so a single shared instance — matching ``debug`` above —
    # accurately reflects that scope. The namespace carries no per-session
    # state.
    profiling: _ProfilingNamespace = _ProfilingNamespace()

    def __init__(
        self,
        devices: Iterable[Device] = (),
        num_threads: int | None = None,
        *,
        custom_extensions: CustomExtensionsType | None = None,
    ) -> None:
        self.num_threads = num_threads

        # Process the provided iterable `devices`.
        final_devices: list[Device] = []
        seen_devices: set[Device] = set()
        for device in devices:
            if device not in seen_devices:
                final_devices.append(device)
                seen_devices.add(device)
        host_cpu = CPU()
        if host_cpu not in seen_devices:
            final_devices.append(host_cpu)
            seen_devices.add(host_cpu)

        custom_extensions_final = []

        if custom_extensions:
            custom_extensions_final = _process_custom_extensions_objects(
                custom_extensions
            )

        self._impl = _InferenceSession(
            final_devices,
            custom_extensions_final,
            num_threads or 0,
        )

        # Register async-safe Python stack trace handler
        # This enables Python stack traces in crash reports without GIL deadlocks
        try:
            faulthandler.register(
                signal.SIGUSR2, file=sys.stderr, all_threads=True, chain=False
            )
        except (OSError, RuntimeError):
            # Ignore errors if SIGUSR2 is already registered or unavailable
            pass

        # Read the op log level from the max-debug.op-log-level config key
        # (covers MODULAR_MAX_DEBUG_OP_LOG_LEVEL env var, modular.cfg, and
        # InferenceSession.debug.op_log_level Python setter).
        if log_level := _InferenceSession.debug.op_log_level:
            self.set_mojo_log_level(log_level)

        # Read the assert level from the max-debug.assert-level Config key.
        if assert_level_str := _InferenceSession.debug.assert_level:
            try:
                assert_level = AssertLevel[assert_level_str.upper()]
            except KeyError as e:
                raise TypeError(
                    f"Invalid assert level ({assert_level_str}). Please use one of: {[x.name for x in AssertLevel]}"
                ) from e
            self.set_mojo_assert_level(assert_level)

        # TODO: Remove this once the new topk kernel is stable.
        if use_old_top_k_kernel := os.getenv("USE_OLD_TOP_K_KERNEL"):
            self.use_old_top_k_kernel(use_old_top_k_kernel)

        if use_fi_topk := os.getenv("USE_FI_TOPK_KERNEL"):
            self.use_fi_topk_kernel(use_fi_topk)

        if val := os.getenv("ENABLE_PER_TENSOR_FP8_QUANTIZE"):
            self.enable_per_tensor_fp8_quantize(val)

        # Read the uninit-read check from the max-debug.uninitialized-read-check
        # Config key.
        if _InferenceSession.debug.uninitialized_read_check:
            # Enable debug allocator poison
            existing = os.environ.get("MODULAR_DEBUG_DEVICE_ALLOCATOR", "")
            if existing:
                if "uninitialized-poison" not in existing:
                    os.environ["MODULAR_DEBUG_DEVICE_ALLOCATOR"] = (
                        existing + ",uninitialized-poison"
                    )
            else:
                os.environ["MODULAR_DEBUG_DEVICE_ALLOCATOR"] = (
                    "uninitialized-poison"
                )
            # Enable compile-time checks
            self._set_mojo_define("MOJO_STDLIB_SIMD_UNINIT_CHECK", "true")

    def __repr__(self) -> str:
        if self.num_threads:
            return f"<modular engine InferenceSession(num_threads={self.num_threads})>"
        else:
            return "<modular engine InferenceSession>"

    def load(
        self,
        model: str | Path | Graph,
        *,
        custom_extensions: CustomExtensionsType | None = None,
        weights_registry: Mapping[str, DLPackArray] | None = None,
    ) -> Model:
        """Loads a trained model and compiles it for inference.

        .. note::

            This method combines compilation and weight binding in a single
            call and will be deprecated over time. New code should call
            :meth:`compile` followed by :meth:`init` instead, which separates
            the two steps and lets you reuse a compiled artifact across
            multiple initializations.

        Args:
            model: A :class:`Graph` instance, or the path to a saved model
                file (for example, a ``.mef`` file).

            custom_extensions: The extensions to load for the model.
                Supports paths to ``.mojoc``/``.mojopkg`` custom ops.

            weights_registry: A mapping from model weight names to their
                values. The values should be DLPack arrays. If an array is a
                read-only NumPy array, you must ensure that its lifetime
                extends beyond the lifetime of the model. Although
                ``weights_registry`` is technically optional, you'll always
                need to load weights in practice.

        Returns:
            The loaded model, compiled and ready to execute.

        Raises:
            RuntimeError: If the path provided is invalid.
        """
        models = self.load_all(
            model,
            custom_extensions=custom_extensions,
            weights_registry=weights_registry,
        )
        if len(models) != 1:
            raise ValueError(
                f"Expected exactly one model in the compiled artifact, but "
                f"got {len(models)}. Use load_all() to load multi-model artifacts."
            )
        return next(iter(models.values()))

    def load_all(
        self,
        model: str | Path | Module | Graph,
        *,
        custom_extensions: CustomExtensionsType | None = None,
        weights_registry: Mapping[str, DLPackArray] | None = None,
    ) -> dict[str, Model]:
        """Loads multiple models and compiles them for inference.

        A compiled ``.mef`` artifact may contain more than one model (for
        example, a vision encoder and a language model compiled together).
        This method returns one :class:`Model` per model encoded in the
        artifact, keyed by the ``sym_name`` of the corresponding ``mo.graph``
        op (preserved through MEF serialization). For single-model
        artifacts, the returned dict has exactly one entry.

        .. note::

            This method combines compilation and weight binding in a single
            call and will be deprecated over time. New code should call
            :meth:`compile` followed by :meth:`init_all` instead, which
            separates the two steps and lets you reuse a compiled artifact
            across multiple initializations.

        Args:
            model: A :class:`max.graph.Module` containing one or more
                ``mo.graph`` ops, the path to a saved multi-model file (for
                example, a ``.mef`` file), or a single :class:`Graph`.

            custom_extensions: The extensions to load for the model.
                Supports paths to ``.mojoc``/``.mojopkg`` custom ops.

            weights_registry: A mapping from model weight names to their
                values. The values should be DLPack arrays. If an array is a
                read-only NumPy array, you must ensure that its lifetime
                extends beyond the lifetime of the model. Although
                ``weights_registry`` is technically optional, you'll always
                need to load weights in practice.

        Returns:
            A mapping from each model's ``sym_name`` to its loaded
            :class:`Model`, ready to execute.

        Raises:
            RuntimeError: If the path provided is invalid.
        """
        compiled = self.compile(model, custom_extensions=custom_extensions)
        return self.init_all(compiled, weights_registry=weights_registry)

    def compile_async(
        self,
        model: str | Path | Module | Graph,
        *,
        custom_extensions: CustomExtensionsType | None = None,
    ) -> CompiledModel:
        """Compiles a model without blocking on the compilation finishing.

        Returns as soon as the compile is scheduled; the returned
        :class:`CompiledModel` wraps a pending compilation that runs on the
        runtime's worker pool. Compilation errors are not raised here — they
        surface when the artifact is awaited, for example by :meth:`init`,
        :meth:`init_all`, or :meth:`CompiledModel.export_mef`. Use
        :meth:`compile` for the synchronous variant that blocks and raises.

        Args:
            model: A :class:`Graph` instance, a :class:`max.graph.Module`
                containing one or more ``mo.graph`` ops, or the path to a
                saved model file (for example, a ``.mef`` file).

            custom_extensions: The extensions to load for the model.
                Supports paths to ``.mojopkg`` custom ops.

        Returns:
            A :class:`CompiledModel` artifact wrapping the pending compilation.
        """
        custom_extensions_final: list[CustomExtensionType] = []
        if custom_extensions is not None:
            custom_extensions_final = _process_custom_extensions_objects(
                custom_extensions
            )

        # Track the MLIR module if we have one so we can enumerate graph
        # names and capture expected-weight metadata for init-time validation.
        module: Module | None = None
        expected_weights: dict[str, Any] | None = None

        if isinstance(model, Path | str):
            handle = self._impl.compile(model, custom_extensions_final)
        elif isinstance(model, Graph):
            module = model.module
            custom_extensions_final.extend(
                _process_custom_extensions_objects(model.kernel_libraries_paths)
            )

            # TODO: if the model has been loaded from a serialized MLIR
            # file, we don't have the _weights attribute available to us
            if hasattr(model, "_weights"):
                expected_weights = {
                    name: weight.value.device
                    for name, weight in model._weights.items()
                }

            # Seed the model module with kernel decls + the opaque-type
            # mapping from the graph's KernelLibrary. The GC pipeline
            # detects the mapping attribute and skips
            # `mogg-import-packages`, so the expensive package-loading
            # step (run once at KernelLibrary construction) doesn't
            # repeat on every compile.
            kernel_library = getattr(model, "_kernel_library", None)
            if kernel_library is not None:
                kernel_library._analysis.seed_kernel_decls(module.mlir_module)

            handle = self._compile_module(module, custom_extensions_final)
        elif isinstance(model, Module):
            module = model
            handle = self._compile_module(module, custom_extensions_final)
        else:
            raise RuntimeError("The model is not a valid path or module.")

        compiled = CompiledModel(
            compiled=handle, expected_weights=expected_weights
        )
        if module is not None:
            compiled._graph_names = tuple(module.top_level_graph_names())
        return compiled

    def compile(
        self,
        model: str | Path | Module | Graph,
        *,
        custom_extensions: CustomExtensionsType | None = None,
    ) -> CompiledModel:
        """Compiles a model without binding weights or device memory.

        Use this when you want to separate compilation from initialization, for
        example to populate a compile cache ahead of time, including in
        cross-compilation scenarios where the target device may not be
        attached. The returned :class:`CompiledModel` requires initialization
        before execution. Pass it to :meth:`init` or :meth:`init_all` to
        produce an executable :class:`Model`.

        Blocks until compilation finishes and raises on failure. Use
        :meth:`compile_async` to schedule compilation without blocking.

        Args:
            model: A :class:`Graph` instance, a :class:`max.graph.Module`
                containing one or more ``mo.graph`` ops, or the path to a
                saved model file (for example, a ``.mef`` file).

            custom_extensions: The extensions to load for the model.
                Supports paths to ``.mojopkg`` custom ops.

        Returns:
            A :class:`CompiledModel` artifact ready to be initialized.

        Raises:
            RuntimeError: If the path provided is invalid or compilation
                fails.
        """
        with _record_phase("compile_seconds"):
            compiled = self.compile_async(
                model, custom_extensions=custom_extensions
            )
            # Synchronously complete the compilation and raise errors.
            compiled._compiled.wait()
        exception = compiled._compiled.exception()
        if exception is None:
            return compiled
        # compile_async surfaces the compile failure here rather than from the
        # compile call, so the Graph/Module wrapping that _compile_module
        # applies to synchronous setup errors is repeated here for the async
        # failure.
        if isinstance(model, (Graph, Module)):
            raise RuntimeError(self._compile_failure_message()) from exception
        raise exception

    def init(
        self,
        compiled: CompiledModel,
        *,
        weights_registry: Mapping[str, DLPackArray] | None = None,
    ) -> Model:
        """Initializes a compiled model with weights for execution.

        Use this to complete the second half of a :meth:`compile`/:meth:`init`
        pair when the artifact contains a single model. For artifacts with
        more than one model, use :meth:`init_all`.

        Args:
            compiled: The compiled artifact returned by :meth:`compile`.

            weights_registry: A mapping from model weight names to their
                values. The values should be DLPack arrays. If an array is a
                read-only NumPy array, you must ensure that its lifetime
                extends beyond the lifetime of the model. Although
                ``weights_registry`` is technically optional, you'll always
                need to load weights in practice.

        Returns:
            The initialized :class:`Model`, ready to execute.
        """
        models = self.init_all(compiled, weights_registry=weights_registry)
        if len(models) != 1:
            raise ValueError(
                f"Expected exactly one model in the compiled artifact, but "
                f"got {len(models)}. Use init_all() to initialize multi-model "
                f"artifacts."
            )
        return next(iter(models.values()))

    def init_all(
        self,
        compiled: CompiledModel,
        *,
        weights_registry: Mapping[str, DLPackArray] | None = None,
    ) -> dict[str, Model]:
        """Initializes all models in a compiled artifact for execution.

        Use this to complete the second half of a
        :meth:`compile`/:meth:`init_all` pair. Returns one :class:`Model` per
        top-level graph in the artifact,
        keyed by ``sym_name``.

        Args:
            compiled: The compiled artifact returned by :meth:`compile`.

            weights_registry: A mapping from model weight names to their
                values. See :meth:`init` for details.

        Returns:
            A mapping from each model's ``sym_name`` to its initialized
            :class:`Model`, ready to execute.
        """
        if is_virtual_device_mode():
            # Virtual device mode can't actually initialize the model, but
            # users (eg. cross compilation, benchmarking) want it to not fail.
            # Return one mock per top-level graph in the artifact so callers
            # that key by graph name still work.
            if not compiled._graph_names:
                raise ValueError(
                    "Cannot initialize a path-compiled artifact in "
                    "virtual-device mode: graph names are unknown without "
                    "an MLIR module to inspect. Initialize on a real device "
                    "instead, or compile from a Graph/Module."
                )
            return {name: mock.Mock(Model) for name in compiled._graph_names}

        weights_registry_real: Mapping[str, DLPackArray] = (
            weights_registry or {}
        )

        # Validate the registry against the expected weights captured at
        # compile time (only available when the source was a Graph with a
        # `_weights` attribute).
        if compiled._expected_weights is not None:
            for (
                weight_name,
                expected_device,
            ) in compiled._expected_weights.items():
                if weight_name not in weights_registry_real:
                    raise ValueError(
                        f"Weight '{weight_name}' is not in the weights registry."
                    )

                registered_weight = weights_registry_real[weight_name]
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

        for weight_name, weight_value in weights_registry_real.items():
            try:
                _raise_if_not_contiguous(weight_value)
            except ValueError as e:
                raise ValueError(
                    f"Weight '{weight_name}' is not contiguous: {str(e)}"
                ) from e

        with _record_phase("init_seconds"):
            models = self._impl._load_all(
                compiled._compiled, weights_registry_real
            )
        result = {m.name: m for m in models}
        if len(result) != len(models):
            raise RuntimeError(
                "Compiled artifact contains models with duplicate sym_names; "
                f"got {[m.name for m in models]}"
            )
        return result

    def _compile_failure_message(self) -> str:
        """Returns the wrapper text for a Graph/Module compilation failure.

        Shared by the synchronous setup-error path in :meth:`_compile_module`
        and the asynchronous compile-failure path in :meth:`compile`, so both
        surface identical guidance.
        """
        msg = (
            "Failed to compile the model. Please file an issue, "
            "all models should be correct by construction and "
            "this error should have been caught during construction."
        )
        if not self.debug.source_tracebacks:
            msg += (
                "\nFor more detailed failure information enable the "
                "`max-debug.source-tracebacks` config key (for example, "
                "`Graph.debug.source_tracebacks = True` or "
                "`MODULAR_DEBUG=source-tracebacks`)."
            )
        return msg

    def _compile_module(
        self,
        module: Module,
        custom_extensions_final: list[CustomExtensionType],
    ) -> _AsyncValue[_CompiledModels]:
        """Compiles an MLIR module under the session's compilation lock.

        Wraps any synchronous setup failure in a ``RuntimeError`` pointing at
        the ``max-debug.source-tracebacks`` config key for richer diagnostics.
        Compilation itself is asynchronous; that failure surfaces when the
        returned value is awaited (see :meth:`compile`).
        """
        try:
            return self._impl.compile(
                module.mlir_module._CAPIPtr,
                custom_extensions_final,
                _derive_pipeline_name(module),
            )
        except Exception as e:
            raise RuntimeError(self._compile_failure_message()) from e

    def set_debug_print_options(
        self,
        style: str | PrintStyle = PrintStyle.COMPACT,
        precision: int = 6,
        output_directory: str | Path | None = None,
    ) -> None:
        """Sets the debug print options.

        Affects debug printing across all model execution using the same
        :class:`InferenceSession`. See :meth:`~max.graph.TensorValue.print`.

        Tensors saved with ``BINARY`` can be loaded using
        :meth:`max.driver.Buffer.mmap`, but you'll have to provide the
        expected dtype and shape. Tensors saved with ``BINARY_MAX_CHECKPOINT``
        are saved with the shape and dtype information and can be loaded with
        :func:`max.driver.buffer.load_max_buffer`.

        .. note::

            Even with ``style`` set to ``NONE``, debug print ops in the graph
            can prevent optimization. If you see performance issues, try fully
            removing debug print ops.

        Args:
            style: The print style for tensor values. One of ``COMPACT``,
                ``FULL``, ``BINARY``, ``BINARY_MAX_CHECKPOINT``, or ``NONE``.
            precision: The digits of precision in the output, used when
                ``style`` is ``FULL``.
            output_directory: The directory to store output tensors, used
                when ``style`` is ``BINARY`` or ``BINARY_MAX_CHECKPOINT``.

        Raises:
            TypeError: If ``style`` is not a valid :class:`PrintStyle`, if
                ``precision`` is not an ``int`` when ``style`` is ``FULL``,
                or if ``output_directory`` is not a ``str`` or
                :class:`~pathlib.Path`.
            ValueError: If ``output_directory`` is empty when ``style`` is
                ``BINARY`` or ``BINARY_MAX_CHECKPOINT``.
        """
        if isinstance(style, str):
            style = cast(str | PrintStyle, getattr(PrintStyle, style, style))
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
        """Sets the accumulation precision for split-k reductions in large matmuls.

        Args:
            precision: The accumulation precision to use, given as a
                ``SplitKReductionPrecision`` member or its name as a string.

        Raises:
            TypeError: If ``precision`` is not a valid
                ``SplitKReductionPrecision`` member or name.
        """
        if not isinstance(precision, SplitKReductionPrecision):
            try:
                precision = SplitKReductionPrecision[precision]
            except Exception as e:
                raise TypeError(
                    f"Invalid precision ({precision}). Please use one of: {[x.name for x in SplitKReductionPrecision]}"
                ) from e

        self._set_mojo_define("SPLITK_REDUCTION_SCHEME", precision)

    def set_mojo_log_level(self, level: str | LogLevel) -> None:
        """Sets the verbosity of Mojo logging in the compiled model.

        Args:
            level: The log level to use, given as a :class:`LogLevel` member
                or its name as a string.

        Raises:
            TypeError: If ``level`` is not a valid :class:`LogLevel` member
                or name.
        """
        if not isinstance(level, LogLevel):
            try:
                level = LogLevel[level]
            except Exception as e:
                raise TypeError(
                    f"Invalid log level ({level}). Please use one of: {[x.name for x in LogLevel]}"
                ) from e

        self._set_mojo_define("LOGGING_LEVEL", level)

    def set_mojo_assert_level(self, level: AssertLevel) -> None:
        """Sets which Mojo asserts are kept in the compiled model.

        .. note::

            Not all kernels are runnable with asserts enabled. If model
            compilation or execution fails at higher assert levels, retry with
            ``AssertLevel.NONE``.

        Args:
            level: The assert level to use. One of ``AssertLevel.NONE``,
                ``AssertLevel.WARN``, ``AssertLevel.SAFE``, or
                ``AssertLevel.ALL``.
        """
        self._set_mojo_define("ASSERT", level)

    def gpu_profiling(self, mode: GPUProfilingMode) -> None:
        """Enables GPU profiling instrumentation for the session.

        Works with NVIDIA Nsight Systems and Nsight Compute. When enabled,
        the runtime adds CUDA driver calls and NVTX markers that allow
        profiling tools to correlate GPU kernel executions with host-side
        code.

        For example, to enable detailed profiling for Nsight Systems
        analysis, call :meth:`gpu_profiling` before :meth:`load`:

        .. code-block:: python

            from max.engine import InferenceSession
            from max.driver import Accelerator

            session = InferenceSession(devices=[Accelerator()])
            session.gpu_profiling("detailed")
            model = session.load(my_graph)

        Then run it with ``nsys``:

        .. code-block:: bash

            nsys profile --trace=cuda,nvtx python example.py

        Instead of calling :meth:`gpu_profiling` in code, you can set the
        ``MODULAR_ENABLE_PROFILING`` environment variable when you call
        ``nsys profile``:

        .. code-block:: bash

            MODULAR_ENABLE_PROFILING=detailed nsys profile --trace=cuda,nvtx python script.py

        Be aware that :meth:`gpu_profiling` overrides the
        ``MODULAR_ENABLE_PROFILING`` environment variable if also used.

        Learn more in `GPU profiling with Nsight Systems </max/gpu-system-profiling>`_.

        .. note::

            Profiling instrumentation adds runtime overhead and should be
            disabled for production deployments.

        Args:
            mode: The profiling mode to set. One of:

                - ``off``: Disable profiling (default).
                - ``on``: Enable basic profiling with NVTX markers for
                  kernel correlation.
                - ``detailed``: Enable detailed profiling with additional
                  Python-level NVTX markers.
        """
        if mode == "off":
            return

        self._set_mojo_define("MODULAR_ENABLE_PROFILING", 1)
        self._set_mojo_define("MODULAR_ENABLE_GPU_PROFILING", 1)
        if mode == "detailed":
            self._set_mojo_define("MODULAR_ENABLE_GPU_PROFILING_DETAILED", 1)

        set_gpu_profiling_state(mode)

    def use_old_top_k_kernel(self, mode: str) -> None:
        """Falls back to the previous top-k kernel implementation.

        By default, the session uses a newer top-k kernel. Use this
        fallback only if you encounter correctness or performance issues
        with the default kernel.

        Args:
            mode: The enable/disable flag. Accepts ``"false"``, ``"off"``,
                ``"no"``, or ``"0"`` to disable. Any other value enables the
                old top-k kernel.
        """
        if mode.lower() in ("false", "off", "no", "0"):
            return

        self._set_mojo_define("USE_OLD_TOP_K_KERNEL", 1)

    def use_fi_topk_kernel(self, mode: str) -> None:
        """Enables the fused-inference top-k kernel.

        Args:
            mode: The enable/disable flag. Accepts ``"false"``, ``"off"``,
                ``"no"``, or ``"0"`` to disable. Any other value enables the
                fused-inference top-k kernel.
        """
        if mode.lower() in ("false", "off", "no", "0"):
            return

        self._set_mojo_define("USE_FI_TOPK_KERNEL", 1)

    def enable_per_tensor_fp8_quantize(self, mode: str) -> None:
        """Enables per-tensor FP8 quantization.

        Args:
            mode: The enable/disable flag. Accepts ``"false"``, ``"off"``,
                ``"no"``, or ``"0"`` to disable. Any other value enables
                per-tensor FP8 quantization.
        """
        if mode.lower() in ("false", "off", "no", "0"):
            return

        self._set_mojo_define("ENABLE_PER_TENSOR_FP8_QUANTIZE", 1)

    def _use_experimental_kernels(self, mode: str) -> None:
        """Enables experimental kernels."""
        if mode.lower() in ("false", "off", "no", "0"):
            return

        self._set_mojo_define("USE_EXPERIMENTAL_KERNELS", 1)

    def _use_vendor_blas(self, mode: str) -> None:
        """Enables vendor BLAS libraries."""
        if mode.lower() in ("false", "off", "no", "0"):
            return

        self._set_mojo_define("MODULE_USE_VENDOR_BLAS", 1)

    def _use_vendor_ccl(self, mode: str) -> None:
        """Enables vendor CCL libraries (NCCL/RCCL) for collective operations."""
        if mode.lower() in ("false", "off", "no", "0"):
            return

        self._set_mojo_define("MODULAR_USE_VENDOR_CCL", 1)

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
        """The devices available to the session, including the host CPU."""
        return self._impl.devices
