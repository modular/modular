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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

"""Modular framework Python bindings."""

import enum
import inspect
import os
import types
from collections.abc import Mapping, Sequence
from typing import Any, overload

import max._core.driver
import max._core.dtype
from max._core.driver import Buffer
from max._core_types.driver import DLPackArray

InputType = DLPackArray | Buffer | int | float | bool

class TensorSpec:
    """
    Defines the properties of a tensor, including its name, shape and data type.

    For usage examples, see :obj:`Model.input_metadata`.
    """

    @property
    def dtype(self) -> max._core.dtype.DType:
        """A tensor data type."""

    @property
    def name(self) -> str:
        """A tensor name."""

    @property
    def shape(self) -> list[int | None] | None:
        """
        The shape of the tensor as a list of integers.

        If a dimension size is unknown/dynamic (such as the batch size), its
        value is ``None``.
        """

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Model:
    """
    A loaded model that you can execute.

    Do not instantiate this class directly. Instead, create it with
    :obj:`InferenceSession`.
    """

    @property
    def devices(self) -> list[max._core.driver.Device]:
        """Returns the device objects used in the Model."""

    @property
    def input_devices(self) -> list[max._core.driver.Device]:
        """
        Devices of the model's input tensors, as a list of :obj:`Device` objects.
        """

    @property
    def input_metadata(self) -> list[TensorSpec]:
        """
        Metadata about the model's input tensors, as a list of :obj:`TensorSpec` objects.

        For example, you can print the input tensor names, shapes, and dtypes:

        .. code-block:: python

            for tensor in model.input_metadata:
                print(f'name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')
        """

    @property
    def output_devices(self) -> list[max._core.driver.Device]:
        """
        Devices of the model's output tensors, as a list of :obj:`Device` objects.
        """

    @property
    def output_metadata(self) -> list[TensorSpec]:
        """
        Metadata about the model's output tensors, as a list of :obj:`TensorSpec` objects.

        For example, you can print the output tensor names, shapes, and dtypes:

        .. code-block:: python

            for tensor in model.output_metadata:
                print(f'name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')
        """

    @property
    def kernel_summaries(self) -> list[str]:
        """
        Kernel fusion summaries from the compiled model.

        Returns a list of strings, one per ``mgp.generic.execute`` kernel in
        the compiled graph.  Each string describes the fused kernel composition,
        e.g. ``"Epilogue(custom__kv_rope, custom__kv_cache_store)"``.
        """

    @property
    def name(self) -> str:
        """
        The symbol name of this model.

        Mirrors the ``sym_name`` of the model's ``mo.graph`` op, preserved
        through MEF serialization. Used by
        :meth:`InferenceSession.load_all` to key the returned dict by graph
        name.
        """

    @property
    def signature(self) -> inspect.Signature:
        """Get input signature for model."""

    def execute(self, *args: InputType) -> list[Buffer]:
        """
        Executes the model with the provided input and returns the outputs.

        For example, if the model has one input tensor:

        .. code-block:: python

            input_tensor = np.random.rand(1, 224, 224, 3)
            model.execute(input_tensor)

        Args:
            args:
              A list of input tensors. We currently support :obj:`np.ndarray`,
              :obj:`torch.Tensor`, and :obj:`max.driver.Buffer` inputs. All
              inputs will be copied to the device that the model is resident on
              prior to executing.

            output_device:
              The device to copy output tensors to. Defaults to :obj:`None`, in
              which case the tensors will remain resident on the same device as
              the model.

        Returns:
            A list of output tensors and Mojo values. The output tensors will be
            resident on the execution device by default (you can change it with
            the ``output_device`` argument).

        Raises:
            RuntimeError: If the given input tensors' shape don't match what
              the model expects.

            TypeError: If the given input tensors' dtype cannot be cast to what
              the model expects.

            ValueError: If positional inputs are not one of the supported
              types, i.e. :obj:`np.ndarray`, :obj:`torch.Tensor`, and
              :obj:`max.driver.Buffer`.
        """

    def __call__(self, *args: InputType, **kwargs: InputType) -> list[Buffer]:
        """
        Executes the model with the provided input and returns the outputs.

        Models can be called with any mixture of positional and named inputs:

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

              * Any tensors implementing the DLPack protocol, such as
                :obj:`np.ndarray`, :obj:`torch.Tensor`
              * Max Driver buffers, i.e. :obj:`max.driver.Buffer`
              * Scalar inputs, i.e. :obj:`bool`, :obj:`float`, :obj:`int`,
                :obj:`np.generic`

            kwargs: Named inputs. We can support the same types supported
              in :obj:`args`.

        Returns:
            A list of output tensors. The output tensors will be
            resident on the execution device.

        Raises:
            RuntimeError: If the given input tensors' shape don't match what
              the model expects.

            TypeError: If the given input tensors' dtype cannot be cast to
              what the model expects.

            ValueError: If positional inputs are not one of the supported
              types, i.e. :obj:`np.ndarray`, :obj:`torch.Tensor`, and
              :obj:`max.driver.Buffer`.

            ValueError: If an input name does not correspond to what the model
              expects.

            ValueError: If any positional and named inputs collide.

            ValueError: If the number of inputs is less than what the model
              expects.
        """

    def __repr__(self) -> str: ...
    def capture(
        self, graph_keys: int | Sequence[int], *inputs: Buffer
    ) -> list[Buffer]:
        """
        Capture execution into a device graph for caller-provided key.

        Capture is best-effort and model-dependent. It records the current execution
        path; models that perform unsupported operations during capture (for example,
        host-device synchronization) will fail to capture. Callers should decide which
        phases are safe to capture (e.g. decode-only in serving).
        """

    def replay(self, graph_keys: int | Sequence[int], *inputs: Buffer) -> None:
        """Replay the captured device graph for the provided key."""

    def debug_verify_replay(
        self, graph_keys: int | Sequence[int], *inputs: Buffer
    ) -> None:
        """
        Execute eagerly and verify the launch trace matches the captured graph.

        This method validates that graph capture correctly represents eager execution
        by running the model and comparing kernel launch traces. Useful for debugging
        graph capture issues.

        Args:
            graph_keys: One graph key per participating device stream.
            inputs: Input buffers matching the captured input signature.

        Raises:
            RuntimeError: If no graph captured or trace verification fails.
        """

    def release_captured_graph(self, graph_keys: int | Sequence[int]) -> None:
        """
        Release a previously captured device graph.

        Drops the device-side graph and its working memory once the last reference
        held by the runtime is released. Releasing a key that was never captured
        is a no-op.

        Args:
            graph_keys: Caller-provided graph key (or per-device keys) identifying
                the captured graph to release.
        """

    def _execute_device_tensors(
        self, tensors: Sequence[max._core.driver.Buffer]
    ) -> list[max._core.driver.Buffer]: ...
    def _capture(
        self,
        graph_keys: Sequence[int],
        inputs: Sequence[max._core.driver.Buffer],
    ) -> list[max._core.driver.Buffer]:
        """Capture execution into a device graph."""

    def _replay(
        self,
        graph_keys: Sequence[int],
        inputs: Sequence[max._core.driver.Buffer],
    ) -> None:
        """Replay the captured device graph."""

    def _debug_verify_replay(
        self,
        graph_keys: Sequence[int],
        inputs: Sequence[max._core.driver.Buffer],
    ) -> None:
        """Debug verify replay against captured graph."""

    def _await_device_graphs(self) -> None:
        """Await all pending device graph instantiations."""

    def _release_captured_graph(self, graph_keys: Sequence[int]) -> None:
        """Release captured device graphs for the given keys."""

    def _export_mef(self, path: str) -> None:
        """
        Exports the compiled model as a mef to a file.

        Args:
          path: The filename where the mef is exported to.
        """

    def reload(self, weights_registry: Mapping[str, Any]) -> None: ...

class DebugConfig:
    """
    Unified debug configuration for MAX inference.

    ``DebugConfig`` is a process-wide singleton accessed through
    :attr:`InferenceSession.debug`. It controls model debugging features
    such as ``NaN`` checks, synchronous GPU execution, stack traces, and IR
    dumping.

    There are two ways to configure debugging options:

    * Set the ``MODULAR_DEBUG`` environment variable to a list of
      kebab-case property names separated by commas. Boolean properties
      can be enabled with just the name; others use ``name=value`` form.
      For example: ``MODULAR_DEBUG=nan-check,assert-level=all``.
    * Set properties directly with the Python API, for example
      ``InferenceSession.debug.<property> = <value>``. Options are
      class-level on :class:`InferenceSession` because they affect
      globally shared infrastructure.

    For the environment variable and config file, the name ``sensible``
    enables a curated default set defined in :attr:`sensible_mode`.
    """

    @property
    def nan_check(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to insert runtime checks after each compiled op that abort if any output contains ``NaN``. Takes effect at model build time.
        """

    @nan_check.setter
    def nan_check(self, arg: bool, /) -> None: ...
    @property
    def uninitialized_read_check(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to instrument buffer reads to detect reads of uninitialized memory. Takes effect at model build time.
        """

    @uninitialized_read_check.setter
    def uninitialized_read_check(self, arg: bool, /) -> None: ...
    @property
    def device_sync_mode(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to force synchronous GPU execution so every device operation waits for completion. This surfaces async errors at their call site but serializes the pipeline. Takes effect at run time.
        """

    @device_sync_mode.setter
    def device_sync_mode(self, arg: bool, /) -> None: ...
    @property
    def stack_trace_on_error(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to print a C++ stack trace whenever a runtime error is raised. Takes effect at run time.
        """

    @stack_trace_on_error.setter
    def stack_trace_on_error(self, arg: bool, /) -> None: ...
    @property
    def stack_trace_on_crash(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to print a C++ stack trace on fatal signals such as ``SIGSEGV`` or ``SIGABRT``. Takes effect at run time.
        """

    @stack_trace_on_crash.setter
    def stack_trace_on_crash(self, arg: bool, /) -> None: ...
    @property
    def source_tracebacks(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to capture Python source locations during graph construction so runtime errors can be traced back to user code. Takes effect at graph build time and is typically set using ``Graph.debug.source_tracebacks``.
        """

    @source_tracebacks.setter
    def source_tracebacks(self, arg: bool, /) -> None: ...
    @property
    def op_log_level(self) -> str:
        r"""
        A string that sets the log level for per-op tracing. One of ``\'\'``, ``'notset'``, ``'trace'``, ``'debug'``, ``'info'``, ``'warning'``, ``'error'``, ``'critical'``. Takes effect at model build time.
        """

    @op_log_level.setter
    def op_log_level(self, arg: str, /) -> None: ...
    @property
    def assert_level(self) -> str:
        r"""
        A string that sets the Mojo assertion level for compiled kernels. One of ``\'\'``, ``'none'``, ``'warn'``, ``'safe'``, ``'all'``. Higher levels enable more runtime checks (e.g. LayoutTensor bounds) at a performance cost. Takes effect at model build time.
        """

    @assert_level.setter
    def assert_level(self, arg: str, /) -> None: ...
    @property
    def print_style(self) -> PrintStyle:
        """
        A :obj:`PrintStyle` value that sets the format for tensor debug printing. Takes effect at run time.
        """

    @print_style.setter
    def print_style(self, arg: PrintStyle, /) -> None: ...
    @property
    def ir_output_dir(self) -> str:
        """
        A string path to the directory into which MAX dumps intermediate compiler IR for inspection. Empty string disables dumping. Takes effect at model build time.
        """

    @ir_output_dir.setter
    def ir_output_dir(self, arg: str, /) -> None: ...
    @property
    def sensible_mode(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to enable a curated default debugging set, including ``nan_check``, ``assert_level='all'``, ``device_sync_mode``, ``stack_trace_on_error``, ``stack_trace_on_crash``, and ``source_tracebacks``. You can override the defaults using individual properties.
        """

    @sensible_mode.setter
    def sensible_mode(self, arg: bool, /) -> None: ...
    def reset(self) -> None:
        """Reset all debug options to their defaults."""

class InferenceSession:
    """
    Manages compilation and execution of MAX models.

    An inference session holds device configuration and compiles graphs
    into executable :class:`Model` objects. It also manages custom
    extensions and debug options.

    .. code-block:: python

        from max._core.engine import InferenceSession
        from max import driver

        devices = [driver.CPU()]
        session = InferenceSession(devices, custom_extensions=[])
        model = session.compile_from_path("model.mef", [])
    """

    def __init__(
        self,
        devices: Sequence[max._core.driver.Device],
        custom_extensions: Sequence[str | os.PathLike],
        num_threads: int = 0,
    ) -> None:
        """
        Creates an inference session for model compilation and execution.

        Args:
            devices: List of devices used for compilation and execution.
            custom_extensions: Paths to custom Mojo extension libraries.
            num_threads (int): Number of execution threads. Defaults to 0,
                which lets the runtime choose automatically.
        """

    def _load_all(
        self, compiled: Model, weights_registry: Mapping[str, Any]
    ) -> list[Model]: ...
    def compile_from_path(
        self,
        model_path: str | os.PathLike,
        custom_extension_paths: Sequence[str | os.PathLike],
    ) -> Model:
        """
        Compiles a model from a file path.

        Args:
            model_path: Path to the compiled model file (for example, a ``.mef`` file).
            custom_extension_paths: Paths to custom Mojo extension libraries.

        Returns:
            Model: The compiled model ready for execution.
        """

    def compile_from_object(
        self,
        model: types.CapsuleType,
        custom_extensions: Sequence[str | os.PathLike],
        pipeline_name: str,
    ) -> Model:
        """
        Compiles a model from an in-memory capsule object.

        Args:
            model: A capsule containing the compiled model object.
            custom_extensions: Paths to custom Mojo extension libraries.
            pipeline_name: Name identifier for the compiled pipeline.

        Returns:
            Model: The compiled model ready for execution.
        """

    def set_debug_print_options(
        self, style: PrintStyle, precision: int, directory: str
    ) -> None:
        """
        Sets debug output options for tensor printing during execution.

        Args:
            style (PrintStyle): The output format style.
            precision (int): Number of decimal places for floating-point values.
            directory (str): Directory path for binary output files.
        """

    @overload
    def set_mojo_define(self, key: str, value: bool) -> None:
        """
        Sets a compile-time Mojo define to a boolean value.

        Args:
            key (str): The define name.
            value (bool): The boolean value to assign.
        """

    @overload
    def set_mojo_define(self, key: str, value: int) -> None:
        """
        Sets a compile-time Mojo define to an integer value.

        Args:
            key (str): The define name.
            value (int): The integer value to assign.
        """

    @overload
    def set_mojo_define(self, key: str, value: str) -> None:
        """
        Sets a compile-time Mojo define to a string value.

        Args:
            key (str): The define name.
            value (str): The string value to assign.
        """

    @property
    def devices(self) -> list[max._core.driver.Device]:
        """Returns the list of devices used by this session."""

    debug: DebugConfig

class PrintStyle(enum.Enum):
    """
    Controls the output format for debug tensor printing.

    Pass one of these values to :meth:`InferenceSession.set_debug_print_options`
    to configure how tensors are printed during execution.
    """

    COMPACT = 0
    """Compact human-readable format."""

    FULL = 1
    """Full verbose format with all tensor details."""

    BINARY = 2
    """Raw binary format."""

    BINARY_MAX_CHECKPOINT = 4
    """Binary checkpoint format compatible with MAX."""

    NONE = 3
    """Disables debug output."""
