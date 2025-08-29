# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
import inspect
import os
from collections.abc import Mapping
from typing import Any, Union, overload

import max._core.driver
import max._core.dtype
from max import mlir
from max._core.driver import Tensor
from max._core_types.driver import DLPackArray

InputType = Union[DLPackArray, Tensor, MojoValue, int, float, bool]

class TensorSpec:
    """
    Defines the properties of a tensor, including its name, shape and
    data type.

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

class MojoValue:
    pass

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
        Metadata about the model's input tensors, as a list of
        :obj:`TensorSpec` objects.

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
        Metadata about the model's output tensors, as a list of
        :obj:`TensorSpec` objects.

        For example, you can print the output tensor names, shapes, and dtypes:

        .. code-block:: python

            for tensor in model.output_metadata:
                print(f'name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')
        """

    @property
    def signature(self) -> inspect.Signature:
        """Get input signature for model."""

    def execute(self, *args: InputType) -> list[Tensor | MojoValue]:
        """
        Executes the model with the provided input and returns the outputs.

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
              :obj:`max.driver.Tensor`.
        """

    def __call__(
        self, *args: InputType, **kwargs: InputType
    ) -> list[Tensor | MojoValue]:
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
              * Max Driver tensors, i.e. :obj:`max.driver.Tensor`
              * Scalar inputs, i.e. :obj:`bool`, :obj:`float`, :obj:`int`,
                :obj:`np.generic`
              * Mojo value inputs, i.e. :obj:`MojoValue` (internal use)

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
              :obj:`max.driver.Tensor`.

            ValueError: If an input name does not correspond to what the model
              expects.

            ValueError: If any positional and named inputs collide.

            ValueError: If the number of inputs is less than what the model
              expects.
        """

    def __repr__(self) -> str: ...
    def _execute_device_tensors(
        self, *tensors: list[max._core.driver.Tensor | MojoValue]
    ) -> list[max._core.driver.Tensor | MojoValue]: ...
    def _export_mef(self, path: str) -> None:
        """
        Exports the compiled model as a mef to a file.

        Args:
          path: The filename where the mef is exported to.
        """

    def _load(self, weights_registry: Mapping[str, Any]) -> None: ...

class InferenceSession:
    def __init__(self, config: dict = {}) -> None: ...
    def compile_from_path(
        self, model_path: str | os.PathLike, config: dict = {}
    ) -> Model: ...
    def compile_from_object(
        self, model: object, config: dict = {}
    ) -> Model: ...
    def set_debug_print_options(
        self, style: PrintStyle, precision: int, directory: str
    ) -> None: ...
    @overload
    def set_mojo_define(self, key: str, value: bool) -> None: ...
    @overload
    def set_mojo_define(self, key: str, value: int) -> None: ...
    @overload
    def set_mojo_define(self, key: str, value: str) -> None: ...
    def set_kernel_logging(self, enabled: bool) -> None:
        """Enable or disable kernel execution logging"""

    def register_runtime_context(self, ctx: mlir.Context) -> None: ...
    @property
    def devices(self) -> list[max._core.driver.Device]: ...

class PrintStyle(enum.Enum):
    COMPACT = 0

    FULL = 1

    BINARY = 2

    BINARY_MAX_CHECKPOINT = 3

    NONE = 4
