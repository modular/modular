# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from sys import version_info
from typing import Any, Optional, Type, Union

import numpy as np
from max.engine.core import DType as _DType
from max.engine.core import InferenceSession as _InferenceSession
from max.engine.core import Model as _Model
from max.engine.core import TensorSpec as _TensorSpec
from max.engine.core import TorchInputSpec as _TorchInputSpec
from max.engine.core import __version__

if version_info.minor <= 8:
    from typing import Dict, List, Tuple
else:
    Dict = dict
    List = list
    Tuple = tuple

version_string = __version__


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

    def execute(
        self, *args, **kwargs
    ) -> Dict[str, Union[np.ndarray, Dict, List, Tuple]]:
        """Executes the model with the provided input and returns the outputs.

        For example, if the model has one input tensor named "input":

        .. code-block:: python

            input_tensor = np.random.rand(1, 224, 224, 3)
            model.execute(input=input_tensor)

        Parameters
        ----------
        ``args``
            Currently not supported. You must specify inputs using ``kwargs``.
        ``kwargs``
            The input tensors, each specified with the appropriate tensor name
            as a keyword and its value as an :obj:`np.ndarray`. You can find the
            tensor names to use as keywords from :obj:`~Model.input_metadata`.

        Returns
        -------
        Dict
            A dictionary of output values, each as an :obj:`np.ndarray`,
            :obj:`Dict`, :obj:`List`, or :obj:`Tuple` identified by its output
            name.

        Raises
        ------
        RuntimeError
            If the given input tensors' name and shape don't match what
            the model expects.

        TypeError
            If the given input tensors' dtype cannot be cast to what the model
            expects.
        """
        if args:
            raise RuntimeError(
                "Execute API only accepts keyword arguments e.g. outs ="
                " model.execute(arg0=np.ones((1,"
                " 10)).astype(np.float32)).Keywords have to be tensor names"
                " which can be queried using the input_metadata API."
            )
        dtype_map = {spec.name: spec.dtype for spec in self.input_metadata}
        for input_name, input_value in kwargs.items():
            if dtype_map[input_name] == DType.unknown:
                # This currently indicates that the value expected by the model
                # internally is not a `GML::Tensor`. We pass the value as-is,
                # since no metadata is available to check the runtime values
                # against.
                kwargs[input_name] = input_value
            elif not isinstance(input_value, np.ndarray):
                # Indicates that the model expects an ndarray
                # (internally `GML::Tensor`), but if the input
                # isn't already an ndarray, then we can attempt to interpret it
                # as a scalar primitive that needs to be converted to an
                # ndarray.
                kwargs[input_name] = np.array(input_value)
            elif input_value.dtype != dtype_map[input_name]:
                try:
                    # the default casting settings of NumPy
                    # are extremely liberal - all data conversions are allowed
                    # e.g. uint8 -> float64 and vice versa. We use `same_kind`
                    # casting instead which casts within the same numerics class.
                    # This can be made stricter to only allow casting that
                    # preserves values.
                    #
                    # NumPy casting creates a copy which is a runtime cost, but
                    # we only pay for it in the event of a dtype mismatch.
                    kwargs[input_name] = input_value.astype(
                        dtype_map[input_name]._to_np_dtype(),
                        casting="same_kind",
                    )
                except TypeError:
                    # We don't expect this branch to be taken in practice but it
                    # is added for completeness' sake. NumPy can theoretically
                    # raise a ComplexWarning but complex numpy tensors are
                    # super rare in ML.
                    raise TypeError(
                        f"Input dtype {input_value.dtype} not compatible with"
                        f" {dtype_map[input_name]}"
                        " required for model"
                        " execution."
                    )
        return self._impl.execute(**kwargs)

    def __repr__(self) -> str:
        return f"Model(inputs={self.input_metadata})"

    @property
    def input_metadata(self) -> List["TensorSpec"]:
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
    def output_metadata(self) -> List["TensorSpec"]:
        """
        Metadata about the model's output tensors, as a list of
        :obj:`TensorSpec` objects.

        For example, you can print the output tensor names, shapes, and dtypes:

        .. code-block:: python

            for tensor in model.ouput_metadata:
                print(f'name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')
        """
        return [TensorSpec._init(spec) for spec in self._impl.output_metadata]


class DType(Enum):
    """The tensor data type."""

    bool = 0
    int8 = 1
    int16 = 2
    int32 = 3
    int64 = 4
    uint8 = 5
    uint16 = 6
    uint32 = 7
    uint64 = 8
    float16 = 9
    float32 = 10
    float64 = 11
    unknown = 12

    @classmethod
    def _from(cls, dtype: _DType):
        obj = cls.__dict__[dtype.name]
        return obj

    def _to(self):
        return _DType.__dict__[self.name]

    def __repr__(self) -> str:
        return self.name

    def _to_np_dtype(self):
        if self == DType.bool:
            return np.bool_
        elif self == DType.int8:
            return np.int8
        elif self == DType.int16:
            return np.int16
        elif self == DType.int32:
            return np.int32
        elif self == DType.int64:
            return np.int64
        elif self == DType.uint8:
            return np.uint8
        elif self == DType.uint16:
            return np.uint16
        elif self == DType.uint32:
            return np.uint32
        elif self == DType.uint64:
            return np.uint64
        elif self == DType.float16:
            return np.float16
        elif self == DType.float32:
            return np.float32
        elif self == DType.float64:
            return np.float64
        return None


class TensorSpec:
    """
    Defines the properties of a tensor, including its name, shape and data type.

    For usage examples, see :obj:`Model.input_metadata`.
    """

    _impl: _TensorSpec

    def __init__(
        self, shape: Optional[List[Optional[int]]], dtype: DType, name: str
    ):
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
    def shape(self) -> Optional[List[int]]:
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

    def __init__(self, shape: Optional[List[Union[int, str]]], dtype: DType):
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
    def shape(self) -> Optional[List[int]]:
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


class InferenceSession:
    """Manages an inference session in which you can load and run models.

    You need an instance of this to load a model as a :obj:`Model` object.
    For example:

    .. code-block:: python

        session = engine.InferenceSession()
        model_path = Path('bert-base-uncased')
        model = session.load(model_path)

    Parameters
    ----------
    num_threads: Optional[int]
        Number of threads to use for the inference session. This parameter
        defaults to the number of physical cores on your machine.
    """

    _impl: _InferenceSession

    def __init__(self, num_threads: Optional[int] = None, **kwargs):
        config = {}
        self.num_threads = num_threads
        if num_threads:
            config["num_threads"] = num_threads
        device = kwargs["device"] if kwargs and "device" in kwargs else None
        if device:
            config["device"] = device
        if "custom_ops_path" in kwargs:
            config["custom_ops_path"] = str(kwargs["custom_ops_path"])
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
        model_path: Union[str, Path],
        *,
        custom_ops_path: Optional[str] = None,
        input_specs: Optional[List["TorchInputSpec"]] = None,
    ) -> Model:
        """Loads a trained model and compiles it for inference.

        Note: PyTorch models must be in TorchScript format.

        Parameters
        ----------
        model_path: Union[str, pathlib.Path]
            Path to a model. May be a TorchScript model or an ONNX model.

        custom_ops_path: str
            The path to your custom ops Mojo package.

        input_specs:
            The tensor specifications (shape and data type) for each of the
            model inputs. This is required when loading serialized TorchScript
            models because they do not include type and shape annotations.

            If the model supports an input with dynamic shapes, use ``None`` as
            the dimension size in ``shape``.

            For example:

            .. code-block:: python

                session = engine.InferenceSession()
                model = session.load(
                    "clip-vit.torchscript",
                    input_specs = [
                        engine.TorchInputSpec(
                            shape=[1, 16], dtype=engine.DType.int32
                        ),
                        engine.TorchInputSpec(
                            shape=[1, 3, 224, 224], dtype=engine.DType.float32
                        ),
                        engine.TorchInputSpec(
                            shape=[1, 16], dtype=engine.DType.int32
                        ),
                    ],
                )

        Returns
        -------
        Model
            The loaded model, compiled and ready to execute.

        Raises
        ------
        RuntimeError
            If the path provided is invalid.
        """

        options_dict = {}

        if custom_ops_path is not None:
            options_dict["custom_ops_path"] = _unwrap_pybind_objects(
                custom_ops_path
            )
        if input_specs is not None:
            options_dict["input_specs"] = _unwrap_pybind_objects(input_specs)

        model_path = Path(str(model_path))
        _model = self._impl.compile(model_path, options_dict)
        _model.load()
        return Model._init(_model)

    def get_torch_custom_op_schemas(self):
        return self._impl.get_torch_custom_op_schemas()


def remove_annotations(cls: Type) -> Type:
    del cls.__annotations__
    return cls


remove_annotations(Model)
remove_annotations(InferenceSession)
remove_annotations(TensorSpec)
remove_annotations(TorchInputSpec)
