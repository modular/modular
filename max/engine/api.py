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

import modular.engine.core as _mecore
from modular.engine.core import DType as _DType
from modular.engine.core import InferenceSession as _InferenceSession
from modular.engine.core import Model as _Model
from modular.engine.core import TensorSpec as _TensorSpec

if version_info.minor <= 8:
    from typing import Dict, List, Tuple
else:
    Dict = dict
    List = list
    Tuple = tuple

version_string = _mecore.__version__


@dataclass
class TensorFlowLoadOptions:
    """Configures how to load TensorFlow models."""

    exported_name: str = field(default="serving_default")
    """The exported name from the TensorFlow model's signature."""

    type: str = "tf"


@dataclass
class CommonLoadOptions:
    """Common options for how to load models."""

    custom_ops_path: str = field(default="")
    """The path from which to load custom ops."""


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

    def execute(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Executes the model with the provided input and returns outputs.

        For example, if the model has one input tensor named "input":

        .. code-block:: python

            input_tensor = np.random.rand(1, 224, 224, 3)
            model.execute(input=input_tensor)

        Parameters
        ----------
        ``kwargs``
            The input tensors, each specified with the approprite tensor name
            as a keyword and passed as an :obj:`np.ndarray`. You can find the
            model's tensor names with :obj:`~Model.input_metadata`.

        Returns
        -------
        Dict
            A dictionary of output tensors, each as an :obj:`np.ndarray`
            identified by its tensor name.

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
            if input_value.dtype != dtype_map[input_name]:
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


class TensorSpec:
    """
    Defines the properties of a tensor, including its name, shape and data type.

    For usage examples, see :obj:`Model.input_metadata` and
    :obj:`Model.output_metadata`.
    """

    _impl: _TensorSpec

    def __init__(self, shape: List[Optional[int]], dtype: DType, name: str):
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
        mlir_shape = [
            str(dim) if dim is not None else "-1" for dim in self.shape
        ]
        shape_str = "x".join(mlir_shape)
        return f"{shape_str}x{self.dtype.name}"

    @property
    def shape(self) -> List[int]:
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


def _unwrap_pybind_objects_dict_factory(
    data: List[Tuple[str, Any]]
) -> Dict[str, Any]:
    """Unwraps pybind objects from python class wrappers."""

    def convert(value: Any) -> Union[List[Any], _TensorSpec, Any]:
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, TensorSpec):
            # Unwrap TensorSpec to _TensorSpec.
            return value._impl
        return value

    return {field: convert(value) for field, value in data}


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

    def __init__(self, num_threads: Optional[int] = None):
        config = {}
        self.num_threads = num_threads
        if num_threads:
            config = {"num_threads": num_threads}
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
        *options: Union[TensorFlowLoadOptions, CommonLoadOptions],
        **kwargs,
    ) -> Model:
        """Loads a trained model and compiles it for inference.

        Parameters
        ----------
        model_path: Union[str, pathlib.Path]
            Path to a model. May be a TensorFlow model in the SavedModel
            format or a traceable PyTorch model.

        *options: Union[TensorFlowLoadOptions, CommonLoadOptions]
            Load options for configuring how the model should be compiled.

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
        for options_obj in options:
            if not is_dataclass(options_obj):
                raise TypeError(
                    "Invalid load options object; must be dataclass."
                )
            # Unwrap dataclass options to a dictionary, and furthermore unwrap
            # any nested objects that are not pybind classes so that the values
            # passed to the compile implementation can be interpreted.
            options_dict.update(
                asdict(
                    options_obj,
                    dict_factory=_unwrap_pybind_objects_dict_factory,
                ).items()
            )

        model_path = Path(str(model_path))
        _model = self._impl.compile(model_path, options_dict)
        _model.load()
        return Model._init(_model)


def remove_annotations(cls: Type) -> Type:
    del cls.__annotations__
    return cls


remove_annotations(Model)
remove_annotations(InferenceSession)
remove_annotations(TensorSpec)
