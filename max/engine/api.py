# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import asdict, dataclass, field
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


@dataclass
class TorchLoadOptions:
    """Configures how to load PyTorch models."""


class Model:
    """A loaded model that you can execute.

    You should not instantiate this class directly. Instead, create a
    :obj:`Model` by passing your model file to :func:`InferenceSession.load()`.
    Then you can run the model by passing your input data to
    :func:`Model.execute()`.
    """

    _impl: _Model

    @classmethod
    def _init(cls, _core_model):
        model = cls()
        model._impl = _core_model
        return model

    def execute(self, **kwargs) -> Dict[str, np.ndarray]:
        """Executes the model with the provided input and returns outputs.

        Parameters
        ----------
        ``*args``
            Input tensors as :obj:`np.ndarray` data.

        Returns
        -------
        np.ndarray
            Output tensors.

        Raises
        ------
        RuntimeError
            If the input tensors don't match what the model expects.
        """
        return self._impl.execute(**kwargs)

    def __repr__(self) -> str:
        return f"Model(inputs={self.input_metadata})"

    @property
    def input_metadata(self) -> List["TensorSpec"]:
        """
        Metadata about the input tensors that the model accepts.

        You can use this to query the tensor shapes and data types like this:

        .. code-block:: python

            for tensor in model.input_metadata:
                print(f'shape: {tensor.shape}, dtype: {tensor.dtype}')
        """
        return [TensorSpec._init(spec) for spec in self._impl.input_metadata]

    @property
    def output_metadata(self) -> List["TensorSpec"]:
        """
        Metadata about the output tensors that the model returns.

        You can use this to query the tensor shapes and data types like this:

        .. code-block:: python

            for tensor in model.ouput_metadata:
                print(f'shape: {tensor.shape}, dtype: {tensor.dtype}')
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


class TensorSpec:
    """
    Defines the properties of a tensor, namely its shape and data type.

    You can get a list of ``TensorSpec`` objects that specify the input tensors
    of a :obj:`Model` from :func:`Model.input_metadata`.
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
        mlir_shape = [str(dim) if dim else "-1" for dim in self.shape]
        shape_str = "x".join(mlir_shape)
        return f"{shape_str}x{self.dtype.name}"

    @property
    def shape(self) -> List[int]:
        """The shape of the tensor as a list of integers.

        If a dimension is indeterminate for a certain axis, such as the first
        axis of a batched tensor, that axis is denoted by ``None``."""
        return self._impl.shape

    @property
    def dtype(self) -> DType:
        """A tensor data type."""
        return DType._from(self._impl.dtype)

    @property
    def name(self) -> str:
        """A tensor name."""
        return self._impl.name


@dataclass
class TorchLoadOptions:
    """Configures how to load PyTorch models."""

    input_specs: List[TensorSpec] = field(default_factory=list)
    """The tensor specifications (shape and data type) for each of the
    model inputs. Required for lowering serialized TorchScript models which
    have no type and shape annotations.
    """


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
        options: Optional[
            Union[TensorFlowLoadOptions, TorchLoadOptions]
        ] = None,
    ) -> Model:
        """Loads a trained model and compiles it for inference.

        Parameters
        ----------
        model_path: Union[str, pathlib.Path]
            Path to a model. May be a TensorFlow model in the SavedModel
            format or a traceable PyTorch model.

        options: Optional[TorchLoadOptions]
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
        if options:
            # Unwrap dataclass options to a dictionary, and furthermore unwrap
            # any nested objects that are not pybind classes so that the values
            # passed to the compile implementation can be interpreted.
            options_dict = asdict(
                options, dict_factory=_unwrap_pybind_objects_dict_factory
            )
            if isinstance(options, TensorFlowLoadOptions):
                options_dict["type"] = "tf"
            elif isinstance(options, TorchLoadOptions):
                options_dict["type"] = "torch"
            else:
                raise TypeError("Invalid compilation options object.")

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
