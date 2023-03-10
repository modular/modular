# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from enum import Enum
from pathlib import Path
from sys import version_info

import modular.engine.core as _mecore
from modular.engine.core import DType as _DType
from modular.engine.core import InferenceSession as _InferenceSession
from modular.engine.core import Model as _Model
from modular.engine.core import TensorSpec as _TensorSpec

if version_info.minor <= 8:
    from typing import List
else:
    List = list

version_string = _mecore.__version__


class Model:
    """A Model object represents a compiled model.

    Model instances are created by calling compile on a native framework model
    (e.g., a Tensorflow SavedModel) using an instance of the InferenceSession
    class.
    """

    _impl: _Model

    @classmethod
    def _init(cls, _core_model):
        model = cls()
        model._impl = _core_model
        return model

    def execute(self, *args) -> None:
        """Executes the model for the provided input tensors and returns outputs as tensors.

        Parameters
        ----------
        ``*args``
            Input tensors, objects of NumPy `ndarray` or TensorFlow `Tensor` type.

        Returns
        -------
        np.ndarray
            Output tensors have type np.ndarray

        Raises
        ------
        RuntimeError
            We raise an exception if input tensors don't match what the model expects.
        """

        # TODO(10070): Fix above docstring after all desired tensor types are supported.
        # The type would likely be Union[np.ndarray, torch.Tensor, tf.Tensor]
        return self._impl.execute(*args)

    def load(self) -> None:
        """Loads the compiled model and makes it ready for execution."""
        self._impl.load()

    @property
    def input_metadata(self) -> List["TensorSpec"]:
        """Metadata about input tensors the model expects."""
        return [TensorSpec._init(spec) for spec in self._impl.input_metadata]


class DType(Enum):
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
    double = 11

    @classmethod
    def _from(cls, dtype: _DType):
        obj = cls.__dict__[dtype.name]
        return obj


class InferenceSession:
    """An InferenceSession object represents a session for compiling and executing models.

    Parameters
    ----------
    config: dict
        Options to configure the inference session. Keys include - number of threads.
    """

    _impl: _InferenceSession

    def __init__(self, config: dict = {}):
        self._impl = _InferenceSession(config)

    def compile(self, model_path: Path, config: dict = {}) -> Model:
        """Compile a saved model file/directory

        We support compiling Tensorflow models in the SavedModel format and traceable PyTorch models

        Parameters
        ----------
        model_path: pathlib.Path
            Path to the saved model

        Returns
        -------
        Model
            A compiled model

        Raises
        ------
        RuntimeError
            We raise an exception if the path provided is invalid.
        """
        _model = self._impl.compile(model_path, config)
        return Model._init(_model)


class TensorSpec:
    """A TensorSpec object represents a tensor's specifications, namely it's shape and data type."""

    _impl: _TensorSpec

    @classmethod
    def _init(cls, _core_tensor_spec):
        tensor_spec = cls()
        tensor_spec._impl = _core_tensor_spec
        return tensor_spec

    @property
    def shape(self) -> List[int]:
        """The shape of the tensor returned as a list of integers.

        If the dimension is indeterminate for a certain axis, like the first
        axis of a batched tensor - it is denoted by `None`."""
        return self._impl.shape

    @property
    def dtype(self) -> DType:
        """Returns the data type of the tensor"""
        return DType._from(self._impl.dtype)


def remove_annotations(cls):
    del cls.__annotations__
    return cls


remove_annotations(Model)
remove_annotations(InferenceSession)
remove_annotations(TensorSpec)
