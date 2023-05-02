# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
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

from dataclasses import asdict, dataclass, field
from typing import Optional, Union

version_string = _mecore.__version__


class Model:
    """A loaded model that you can execute.

    You should not instantiate this class directly. Instead create a
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

    def execute(self, *args) -> None:
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

        # TODO(10070): Fix above docstring after all desired tensor types are supported.
        # The type would likely be Union[np.ndarray, torch.Tensor, tf.Tensor]
        return self._impl.execute(*args)

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


class DType(Enum):
    """The tensor data type."""

    bool = 0
    si8 = 1
    si16 = 2
    si32 = 3
    si64 = 4
    ui8 = 5
    ui16 = 6
    ui32 = 7
    ui64 = 8
    f16 = 9
    f32 = 10
    f64 = 11

    @classmethod
    def _from(cls, dtype: _DType):
        obj = cls.__dict__[dtype.name]
        return obj


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
        if num_threads:
            config = {"num_threads": num_threads}
        self._impl = _InferenceSession(config)

    def load(self, model_path: Union[str, Path]) -> Model:
        """Loads a trained model and compiles it for inference.

        Parameters
        ----------
        model_path: Union[str, pathlib.Path]
            Path to a model. May be a TensorFlow model in the SavedModel
            format or a traceable PyTorch model.

        Returns
        -------
        Model
            The loaded model, compiled and ready to execute.

        Raises
        ------
        RuntimeError
            If the path provided is invalid.
        """
        model_path = Path(str(model_path))
        _model = self._impl.compile(model_path)
        _model.load()
        return Model._init(_model)


class TensorSpec:
    """
    Defines the properties of a tensor, namely its shape and data type.

    You can get a list of ``TensorSpec`` objects that specify the input tensors
    of a :obj:`Model` from :func:`Model.input_metadata`.
    """

    _impl: _TensorSpec

    @classmethod
    def _init(cls, _core_tensor_spec):
        tensor_spec = cls()
        tensor_spec._impl = _core_tensor_spec
        return tensor_spec

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


def remove_annotations(cls):
    del cls.__annotations__
    return cls


remove_annotations(Model)
remove_annotations(InferenceSession)
remove_annotations(TensorSpec)
