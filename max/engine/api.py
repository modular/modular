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


class ModelKind(str, Enum):
    """The model format."""

    PYTORCH_MODEL = "PyTorch Model"
    TENSORFLOW_MODEL = "TensorFlow Model"
    UNKNOWN_MODEL = "Unknown Model"


@dataclass
class TensorFlowLoadOptions:
    """Configures how to load TensorFlow saved models."""

    exported_name: str = field(default="serving_default")
    """The exported name from the TF model's signature."""

    compatibility_mode: bool = field(default=False)
    """Indicates whether or not the model will fall back to using
    TF kernels."""


@dataclass
class TorchLoadOptions:
    """Configures how to load PyTorch models."""


class Model:
    """A loaded model that you can execute with the Modular Engine.

    You should not instantiate this class directly. Instead create a
    :obj:`Model` by passing your model file to :func:`InferenceSession.load()`.
    Then you can run the model by passing your input data to
    :func:`Model.execute()`.
    """

    _impl: _Model
    _kind: ModelKind

    @classmethod
    def _init(cls, _core_model, _kind: ModelKind):
        model = cls()
        model._impl = _core_model
        model._kind = _kind
        return model

    def execute(self, *args) -> None:
        """Executes the model with the provided input tensors and returns
        outputs as tensors.

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

    def init(self) -> None:
        """Initializes the loaded model and prepares it for execution.

        Calling this is optional to warm-up the model; you can instead just
        call :func:`execute()` and it will call this for you."""
        self._impl.load()

    @property
    def input_metadata(self) -> List["TensorSpec"]:
        """Metadata about the input tensors that the model expects."""
        return [TensorSpec._init(spec) for spec in self._impl.input_metadata]

    @property
    def kind(self) -> str:
        """The :obj:`ModelKind` value."""
        return self._kind.value


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
    config: dict
        Configuration details about the inference session, such as the number
        of threads.
    """

    _impl: _InferenceSession

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self._impl = _InferenceSession(config)

    def load(
        self,
        model_path: Path,
        options: Optional[
            Union[TensorFlowLoadOptions, TorchLoadOptions]
        ] = None,
    ) -> Model:
        """Loads a trained model and compiles it for inference.

        Parameters
        ----------
        model_path: pathlib.Path
            Path to a model. May be a Tensorflow model in the SavedModel
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
        options_dict = {}
        if options:
            options_dict = asdict(options)
            if isinstance(options, TensorFlowLoadOptions):
                options_dict["type"] = "tf"
            elif isinstance(options, TorchLoadOptions):
                options_dict["type"] = "torch"
            else:
                raise TypeError("Invalid compilation options object.")

        # TODO(#12573): Remove this (demo compatibility)
        model_path = Path(str(model_path).replace(".pt", ".onnx"))
        _model = self._impl.compile(model_path, options_dict)

        if model_path.suffix == ".onnx":
            model_kind = ModelKind.PYTORCH_MODEL
        elif os.path.isdir(model_path):
            model_kind = ModelKind.TENSORFLOW_MODEL
        else:
            model_kind = ModelKind.UNKNOWN_MODEL

        return Model._init(_model, model_kind)


class TensorSpec:
    """Represents a tensor's specifications, namely its shape and data type."""

    _impl: _TensorSpec

    @classmethod
    def _init(cls, _core_tensor_spec):
        tensor_spec = cls()
        tensor_spec._impl = _core_tensor_spec
        return tensor_spec

    @property
    def shape(self) -> List[int]:
        """The shape of the tensor as a list of integers.

        If a dimension is indeterminate for a certain axis, like the first
        axis of a batched tensor, that axis is denoted by ``None``."""
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
