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
    """ModelKind represents the to-be-loaded model's framework"""

    PYTORCH_MODEL = "PyTorch Model"
    TENSORFLOW_MODEL = "TensorFlow Model"
    TFLITE_MODEL = "TFLite Model"
    UNKNOWN_MODEL = "Unknown Model"


class TFSavedModelVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"


@dataclass
class TensorFlowLoadOptions:
    """TensorFlowLoadOptions is a class that can be used to configure loading of TensorFlow saved models"""

    saved_model_version: TFSavedModelVersion = field(
        default=TFSavedModelVersion.V1
    )
    exported_name: str = field(default="serving_default")
    compatibility_mode: bool = field(default=False)


@dataclass
class TFLiteLoadOptions:
    """TFLiteLoadOptions is a class that can be used to configure loading of TFLite models"""


@dataclass
class TorchLoadOptions:
    """TorchLoadOptions is a class that can be used to configure loading of PyTorch models"""


class Model:
    """A Model object represents a loaded model.

    Model instances are created by calling load on a native framework model
    (e.g., a Tensorflow SavedModel) using an instance of the InferenceSession
    class.
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

    def init(self) -> None:
        """Initializes the loaded model and makes it ready for execution."""
        self._impl.load()

    @property
    def input_metadata(self) -> List["TensorSpec"]:
        """Metadata about input tensors the model expects."""
        return [TensorSpec._init(spec) for spec in self._impl.input_metadata]

    @property
    def kind(self) -> str:
        return self._kind.value


class DType(Enum):
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
    """An InferenceSession object represents a session for compiling and executing models.

    Parameters
    ----------
    config: dict
        Options to configure the inference session. Keys include - number of threads.
    """

    _impl: _InferenceSession

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self._impl = _InferenceSession(config)

    def load(
        self,
        model_path: Path,
        options: Optional[
            Union[TensorFlowLoadOptions, TorchLoadOptions, TFLiteLoadOptions]
        ] = None,
    ) -> Model:
        """Load a saved model file/directory

        We support compiling Tensorflow models in the SavedModel format and traceable PyTorch models

        Parameters
        ----------
        model_path: pathlib.Path
            Path to the saved model

        Returns
        -------
        Model
            A loaded model

        Raises
        ------
        RuntimeError
            We raise an exception if the path provided is invalid.
        """
        options_dict = {}
        if options:
            options_dict = asdict(options)
            if isinstance(options, TensorFlowLoadOptions):
                options_dict["type"] = "tf"
            elif isinstance(options, TorchLoadOptions):
                options_dict["type"] = "torch"
            elif isinstance(options, TFLiteLoadOptions):
                options_dict["type"] = "tflite"
            else:
                raise TypeError("Invalid compilation options object.")

        _model = self._impl.compile(model_path, options_dict)

        # TODO: Fix when real PyTorch support lands
        if model_path.suffix == ".onnx":
            model_kind = ModelKind.PYTORCH_MODEL
        elif model_path.suffix == ".tflite":
            model_kind = ModelKind.TFLITE_MODEL
        elif os.path.isdir(model_path):
            model_kind = ModelKind.TENSORFLOW_MODEL
        else:
            model_kind = ModelKind.UNKNOWN_MODEL

        return Model._init(_model, model_kind)


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
