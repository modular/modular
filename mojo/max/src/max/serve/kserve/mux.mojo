# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides a basic multi-model server."""

from collections import Dict, Optional
from utils.variant import Variant
from max.engine import InferenceSession, InputSpec, Model, TensorMap
from memory import UnsafePointer

from .service import InferenceService
from .types import InferenceRequest, InferenceResponse, CInferenceResponse


@value
struct FileModel:
    """File-backed model artifact details."""

    var name: String
    """The name of the model."""
    var version: String
    """The version of the model."""

    var path: String
    """A string representing the path of the model file."""
    var input_specs: Optional[List[InputSpec]]
    """ The input specification of the model."""


struct MuxInferenceService(InferenceService):
    """Inference service that multiplexes across a list of models."""

    alias versions_dict_type = Dict[String, UnsafePointer[Model]]
    alias model_dict_type = Dict[String, UnsafePointer[Self.versions_dict_type]]

    var _models: List[Model]
    var _version_dicts: List[Self.versions_dict_type]
    var _model_dict: Self.model_dict_type
    var _session: InferenceSession

    fn __init__(
        inout self,
        models: List[FileModel],
        owned session: InferenceSession,
    ) raises:
        self._session = session^
        self._model_dict = Self.model_dict_type()
        self._version_dicts = List[Self.versions_dict_type](
            capacity=len(models)
        )
        self._models = List[Model](capacity=len(models))
        for model in models:
            var name = model[].name
            if name not in self._model_dict:
                self._version_dicts.append(Self.versions_dict_type())
                var back = self._version_dicts[-1]
                self._model_dict[name] = UnsafePointer.address_of(back)

            var version = model[].version
            var versioned = self._model_dict[name]
            if version not in versioned[]:
                self._models.append(
                    self._session.load(
                        model[].path, input_specs=model[].input_specs
                    )
                )
                var back = self._models[-1]
                versioned[][version] = UnsafePointer.address_of(back)
            else:
                raise Error(
                    "Cannot add duplicate version: "
                    + version
                    + " for model: "
                    + name
                )

    fn init[
        add_models: fn (List[Model]) capturing raises -> None
    ](inout self) raises -> None:
        add_models(self._models)

    fn infer(
        inout self, request: InferenceRequest, inout response: InferenceResponse
    ) raises -> None:
        var outputs = self.infer(request)
        response.set_output_tensors(request.get_outputs(), outputs^)

    fn infer(inout self, request: InferenceRequest) raises -> TensorMap:
        var name = request.get_model_name()
        var version = request.get_model_version()
        var model = self._model_dict[name][][version]

        var inputs = request.get_input_tensors()
        var outputs = model[].execute(inputs)

        return outputs^
