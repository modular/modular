# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides a basic multi-model server."""

from utils.variant import Variant
from max.engine import InferenceSession, InputSpec, Model, TensorMap

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

    alias model_dict_type = Dict[String, UnsafePointer[Model]]

    var _models: List[Model]
    var _model_dict: Self.model_dict_type
    var _session: InferenceSession

    fn __init__(
        inout self,
        models: List[FileModel],
        owned session: InferenceSession,
    ) raises:
        self._session = session^
        self._model_dict = Self.model_dict_type()
        self._models = List[Model](capacity=len(models))
        for model in models:
            var name = model[].name
            if name not in self._model_dict:
                var m = self._session.load(
                    model[].path, input_specs=model[].input_specs
                )
                var p = UnsafePointer[Model].alloc(1)
                p.init_pointee_move(m^)
                self._model_dict[name] = p
            else:
                raise Error(
                    "Cannot add duplicate version: "
                    + ""
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
        if name not in self._model_dict:
            raise "model not found!"
        var model = self._model_dict[name]
        var inputs = request.get_input_tensors()
        var outputs = model[].execute(inputs)

        return outputs^

    fn __del__(owned self: Self):
        _ = self._models^
        _ = self._model_dict^
        _ = self._session^
