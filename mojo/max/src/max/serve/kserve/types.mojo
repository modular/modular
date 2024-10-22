# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Includes common data types for KServe services."""

from sys.ffi import DLHandle

from max.engine import InferenceSession, TensorMap

from ._types import (
    _get_tensors,
    _set_tensors,
    CInferenceRequest,
    CInferenceResponse,
)
from .._batch import CBatch


# ===----------------------------------------------------------------------=== #
# InferenceRequest
# ===----------------------------------------------------------------------=== #


struct InferenceRequest(Movable):
    var _impl: CInferenceRequest
    var _session: InferenceSession

    fn __init__(
        inout self,
        owned impl: CInferenceRequest,
        owned session: InferenceSession,
    ):
        self._impl = impl^
        self._session = session^

    fn __moveinit__(inout self, owned existing: Self):
        self._impl = existing._impl^
        self._session = existing._session^

    fn get_model_name(self) raises -> String:
        return str(self._impl.model_name())

    fn get_model_version(self) raises -> String:
        return str(self._impl.model_version())

    fn get_input_tensors(self) raises -> TensorMap:
        """Returns all input tensors.

        Returns:
            A tensor map containing all input tensors.
        """
        return _get_tensors[
            CInferenceRequest._InputsSizeFnName,
            CInferenceRequest._InputAtFnName,
        ](self._impl._lib, self._impl._ptr, self._session)

    fn set_input_tensors(self, names: List[String], map: TensorMap) raises:
        """Sets input tensors to those with matching names from map.

        Args:
            names: Names of selected input tensors.
            map: Map of all input tensors.
        """
        _set_tensors[CInferenceRequest._AddInputFnName,](
            self._impl._lib, self._impl._ptr, names, map
        )

    fn get_outputs(self) -> List[String]:
        """Returns all requested output names.

        Returns:
            A list containing all requested output names.
        """
        # TODO: Pass back an array.
        var result = List[String](capacity=int(self._impl.outputs_size()))
        for i in range(self._impl.outputs_size()):
            result.append(self._impl.output_at(i).__str__())
        return result^

    fn set_outputs(self, outputs: List[String]) -> None:
        for output in outputs:
            self._impl.add_output(output[].as_string_slice())


# ===----------------------------------------------------------------------=== #
# InferenceResponse
# ===----------------------------------------------------------------------=== #


struct InferenceResponse(Movable):
    var _impl: CInferenceResponse
    var _session: InferenceSession

    fn __init__(
        inout self,
        owned impl: CInferenceResponse,
        owned session: InferenceSession,
    ):
        self._impl = impl^
        self._session = session^

    fn __moveinit__(inout self, owned existing: Self):
        self._impl = existing._impl^
        self._session = existing._session^

    fn get_output_tensors(self) raises -> TensorMap:
        """Returns all output tensors.

        Returns:
            A tensor map containing all output tensors.
        """
        return _get_tensors[
            CInferenceResponse._OutputsSizeFnName,
            CInferenceResponse._OutputAtFnName,
        ](self._impl._lib, self._impl._ptr, self._session)

    fn set_output_tensors(self, names: List[String], map: TensorMap) raises:
        """Sets output tensors to those with matching names from map.

        Args:
            names: Names of selected output tensors.
            map: Map of all output tensors.
        """
        _set_tensors[CInferenceResponse._AddOutputFnName](
            self._impl._lib, self._impl._ptr, names, map
        )


# ===----------------------------------------------------------------------=== #
# InferenceBatch
# ===----------------------------------------------------------------------=== #


struct InferenceBatch(Sized, Movable):
    var _impl: CBatch
    var _session: InferenceSession

    fn __init__(inout self, lib: DLHandle, owned session: InferenceSession):
        self._impl = CBatch(lib)
        self._session = session^

    fn __moveinit__(inout self: Self, owned existing: Self):
        self._impl = existing._impl^
        self._session = existing._session^

    fn __len__(self) -> Int:
        return int(self._impl.size())

    fn request_at(self, index: Int64) -> InferenceRequest:
        return InferenceRequest(
            self._impl.request_at[CInferenceRequest](index),
            self._session,
        )

    fn response_at(self, index: Int64) -> InferenceResponse:
        return InferenceResponse(
            self._impl.response_at[CInferenceResponse](index),
            self._session,
        )
