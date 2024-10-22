# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides basic KServe client functionality."""

from sys.ffi import DLHandle
from runtime.asyncrt import run, ChainPromise
from utils.variant import Variant

from max.engine import InferenceSession, TensorMap
from max._utils import handle_from_config

from ._types import CInferenceResponse, CInferenceRequest
from .types import InferenceResponse, InferenceRequest
from ._client import CGRPCClient, ClientResult


struct GRPCClient:
    """Inference client implementing the KServe protocol over gRPC."""

    var _lib: DLHandle
    var _impl: CGRPCClient
    var _session: InferenceSession

    fn __init__(
        inout self,
        address: String,
        owned session: InferenceSession,
    ) raises:
        """Constructs a gRPC inference client.

        Args:
            address: Address to connect to.
            session: Current inference context.
        """
        self._lib = handle_from_config("serving", ".serve_lib")
        self._session = session^
        self._impl = CGRPCClient(self._lib, address.as_string_slice())
        self._impl.run()

    fn __moveinit__(inout self: Self, owned existing: Self):
        self._lib = existing._lib
        self._impl = existing._impl^
        self._session = existing._session^

    async fn _infer(
        inout self, request: InferenceRequest
    ) raises -> InferenceResponse:
        await ChainPromise(self._impl.model_infer(request._impl))
        var result = self._impl.take_infer_result(request._impl)
        if result[].code != 0:
            # The response should be null in this case.
            var s = str(result[].error)
            ClientResult.free(self._impl._lib, result)
            raise Error(s)
        else:
            # This hands ownership of the response to the underlying
            # InferenceRequest object. It will be freed separately when this
            # object is freed.
            var response = InferenceResponse(
                CInferenceResponse(
                    self._impl._lib, result[].response, owning=True
                ),
                self._session,
            )
            ClientResult.free(self._impl._lib, result)
            return response^

    fn _make_inference_request(
        inout self,
        name: String,
        version: String,
        inputs: TensorMap,
        outputs: List[String],
    ) raises -> InferenceRequest:
        var request = InferenceRequest(
            self._impl.create_infer_request(
                name.as_string_slice(), version.as_string_slice()
            ),
            self._session,
        )
        request.set_input_tensors(inputs.keys(), inputs)
        request.set_outputs(outputs)
        return request^

    fn _make_inference_response(
        inout self,
        name: String,
        version: String,
        inputs: TensorMap,
        outputs: List[String],
    ) raises -> InferenceRequest:
        var request = InferenceRequest(
            self._impl.create_infer_request(
                name.as_string_slice(), version.as_string_slice()
            ),
            self._session,
        )
        request.set_input_tensors(inputs.keys(), inputs)
        request.set_outputs(outputs)
        return request^

    fn infer(
        inout self,
        name: String,
        version: String,
        inputs: TensorMap,
        outputs: List[String],
    ) raises -> InferenceResponse:
        var request = self._make_inference_request(
            name, version, inputs, outputs
        )
        return run(self._infer(request))
