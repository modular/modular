# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides basic KServe service/server implementations."""

from sys.ffi import DLHandle
from tensor import TensorSpec
from runtime.llcl import Runtime
from time import now
from utils.variant import Variant

from max.engine import InferenceSession, InputSpec, TensorMap
from max.engine._utils import handle_from_config

from ._client_impl import KServeClientAsync
from .._serve_rt import TensorView, InferenceRequestImpl, InferenceResponseImpl


struct GRPCInferenceClient:
    """Inference client implementing the KServe protocol over gRPC."""

    var _lib: DLHandle
    var _session: InferenceSession
    var _impl: KServeClientAsync

    @staticmethod
    fn create(
        address: String, owned session: InferenceSession
    ) raises -> GRPCInferenceClient:
        return GRPCInferenceClient(address, session^)

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
        self._impl = KServeClientAsync(self._lib, address, self._session)
        self._impl.run()

    fn infer(
        inout self,
        name: String,
        version: String,
        inputs: TensorMap,
        outputs: List[String],
    ) raises -> InferenceResponseImpl:
        var rt = Runtime()
        var result = Variant[InferenceResponseImpl, Error](Error())
        rt.run(self.async_infer(name, version, inputs, outputs, result))
        if result.isa[Error]():
            raise result.unsafe_take[Error]()
        else:
            return result.unsafe_take[InferenceResponseImpl]()

    # TODO: Add TensorMap variant that owns tensors.
    async fn async_infer(
        inout self,
        name: String,
        version: String,
        inputs: TensorMap,
        outputs: List[String],
        inout result: Variant[InferenceResponseImpl, Error],
    ) -> None:
        var request = self._impl.create_infer_request(name, version)
        try:
            request.set_input_tensors(inputs.keys(), inputs)
            request.set_outputs(outputs)
            var response = Variant[InferenceResponseImpl, Error](Error())
            await self._impl.model_infer(request, response)
            result = response^
        except e:
            result.set[Error](e)
