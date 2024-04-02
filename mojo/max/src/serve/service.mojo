# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Includes traits and common data types for inference services."""

from max.engine import TensorMap


@value
struct ModelInfo:
    """Model artifact details."""

    var name: String
    var path: String
    var version: Int


trait InferenceRequest(Movable):
    """A trait for singular inference requests."""

    fn get_input_tensors(self) raises -> TensorMap:
        """Returns all input tensors.

        Returns:
            A tensor map containing all input tensors.
        """
        ...

    fn get_requested_outputs(self) -> List[String]:
        """Returns all requested output names.

        Returns:
            A list containing all requested output names.
        """
        ...

    fn set_input_tensors(self, names: List[String], map: TensorMap) raises:
        """Sets input tensors to those with matching names from map.

        Args:
            names: Names of selected input tensors.
            map: Map of all input tensors.
        """
        ...


trait InferenceResponse(Movable):
    """A trait for singular inference responses."""

    fn get_output_tensors(self) raises -> TensorMap:
        """Returns all output tensors.

        Returns:
            A tensor map containing all output tensors.
        """
        ...

    fn set_output_tensors(self, names: List[String], map: TensorMap) raises:
        """Sets output tensors to those with matching names from map.

        Args:
            names: Names of selected output tensors.
            map: Map of all output tensors.
        """
        ...


trait InferenceService:
    """A trait for services that serve model inference requests."""

    fn infer[
        req_type: InferenceRequest, resp_type: InferenceResponse
    ](self, request: req_type, inout response: resp_type) raises -> None:
        """Runs a single inference request.

        Parameters:
            req_type: Type of request.
            resp_type: Type of response.

        Args:
            request: Request object.
            response: Response object.
        """
        ...

    async fn async_infer[
        req_type: InferenceRequest, resp_type: InferenceResponse
    ](self, request: req_type, inout response: resp_type) -> None:
        """Asynchronously runs a single inference request.

        Parameters:
            req_type: Type of request.
            resp_type: Type of response.

        Args:
            request: Request object.
            response: Response object.
        """
        ...
