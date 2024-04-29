# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Includes common traits and data types for inference services."""

from max.engine import TensorMap


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
    ](self, request: req_type, inout response: resp_type) -> None:
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


@value
struct FileModel:
    """File-backed model artifact details."""

    var name: String
    var path: String
    var version: Int
