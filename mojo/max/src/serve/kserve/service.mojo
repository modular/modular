# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Service definition for KServe-based APIs."""

from utils.variant import Variant
from max.engine import Model
from .types import InferenceRequest, InferenceResponse


trait InferenceService:
    """A trait for services that serve model inference requests."""

    fn init[
        add_models: fn (List[Model]) capturing raises -> None
    ](inout self) raises -> None:
        """Called prior to serving.

        Parameters:
            add_models: Callback to add models to service metadata.
        """
        ...

    fn infer(
        inout self, request: InferenceRequest, inout response: InferenceResponse
    ) raises -> None:
        """Ssynchronously runs a single inference request.

        Args:
            request: Request object.
            response: Response object.
        """
        ...
