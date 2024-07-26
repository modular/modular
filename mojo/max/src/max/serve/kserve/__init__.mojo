# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""KServe client and server implementation."""

from .types import InferenceRequest, InferenceResponse, InferenceBatch
from .server import GRPCServer
from .client import GRPCClient
from .mux import FileModel, MuxInferenceService
