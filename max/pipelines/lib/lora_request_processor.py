# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""LoRA ZMQ request handler for processing LoRA operations."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Generic, TypeVar

import msgspec
from max.interfaces.lora import (
    LoRAOperation,
    LoRARequest,
    LoRAResponse,
    LoRAStatus,
)
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket

if TYPE_CHECKING:
    from .lora import LoRAManager

logger = logging.getLogger("max.serve")

ReqId = TypeVar("ReqId")


class LoRARequestProcessor(Generic[ReqId]):
    """
    Processes LoRA requests by delegating operations to a LoRAManager.

    This class acts as a bridge between the LoRA queue system and the LoRA manager,
    processing load/unload/list operations and returning appropriate responses.
    """

    def __init__(
        self,
        manager: LoRAManager,
        zmq_request_endpoint: str,
        zmq_response_endpoint: str,
    ):
        """
        Initialize the LoRA request processor.

        Args:
            manager: The LoRAManager instance to handle operations.
        """
        self.manager = manager

        self._request_socket = ZmqPullSocket[tuple[ReqId, LoRARequest]](
            endpoint=zmq_request_endpoint,
            deserialize=msgspec.msgpack.Decoder(
                type=tuple[ReqId, LoRARequest]
            ).decode,
        )

        self._response_socket = ZmqPushSocket[tuple[ReqId, LoRAResponse]](
            endpoint=zmq_response_endpoint,
            serialize=msgspec.msgpack.Encoder().encode,
        )

        self._zmq_thread = threading.Thread(
            target=self._process_zmq_requests,
            daemon=True,
            name="LoRARequestProcessor",
        )
        self._zmq_running = True
        self._zmq_thread.start()

    def _process_zmq_requests(self):
        """Process ZMQ requests in background thread."""
        logger.info("LoRA request handler thread started.")
        while self._zmq_running:
            try:
                try:
                    req_id, request = self._request_socket.get_nowait()

                    response = self._handle_zmq_request(request)

                    self._response_socket.put_nowait((req_id, response))

                except queue.Empty:
                    time.sleep(0.01)

            except Exception as e:
                logger.exception(f"Error processing LoRA request: {e}")

        logger.info("LoRA ZMQ handler thread stopped")

    def _handle_zmq_request(self, request: LoRARequest) -> LoRAResponse:
        """
        Handle a single LoRA request with thread-safe access.

        Args:
            request: The LoRA request to process.

        Returns:
            LoRAResponse with the result of the operation.
        """
        try:
            if request.operation == LoRAOperation.LOAD:
                return self._handle_load_request(request)
            elif request.operation == LoRAOperation.UNLOAD:
                return self._handle_unload_request(request)
            else:
                return self._handle_list_request()
        except Exception as e:
            logger.exception(
                f"Unexpected error handling LoRA request {request}: {e}"
            )
            error_detail = str(e) if str(e) else "Unknown error"

            if request.operation == LoRAOperation.LOAD:
                return LoRAResponse(
                    LoRAStatus.LOAD_ERROR,
                    f"Unexpected error loading LoRA adapter: {error_detail}",
                )
            elif request.operation == LoRAOperation.UNLOAD:
                return LoRAResponse(
                    LoRAStatus.UNLOAD_ERROR,
                    f"Unexpected error unloading LoRA adapter: {error_detail}",
                )
            else:
                return LoRAResponse(
                    LoRAStatus.LOAD_ERROR,  # Use LOAD_ERROR as generic error
                    f"Unexpected error listing LoRA adapters: {error_detail}",
                )

    def _handle_load_request(self, request: LoRARequest) -> LoRAResponse:
        """Handle LoRA load request."""
        lora_name = request.lora_name
        lora_path = request.lora_path

        status = self.manager.load_adapter(f"{lora_name}={lora_path}")

        if status == LoRAStatus.SUCCESS:
            message = f"LoRA adapter '{lora_name}' loaded successfully"
        elif status == LoRAStatus.LOAD_NAME_EXISTS:
            message = f"LoRA adapter name '{lora_name}' already exists with different path"
        elif status == LoRAStatus.LOAD_INVALID_PATH:
            message = f"Invalid LoRA adapter path: '{lora_path}'. Path must exist locally (remote repositories are not supported)"
        elif status == LoRAStatus.LOAD_INVALID_ADAPTER:
            message = f"Invalid LoRA adapter at '{lora_path}'. Ensure the adapter has the correct format and required files"
        elif status == LoRAStatus.LOAD_ERROR:
            message = f"Unexpected error loading LoRA adapter '{lora_name}'"
        else:
            message = f"Failed to load LoRA adapter '{lora_name}' with status: {status.value}"

        return LoRAResponse(status, message)

    def _handle_unload_request(self, request: LoRARequest) -> LoRAResponse:
        """Handle LoRA unload request."""
        status = self.manager.unload_adapter(request.lora_name)  # type: ignore

        if status == LoRAStatus.SUCCESS:
            message = (
                f"LoRA adapter '{request.lora_name}' unloaded successfully"
            )
        elif status == LoRAStatus.UNLOAD_NAME_NONEXISTENT:
            message = f"LoRA adapter '{request.lora_name}' not found"
        elif status == LoRAStatus.UNLOAD_ERROR:
            message = f"Error unloading LoRA adapter '{request.lora_name}'"
        else:
            message = f"Failed to unload LoRA adapter '{request.lora_name}' with status: {status.value}"

        return LoRAResponse(status, message)

    def _handle_list_request(self) -> LoRAResponse:
        """Handle LoRA list request."""
        loras = self.manager.loras
        return LoRAResponse(LoRAStatus.SUCCESS, loras)

    def __del__(self):
        """Cleanup when the object is destroyed."""
        if self._zmq_running:
            self._zmq_running = False
            if self._zmq_thread.is_alive():
                self._zmq_thread.join(timeout=5)

            self._request_socket.close()
            self._response_socket.close()
