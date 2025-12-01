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

"""LoRA queue implementation for managing LoRA adapter loading/unloading."""

from __future__ import annotations

import asyncio
import logging
import queue

from max.interfaces import (
    LoRAOperation,
    LoRARequest,
    LoRAResponse,
    LoRAStatus,
    RequestID,
)
from max.interfaces.lora import LORA_REQUEST_ENDPOINT, LORA_RESPONSE_ENDPOINT
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket

logger = logging.getLogger("max.serve")


class LoRAQueue:
    """Queue for managing LoRA adapter load/unload/list requests."""

    def __init__(self, zmq_endpoint_base: str, lora_paths: list[str] | None):
        self._request_socket = ZmqPushSocket[tuple[RequestID, LoRARequest]](
            endpoint=f"{zmq_endpoint_base}-{LORA_REQUEST_ENDPOINT}",
            payload_type=tuple[RequestID, LoRARequest],
        )
        self._response_socket = ZmqPullSocket[tuple[RequestID, LoRAResponse]](
            endpoint=f"{zmq_endpoint_base}-{LORA_RESPONSE_ENDPOINT}",
            payload_type=tuple[RequestID, LoRAResponse],
        )

        self._loaded_loras: list[str] = []

        if lora_paths:
            self._loaded_loras = [
                lora.split("=")[0] if lora.find("=") != -1 else lora
                for lora in lora_paths
            ]

    def list_loras(self) -> list[str]:
        return self._loaded_loras

    async def get_response(
        self, req_id: RequestID, request: LoRARequest, timeout: float = 30.0
    ) -> LoRAResponse:
        """
        Send a LoRA request and poll for the response.

        Since LoRA operations are infrequent, we poll directly instead of
        using a continuous background worker to avoid performance overhead.
        """
        # Send the request
        try:
            self._request_socket.put_nowait((req_id, request))
        except Exception as e:
            return LoRAResponse(
                status=LoRAStatus.UNSPECIFIED_ERROR,
                message=f"Failed to send LoRA request: {e}",
            )

        start_time = asyncio.get_event_loop().time()
        while True:
            try:
                response_id, response = self._response_socket.get_nowait()
                if response_id == req_id:
                    # Update loaded loras list on success
                    if response.status == LoRAStatus.SUCCESS:
                        if request.operation == LoRAOperation.LOAD:
                            self._loaded_loras.append(request.lora_name)
                        elif request.operation == LoRAOperation.UNLOAD:
                            self._loaded_loras.remove(request.lora_name)
                    return response
                else:
                    # Not our response. This shouldn't happen but log it
                    logger.warning(
                        "Received response for unexpected request_id: %s (expected %s)",
                        response_id,
                        req_id,
                    )
            except queue.Empty:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    return LoRAResponse(
                        status=LoRAStatus.UNSPECIFIED_ERROR,
                        message=f"Timeout waiting for LoRA response after {timeout}s",
                    )
                # Sleep briefly before polling again (this is fine since LoRA ops are infrequent)
                await asyncio.sleep(0.005)
