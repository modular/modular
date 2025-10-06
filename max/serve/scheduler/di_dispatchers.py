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

from __future__ import annotations

import logging

from max.nn.kv_cache import KVTransferEngineMetadata
from max.serve.kvcache_agent.dispatcher_v2 import (
    DispatcherClientV2,
    DispatcherServerV2,
)
from max.serve.scheduler.base import (
    CancelRequest,
    PrefillRequest,
    PrefillResponse,
)

logger = logging.getLogger("max.serve")

RequestType = PrefillRequest | KVTransferEngineMetadata | CancelRequest
ReplyType = PrefillResponse | KVTransferEngineMetadata


class PrefillDispatcherServerV2(DispatcherServerV2[RequestType, ReplyType]):
    def __init__(self, bind_addr: str):
        logger.info(f"Starting Prefill Dispatcher Server on {bind_addr}")
        super().__init__(
            endpoint=bind_addr,
            request_type=RequestType,
            reply_type=ReplyType,
        )


class DecodeDispatcherClientV2(DispatcherClientV2[RequestType, ReplyType]):
    def __init__(self, bind_addr: str, default_dest_addr: str | None):
        logger.info(
            f"Starting Decode Dispatcher Client on {bind_addr} with default destination address {default_dest_addr}"
        )
        super().__init__(
            bind_addr=bind_addr,
            default_dest_addr=default_dest_addr,
            request_type=RequestType,
            reply_type=ReplyType,
        )
