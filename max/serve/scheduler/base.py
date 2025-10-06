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
import asyncio
from enum import Enum

import msgspec
from max.interfaces import RequestID
from max.nn.kv_cache import XferReqData
from max.pipelines.core import TextAndVisionContext, TextContext


class SchedulerProgress(Enum):
    """Indicates whether a scheduler made progress during an iteration."""

    MADE_PROGRESS = "made_progress"
    NO_PROGRESS = "no_progress"


async def sleep_with_backoff(count_no_progress: int) -> None:
    """A basic strategy to avoid busy waiting.

    This function sleeps with a linear backoff.
    The first sleep of 0 enables other async threads to run but otherwise does not sleep.
    The step size is 1ms because of limitations around asyncio to sleep with finer granularity.
    The maximum sleep is 10ms because it resolves CPU usage overhead while maintaining minimal waiting.
    """

    ms_to_sleep = min(max(0, count_no_progress), 10)
    await asyncio.sleep(ms_to_sleep * 0.001)


class PrefillRequest(
    msgspec.Struct, tag=True, omit_defaults=True, kw_only=True
):
    """A request for prefill (context encoding) processing.

    Contains the request ID, input context, and transfer engine details needed to
    process a prefill request through the pipeline and transfer KV cache data.

    Attributes:
        id: Unique identifier for this request
        context: The input context containing the request data and state
        transfer_engine_name: Name of the transfer engine to use for KV cache transfers
        block_ids: List of block IDs allocated for KV cache storage
    """

    id: RequestID
    context: TextContext | TextAndVisionContext
    transfer_engine_name: str
    block_ids: list[int]


class PrefillResponse(
    msgspec.Struct, tag=True, omit_defaults=True, kw_only=True
):
    """A response for prefill (context encoding) processing.

    Contains the request ID and input context needed to run decode
    and generate tokens based on the prefill finished.

    Attributes:
        id: Unique identifier for this request
        context: The input context containing the request data and state
        transfer_metadata: The transfer metadata for the KV cache transfers
    """

    id: RequestID
    generated_token_id: int
    transfer_metadata: XferReqData


class CancelRequest(msgspec.Struct, tag=True, omit_defaults=True, kw_only=True):
    """A request to cancel an ongoing request.

    Used to signal that a specific request should be cancelled and its resources
    should be freed. This is typically used to cancel prefill or decode requests
    that are no longer needed.

    Attributes:
        id: Unique identifier of the request to cancel
    """

    id: RequestID
