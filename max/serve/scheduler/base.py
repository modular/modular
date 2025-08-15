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
from abc import ABC, abstractmethod
from typing import Union

import msgspec
from max.interfaces import RequestID
from max.nn.kv_cache import XferReqData
from max.pipelines.core import TextAndVisionContext, TextContext


class Scheduler(ABC):
    """Abstract base class defining the interface for schedulers."""

    @abstractmethod
    def run_iteration(self):
        """The core scheduler routine that creates and executes batches.

        This method should implement the core scheduling logic including:
        - Batch creation and management
        - Request scheduling
        """
        pass


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
    context: Union[TextContext, TextAndVisionContext]
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
    """

    id: RequestID
    context: Union[TextContext, TextAndVisionContext]
    transfer_metadata: XferReqData
