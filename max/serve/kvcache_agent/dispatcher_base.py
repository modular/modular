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

from enum import Enum
from typing import Any

import msgspec


class MessageType(Enum):
    """Enumeration of supported message types for local communication."""

    PREFILL_REQUEST = "prefill_request"
    PREFILL_RESPONSE = "prefill_response"


class ReplyContext(
    msgspec.Struct,
    tag=True,
    omit_defaults=True,
    frozen=True,
):
    """
    Transport-agnostic reply context containing routing information for message replies.

    Encapsulates all necessary information to send a reply back to the original sender,
    including transport-specific metadata for proper routing.
    """

    reply_address: str
    correlation_id: str
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)
