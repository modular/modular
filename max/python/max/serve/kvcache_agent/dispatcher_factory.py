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

from dataclasses import dataclass, field
from enum import Enum


class TransportType(Enum):
    DYNAMIC_ZMQ = "dynamic_zmq"


@dataclass
class DynamicZmqTransportConfig:
    """Configuration for DynamicZmqTransport."""

    bind_address: str = "tcp://127.0.0.1:5555"
    default_destination_address: str | None = None


@dataclass
class DispatcherConfig:
    """Main configuration for dispatcher creation."""

    # Transport type to use for creating the transport
    transport: TransportType = field(default=TransportType.DYNAMIC_ZMQ)

    # Transport configuration to use for creating the transport
    transport_config: DynamicZmqTransportConfig = field(
        default_factory=lambda: DynamicZmqTransportConfig()
    )
