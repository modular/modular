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

"""
The overall architecture is as follows:

Client Side:                    Server Side:
┌─────────────────┐            ┌─────────────────┐
│ DispatcherClient│            │ DispatcherClient│
│     (sender)    │            │   (receiver)    │
└─────────┬───────┘            └─────────┬───────┘
          │                              │
┌─────────▼───────┐            ┌─────────▼───────┐
│ DispatcherService│            │DispatcherService│
│   (transport)   │◄──────────►│   (transport)   │
└─────────────────┘            └─────────────────┘
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import zmq
from max.serve.kvcache_agent.dispatcher_client import DispatcherClient
from max.serve.kvcache_agent.dispatcher_service import DispatcherService
from max.serve.kvcache_agent.dispatcher_transport import (
    DispatcherTransport,
    DynamicZmqTransport,
)
from max.serve.queue.zmq_queue import generate_zmq_ipc_path

logger = logging.getLogger(__name__)


class TransportType(Enum):
    DYNAMIC_ZMQ = "dynamic_zmq"


class TransportFactory:
    """
    Factory class for creating transport instances.

    Provides convenient factory methods for creating different types of transport
    implementations with proper configuration and validation.
    """

    @dataclass
    class DynamicZmqTransportConfig:
        """Configuration for DynamicZmqTransport."""

        bind_address: str = "tcp://127.0.0.1:5555"
        instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
        default_destination_address: Optional[str] = None

    @staticmethod
    def create_dynamic_zmq_transport(
        zmq_ctx: zmq.Context,
        config: DynamicZmqTransportConfig,
    ) -> DynamicZmqTransport:
        """
        Create a dynamic ZMQ transport for N:M communication patterns.
        """
        logger.debug(
            f"Creating DynamicZmqTransport: bind_address={config.bind_address}, instance_id={config.instance_id}"
        )
        return DynamicZmqTransport(
            zmq_ctx=zmq_ctx,
            bind_address=config.bind_address,
            instance_id=config.instance_id,
            default_destination_address=config.default_destination_address,
        )

    @classmethod
    def create_transport_from_config(
        cls,
        transport_type: TransportType,
        config: DynamicZmqTransportConfig,
        zmq_ctx: zmq.Context,
    ) -> DispatcherTransport:
        """
        Create transport instance from configuration object.
        """
        if transport_type == TransportType.DYNAMIC_ZMQ:
            assert isinstance(config, cls.DynamicZmqTransportConfig)
            return cls.create_dynamic_zmq_transport(
                zmq_ctx=zmq_ctx,
                config=config,
            )


@dataclass
class DispatcherConfig:
    """Main configuration for dispatcher creation."""

    # Transport type to use for creating the transport
    transport: TransportType = field(default=TransportType.DYNAMIC_ZMQ)

    # Transport configuration to use for creating the transport
    transport_config: TransportFactory.DynamicZmqTransportConfig = field(
        default_factory=lambda: TransportFactory.DynamicZmqTransportConfig()
    )


class DispatcherFactory:
    """
    Simple factory for creating dispatcher servers and clients.

    Can be passed between processes and used to create servers/clients
    with the appropriate ZMQ context for each process.
    """

    def __init__(
        self,
        config: DispatcherConfig,
    ):
        """
        Initialize factory with a transport instance.
        """
        self._config = config
        self._service_to_client = generate_zmq_ipc_path()
        self._client_to_service = generate_zmq_ipc_path()

    def create_service(self, zmq_ctx: zmq.Context) -> DispatcherService:
        """
        Create a dispatcher service using the provided ZMQ context.
        """
        transport = TransportFactory.create_transport_from_config(
            transport_type=self._config.transport,
            config=self._config.transport_config,
            zmq_ctx=zmq_ctx,
        )
        return DispatcherService(
            zmq_ctx=zmq_ctx,
            send_endpoint=self._service_to_client,
            recv_endpoint=self._client_to_service,
            transport=transport,
        )

    def create_client(self, zmq_ctx: zmq.Context) -> DispatcherClient:
        """
        Create a dispatcher client using the provided ZMQ context.
        """
        return DispatcherClient(
            zmq_ctx=zmq_ctx,
            send_endpoint=self._client_to_service,
            recv_endpoint=self._service_to_client,
        )
