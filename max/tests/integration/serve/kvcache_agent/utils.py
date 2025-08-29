# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from max.serve.kvcache_agent.dispatcher_factory import (
    DispatcherConfig,
    TransportFactory,
    TransportType,
)
from max.serve.queue.zmq_queue import generate_zmq_inproc_endpoint


def create_dispatcher_config() -> tuple[str, DispatcherConfig]:
    bind_address = generate_zmq_inproc_endpoint()
    return bind_address, DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=bind_address,
        ),
    )
