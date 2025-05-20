# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests basic NIXL functionality"""

from typing import cast

import numpy as np
import pytest
from max._core.nixl import (
    Agent,
    AgentConfig,
    MemoryType,
    RegistrationDescriptorList,
    Status,
)
from max.driver import CPU, Accelerator, Device, Tensor
from numpy.typing import ArrayLike


def create_agent(
    name: str = "test_agent", listen_port: int = 8047
) -> tuple[Agent, int]:
    # Stand up Agent
    agent = Agent(
        name,
        AgentConfig(
            use_prog_thread=False,
            use_listen_thread=True,
            listen_port=listen_port,
        ),
    )

    # ucx should be available as a plugin.
    assert "ucx" in agent.get_available_plugins()

    # Create ucx backend
    ucx_params = agent.get_plugin_params("ucx")
    ucx_backend = agent.create_backend(type="ucx", init_params=ucx_params[0])

    return agent, ucx_backend


def test_agent_registration():
    _ = create_agent()


def test_remote_agent_registration():
    agent_1, _ = create_agent("test_agent_1", 8047)
    agent_2, _ = create_agent("test_agent_2", 8057)

    # Get Agent Metadata from Agents
    agent_1_md = agent_1.get_local_metadata()
    agent_2_md = agent_2.get_local_metadata()

    # Register Agent with Pair
    agent_2.load_remote_metadata(agent_1_md)
    agent_1.load_remote_metadata(agent_2_md)


@pytest.mark.parametrize("device", [CPU(), Accelerator()])
def test_memory_registration(device: Device):
    agent, ucx_backend = create_agent()

    buffer = Tensor.from_numpy(np.ones((100, 100))).to(device)

    # if the descriptors are sorted for descs and device ID.
    # sort criteria has the comparison order of dev_id, then addr, then len.
    sorted = True

    memory_type = MemoryType.DRAM if device.is_host else MemoryType.VRAM

    registration_descriptor = RegistrationDescriptorList(
        type=memory_type,
        # This cast should not be necessary.
        # descs should accept any object that implements __dlpack__ instead.
        descs=[cast(ArrayLike, buffer)],
        sorted=sorted,
    )

    # Register Memory
    status = agent.register_memory(
        descs=registration_descriptor,
        backends=[ucx_backend],
    )
    assert status == Status.SUCCESS
