# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test NIXL Agent Registration"""

import pytest
from max._core.nixl import Agent, AgentConfig


def test_agent_registration():
    # Stand up Agent
    agent = Agent(
        "test_agent",
        AgentConfig(
            use_prog_thread=False, use_listen_thread=True, listen_port=8047
        ),
    )

    # ucx should be available as a plugin.
    assert "ucx" in agent.get_available_plugins()

    # Create ucx backend
    ucx_params = agent.get_plugin_params("ucx")
    agent.create_backend(type="ucx", init_params=ucx_params[0])


@pytest.mark.skip("TODO: E2EOPT-241")
def test_remote_agent_registration():
    # Stand up Agent 1
    agent_1 = Agent(
        "test_agent",
        AgentConfig(
            use_prog_thread=False, use_listen_thread=True, listen_port=8047
        ),
    )

    # ucx should be available as a plugin.
    assert "ucx" in agent_1.get_available_plugins()

    # Create ucx backend
    ucx_params = agent_1.get_plugin_params("ucx")
    agent_1.create_backend(type="ucx", init_params=ucx_params[0])

    # Stand up Agent 2
    agent_2 = Agent(
        "test_agent_2",
        AgentConfig(
            use_prog_thread=False, use_listen_thread=True, listen_port=8057
        ),
    )

    # ucx should be available as a plugin.
    assert "ucx" in agent_2.get_available_plugins()

    # Create ucx backend
    ucx_params = agent_2.get_plugin_params("ucx")
    agent_2.create_backend(type="ucx", init_params=ucx_params[0])

    # Get Agent Metadata from Agents
    agent_1_md = agent_1.get_local_metadata()
    agent_2_md = agent_2.get_local_metadata()

    # Register Agent with Pair
    agent_2.load_remote_metadata(agent_1_md)
    agent_1.load_remote_metadata(agent_2_md)
