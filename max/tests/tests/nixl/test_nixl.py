# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests basic NIXL functionality"""

from max._core.nixl import Agent, AgentConfig


def create_agent(name: str, listen_port: int) -> Agent:
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
    agent.create_backend(type="ucx", init_params=ucx_params[0])

    return agent


def test_agent_registration():
    _ = create_agent("test_agent", 8047)


def test_remote_agent_registration():
    agent_1 = create_agent("test_agent_1", 8047)
    agent_2 = create_agent("test_agent_2", 8057)

    # Get Agent Metadata from Agents
    agent_1_md = agent_1.get_local_metadata()
    agent_2_md = agent_2.get_local_metadata()

    # Register Agent with Pair
    agent_2.load_remote_metadata(agent_1_md)
    agent_1.load_remote_metadata(agent_2_md)
