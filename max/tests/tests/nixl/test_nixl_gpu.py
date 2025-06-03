# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests basic NIXL functionality"""

import time
from typing import cast

import numpy as np
import pytest
import torch
from max._core.nixl import (
    Agent,
    AgentConfig,
    MemoryType,
    RegistrationDescriptorList,
    Status,
    TransferDescriptorList,
    TransferOpType,
)
from max.driver import CPU, Accelerator, Device, Tensor
from numpy.typing import ArrayLike


def _get_tensor_base_addr(tensor: Tensor) -> int:
    """Get the base address of a tensor."""
    return torch.from_dlpack(tensor).data_ptr()


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
    agent_1, _ = create_agent("agent_1", 8047)
    agent_2, _ = create_agent("agent_2", 8057)

    # Get Agent Metadata from Agents
    agent_1_md = agent_1.get_local_metadata()
    agent_2_md = agent_2.get_local_metadata()

    # Register Agent with Pair
    remote_bytes_1 = agent_2.load_remote_metadata(agent_1_md)
    assert remote_bytes_1.decode() == "agent_1"
    remote_bytes_2 = agent_1.load_remote_metadata(agent_2_md)
    assert remote_bytes_2.decode() == "agent_2"


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

    # Test append()
    buffer2 = Tensor.from_numpy(np.ones((50, 50))).to(device)
    registration_descriptor.append(cast(ArrayLike, buffer2))

    # Register Memory
    status = agent.register_memory(
        descs=registration_descriptor,
        backends=[ucx_backend],
    )
    assert status == Status.SUCCESS


@pytest.mark.parametrize("device", [CPU()])
def test_memory_transfer(device: Device):
    buffer_size_1 = 1024
    buffer_magic_val_1 = 99
    buffer_size_2 = 2048
    buffer_magic_val_2 = 42
    agent_1_name = "agent_1"
    agent_2_name = "agent_2"

    buffer_1 = Tensor.from_numpy(
        np.full((buffer_size_1,), buffer_magic_val_1, dtype=np.int8)
    ).to(device)
    buffer_2 = Tensor.from_numpy(
        np.full((buffer_size_2,), buffer_magic_val_2, dtype=np.int8)
    ).to(device)

    memory_type = MemoryType.DRAM if device.is_host else MemoryType.VRAM

    # Create Agents
    agent_1, ucx_backend_1 = create_agent(agent_1_name, listen_port=8037)
    agent_2, ucx_backend_2 = create_agent(agent_2_name, listen_port=8042)

    # Register Memory
    reg_dlist_1 = RegistrationDescriptorList(
        type=memory_type, descs=[cast(ArrayLike, buffer_1)], sorted=True
    )
    agent_1.register_memory(descs=reg_dlist_1, backends=[ucx_backend_1])

    reg_dlist_2 = RegistrationDescriptorList(
        type=memory_type, descs=[cast(ArrayLike, buffer_2)], sorted=True
    )
    agent_2.register_memory(descs=reg_dlist_2, backends=[ucx_backend_2])

    # Get Agent Metadata from Agents
    agent_1_md = agent_1.get_local_metadata()
    agent_2_md = agent_2.get_local_metadata()

    # Register Agent with Pair
    remote_bytes_2 = agent_2.load_remote_metadata(agent_1_md)
    assert remote_bytes_2.decode() == agent_1_name
    remote_bytes_1 = agent_1.load_remote_metadata(agent_2_md)
    assert remote_bytes_1.decode() == agent_2_name

    # Compute buffer regions to transfer
    # The offsets and transfer sizes are arbitrarily chosen for this test.
    dst_offset = buffer_size_1 // 4
    src_offset = 0
    bytes_to_copy = buffer_size_1 // 2
    src_base = _get_tensor_base_addr(buffer_1) + src_offset
    dst_base = _get_tensor_base_addr(buffer_2) + dst_offset

    # Create xfer request
    xfer_dlist_src = TransferDescriptorList(
        type=memory_type,
        descs=[(src_base, bytes_to_copy, memory_type.value)],
    )
    xfer_dlist_dst = TransferDescriptorList(
        type=memory_type,
        descs=[(dst_base, bytes_to_copy, memory_type.value)],
    )
    secret_message = "mojo is so fast 🔥"
    xfer_req = agent_1.create_transfer_request(
        operation=TransferOpType.WRITE,
        local_descs=xfer_dlist_src,
        remote_descs=xfer_dlist_dst,
        remote_agent=agent_2_name,
        notif_msg=secret_message,
    )

    status = agent_1.post_transfer_request(xfer_req)
    assert status in [Status.SUCCESS, Status.IN_PROG], (
        "Failed to post transfer request"
    )

    check_status = Status.IN_PROG
    notif_received = False
    start_time = time.time()
    timeout = 5.0
    notif_map: dict[str, list[bytes]] = {}

    while (check_status != Status.SUCCESS or not notif_received) and (
        time.time() - start_time
    ) < timeout:
        if check_status != Status.SUCCESS:
            check_status = agent_1.get_transfer_status(xfer_req)
            assert check_status in [Status.SUCCESS, Status.IN_PROG], (
                "Received unexpected transfer status"
            )

        if not notif_received:
            notif_map = agent_2.get_notifs()
            if agent_1_name in notif_map and len(notif_map[agent_1_name]) > 0:
                notif_received = True

        # Yield/Sleep briefly to avoid busy-waiting
        time.sleep(0.001)

    assert check_status == Status.SUCCESS, (
        f"Transfer did not complete in {timeout} seconds"
    )
    assert agent_1_name in notif_map, "Notification map missing agent 1"
    assert len(notif_map[agent_1_name]) == 1, (
        "Incorrect number of notifications received"
    )
    received_message = notif_map[agent_1_name][0].decode()
    assert received_message == secret_message, "Notification message mismatch"

    def s2ms(s: float) -> float:
        return s * 1000

    # Should take only a couple ms
    print(f"Transfer completed in {s2ms(time.time() - start_time):.2f} ms")
    print(f"Agent 2 received notification from Agent 1: '{received_message}'")

    # Verify transferred data

    # Buffer 1 should be unchanged
    buffer_1_expected = np.full(
        (buffer_size_1,), buffer_magic_val_1, dtype=np.int8
    )
    assert np.array_equal(buffer_1.to_numpy(), buffer_1_expected), (
        "Transferred data does not match expected data"
    )

    # Buffer 2 should be mostly 42's except for some region of 99's
    buffer_2_expected = np.full(
        (buffer_size_2,), buffer_magic_val_2, dtype=np.int8
    )
    buffer_2_expected[dst_offset : dst_offset + bytes_to_copy] = (
        buffer_magic_val_1
    )
    assert np.array_equal(buffer_2.to_numpy(), buffer_2_expected), (
        "Transferred data does not match expected data"
    )

    # Cleanup
    status = agent_1.release_transfer_request(xfer_req)
    assert status == Status.SUCCESS, "Failed to release transfer request"

    status = agent_2.deregister_memory(reg_dlist_2)
    assert status == Status.SUCCESS, "Failed to deregister memory"

    status = agent_1.invalidate_remote_metadata(agent_2_name)
    assert status == Status.SUCCESS, "Failed to invalidate remote metadata"

    status = agent_2.invalidate_remote_metadata(agent_1_name)
    assert status == Status.SUCCESS, "Failed to invalidate remote metadata"

    status = agent_1.deregister_memory(reg_dlist_1)
    assert status == Status.SUCCESS, "Failed to deregister memory"
