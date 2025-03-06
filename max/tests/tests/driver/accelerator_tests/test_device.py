# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.graph import DeviceKind, DeviceRef


def test_accelerator_device():
    # We should be able to create a Accelerator device.
    dev = Accelerator()
    assert "gpu" in str(dev)
    assert not dev.is_host


def test_accelerator_is_compatible():
    accelerator = Accelerator()
    assert accelerator.is_compatible


def test_accelerator_device_label_id():
    # Test the label property and attempt to map to graph.DeviceRef.
    dev_id = 0
    default_device = Accelerator()
    device = Accelerator(id=dev_id)
    assert "gpu" in device.label
    assert dev_id == device.id
    assert dev_id == default_device.id
    dev_from_runtime = DeviceRef(device.label, device.id)
    dev1_from_runtime = DeviceRef(DeviceKind(device.label), device.id)
    assert dev_from_runtime == DeviceRef.GPU(dev_id)
    assert dev1_from_runtime == DeviceRef.GPU(dev_id)


def scoped_device():
    _ = Accelerator(0)  # NOTE: device ID is intentionally explicit.


def test_stress_accelerator_device():
    # We should be able to call Accelerator() many times, and get cached outputs.
    devices = [Accelerator() for _ in range(64)]
    assert len({id(dev) for dev in devices}) == 1

    # TODO(MSDK-1220): move this before the above assert when the context no
    # longer leaks. Until then, this should still test that the default device
    # ID and explicit 0 ID share a device cache entry.
    for _ in range(64):
        scoped_device()


def test_equality():
    # We should be able to test the equality of devices.
    cpu = CPU()
    accel = Accelerator()

    assert cpu != accel


def test_stats():
    # We should be able to query utilization stats for the device.
    accel = Accelerator()
    stats = accel.stats
    assert "free_memory" in stats
    assert "total_memory" in stats


def test_accelerator_can_access_self():
    """Accelerator should not be able to access itself."""
    accel = Accelerator()
    assert not accel.can_access(accel), "Device should not access itself."


def test_accelerator_can_access_cpu():
    """Accelerator should typically not have direct peer access to CPU."""
    gpu = Accelerator()
    cpu = CPU()
    assert not gpu.can_access(cpu), "GPU should not directly access CPU memory."


def test_cpu_can_access_accelerator():
    """CPUs normally cannot directly access accelerator memory."""
    gpu = Accelerator()
    cpu = CPU()
    assert not cpu.can_access(gpu), (
        "CPUs shouldn't be able to access accelerator memory."
    )


def test_accelerator_peer_access():
    """Test peer access between multiple accelerators."""
    num_accelerators = accelerator_count()
    if num_accelerators < 2:
        pytest.skip("Test requires at least two accelerators.")

    gpu0 = Accelerator(id=0)
    gpu1 = Accelerator(id=1)

    can_access_0_to_1 = gpu0.can_access(gpu1)
    can_access_1_to_0 = gpu1.can_access(gpu0)

    # Typically, peer access is symmetric, but hardware-dependent.
    assert can_access_0_to_1 == can_access_1_to_0, (
        "Peer access should be symmetric."
    )


def test_cpu_can_access_cpu():
    """CPU should not report peer access to itself."""
    cpu = CPU()
    another_cpu = CPU()
    assert not cpu.can_access(another_cpu), "CPU should not access another CPU."
