# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from max.driver import CPU, Accelerator
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
    assert len({id(dev._device) for dev in devices}) == 1

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
    assert "timestamp" in stats
    assert "free_memory" in stats
    assert "total_memory" in stats
