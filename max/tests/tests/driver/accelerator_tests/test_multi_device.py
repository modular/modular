# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType


def test_accelerator_peer_access() -> None:
    """Test peer access between multiple accelerators."""
    gpu0 = Accelerator(id=0)
    gpu1 = Accelerator(id=1)

    can_access_0_to_1 = gpu0.can_access(gpu1)
    can_access_1_to_0 = gpu1.can_access(gpu0)

    # Typically, peer access is symmetric, but hardware-dependent.
    assert can_access_0_to_1 == can_access_1_to_0, (
        "Peer access should be symmetric."
    )


def test_to_multiple_devices() -> None:
    cpu = CPU()
    acc0 = Accelerator(id=0)
    acc1 = Accelerator(id=1)

    tensor = Tensor(dtype=DType.int32, shape=(3, 3), device=cpu)
    # GEX-2624: CPU to CPU copies not supported via Tensor.to()
    tensors = tensor.to([acc0, acc1])
    assert len(tensors) == 2
    assert tensors[0].device == acc0
    assert tensors[1].device == acc1


def test_from_device() -> None:
    cpu = CPU()
    acc0 = Accelerator(id=0)
    acc1 = Accelerator(id=1)

    tensor = Tensor(dtype=DType.int32, shape=(3, 3), device=acc0)
    tensors = tensor.to([cpu, acc0, acc1])
    assert len(tensors) == 3
    assert tensors[0].device == cpu
    assert tensors[1].device == acc0
    assert tensors[2].device == acc1
