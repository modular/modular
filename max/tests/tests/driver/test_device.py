# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.driver import CPU, CUDA


def test_cpu_device():
    # We should be able to create a CPU device.
    cpu = CPU()
    assert "cpu" in str(cpu)
    assert cpu.is_host


@pytest.mark.skip(reason="MSDK-834")
def test_cuda_device_creation_error():
    # Creating a CUDA device on a machine without a GPU should raise an error.
    with pytest.raises(ValueError, match="failed to create device:"):
        _ = CUDA()
