# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import getenv
from subprocess import run

import max.driver as md
import pytest

from modular.utils.misc import has_gpu


def test_cpu_device():
    cpu = md.CPU()
    assert "cpu" in str(cpu)


def _cuda_available() -> bool:
    return has_gpu()


@pytest.mark.skipif(not _cuda_available(), reason="Requires CUDA")
def test_cuda_device():
    cuda = md.CUDA()
    assert "cuda" in str(cuda)


@pytest.mark.skip(reason="MSDK-834")
@pytest.mark.skipif(_cuda_available(), reason="Should not have CUDA")
def test_cuda_device_creation_error():
    with pytest.raises(ValueError, match="failed to create device:"):
        _ = md.CUDA()
