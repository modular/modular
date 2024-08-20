# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import getenv
from subprocess import run

import max.driver as md
import pytest


def test_cpu_device():
    cpu = md.CPU()
    assert "cpu" in str(cpu)


def _cuda_available() -> bool:
    output = run(getenv("MODULAR_IS_CUDA_AVAILABLE") or "is-cuda-available")
    return output.returncode == 0


@pytest.mark.skipif(not _cuda_available(), reason="Requires CUDA")
def test_cuda_device():
    cuda = md.CUDA()
    assert "cuda" in str(cuda)
