# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import max.driver as md


def test_cpu_device():
    cpu = md.CPU()
    assert "CPU" in str(cpu)
