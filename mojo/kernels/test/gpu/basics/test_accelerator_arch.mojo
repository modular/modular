# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from sys.info import _accelerator_arch

from testing import *


def main():
    var accelerator_arch = _accelerator_arch()

    assert_true(
        accelerator_arch == "nvidia:80"
        or accelerator_arch == "nvidia:84"
        or accelerator_arch == "nvidia:86"
        or accelerator_arch == "nvidia:89"
        or accelerator_arch == "nvidia:90"
        or accelerator_arch == "amdgpu:94"
    )
