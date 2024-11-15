# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from testing import *
from sys.info import _accelerator_arch


def main():
    var accelerator_arch = _accelerator_arch()

    assert_true(
        accelerator_arch == "nvidia:sm80"
        or accelerator_arch == "nvidia:sm84"
        or accelerator_arch == "nvidia:sm86"
        or accelerator_arch == "nvidia:sm89"
        or accelerator_arch == "nvidia:sm90"
    )
