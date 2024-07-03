# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.driver.driver_core import __version__
from max.driver.driver_core import cpu_device as _cpu_device


class CPU:
    """This represents an instance of a logical cpu device."""

    def __init__(self):
        # FIXME: Use device descriptors
        self._device = _cpu_device(-1)

    def __str__(self) -> str:
        return str(self._device)
