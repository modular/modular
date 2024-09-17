# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s


from nvml import Device
from nvml.nvml import _get_nvml_library_path


fn has_nvml_library() -> Bool:
    try:
        _ = _get_nvml_library_path()
        return True
    except:
        return False


def main():
    if not has_nvml_library():
        return

    var dev = Device(0)
    for clock in dev.mem_clocks():
        # CHECK: Clock =
        print("Clock =", clock[])
