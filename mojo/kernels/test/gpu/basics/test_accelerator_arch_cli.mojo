# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug --target-accelerator=nvidia:sm80 %s | FileCheck --check-prefix=CHECK-NV80 %s
# RUN: %mojo-no-debug --target-accelerator=nvidia:sm90 %s | FileCheck --check-prefix=CHECK-NV90 %s
# RUN: %mojo-no-debug --target-accelerator=some_amd:300 %s | FileCheck --check-prefix=CHECK-A300 %s
# RUN: %mojo-no-debug --target-accelerator=quantum:3000 %s | FileCheck --check-prefix=CHECK-Q3000 %s

from sys.info import _accelerator_arch


def main():
    # CHECK-NV80: nvidia:sm80
    # CHECK-NV90: nvidia:sm90
    # CHECK-A300: some_amd:300
    # CHECK-Q3000: quantum:3000
    print(_accelerator_arch())
