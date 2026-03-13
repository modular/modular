# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo-build %s -o %t/this_dir_cant_exist/output 2>&1 | FileCheck %s --check-prefix=INVALID_OUTPUT_DIR
# INVALID_OUTPUT_DIR: error: unable to write file. The path '{{.*}}' does not exist.


def main():
    return
