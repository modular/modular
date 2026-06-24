# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -split-input-file -I=%S/inputs -verify-diagnostics %s | FileCheck %s

# This test checks that we pick the source package named test_package over the
# identically named test_package.mojo module in the same directory. If we did,
# we wouldn't see lit.package below; we'd see lit.file_module.

# CHECK: lit.package @test_package
import test_package

def main():
  test_package.method_defined_in_init()
