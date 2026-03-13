# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that JSON diagnostics include fixit information.
# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s 2>&1 | FileCheck %s

# Verify the JSON contains fixit with expected structure.
# CHECK: "diagnostic":{
# CHECK-SAME: "fixIts":[{
# CHECK-SAME: "end":{
# CHECK-SAME: "start":{
# CHECK-SAME: "text":"origin_of"


def test_fixit[T: AnyType](a: T):
    # __origin_of is deprecated and suggests origin_of as a fixit
    _ = __origin_of(a)


def main():
    test_fixit(1)
