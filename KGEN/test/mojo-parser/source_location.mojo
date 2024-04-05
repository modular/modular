# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s

from builtin._location import _SourceLocation


# CHECK-LABEL: lit.func @"foo()"
fn foo() -> _SourceLocation:
    # CHECK: kgen.param.constant: !SourceLocation =
    # CHECK: file_name: !StringLiteral = {:string "{{.*}}KGEN/test/mojo-parser/source_location.mojo"},
    # CHECK: function_name: !StringLiteral = {:string "foo"},
    # CHECK: line: !Int = {18}}>
    return __source_location()


# CHECK-LABEL: lit.func @"bar
fn bar[x: _SourceLocation = __source_location()]() -> _SourceLocation:
    # CHECK: file_name: !StringLiteral = {:string "{{.*}}KGEN/test/mojo-parser/source_location.mojo"},
    # CHECK: function_name: !StringLiteral = {:string "None"},
    # CHECK: line: !Int = {22}}>
    return x
