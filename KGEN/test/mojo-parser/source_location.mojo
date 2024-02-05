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
    # CHECK: file_name: !StringLiteral = #lit.struct<{value: string = "{{.*}}KGEN/test/mojo-parser/source_location.mojo"}>,
    # CHECK: function_name: !StringLiteral = #lit.struct<{value: string = "foo"}>,
    # CHECK: line: !Int = #lit.struct<{value = 18}>}>>
    return __source_location()
