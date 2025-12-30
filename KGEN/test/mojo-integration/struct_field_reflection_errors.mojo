# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test compile-time errors for struct field reflection APIs.

# RUN: not kgen %s -elaborate -D TEST_NONEXISTENT_INDEX=1 2>&1 | FileCheck %s --check-prefix=CHECK-INDEX
# RUN: not kgen %s -elaborate -D TEST_NONEXISTENT_TYPE=1 2>&1 | FileCheck %s --check-prefix=CHECK-TYPE

from sys import env_get_bool

from reflection import (
    struct_field_index_by_name,
    struct_field_type_by_name,
    get_type_name,
)


struct TestStruct:
    var x: Int
    var y: Float64


# Test that struct_field_index_by_name produces an error for non-existent field.
# Use constrained to force compile-time evaluation.
fn test_nonexistent_field_index():
    @parameter
    if env_get_bool["TEST_NONEXISTENT_INDEX", False]():
        # CHECK-INDEX: has no field named 'nonexistent'
        constrained[
            struct_field_index_by_name[TestStruct, "nonexistent"]() == 0,
            "should not reach here",
        ]()


# Test that struct_field_type_by_name produces an error for non-existent field.
fn test_nonexistent_field_type():
    @parameter
    if env_get_bool["TEST_NONEXISTENT_TYPE", False]():
        # CHECK-TYPE: has no field named 'missing_field'
        comptime field_type = struct_field_type_by_name[
            TestStruct, "missing_field"
        ]()
        # Force evaluation by using the type
        constrained[
            get_type_name[field_type.T]() == "Int",
            "should not reach here",
        ]()


fn main():
    test_nonexistent_field_index()
    test_nonexistent_field_type()
