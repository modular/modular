# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test compile-time errors for struct field reflection APIs.

# RUN: not kgen %s -elaborate -D TEST_NONEXISTENT_INDEX=1 2>&1 | FileCheck %s --check-prefix=CHECK-INDEX
# RUN: not kgen %s -elaborate -D TEST_NONEXISTENT_TYPE=1 2>&1 | FileCheck %s --check-prefix=CHECK-TYPE
# RUN: not kgen %s -elaborate -D TEST_OFFSET_NONEXISTENT_FIELD=1 2>&1 | FileCheck %s --check-prefix=CHECK-OFFSET-NAME
# RUN: not kgen %s -elaborate -D TEST_OFFSET_OUT_OF_BOUNDS=1 2>&1 | FileCheck %s --check-prefix=CHECK-OFFSET-INDEX
# RUN: not kgen %s -elaborate -D TEST_OFFSET_NEGATIVE_INDEX=1 2>&1 | FileCheck %s --check-prefix=CHECK-OFFSET-NEGATIVE

from sys import env_get_bool

from reflection import (
    struct_field_index_by_name,
    struct_field_type_by_name,
    offset_of,
    get_type_name,
)


struct TestStruct:
    var x: Int
    var y: Float64


# Test that struct_field_index_by_name produces an error for non-existent field.
# Use constrained to force compile-time evaluation.
fn test_nonexistent_field_index():
    comptime if env_get_bool["TEST_NONEXISTENT_INDEX", False]():
        # CHECK-INDEX: has no field named 'nonexistent'
        comptime assert (
            struct_field_index_by_name[TestStruct, "nonexistent"]() == 0
        ), "should not reach here"


# Test that struct_field_type_by_name produces an error for non-existent field.
fn test_nonexistent_field_type():
    comptime if env_get_bool["TEST_NONEXISTENT_TYPE", False]():
        # CHECK-TYPE: has no field named 'missing_field'
        comptime field_type = struct_field_type_by_name[
            TestStruct, "missing_field"
        ]()
        # Force evaluation by using the type
        comptime assert (
            get_type_name[field_type.T]() == "Int"
        ), "should not reach here"


# Test that offset_of[name=] produces an error for non-existent field.
fn test_offset_nonexistent_field():
    comptime if env_get_bool["TEST_OFFSET_NONEXISTENT_FIELD", False]():
        # CHECK-OFFSET-NAME: has no field named 'does_not_exist'
        comptime assert (
            offset_of[TestStruct, name="does_not_exist"]() == 0
        ), "should not reach here"


# Test that offset_of[index=] produces an error for out-of-bounds index.
fn test_offset_out_of_bounds():
    comptime if env_get_bool["TEST_OFFSET_OUT_OF_BOUNDS", False]():
        # CHECK-OFFSET-INDEX: field index 99 is out of bounds for struct with 2 fields
        comptime assert (
            offset_of[TestStruct, index=99]() == 0
        ), "should not reach here"


# Test that offset_of[index=] produces an error for negative index.
fn test_offset_negative_index():
    comptime if env_get_bool["TEST_OFFSET_NEGATIVE_INDEX", False]():
        # CHECK-OFFSET-NEGATIVE: field index -1 is out of bounds for struct with 2 fields
        comptime assert (
            offset_of[TestStruct, index= -1]() == 0
        ), "should not reach here"


fn main():
    test_nonexistent_field_index()
    test_nonexistent_field_type()
    test_offset_nonexistent_field()
    test_offset_out_of_bounds()
    test_offset_negative_index()
