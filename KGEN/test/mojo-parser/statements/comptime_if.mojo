# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s --verify-diagnostics | FileCheck %s

# ===----------------------------------------------------------------------=== #
# comptime if
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"comptime_if_basic{{.*}}"<a: i1>()
def comptime_if_basic[a: __mlir_type.i1]():
    # CHECK: kgen.param.if <a> {
    comptime if a:
        # CHECK: lit.var.decl "inside" var
        var inside: Int
    # CHECK: kgen.param.yield
    # CHECK: }


# CHECK-LABEL: lit.fn @"comptime_if_elif{{.*}}"<a: i1, b: !Bool>()
def comptime_if_elif[a: __mlir_type.i1, b: Bool]():
    # CHECK: kgen.param.if <a> {
    comptime if a:
        # CHECK: lit.var.decl "inside_1" var
        var inside_1: Int
    # CHECK: } else {
    # CHECK:     kgen.param.if <#lit.struct.extract<:!Bool b, "_mlir_value">> {
    elif b:
        # CHECK:     lit.var.decl "inside_2" var
        var inside_2: Int
    # CHECK:     kgen.param.yield
    # CHECK:   }
    # CHECK:   kgen.param.yield
    # CHECK: }


# CHECK-LABEL: lit.fn @"comptime_if_else{{.*}}"<a: i1>()
def comptime_if_else[a: __mlir_type.i1]():
    # CHECK: kgen.param.if <a> {
    comptime if a:
        # CHECK: lit.var.decl "inside_then" var
        var inside_then: Int
    # CHECK: } else {
    else:
        # CHECK: lit.var.decl "inside_else" var
        var inside_else: Int
    # CHECK: }


# ===----------------------------------------------------------------------=== #
# @parameter if (legacy syntax - same IR as comptime if)
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"param_if{{.*}}"<a: i1, b: !Bool>()
def param_if[a: __mlir_type.i1, b: Bool]():
    # CHECK: kgen.param.if <a> {
    # expected-warning @+1 {{'@parameter if' is deprecated, use 'comptime if' instead}}
    @parameter
    if a:
        # CHECK: lit.var.decl "inside_1" var
        var inside_1: Int
    # CHECK: } else {
    # CHECK:     kgen.param.if <#lit.struct.extract<:!Bool b, "_mlir_value">> {
    elif b:
        # CHECK:     lit.var.decl "inside_2" var
        var inside_2: Int
    # CHECK:     kgen.param.yield
    # CHECK:   }
    # CHECK:   kgen.param.yield
    # CHECK: }


# CHECK-LABEL: lit.fn @"param_if_andor_i1{{.*}}"<a: i1, b: i1>()
def param_if_andor_i1[a: __mlir_type.i1, b: __mlir_type.i1]():
    # CHECK: kgen.param.if <cond(a, b, a)>
    # expected-warning @+1 {{'@parameter if' is deprecated, use 'comptime if' instead}}
    @parameter
    if a and b:
        # CHECK:   lit.var.decl "v" var
        var v: Int
    # CHECK:   kgen.param.yield
    # CHECK: } else {
    # CHECK: kgen.param.if <cond(a, a, b)>
    elif a or b:
        # CHECK:   lit.var.decl "w" var
        var w: Int


# CHECK-LABEL: lit.fn @"param_if_and{{.*}}"<a: !Bool, b: !Bool>()
def param_if_and[a: Bool, b: Bool]():
    # CHECK: kgen.param.if <#lit.struct.extract<:!Bool cond(#lit.struct.extract<:!Bool a, "_mlir_value">, b, a), "_mlir_value">>
    # expected-warning @+1 {{'@parameter if' is deprecated, use 'comptime if' instead}}
    @parameter
    if a and b:
        # CHECK:   lit.var.decl "v" var
        var v: Int
    # CHECK:   kgen.param.yield
    # CHECK: }
