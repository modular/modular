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
fn comptime_if_basic[a: __mlir_type.i1]():
    # CHECK: kgen.param.if <a> {
    comptime if a:
        # CHECK: lit.var.decl "inside" var
        var inside: Int
    # CHECK: kgen.param.yield
    # CHECK: }


# CHECK-LABEL: lit.fn @"comptime_if_elif{{.*}}"<a: i1, b: !Bool>()
fn comptime_if_elif[a: __mlir_type.i1, b: Bool]():
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
fn comptime_if_else[a: __mlir_type.i1]():
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
fn param_if[a: __mlir_type.i1, b: Bool]():
    # CHECK: kgen.param.if <a> {
    #expected-warning @+1 {{'@parameter if' is deprecated, use 'comptime if' instead}}
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
fn param_if_andor_i1[a: __mlir_type.i1, b: __mlir_type.i1]():
    # CHECK: kgen.param.if <cond(a, b, a)>
    #expected-warning @+1 {{'@parameter if' is deprecated, use 'comptime if' instead}}
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
fn param_if_and[a: Bool, b: Bool]():
    # CHECK: kgen.param.if <#lit.struct.extract<:!Bool cond(#lit.struct.extract<:!Bool a, "_mlir_value">, b, a), "_mlir_value">>
    #expected-warning @+1 {{'@parameter if' is deprecated, use 'comptime if' instead}}
    @parameter
    if a and b:
        # CHECK:   lit.var.decl "v" var
        var v: Int
    # CHECK:   kgen.param.yield
    # CHECK: }

# Minimal local stand-in for std.sys.is_run_in_comptime_interpreter, matching its definition.
@always_inline("nodebug")
fn is_run_in_comptime_interpreter() -> Bool:
    return __mlir_op.`kgen.is_run_in_comptime_interpreter`()


# Warnings appear on stderr before IR output, so all CHECK/CHECK-NOT lines
# for warnings must be ordered before any CHECK-LABEL that matches IR.

fn test_direct_call():
    #expected-warning @+1 {{'is_run_in_comptime_interpreter()' is always true as a 'comptime if' condition; use runtime 'if'}}
    comptime if is_run_in_comptime_interpreter():
        var x: Int


fn test_negated_call():
    #expected-warning @+1 {{'is_run_in_comptime_interpreter()' is always true as a 'comptime if' condition; use runtime 'if'}}
    comptime if not is_run_in_comptime_interpreter():
        var x: Int


# No warning for a regular compile-time parameter.
fn test_no_warning_for_regular_param[cond: __mlir_type.i1]():
    comptime if cond:
        var x: Int


fn test_elif_also_warns[cond: __mlir_type.i1]():
    #expected-warning @+1 {{'is_run_in_comptime_interpreter()' is always true as a 'comptime if' condition; use runtime 'if'}}
    comptime if is_run_in_comptime_interpreter():
        var x: Int
    #expected-warning @+1 {{'is_run_in_comptime_interpreter()' is always true as a 'comptime if' condition; use runtime 'if'}}
    elif is_run_in_comptime_interpreter():
        var y: Int

fn test_and_call():
    #expected-warning @+1 {{'is_run_in_comptime_interpreter()' is always true as a 'comptime if' condition; use runtime 'if'}}
    comptime if is_run_in_comptime_interpreter() and True:
        var x: Int

fn test_or_call():
    #expected-warning @+1 {{'is_run_in_comptime_interpreter()' is always true as a 'comptime if' condition; use runtime 'if'}}
    comptime if is_run_in_comptime_interpreter() or False:
        var x: Int

fn test_nested_call():
    #expected-warning @+1 {{'is_run_in_comptime_interpreter()' is always true as a 'comptime if' condition; use runtime 'if'}}
    comptime if ((not (is_run_in_comptime_interpreter() and True)) or (not is_run_in_comptime_interpreter())):
        var x: Int
