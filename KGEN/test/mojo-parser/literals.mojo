# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s


fn var_let_decls():
    # CHECK: %xx = lit.var.decl "xx" var
    # CHECK: %[[V1:.*]] = kgen.param.constant: !Int = <{42}>
    # CHECK: lit.ref.store %[[V1]], %xx
    var xx = 42

    # CHECK: lit.alias.decl {{.*}}il{{.*}}@IntLiteral<:!pop.int_literal 43>
    alias il = 43

    # CHECK: %yy = lit.var.decl "yy" var
    # CHECK: [[TMP:%.*]] = kgen.param.constant: {{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<1|1>>
    # CHECK: %[[V3:.*]] = lit.call {{.*}}SIMD::@"__init__{{.*}}([[TMP]])
    # CHECK: lit.ref.store %[[V3]], %yy
    var yy = 1.0

    # CHECK: lit.alias.decl {{.*}}fl1{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<2|1>> = <*?>
    alias fl1 = 2.0
    # CHECK: lit.alias.decl {{.*}}fl2{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<63|10>> = <*?>
    alias fl2 = 6.3
    # CHECK: lit.alias.decl {{.*}}fl3{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<41|2>> = <*?> 
    alias fl3 = 20.5
    # CHECK: lit.alias.decl {{.*}}fl4{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<-41|2>> = <*?> 
    alias fl4 = -20.5
    # CHECK: lit.alias.decl {{.*}}fl5{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<neg_zero>> = <*?>
    alias fl5 = -0.0

    # TODO - Python raises an error when dividing by zero.  We need support for
    # parameter-time evaluation of `raise` to support that semantics, in which
    # case these will be static errors instead.
    # CHECK: lit.alias.decl {{.*}}flDivZero{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<nan>> = <*?> 
    alias flDivZero = 5.0 / 0.0
    # CHECK: lit.alias.decl {{.*}}flDivNegZero{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<nan>> = <*?> 
    alias flDivNegZero = 5.0 / -0.0

    # CHECK: %str = lit.var.decl {{.*}} : !lit.ref<{{.*}}StringLiteral<:string "hello">
    # CHECK: [[CONST:%.*]] = kgen.param.constant: {{.*}}@StringLiteral<:string "hello"> = <*?>
    # CHECK: lit.ref.store [[CONST]], %str
    var str = "hello"
