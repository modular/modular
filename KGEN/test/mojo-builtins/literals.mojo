# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s


fn var_let_decls():
    # CHECK: %[[V0:.*]] = kgen.param.constant: !Int = <#lit.struct<{value = 10}>>
    # CHECK: %x = lit.letreg.decl "x" = %[[V0]] : !Int
    let x = 10

    # CHECKL: %xx = lit.varlet.decl "xx"  var
    # CHECKL: %[[V1:.*]] = kgen.param.constant: !Int = <#lit.struct<{value = 42}>>
    # CHECKL: lit.ref.store %[[V1]], %xx
    var xx = 42

    # CHECK: %[[V4:.*]] = kgen.param.constant: !FloatLiteral = <{{.*}}"1"{{.*}}>
    # CHECK: %y = lit.letreg.decl "y" = %[[V4]] : !FloatLiteral
    let y = 1.0

    # CHECK: %yy = lit.varlet.decl "yy"  var
    # CHECK: %[[V3:.*]] = kgen.param.constant: !FloatLiteral = <#lit.struct<{value: scalar<f64> = "1"}>>
    # CHECK: lit.ref.store %[[V3]], %yy
    var yy = 1.0

    # CHECK: kgen.param.constant: !StringLiteral = <#lit.struct<{value: string = "hello"}>>
    let const_str = "hello"

    # CHECK: %str = lit.varlet.decl {{.*}} : !lit.ref<!StringLiteral,
    # CHECK: [[CONST:%.*]] = kgen.param.constant: {{.*}} = "hello"
    # CHECK: lit.ref.store [[CONST]], %str
    var str = "hello"
