# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s


fn var_let_decls():
    # CHECK: %xx = lit.varlet.decl "xx"  var
    # CHECK: %[[V1:.*]] = kgen.param.constant: !Int = <{42}>
    # CHECK: lit.ref.store %[[V1]], %xx
    var xx = 42

    # CHECK: %yy = lit.varlet.decl "yy"  var
    # CHECK: %[[V3:.*]] = kgen.param.constant: !FloatLiteral = <{:scalar<f64> "1"}>
    # CHECK: lit.ref.store %[[V3]], %yy
    var yy = 1.0

    # CHECK: %str = lit.varlet.decl {{.*}} : !lit.ref<!StringLiteral,
    # CHECK: [[CONST:%.*]] = kgen.param.constant: !StringLiteral = <{:string "hello"}>
    # CHECK: lit.ref.store [[CONST]], %str
    var str = "hello"
