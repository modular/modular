# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s


fn var_let_decls():
    # CHECK: %xx = lit.var.decl "xx"  var
    # CHECK: %[[V1:.*]] = kgen.param.constant: !Int = <{42}>
    # CHECK: lit.ref.store %[[V1]], %xx
    var xx = 42

    # CHECK: lit.alias.decl {{.*}}il{{.*}}: !IntLiteral = <{:!kgen.int_literal 43}>
    alias il = 43

    # CHECK: %yy = lit.var.decl "yy"  var
    # CHECK: %[[V3:.*]] = kgen.param.constant: !FloatLiteralOld = <{:scalar<f64> "1"}>
    # CHECK: lit.ref.store %[[V3]], %yy
    var yy = 1.0

    # CHECK: lit.alias.decl {{.*}}fl{{.*}}: !FloatLiteral = <{:!kgen.float_literal #kgen.float_literal<normal (2|1)>}>
    alias fl = 2.0

    # CHECK: %str = lit.var.decl {{.*}} : !lit.ref<!StringLiteral,
    # CHECK: [[CONST:%.*]] = kgen.param.constant: !StringLiteral = <{:string "hello"}>
    # CHECK: lit.ref.store [[CONST]], %str
    var str = "hello"
