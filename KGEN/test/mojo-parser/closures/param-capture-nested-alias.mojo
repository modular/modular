# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | kgen-opt -verify-parameters | FileCheck %s

alias int = __mlir_type.index


fn use[x: int]():
    pass


fn param[x: int]():
    # CHECK: lit.func *"param_closure
    @parameter
    fn param_closure():
        # CHECK: decl [[y:.*]] =
        alias y = x

        # CHECK: !lit.ref<{{.*}}_CI_{{.*}}escaping0"<[[y]]>
        fn closure() escaping:
            use[y]()
