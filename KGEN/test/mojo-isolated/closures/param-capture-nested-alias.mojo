# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s


fn use[x: int]():
    pass


fn param[x: int]():
    # CHECK: lit.func *"param_closure
    @parameter
    fn param_closure():
        # CHECK: lit.alias.decl [[y:.*]] =
        alias y = x

        # CHECK: !lit.ref<{{.*}}_CI_{{.*}}escaping0"<[[y]]>
        fn closure() escaping:
            use[y]()
