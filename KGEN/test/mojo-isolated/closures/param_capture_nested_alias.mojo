# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


fn use[x: int]():
    pass


fn param[x: int]():
    # CHECK: lit.func *"param_closure
    @parameter
    fn param_closure[y: int]():

        # CHECK: !lit.ref<{{.*}}_CI_{{.*}}escaping0"<y>
        fn closure() escaping:
            use[y]()
