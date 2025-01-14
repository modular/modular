# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


fn use[x: Index]():
    pass


fn param[x: Index]():
    # CHECK: lit.fn *"param_closure
    @parameter
    fn param_closure[y: Index]():

        # CHECK: !lit.ref<{{.*}}_CI_{{.*}}escaping0"<y>
        fn closure() escaping:
            use[y]()
