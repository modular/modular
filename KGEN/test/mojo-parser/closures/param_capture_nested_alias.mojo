# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


fn use[x: Int]():
    pass


fn param[x: Int]():
    # CHECK: lit.fn *"param_closure
    @parameter
    fn param_closure[y: Int]():

        # CHECK: !lit.ref<!lit.struct<#escaping0 <:!Int y>
        fn closure() escaping:
            use[y]()
