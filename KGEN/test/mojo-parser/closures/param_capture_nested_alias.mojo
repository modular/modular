# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


def use[x: Int]():
    pass


def param[x: Int]():
    # CHECK: lit.fn *"param_closure
    @parameter
    def param_closure[y: Int]():

        # CHECK: !lit.ref<!lit.struct<#escaping0 <:!Int y>
        def closure() escaping:
            use[y]()
