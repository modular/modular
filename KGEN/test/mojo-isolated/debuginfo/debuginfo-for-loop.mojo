# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s | FileCheck %s

# CHECK: #[[LOCAL_VAR_I:.*]] = #debuginfo.local_variable<scope = #[[FOR_SP:.*]], name = "i", {{.*}}, line = [[LN:[0-9]+]], arg = 1


# CHECK-LABEL: lit.func @"structured_for_loop()"
fn structured_for_loop() -> __mlir_type.index:
    # CHECK: %0 = hlcf.loop (%arg0 loc({{.*}}) = %index0 : index) -> index {
    __mlir_region loop_body(i: __mlir_type.index):
        # CHECK-NEXT: debuginfo.value #[[LOCAL_VAR_I]] = %arg0 : index
        # CHECK-NEXT: %index1 = kgen.param.constant = <1>
        # CHECK-NEXT: %1 = index.add %arg0, %index1 loc(#[[FOR_ADD_LOC:.*]])
        # CHECK-NEXT: hlcf.continue %1 : index loc(#[[FOR_YIELD_LOC:.*]])
        __mlir_op.`hlcf.continue`(__mlir_op.`index.add`(i, `1`))

    # CHECK: lit.return %0 : index
    return __mlir_op.`hlcf.loop`[
        _type = __mlir_type.index, _region = __mlir_attr.`"loop_body"`
    ](`0`)
