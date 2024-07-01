# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo package %S/../../Inputs/test_user_op -o %t.mojopkg
# RUN: not %mojo %s %S/model_invalid_op.mlir %t.mojopkg 2>&1 | FileCheck %s

from sys import argv

from max.engine import InferenceSession
from pathlib import Path

# CHECK: user_invalid.mojo:{{.*}} error: call expansion failed
# CHECK-NEXT: constrained[False, "oops"]()
# CHECK: constrained.mojo:{{.*}} note: constraint failed: oops
# CHECK: error: failed to run the pass manager
# CHECK: model_invalid_op.mlir:{{.*}} error: KGEN elaboration failed
# CHECK-NEXT: mo.graph @elaboration_error_in_kernel()


def main():
    session = InferenceSession()
    args = argv()
    model_path = args[1]
    user_defined_ops_path = args[2]
    _ = session.load(
        Path(model_path),
        custom_ops_paths=List[Path](Path(user_defined_ops_path)),
    )
