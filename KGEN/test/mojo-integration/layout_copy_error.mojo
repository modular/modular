# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that Layout is not implicitly copyable.
# This is a common error that users were hitting,
# so it's good to make sure we don't regress.

# RUN: not kgen %s -elaborate 2>&1 | FileCheck %s

from layout import Layout, LayoutTensor


def kernel(tensor: LayoutTensor):
    # CHECK: cannot materialize comptime value of type 'Layout' to runtime because it is not 'ImplicitlyCopyable'
    var size = tensor.layout.size()
