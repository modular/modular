# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Index import StaticIntTuple, Index
from TargetInfo import has_avx512f, has_neon

# conv uses a different kernel than matmul
fn get_conv_a_row_size() -> Int:
    @parameter
    if has_neon():
        return 8
    elif has_avx512f():
        return 5
    return 5


fn get_conv_pack_inner_size() -> Int:
    @parameter
    if has_neon():
        return 2
    elif has_avx512f():
        return 4
    return 4
