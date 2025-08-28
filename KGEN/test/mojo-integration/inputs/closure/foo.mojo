# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn printIt[kernel: fn (x: Int) unified -> Int](func: kernel, y: Int):
    print(func(y))
