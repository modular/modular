# RUN: not %mojo %s 2>&1 | FileCheck %s
def main():
    # CHECK: SIMD: expected 4 elements, received 2
    var v2: SIMD[DType.int32, 4] = [1, 2]
