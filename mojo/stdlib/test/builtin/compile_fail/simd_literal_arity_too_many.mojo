# RUN: not %mojo %s 2>&1 | FileCheck %s
def main():
    # CHECK: SIMD: expected 2 elements, received 4
    var v1: SIMD[DType.int32, 2] = [1, 2, 3, 4]
