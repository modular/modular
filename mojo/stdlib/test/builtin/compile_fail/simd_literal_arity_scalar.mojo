# RUN: not %mojo %s 2>&1 | FileCheck %s
def main():
    # CHECK: SIMD: expected 1 elements, received 3
    var x: Int = [42, 43, 44]
