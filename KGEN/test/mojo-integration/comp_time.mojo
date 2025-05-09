# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


fn diverge_comptime(i: Int) -> Int:
    var t = 0
    # Intentionally have then and else branches mismatch logic for testing.
    if __mlir_op.`kgen.is_compile_time`():
        t = i + 40
    else:
        t = i + 1

    return t + 1


fn test_is_compile_time():
    # CHECK-LABEL: testing test_is_compile_time
    print("testing test_is_compile_time")

    # CHECK: interpret value: 42
    alias a = diverge_comptime(1)
    print("interpret value:", a)

    # CHECK: runtime value: 3
    var b = diverge_comptime(1)
    print("runtime value:", b)


fn might_throw(cond: Bool) -> Int:
    var result = 0
    try:
        if cond:
            raise "something"

        result = 4
    except e:
        return len(String(e)) * 4

    else:
        result += 1
    finally:
        result += 2

    return result


# MOCO-246: Test EH at comptime.
fn test_exception_handling():
    # CHECK-LABEL: test_exception_handling
    print("test_exception_handling")

    # CHECK: interpret value: 36
    alias a = might_throw(True)
    print("interpret value:", a)

    # CHECK: run value: 36
    print("run value:", might_throw(True))

    # CHECK: interpret value: 7
    alias b = might_throw(False)
    print("interpret value:", b)

    # CHECK: run value: 7
    print("run value:", might_throw(False))


fn main():
    test_is_compile_time()
    test_exception_handling()
