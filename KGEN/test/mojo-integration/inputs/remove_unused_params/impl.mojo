# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct FactorialComputer(Copyable, Movable):
    fn compute_method(self, depth: UInt8) -> UInt8:
        if depth == 0:
            return 1
        return depth * self.compute_method(depth - 1)


fn compute_unusedPost(depth: UInt8, unused: Bool) -> UInt8:
    if depth == 0:
        return 1
    return depth * compute_unusedPost(depth - 1, unused)


fn compute_unusedPre(unused: Bool, depth: UInt8) -> UInt8:
    if depth == 0:
        return 1
    return depth * compute_unusedPre(unused, depth - 1)


@export
fn use_it():
    var a = FactorialComputer().compute_method(2)
    print(a)
    var b = compute_unusedPost(2, False)
    print(b)
    var c = compute_unusedPre(False, 2)
    print(c)
