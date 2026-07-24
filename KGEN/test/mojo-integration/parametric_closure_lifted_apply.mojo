# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s

# Regression test for https://github.com/modular/modular/issues/6805: binding
# a parametric capturing closure through two levels of higher-order generics
# crashed generator specialization when the closure's return type has a field
# sized by derived comptime aliases (lifted vs concrete `apply` skew).


trait Cipher:
    comptime BLOCK_SIZE: Int


@fieldwise_init
struct MyCipher[KeySize: Int](Cipher, Copyable, Movable):
    comptime BLOCK_SIZE: Int = 16
    comptime NK: Int = Self.KeySize // 4
    comptime WORDS_SIZE: Int = 4 * (Self.NK + 7)
    var w: InlineArray[UInt32, Self.WORDS_SIZE]


def check[
    C: Cipher & Copyable & ImplicitlyDeletable,
    KeySize: Int,
    cipher_init: def(InlineArray[UInt8, KeySize]) raises capturing[_] -> C,
](n: Int) raises:
    var k = InlineArray[UInt8, KeySize](fill=0)
    var c = cipher_init(k)
    print("checked KeySize =", KeySize, "n =", n)


def run[
    check: def[
        C: Cipher & Copyable & ImplicitlyDeletable,
        KeySize: Int,
        cipher_init: def(InlineArray[UInt8, KeySize]) raises capturing[_] -> C,
    ](Int) raises capturing[_],
](n: Int) raises:
    @parameter
    def make[
        KeySize: Int
    ](key: InlineArray[UInt8, KeySize]) raises -> MyCipher[KeySize]:
        return MyCipher[KeySize](
            InlineArray[UInt32, MyCipher[KeySize].WORDS_SIZE](fill=0)
        )

    check[MyCipher[16], 16, make[16]](n)
    check[MyCipher[24], 24, make[24]](n)
    check[MyCipher[32], 32, make[32]](n)


def main() raises:
    # CHECK: checked KeySize = 16 n = 3
    # CHECK: checked KeySize = 24 n = 3
    # CHECK: checked KeySize = 32 n = 3
    run[check](3)
