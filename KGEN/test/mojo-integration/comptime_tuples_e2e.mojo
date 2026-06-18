# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo -O0 %s | FileCheck %s

from std.utils.variant import Variant
from std.memory import alloc


# NOTE: This struct uses UnsafePointer-based storage instead of List because
# List requires T: Copyable at the field declaration site, which creates a
# circular dependency with Variant's conditional Copyable conformance:
#   MTuple contains List[Variant[T, MTuple[T]]]
#   -> List requires Variant[T, MTuple[T]]: Copyable
#   -> requires MTuple[T]: Copyable (conditional on Variant)
#   -> MTuple's fields include List[Variant[T, MTuple[T]]]... (cycle)
# UnsafePointer avoids this because it doesn't constrain its element type,
# and Copyable checks in method bodies resolve after the struct is defined.
struct MTuple[T: ImplicitlyCopyable](ImplicitlyCopyable, Writable):
    comptime Element = Variant[Self.T, Self]
    var _data: UnsafePointer[Self.Element, MutUntrackedOrigin]
    var _len: Int
    var _cap: Int

    @always_inline
    def __init__(out self):
        self._data = UnsafePointer[
            Self.Element, MutUntrackedOrigin
        ].unsafe_dangling()
        self._len = 0
        self._cap = 0

    @always_inline
    def __init__(out self, var value: Self.Element):
        self._cap = 4
        self._data = alloc[Self.Element](self._cap)
        self._data.init_pointee_move(value^)
        self._len = 1

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self._data = move._data
        self._len = move._len
        self._cap = move._cap
        move._data = UnsafePointer[
            Self.Element, MutUntrackedOrigin
        ].unsafe_dangling()
        move._len = 0
        move._cap = 0

    def __init__(out self, *, copy: Self):
        self._len = copy._len
        self._cap = copy._len
        if copy._len > 0:
            self._data = alloc[Self.Element](copy._len)
            for i in range(copy._len):
                (self._data + i).init_pointee_copy(copy._data[i])
        else:
            self._data = UnsafePointer[
                Self.Element, MutUntrackedOrigin
            ].unsafe_dangling()

    def __del__(deinit self):
        for i in range(self._len):
            (self._data + i).destroy_pointee()
        if self._cap > 0:
            self._data.free()

    def _grow_if_needed(mut self):
        if self._len >= self._cap:
            var new_cap = self._cap * 2 if self._cap > 0 else 4
            var new_data = alloc[Self.Element](new_cap)
            for i in range(self._len):
                (new_data + i).init_pointee_move(
                    (self._data + i).take_pointee()
                )
            if self._cap > 0:
                self._data.free()
            self._data = new_data
            self._cap = new_cap

    def _append(mut self, value: Self.Element):
        self._grow_if_needed()
        (self._data + self._len).init_pointee_copy(value)
        self._len += 1

    @always_inline
    def cons(self, var other: Self) -> Self:
        var new = self
        for i in range(other._len):
            new._append(other._data[i])
        return new

    @always_inline
    def __add__(self, var other: Self) -> Self:
        var new = Self()
        for i in range(self._len):
            new._append(self._data[i])
        for i in range(other._len):
            new._append(other._data[i])
        return new

    def write_to(self, mut writer: Some[Writer]):
        writer.write("(")

        for i in range(self._len):
            if self._data[i].isa[Int]():
                var value = self._data[i]
                writer.write(value[Int])
            elif self._data[i].isa[MTuple[Self.T]]():
                var value = self._data[i]
                writer.write(value[MTuple[Self.T]])
            else:
                writer.write("?")
            if i < self._len - 1:
                writer.write(", ")

        writer.write(")")


comptime IntTuple = MTuple[Int]


def main():
    comptime tup = IntTuple(IntTuple(3) + IntTuple(4))
    # CHECK: (3, 4)
    print(tup)
    add_print[tup]()


def add_print[x: IntTuple]():
    comptime tup = x + IntTuple(4)
    # CHECK: ((3, 4), 4)
    print(tup)
