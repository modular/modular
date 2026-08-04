# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


trait Nameable:
    def name(self) -> Int:
        ...


@fieldwise_init
struct Thing(Copyable, Movable):
    var value: Int

    def get(self) -> Int:
        return self.value
