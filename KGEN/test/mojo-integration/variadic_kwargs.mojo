# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


def takes_int_variadic_kwargs(**kwargs: Int) raises:
    var key = "stuff"
    # CHECK: stuff 8
    print(key, kwargs[key])
    # CHECK: x 9
    print("x", kwargs["x"])

    try:
        _ = kwargs["non-existent"]
    except:
        # CHECK: non-existent key not found (as expected)
        print("non-existent key not found (as expected)")


trait Resettable(ImplicitlyCopyable, ImplicitlyDeletable):
    def reset(mut self):
        ...

    def get(self) -> Int:
        ...


@fieldwise_init
struct MemOnly(Resettable):
    var value: Int

    def get(self) -> Int:
        return self.value

    def reset(mut self):
        self.value = 0


def takes_mem_only_variadic_kwargs[T: Resettable](var **kwargs: T) raises:
    var key = "fizzbuzz"
    # CHECK: fizzbuzz 13
    print(key, kwargs[key].get())
    # CHECK: y 42
    print("y", kwargs["y"].get())

    try:
        _ = kwargs[""]
    except:
        # CHECK: empty key not found
        print("empty key not found")

    kwargs[key].reset()
    # CHECK: fizzbuzz now 0
    print(key, "now", kwargs[key].get())


def main() raises:
    takes_int_variadic_kwargs(x=9, stuff=8)

    var m = MemOnly(42)
    takes_mem_only_variadic_kwargs(y=m, fizzbuzz=MemOnly(13))
    # CHECK: m outside 42
    print("m outside", m.value)
