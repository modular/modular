# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


fn takes_int_variadic_kwargs(**kwargs: Int) raises:
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


trait Resetable(Copyable, Movable):
    fn reset(mut self):
        ...

    fn get(self) -> Int:
        ...


@value
struct MemOnly(Resetable):
    var value: Int

    fn get(self) -> Int:
        return self.value

    fn reset(mut self):
        self.value = 0


fn takes_mem_only_variadic_kwargs[T: Resetable](owned **kwargs: T) raises:
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


fn main() raises:
    takes_int_variadic_kwargs(x=9, stuff=8)

    var m = MemOnly(42)
    takes_mem_only_variadic_kwargs(y=m, fizzbuzz=MemOnly(13))
    # CHECK: m outside 42
    print("m outside", m.value)
