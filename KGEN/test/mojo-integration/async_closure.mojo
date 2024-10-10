# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s

from runtime.asyncrt import run


@always_inline
async fn just_call_it[func: async fn[a: Int] (Int) capturing -> Int]() -> Int:
    var coro = func[3](2)
    var result = await coro^
    return result


fn foobar[pref: Int](a: Int):
    @parameter
    async fn but_async[c: Int](b: Int) -> Int:
        return a + b + c

    var coro = just_call_it[but_async]()
    print(pref)
    print(run(coro^))


fn main():
    # CHECK: 10
    # CHECK: 6
    foobar[10](1)
    # CHECK: 20
    # CHECK: 7
    foobar[20](2)
