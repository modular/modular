# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s
# RUN: %mojo -debug-level full %s | FileCheck %s

from IO import print


@always_inline
async fn just_call_it[func: async fn[a: Int] (Int) capturing -> Int]() -> Int:
    let coro: Coroutine[Int] = func[3](2)
    let result = await coro
    return result


fn foobar[pref: Int](a: Int):
    @parameter
    async fn but_async[c: Int](b: Int) -> Int:
        return a + b + c

    let coro: Coroutine[Int] = just_call_it[but_async]()
    print(pref)
    print(coro())


fn main():
    # CHECK: 10
    # CHECK: 6
    foobar[10](1)
    # CHECK: 20
    # CHECK: 7
    foobar[20](2)
