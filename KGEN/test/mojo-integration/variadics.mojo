# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


# This isn't copyable or movable, but it is talkative!
struct TalkativeMem(Stringable):
    var state: Int

    fn __init__(inout self, state: Int):
        self.state = state
        print("initializing ", end="")
        print(state)

    fn __del__(owned self):
        print("destroying ", end="")
        print(self.state)

    fn __str__(self) -> String:
        return "talkative " + self.state.__str__()


# This isn't copyable or movable, but it is talkative!
@register_passable
struct TalkativeReg(Stringable):
    var state: Int

    fn __init__(inout self, state: Int):
        self.state = state
        print("initializing ", end="")
        print(state)

    fn __del__(owned self):
        print("destroying ", end="")
        print(self.state)

    fn __str__(self) -> String:
        return "talkative " + self.state.__str__()


# ===----------------------------------------------------------------------=== #
# Inout varargs
# ===----------------------------------------------------------------------=== #


fn test_inout_varargs():
    # CHECK: -- Testing inout varargs
    print("-- Testing inout varargs")
    var s1: String = "hello"
    var s2: String = "konnichiwa"
    var s3: String = "bonjour"

    fn make_worldly(inout *strs: String):
        for i in range(len(strs)):
            strs[i] += " world"

    make_worldly(s1, s2, s3)
    print(s1)  # CHECK-NEXT: hello world
    print(s2)  # CHECK-NEXT: konnichiwa world
    print(s3)  # CHECK-NEXT: bonjour world

    # CHECK: -- Testing inout varargs destructors
    print("-- Testing inout varargs destructors")
    var v1 = TalkativeMem(1)  # CHECK-NEXT: initializing 1
    var v2 = TalkativeMem(2)  # CHECK-NEXT: initializing 2

    fn double(inout *tm: TalkativeMem):
        for i in range(len(tm)):
            tm[i].state *= 2

    double(v1, v2)
    # CHECK-NEXT: destroying 4
    # CHECK-NEXT: destroying 2


# ===----------------------------------------------------------------------=== #
# Owned varargs
# ===----------------------------------------------------------------------=== #


fn test_owned_varargs():
    # CHECK: -- testing owned mem varargs
    print("\n-- testing owned mem varargs")

    var v1 = TalkativeMem(1)  # CHECK-NEXT: initializing 1
    var v2 = TalkativeMem(2)  # CHECK-NEXT: initializing 2
    var v3 = TalkativeMem(3)  # CHECK-NEXT: initializing 3

    fn handle_owned_mem(owned *strs: TalkativeMem):
        # owned arguments are mutable and live as long as they are used.
        for i in range(len(strs)):
            strs[i].state *= 2

        # So they should die here, after the loop, before the print.
        # CHECK-NEXT: destroying 6
        # CHECK-NEXT: destroying 4
        # CHECK-NEXT: destroying 2

        # CHECK-NEXT: after last use
        print("after last use")

    handle_owned_mem(v1^, v2^, v3^)

    # CHECK-NEXT: after call
    print("after call")


fn test_owned_reg_varargs():
    # CHECK: -- testing owned reg varargs
    print("\n-- testing owned reg varargs")

    var v1 = TalkativeReg(1)  # CHECK-NEXT: initializing 1
    var v2 = TalkativeReg(2)  # CHECK-NEXT: initializing 2
    var v3 = TalkativeReg(3)  # CHECK-NEXT: initializing 3

    fn handle_owned_reg(owned *strs: TalkativeReg):
        # owned arguments are mutable and live as long as they are used.
        for s in strs:
            s[].state *= 2

        # So they should die here, after the loop, before the print.
        # CHECK-NEXT: destroying 6
        # CHECK-NEXT: destroying 4
        # CHECK-NEXT: destroying 2

        # CHECK-NEXT: after last use
        print("after last use")

    handle_owned_reg(v1^, v2^, v3^)

    # CHECK-NEXT: after call
    print("after call")


# ===----------------------------------------------------------------------=== #
# owned variadic packs
# ===----------------------------------------------------------------------=== #


fn test_owned_variadic_pack[*Ts: Stringable](owned *pack: *Ts):
    print("-- testing owned variadic pack with", len(pack), "elements")

    # TODO: Test mutation of the value not just reading of the value.
    fn process[T: Stringable](a: T):
        print("hello", a)

    pack.each[process]()


fn test_owned_variadic_pack():
    # CHECK: -- testing owned variadic pack with 2 elements
    # CHECK: hello foo
    # CHECK: hello 42
    test_owned_variadic_pack("foo", 42)
    print("")

    # CHECK: initializing 1
    # CHECK: initializing 2
    # CHECK: initializing 3
    # CHECK: -- testing owned variadic pack with 3 elements
    # CHECK: hello talkative 1
    # CHECK: hello talkative 2
    # CHECK: hello talkative 3
    # CHECK: destroying 3
    # CHECK: destroying 2
    # CHECK: destroying 1
    test_owned_variadic_pack(TalkativeMem(1), TalkativeMem(2), TalkativeMem(3))
    print("")


fn test_inout_variadic_pack[*Ts: Stringable](inout *pack: *Ts):
    print("-- testing inout variadic pack with", len(pack), "elements")

    # TODO: Test mutation of the value not just reading of the value.
    fn process[T: Stringable](a: T):
        print("hello", a)

    pack.each[process]()


fn test_inout_variadic_pack():
    # CHECK: -- testing inout variadic pack with 2 elements
    # CHECK: hello foo
    # CHECK: hello 42
    var string = "foo"
    var integer = 42
    test_inout_variadic_pack(string, integer)
    print("")

    # CHECK: initializing 1
    # CHECK: initializing 2
    # CHECK: -- testing inout variadic pack with 2 elements
    # CHECK: hello talkative 1
    # CHECK: hello talkative 2
    # CHECK: destroying 1
    # CHECK: after call
    # CHECK: destroying 2
    var m1 = TalkativeMem(1)
    var m2 = TalkativeMem(2)
    test_inout_variadic_pack(m1, m2)
    print("after call")
    _ = m2^
    print("")


fn main():
    test_inout_varargs()
    test_owned_varargs()
    test_owned_reg_varargs()
    test_owned_variadic_pack()
    test_inout_variadic_pack()
