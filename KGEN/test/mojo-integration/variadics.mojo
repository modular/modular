# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


# This isn't copyable or movable, but it is talkative!
struct TalkativeMem(Stringable, Formattable):
    var state: Int

    fn __init__(inout self, state: Int):
        self.state = state
        print("initializing", state)

    fn __del__(owned self):
        print("destroying", self.state)

    fn __str__(self) -> String:
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        writer.write("talkative ", self.state)


# This isn't copyable or movable, but it is talkative!
@register_passable
struct TalkativeReg(Stringable, Formattable):
    var state: Int

    fn __init__(inout self, state: Int):
        self.state = state
        print("initializing", state)

    fn __del__(owned self):
        print("destroying", self.state)

    fn __str__(self) -> String:
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        writer.write("talkative ", self.state)


# This is copyable, movable, and talkative!  It doesn't print on move.
@register_passable
struct TalkativeCopableReg(Stringable, Formattable):
    var state: Int

    fn __init__(inout self, state: Int):
        self.state = state
        print("initializing", state)

    fn __copyinit__(inout self, existing: Self):
        self.state = existing.state
        print("copying", self.state)

    fn __del__(owned self):
        print("destroying", self.state)

    fn __str__(self) -> String:
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        writer.write("talkative ", self.state)


# This is copyable, movable, and talkative!  It prints on move.
struct TalkativeCopableMovableMem(Stringable, Formattable, CollectionElement):
    var state: Int

    fn __init__(inout self, state: Int):
        self.state = state
        print("initializing", state)

    fn __copyinit__(inout self, existing: Self):
        self.state = existing.state
        print("copying", self.state)

    fn __moveinit__(inout self, owned existing: Self):
        self.state = existing.state
        print("moving", self.state)

    fn __del__(owned self):
        print("destroying", self.state)

    fn __str__(self) -> String:
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        writer.write("talkative ", self.state)


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


fn test_non_trivial_reg_varargs():
    # CHECK: -- test_non_trivial_reg_varargs
    print("\n-- test_non_trivial_reg_varargs")

    fn callee(*args: TalkativeCopableReg):
        var arg: TalkativeCopableReg

        if len(args) == 1:
            arg = args[0]
        else:
            arg = args[1]
        print("here")
        print(arg)

    # CHECK-NEXT: initializing 1
    # CHECK-NEXT: copying 1
    # CHECK-NEXT: here
    # CHECK-NEXT: talkative 1
    # CHECK-NEXT: destroying 1
    # CHECK-NEXT: destroying 1
    callee(TalkativeCopableReg(1))

    # CHECK-NEXT: initializing 2
    # CHECK-NEXT: initializing 3
    # CHECK-NEXT: copying 3
    # CHECK-NEXT: here
    # CHECK-NEXT: talkative 3
    # CHECK-NEXT: destroying 3
    # CHECK-NEXT: destroying 3
    # CHECK-NEXT: destroying 2
    callee(TalkativeCopableReg(2), TalkativeCopableReg(3))


# ===----------------------------------------------------------------------=== #
# owned variadic packs
# ===----------------------------------------------------------------------=== #


fn owned_variadic_pack[*Ts: Formattable](owned *pack: *Ts):
    print("-- testing owned variadic pack with", len(pack), "elements")

    # TODO: Test mutation of the value not just reading of the value.
    @parameter
    fn process[T: Formattable](a: T):
        print("hello", a)

    pack.each[process]()

    print("owned_variadic_pack done")


fn test_owned_variadic_pack():
    # CHECK: -- testing owned variadic pack with 0 elements
    owned_variadic_pack()
    # CHECK-NEXT: owned_variadic_pack done
    # CHECK-NEXT: done zero
    print("done zero")

    # CHECK-NEXT: -- testing owned variadic pack with 2 elements
    # CHECK-NEXT: hello foo
    # CHECK-NEXT: hello 42
    # CHECK-NEXT: owned_variadic_pack done
    owned_variadic_pack("foo", 42)
    # CHECK-NEXT: done two
    print("done two")

    # CHECK-NEXT: initializing 1
    # CHECK-NEXT: initializing 2
    # CHECK-NEXT: initializing 3
    # CHECK-NEXT: -- testing owned variadic pack with 3 elements
    # CHECK-NEXT: hello talkative 1
    # CHECK-NEXT: hello talkative 2
    # CHECK-NEXT: hello talkative 3
    # CHECK-NEXT: destroying 3
    # CHECK-NEXT: destroying 2
    # CHECK-NEXT: destroying 1
    # CHECK-NEXT: owned_variadic_pack done
    owned_variadic_pack(TalkativeMem(1), TalkativeMem(2), TalkativeMem(3))

    # CHECK-NEXT: done three
    print("done three")


fn inout_variadic_pack[*Ts: Formattable](inout *pack: *Ts):
    print("-- testing inout variadic pack with", len(pack), "elements")

    # TODO: Test mutation of the value not just reading of the value.
    @parameter
    fn process[T: Formattable](a: T):
        print("hello", a)

    pack.each[process]()


fn test_inout_variadic_pack():
    # CHECK: -- testing inout variadic pack with 2 elements
    # CHECK: hello foo
    # CHECK: hello 42
    var string = "foo"
    var integer = 42
    inout_variadic_pack(string, integer)
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
    inout_variadic_pack(m1, m2)
    print("after call")
    _ = m2^
    print("")


fn borrowed_variadic_pack[*Ts: Formattable](*pack: *Ts):
    print("-- testing borrowed variadic pack with", len(pack), "elements")

    @parameter
    fn process[T: Formattable](a: T):
        print("hello", a)

    pack.each[process]()


fn test_borrowed_variadic_pack():
    # CHECK: -- testing borrowed variadic pack with 2 elements
    # CHECK: hello foo
    # CHECK: hello 42
    borrowed_variadic_pack("foo", 42)
    print("")

    # CHECK: initializing 2
    # CHECK: initializing 1
    # CHECK: -- testing borrowed variadic pack with 2 elements
    # CHECK: hello talkative 1
    # CHECK: hello talkative 2
    # CHECK: destroying 1
    # CHECK: after call
    # CHECK: destroying 2
    var m2 = TalkativeMem(2)
    borrowed_variadic_pack(TalkativeMem(1), m2)
    print("after call")
    _ = m2^
    print("")


# Sum all the specified values together.
fn sum_intable[*Ts: Intable](*pack: *Ts) -> Int:
    var result = 0

    @parameter
    fn process[T: Intable](a: T):
        result += int(a)

    pack.each[process]()
    return result


# Check to see if we can do packs at comptime.
fn test_comptime_pack():
    # CHECK-LABEL: test_comptime_pack
    print("test_comptime_pack")

    var str1 = sum_intable(4, 5.0, 7)
    print(str1)
    # CHECK: 16

    alias str2 = sum_intable(4, 5.0, 7)
    print(str2)
    # CHECK: 16


fn use_value[T: AnyType](value: T):
    pass


fn test_tuple():
    # CHECK-LABEL: -- test_tuple
    print("-- test_tuple")

    # TODO: The initializer for tuple is copying+destroying the elements
    # unnecessarily.

    # CHECK-NEXT: initializing 1
    # CHECK-NEXT: initializing 2
    # CHECK-NEXT: moving 1
    # CHECK-NEXT: moving 2
    var t1 = TalkativeCopableMovableMem(1), TalkativeCopableMovableMem(2)

    # CHECK-NEXT: p1: talkative 2 before copy
    print("p1:", t1[1], "before copy")

    # TODO(MSTDL-487):
    #   Restore implicit copying to Tuple.
    #   Don't forget to uncomment block ~15 lines down too!
    # # COM: CHECK-NEXT: copying 1
    # # COM: CHECK-NEXT: copying 2
    # var t2 = t1

    # # COM: CHECK-NEXT: p2: talkative 1 before transfer
    # print("p2:", t2[0], "before transfer")

    # CHECK-NEXT: moving 1
    # CHECK-NEXT: moving 2
    var t3 = t1^

    # # COM: CHECK-NEXT: before use t2
    # print("before use t2")

    # use_value(t2)
    # # COM: CHECK-NEXT: destroying 1
    # # COM: CHECK-NEXT: destroying 2

    # CHECK-NEXT: before use t3
    print("before use t3")
    use_value(t3)
    # CHECK-NEXT: destroying 1
    # CHECK-NEXT: destroying 2

    # CHECK-NEXT: test_tuple done!
    print("test_tuple done!")


fn main():
    test_inout_varargs()
    test_owned_varargs()
    test_owned_reg_varargs()
    test_non_trivial_reg_varargs()
    test_owned_variadic_pack()
    test_inout_variadic_pack()
    test_borrowed_variadic_pack()
    test_comptime_pack()
    test_tuple()
