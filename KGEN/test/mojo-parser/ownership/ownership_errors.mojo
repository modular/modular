# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics


struct Empty(ImplicitlyCopyable):
    def __init__(out self):
        pass


struct MemExample(ImplicitlyCopyable):
    var x: Int
    var y: Int

    # expected-error @+1 {{'self.x' is uninitialized at the implicit return from this function}}
    def __init__(out self):  # expected-note {{'self' declared here}}
        pass

    # expected-error @+1 {{'self.y' is uninitialized at the implicit return from this function}}
    def __init__(
        out self, *, copy: Self  # expected-note {{'self' declared here}}
    ):
        self.x = copy.x

    def noop(self):
        pass

    def consume(var self):
        pass

    def __del__(deinit self):
        pass


def use(x: MemExample):
    pass


def use_inout(mut x: MemExample):
    pass


struct RegExample(ImplicitlyCopyable, RegisterPassable):
    var regstate: Int

    def __init__(out self):
        self.regstate = 1

    def __init__(out self, *, copy: Self):
        self.regstate = 12

    def __del__(deinit self):
        pass

    def consume(var self):
        pass


def use(x: RegExample):
    pass


##===----------------------------------------------------------------------===##
# Simple cases.
##===----------------------------------------------------------------------===##


def use_of_empty() -> Empty:
    var a: Empty  # expected-note {{'a' declared here}}
    return a  # expected-error {{use of uninitialized value 'a'}}


def use_of_init() -> RegExample:
    var a = RegExample()
    return a  # OK!


def use_of_init2() -> MemExample:
    var a = MemExample()
    return a  # OK!


def use_of_uninit() -> RegExample:
    var a: RegExample  # expected-note {{'a' declared here}}
    return a  # expected-error {{use of uninitialized value 'a'}}


def use_of_uninit2() -> MemExample:
    var a: MemExample  # expected-note {{'a' declared here}}
    return a  # expected-error {{use of uninitialized value 'a'}}


def invalid_partial_init() -> MemExample:
    var a: MemExample  # expected-note {{'a' declared here}}
    a.x = 42
    a.y = 42
    return a  # expected-error {{'a' used with all fields manually initialized but without calling an '__init__' method}}


def field_sensitive():
    var a: MemExample  # expected-note {{'a' declared here}}
    a.x = 1
    use(a)  # expected-error {{use of uninitialized value 'a.y'}}

    var b = MemExample()
    b.x = 1
    b.y = 2
    use(b)  # Ok


# Issue #12859, bad location info
def take_int(a: Int):
    pass


def uninit_lvalue_int():
    var x: Int  # expected-note {{'x' declared here}}
    take_int(x)  # expected-error {{use of uninitialized value 'x'}}


# Return-specific errors.
def return_error1(mut a: MemExample):  # expected-note {{'a' declared here}}
    a^.consume()
    return  # expected-error {{'a' is uninitialized at return from this function}}


# expected-error @+1 {{'a' is uninitialized at the implicit return from this function}}
def return_error2(mut a: RegExample):  # expected-note {{'a' declared here}}
    a^.consume()


##===----------------------------------------------------------------------===##
# Control flow
##===----------------------------------------------------------------------===##


def use_of_uninit_if(cond: Bool):
    var a: MemExample
    if cond:
        a = MemExample()
    else:
        a = MemExample()
    use(a)  # Ok

    var b: MemExample  # expected-note {{'b' declared here}}
    if cond:
        b = MemExample()
    use(b)  # expected-error {{use of uninitialized value 'b'}}

    var c: MemExample
    if True:
        c = MemExample()
    use(c)  # Ok.


def use_of_uninit_while(cond: Bool):
    var a: MemExample
    if cond:
        # Infinite loop never returns
        while True:
            pass
    else:
        a = MemExample()
    use(a)  # Ok

    var b: RegExample  # expected-note {{'b' declared here}}
    while cond:
        b = RegExample()

    use(b)  # expected-error {{use of uninitialized value 'b'}}


def use_of_uninit_raise(cond: Bool):
    var a: MemExample
    try:
        raise Error()
    except:
        a = MemExample()
        pass
    use(a)  # ok

    var b: MemExample
    try:
        b = MemExample()
        raise Error()
    except:
        pass
    use(b)  # ok

    var c: MemExample  # expected-note {{'c' declared here}}
    try:
        if cond:
            c = MemExample()
            raise Error()
        else:
            raise Error()
    except:
        pass
    use(c)  # expected-error {{use of uninitialized value 'c'}}

    var d: MemExample
    try:
        if cond:
            raise Error()
    except:
        d = MemExample()
    else:
        d = MemExample()
    use(d)  # Ok


def may_raise() raises -> MemExample:
    return MemExample()


def reassign_might_raise():
    var value = MemExample()
    try:
        # 'value' is passed directly as the MLValue slot to the raising call,
        # meaning the current value has to be destroyed before the call.
        value = may_raise()
        _ = value
    except:
        # If the call raises, then the value is known to be uninitialized.
        _ = value


# expected-note @below {{'out' declared here}}
def uninitialized_result(c: Bool, out out: MemExample):
    if c:
        # expected-error @below {{'out' is uninitialized at return from this function}}
        return
    out = MemExample()


def test_unreachable_after_abort():
    abort()
    # expected-warning @+1 {{unreachable code after function that never returns}}
    var x = 4 + 5


@no_inline
def throwonly() raises -> Never:
    abort()


def test_unreachable_after_throwonly() raises:
    throwonly()
    # expected-warning @+1 {{unreachable code after function that never returns}}
    var x = 4 + 5


def test_unreachable_after_comptime_assert_false():
    comptime assert False
    # expected-warning @+1 {{unreachable code after compile-time assertion failure}}
    return


##===----------------------------------------------------------------------===##
# Complex aggregates
##===----------------------------------------------------------------------===##


struct TwoRegs(ImplicitlyCopyable):
    var reg1: RegExample
    var reg2: RegExample

    def __init__(out self):
        self.reg1 = RegExample()
        self.reg2 = RegExample()


struct TwoRegsRP(Copyable, RegisterPassable):
    var reg1: RegExample
    var reg2: RegExample

    def __init__(out self):
        self.reg1 = RegExample()
        self.reg2 = RegExample()


struct MoreComplexExample(ImplicitlyCopyable):
    var mem: MemExample
    var reg: TwoRegs

    def __init__(out self):
        var result: MoreComplexExample  # expected-note {{'result' declared here}}
        result.mem = MemExample()
        result.reg.reg2 = RegExample()
        self = result  # expected-error {{use of uninitialized value 'result.reg.reg1'}}

    def __del__(deinit self):
        pass


def use(x: MoreComplexExample):
    pass


def testClosure(a: Bool):
    if a:
        return

    @always_inline
    def thing() capturing -> MemExample:
        var x: MemExample  # expected-note {{'x' declared here}}
        return x  # expected-error {{use of uninitialized value 'x'}}

    _ = thing()


# expected-error @+1 {{field 'x.mem' destroyed out of the middle of a value, preventing the overall value from being destroyed}}
def disableDtor(var x: MoreComplexExample):
    x.mem^.consume()


def fieldConsumeError(
    var w: MoreComplexExample,  # expected-note {{'w' declared here}}
    # expected-error @+1 {{field 'x.mem' destroyed out of the middle of a value, preventing the overall value from being destroyed}}
    var x: MoreComplexExample,
    var y: MoreComplexExample,
    var z: MoreComplexExample,  # expected-note {{'z' declared here}}
):
    x.mem^.consume()

    # This is ok because we replace the mem field.
    y.mem^.consume()

    y.mem = MemExample()

    z.mem^.consume()
    z.mem^.consume()  # expected-error {{use of uninitialized value 'z.mem'}}
    z.mem = MemExample()

    w.mem^.consume()
    use(w)  # expected-error {{use of uninitialized value 'w.mem'}}

    # expected-warning @+2 {{assignment to 'twoRegsRP' was never used; assign to '_' instead?}}
    # expected-error @+1 {{field 'twoRegsRP.reg1' destroyed out of the middle of a value, preventing the overall value from being destroyed}}
    var twoRegsRP = TwoRegsRP()
    twoRegsRP.reg1^.consume()

    var single1 = MemExample()  # expected-note {{'single1' declared here}}
    single1^.consume()
    _ = single1  # expected-error {{use of uninitialized value 'single1'}}


# https://github.com/modularml/modular/issues/15404
struct SimpleStructNoDtor:
    def __init__(out self):
        pass


def consume(var x: SimpleStructNoDtor):
    pass


def issue15404():
    var c = SimpleStructNoDtor()  # expected-note {{'c' declared here}}
    consume(c^)
    consume(c^)  # expected-error {{use of uninitialized value 'c'}}


@fieldwise_init
struct SP[n: Int]:
    pass


def test_no_unused_warning() -> Int:
    # expected-warning @+1 {{assignment to 's' was never used; assign to '_' instead?}}
    s = SP[2]()
    # This is syntactically a use, but n is a parameter, so the warning does show up.
    return s.n


struct TestUnused[T: RPTTrait](TrivialRegisterPassable):
    var thing: Self.T

    def __init__(out self, xyz: Self):
        var other = xyz  # should not warn about dead store.
        self.thing = other.thing
        _ = self.thing


trait RPTTrait(TrivialRegisterPassable):
    pass


##===----------------------------------------------------------------------===##
# incorrect warnings
##===----------------------------------------------------------------------===##


# Consumption of struct works only on definition of __del__
# https://github.com/modular/mojo/issues/734
struct StructWithNoDel:
    var x: Int

    @implicit
    def __init__(out self, a: Int):
        self.x = a


def take(var x: StructWithNoDel):
    pass


def testStructWithNoDel():
    var l = StructWithNoDel(100)
    # expected-error @below {{value 'l' cannot be consumed, because 'l.x' is used later}}
    take(l^)
    l.x = 10


# expected-note @+1 {{'x' declared here}}
def inout_restored_at_throw(mut x: MemExample) raises:
    # x is uninit after this point, needs to be restored if an
    # error is thrown.
    x^.consume()
    raise Error()  # expected-error {{use of uninitialized value 'x'}}


# Invalid error field 'w.x.y' destroyed out of the middle of a value, preventing the overall value from being destroyed
# https://github.com/modular/mojo/issues/1535
@fieldwise_init
struct NestedInt(ImplicitlyCopyable):
    var y: Int


@fieldwise_init
struct WrapperNestedInt:
    var x: NestedInt


@fieldwise_init
struct TrivialRange(Iterator, TrivialRegisterPassable):
    comptime Element = Int

    def __iter__(self) -> Self:
        return self

    def __next__(mut self) raises StopIteration -> Int:
        return 1

    def __len__(self) -> Int:
        return 1


def testWrapperNestedInt():
    var w = WrapperNestedInt(NestedInt(0))
    for _ in TrivialRange():
        w.x.y = 0


# ===----------------------------------------------------------------------=== #
# More complex references
# ===----------------------------------------------------------------------=== #


def testConditionalImmut(cond: __mlir_type.i1):
    var a = MemExample()
    var b: MemExample  # expected-note {{'b' declared here}}

    var aptr = Pointer(to=a)
    # expected-error @+1 {{use of uninitialized value 'b'}}
    var bptr = Pointer(to=b)
    var cptr = aptr if cond else bptr

    cptr[].noop()


def testConditionalMut(cond: __mlir_type.i1):
    var a = MemExample()
    var b: MemExample  # expected-note {{'b' declared here}}

    # expected-error @+1 {{use of uninitialized value 'b'}}
    var cptr = Pointer(to=a) if cond else Pointer(to=b)
    cptr[] = MemExample()


# CheckLifetimes cannot call MemExample.__del__ because 'self' is in the default
# address space.
def bad_addr_space[
    addr_space: AddressSpace
](ptr: UnsafePointer[MemExample, address_space=addr_space, ...]):
    # expected-error @+1 {{cannot destroy value in non-default address space}}
    _ = __get_address_as_owned_value(ptr.address)


# Returning a reference to the caller's stack.
# https://github.com/modularml/modular/issues/38421
# This is valid to declare...
def return_owned_arg_ref(var x: String) -> Pointer[String, origin_of(x)]:
    return Pointer(to=x)


def test38421():
    # this is getting a reference to the expression temporary for the string.
    # expected-note @+1 {{'(expression temporary)' declared here}}
    var reference = return_owned_arg_ref("abc")

    # This is an error since the rvalue temp slot is uninitialized here.
    # expected-error @+1 {{potential indirect access to uninitialized value '(expression temporary)'}}
    _ = reference[].__len__()


@fieldwise_init
struct MovableStuff(Movable):
    pass


def test_cannot_consume_indirect_references():
    # expected-warning @+1 {{assignment to 'a' was never used}}
    var a = MovableStuff()
    # expected-warning @+1 {{assignment to 'b' was never used}}
    var b = MovableStuff()

    @parameter
    def callback():
        # expected-error @+1 {{cannot consume indirect references to values}}
        b = a^


# ===----------------------------------------------------------------------=== #
# Computed LValues
# ===----------------------------------------------------------------------=== #


def get_inout_ref(mut x: String) -> ref[x] String:
    return x


struct StrArray:
    def __getitem__(self, x: Int) -> String:
        return String()

    def __setitem__(mut self, x: Int, var value: String):
        pass


def test_inout_ref(mut v: StrArray, i: Int):
    # expected-note @below {{'(expression temporary)' declared here}}
    # expected-error @below {{use of uninitialized value '(expression temporary)'}}
    var r = Pointer(to=get_inout_ref(v[i]))

    _ = r[]


def test_uninit_store_trivial():
    var example = TrivialAggregate()
    example.a = 1
    # expected-warning @+1 {{assignment to 'example.b' was never used}}
    example.b = 2


def test_owned_warning(var arg: TrivialAggregate):
    # expected-warning @+1 {{assignment to 'arg' was never used}}
    arg = TrivialAggregate()


struct TrivialAggregate(TrivialRegisterPassable):
    var a: Int
    var b: Int

    def __init__(out self):
        self.a = 0
        self.b = 0


def param_for_merge_diagnostic():
    # NOTE: shouldn't produce a "unused store" warning.
    var array_ptr = Int()

    comptime for _ in TrivialRange():
        _ = array_ptr._mlir_value


def raises_ret_int() raises -> Int:
    return 4


def test_trivial_consume():
    var outshape: Int  # expected-note {{'outshape' declared here}}
    try:
        outshape = raises_ret_int()
    except:
        pass

    # expected-error @+1 {{use of uninitialized value 'outshape'}}
    _ = outshape


def test_unused_var(mut mut_arg: Int):
    # expected-warning @+1 {{assignment to 'x' was never used; assign to '_' instead?}}
    var x: Int = 0

    # expected-warning @+1 {{variable 'y' was never used, remove it?}}
    var y: Int

    # expected-warning @+1 {{ref 'z' was never used, remove it?}}
    ref z = mut_arg


@explicit_destroy("Use `consume() method` to finalize")
@fieldwise_init
struct LinearType:
    def consume(deinit self):
        pass

    def use(self):
        pass


def do_something() raises:
    pass


def test_linear_type() raises:
    var tok1 = LinearType()

    # expected-error @+1 {{'tok1' abandoned without being explicitly destroyed: Use `consume() method` to finalize}}
    tok1.use()

    # MOCO-2275: Poor error when raises interacts with @explicit_destroy type
    var tok2 = LinearType()
    # expected-error @below {{'tok2' abandoned without being explicitly destroyed: Use `consume() method` to finalize}}
    # expected-note @below {{value was not consumed when an error is thrown}}
    do_something()

    tok2^.consume()


def test_linear_no_return_complex_lifetime(
    a: LinearType, mut b: LinearType, mut c: LinearType
):
    _ = a
    b^.consume()
    abort()  # Doesn't require 'a' or 'c' to be destroyed.


# Hard case for conditional lifetime analysis on abort.  This should compile.
@explicit_destroy
struct ReducedVariant:
    var _storage: ReducedStorage

    def take[T: Movable](deinit self, cond: Bool) -> T:
        if cond:
            abort()
        return self._storage^.take[T]()


@explicit_destroy
struct ReducedStorage:
    def take[U: Movable](deinit self) -> U:
        abort()


@explicit_destroy("This needs explicit destruction when T isn't linear")
struct ConditionallyLinearType[T: AnyType](
    ImplicitlyDestructible where conforms_to(T, ImplicitlyDestructible)
):
    # var data: Self.T

    def __init__(out self):
        pass

    def use(self):
        pass

    def __del__(deinit self) where conforms_to(Self.T, ImplicitlyDestructible):
        pass  # self.data^.__del__()


def testConditionallyLinearType():
    var c = ConditionallyLinearType[String]()
    c.use()

    var d = ConditionallyLinearType[LinearType]()
    # expected-error @+1 {{'d' abandoned without being explicitly destroyed: This needs explicit destruction when T isn't linear}}
    d.use()


# ===----------------------------------------------------------------------=== #
# Trait-bound fields
# ===----------------------------------------------------------------------=== #


struct Pair[T: Movable & ImplicitlyDestructible](Movable):
    var first: Self.T
    var second: Self.T

    def __init__(out self, var first: Self.T, var second: Self.T):
        self.first = first^
        self.second = second^


def sink[T: AnyType](x: T):
    pass


def test_trait_bound_field():
    # @expected-error @below {{field '(expression temporary).first' destroyed out of the middle of a value, preventing the overall value from being destroyed}}
    var r = Pair[List[Int]](List[Int](), List[Int]()).first^
    # To prevent optimization/warning on unused object
    sink(r)
