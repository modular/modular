# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics


struct Empty:
    fn __init__(inout self):
        pass

    fn __copyinit__(inout self, existing: Self):
        pass


struct MemExample:
    var x: Int
    var y: Int

    # expected-error @+1 {{'self.x' is uninitialized at the implicit return from this function}}
    fn __init__(inout self):  # expected-note {{'self' declared here}}
        pass

    # expected-error @below {{'self.y' is uninitialized at the implicit return from this function}}
    fn __copyinit__(
        inout self,  # expected-note {{'self' declared here}}
        existing: Self,
    ):
        self.x = existing.x

    fn noop(self):
        pass

    fn consume(owned self):
        pass

    fn __del__(owned self):
        pass


fn use(x: MemExample):
    pass


fn use_inout(inout x: MemExample):
    pass


@register_passable
struct RegExample:
    var regstate: Int

    fn __init__(inout self):
        self.regstate = 1

    fn __copyinit__(inout self, existing: Self):
        self.regstate = 12

    fn __del__(owned self):
        pass


fn use(x: RegExample):
    pass


##===----------------------------------------------------------------------===##
# Simple cases.
##===----------------------------------------------------------------------===##


fn use_of_empty() -> Empty:
    var a: Empty  # expected-note {{'a' declared here}}
    return a  # expected-error {{use of uninitialized value 'a'}}


fn use_of_init() -> RegExample:
    var a = RegExample()
    return a  # OK!


fn use_of_init2() -> MemExample:
    var a = MemExample()
    return a  # OK!


fn use_of_uninit() -> RegExample:
    var a: RegExample  # expected-note {{'a' declared here}}
    return a  # expected-error {{use of uninitialized value 'a'}}


fn use_of_uninit2() -> MemExample:
    var a: MemExample  # expected-note {{'a' declared here}}
    return a  # expected-error {{use of uninitialized value 'a'}}


fn invalid_partial_init() -> MemExample:
    var a: MemExample  # expected-note {{'a' declared here}}
    a.x = 42
    a.y = 42
    return a  # expected-error {{'a' used with all fields manually initialized but without calling an '__init__' method}}


fn field_sensitive():
    var a: MemExample  # expected-note {{'a' declared here}}
    a.x = 1
    use(a)  # expected-error {{use of uninitialized value 'a.y'}}

    var b = MemExample()
    b.x = 1
    b.y = 2
    use(b)  # Ok


# Issue #12859, bad location info
fn take_int(a: Int):
    pass


fn uninit_lvalue_int():
    var x: Int  # expected-note {{'x' declared here}}
    take_int(x)  # expected-error {{use of uninitialized value 'x'}}


# Return-specific errors.
fn return_error1(inout a: MemExample):  # expected-note {{'a' declared here}}
    _ = a^
    return  # expected-error {{'a' is uninitialized at return from this function}}


# expected-error @+1 {{'a' is uninitialized at the implicit return from this function}}
fn return_error2(inout a: RegExample):  # expected-note {{'a' declared here}}
    _ = a^


##===----------------------------------------------------------------------===##
# Control flow
##===----------------------------------------------------------------------===##


fn use_of_uninit_if(cond: Bool):
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


fn use_of_uninit_while(cond: Bool):
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


fn use_of_uninit_raise(cond: Bool, err: Error):
    var a: MemExample
    try:
        raise err
    except:
        a = MemExample()
        pass
    use(a)  # ok

    var b: MemExample
    try:
        b = MemExample()
        raise err
    except:
        pass
    use(b)  # ok

    var c: MemExample  # expected-note {{'c' declared here}}
    try:
        if cond:
            c = MemExample()
            raise err
        else:
            raise err
    except:
        pass
    use(c)  # expected-error {{use of uninitialized value 'c'}}

    var d: MemExample
    try:
        if cond:
            raise err
    except:
        d = MemExample()
    else:
        d = MemExample()
    use(d)  # Ok


fn may_raise() raises -> MemExample:
    return MemExample()


fn reassign_might_raise():
    var value = MemExample()  # expected-note {{'value' declared here}}
    try:
        # 'value' is passed directly as the MLValue slot to the raising call,
        # meaning the current value has to be destroyed before the call.
        value = may_raise()
    except:
        # If the call raises, then the value is known to be uninitialized.
        _ = value  # expected-error {{use of uninitialized value 'value'}}


@__named_result(out)
# expected-note @below {{'out' declared here}}
fn uninitialized_result(c: Bool) -> MemExample:
    if c:
        # expected-error @below {{'out' is uninitialized at return from this function}}
        return
    out = MemExample()


##===----------------------------------------------------------------------===##
# Complex aggregates
##===----------------------------------------------------------------------===##


struct TwoRegs:
    var reg1: RegExample
    var reg2: RegExample

    fn __init__(inout self):
        self.reg1 = RegExample()
        self.reg2 = RegExample()

    fn __copyinit__(inout self, existing: Self):
        self.reg1 = existing.reg1
        self.reg2 = existing.reg2


@register_passable
struct TwoRegsRP:
    var reg1: RegExample
    var reg2: RegExample

    fn __init__(inout self):
        self.reg1 = RegExample()
        self.reg2 = RegExample()

    fn __copyinit__(inout self, existing: Self):
        self.reg1 = existing.reg1
        self.reg2 = existing.reg2


struct MoreComplexExample:
    var mem: MemExample
    var reg: TwoRegs

    fn __init__(inout self):
        var result: MoreComplexExample  # expected-note {{'result' declared here}}
        result.mem = MemExample()
        result.reg.reg2 = RegExample()
        self = result  # expected-error {{use of uninitialized value 'result.reg.reg1'}}

    fn __copyinit__(inout self, existing: Self):
        self.mem = existing.mem
        self.reg = existing.reg

    fn __del__(owned self):
        pass


fn use(x: MoreComplexExample):
    pass


fn testClosure(a: Bool):
    if a:
        return

    @always_inline
    fn thing() -> MemExample:
        var x: MemExample  # expected-note {{'x' declared here}}
        return x  # expected-error {{use of uninitialized value 'x'}}

    _ = thing()


# expected-error @+1 {{field 'x.mem' destroyed out of the middle of a value, preventing the overall value from being destroyed}}
fn disableDtor(owned x: MoreComplexExample):
    _ = x.mem^


fn badMarkDestroyed(owned x: MoreComplexExample):
    # expected-error @+1 {{cannot mark subobjects destroyed}}
    __mlir_op.`lit.ownership.mark_destroyed`[_type=None](
        __get_mvalue_as_litref(x.mem)
    )


# expected-error @+3 {{field 'x.mem' destroyed out of the middle of a value, preventing the overall value from being destroyed}}
fn fieldConsumeError(
    owned w: MoreComplexExample,  # expected-note {{'w' declared here}}
    owned x: MoreComplexExample,
    owned y: MoreComplexExample,
    owned z: MoreComplexExample,  # expected-note {{'z' declared here}}
):
    _ = x.mem^  # Error diagnosed above.

    # This is ok because we replace the mem field.
    _ = y.mem^
    y.mem = MemExample()

    _ = z.mem^
    _ = z.mem^  # expected-error {{use of uninitialized value 'z.mem'}}
    z.mem = MemExample()

    _ = w.mem^
    use(w)  # expected-error {{use of uninitialized value 'w.mem'}}

    # expected-error @+1 {{field 'twoRegsRP.reg1' destroyed out of the middle of a value, preventing the overall value from being destroyed}}
    var twoRegsRP = TwoRegsRP()
    _ = twoRegsRP.reg1^

    var single1 = MemExample()  # expected-note {{'single1' declared here}}
    _ = single1^
    _ = single1  # expected-error {{use of uninitialized value 'single1'}}


# https://github.com/modularml/modular/issues/15404
struct SimpleStructNoDtor:
    fn __init__(inout self):
        pass


fn consume(owned x: SimpleStructNoDtor):
    pass


fn issue15404():
    var c = SimpleStructNoDtor()  # expected-note {{'c' declared here}}
    consume(c^)
    consume(c^)  # expected-error {{use of uninitialized value 'c'}}


##===----------------------------------------------------------------------===##
# incorrect warnings
##===----------------------------------------------------------------------===##


# Consumption of struct works only on definition of __del__
# https://github.com/modularml/mojo/issues/734
struct StructWithNoDel:
    var x: Int

    fn __init__(inout self, a: Int):
        self.x = a


fn take(owned x: StructWithNoDel):
    pass


fn testStructWithNoDel():
    var l = StructWithNoDel(100)
    # expected-error @below {{value 'l' cannot be consumed, because 'l.x' is used later}}
    take(l^)
    l.x = 10


# expected-note @+1 {{'x' declared here}}
fn inout_restored_at_throw(inout x: MemExample, err: Error) raises:
    # x is uninit after this point, needs to be restored if an
    # error is thrown.
    _ = x^
    raise err  # expected-error {{use of uninitialized value 'x'}}


# Invalid error field 'w.x.y' destroyed out of the middle of a value, preventing the overall value from being destroyed
# https://github.com/modularml/mojo/issues/1535
@value
struct NestedInt:
    var y: Int


@value
struct WrapperNestedInt:
    var x: NestedInt


@value
@register_passable("trivial")
struct TrivialRange:
    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) -> Int:
        return 1

    fn __len__(self) -> Int:
        return 1


fn testWrapperNestedInt():
    var w = WrapperNestedInt(NestedInt(0))
    for i in TrivialRange():
        w.x.y = 0


# ===----------------------------------------------------------------------=== #
# More complex references
# ===----------------------------------------------------------------------=== #


fn testConditionalImmut(cond: __mlir_type.i1):
    var a = MemExample()
    var b: MemExample  # expected-note {{'b' declared here}}

    var aref = Reference(a).value
    # expected-error @+1 {{use of uninitialized value 'b'}}
    var bref = Reference(b).value
    var cref = aref if cond else bref

    Reference(__get_litref_as_mvalue(cref))[].noop()


fn testConditionalMut(cond: __mlir_type.i1):
    var a = MemExample()
    var b: MemExample  # expected-note {{'b' declared here}}

    # expected-error @+1 {{use of uninitialized value 'b'}}
    var cref = Reference(a).value if cond else Reference(b).value

    Reference(__get_litref_as_mvalue(cref))[] = MemExample()


# CheckLifetimes cannot call MemExample.__del__ because 'self' is in the default
# address space.
fn bad_addr_space[
    addr_space: AddressSpace
](ptr: UnsafePointer[MemExample, addr_space]):
    # expected-error @+1 {{cannot destroy value in non-default address space}}
    _ = __get_address_as_owned_value(ptr.address)


# Returning a reference to the caller's stack.
# https://github.com/modularml/modular/issues/38421
# This is valid to declare...
fn return_owned_arg_ref(owned x: String) -> Reference[String, __lifetime_of(x)]:
    return x


fn test38421():
    # this is getting a reference to the expression temporary for the string.
    # expected-note @+1 {{'(expression temporary)' declared here}}
    var reference = return_owned_arg_ref(String("abc"))

    # This is an error since the rvalue temp slot is uninitialized here.
    # expected-error @+1 {{potential indirect access to uninitialized value '(expression temporary)'}}
    _ = reference[].__len__()
