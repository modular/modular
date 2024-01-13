# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics

struct Empty:
    fn __init__(inout self):
        pass

    fn __copyinit__(inout self, existing: Self):
        pass


struct MemExample:
    var x: Int
    var y: Int

    # expected-error @+1 {{'self.x' is uninitialized at the implicit return from this function}}
    fn __init__(inout self): # expected-note {{'self' declared here}}
        pass

    # expected-error @+1 {{'self.y' is uninitialized at the implicit return from this function}}
    fn __copyinit__(inout self, existing: Self): # expected-note {{'self' declared here}}
        self.x = existing.x

    fn noop(self): pass
    fn consume(owned self): pass
    fn __del__(owned self): pass

fn use(x: MemExample): pass
fn use_inout(inout x: MemExample): pass


@register_passable
struct RegExample:
    var regstate: Int

    fn __init__() -> Self:
        return RegExample {regstate: 1}

    fn __copyinit__(self) -> Self:
        return RegExample {regstate: 12}

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
    var a = RegExample()  # expected-warning {{never mutated}}
    return a  # OK!


fn use_of_init2() -> MemExample:
    var a = MemExample()  # expected-warning {{never mutated}}
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

fn invalid_mutation():
  let x = MemExample() # expected-note {{'x' declared here}}
  use_inout(x) # expected-error {{invalid mutation of immutable value 'x'}}


# Issue #12859, bad location info
fn take_int(a: Int):
    pass


fn uninit_lvalue_int():
    var x: Int  # expected-note {{'x' declared here}}
    take_int(x)  # expected-error {{use of uninitialized value 'x'}}


# Return-specific errors.
fn return_error1(inout a: MemExample): # expected-note {{'a' declared here}}
    _ = a ^
    return  # expected-error {{'a' is uninitialized at return from this function}}


# expected-error @+1 {{'a' is uninitialized at the implicit return from this function}}
fn return_error2(inout a: RegExample): # expected-note {{'a' declared here}}
    _ = a ^


##===----------------------------------------------------------------------===##
# Control flow
##===----------------------------------------------------------------------===##


fn use_of_uninit_if(cond: Bool):
    var a: MemExample  # expected-warning {{never mutated}}
    if cond:
        a = MemExample()
    else:
        a = MemExample()
    use(a)  # Ok

    var b: MemExample  # expected-note {{'b' declared here}}
    if cond:
        b = MemExample()
    use(b)  # expected-error {{use of uninitialized value 'b'}}

    var c: MemExample  # expected-warning {{never mutated}}
    if True:
        c = MemExample()
    use(c)  # Ok.


fn use_of_uninit_while(cond: Bool):
    var a: MemExample  # expected-warning {{never mutated}}
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
    var a: MemExample  # expected-warning {{'a' was declared as a 'var' but never mutated, consider switching to a 'let'}}
    try:
        raise err
    except:
        a = MemExample()
        pass
    use(a)  # ok

    var b: MemExample  # expected-warning {{'b' was declared as a 'var' but never mutated, consider switching to a 'let'}}
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

    var d: MemExample  # expected-warning {{'d' was declared as a 'var' but never mutated, consider switching to a 'let'}}
    try:
        if cond:
            raise err
    except:
        d = MemExample()
    else:
        d = MemExample()
    use(d)  # Ok


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

    fn __init__() -> Self:
        return Self {reg1: RegExample(), reg2: RegExample()}

    fn __copyinit__(existing: Self) -> Self:
        return Self {reg1: existing.reg1, reg2: existing.reg2}


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
    __mlir_op.`lit.ownership.mark_destroyed`[_type=None](
        __get_ref_from_value(x.mem)
    )


# expected-error @+3 {{field 'x.mem' destroyed out of the middle of a value, preventing the overall value from being destroyed}}
fn fieldConsumeError(
    owned w: MoreComplexExample,  # expected-note {{'w' declared here}}
    owned x: MoreComplexExample,
    owned y: MoreComplexExample,
    owned z: MoreComplexExample,  # expected-note {{'z' declared here}}
):
    _ = x.mem ^  # Error diagnosed above.

    # This is ok because we replace the mem field.
    _ = y.mem ^
    y.mem = MemExample()

    _ = z.mem ^
    _ = z.mem ^  # expected-error {{use of uninitialized value 'z.mem'}}
    z.mem = MemExample()

    _ = w.mem ^
    use(w)  # expected-error {{use of uninitialized value 'w.mem'}}

    # expected-error @+1 {{field 'twoRegsRP.reg1' destroyed out of the middle of a value, preventing the overall value from being destroyed}}
    var twoRegsRP = TwoRegsRP()
    _ = twoRegsRP.reg1 ^

    var single1 = MemExample()  # expected-note {{'single1' declared here}}
    _ = single1 ^
    _ = single1  # expected-error {{use of uninitialized value 'single1'}}


# https://github.com/modularml/modular/issues/15404
struct SimpleStructNoDtor:
    fn __init__(inout self):
        pass


fn consume(owned x: SimpleStructNoDtor):
    pass


fn issue15404():
    let c = SimpleStructNoDtor()  # expected-note {{'c' declared here}}
    consume(c ^)
    consume(c ^)  # expected-error {{use of uninitialized value 'c'}}


##===----------------------------------------------------------------------===##
# var -> let and incorrect let warnings
##===----------------------------------------------------------------------===##


fn use(a: Int):
    pass


fn testVarToLet(cond: Bool):
    var a: Int  # expected-warning {{'a' was declared as a 'var' but never mutated, consider switching to a 'let'}}
    if cond:
        a = 1
    else:
        a = 2
    use(a)

    var b: Int  # expected-warning {{consider switching to a 'let'}}
    if cond:
        b = 1
        use(b)

    var c: Int
    if cond:
        c = 1
    c = 2  # This is correct to be a var.
    use(c)

    let d: TwoRegs
    d = TwoRegs()
    let e = d ^  # Consume from let is fine.


fn invalid_let():
    let twoRegs = TwoRegs()  # expected-note {{'twoRegs' declared here}}
    twoRegs.reg1 = (
        RegExample()  # expected-error {{invalid mutation of immutable value 'twoRegs.reg1'}}
    )

# Consumption of struct works only on definition of __del__
# https://github.com/modularml/mojo/issues/734
struct StructWithNoDel:
    var x: Int
    fn __init__(inout self, a: Int):
        self.x = a
fn take(owned x: StructWithNoDel): pass
fn testStructWithNoDel():
    var l = StructWithNoDel(100)
    take(l^)  # expected-error {{value 'l' cannot be consumed, because 'l.x' is used later}}
    l.x = 10


# expected-note @+1 {{'x' declared here}}
fn inout_restored_at_throw(inout x: MemExample, err: Error) raises:
   # x is uninit after this point, needs to be restored if an
   # error is thrown.
   _ = x^
   raise err # expected-error {{use of uninitialized value 'x'}}

# Invalid error field 'w.x.y' destroyed out of the middle of a value, preventing the overall value from being destroyed
# https://github.com/modularml/mojo/issues/1535
@value
struct NestedInt:
    var y: Int
@value
struct WrapperNestedInt:
    var x: NestedInt
fn testWrapperNestedInt():
    var w = WrapperNestedInt(NestedInt(0))
    for i in range(0, 1):
        w.x.y = 0


# ===----------------------------------------------------------------------=== #
# More complex references
# ===----------------------------------------------------------------------=== #

fn testConditional(cond: __mlir_type.i1):
  let a = MemExample()
  let b : MemExample # expected-note {{'b' declared here}}

  let aref = __get_ref_from_value(a)
  let bref = __get_ref_from_value(b)
  let cref = aref if cond else bref

  # expected-error @+1 {{use of uninitialized value 'b'}}
  __get_value_from_ref(cref).noop()

  # expected-error @+1 {{cannot consume indirect references to values}}
  __get_value_from_ref(cref)^.consume()
