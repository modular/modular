# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s
# RUN: %parse-mojo-isolated %s --debug-level full -o /dev/null

# Error Handling related CheckLifetimes tests.


fn use(x: Int):
    pass


fn use(x: String):
    pass


def use_and_raise(x: Int):
    pass


# CHECK-LABEL: lit.struct.decl @RegExample
# CHECK: destructor {{.*}}RegExample::@"__del__
@register_passable
struct RegExample:
    fn __init__(inout self):
        return

    fn __copyinit__(
        inout self, existing: Self
    ):  # CHECK: lit.func @"__copyinit__
        return

    # Test a raising constructor.
    # CHECK-LABEL: lit.func @"__init__{{.*}}(%self: !lit.ref<!RegExample, {{.*}}> init_self, |, %a: {{.*}}, %b: {{.*}}, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error) throws -> i1
    fn __init__(inout self, a: MemExample, b: MemExample) raises:
        # CHECK-NOT: __del__
        # CHECK: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
        # CHECK-NEXT: kgen.return [[FALSE]]
        return

    fn noop(self):
        pass

    fn __del__(owned self):
        pass

    fn mutate(inout self):
        pass


struct MemExample:
    var x: Int

    fn __init__(inout self):
        self.x = 42
        pass

    fn noop(self):
        pass

    fn __moveinit__(inout self, owned existing: Self):
        self.x = existing.x

    fn __copyinit__(inout self, existing: Self):
        self.x = existing.x

    fn __bool__(self) -> Bool:
        return True

    fn __del__(owned self):
        pass


# Use of uninitialized value after call to def function


# CHECK-LABEL: lit.func @"error_handling_int_let
# https://github.com/modularml/modular/issues/25419
def error_handling_int_let():
    # CHECK: lit.var.decl "x"
    var x: Int = 1
    _ = use_and_raise(x)
    use(x)


fn somethingThatRaises() raises:
    pass


# CHECK-LABEL: lit.func @"thing_that_raises
fn thing_that_raises(c: __mlir_type.i1) raises -> MemExample:
    # CHECK-NEXT: [[RESULT:%.*]] = lit.var.decl "anonymous*" synth : !lit.ref<none,
    # CHECK-NEXT: [[IS_ERR:%.*]] = lit.call {{.*}}somethingThatRaises{{.*}}(%__error__, [[RESULT]])
    # CHECK-NEXT: hlcf.if [[IS_ERR]]
    # CHECK-NEXT:   mark_consumed [[RESULT]]
    # CHECK-NEXT:   [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
    # CHECK-NEXT:   lit.error_return [[TRUE]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   mark_consumed %__error__
    # CHECK-NEXT:   yield
    # CHECK-NEXT: }
    somethingThatRaises()

    # CHECK-NEXT:   hlcf.elif {
    # CHECK-NEXT:     hlcf.elif.yield %c
    # CHECK-NEXT:   } then {
    # CHECK-NEXT:      = lit.call {{.*}}__init__
    # CHECK-NOT:       __del__
    # CHECK: kgen.return
    if c:
        return MemExample()
    # CHECK-NEXT:  } else {
    raise Error("TypeError: cannot invert values of this type")


struct RaisingInit:
    var stream: Int

    fn __init__(inout self, flags: Int = 0) raises:
        var stream = 4
        # This can raise, but 'self' doesn't need to be initialized.
        _ = somethingThatRaises()
        self.stream = stream


# CHECK-LABEL: lit.func @"finally_may_raise
fn finally_may_raise() raises:
    # CHECK: lit.try
    try:
        # CHECK-NEXT: call {{.*}}__init__{{.*}}(%__try_error__)
        # CHECK-NEXT: lit.try.raise
        raise Error()
        # CHECK-NEXT: except
        # CHECK-NEXT: [[MOVE:%.*]] = lit.load.consume %__try_error__
        # CHECK-NEXT: lit.ref.store [[MOVE]], %__error__
        # CHECK-NEXT: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
        # CHECK-NEXT: %__finally_error__ = lit.var.decl
        # CHECK-NEXT: lit.try
        # CHECK-NEXT:   [[RESULT:%.*]] = lit.var.decl
        # CHECK-NEXT:   [[IS_ERR:%.*]] = lit.call {{.*}}somethingThatRaises{{.*}}(%__finally_error__, [[RESULT]])
        # CHECK-NEXT:   if [[IS_ERR]]
        # CHECK-NEXT:     [[ERR:%.*]] = lit.ref.load %__error__
        # CHECK-NEXT:     call {{.*}}__del__{{.*}}([[ERR]])
        # CHECK-NEXT:     mark_consumed [[RESULT]]
        # CHECK:      except
        # CHECK-NEXT:   [[MOVE:%.*]] = lit.load.consume %__finally_error__
        # CHECK-NEXT:   lit.ref.store [[MOVE]], %__error__
        # CHECK:      else
        # CHECK-NEXT:   lit.try.yield
        # CHECK-NEXT: }
        # CHECK-NEXT: lit.error_return [[TRUE]]
    finally:
        somethingThatRaises()


@value
struct ThrowingExit:
    fn __enter__(self):
        pass

    fn __exit__(self) raises:
        pass

    fn __exit__(self, e: Error) raises -> Bool:
        return False


# CHECK-LABEL: lit.func @"context_mgr_exit_raises
fn context_mgr_exit_raises() raises:
    # CHECK:      [[MOVE:%.*]] = lit.load.consume %__with_error__
    # CHECK-NEXT: lit.ref.store [[MOVE]], %__error__
    # CHECK-NEXT: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
    # CHECK-NEXT: %__finally_error__ = lit.var.decl
    # CHECK-NEXT: lit.try
    # CHECK-NEXT:   [[DID_ERR:%.*]] = lit.ref.load %__with_exc__
    # CHECK-NEXT:   hlcf.if [[DID_ERR]]
    # CHECK-NEXT:     [[IMM:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT:     [[BOOL:%.*]] = lit.var.decl
    # CHECK-NEXT:     [[IS_ERR:%.*]] = lit.call {{.*}}__exit__{{.*}}([[IMM]], %__finally_error__, [[BOOL]])
    # CHECK-NEXT:     call {{.*}}__del__{{.*}}(%$CONTEXTMGR)
    # CHECK-NEXT:     if [[IS_ERR]]
    # CHECK-NEXT:       [[ERR:%.*]] = lit.ref.load %__error__
    # CHECK-NEXT:       call {{.*}}__del__{{.*}}([[ERR]])
    # CHECK-NEXT:       mark_consumed [[BOOL]]
    # CHECK:          else
    # CHECK-NEXT:       mark_consumed %__finally_error__
    # CHECK-NEXT:       yield
    # CHECK:        else
    # CHECK-NEXT:     call {{.*}}__del__{{.*}}(%$CONTEXTMGR)
    # CHECK:      except
    # CHECK-NEXT:   [[MOVE:%.*]] = lit.load.consume %__finally_error__
    # CHECK-NEXT:   lit.ref.store [[MOVE]], %__error__
    # CHECK:      else
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: }
    # CHECK-NEXT: lit.error_return [[TRUE]]
    with ThrowingExit():
        raise Error()


fn may_throw() raises -> RegExample:
    return RegExample()


# CHECK-LABEL: lit.func @"propagate_reg_error
fn propagate_reg_error() raises:
    # CHECK-NEXT: [[RESULT:%.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!RegExample,
    # CHECK-NEXT: %0 = lit.call {{.*}}may_throw{{.*}}(%__error__, [[RESULT]])
    # CHECK-NEXT: if %0
    # CHECK:        lit.error_return
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   [[VALUE:%.*]] = lit.ref.load [[RESULT]]
    # CHECK-NEXT:   lit.call {{.*}}@RegExample::@"__del__{{.*}}([[VALUE]])
    # CHECK-NEXT:   mark_consumed %__error__
    # CHECK-NEXT:   yield
    # CHECK-NEXT: }
    _ = may_throw()
    # CHECK-NEXT: %none = kgen.param.constant: none
    # CHECK-NEXT: lit.ref.store %none, %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: kgen.return [[FALSE]]


# CHECK-LABEL: lit.struct.decl @BigRegExample
@register_passable
struct BigRegExample:
    var a: RegExample
    var b: RegExample

    # Test a raising constructor.
    # CHECK-LABEL: lit.func @"__init__{{.*}}MemExample{{.*}}MemExample
    fn __init__(inout self, a: MemExample, b: MemExample) raises:
        # CHECK-NEXT: [[SELF:%.*]] = kgen.rebind %self
        # CHECK-NEXT: [[A_REF:%.*]] = lit.ref.struct.ger [[SELF]][a]
        # CHECK-NEXT: [[A:%.*]] = lit.call {{.*}}__init__{{.*}}([[A_REF]])
        self.a = RegExample()
        # CHECK-NEXT: [[B_REF:%.*]] = lit.ref.struct.ger [[SELF]][b]
        # CHECK-NEXT: [[B:%.*]] = lit.call {{.*}}__init__{{.*}}([[B_REF]])
        self.b = RegExample()
        # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
        # CHECK-NEXT: kgen.return [[FALSE]]


struct MyStringReturningCtx:
    var s: String

    fn __init__(inout self):
        self.s = "hey"

    fn __enter__(owned self) -> Self:
        return self^

    fn __moveinit__(inout self, owned existing: Self):
        self.s = existing.s

    fn read(self) raises -> String:
        return ""


# CHECK: lit.func @"testErrorReturn
fn testErrorReturn() raises:
    var input: String
    # CHECK: try
    with MyStringReturningCtx() as ctx:
        # CHECK-NOT: @MyStringReturningCtx::@"__del__
        var x = ctx.read()
        input = "hello"
    # CHECK: except
    use(input)


# COM: Test partial destruction of initialized fields upon an error return.
struct Field:
    fn __copyinit__(inout self, existing: Self):
        pass


# CHECK-LABEL: lit.struct.decl @DestructSome
struct DestructSome:
    var a: Field
    var b: Field

    # CHECK-LABEL: lit.func @"__init__
    fn __init__(inout self, a: Field, b: Field) raises:
        # CHECK:      call {{.*}}somethingThatRaises
        # CHECK-NEXT: if
        # CHECK-NEXT:   mark_consumed
        # CHECK-NEXT:   kgen.param.constant
        # CHECK-NEXT:   lit.error_return
        somethingThatRaises()

        # CHECK: [[FIELD:%.*]] = lit.ref.struct.ger %self[a]
        # CHECK-NEXT: __copyinit__{{.*}}([[FIELD]], %a)
        self.a = a

        # CHECK:      call {{.*}}somethingThatRaises
        # CHECK-NEXT: if
        # CHECK-NEXT:   [[FIELD:%.*]] = lit.ref.struct.ger %self[a]
        # CHECK-NEXT:   __del__{{.*}}([[FIELD]])
        # CHECK-NEXT:   mark_consumed %anonymous
        # CHECK-NEXT:   kgen.param.constant
        # CHECK-NEXT:   lit.error_return
        somethingThatRaises()

        # CHECK: [[FIELD:%.*]] = lit.ref.struct.ger %self[b]
        # CHECK-NEXT: __copyinit__{{.*}}([[FIELD]], %b)
        self.b = b

        # At this point 'self' is fully initialized, so any exit out should
        # destroy the whole thing.

        # CHECK:      call {{.*}}somethingThatRaises
        # CHECK-NEXT: if
        # CHECK-NEXT:   __del__{{.*}}(%self)
        # CHECK-NEXT:   mark_consumed %anonymous
        # CHECK-NEXT:   kgen.param.constant
        # CHECK-NEXT:   lit.error_return
        somethingThatRaises()


fn borrow_and_return(value: MemExample) raises -> MemExample:
    return value


fn use(err: Error):
    pass


# CHECK-LABEL: lit.func @"raising_use
fn raising_use(owned value: MemExample):
    try:
        # CHECK:      [[BORROW:%.*]] = lit.ref.immut %value
        # CHECK-NEXT: [[VAL:%.*]] = lit.var.decl "anonymous*"
        # CHECK-NEXT: [[IS_ERR:%.*]] = lit.call {{.*}}borrow_and_return{{.*}}([[BORROW]], %__try_error__, [[VAL]])
        # CHECK-NEXT: call {{.*}}@MemExample::@"__del__{{.*}}(%value)
        # CHECK-NEXT: if [[IS_ERR]]
        # CHECK-NEXT:   [[ERR:%.*]] = lit.ref.load %__try_error__
        # CHECK-NEXT:   call {{.*}}@Error::@"__del__{{.*}}([[ERR]])
        # CHECK-NEXT:   mark_consumed [[VAL]]
        # CHECK-NEXT:   lit.try.raise
        # CHECK-NEXT: } else {
        # CHECK-NEXT:   call {{.*}}@MemExample::@"__del__{{.*}}([[VAL]])
        # CHECK-NEXT:   mark_consumed %__try_error__
        _ = borrow_and_return(value)
    except:
        pass


# CHECK-LABEL: lit.struct.decl @ThrowingSelfInit
struct ThrowingSelfInit:
    var x: Int

    # CHECK-LABEL: lit.func @"__init__
    fn __init__(inout self) raises:
        self.x = 0

    # CHECK-LABEL: lit.func @"__init__
    fn __init__(inout self, x: Int) raises:
        # CHECK-NEXT: [[IS_ERR:%.*]] = lit.call {{.*}}__init__{{.*}}(%self, %__error__)
        # CHECK-NEXT: if [[IS_ERR]]
        # CHECK-NEXT:   mark_consumed %self
        # CHECK-NEXT:   [[TRUE:%.*]] = kgen.param.constant
        # CHECK-NEXT:   error_return [[TRUE]]
        # CHECK-NEXT: else
        # CHECK-NEXT:   mark_consumed %__error__
        # CHECK-NEXT:   yield
        self = ThrowingSelfInit()

    # CHECK-LABEL: lit.func @"__init__
    fn __init__(inout self, x: Int, y: Int) raises:
        # CHECK-NEXT: [[IS_ERR:%.*]] = lit.call {{.*}}__init__{{.*}}(%self, %__error__)
        # CHECK:      else
        # CHECK-NEXT:   call {{.*}}__del__{{.*}}(%self)
        # CHECK-NEXT:   mark_consumed %__error__
        # CHECK-NEXT:   yield
        self = ThrowingSelfInit()
        # CHECK:      lit.call {{.*}}__init__{{.*}}(%self, %__error__)
        self = ThrowingSelfInit()


# CHECK-LABEL: lit.func @"emplace_error
fn emplace_error() raises:
    # CHECK: lit.call {{.*}}Error::@"__init__{{.*}}(%__error__)
    # CHECK: lit.error_return
    __get_nearest_error_slot() = Error()
    __mlir_op.`lit.raise`()


struct InitFieldsDestroyedInThrowingConstructor:
    var x: MemExample

    fn __init__(inout self):
        self.x = MemExample()

    # CHECK-LABEL: lit.func @"__init__({{.*}}::InitFieldsDestroyedInThrowingConstructor=&,__mlir_type.i1)"
    fn __init__(inout self, cond: __mlir_type.`i1`) raises:
        self = InitFieldsDestroyedInThrowingConstructor()
        # CHECK:      hlcf.elif {
        # CHECK-NEXT:   hlcf.elif.yield %cond : i1
        # CHECK-NEXT: } then {
        # CHECK-NEXT:   lit.call {{.*}}__del__{{.*}}(%self)
        # CHECK-NEXT:   lit.call @{{.*}}::@Error::@"__init__
        # CHECK-NEXT:   kgen.param.constant
        # CHECK-NEXT:   lit.error_return
        # CHECK-NEXT: } else {
        # CHECK-NEXT:   hlcf.yield
        # CHECK-NEXT: }
        if cond:
            raise Error()
