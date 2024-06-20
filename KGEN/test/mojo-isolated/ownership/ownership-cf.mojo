# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s
# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo --debug-level full -o /dev/null

# Control flow related CheckLifetimes tests.


fn use(err: Error):
    pass


fn use(str: String):
    pass


fn use(a: MemExample):
    pass


fn use_mut(inout a: MemExample):
    pass


# CHECK-LABEL: lit.struct.decl @MemExample
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


# CHECK-LABEL: lit.func @"if_examples
fn if_examples(cond: __mlir_type.i1):
    # CHECK: %a = lit.var.decl
    var a: MemExample

    # CHECK-NEXT: %b = lit.var.decl
    # CHECK-NEXT: lifetime.start %b
    # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%b)
    # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%b)
    # CHECK-NEXT: lifetime.end %b
    var b = MemExample()

    # CHECK: hlcf.elif
    # CHECK-NEXT: hlcf.elif.yield
    # CHECK-NEXT: } then {
    if cond:
        # CHECK-NEXT: lifetime.start %a
        # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%a)
        # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%a)
        # CHECK-NEXT: lifetime.end %a
        a = MemExample()
    # CHECK-NEXT: hlcf.yield
    # CHECK-NEXT: } else {
    else:
        # CHECK-NEXT: lifetime.start %b
        # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%b)
        # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%b)
        # CHECK-NEXT: lifetime.end %b
        b = MemExample()
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: }

    # CHECK-NEXT: %c = lit.var.decl
    # CHECK-NEXT: lifetime.start %c
    # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%c)
    var c = MemExample()
    # CHECK: hlcf.elif {
    # CHECK-NEXT: hlcf.elif.yield %cond
    # CHECK-NEXT: } then {
    if cond:
        # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%c)
        # CHECK-NEXT: lifetime.end %c
        # CHECK-NEXT: lifetime.start %c
        # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%c)
        c = MemExample()
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: } else {
    else:
        pass
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: }
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %c
    # CHECK-NEXT: lit.call @{{.*}}noop{{.*}}([[IMMREF]])
    c.noop()
    # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%c)
    # CHECK-NEXT: lifetime.end %c

    # CHECK-NEXT: %d = lit.var.decl "d"
    # CHECK-NEXT: lifetime.start %d
    # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%d)
    var d = MemExample()

    # CHECK: hlcf.elif {
    # CHECK-NEXT: [[ONE:%[0-9]+]] = kgen.param.constant: i1 = <1>
    # CHECK-NEXT: hlcf.elif.yield [[ONE]]
    # CHECK-NEXT: } then {
    if True:
        # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %d
        # CHECK-NEXT: lit.call @{{.*}}noop{{.*}}([[IMMREF]])
        d.noop()
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: }

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %d
    # CHECK-NEXT: lit.call @{{.*}}noop{{.*}}([[IMMREF]])
    d.noop()
    # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%d)
    # CHECK-NEXT: lifetime.end %d


# CHECK-LABEL: lit.func @"try_examples
fn try_examples(cond: __mlir_type.i1, err: Error):
    # CHECK-NEXT: %a = lit.var.decl
    var a: MemExample
    # CHECK: lit.try
    # CHECK-NOT: %a
    try:
        # CHECK-NEXT: lifetime.start %__try_error__
        # CHECK-NEXT: [[ERR:%.*]] = lit.call {{.*}}Error::@"__copyinit__{{.*}}(%__try_error__, %err)
        raise err
        # COM: The error value isn't used on the except branch, so it is immediately
        # COM: destroyed.
        # CHECK-NEXT: [[ERR:%.*]] = lit.ref.load %__try_error__
        # CHECK-NEXT: lit.call @{{.*}}Error::@"__del__{{.*}}([[ERR]])
    # CHECK: } except {
    except:
        # CHECK-NEXT: lifetime.start %a
        # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%a)
        a = MemExample()
        # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: }
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
    # CHECK-NEXT: lit.call @{{.*}}noop{{.*}}([[IMMREF]])
    a.noop()  # ok
    # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%a)
    # CHECK-NEXT: lifetime.end %a

    # CHECK-NEXT: %b = lit.var.decl
    var b: MemExample
    # CHECK-NEXT: [[ERRSLOT:%.*]] = lit.var.decl "e"
    # CHECK-NEXT: lit.try {
    try:
        # CHECK-NEXT: lifetime.start %b
        # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%b)
        b = MemExample()
        # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%b)
        # CHECK-NEXT: lifetime.end %b
        raise err
    # CHECK: } except {
    # CHECK-NEXT: [[ERR:%.*]] = lit.ref.load [[ERRSLOT]]
    # CHECK-NEXT: lit.call {{.*}}use{{.*}}([[ERR]])
    # CHECK-NEXT: [[ERR:%.*]] = lit.ref.load [[ERRSLOT]]
    # CHECK-NEXT: lit.call @{{.*}}Error::@"__del__{{.*}}([[ERR]])
    # CHECK-NEXT: lifetime.end [[ERRSLOT]]
    # CHECK-NEXT: lit.try.yield
    except e:
        use(e)
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: }

    # CHECK-NEXT: lifetime.start %b
    # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%b)
    b = MemExample()
    # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%b)
    # CHECK-NEXT: lifetime.end %b

    # CHECK-NEXT: %c = lit.var.decl
    var c: MemExample
    # CHECK-NEXT: [[ERRSLOT:%.*]] = lit.var.decl "e"
    # CHECK-NEXT: lit.try {
    try:
        # CHECK-NEXT: lifetime.start %c
        # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%c)
        c = MemExample()
        # CHECK-NOT: %c
        raise err
    # CHECK: } except {
    # CHECK-NEXT: [[ERR:%.*]] = lit.ref.load [[ERRSLOT]]
    # CHECK-NEXT: lit.call {{.*}}use{{.*}}([[ERR]])
    # CHECK-NEXT: [[ERR:%.*]] = lit.ref.load [[ERRSLOT]]
    # CHECK-NEXT: lit.call @{{.*}}Error::@"__del__{{.*}}([[ERR]])
    # CHECK-NEXT: lifetime.end [[ERRSLOT]]
    # CHECK-NEXT: lit.try.yield
    except e:
        use(e)
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: }
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %c
    # CHECK-NEXT: lit.call @{{.*}}noop{{.*}}([[IMMREF]])
    c.noop()
    # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%c)
    # CHECK-NEXT: lifetime.end %c

    # CHECK-NEXT: %d = lit.var.decl
    var d: MemExample
    # CHECK-NEXT: [[ERRSLOT:%.*]] = lit.var.decl "e"
    # CHECK-NEXT: lit.try {
    try:
        # CHECK-NEXT:  hlcf.elif
        # CHECK-NEXT:  hlcf.elif.yield
        if cond:
            raise err
        # CHECK-NOT: %d
    # CHECK: } except {
    except e:
        # CHECK: call {{.*}}Error::@"__del__
        use(e)
        # CHECK: lit.call @{{.*}}__init__{{.*}}(%d)
        d = MemExample()
        # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: } else {
    else:
        # CHECK-NEXT: lifetime.start %d
        # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%d)
        d = MemExample()
        # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: }

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %d
    # CHECK-NEXT: lit.call @{{.*}}noop{{.*}}([[IMMREF]])
    d.noop()
    # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%d)
    # CHECK-NEXT: lifetime.end %d


# CHECK-LABEL: lit.func @"chris_lifetime_example
fn chris_lifetime_example(a: Bool, b: Bool):
    var x: MemExample
    # CHECK: lit.try
    try:
        # CHECK: lit.try
        try:
            # CHECK: hlcf.elif
            if a:
                # CHECK: __init__{{.*}}(%x)
                x = MemExample()
                # CHECK: lit.try.raise
                raise Error()
        # CHECK: except
        # CHECK: hlcf.elif
        # CHECK-NEXT: lit.call
        # CHECK-NEXT: hlcf.elif.yield
        # CHECK-NEXT: } then {
        # CHECK-NEXT: __del__{{.*}}(%x)
        # CHECK: return
        # CHECK: else
        # CHECK: lit.try.raise
        # CHECK: else
        # CHECK: hlcf.elif
        # CHECK: return
        # CHECK: else
        finally:
            if b:
                return
    # CHECK: except
    except:
        # CHECK: [[DEAD:%.*]] = lit.transfer_mem_ownership %x
        # CHECK: __del__{{.*}}([[DEAD]])
        _ = x^
    # CHECK: else
    # CHECK-NEXT: lit.try.yield


# CHECK-LABEL: lit.func @"loop_example
fn loop_example(cond1: __mlir_type.i1, cond2: __mlir_type.i1):
    # CHECK-NEXT: %a = lit.var.decl "a"
    var a: MemExample
    # CHECK-NEXT: %b = lit.var.decl "b"
    var b: MemExample
    # CHECK-NEXT: %c = lit.var.decl "c"
    var c: MemExample

    # CHECK-NEXT: lifetime.start %a
    # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%a)
    a = MemExample()

    # Unneeded boilerplate due to 'while True':
    # CHECK-NEXT: hlcf.loop "_loop_0" {
    # CHECK-NEXT:  = kgen.param.constant: i1 = <1>
    # CHECK-NEXT:      hlcf.if
    # CHECK-NEXT:        hlcf.yield
    # CHECK-NEXT:      } else {
    # CHECK-NEXT:        kgen.unreachable
    # CHECK-NEXT:      }
    while True:
        # CHECK-NEXT: lifetime.start %c
        # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%c)
        c = MemExample()
        # CHECK-NEXT: hlcf.elif {
        # CHECK-NEXT: hlcf.elif.yield %cond2 : i1
        # CHECK-NEXT: } then {
        if cond2:
            # CHECK-NEXT: lifetime.start %b
            # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%b)
            b = MemExample()
            # CHECK-NEXT: hlcf.break
            break
        # CHECK-NEXT: } else {
        # CHECK-NEXT:   lit.call @{{.*}}__del__{{.*}}(%a)
        # CHECK-NEXT:   lifetime.end %a
        # CHECK-NEXT:   lit.call @{{.*}}__del__{{.*}}(%c)
        # CHECK-NEXT:   lifetime.end %c
        # CHECK-NEXT:   hlcf.yield
        # CHECK-NEXT: }

        # CHECK-NEXT: lifetime.start %a
        # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%a)
        # CHECK-NEXT: hlcf.continue
        a = MemExample()
    # CHECK-NEXT: }

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
    # CHECK-NEXT: lit.call @{{.*}}noop{{.*}}([[IMMREF]])
    a.noop()
    # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%a)
    # CHECK-NEXT: lifetime.end %a

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %b
    # CHECK-NEXT: lit.call @{{.*}}noop{{.*}}([[IMMREF]])
    b.noop()
    # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%b)
    # CHECK-NEXT: lifetime.end %b

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %c
    # CHECK-NEXT: lit.call @{{.*}}noop{{.*}}([[IMMREF]])
    c.noop()
    # CHECK-NEXT: lit.call @{{.*}}__del__{{.*}}(%c)
    # CHECK-NEXT: lifetime.end %c


# CHECK-LABEL: lit.struct.decl @TestLoopWithWholeObjectBit
struct TestLoopWithWholeObjectBit:
    var field: MemExample

    # CHECK: lit.func @"__init__
    fn __init__(inout self, cond: __mlir_type.i1):
        # CHECK-NEXT: %buf = lit.var.decl "buf"
        # CHECK-NEXT: lifetime.start %buf
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%buf)
        var buf = MemExample()

        # CHECK-NEXT: hlcf.loop "_loop_0" {
        # CHECK-NEXT:   hlcf.if %cond {
        # CHECK-NEXT:     hlcf.yield
        # CHECK-NEXT:   } else {
        # CHECK-NEXT:     hlcf.break
        # CHECK-NEXT:   }
        while cond:
            # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %buf
            # CHECK-NEXT:   lit.call {{.*}}noop{{.*}}([[IMMREF]])
            # CHECK-NEXT:   hlcf.continue
            buf.noop()
        # CHECK-NEXT: }

        # CHECK-NEXT: [[TRANSFER_REF:%.*]] = lit.transfer_mem_ownership %buf
        # CHECK-NEXT: [[FIELD_REF:%.*]] = lit.ref.struct.ger %self[field]
        # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}([[FIELD_REF]], [[TRANSFER_REF]])
        # CHECK-NEXT: lifetime.end %buf
        # CHECK-NEXT: %none = kgen.param.constant
        # CHECK-NEXT: kgen.return
        self.field = buf^


# CHECK-LABEL: lit.func @"testInfiniteloop
fn testInfiniteloop():
    # CHECK-NEXT:  hlcf.loop "_loop_0" {
    # CHECK-NEXT:    %0 = kgen.param.constant: i1 = <1>
    # CHECK-NEXT:    hlcf.if %0 {
    # CHECK-NEXT:      hlcf.yield
    # CHECK-NEXT:    } else {
    # CHECK-NEXT:      kgen.unreachable
    # CHECK-NEXT:    }
    while True:
        # CHECK-NEXT:  %localThing = lit.var.decl
        # CHECK-NEXT:  lifetime.start %localThing
        # CHECK-NEXT:  lit.call {{.*}}__init__{{.*}}(%localThing)
        # CHECK-NEXT:  [[IMMREF:%.*]] = lit.ref.immut %localThing
        # CHECK-NEXT:  lit.call {{.*}}noop{{.*}}([[IMMREF]])
        # CHECK-NEXT:  lit.call {{.*}}__del__{{.*}}(%localThing)
        # CHECK-NEXT:  lifetime.end %localThing
        var localThing = MemExample()
        localThing.noop()
    # CHECK-NEXT:      hlcf.continue
    # CHECK-NEXT:  }


@value
@register_passable("trivial")
struct TrivialRange:
    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) -> Int:
        return 1

    fn __len__(self) -> Int:
        return 1


# Issue #98: https://github.com/modularml/mojo/issues/98
# CHECK-LABEL: lit.func @"mojo98
fn mojo98(n: Int):
    var a = MemExample()
    for i in TrivialRange():
        a.x = i


struct MyStringReturningCtx:
    var s: String

    fn __init__(inout self):
        self.s = "hey"

    fn __enter__(owned self) -> Self:
        return self^

    fn __moveinit__(inout self, owned existing: Self):
        self.s = ""

    fn read(self) raises -> String:
        return ""


# CHECK-LABEL: lit.func @"testErrorReturn
fn testErrorReturn() raises:
    var input: String
    # CHECK: try
    with MyStringReturningCtx() as ctx:
        # CHECK-NOT: @MyStringReturningCtx::@"__del__
        var x = ctx.read()
        input = "hello"
    # CHECK: except
    use(input)


fn marker():
    pass


# CHECK-LABEL: lit.func @"test_param_for1
# MOCO-831
fn test_param_for1(cond: Bool, cond2: Bool):
    # CHECK-NEXT: %mem = lit.var.decl
    # CHECK-NEXT: lifetime.start %mem
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mem)
    var mem = MemExample()

    # CHECK: kgen.param.for *"x`1"{{.*}}!kgen.none>
    # CHECK-NEXT: {
    @parameter
    for x in TrivialRange():
        # Make sure nothing sneaks in here.
        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()

        # CHECK-NEXT: hlcf.elif {
        # ... cond ...
        # CHECK: } then {
        if cond:
            # CHECK: lit.call {{.*}}__del__{{.*}}(%mem)
            # CHECK-NEXT: lifetime.end %mem
            # CHECK-NEXT: lit.call {{.*}}marker()
            marker()
            # CHECK-NEXT: kgen.param.for.break
            break

        # CHECK: hlcf.elif {
        # ... cond2 ...
        # CHECK: } then {
        if cond2:
            # CHECK: lit.call {{.*}}__del__{{.*}}(%mem)
            # CHECK-NEXT: lifetime.end %mem
            # CHECK-NEXT: lit.call {{.*}}marker()
            marker()
            # CHECK-NEXT: kgen.param.for.break
            break
        # CHECK-NEXT: } else {
        # CHECK-NEXT:   hlcf.yield
        # CHECK-NEXT: }
        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()
        # CHECK-NEXT: kgen.param.for.continue

    # CHECK-NEXT: } else {
    else:
        # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut %mem
        # CHECK-NEXT: lit.call {{.*}}use{{.*}}([[TMP]])
        use(mem)
        # CHECK: lit.call {{.*}}__del__{{.*}}(%mem)
        # CHECK-NEXT: lifetime.end %mem

        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()
        # CHECK-NEXT: kgen.param.yield


# CHECK-LABEL: lit.func @"test_param_for2
# MOCO-831
fn test_param_for2():
    # CHECK: lit.call {{.*}}__init__{{.*}}(%mem)
    var mem = MemExample()

    # CHECK: kgen.param.for *"x`1"{{.*}}!kgen.none>
    # CHECK-NEXT: {
    @parameter
    for x in TrivialRange():
        # Make sure nothing sneaks in here.
        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()

        # CHECK-NEXT: lit.call {{.*}}use_mut{{.*}}(%mem)
        use_mut(mem)
        # CHECK-NEXT: kgen.param.for.continue

    # CHECK-NEXT: } else {
    else:
        # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem)
        # CHECK-NEXT: lifetime.end %mem
        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()

        # CHECK-NEXT: kgen.param.yield


# CHECK-LABEL: lit.func @"test_elif
fn test_elif(cond: Bool, cond2: Bool):
    var mem1 = MemExample()
    var mem2 = MemExample()
    var mem3 = MemExample()

    # CHECK: hlcf.elif {
    # CHECK-NEXT:  __mlir_i1__
    # CHECK-NEXT: hlcf.elif.yield
    # CHECK-NEXT: } then {
    if cond:
        # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem2)
        # CHECK-NEXT: lifetime.end %mem2
        # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem3)
        # CHECK-NEXT: lifetime.end %mem3
        # CHECK-NEXT: lit.call {{.*}}use_mut{{.*}}(%mem1)
        use_mut(mem1)
        # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem1)
        # CHECK-NEXT: lifetime.end %mem1

        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()
        # CHECK-NEXT: hlcf.yield
    # CHECK-NEXT: } {
    # mem1 never used at this point, destroy in the condition.
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem1)
    # CHECK-NEXT: lifetime.end %mem1
    # CHECK-NEXT: __mlir_i1__
    # CHECK-NEXT: hlcf.elif.yield

    # CHECK-NEXT: } then {
    elif cond2:
        # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem2)
        # CHECK-NEXT: lifetime.end %mem2
        # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem3)
        # CHECK-NEXT: lifetime.end %mem3

        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()
        # CHECK-NEXT: hlcf.yield
    # CHECK-NEXT: } {

    # Last use of mem2 is in this condition.
    # CHECK-NEXT: lit.ref.struct.ger %mem2[x]
    # CHECK-NEXT: lit.ref.load
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem2)
    # CHECK: hlcf.elif.yield

    # CHECK-NEXT: } then {
    elif mem2.x == 0:
        # CHECK-NEXT: lit.call {{.*}}use_mut{{.*}}(%mem3)
        use_mut(mem3)
        # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem3)
        # CHECK-NEXT: lifetime.end %mem3

        # CHECK-NEXT: lit.call {{.*}}marker()
        marker()
        # CHECK-NEXT: hlcf.yield

    # CHECK-NEXT: } else {
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%mem3)
    # CHECK-NEXT: lifetime.end %mem3
    # CHECK-NEXT: hlcf.yield
