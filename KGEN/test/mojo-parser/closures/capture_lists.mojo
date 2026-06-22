# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file| FileCheck %s

def takeIt[T: def (prefix: String) -> String, //](state: T, prefix:String):
    _ = state(prefix)



struct MoveMe(Movable):
    var x:Int

def use(a:String, d:MoveMe):
    pass

# CHECK:  lit.fn @"moveMeUser
def moveMeUser(byCopy:String, prefix:String, var byMove: MoveMe):
    # CHECK: [[V0:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: lit.call {{.*}}::@String::@"__init__{{.*}}(%byCopy, [[V0]]){{.*}}(*, "copy":
    # CHECK: [[V1:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: lit.call {{.*}}::@MoveMe::@"__init__{{.*}}(%byMove, [[V1]]){{.*}}(*, "move":
    # CHECK: [[V0]][@{{.*}}::@String::@"__init__(copy:::String)
    def myclosure(prefix: String) {var byCopy, var byMove^} -> String:
        use(byCopy, byMove)
        return prefix

    takeIt(myclosure, prefix)

# // -----

# COM: Trivial Capture

# CHECK:  lit.fn @"make_closure
def make_closure(x: Int):
    # CHECK: [[V0:%.*]] = lit.closure.init[{{.*}}](%x)
    def my_closure(y: Int) {var x} -> Int:
        return x + y

# // -----

# COM: Nested Captures

def use(y:String):
    pass

def make_closure(x: Int, str:String):
    # CHECK: [[COPY:%.*]] = lit.var.decl "anonymous*"
    # CHECK: lit.closure.init[{{.*}}](%x, [[COPY]]
    # CHECK: [[COPY_OF_COPY:%.*]] = lit.var.decl "anonymous*"
    # CHECK: lit.closure.init[{{.*}}]({{.*}}, [[COPY_OF_COPY]]
    def my_closure(y: Int) {var x, var str} -> Int:
        def my_nested_closure(z: Int) {var x, var str} -> Int:
            use(str)
            return x

        return x + y

# // -----

# COM: Verify mutability casts are inserted

def takeIt[T: def () -> None, //](state: T):
    state()

def takesImmut(str: String):
    pass

def takesMut(mut str: String):
    pass

# CHECK: lit.fn @"no_castsImmut({{.*}})"[imm *"byRef`"
def no_castsImmut(byRef:String):
    # CHECK-NEXT: %byRef[ref: imm *"byRef`"]
    def myclosure() {read byRef}:
        takesImmut(byRef)

    takeIt(myclosure)

# CHECK: lit.fn @"no_castsMut({{.*}})"[mut *"byRefMut`"
def no_castsMut(mut byRefMut: String):
    # CHECK-NEXT: %byRefMut[ref: mut *"byRefMut`"
    def myclosure() {mut byRefMut}:
        takesImmut(byRefMut)

    takeIt(myclosure)

# CHECK: lit.fn @"casts({{.*}})"[mut *"byRefMut`"
def casts(mut byRefMut: String):
    # CHECK: [[V0:%.*]] = lit.ref.immut %byRefMut : <!String, mut *"byRefMut`">
    # CHECK: lit.closure.init[{{.*}}]([[V0]]
    def myclosure() {read byRefMut}:
        takesImmut(byRefMut)

    takeIt(myclosure)

# // -----

# COM: Verify ref capture preserves original mutability

def use(ref a: String, ref b: String, ref c: String):
    pass


# COM: Capture-all-by-ref preserves each value's original mutability
# CHECK-LABEL: lit.fn @"captureAllByRef
def captureAllByRef(A: String, mut B: String, ref C: String):
    # CHECK-NOT: lit.ref.immut
    # CHECK: lit.closure.init{{.*}}%A[ref: imm
    # CHECK-SAME: %B[ref: mut
    # CHECK-SAME: %C[ref: mut=
    def refAll() {ref}:
        use(A, B, C)
        pass


# // -----

# COM: Ensure "capture all by" emits the correct IR.

def takeIt[T: def () -> String, //](state: T):
    _ = state()


def use(a: String, d: String):
    pass


# CHECK-LABEL:  lit.fn @"toy
def toy(A: String, B: String, mut C: String, mut D: String):
    # CHECK: (%A[ref: imm *"A`"], %B[ref: imm *"B`1"])
    def readAll() {read} -> String:
        use(A, B)
        return A
    takeIt(readAll)

    # CHECK: @String::@"__init__{{.*}}"{{.*}}*, "copy"
    # CHECK: @String::@"__init__{{.*}}"{{.*}}*, "copy"
    # CHECK: lit.closure.init
    def copyAll() {var} -> String:
        use(A, B)
        return C

    takeIt(copyAll)

    # CHECK: @String::@"__init__{{.*}}"{{.*}}*, "move"
    # CHECK: @String::@"__init__{{.*}}"{{.*}}*, "move"
    # CHECK: lit.closure.init
    def moveAll() {var^} -> String:
        use(C, D)
        return D
    takeIt(moveAll)


# COM: Ensure multiple references to the same capture result in a single copy

struct MyCopyableType(ImplicitlyCopyable):
    def __init__(out self, *, copy: Self):
        pass

def use(y: MyCopyableType, wy:MyCopyableType):
    pass

# CHECK: lit.fn @"testOnce
def testOnce(x: MyCopyableType):
    # CHECK-COUNT: 1 @MyCopyableType::@"__init__{{.*}}"{{.*}}*, "copy"
    def myclosure() {var}:
        use(x, x)

# // -----

# COM: Trailing commas are supported

def callIt(x: String, x1: String, x2: String, x3: String, x4: String) raises:
    pass


@no_inline
def takeIt[T: def () raises -> None](impl: T) raises:
    impl()


# CHECK-LABEL: lit.fn @"longCaptureLists
def longCaptureLists(
    mut something: String,
    mut something1: String,
    mut something2: String,
    mut something3: String,
    mut something4: String,
    mut something5: String,
) raises:
    # CHECK: lit.closure.init
    def closure() raises {
        var something,
        mut something2,
        read something3,
        mut something4,
        read something5,
    }:
        callIt(something, something2, something3, something4, something5)

    takeIt(closure)
