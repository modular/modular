# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file| FileCheck %s

fn takeIt[T: fn (prefix: String) unified -> String, //](state: T, prefix:String):
    _ = state(prefix)



struct MoveMe(Movable):
    var x:Int
    fn __moveinit__(out self, deinit other: Self):
        self.x = other.x

fn use(a:String, d:MoveMe):
    pass

# CHECK:  lit.fn @"moveMeUser
fn moveMeUser(byCopy:String, prefix:String, var byMove: MoveMe):
    # CHECK: [[V0:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: lit.call {{.*}}::@String::@"__copyinit__(::String)"[{{.*}}](%byCopy, [[V0]])
    # CHECK: [[V1:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: lit.call {{.*}}::@MoveMe::@"__moveinit__({{.*}})"[{{.*}}](%byMove, [[V1]])
    # CHECK: [[V0]][@{{.*}}::@String::@"__copyinit__({{.*}})" !lit.generator<{{.*}}>, @{{.*}}::@String::@"__moveinit__({{.*}})" !lit.generator<{{.*}}>, @{{.*}}::@String::@"__del__({{.*}})" !lit.generator<{{.*}}>], [[V1]]
    fn myclosure(prefix: String) unified {var byCopy, var byMove^} -> String:
        use(byCopy, byMove)
        return prefix

    takeIt(myclosure, prefix)

# // -----

# COM: Trivial Capture

# CHECK:  lit.fn @"make_closure
fn make_closure(x: Int):
    # CHECK: [[V0:%.*]] = lit.closure.init[{{.*}}](%x)
    fn my_closure(y: Int) unified {var x} -> Int:
        return x + y

# // -----

# COM: Nested Captures

fn use(y:String):
    pass

fn make_closure(x: Int, str:String):
    # CHECK: [[COPY:%.*]] = lit.var.decl "anonymous*"
    # CHECK: lit.closure.init[{{.*}}](%x, [[COPY]]
    # CHECK: [[COPY_OF_COPY:%.*]] = lit.var.decl "anonymous*"
    # CHECK: lit.closure.init[{{.*}}](%x, [[COPY_OF_COPY]]
    fn my_closure(y: Int) unified {var x, var str} -> Int:
        fn my_nested_closure(z: Int) unified {var x, var str} -> Int:
            use(str)
            return x

        return x + y

# // -----

# COM: Verify mutability casts are inserted

fn takeIt[T: fn () unified -> None, //](state: T):
    state()

fn takesImmut(str: String):
    pass

fn takesMut(mut str: String):
    pass

# CHECK: lit.fn @"no_castsImmut({{.*}})"[imm *"byRef`"
fn no_castsImmut(byRef:String):
    # CHECK-NEXT: %byRef[ref: imm *"byRef`"]
    fn myclosure() unified {read byRef}:
        takesImmut(byRef)

    takeIt(myclosure)

# CHECK: lit.fn @"no_castsMut({{.*}})"[mut *"byRefMut`"
fn no_castsMut(mut byRefMut: String):
    # CHECK-NEXT: %byRefMut[ref: mut *"byRefMut`"
    fn myclosure() unified {mut byRefMut}:
        takesImmut(byRefMut)

    takeIt(myclosure)

# CHECK: lit.fn @"casts({{.*}})"[mut *"byRefMut`"
fn casts(mut byRefMut: String):
    # CHECK: [[V0:%.*]] = lit.ref.immut %byRefMut : <!String, mut *"byRefMut`">
    # CHECK: lit.closure.init[{{.*}}]([[V0]]
    fn myclosure() unified {read byRefMut}:
        takesImmut(byRefMut)

    takeIt(myclosure)

# // -----

# COM: Ensure "capture all by" emits the correct IR.

fn takeIt[T: fn () unified -> String, //](state: T):
    _ = state()


fn use(a: String, d: String):
    pass


# CHECK-LABEL:  lit.fn @"toy
fn toy(A: String, B: String, mut C: String, mut D: String):
    # CHECK: (%A[ref: imm *"A`"], %B[ref: imm *"B`1"])
    fn readAll() unified {read} -> String:
        use(A, B)
        return A
    takeIt(readAll)

    # CHECK: @String::@"__copyinit__
    # CHECK: @String::@"__copyinit__
    # CHECK: lit.closure.init
    fn copyAll() unified {var} -> String:
        use(A, B)
        return C

    takeIt(copyAll)

    # CHECK: @String::@"__moveinit__
    # CHECK: @String::@"__moveinit__
    # CHECK: lit.closure.init
    fn moveAll() unified {var^} -> String:
        use(C, D)
        return D
    takeIt(moveAll)


# COM: Ensure multiple references to the same capture result in a single copy

struct MyCopyableType(ImplicitlyCopyable):
    fn __copyinit__(out self, other: Self):
        pass

fn use(y: MyCopyableType, wy:MyCopyableType):
    pass

# CHECK: lit.fn @"testOnce
fn testOnce(x: MyCopyableType):
    # CHECK-COUNT: 1 @MyCopyableType::@"__copyinit__
    fn myclosure() unified {var}:
        use(x, x)

# // -----

# COM: Trailing commas are supported

def callIt(x: String, x1: String, x2: String, x3: String, x4: String):
    pass


@no_inline
def takeIt[T: def () unified -> None](impl: T):
    impl()


# CHECK-LABEL: lit.fn @"longCaptureLists
def longCaptureLists(
    mut something: String,
    mut something1: String,
    mut something2: String,
    mut something3: String,
    mut something4: String,
    mut something5: String,
):
    # CHECK: lit.closure.init
    def closure() unified {
        var something,
        mut something2,
        read something3,
        mut something4,
        read something5,
    }:
        callIt(something, something2, something3, something4, something5)

    takeIt(closure)
