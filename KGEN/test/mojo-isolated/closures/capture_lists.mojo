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

fn use(a:String, b:String, c:String, d:MoveMe):
    pass

# CHECK:  lit.fn @"toy
fn toy(byCopy:String, byRef:String, prefix:String, byRefMut: String, var byMove: MoveMe):
    # CHECK: [[V0:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: lit.call @{{.*}}::@String::@"__copyinit__(::String)"[{{.*}}](%byCopy, [[V0]])
    # CHECK: [[V1:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: lit.call @{{.*}}::@MoveMe::@"__moveinit__({{.*}})"[{{.*}}](%byMove, [[V1]])
    # CHECK: %byRef[ref: imm *"byRefMut`3"], [[V0]]
    # CHECK-SAME: [@{{.*}}::@String::@"__copyinit__({{.*}})" !lit.generator<{{.*}}>, @{{.*}}::@String::@"__moveinit__({{.*}})" !lit.generator<{{.*}}>, @{{.*}}::@String::@"__del__({{.*}})" !lit.generator<{{.*}}>],
    # CHECK-SAME: %byRefMut[ref: imm *"byRef`1"], [[V1]]
    fn myclosure(prefix: String) unified {read byRef, var byCopy, mut byRefMut, var byMove^} -> String:
        use(byRef, byCopy, byRefMut, byMove)
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
