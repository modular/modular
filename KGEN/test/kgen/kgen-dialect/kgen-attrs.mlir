// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #kgen.fn_metadata<["someRef", "v"], [13 : index, 17 : i64], [3.140000e+00 : f32]>
"some.op"() {metadata = #kgen.fn_metadata<["someRef", "v"], [13 : index, 17: i64], [3.14: f32]>} : () -> ()

// CHECK: #kgen.fn_metadata<[], [], []>
"some.op"() {metadata = #kgen.fn_metadata<[], [], []>} : () -> ()

// CHECK: *"mangled_fn{{.*}}$int
"some.op"() {decl = #kgen<param.decl *"mangled_fn(Pointer[!kgen.declref<_\22$int\22::_Int>])" : index>} : () -> ()

kgen.generator @return_one() -> index {
  %0 = index.constant 1
  kgen.return %0 : index
}

// CHECK: a = #kgen.concretetype.constant
// CHECK-SAME: b = #kgen.parameterizedtype.constant
"some.op"() {
  a = #kgen.parameterizedtype.constant<!pop.array<1, i1>>,
  b = #kgen.parameterizedtype.constant<!pop.array<apply(:() -> index @return_one), i1>>
} : () -> ()

// CHECK: #kgen.param.index.ref<0, false, 0> : index
"some.op"() {ref = #kgen.param.index.ref<0, false, 0> : index} : () -> ()

// CHECK: #kgen.int_literal<5> : !kgen.int_literal
"some.op"() {data = #kgen.int_literal<5> : !kgen.int_literal} : () -> ()

// CHECK: #kgen.env<{bar = 1 : index, foo}>
"some.op"() {env = #kgen.env<{bar = 1 : index, foo}>} : () -> ()

// CHECK: #kgen<decorators[1 : i64]>
"some.op"() {decorators = #kgen<decorators[1 : i64]>} : () -> ()

// CHECK: #kgen.int_literal<1234>
// CHECK-SAME: #kgen.int_literal<12345678901234567899012345678901234567890>
"some.op"() {a = #kgen.int_literal<1234> : !kgen.int_literal,
             b = #kgen.int_literal<12345678901234567899012345678901234567890> : !kgen.int_literal} : () -> ()
