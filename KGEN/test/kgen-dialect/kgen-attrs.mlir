// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #kgen.metadata<[byref, byval], [13 : index, 17 : i64], throws>
"some.op"() {metadata = #kgen.metadata<[byref, byval], [13 : index, 17: i64], throws>} : () -> ()

// CHECK: #kgen.metadata<[], [], none>
"some.op"() {metadata = #kgen.metadata<[], [], none>} : () -> ()

// CHECK: *"mangled_fn{{.*}}$Int
"some.op"() {decl = #kgen<param.decl *"mangled_fn(Pointer[!kgen.declref<_\22$Int\22::_Int>])" : index>} : () -> ()

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

"some.op"() {
  // CHECK: a = #kgen.param.region<"aRegion", []> : !kgen.signature<() -> ()>,
  a = #kgen<param.region<"aRegion" , []>> : !kgen.signature<() -> ()>,
  // CHECK-SAME: b = #kgen.param.region<"bRegion", [ foo = 1 : i64]> : !kgen.signature<(index) -> index>
  b = #kgen<param.region<"bRegion" , [foo = 1]>> : !kgen.signature<(index) -> (index)>
} : () -> ()
