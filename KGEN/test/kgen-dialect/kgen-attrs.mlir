// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #kgen.metadata<[byref, byval], [13 : index, 17 : i64], throws>
"someop"() {metadata = #kgen.metadata<[byref, byval], [13 : index, 17: i64], throws>} : () -> ()

// CHECK: #kgen.metadata<[], [], none>
"someop"() {metadata = #kgen.metadata<[], [], none>} : () -> ()

// CHECK: *"mangled_fn{{.*}}$Int
"someop"() {decl = #kgen<param.decl *"mangled_fn(Pointer[!kgen.declref<_\22$Int\22::_Int>])" : index>} : () -> ()

kgen.generator @return_one() -> index {
  %0 = index.constant 1
  kgen.return %0 : index
}

// CHECK: a = #kgen.concretetype.constant
// CHECK-SAME: b = #kgen.parameterizedtype.constant
"someop"() {
  a = #kgen.parameterizedtype.constant<!pop.array<1, i1>>,
  b = #kgen.parameterizedtype.constant<!pop.array<apply(:() -> index @return_one), i1>>
} : () -> ()
