// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #kgen.metadata<[byref, byval], throws>
"someop"() {metadata = #kgen.metadata<[byref, byval], throws>} : () -> ()

// CHECK: #kgen.metadata<[], none>
"someop"() {metadata = #kgen.metadata<[], none>} : () -> ()

// CHECK: *"mangled_fn{{.*}}$Int
"someop"() {decl = #kgen<param.decl *"mangled_fn(Pointer[!kgen.declref<_\22$Int\22::_Int>])" : index>} : () -> ()
