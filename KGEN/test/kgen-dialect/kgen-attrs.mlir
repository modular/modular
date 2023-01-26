// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #kgen.conventions<[byref, byval], throws>
"someop"() {conventions = #kgen.conventions<[byref, byval], throws>} : () -> ()

// CHECK: #kgen.conventions<[], none>
"someop"() {conventions = #kgen.conventions<[], none>} : () -> ()

// CHECK: *"mangled_fn{{.*}}$Int
"someop"() {decl = #kgen<param.decl *"mangled_fn(Pointer[!kgen.declref<_\22$Int\22::_Int>])" : index>} : () -> ()
