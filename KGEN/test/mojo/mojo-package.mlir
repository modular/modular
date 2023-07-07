// RUN: mojo package %s --name mojo-test-pkg | FileCheck %s

// CHECK: lit.package @"mojo-test-pkg" attributes {archiveBytes = {{.*}}, compiledFor = {{.*}}
// We just use this package as a substitute for the frontend work that's going
// to land soon, which will be able to generate one of these from a directory.
lit.package @mypackage {
  // CHECK: lit.package @inner1
  lit.package @inner1 {
    // CHECK: lit.file_module @myfile
    lit.file_module @myfile {
      // CHECK: lit.struct.decl @aStruct
      lit.struct.decl @aStruct {
        // CHECK: lit.func @parametric<{{.*}}>() -> index
        lit.func @parametric<aParameter>() -> index {
          // CHECK-NEXT: kgen.param.constant = <{{.*}}>
          %0 = kgen.param.constant = <aParameter>
          // CHECK-NEXT: kgen.return
          kgen.return %0 : index
        }

        // CHECK: lit.func @foo() attributes {postElaborationBodyRef = {{.*}}
        lit.func @foo() {
          // CHECK-NEXT: lit.extern_func
          kgen.return
        }
      }

      kgen.export @mypackage::@inner1::@myfile::@aStruct::@foo
    }
  }
  // CHECK: lit.package @inner2
  lit.package @inner2 {
    // CHECK: lit.file_module @myfile2
    lit.file_module @myfile2 {
      // CHECK: lit.func @bar() -> index attributes {postElaborationBodyRef = {{.*}}
      lit.func @bar() -> index {
        %0 = kgen.param.constant = <3>
        // CHECK-NEXT: lit.extern_func
        kgen.return %0 : index
      }

      kgen.export @mypackage::@inner2::@myfile2::@bar
    }
  }
}

// CHECK: {-#
// CHECK: dialect_resources
// CHECK: builtin
// CHECK: archive_{{.*}}: {{.*}}
// CHECK: bytecode_{{.*}}: {{.*}}
// CHECK: bytecode_{{.*}}: {{.*}}

