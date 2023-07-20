// RUN: kgen-opt -allow-unregistered-dialect -split-input-file -lower-lit %s | FileCheck %s

// CHECK-NOT: lit.file_module

lit.file_module @module {
  // CHECK: kgen.generator @"module::test"()
  lit.func @test()  {
    kgen.return
  }

  lit.struct.decl @Adder<size> {
    // CHECK-LABEL: kgen.generator @"module::Adder::__add__"<size>(%arg0: !kgen.declref<@"module::Adder"<size = size>>) {
    // CHECK-NEXT:    kgen.call @"module::test"() : () -> ()
    lit.func @__add__(%self: !kgen.declref<@module::@Adder<size = size>>)  {
      kgen.call @module::@test() : () -> ()
      kgen.return
    }
  }

  // CHECK-LABEL: lit.struct.decl @"module::Adder"<size> {
}

// CHECK-LABEL: kgen.generator @caller(%arg0: !kgen.declref<@"module::Adder"<size = 10>>)
lit.func @caller(%ref: !kgen.declref<@module::@Adder<size = 10>>)  {
  // CHECK: kgen.call @"module::Adder::__add__"
  kgen.call @module::@Adder::@__add__<10>(%ref) : (!kgen.declref<@module::@Adder<size = 10>>) -> ()
  kgen.return
}

// -----

// CHECK-NOT: lit.
lit.package @package {
  lit.file_module @module {
    // CHECK: kgen.link "lib.a" as @"package::module::lib"
    kgen.link "lib.a" as @lib

    // CHECK: kgen.generator export @"package::module::foo"()
    lit.func export @foo() {
      kgen.return
    }
  }
}

// -----

lit.file_module @module {
   // CHECK-NOT: lit.alias.decl
   lit.alias.decl A = <42>
}
