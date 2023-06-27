// RUN: kgen-opt -allow-unregistered-dialect -split-input-file -lower-lit %s | FileCheck %s

// CHECK-NOT: lit.file_module

lit.file_module @module {
  // CHECK: kgen.generator @"module::test"()
  lit.func @test()  {
    kgen.return
  }

  lit.struct.decl @Adder<size> {
    %base = lit.varlet.decl "base", var = true, synth=false : <index>

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
// CHECK: kgen.generator @"package::subpackage::module::foo"()
// CHECK: kgen.export @"package::subpackage::module::foo"

lit.package @package {
  lit.package @subpackage {
    lit.file_module @module {
      lit.func @foo() {
        kgen.return
      }
      kgen.export @package::@subpackage::@module::@foo
    }
  }
}

// -----

lit.file_module @module {
   // CHECK-NOT: kgen.param.declare
   kgen.param.declare A = <42>
}
