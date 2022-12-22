// RUN: kgen-opt -allow-unregistered-dialect -split-input-file -lower-lit %s | FileCheck %s

// CHECK-NOT: lit.file_module

lit.file_module @module {
  // CHECK: kgen.generator @"module::test"()
  lit.func @test()  {
    kgen.return
  }

  kgen.struct.decl @Adder<size> {
    %base = lit.var.decl "base" : <index>

    // CHECK-LABEL: kgen.generator @"module::Adder::__add__"<size>(%arg0: !kgen.declref<@"module::Adder"<size = size>>) {
    // CHECK-NEXT:    kgen.call @"module::test"() : () -> ()
    lit.func @__add__(%self: !kgen.declref<@module::@Adder<size = size>>)  {
      kgen.call @module::@test() : () -> ()
      kgen.return
    }
  }

  // CHECK-LABEL: kgen.struct.decl @"module::Adder"<size> {
}

// CHECK-LABEL: kgen.generator @caller(%arg0: !kgen.declref<@"module::Adder"<size = 10>>)
lit.func @caller(%ref: !kgen.declref<@module::@Adder<size = 10>>)  {
  // CHECK: kgen.call @"module::Adder::__add__"
  kgen.call @module::@Adder::@__add__<size = 10>(%ref) : (!kgen.declref<@module::@Adder<size = 10>>) -> ()
  kgen.return
}
