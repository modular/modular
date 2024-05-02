// RUN: kgen-opt %s -verify-parameters=simplify=true | FileCheck %s

lit.file_module @module {
  lit.func @store_load(%a: index) -> !kgen.struct<(index, index)> {
    %x = lit.var.decl "x" var : !lit.ref<index, mut lt>
    lit.ref.store %a, %x : <index, mut lt>
    %0 = lit.ref.load %x : <index, mut lt>
    %1 = lit.load.consume %x : !lit.ref<index, mut lt>
    %2 = kgen.struct.create(%0, %1) : !kgen.struct<(index, index)>
    kgen.return %2 : !kgen.struct<(index, index)>
  }
}

// CHECK-LABEL: lit.func @interpret
lit.func @interpret() {
  // CHECK-NEXT: <{ 42, 42 }>
  kgen.param.constant: struct<(index, index)> = <apply(:!lit.signature<("a": index) -> !kgen.struct<(index, index)>> @module::@store_load, 42)>
  kgen.return
}

lit.struct.decl @Int register_passable_trivial {
  lit.struct.field value : index
}

lit.func @ger_load(%a: !lit.declref<@Int>) -> index {
  %x = lit.var.decl "x" var : !lit.ref<@Int, mut lt>
  lit.ref.store %a, %x : <@Int, mut lt>
  %0 = lit.ref.struct.ger %x[value] : <index, mut lt> from @Int
  %1 = lit.ref.load %0 : <index, mut lt>
  kgen.return %1 : index
}

// CHECK-LABEL: lit.func @interpret_ger
lit.func @interpret_ger() {
  // CHECK-NEXT: constant = <42>
  kgen.param.constant = <apply(:!lit.signature<("a": !lit.declref<@Int>) -> index> @ger_load, { 42 })>
  kgen.return
}
