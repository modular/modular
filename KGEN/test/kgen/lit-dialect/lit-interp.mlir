// RUN: kgen-opt %s -verify-parameters='simplify=true enable-interp=true' | FileCheck %s

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

lit.func @ger_load(%a: !lit.struct<@Int>) -> index {
  %x = lit.var.decl "x" var : !lit.ref<@Int, mut lt>
  lit.ref.store %a, %x : <@Int, mut lt>
  %0 = lit.ref.struct.ger %x[value] : <index, mut lt> from @Int
  %1 = lit.ref.load %0 : <index, mut lt>
  kgen.return %1 : index
}

// CHECK-LABEL: lit.func @interpret_ger
lit.func @interpret_ger() {
  // CHECK-NEXT: constant = <42>
  kgen.param.constant = <apply(:!lit.signature<("a": !lit.struct<@Int>) -> index> @ger_load, { 42 })>
  kgen.return
}

lit.func @load_undef_var() -> index {
  %x = lit.var.decl "x" var : !lit.ref<index, mut lt>
  %0 = lit.ref.load %x : <index, mut lt>
  kgen.return %0 : index
}

// CHECK-LABEL: lit.func @interpret_undef_var
lit.func @interpret_undef_var() {
  // CHECK-NEXT: constant = <*?>
  kgen.param.constant = <apply(:!lit.signature<() -> index> @load_undef_var)>
  kgen.return
}

lit.struct.decl @Pair register_passable_trivial {
  lit.struct.field first : !lit.struct<@Int>
  lit.struct.field second : !lit.struct<@Int>
}

lit.func @load_undef_ger() -> index {
  %x = lit.var.decl "x" var : !lit.ref<@Int, mut lt>
  %0 = lit.ref.struct.ger %x[value] : <index, mut lt> from @Int
  %1 = lit.ref.load %0 : <index, mut lt>
  kgen.return %1 : index
}

// CHECK-LABEL: lit.func @interpret_undef_ger_load
lit.func @interpret_undef_ger_load() {
  // CHECK-NEXT: constant = <*?>
  kgen.param.constant = <apply(:!lit.signature<() -> index> @load_undef_ger)>
  kgen.return
}

lit.func @double_ger_store_load(%a: index) -> !lit.struct<@Int> {
  %x = lit.var.decl "x" var : !lit.ref<@Pair, mut lt>
  %0 = lit.ref.struct.ger %x[first] : <@Int, mut lt> from @Pair
  %1 = lit.ref.struct.ger %0[value] : <index, mut lt> from @Int
  lit.ref.store %a, %1 : <index, mut lt>
  %2 = lit.ref.load %0 : <@Int, mut lt>
  kgen.return %2 : !lit.struct<@Int>
}

lit.func @initialize_pair(%a: index, %b: index) -> !lit.struct<@Pair> {
  %x = lit.var.decl "x" var : !lit.ref<@Pair, mut lt>
  %0 = lit.ref.struct.ger %x[first] : <@Int, mut lt> from @Pair
  %1 = lit.ref.struct.ger %0[value] : <index, mut lt> from @Int
  lit.ref.store %a, %1 : <index, mut lt>
  %2 = lit.ref.struct.ger %x[second] : <@Int, mut lt> from @Pair
  %3 = lit.ref.struct.ger %2[value] : <index, mut lt> from @Int
  lit.ref.store %b, %3 : <index, mut lt>
  %4 = lit.ref.load %x : <@Pair, mut lt>
  kgen.return %4 : !lit.struct<@Pair>
}

// CHECK-LABEL: lit.func @interpret_ger_store
lit.func @interpret_ger_store() {
  // CHECK-NEXT: constant: @Int = <{42}>
  kgen.param.constant: @Int = <apply(:!lit.signature<("a": index) -> !lit.struct<@Int>> @double_ger_store_load, 42)>
  // CHECK-NEXT: constant: @Pair = <{first: @Int = {11}, second: @Int = {22}}>
  kgen.param.constant: @Pair = <apply(:!lit.signature<("a": index, "b": index) -> !lit.struct<@Pair>> @initialize_pair, 11, 22)>
  kgen.return
}

lit.func @partial_ger_store(%i: index) -> !lit.struct<@Pair> {
  %x = lit.var.decl "x" arg : !lit.ref<@Pair, mut lt>
  %0 = lit.ref.struct.ger %x[first] : <@Int, mut lt> from @Pair
  %1 = lit.ref.struct.ger %0[value] : <index, mut lt> from @Int
  %2 = lit.ref.struct.ger %x[second] : <@Int, mut lt> from @Pair
  %3 = lit.ref.struct.ger %2[value] : <index, mut lt> from @Int
  lit.ref.store %i, %1 : <index, mut lt>
  lit.ref.store %i, %3 : <index, mut lt>
  %4 = lit.load.consume %x : !lit.ref<@Pair, mut lt>
  kgen.return %4 : !lit.struct<@Pair>
}

// CHECK-LABEL: lit.func @interpret_partial_ger_store
lit.func @interpret_partial_ger_store() {
  // CHECK-NEXT: @Pair = <{first: @Int = {22}, second: @Int = {22}}>
  kgen.param.constant: @Pair = <apply(:!lit.signature<("i": index) -> !lit.struct<@Pair>> @partial_ger_store, 22)>
  kgen.return
}
