// RUN: kgen-opt %s -check-lifetimes -verify-diagnostics | FileCheck %s

lit.file_module @"$check_lifetimes" {
  // struct Struct:
  lit.struct.decl @Struct {
    //   var a: __mlir_type.index
    lit.struct.field a : index

    //   fn __init__(self&: Self):
    //     self.a = 1
    lit.func @"__init__($check_lifetimes::Struct=&)"(%self: !pop.pointer<@"$check_lifetimes"::@Struct> byref_result) -> !lit.none attributes {isStatic} {
      %0 = lit.struct.gep %self[a] : <index> from <@"$check_lifetimes"::@Struct>
      %idx1 = index.constant 1
      pop.store %idx1, %0 : !pop.pointer<index>

      %none = kgen.param.constant: !lit.none = <#lit.none>
      kgen.return %none : !lit.none
    }
    // fn __del___(owned self): pass
    lit.func @__del___(%self: !pop.pointer<@"$check_lifetimes"::@Struct> owned_in_mem) -> !lit.none {
      %none = kgen.param.constant: !lit.none = <#lit.none>
      kgen.return %none : !lit.none
    }
  }

  // fn useDtor(a: Struct, owned b: Struct):

  // CHECK-LABEL: lit.func @useDtor
  lit.func @useDtor(%a: !pop.pointer<@"$check_lifetimes"::@Struct> borrow_in_mem, %b: !pop.pointer<@"$check_lifetimes"::@Struct> owned_in_mem) -> !lit.none {
    // This gets removed by the check lifetimes pass.
    %b.arg = lit.owned.arg.decl "b", %b, #kgen.symbol.constant<@"$check_lifetimes"::@Struct::@__del___ > : !kgen.signature<(!pop.pointer<@"$check_lifetimes"::@Struct> owned_in_mem) -> !lit.none> : <@"$check_lifetimes"::@Struct>

    // b.a = 42
    // CHECK-NEXT: %0 = lit.struct.gep %b[a]
    %b_a = lit.struct.gep %b.arg[a] : <index> from <@"$check_lifetimes"::@Struct>
    %idx42 = index.constant 42
    pop.store %idx42, %b_a : !pop.pointer<index>


    // var c = Struct()
    %c = lit.varlet.decl "c", var = true, #kgen.symbol.constant<@"$check_lifetimes"::@Struct::@__del___ > : !kgen.signature<(!pop.pointer<@"$check_lifetimes"::@Struct> owned_in_mem) -> !lit.none> : <@"$check_lifetimes"::@Struct>
    %0 = kgen.call @"$check_lifetimes"::@Struct::@"__init__($check_lifetimes::Struct=&)"(%c) : (!pop.pointer<@"$check_lifetimes"::@Struct> byref_result) -> !lit.none


    %none = kgen.param.constant: !lit.none = <#lit.none>
    kgen.return %none : !lit.none
  }
}
