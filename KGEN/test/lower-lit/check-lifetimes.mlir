// RUN: kgen-opt %s -check-lifetimes -verify-diagnostics | FileCheck %s

lit.file_module @check_lifetimes {
  // struct Struct:
  lit.struct.decl @Struct {
    //   var a: __mlir_type.index
    lit.struct.field a : index

    //   fn __init__(inout self: Self):
    //     self.a = 1
    lit.func @"__init__check_lifetimes:Struct=&)"(%self: !pop.pointer<@check_lifetimes::@Struct> init_self) -> !lit.none attributes {isStatic} {
      %0 = lit.struct.gep %self[a] : <index> from <@check_lifetimes::@Struct>
      %idx1 = index.constant 1
      pop.store %idx1, %0 : !pop.pointer<index>

      %none = kgen.param.constant: !lit.none = <#lit.none>
      kgen.return %none : !lit.none
    }

    // fn __copyinit__(inout self, existing: Self):
    lit.func @__copyinit__(
        %self: !pop.pointer<@check_lifetimes::@Struct> init_self,
        %existing: !pop.pointer<@check_lifetimes::@Struct> borrow_in_mem) -> !lit.none {
      %0 = lit.struct.gep %existing[a] : <index> from <@check_lifetimes::@Struct>
      %1 = pop.load %0 : !pop.pointer<index>
      %2 = lit.struct.gep %self[a] : <index> from <@check_lifetimes::@Struct>
      pop.store %1, %2 : !pop.pointer<index>
      %none = kgen.param.constant: !lit.none = <#lit.none>
      kgen.return %none : !lit.none
    }

    // fn __del__(owned self): pass
    lit.func @__del__(%self: !pop.pointer<@check_lifetimes::@Struct> owned_in_mem) -> !lit.none {
      %none = kgen.param.constant: !lit.none = <#lit.none>
      kgen.return %none : !lit.none
    }
  }

  // fn useDtor(a: Struct, owned b: Struct):

  // CHECK-LABEL: lit.func @useDtor
  lit.func @useDtor(
    %a: !pop.pointer<@check_lifetimes::@Struct> borrow_in_mem,
    %b: !pop.pointer<@check_lifetimes::@Struct> owned_in_mem) -> !lit.none {

    // b.a = 42
    // CHECK-NEXT: %0 = lit.struct.gep %b[a]
    %b_a = lit.struct.gep %b[a] : <index> from <@check_lifetimes::@Struct>
    %idx42 = index.constant 42
    pop.store %idx42, %b_a : !pop.pointer<index>


    // var c = Struct()
    // expected-warning @+1 {{'c' was declared as a 'var' but never mutated, consider switching to a 'let'}}
    %c = lit.varlet.decl "c", var = true, synth = false : <@check_lifetimes::@Struct>
    %0 = kgen.call @check_lifetimes::@Struct::@"__init__check_lifetimes:Struct=&)"(%c) : (!pop.pointer<@check_lifetimes::@Struct> byref_result) -> !lit.none

    %none = kgen.param.constant: !lit.none = <#lit.none>
    kgen.return %none : !lit.none
  }

  // fn indirectCall(a: Struct):
  lit.func @indirectCall(%a: !pop.pointer<@check_lifetimes::@Struct> borrow_in_mem) {
    // @noncapturing fn byrefResultFn(x: Struct) -> Struct:
    kgen.param.declare.region byrefResultFn = (
        %result: !pop.pointer<@check_lifetimes::@Struct> byref_result,
        %x: !pop.pointer<@check_lifetimes::@Struct> borrow_in_mem) {
      kgen.call @check_lifetimes::@Struct::@__copyinit__(%result, %x)
          : (!pop.pointer<@check_lifetimes::@Struct> byref_result,
             !pop.pointer<@check_lifetimes::@Struct> borrow_in_mem) -> ()
      kgen.return
    }

    // var c = byrefResultFn(x)
    %callee = kgen.create_closure[(
        !pop.pointer<@check_lifetimes::@Struct> byref_result,
        !pop.pointer<@check_lifetimes::@Struct> borrow_in_mem) -> (): byrefResultFn]()
    %c = lit.varlet.decl "c", var = true, synth = false : <@check_lifetimes::@Struct>
    kgen.call_signature %callee(%c, %a) :
        (!pop.pointer<@check_lifetimes::@Struct> byref_result,
         !pop.pointer<@check_lifetimes::@Struct> borrow_in_mem) -> ()

    %0 = lit.struct.gep %c[a] : <index> from <@check_lifetimes::@Struct>
    pop.load %0 : !pop.pointer<index>

    kgen.return
  }
}
