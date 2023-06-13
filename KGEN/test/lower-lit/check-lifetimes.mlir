// RUN: kgen-opt %s -split-input-file -check-lifetimes -verify-diagnostics | FileCheck %s

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

// -----

lit.struct.decl @S attributes {destructor = #kgen.symbol.constant<@S::@"__del__" > : !kgen.signature<(!pop.pointer<@S> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : index
  lit.func @__init__(%self: !pop.pointer<@S> init_self) -> !lit.none {
    %0 = lit.struct.gep %self[a] : <index> from <@S>
    %idx1 = index.constant 1
    pop.store %idx1, %0 : !pop.pointer<index>
    %none = kgen.param.constant: !lit.none = <#lit.none>
    kgen.return %none : !lit.none
  }
  lit.func @__del__(%self: !pop.pointer<@S> owned_in_mem) -> !lit.none {
    %none = kgen.param.constant: !lit.none = <#lit.none>
    kgen.return %none : !lit.none
  }
}

lit.func @verify_destructor_post_throw() -> !lit.none {
  lit.try {
    %x = lit.varlet.decl "x", var = false, synth = false : <@S>
    %1 = kgen.call @foo(%x) : (!pop.pointer<@S> byref_result) throws -> !pop.variant<@Error, !lit.none>
    // CHECK: %[[VAR0:.*]] = lit.handle_variant %1, %x : (!pop.variant<@Error, !lit.none>, !pop.pointer<@S>) -> !lit.none {
    // CHECK: %[[VAR1:.*]] = pop.variant.get %1 : !pop.variant<@Error, !lit.none> as !lit.none
    // CHECK: lit.yield %[[VAR1]] : !lit.none
    // CHECK: } else {
    // CHECK: %[[VAR2:.*]] = pop.variant.get %1 : !pop.variant<@Error, !lit.none> as !kgen.declref<@Error>
    // CHECK: lit.try.raise %[[VAR2]] : !kgen.declref<@Error>
    // CHECK: }
    %2 = lit.handle_variant %1, %x: (!pop.variant<@Error, !lit.none>, !pop.pointer<@S>) -> !lit.none {
      %4 = pop.variant.get %1 : !pop.variant<@Error, !lit.none> as !lit.none
      lit.yield %4 : !lit.none
    } else {
      %4 = pop.variant.get %1 : !pop.variant<@Error, !lit.none> as !kgen.declref<@Error>
      lit.try.raise %4 : !kgen.declref<@Error>
    }
    // CHECK: kgen.call @S::@__del__(%x) : (!pop.pointer<@S> owned_in_mem) -> !lit.none
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  } except (%arg0: !kgen.declref<@Error>) {
    lit.try.yield
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  lit.return %0 : !lit.none
  lit.end_func
}

lit.func @verify_callee_destroys(%c: i1) -> !lit.none {
  %s = lit.varlet.decl "s", var = false, synth = false : <@S>
  %2 = kgen.call @S::@__init__(%s) : (!pop.pointer<@S> init_self) -> !lit.none
  lit.try {
    hlcf.if %c {
      %5 = kgen.call @mightThrow() : () throws -> !pop.variant<@Error, !lit.none>
  	  %6 = lit.handle_variant %5 : (!pop.variant<@Error, !lit.none>) -> !lit.none {
        %10 = pop.variant.get %5 : !pop.variant<@Error, !lit.none> as !lit.none
        lit.yield %10 : !lit.none
  	  } else {
  	    // CHECK: %[[VAR0:.*]] = kgen.call @S::@__del__(%s) : (!pop.pointer<@S> owned_in_mem) -> !lit.none
  	    // CHECK-NEXT: %[[VAR1:.*]] = pop.variant.get %2 : !pop.variant<@Error, !lit.none> as !kgen.declref<@Error>
        %10 = pop.variant.get %5 : !pop.variant<@Error, !lit.none> as !kgen.declref<@Error>
        lit.try.raise %10 : !kgen.declref<@Error>
      }
      %7 = lit.struct.gep %s[a] : <index> from <@S>
      // CHECK: %[[VAR2:.*]] = pop.load %4 : !pop.pointer<index>
      // CHECK-NEXT: %[[VAR3:.*]] = kgen.call @S::@__del__(%s) : (!pop.pointer<@S> owned_in_mem) -> !lit.none
      %8 = pop.load %7 : !pop.pointer<index>
      %9 = kgen.call @print(%8) : (index) -> !lit.none
  	  hlcf.yield
    } else {
  	  hlcf.yield
	}
    lit.try.yield
  } except (%arg0: !kgen.declref<@Error>) {
    lit.try.yield
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  %3 = kgen.param.constant: !lit.none = <#lit.none>
  lit.return %3 : !lit.none
  lit.end_func
}
