// RUN: kgen-opt %s -split-input-file -check-lifetimes -verify-diagnostics | FileCheck %s

lit.file_module @check_lifetimes {
  // struct Struct:
  lit.struct.decl @Struct {
    //   var a: __mlir_type.index
    lit.struct.field a : index

    //   fn __init__(inout self: Self):
    //     self.a = 1
    lit.func @"__init__check_lifetimes:Struct=&)"(%self: !kgen.pointer<@check_lifetimes::@Struct> init_self) -> !lit.none attributes {isStatic} {
      %0 = lit.struct.gep %self[a] : <index> from <@check_lifetimes::@Struct>
      %idx1 = index.constant 1
      pop.store %idx1, %0 : !kgen.pointer<index>

      %none = kgen.param.constant: !lit.none = <#lit.none>
      kgen.return %none : !lit.none
    }

    // fn __copyinit__(inout self, existing: Self):
    lit.func @__copyinit__(
        %self: !kgen.pointer<@check_lifetimes::@Struct> init_self,
        %existing: !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem) -> !lit.none {
      %0 = lit.struct.gep %existing[a] : <index> from <@check_lifetimes::@Struct>
      %1 = pop.load %0 : !kgen.pointer<index>
      %2 = lit.struct.gep %self[a] : <index> from <@check_lifetimes::@Struct>
      pop.store %1, %2 : !kgen.pointer<index>
      %none = kgen.param.constant: !lit.none = <#lit.none>
      kgen.return %none : !lit.none
    }

    // fn __del__(owned self): pass
    lit.func @__del__(%self: !kgen.pointer<@check_lifetimes::@Struct> owned_in_mem) -> !lit.none {
      %none = kgen.param.constant: !lit.none = <#lit.none>
      kgen.return %none : !lit.none
    }
  }

  // fn useDtor(a: Struct, owned b: Struct):

  // CHECK-LABEL: lit.func @useDtor
  lit.func @useDtor(
    %a: !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem,
    %b: !kgen.pointer<@check_lifetimes::@Struct> owned_in_mem) -> !lit.none {

    // b.a = 42
    // CHECK-NEXT: %0 = lit.struct.gep %b[a]
    %b_a = lit.struct.gep %b[a] : <index> from <@check_lifetimes::@Struct>
    %idx42 = index.constant 42
    pop.store %idx42, %b_a : !kgen.pointer<index>


    // var c = Struct()
    // expected-warning @+1 {{'c' was declared as a 'var' but never mutated, consider switching to a 'let'}}
    %c = lit.varlet.decl "c" var : !lit.ref<mut @check_lifetimes::@Struct, *"life">
    %1 = lit.ref.to_pointer %c : !lit.ref<mut @check_lifetimes::@Struct, *"life">
    %0 = kgen.call @check_lifetimes::@Struct::@"__init__check_lifetimes:Struct=&)"(%1) : (!kgen.pointer<@check_lifetimes::@Struct> byref_result) -> !lit.none

    %none = kgen.param.constant: !lit.none = <#lit.none>
    kgen.return %none : !lit.none
  }

  // fn indirectCall(a: Struct):
  lit.func @indirectCall(%a: !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem) {
    // @noncapturing fn byrefResultFn(x: Struct) -> Struct:
    lit.func byrefResultFn(
        %result: !kgen.pointer<@check_lifetimes::@Struct> byref_result,
        %x: !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem) {
      kgen.call @check_lifetimes::@Struct::@__copyinit__(%result, %x)
          : (!kgen.pointer<@check_lifetimes::@Struct> byref_result,
             !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem) -> ()
      kgen.return
    }

    // var c = byrefResultFn(x)
    %callee = kgen.create_closure[(
        !kgen.pointer<@check_lifetimes::@Struct> byref_result,
        !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem) -> (): byrefResultFn]()
    %c = lit.varlet.decl "c" var : !lit.ref<mut @check_lifetimes::@Struct, *"life">
    %1 = lit.ref.to_pointer %c : !lit.ref<mut @check_lifetimes::@Struct, *"life">
    kgen.call_signature %callee(%1, %a) :
        (!kgen.pointer<@check_lifetimes::@Struct> byref_result,
         !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem) -> ()

    %0 = lit.ref.struct.ger %c[a] : !lit.ref<mut index, *"life"> from !lit.ref<mut @check_lifetimes::@Struct, *"life">
    lit.ref.load %0 : !lit.ref<mut index, *"life">

    kgen.return
  }
}

// -----

lit.struct.decl @Error {
  lit.struct.field a : index
}

lit.struct.decl @S attributes {destructor = #kgen.symbol.constant<@S::@"__del__" > : !kgen.signature<(!kgen.pointer<@S> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : index
  lit.func @__init__(%self: !kgen.pointer<@S> init_self) -> !lit.none {
    %0 = lit.struct.gep %self[a] : <index> from <@S>
    %idx1 = index.constant 1
    pop.store %idx1, %0 : !kgen.pointer<index>
    %none = kgen.param.constant: !lit.none = <#lit.none>
    kgen.return %none : !lit.none
  }
  lit.func @__del__(%self: !kgen.pointer<@S> owned_in_mem) -> !lit.none {
    %none = kgen.param.constant: !lit.none = <#lit.none>
    kgen.return %none : !lit.none
  }
}

lit.func @verify_destructor_post_throw() -> !lit.none {
  lit.try {
    %x = lit.varlet.decl "x" : !lit.ref<mut @S, *"life">
    %xptr = lit.ref.to_pointer %x : !lit.ref<mut @S, *"life">
    // CHECK: [[V:%.*]] = kgen.call @foo(%1)
    %1 = kgen.call @foo(%xptr) : (!kgen.pointer<@S> byref_result) throws -> !pop.variant<@Error, !lit.none>
    // CHECK: [[VAR0:%.*]] = lit.handle_variant [[V]], %1 : (!pop.variant<@Error, !lit.none>, !kgen.pointer<@S>) -> !lit.none {
    // CHECK: [[VAR1:%.*]] = pop.variant.get [[V]] : !pop.variant<@Error, !lit.none> as !lit.none
    // CHECK: lit.yield [[VAR1]]
    // CHECK: } else {
    // CHECK: [[VAR2:%.*]] = pop.variant.get [[V]] : !pop.variant<@Error, !lit.none> as !kgen.declref<@Error>
    // CHECK: lit.try.raise [[VAR2]]
    // CHECK: }
    %2 = lit.handle_variant %1, %xptr: (!pop.variant<@Error, !lit.none>, !kgen.pointer<@S>) -> !lit.none {
      %4 = pop.variant.get %1 : !pop.variant<@Error, !lit.none> as !lit.none
      lit.yield %4 : !lit.none
    } else {
      %4 = pop.variant.get %1 : !pop.variant<@Error, !lit.none> as !kgen.declref<@Error>
      lit.try.raise %4 : !kgen.declref<@Error>
    }
    // CHECK: kgen.call @S::@__del__({{.*}}) : (!kgen.pointer<@S> owned_in_mem) -> !lit.none
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

// CHECK-LABEL: lit.func @verify_callee_destroys
lit.func @verify_callee_destroys(%c: i1) -> !lit.none {
  %s = lit.varlet.decl "s" : !lit.ref<mut @S, *"SLife">
  %sptr = lit.ref.to_pointer %s : !lit.ref<mut @S, *"SLife">
  // CHECK: [[PTR:%.*]] = lit.ref.to_pointer %s
  // CHECK: kgen.call @S::@__init__([[PTR]])
  %2 = kgen.call @S::@__init__(%sptr) : (!kgen.pointer<@S> init_self) -> !lit.none
  lit.try {
    hlcf.if %c {
      %5 = kgen.call @mightThrow() : () throws -> !pop.variant<@Error, !lit.none>
  	  %6 = lit.handle_variant %5 : (!pop.variant<@Error, !lit.none>) -> !lit.none {
        %10 = pop.variant.get %5 : !pop.variant<@Error, !lit.none> as !lit.none
        lit.yield %10 : !lit.none
  	  } else {
        // CHECK: [[PTR:%.*]] = lit.ref.to_pointer %s
  	    // CHECK-NEXT: [[VAR0:%.*]] = kgen.call @S::@__del__([[PTR]])
  	    // CHECK-NEXT: [[VAR1:%.*]] = pop.variant.get
        %10 = pop.variant.get %5 : !pop.variant<@Error, !lit.none> as !kgen.declref<@Error>
        lit.try.raise %10 : !kgen.declref<@Error>
      }
      %7 = lit.ref.struct.ger %s[a] : !lit.ref<mut index, *"SLife"> from !lit.ref<mut @S, *"SLife">
      // CHECK:  = lit.ref.load
      // CHECK-NEXT: [[PTR:%.*]] = lit.ref.to_pointer %s
      // CHECK-NEXT: kgen.call @S::@__del__([[PTR]])
      %8 = lit.ref.load %7 : !lit.ref<mut index, *"SLife">
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

// -----

// COM: Test initialized fields are destroyed before error return.

lit.struct.decl @Error {
  lit.struct.field a : index
}

lit.struct.decl @S attributes {destructor = #kgen.symbol.constant<@S::@__del__> : !kgen.signature<(!kgen.pointer<@S> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : index
}

lit.struct.decl @DestructSome attributes {destructor = #kgen.symbol.constant<@DestructSome::@__del__> : !kgen.signature<(!kgen.pointer<@DestructSome> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : !kgen.declref<@S>
  lit.struct.field stole : !kgen.declref<@S>
  lit.struct.field uninitialized : !kgen.declref<@S>
  lit.struct.field byinit: !kgen.declref<@S>
  lit.struct.field register : index

  lit.func @__init__(%self: !kgen.pointer<@DestructSome> init_self, %cond: i1,
                     %x: !kgen.pointer<@S> borrow_in_mem,
                     %y: !kgen.pointer<@S> borrow_in_mem,
                     %takeMe: !kgen.pointer<@S> owned_in_mem,
                     %reg: index
                     ) throws -> !pop.variant<@Error, !lit.none> {
    %0 = lit.struct.gep %self[a] : <@S> from <@DestructSome>
    %1 = kgen.call @S::@__copyinit__(%0, %x) : (!kgen.pointer<@S> init_self, !kgen.pointer<@S> borrow_in_mem) -> !lit.none

    %100 = lit.struct.gep %self[register] : <index> from <@DestructSome>
    pop.store %reg, %100 : !kgen.pointer<index>

    %103 = lit.struct.gep %self[stole] : <@S> from <@DestructSome>
    %104 = kgen.call @S::@__moveinit__(%103, %takeMe) : (!kgen.pointer<@S> init_self, !kgen.pointer<@S> owned_in_mem) -> !lit.none

    %105 = lit.struct.gep %self[byinit] : <@S> from <@DestructSome>
    %106 = kgen.call @S::@__init__(%105) : (!kgen.pointer<@S> init_self) -> !lit.none
    // CHECK: hlcf.if %cond {
    // CHECK-NEXT: [[VAR0:%.*]] = kgen.call @Error::@__init__() : () ownedresult -> !kgen.declref<@Error>
    // CHECK-NEXT: [[VAR1:%.*]] = pop.variant.create [[VAR0]] : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
    // CHECK-NEXT: [[VAR2:%.*]] = lit.struct.gep %self[a] : <@S> from <@DestructSome>
    // CHECK-NEXT: [[VAR3:%.*]] = kgen.call @S::@__del__([[VAR2]]) : (!kgen.pointer<@S> owned_in_mem) -> !lit.none
    // CHECK-NEXT: [[VAR4:%.*]] = lit.struct.gep %self[stole] : <@S> from <@DestructSome>
    // CHECK-NEXT: [[VAR5:%.*]] = kgen.call @S::@__del__([[VAR4]]) : (!kgen.pointer<@S> owned_in_mem) -> !lit.none
    // CHECK-NEXT: [[VAR6:%.*]] = lit.struct.gep %self[byinit] : <@S> from <@DestructSome>
    // CHECK-NEXT: [[VAR7:%.*]] = kgen.call @S::@__del__([[VAR6]]) : (!kgen.pointer<@S> owned_in_mem) -> !lit.none
    // CHECK-NEXT: lit.error_return [[VAR1]] : <@Error, !lit.none>
    // CHECK-NEXT: } else {
    // CHECK-NEXT: hlcf.yield
    // CHECK-NEXT: }
    hlcf.if %cond {
      %12 = kgen.call @Error::@__init__() : () ownedresult -> !kgen.declref<@Error>
      %13 = pop.variant.create %12 : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
      lit.error_return %13 : !pop.variant<@Error, !lit.none>
    } else {
      hlcf.yield
    }
    %2 = lit.struct.gep %self[uninitialized] : <@S> from <@DestructSome>
    %3 = kgen.call @S::@"__copyinit__"(%2, %y) : (!kgen.pointer<@S> init_self, !kgen.pointer<@S> borrow_in_mem) -> !lit.none
    %none = kgen.param.constant: !lit.none = <#lit.none>
    %14 = pop.variant.create %none : !lit.none -> !pop.variant<@Error, !lit.none>
    kgen.return %14 : !pop.variant<@Error, !lit.none>
  }
}

lit.struct.decl @DestructNone attributes {destructor = #kgen.symbol.constant<@DestructNone::@__del__> : !kgen.signature<(!kgen.pointer<@DestructNone> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : !kgen.declref<@S>
  lit.struct.field stole : !kgen.declref<@S>
  lit.struct.field uninitialized : !kgen.declref<@S>
  lit.struct.field register : index

  lit.func @__init__(%self: !kgen.pointer<@DestructNone> init_self, %cond: i1,
                     %x: !kgen.pointer<@S> borrow_in_mem,
                     %y: !kgen.pointer<@S> borrow_in_mem,
                     %takeMe: !kgen.pointer<@S> owned_in_mem,
                     %reg: index
                     ) throws -> !pop.variant<@Error, !lit.none> {
    // CHECK: hlcf.if %cond {
    // CHECK-NEXT: %[[VAR0:.*]] = kgen.call @Error::@__init__() : () ownedresult -> !kgen.declref<@Error>
    // CHECK-NEXT: %[[VAR1:.*]] = pop.variant.create %[[VAR0]] : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
    // CHECK-NEXT: lit.error_return %[[VAR1]] : <@Error, !lit.none>
    // CHECK-NEXT: } else {
    // CHECK-NEXT: hlcf.yield
    // CHECK-NEXT: }
    hlcf.if %cond {
      %12 = kgen.call @Error::@__init__() : () ownedresult -> !kgen.declref<@Error>
      %13 = pop.variant.create %12 : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
      lit.error_return %13 : !pop.variant<@Error, !lit.none>
    } else {
        hlcf.yield
    }
    %0 = lit.struct.gep %self[a] : <@S> from <@DestructNone>
    %1 = kgen.call @S::@__copyinit__(%0, %x) : (!kgen.pointer<@S> init_self, !kgen.pointer<@S> borrow_in_mem) -> !lit.none

    %100 = lit.struct.gep %self[register] : <index> from <@DestructNone>
    pop.store %reg, %100 : !kgen.pointer<index>

    %103 = lit.struct.gep %self[stole] : <@S> from <@DestructNone>
    %104 = kgen.call @S::@__moveinit__(%103, %takeMe) : (!kgen.pointer<@S> init_self, !kgen.pointer<@S> owned_in_mem) -> !lit.none

    %2 = lit.struct.gep %self[uninitialized] : <@S> from <@DestructNone>
    %3 = kgen.call @S::@"__copyinit__"(%2, %y) : (!kgen.pointer<@S> init_self, !kgen.pointer<@S> borrow_in_mem) -> !lit.none
    %none = kgen.param.constant: !lit.none = <#lit.none>
    %14 = pop.variant.create %none : !lit.none -> !pop.variant<@Error, !lit.none>
    kgen.return %14 : !pop.variant<@Error, !lit.none>
  }
}

lit.struct.decl @DestructFull attributes {destructor = #kgen.symbol.constant<@DestructFull::@__del__> : !kgen.signature<(!kgen.pointer<@DestructFull> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : !kgen.declref<@S>
  lit.struct.field stole : !kgen.declref<@S>
  lit.struct.field uninitialized : !kgen.declref<@S>
  lit.struct.field register : index

  lit.func @__init__(%self: !kgen.pointer<@DestructFull> init_self, %cond: i1,
                     %x: !kgen.pointer<@S> borrow_in_mem,
                     %y: !kgen.pointer<@S> borrow_in_mem,
                     %takeMe: !kgen.pointer<@S> owned_in_mem,
                     %reg: index
                     ) throws -> !pop.variant<@Error, !lit.none> {

    %0 = lit.struct.gep %self[a] : <@S> from <@DestructFull>
    %1 = kgen.call @S::@__copyinit__(%0, %x) : (!kgen.pointer<@S> init_self, !kgen.pointer<@S> borrow_in_mem) -> !lit.none

    %100 = lit.struct.gep %self[register] : <index> from <@DestructFull>
    pop.store %reg, %100 : !kgen.pointer<index>

    %103 = lit.struct.gep %self[stole] : <@S> from <@DestructFull>
    %104 = kgen.call @S::@__moveinit__(%103, %takeMe) : (!kgen.pointer<@S> init_self, !kgen.pointer<@S> owned_in_mem) -> !lit.none

    %2 = lit.struct.gep %self[uninitialized] : <@S> from <@DestructFull>
    %3 = kgen.call @S::@"__copyinit__"(%2, %y) : (!kgen.pointer<@S> init_self, !kgen.pointer<@S> borrow_in_mem) -> !lit.none
    hlcf.if %cond {
      %12 = kgen.call @Error::@__init__() : () ownedresult -> !kgen.declref<@Error>
      %13 = pop.variant.create %12 : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
      // CHECK: %[[VAR0:.*]] = kgen.call @DestructFull::@__del__(%self) : (!kgen.pointer<@DestructFull> owned_in_mem) -> !lit.none
      lit.error_return %13 : !pop.variant<@Error, !lit.none>
    } else {
        hlcf.yield
    }

    %none = kgen.param.constant: !lit.none = <#lit.none>
    %14 = pop.variant.create %none : !lit.none -> !pop.variant<@Error, !lit.none>
    kgen.return %14 : !pop.variant<@Error, !lit.none>
  }
}

// -----

// COM: Test all fields are destroyed in object destructor

lit.struct.decl @S attributes {destructor = #kgen.symbol.constant<@S::@__del__> : !kgen.signature<(!kgen.pointer<@S> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : index
}

lit.struct.decl @HasMemFields attributes {destructor = #kgen.symbol.constant<@HasMemFields::@__del__> : !kgen.signature<(!kgen.pointer<@HasMemFields> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : !kgen.declref<@S>
  lit.struct.field stole : !kgen.declref<@S>
  lit.struct.field uninitialized : !kgen.declref<@S>
  lit.struct.field register : index

  lit.func @__del__(%self: !kgen.pointer<@HasMemFields> owned_in_mem) -> !lit.none {
    // CHECK: %[[VAR0:.*]] = lit.struct.gep %self[a] : <@S> from <@HasMemFields>
    // CHECK: %[[VAR1:.*]] = kgen.call @S::@__del__(%[[VAR0]]) : (!kgen.pointer<@S> owned_in_mem) -> !lit.none
    // CHECK: %[[VAR2:.*]] = lit.struct.gep %self[stole] : <@S> from <@HasMemFields>
    // CHECK: %[[VAR3:.*]] = kgen.call @S::@__del__(%[[VAR2]]) : (!kgen.pointer<@S> owned_in_mem) -> !lit.none
    // CHECK: %[[VAR4:.*]] = lit.struct.gep %self[uninitialized] : <@S> from <@HasMemFields>
    // CHECK: %[[VAR5:.*]] = kgen.call @S::@__del__(%[[VAR4]]) : (!kgen.pointer<@S> owned_in_mem) -> !lit.none
    // CHECK-NOT: kgen.call @HasMemFields::@__del__(%self) : (!kgen.pointer<@HasMemFields> owned_in_mem) -> !lit.none
    lit.ownership.mark_destroyed %self : !kgen.pointer<@HasMemFields>
    %none = kgen.param.constant: !lit.none = <#lit.none>
    kgen.return %none : !lit.none
  }
}

// -----

// COM: Verify that initialized values are masked out of the function value set.

lit.struct.decl @MyStruct attributes {destructor = #kgen.symbol.constant<@MyStruct::@__del__> : !kgen.signature<(!kgen.pointer<@MyStruct> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : index
}


lit.func @nestedLocalValueThatNeedsDestruct(%cond1: i1, %cond2: i1) -> !lit.none {
  %1 = kgen.param.constant: !lit.none = <#lit.none>
  hlcf.if %cond1 {
    kgen.return %1 : !lit.none
  } else {
    // CHECK: hlcf.if %cond2 {
    // CHECK: kgen.return %0 : !lit.none
    // CHECK: } else {
    // CHECK: hlcf.yield
    // CHECK: }
    hlcf.if %cond2 {
      kgen.return %1 : !lit.none
    } else {
      hlcf.yield
    }
    %anonymous2A = lit.varlet.decl "anonymous*" var synth : !lit.ref<mut @MyStruct, *"life">
    %ptr = lit.ref.to_pointer %anonymous2A : !lit.ref<mut @MyStruct, *"life">
    %3 = kgen.call @MyStruct::@__init__(%ptr) : (!kgen.pointer<@MyStruct> init_self) -> !lit.none
    // CHECK: kgen.call @use(
    %6 = kgen.call @use(%ptr) : (!kgen.pointer<@MyStruct> borrow_in_mem) -> !lit.none
    // CHECK: [[PTR:%.*]] = lit.ref.to_pointer %anonymous2A
    // CHECK: kgen.call @MyStruct::@__del__([[PTR]])
    hlcf.yield
  }
  kgen.return %1 : !lit.none
}

lit.globalvar.decl @x : !kgen.declref<@MyStruct> isVar {}, {}

// CHECK-LABEL: lit.func @byref_result_global_ref
lit.func @byref_result_global_ref() {
  // CHECK-NEXT: lit.globalvar.ref @x
  %0 = lit.globalvar.ref @x : <@MyStruct>
  // CHECK-NEXT: call @MyStruct::@__del__
  // CHECK-NEXT: call @memory_result
  kgen.call @memory_result(%0) : !lit.signature<(!kgen.pointer<@MyStruct> byref_result) -> ()>
  kgen.return
}

// CHECK-LABEL: lit.func @global_ref_no_use
lit.func @global_ref_no_use() {
  // CHECK-NOT: call @MyStruct::@__del__
  %0 = lit.globalvar.ref @x : <@MyStruct>
  kgen.return
}

// -----

lit.struct.decl @MyRegStruct attributes {destructor = #kgen.symbol.constant<@MyRegStruct::@__del__> : !kgen.signature<(!kgen.declref<@MyRegStruct>) -> !lit.none>} {
  lit.struct.field a : index
}

lit.globalvar.decl @y : !kgen.declref<@MyRegStruct> isVar {}, {}

// CHECK-LABEL: lit.func @global_ref_reg_store
lit.func @global_ref_reg_store(%x: !kgen.declref<@MyRegStruct> borrow) {
  // CHECK-NEXT: %0 = lit.globalvar.ref @y
  %0 = lit.globalvar.ref @y : <@MyRegStruct>
  // CHECK-NEXT: %1 = pop.load %0
  // CHECK-NEXT: call @MyRegStruct::@__del__(%1)
  // CHECK-NEXT: pop.store %x, %0
  pop.store %x, %0 : !kgen.pointer<@MyRegStruct>
  kgen.return
}

// -----

// COM: Verify that we don't traverse external functions.

lit.struct.decl @MyStruct attributes {destructor = #kgen.symbol.constant<@MyStruct::@__del__ > : !kgen.signature<(!kgen.pointer<@MyStruct> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : index
}

// CHECK-LABEL: @external_func
// CHECK-NEXT: lit.extern_func
lit.func @external_func(%arg: !kgen.pointer<@MyStruct> owned_in_mem) attributes {preCompiledModuleRef = @package} {
  lit.extern_func
}

// -----

// COM: debuginfo.value ops may reference values that are not initialized (e.g.
// COM: init_self arguments in __init__ functions). We check here that this does
// COM: not cause an error in the pass.

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_C,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "foo",
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 1
> : !debuginfo.unresolved<!kgen.pointer<@MyStruct>>

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

lit.struct.decl @SomeData {
}

lit.struct.decl @MyStruct attributes {destructor = #kgen.symbol.constant<@MyStruct::@__del__ > : !kgen.signature<(!kgen.pointer<@MyStruct> owned_in_mem) -> !lit.none>} {
  lit.struct.field str : !kgen.declref<@SomeData>
}

// CHECK-LABEL: lit.func @init
lit.func @init(%self: !kgen.pointer<@MyStruct> init_self) {
  // CHECK-NEXT: debuginfo.value #local_variable
  debuginfo.value #local_variable = %self : !kgen.pointer<@MyStruct> loc(#loc)
  // CHECK-NOT: __del__
  %2 = kgen.call @bar(%self) : (!kgen.pointer<@MyStruct> init_self) -> !lit.none loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)

// -----

// COM: Test that destructors are inserted for error instances.

!Error = !kgen.declref<@Error>

// CHECK-LABEL: lit.struct.decl @Error
lit.struct.decl @Error attributes {destructor = #kgen.symbol.constant<@Error::@__del__ > : !kgen.signature<(!Error) -> !lit.none>, registerPassable = 1 : i8}  {
  lit.struct.field a : index
  lit.func @__init__() -> !Error {
     %idx0 = index.constant 0
     %0 = lit.struct.create(a=%idx0) : (index) -> !Error
     kgen.return %0 : !Error
  }
}

lit.func @doSomething(%e: !Error borrow) {
   kgen.return
}

lit.func @i_raise() throws -> !pop.variant<!Error, index> {
  %0 = kgen.call @Error::@__init__() : () ownedresult -> !Error
  %1 = pop.variant.create %0 : !Error -> !pop.variant<!Error, index>
  lit.error_return %1 : <!Error, index>
}

// CHECK-LABEL: lit.func @eatErrorNoRef
lit.func @eatErrorNoRef() {
  lit.try {
    %3 = kgen.call @i_raise() : !lit.signature<() throws -> !pop.variant<!Error, index>>
    %4 = lit.handle_variant %3 : (!pop.variant<!Error, index>) -> index {
      %6 = pop.variant.get %3 : !pop.variant<!Error, index> as index
      lit.yield %6 : index
    } else {
      %6 = pop.variant.get %3 : !pop.variant<!Error, index> as !Error
      lit.try.raise %6 : !Error
    }
    lit.try.yield
  } except (%arg0: !Error) {
    // CHECK: except
    // CHECK-NEXT: %0 = kgen.call @Error::@__del__(%arg0) : (!kgen.declref<@Error>) -> !lit.none
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  } else {
    lit.try.yield
  }
  kgen.return
}

// CHECK-LABEL: lit.func @eatErrorRef
lit.func @eatErrorRef() {
  lit.try {
    %3 = kgen.call @i_raise() : !lit.signature<() throws -> !pop.variant<!Error, index>>
    %4 = lit.handle_variant %3 : (!pop.variant<!Error, index>) -> index {
      %6 = pop.variant.get %3 : !pop.variant<!Error, index> as index
      lit.yield %6 : index
    } else {
      %6 = pop.variant.get %3 : !pop.variant<!Error, index> as !Error
      lit.try.raise %6 : !Error
    }
    lit.try.yield
  } except (%arg0: !Error) {
    // CHECK: except
    // CHECK-NEXT: kgen.call @doSomething(%arg0)
    // CHECK-NEXT: kgen.call @Error::@__del__(%arg0) : (!kgen.declref<@Error>) -> !lit.none
    kgen.call @doSomething(%arg0) : !lit.signature<("e": !Error borrow) -> !lit.none>
    lit.try.yield
  } else {
    lit.try.yield
  }
  kgen.return
}

// -----

// COM: Check variadic arguments.

!RegPassable = !kgen.declref<@RegPassable>
lit.struct.decl @RegPassable attributes {registerPassable = 1 : i8}  {}

lit.func @reg_passable_owned(%a: !kgen.variadic<!RegPassable>) vararg {
  lit.end_func
}

// COM: TODO(#21861): support variadic arguments
// expected-error @below {{passing variadic arguments by reference is not supported yet (hint: pass register-passable types as `owned` or `borrowed` and memory-only types as `borrowed` if possible)}}
lit.func @reg_passable_inout(%a: !kgen.variadic<pointer<!RegPassable>> byref) vararg {
  lit.end_func
}

lit.func @"reg_passable_borrowed(,$test::RegPassable*)"(%a: !kgen.variadic<!RegPassable> borrow) vararg {
  lit.end_func
}

!MemOnly = !kgen.declref<@MemOnly>
lit.struct.decl @MemOnly {}

// COM: TODO(#21861): support variadic arguments
// expected-error @below {{passing variadic arguments of memory-only types as `owned` is not supported yet (hint: pass as `borrowed` if possible)}}
lit.func @"mem_only_owned(,$test::MemOnly*)"(%a: !kgen.variadic<pointer<!MemOnly>> owned_in_mem) vararg {
  lit.end_func
}

// COM: TODO(#21861): support variadic arguments
// expected-error @below {{passing variadic arguments by reference is not supported yet (hint: pass register-passable types as `owned` or `borrowed` and memory-only types as `borrowed` if possible)}}
lit.func @"mem_only_inout(,$test::MemOnly&*)"(%a: !kgen.variadic<pointer<!MemOnly>> byref) vararg {
  lit.end_func
}

lit.func @"mem_only_borrowed(,$test::MemOnly*)"(%a: !kgen.variadic<pointer<!MemOnly>> borrow_in_mem) vararg {
  lit.end_func
}
