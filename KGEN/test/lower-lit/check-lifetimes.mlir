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

// -----

// COM: Test initialized fields are destroyed before error return.

lit.struct.decl @Error {
  lit.struct.field a : index
}

lit.struct.decl @S attributes {destructor = #kgen.symbol.constant<@S::@__del__> : !kgen.signature<(!pop.pointer<@S> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : index
}

lit.struct.decl @DestructSome attributes {destructor = #kgen.symbol.constant<@DestructSome::@__del__> : !kgen.signature<(!pop.pointer<@DestructSome> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : !kgen.declref<@S>
  lit.struct.field stole : !kgen.declref<@S>
  lit.struct.field uninitialized : !kgen.declref<@S>
  lit.struct.field byinit: !kgen.declref<@S>
  lit.struct.field register : index

  lit.func @__init__(%self: !pop.pointer<@DestructSome> init_self, %cond: i1,
                     %x: !pop.pointer<@S> borrow_in_mem,
                     %y: !pop.pointer<@S> borrow_in_mem,
                     %takeMe: !pop.pointer<@S> owned_in_mem,
                     %reg: index
                     ) throws -> !pop.variant<@Error, !lit.none> {
    %0 = lit.struct.gep %self[a] : <@S> from <@DestructSome>
    %1 = kgen.call @S::@__copyinit__(%0, %x) : (!pop.pointer<@S> init_self, !pop.pointer<@S> borrow_in_mem) -> !lit.none

    %100 = lit.struct.gep %self[register] : <index> from <@DestructSome>
    pop.store %reg, %100 : !pop.pointer<index>

    %103 = lit.struct.gep %self[stole] : <@S> from <@DestructSome>
    %104 = kgen.call @S::@__moveinit__(%103, %takeMe) : (!pop.pointer<@S> init_self, !pop.pointer<@S> owned_in_mem) -> !lit.none

    %105 = lit.struct.gep %self[byinit] : <@S> from <@DestructSome>
    %106 = kgen.call @S::@__init__(%105) : (!pop.pointer<@S> init_self) -> !lit.none
    // CHECK: hlcf.if %cond {
    // CHECK-NEXT: [[VAR0:%.*]] = kgen.call @Error::@__init__() : () ownedresult -> !kgen.declref<@Error>
    // CHECK-NEXT: [[VAR1:%.*]] = pop.variant.create [[VAR0]] : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
    // CHECK-NEXT: [[VAR2:%.*]] = lit.struct.gep %self[a] : <@S> from <@DestructSome>
    // CHECK-NEXT: [[VAR3:%.*]] = kgen.call @S::@__del__([[VAR2]]) : (!pop.pointer<@S> owned_in_mem) -> !lit.none
    // CHECK-NEXT: [[VAR4:%.*]] = lit.struct.gep %self[stole] : <@S> from <@DestructSome>
    // CHECK-NEXT: [[VAR5:%.*]] = kgen.call @S::@__del__([[VAR4]]) : (!pop.pointer<@S> owned_in_mem) -> !lit.none
    // CHECK-NEXT: [[VAR6:%.*]] = lit.struct.gep %self[byinit] : <@S> from <@DestructSome>
    // CHECK-NEXT: [[VAR7:%.*]] = kgen.call @S::@__del__([[VAR6]]) : (!pop.pointer<@S> owned_in_mem) -> !lit.none
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
    %3 = kgen.call @S::@"__copyinit__"(%2, %y) : (!pop.pointer<@S> init_self, !pop.pointer<@S> borrow_in_mem) -> !lit.none
    %none = kgen.param.constant: !lit.none = <#lit.none>
    %14 = pop.variant.create %none : !lit.none -> !pop.variant<@Error, !lit.none>
    kgen.return %14 : !pop.variant<@Error, !lit.none>
  }
}

lit.struct.decl @DestructNone attributes {destructor = #kgen.symbol.constant<@DestructNone::@__del__> : !kgen.signature<(!pop.pointer<@DestructNone> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : !kgen.declref<@S>
  lit.struct.field stole : !kgen.declref<@S>
  lit.struct.field uninitialized : !kgen.declref<@S>
  lit.struct.field register : index

  lit.func @__init__(%self: !pop.pointer<@DestructNone> init_self, %cond: i1,
                     %x: !pop.pointer<@S> borrow_in_mem,
                     %y: !pop.pointer<@S> borrow_in_mem,
                     %takeMe: !pop.pointer<@S> owned_in_mem,
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
    %1 = kgen.call @S::@__copyinit__(%0, %x) : (!pop.pointer<@S> init_self, !pop.pointer<@S> borrow_in_mem) -> !lit.none

    %100 = lit.struct.gep %self[register] : <index> from <@DestructNone>
    pop.store %reg, %100 : !pop.pointer<index>

    %103 = lit.struct.gep %self[stole] : <@S> from <@DestructNone>
    %104 = kgen.call @S::@__moveinit__(%103, %takeMe) : (!pop.pointer<@S> init_self, !pop.pointer<@S> owned_in_mem) -> !lit.none

    %2 = lit.struct.gep %self[uninitialized] : <@S> from <@DestructNone>
    %3 = kgen.call @S::@"__copyinit__"(%2, %y) : (!pop.pointer<@S> init_self, !pop.pointer<@S> borrow_in_mem) -> !lit.none
    %none = kgen.param.constant: !lit.none = <#lit.none>
    %14 = pop.variant.create %none : !lit.none -> !pop.variant<@Error, !lit.none>
    kgen.return %14 : !pop.variant<@Error, !lit.none>
  }
}

lit.struct.decl @DestructFull attributes {destructor = #kgen.symbol.constant<@DestructFull::@__del__> : !kgen.signature<(!pop.pointer<@DestructFull> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : !kgen.declref<@S>
  lit.struct.field stole : !kgen.declref<@S>
  lit.struct.field uninitialized : !kgen.declref<@S>
  lit.struct.field register : index

  lit.func @__init__(%self: !pop.pointer<@DestructFull> init_self, %cond: i1,
                     %x: !pop.pointer<@S> borrow_in_mem,
                     %y: !pop.pointer<@S> borrow_in_mem,
                     %takeMe: !pop.pointer<@S> owned_in_mem,
                     %reg: index
                     ) throws -> !pop.variant<@Error, !lit.none> {

    %0 = lit.struct.gep %self[a] : <@S> from <@DestructFull>
    %1 = kgen.call @S::@__copyinit__(%0, %x) : (!pop.pointer<@S> init_self, !pop.pointer<@S> borrow_in_mem) -> !lit.none

    %100 = lit.struct.gep %self[register] : <index> from <@DestructFull>
    pop.store %reg, %100 : !pop.pointer<index>

    %103 = lit.struct.gep %self[stole] : <@S> from <@DestructFull>
    %104 = kgen.call @S::@__moveinit__(%103, %takeMe) : (!pop.pointer<@S> init_self, !pop.pointer<@S> owned_in_mem) -> !lit.none

    %2 = lit.struct.gep %self[uninitialized] : <@S> from <@DestructFull>
    %3 = kgen.call @S::@"__copyinit__"(%2, %y) : (!pop.pointer<@S> init_self, !pop.pointer<@S> borrow_in_mem) -> !lit.none
    hlcf.if %cond {
      %12 = kgen.call @Error::@__init__() : () ownedresult -> !kgen.declref<@Error>
      %13 = pop.variant.create %12 : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
      // CHECK: %[[VAR0:.*]] = kgen.call @DestructFull::@__del__(%self) : (!pop.pointer<@DestructFull> owned_in_mem) -> !lit.none
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

lit.struct.decl @S attributes {destructor = #kgen.symbol.constant<@S::@__del__> : !kgen.signature<(!pop.pointer<@S> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : index
}

lit.struct.decl @HasMemFields attributes {destructor = #kgen.symbol.constant<@HasMemFields::@__del__> : !kgen.signature<(!pop.pointer<@HasMemFields> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : !kgen.declref<@S>
  lit.struct.field stole : !kgen.declref<@S>
  lit.struct.field uninitialized : !kgen.declref<@S>
  lit.struct.field register : index

  lit.func @__del__(%self: !pop.pointer<@HasMemFields> owned_in_mem) -> !lit.none {
    // CHECK: %[[VAR0:.*]] = lit.struct.gep %self[a] : <@S> from <@HasMemFields>
    // CHECK: %[[VAR1:.*]] = kgen.call @S::@__del__(%[[VAR0]]) : (!pop.pointer<@S> owned_in_mem) -> !lit.none
    // CHECK: %[[VAR2:.*]] = lit.struct.gep %self[stole] : <@S> from <@HasMemFields>
    // CHECK: %[[VAR3:.*]] = kgen.call @S::@__del__(%[[VAR2]]) : (!pop.pointer<@S> owned_in_mem) -> !lit.none
    // CHECK: %[[VAR4:.*]] = lit.struct.gep %self[uninitialized] : <@S> from <@HasMemFields>
    // CHECK: %[[VAR5:.*]] = kgen.call @S::@__del__(%[[VAR4]]) : (!pop.pointer<@S> owned_in_mem) -> !lit.none
    // CHECK-NOT: kgen.call @HasMemFields::@__del__(%self) : (!pop.pointer<@HasMemFields> owned_in_mem) -> !lit.none
    lit.ownership.mark.destroyed %self : !pop.pointer<@HasMemFields>
    %none = kgen.param.constant: !lit.none = <#lit.none>
    kgen.return %none : !lit.none
  }
}

// -----

// COM: Verify that initialized values are masked out of the function value set.

lit.struct.decl @MyStruct attributes {destructor = #kgen.symbol.constant<@MyStruct::@__del__ > : !kgen.signature<(!pop.pointer<@MyStruct> owned_in_mem) -> !lit.none>} {
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
    %anonymous2A = lit.varlet.decl "anonymous*", var = true, synth = true : <@MyStruct>
    %3 = kgen.call @MyStruct::@__init__(%anonymous2A) : (!pop.pointer<@MyStruct> init_self) -> !lit.none
    %6 = kgen.call @use(%anonymous2A) : (!pop.pointer<@MyStruct> borrow_in_mem) vararg -> !lit.none
    // CHECK: kgen.call @MyStruct::@__del__(%anonymous2A) : (!pop.pointer<@MyStruct> owned_in_mem) -> !lit.none
    hlcf.yield
  }
  kgen.return %1 : !lit.none
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
> : !debuginfo.unresolved<index>


lit.struct.decl @SomeData {
}

lit.struct.decl @MyStruct attributes {destructor = #kgen.symbol.constant<@MyStruct::@__del__ > : !kgen.signature<(!pop.pointer<@MyStruct> owned_in_mem) -> !lit.none>} {
  lit.struct.field str : !kgen.declref<@SomeData>
}

// CHECK: lit.func @init
lit.func @init(%self: !pop.pointer<@MyStruct> init_self) {
  // CHECK-NEXT: debuginfo.value #local_variable
  debuginfo.value #local_variable = %self : !pop.pointer<@MyStruct>
  // CHECK-NOT: __del__
  %2 = kgen.call @bar(%self) : (!pop.pointer<@MyStruct> init_self) -> !lit.none
  kgen.return
}
