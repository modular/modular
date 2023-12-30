// RUN: kgen-opt %s -split-input-file -check-lifetimes -verify-diagnostics | FileCheck %s

lit.file_module @check_lifetimes {
  // struct Struct:
  lit.struct.decl @Struct {
    //   var a: __mlir_type.index
    lit.struct.field a : index

    //   fn __init__(inout self: Self):
    //     self.a = 1
    lit.func @"__init__check_lifetimes:Struct=&)"(%self: !kgen.pointer<@check_lifetimes::@Struct> init_self) -> !kgen.none attributes {isStatic} {
      %0 = lit.struct.gep %self[a] : <index> from <@check_lifetimes::@Struct>
      %idx1 = index.constant 1
      pop.store %idx1, %0 : !kgen.pointer<index>

      %none = kgen.param.constant: none = <#kgen.none>
      kgen.return %none : !kgen.none
    }

    // fn __copyinit__(inout self, existing: Self):
    lit.func @__copyinit__(
        %self: !kgen.pointer<@check_lifetimes::@Struct> init_self,
        %existing: !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem) -> !kgen.none {
      %0 = lit.struct.gep %existing[a] : <index> from <@check_lifetimes::@Struct>
      %1 = pop.load %0 : !kgen.pointer<index>
      %2 = lit.struct.gep %self[a] : <index> from <@check_lifetimes::@Struct>
      pop.store %1, %2 : !kgen.pointer<index>
      %none = kgen.param.constant: none = <#kgen.none>
      kgen.return %none : !kgen.none
    }

    // fn __del__(owned self): pass
    lit.func @__del__[dellife](%self: !lit.ref<mut @check_lifetimes::@Struct, dellife> owned_in_mem) -> !kgen.none {
      %none = kgen.param.constant: none = <#kgen.none>
      kgen.return %none : !kgen.none
    }
  }

  // fn useDtor(a: Struct, owned b: Struct):

  // CHECK-LABEL: lit.func @useDtor
  lit.func @useDtor(
    %a: !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem,
    %b: !lit.ref<mut @check_lifetimes::@Struct, #lit.lifetime> owned_in_mem) -> !kgen.none {

    // b.a = 42
    // CHECK-NEXT: %0 = lit.ref.struct.ger %b[a]
    %b_a = lit.ref.struct.ger %b[a] : <mut index, #lit.lifetime> from <mut @check_lifetimes::@Struct, #lit.lifetime>
    %idx42 = index.constant 42
    lit.ref.store %idx42, %b_a : !lit.ref<mut index, #lit.lifetime>


    // var c = Struct()
    // expected-warning @+1 {{'c' was declared as a 'var' but never mutated, consider switching to a 'let'}}
    %c = lit.varlet.decl "c" var : !lit.ref<mut @check_lifetimes::@Struct, *"life">
    %1 = lit.ref.to_pointer %c : !lit.ref<mut @check_lifetimes::@Struct, *"life">
    %0 = lit.call @check_lifetimes::@Struct::@"__init__check_lifetimes:Struct=&)"(%1) : !lit.signature<(!kgen.pointer<@check_lifetimes::@Struct> byref_result) -> !kgen.none>

    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }

  // fn indirectCall(a: Struct):
  lit.func @indirectCall(%a: !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem) {
    // @noncapturing fn byrefResultFn(x: Struct) -> Struct:
    lit.func byrefResultFn(
        %result: !kgen.pointer<@check_lifetimes::@Struct> byref_result,
        %x: !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem) {
      lit.call @check_lifetimes::@Struct::@__copyinit__(%result, %x)
          : !lit.signature<(!kgen.pointer<@check_lifetimes::@Struct> byref_result,
                            !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem) -> !kgen.none>
      kgen.return
    }

    // var c = byrefResultFn(x)
    %callee = kgen.create_closure[!lit.signature<(
        !kgen.pointer<@check_lifetimes::@Struct> byref_result,
        !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem) -> !kgen.none>: byrefResultFn]()
    %c = lit.varlet.decl "c" var : !lit.ref<mut @check_lifetimes::@Struct, *"life">
    %1 = lit.ref.to_pointer %c : !lit.ref<mut @check_lifetimes::@Struct, *"life">
    lit.call_signature %callee(%1, %a) :
        !lit.signature<(!kgen.pointer<@check_lifetimes::@Struct> byref_result,
         !kgen.pointer<@check_lifetimes::@Struct> borrow_in_mem) -> !kgen.none>

    %0 = lit.ref.struct.ger %c[a] : !lit.ref<mut index, *"life"> from !lit.ref<mut @check_lifetimes::@Struct, *"life">
    lit.ref.load %0 : !lit.ref<mut index, *"life">

    kgen.return
  }
}

// -----

lit.struct.decl @Error {
  lit.struct.field a : index
}

lit.struct.decl @S attributes {destructor =
#kgen.symbol.constant<@S::@"__del__" > : !lit.signature<[1](!lit.ref<mut @S, *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
  lit.func @__init__(%self: !kgen.pointer<@S> init_self) -> !kgen.none {
    %0 = lit.struct.gep %self[a] : <index> from <@S>
    %idx1 = index.constant 1
    pop.store %idx1, %0 : !kgen.pointer<index>
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
  lit.func @__del__[dellife](%self: !lit.ref<mut @S, dellife> owned_in_mem) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

lit.func @verify_destructor_post_throw() -> !kgen.none {
  lit.try {
    %x = lit.varlet.decl "x" let : !lit.ref<mut @S, *"life">
    // CHECK: [[XPTR:%.*]] = lit.ref.to_pointer %x
    %xptr = lit.ref.to_pointer %x : !lit.ref<mut @S, *"life">
    // CHECK: [[V:%.*]] = lit.call @foo([[XPTR]])
    %1 = lit.call @foo(%xptr) : !lit.signature<(!kgen.pointer<@S> byref_result) throws -> !kgen.variant<@Error, none>>
    // CHECK: [[VAR0:%.*]] = lit.handle_variant [[V]], [[XPTR]] : (!kgen.variant<@Error, none>, !kgen.pointer<@S>) -> !kgen.none {
    // CHECK: [[VAR1:%.*]] = kgen.variant.get [[V]], 1 : <@Error, none>
    // CHECK: lit.yield [[VAR1]]
    // CHECK: } else {
    // CHECK: [[VAR2:%.*]] = kgen.variant.get [[V]], 0 : <@Error, none>
    // CHECK: lit.try.raise [[VAR2]]
    // CHECK: }
    %2 = lit.handle_variant %1, %xptr: (!kgen.variant<@Error, none>, !kgen.pointer<@S>) -> !kgen.none {
      %4 = kgen.variant.get %1, 1 : <@Error, none>
      lit.yield %4 : !kgen.none
    } else {
      %4 = kgen.variant.get %1, 0 : <@Error, none>
      lit.try.raise %4 : !kgen.declref<@Error>
    }
    // CHECK: lit.call @S::@__del__[life](%x) : !lit.signature<[1](!lit.ref<mut @S, *[0,0]> owned_in_mem)
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  } except (%arg0: !kgen.declref<@Error>) {
    lit.try.yield
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  %0 = kgen.param.constant: none = <#kgen.none>
  lit.return %0 : !kgen.none
  lit.end_func
}

// CHECK-LABEL: lit.func @verify_callee_destroys
lit.func @verify_callee_destroys(%c: i1) -> !kgen.none {
  %s = lit.varlet.decl "s" let : !lit.ref<mut @S, *"SLife">
  %sptr = lit.ref.to_pointer %s : !lit.ref<mut @S, *"SLife">
  // CHECK: [[PTR:%.*]] = lit.ref.to_pointer %s
  // CHECK: lit.call @S::@__init__([[PTR]])
  %2 = lit.call @S::@__init__(%sptr) : !lit.signature<(!kgen.pointer<@S> init_self) -> !kgen.none>
  lit.try {
    hlcf.if %c {
      %5 = lit.call @mightThrow() : !lit.signature<() throws -> !kgen.variant<@Error, none>>
  	  %6 = lit.handle_variant %5 : (!kgen.variant<@Error, none>) -> !kgen.none {
        %10 = kgen.variant.get %5, 1 : <@Error, none>
        lit.yield %10 : !kgen.none
  	  } else {
  	    // CHECK: [[VAR0:%.*]] = lit.call @S::@__del__[SLife](%s)
  	    // CHECK-NEXT: [[VAR1:%.*]] = kgen.variant.get
        %10 = kgen.variant.get %5, 0 : <@Error, none>
        lit.try.raise %10 : !kgen.declref<@Error>
      }
      %7 = lit.ref.struct.ger %s[a] : !lit.ref<mut index, *"SLife"> from !lit.ref<mut @S, *"SLife">
      // CHECK:  = lit.ref.load
      // CHECK-NEXT: lit.call @S::@__del__[SLife](%s)
      %8 = lit.ref.load %7 : !lit.ref<mut index, *"SLife">
      %9 = lit.call @print(%8) : !lit.signature<(index owned) -> !kgen.none>
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
  %3 = kgen.param.constant: none = <#kgen.none>
  lit.return %3 : !kgen.none
  lit.end_func
}

// -----

// COM: Test initialized fields are destroyed before error return.

lit.struct.decl @Error {
  lit.struct.field a : index
}

lit.struct.decl @S attributes {
  destructor =
    #kgen.symbol.constant<@S::@__del__> : !lit.signature<[1](!lit.ref<mut @S, *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

lit.struct.decl @DestructSome attributes {
  destructor = #kgen.symbol.constant<@DestructSome::@__del__> : !lit.signature<[1](!lit.ref<mut @DestructSome, *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : !kgen.declref<@S>
  lit.struct.field stole : !kgen.declref<@S>
  lit.struct.field uninitialized : !kgen.declref<@S>
  lit.struct.field byinit: !kgen.declref<@S>
  lit.struct.field register : index

  lit.func @__init__(%self: !lit.ref<mut @DestructSome, #lit.lifetime> init_self, %cond: i1,
                     %x: !kgen.pointer<@S> borrow_in_mem,
                     %y: !kgen.pointer<@S> borrow_in_mem,
                     %takeMe: !lit.ref<mut @S, #lit.lifetime> owned_in_mem,
                     %reg: index
                     ) throws -> !kgen.variant<@Error, none> {
    %0 = lit.ref.struct.ger %self[a] : <mut @S, #lit.lifetime> from <mut @DestructSome, #lit.lifetime>
    %1 = lit.call @S::@__copyinit__(%0, %x) : !lit.signature<(!lit.ref<mut @S, #lit.lifetime> init_self, !kgen.pointer<@S> borrow_in_mem) -> !kgen.none>

    %100 = lit.ref.struct.ger %self[register] : <mut index, #lit.lifetime> from <mut @DestructSome, #lit.lifetime>
    lit.ref.store %reg, %100 : !lit.ref<mut index, #lit.lifetime>

    %103 = lit.ref.struct.ger %self[stole] : <mut @S, #lit.lifetime> from <mut @DestructSome, #lit.lifetime>
    %104 = lit.call @S::@__moveinit__(%103, %takeMe) : !lit.signature<(!lit.ref<mut @S, #lit.lifetime> init_self, !lit.ref<mut @S, #lit.lifetime> owned_in_mem) -> !kgen.none>

    %105 = lit.ref.struct.ger %self[byinit] : <mut @S, #lit.lifetime> from <mut @DestructSome, #lit.lifetime>
    %106 = lit.call @S::@__init__(%105) : !lit.signature<(!lit.ref<mut @S, #lit.lifetime> init_self) -> !kgen.none>
    // CHECK: hlcf.if %cond {
    // CHECK-NEXT: [[VAR0:%.*]] = lit.call @Error::@__init__()
    // CHECK-NEXT: [[VAR1:%.*]] = kgen.variant.create [[VAR0]], 0 : <@Error, none>
    // CHECK-NEXT: [[VAR2:%.*]] = lit.ref.struct.ger %self[a]
    // CHECK-NEXT: [[VAR3:%.*]] = lit.call @S::@__del__{{.*}}([[VAR2]])
    // CHECK-NEXT: [[VAR4:%.*]] = lit.ref.struct.ger %self[stole]
    // CHECK-NEXT: [[VAR5:%.*]] = lit.call @S::@__del__{{.*}}([[VAR4]])
    // CHECK-NEXT: [[VAR6:%.*]] = lit.ref.struct.ger %self[byinit]
    // CHECK-NEXT: [[VAR7:%.*]] = lit.call @S::@__del__{{.*}}([[VAR6]])
    // CHECK-NEXT: lit.error_return [[VAR1]] : <@Error, none>
    // CHECK-NEXT: } else {
    // CHECK-NEXT: hlcf.yield
    // CHECK-NEXT: }
    hlcf.if %cond {
      %12 = lit.call @Error::@__init__() : !lit.signature<() ownedresult -> !kgen.declref<@Error>>
      %13 = kgen.variant.create %12, 0 : <@Error, none>
      lit.error_return %13 : !kgen.variant<@Error, none>
    } else {
      hlcf.yield
    }
    %2 = lit.ref.struct.ger %self[uninitialized] : <mut @S, #lit.lifetime> from <mut @DestructSome, #lit.lifetime>
    %3 = lit.call @S::@"__copyinit__"(%2, %y) : !lit.signature<(!lit.ref<mut @S, #lit.lifetime> init_self, !kgen.pointer<@S> borrow_in_mem) -> !kgen.none>
    %none = kgen.param.constant: none = <#kgen.none>
    %14 = kgen.variant.create %none, 1 : <@Error, none>
    kgen.return %14 : !kgen.variant<@Error, none>
  }
}

lit.struct.decl @DestructNone attributes {
    destructor = #kgen.symbol.constant<@DestructNone::@__del__> : !lit.signature<[1](!lit.ref<mut @DestructNone, *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : !kgen.declref<@S>
  lit.struct.field stole : !kgen.declref<@S>
  lit.struct.field uninitialized : !kgen.declref<@S>
  lit.struct.field register : index

  lit.func @__init__(%self: !kgen.pointer<@DestructNone> init_self, %cond: i1,
                     %x: !kgen.pointer<@S> borrow_in_mem,
                     %y: !kgen.pointer<@S> borrow_in_mem,
                     %takeMe: !lit.ref<mut @S, #lit.lifetime> owned_in_mem,
                     %reg: index
                     ) throws -> !kgen.variant<@Error, none> {
    // CHECK: hlcf.if %cond {
    // CHECK-NEXT: %[[VAR0:.*]] = lit.call @Error::@__init__()
    // CHECK-NEXT: %[[VAR1:.*]] = kgen.variant.create %[[VAR0]], 0 : <@Error, none>
    // CHECK-NEXT: lit.error_return %[[VAR1]] : <@Error, none>
    // CHECK-NEXT: } else {
    // CHECK-NEXT: hlcf.yield
    // CHECK-NEXT: }
    hlcf.if %cond {
      %12 = lit.call @Error::@__init__() : !lit.signature<() ownedresult -> !kgen.declref<@Error>>
      %13 = kgen.variant.create %12, 0 : <@Error, none>
      lit.error_return %13 : !kgen.variant<@Error, none>
    } else {
        hlcf.yield
    }
    %0 = lit.struct.gep %self[a] : <@S> from <@DestructNone>
    %1 = lit.call @S::@__copyinit__(%0, %x) : !lit.signature<(!kgen.pointer<@S> init_self, !kgen.pointer<@S> borrow_in_mem) -> !kgen.none>

    %100 = lit.struct.gep %self[register] : <index> from <@DestructNone>
    pop.store %reg, %100 : !kgen.pointer<index>

    %103 = lit.struct.gep %self[stole] : <@S> from <@DestructNone>
    %104 = lit.call @S::@__moveinit__(%103, %takeMe) : !lit.signature<(!kgen.pointer<@S> init_self, !lit.ref<mut @S, #lit.lifetime> owned_in_mem) -> !kgen.none>

    %2 = lit.struct.gep %self[uninitialized] : <@S> from <@DestructNone>
    %3 = lit.call @S::@"__copyinit__"(%2, %y) : !lit.signature<(!kgen.pointer<@S> init_self, !kgen.pointer<@S> borrow_in_mem) -> !kgen.none>
    %none = kgen.param.constant: none = <#kgen.none>
    %14 = kgen.variant.create %none, 1 : <@Error, none>
    kgen.return %14 : !kgen.variant<@Error, none>
  }
}

lit.struct.decl @DestructFull attributes {destructor = #kgen.symbol.constant<@DestructFull::@__del__> : !lit.signature<[1](!lit.ref<mut @DestructFull, #lit.lifetime> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : !kgen.declref<@S>
  lit.struct.field stole : !kgen.declref<@S>
  lit.struct.field uninitialized : !kgen.declref<@S>
  lit.struct.field register : index

  lit.func @__init__(%self: !lit.ref<mut @DestructFull, #lit.lifetime> init_self, %cond: i1,
                     %x: !kgen.pointer<@S> borrow_in_mem,
                     %y: !kgen.pointer<@S> borrow_in_mem,
                     %takeMe: !lit.ref<mut @S, #lit.lifetime> owned_in_mem,
                     %reg: index
                     ) throws -> !kgen.variant<@Error, none> {

    %0 = lit.ref.struct.ger %self[a] : <mut @S, #lit.lifetime> from <mut @DestructFull, #lit.lifetime>
    %1 = lit.call @S::@__copyinit__(%0, %x) : !lit.signature<(!lit.ref<mut @S, #lit.lifetime> init_self, !kgen.pointer<@S> borrow_in_mem) -> !kgen.none>

    %100 = lit.ref.struct.ger %self[register] : <mut index, #lit.lifetime> from <mut @DestructFull, #lit.lifetime>
    lit.ref.store %reg, %100 : !lit.ref<mut index, #lit.lifetime>

    %103 = lit.ref.struct.ger %self[stole] : <mut @S, #lit.lifetime> from <mut @DestructFull, #lit.lifetime>
    %104 = lit.call @S::@__moveinit__(%103, %takeMe) : !lit.signature<(!lit.ref<mut @S, #lit.lifetime> init_self, !lit.ref<mut @S, #lit.lifetime> owned_in_mem) -> !kgen.none>

    %2 = lit.ref.struct.ger %self[uninitialized] : <mut @S, #lit.lifetime> from <mut @DestructFull, #lit.lifetime>
    %3 = lit.call @S::@"__copyinit__"(%2, %y) : !lit.signature<(!lit.ref<mut @S, #lit.lifetime> init_self, !kgen.pointer<@S> borrow_in_mem) -> !kgen.none>
    hlcf.if %cond {
      %12 = lit.call @Error::@__init__() : !lit.signature<() ownedresult -> !kgen.declref<@Error>>
      %13 = kgen.variant.create %12, 0 : <@Error, none>
      // CHECK: %[[VAR0:.*]] = lit.call @DestructFull::@__del__{{.*}}(%self)
      lit.error_return %13 : !kgen.variant<@Error, none>
    } else {
        hlcf.yield
    }

    %none = kgen.param.constant: none = <#kgen.none>
    %14 = kgen.variant.create %none, 1 : <@Error, none>
    kgen.return %14 : !kgen.variant<@Error, none>
  }
}

// -----

// COM: Test all fields are destroyed in object destructor

lit.struct.decl @S attributes {destructor = #kgen.symbol.constant<@S::@__del__> : !lit.signature<[1](!lit.ref<mut @S, *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

lit.struct.decl @HasMemFields attributes {destructor = #kgen.symbol.constant<@HasMemFields::@__del__> : !lit.signature<[1](!lit.ref<mut @HasMemFields, *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : !kgen.declref<@S>
  lit.struct.field stole : !kgen.declref<@S>
  lit.struct.field uninitialized : !kgen.declref<@S>
  lit.struct.field register : index

  lit.func @__del__[dellife](%self: !lit.ref<mut @HasMemFields, dellife> owned_in_mem) -> !kgen.none {
    // CHECK: %[[VAR0:.*]] = lit.ref.struct.ger %self[a]
    // CHECK: %[[VAR1:.*]] = lit.call @S::@__del__[dellife](%[[VAR0]])
    // CHECK: %[[VAR2:.*]] = lit.ref.struct.ger %self[stole]
    // CHECK: %[[VAR3:.*]] = lit.call @S::@__del__[dellife](%[[VAR2]])
    // CHECK: %[[VAR4:.*]] = lit.ref.struct.ger %self[uninitialized]
    // CHECK: %[[VAR5:.*]] = lit.call @S::@__del__[dellife](%[[VAR4]])
    // CHECK-NOT: lit.call @HasMemFields::@__del__{{.*}}(%self)
    lit.ownership.mark_destroyed %self : !lit.ref<mut @HasMemFields, dellife>
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

// -----

// COM: Verify that initialized values are masked out of the function value set.

lit.struct.decl @MyStruct attributes {destructor = #kgen.symbol.constant<@MyStruct::@__del__> : !lit.signature<[1](!lit.ref<mut @MyStruct, *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}


lit.func @nestedLocalValueThatNeedsDestruct(%cond1: i1, %cond2: i1) -> !kgen.none {
  %1 = kgen.param.constant: none = <#kgen.none>
  hlcf.if %cond1 {
    kgen.return %1 : !kgen.none
  } else {
    // CHECK: hlcf.if %cond2 {
    // CHECK: kgen.return %none : !kgen.none
    // CHECK: } else {
    // CHECK: hlcf.yield
    // CHECK: }
    hlcf.if %cond2 {
      kgen.return %1 : !kgen.none
    } else {
      hlcf.yield
    }
    %anonymous2A = lit.varlet.decl "anonymous*" let : !lit.ref<mut @MyStruct, *"life"> {isSynthetic}
    %ptr = lit.ref.to_pointer %anonymous2A : !lit.ref<mut @MyStruct, *"life">
    %3 = lit.call @MyStruct::@__init__(%ptr) : !lit.signature<(!kgen.pointer<@MyStruct> init_self) -> !kgen.none>
    // CHECK: lit.call @use(
    %6 = lit.call @use(%ptr) : !lit.signature<(!kgen.pointer<@MyStruct> borrow_in_mem) -> !kgen.none>
    // CHECK: lit.call @MyStruct::@__del__[life](%anonymous2A)
    hlcf.yield
  }
  kgen.return %1 : !kgen.none
}

lit.globalvar.decl @x : !kgen.declref<@MyStruct> isVar {}, {}

// CHECK-LABEL: lit.func @byref_result_global_ref
lit.func @byref_result_global_ref() {
  // CHECK-NEXT: lit.globalvar.ref @x
  %0 = lit.globalvar.ref @x : <@MyStruct>
  // CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %0
  // CHECK-NEXT: lit.call @MyStruct::@__del__{{.*}}(%1)
  // CHECK-NEXT: call @memory_result
  lit.call @memory_result(%0) : !lit.signature<(!kgen.pointer<@MyStruct> byref_result) -> ()>
  kgen.return
}

// CHECK-LABEL: lit.func @global_ref_no_use
lit.func @global_ref_no_use() {
  // CHECK-NOT: call @MyStruct::@__del__
  %0 = lit.globalvar.ref @x : <@MyStruct>
  kgen.return
}

// -----

lit.struct.decl @MyRegStruct attributes {destructor = #kgen.symbol.constant<@MyRegStruct::@__del__> : !lit.signature<(!kgen.declref<@MyRegStruct>) -> !kgen.none>} {
  lit.struct.field a : index
}

lit.globalvar.decl @y : !kgen.declref<@MyRegStruct> isVar {}, {}

// CHECK-LABEL: lit.func @global_ref_reg_store
lit.func @global_ref_reg_store(%x: !kgen.declref<@MyRegStruct> borrow) {
  // CHECK-NEXT: %0 = lit.globalvar.ref @y
  %0 = lit.globalvar.ref @y : <@MyRegStruct>
  // CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %0
  // CHECK-NEXT: %2 = lit.ref.load %1
  // CHECK-NEXT: call @MyRegStruct::@__del__(%2)
  // CHECK-NEXT: pop.store %x, %0
  pop.store %x, %0 : !kgen.pointer<@MyRegStruct>
  kgen.return
}

// -----

// COM: Verify that we don't traverse external functions.

lit.struct.decl @MyStruct attributes {destructor = #kgen.symbol.constant<@MyStruct::@__del__ > : !lit.signature<[1](!lit.ref<mut @MyStruct, *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

// CHECK-LABEL: @external_func
// CHECK-NEXT: lit.extern_func
lit.func @external_func(%arg: !lit.ref<mut @MyStruct, #lit.lifetime> owned_in_mem) attributes {preCompiledModuleRef = @package, preElaborationName = "external_func"} {
  lit.extern_func
}

// -----

// COM: debuginfo.value ops may reference values that are not initialized (e.g.
// COM: init_self arguments in __init__ functions). We check here that this does
// COM: not cause an error in the pass.

#file = #debuginfo.file<"foo.c" in "/mlir/">
#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<!kgen.pointer<@MyStruct>>

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

lit.struct.decl @SomeData {
}

lit.struct.decl @MyStruct attributes {destructor = #kgen.symbol.constant<@MyStruct::@__del__ > : !lit.signature<[1](!lit.ref<mut @MyStruct, *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field str : !kgen.declref<@SomeData>
}

// CHECK-LABEL: lit.func @init
lit.func @init(%self: !kgen.pointer<@MyStruct> init_self) {
  // CHECK-NEXT: debuginfo.value #local_variable
  debuginfo.value #local_variable = %self : !kgen.pointer<@MyStruct> loc(#loc)
  // CHECK-NOT: __del__
  %2 = lit.call @bar(%self) : !lit.signature<(!kgen.pointer<@MyStruct> init_self) -> !kgen.none> loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)

// -----

// COM: Test that destructors are inserted for error instances.

!Error = !kgen.declref<@Error>

// CHECK-LABEL: lit.struct.decl @Error
lit.struct.decl @Error register_passable attributes {destructor = #kgen.symbol.constant<@Error::@__del__ > : !lit.signature<(!Error) -> !kgen.none>} {
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

lit.func @i_raise() throws -> !kgen.variant<!Error, index> {
  %0 = lit.call @Error::@__init__() : !lit.signature<() ownedresult -> !Error>
  %1 = kgen.variant.create %0, 0 : <!Error, index>
  lit.error_return %1 : <!Error, index>
}

// CHECK-LABEL: lit.func @eatErrorNoRef
lit.func @eatErrorNoRef() {
  lit.try {
    %3 = lit.call @i_raise() : !lit.signature<() throws -> !kgen.variant<!Error, index>>
    %4 = lit.handle_variant %3 : (!kgen.variant<!Error, index>) -> index {
      %6 = kgen.variant.get %3, 1 : <!Error, index>
      lit.yield %6 : index
    } else {
      %6 = kgen.variant.get %3, 0 : <!Error, index>
      lit.try.raise %6 : !Error
    }
    lit.try.yield
  } except (%arg0: !Error) {
    // CHECK: except
    // CHECK-NEXT: %0 = lit.call @Error::@__del__(%arg0)
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
    %3 = lit.call @i_raise() : !lit.signature<() throws -> !kgen.variant<!Error, index>>
    %4 = lit.handle_variant %3 : (!kgen.variant<!Error, index>) -> index {
      %6 = kgen.variant.get %3, 1 : <!Error, index>
      lit.yield %6 : index
    } else {
      %6 = kgen.variant.get %3, 0 : <!Error, index>
      lit.try.raise %6 : !Error
    }
    lit.try.yield
  } except (%arg0: !Error) {
    // CHECK: except
    // CHECK-NEXT: lit.call @doSomething(%arg0)
    // CHECK-NEXT: lit.call @Error::@__del__(%arg0)
    lit.call @doSomething(%arg0) : !lit.signature<("e": !Error borrow) -> !kgen.none>
    lit.try.yield
  } else {
    lit.try.yield
  }
  kgen.return
}

// -----

// COM: Check variadic arguments.

!RegPassable = !kgen.declref<@RegPassable>
lit.struct.decl @RegPassable register_passable {}

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
lit.func @"mem_only_owned(,$test::MemOnly*)"(%a: !kgen.variadic<!lit.ref<mut !MemOnly, #lit.lifetime>> owned_in_mem) vararg {
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

// -----

// COM: Copy-del elision of register-passable value, where the argument is an
// COM: owned register-passable letreg decl.

!Reg = !kgen.declref<@Reg>
lit.struct.decl @Reg register_passable attributes {
    copyInit = #kgen.symbol.constant<@Reg::@__copyinit__> : !lit.signature<(!Reg borrow) ownedresult -> !Reg>,
    destructor = #kgen.symbol.constant<@Reg::@__del__> : !lit.signature<(!Reg) -> !kgen.none>
} {
  lit.func @__del__(%self: !Reg, |) {
    kgen.return
  }
  lit.func @__copyinit__(%other: !Reg borrow) ownedresult -> !Reg attributes {specialFnKind = 7 : i8} {
    kgen.unreachable
  }
}

// CHECK-LABEL: lit.func @copy_del_reg_value
lit.func @copy_del_reg_value() {
  %0 = kgen.param.materialize: !Reg = <#lit.struct<{}>>
  %x = lit.letreg.decl "x" = %0 : !Reg
  // CHECK: call @Reg::@__del__(%x)
  %1 = lit.call @Reg::@__copyinit__(%x) : !lit.signature<(!Reg borrow, |) ownedresult -> !Reg>
  kgen.return
}

// -----

!MemType = !kgen.declref<@MemType, !lit.metatype<@MemType>>
!Error = !kgen.declref<@Error, !lit.metatype<@Error>>

lit.struct.decl @Error register_passable attributes {
  destructor = #kgen.symbol.constant<@Error::@__del__ > : !lit.signature<(!Error) -> !kgen.none>} {
  lit.struct.field a : index
  lit.func @__init__() -> !Error {
     %idx0 = index.constant 0
     %0 = lit.struct.create(a=%idx0) : (index) -> !Error
     kgen.return %0 : !Error
  }
}

lit.struct.decl @MemType attributes {destructor = #kgen.symbol.constant<@MemType::@"__del__" > : !lit.signature<[1]("self": !lit.ref<mut !MemType, *[0,0]> owned_in_mem) -> !kgen.none>}  {
  // CHECK-NOT: kgen.call @MemType::@__del__
  lit.func @i_raise(%self[self]: !kgen.pointer<!MemType> borrow_in_mem) throws -> !kgen.variant<!Error, index> {
    %0 = kgen.call @Error::@__init__() : !lit.signature<() ownedresult -> !kgen.declref<@Error, !lit.metatype<@Error>>>
    %1 = kgen.variant.create %0, 0 : <@Error : metatype<@Error>, index>
    lit.error_return %1 : <@Error : metatype<@Error>, index>
  }
}

lit.struct.decl @RegType register_passable {}

!RegType = !kgen.declref<@RegType>

lit.func @use_value(%arg0: !RegType borrow) {
  kgen.return
}

// COM: Just check that this is a valid borrow.
lit.func @sbvalue_to_mbvalue(%arg0: !RegType owned) {
  %x = lit.letreg.decl "x" = %arg0 : !RegType
  %0 = pop.stack_allocation 1 x !RegType
  lit.store.borrow %x, %0 : <!RegType>
  lit.call @use_value(%x) : !lit.signature<(!RegType) -> ()>
  kgen.return
}

lit.trait.decl @Destructable attributes {
  dtorSig = !lit.signature<[1]<regtype, !kgen.paramref<*(0,0)>>(!lit.ref<mut :!kgen.paramref<*(0,0)> *(0,1), *[0,0]>) -> !kgen.none>
} {}

// CHECK-LABEL: lit.func @destroy_generic
lit.func @destroy_generic<T: trait<@Destructable>>(%x: !lit.ref<mut :trait<@Destructable> T, #lit.lifetime> owned_in_mem) {
  // CHECK: lit.call_param[!lit.signature<[1](!lit.ref<mut :trait<@Destructable> T, *[0,0]>) -> !kgen.none>: get_type_method(:trait<@Destructable> T, "__del__")][#lit.lifetime](%x)
  kgen.return
}

// -----

// https://github.com/modularml/modular/issues/25211
lit.struct.decl @Int register_passable attributes {destructor = #kgen.symbol.constant<@Int::@__del__ > : !kgen.signature<!lit.signature<(!kgen.declref<@Int>) -> !kgen.none>>}  {
}
lit.func @y(%arg1[arg1]: !kgen.declref<@Int> borrow) {
  kgen.return
}
lit.func @x(%arg0[arg0]: !kgen.declref<@Int>) {
  // expected-warning @+1 {{'x' was declared as a 'var' but never mutated, consider switching to a 'let'}}
  %x = lit.varlet.decl "x"  var : !lit.ref<mut @Int, a>
  %0 = lit.ref.to_pointer %x : <mut @Int, a>
  // expected-error @+1 {{value 'arg0' cannot be consumed, because it is used later}}
  pop.store %arg0, %0 : !kgen.pointer<@Int>
  %1 = lit.ref.load %x : <mut @Int, a>
  %2 = kgen.call @Int::@__del__(%1) : !lit.signature<(!kgen.declref<@Int>) -> !kgen.none>
  kgen.call @y(%arg0) : (!kgen.declref<@Int> borrow) -> ()
  kgen.return
}

// -----

!Thing = !kgen.declref<@Thing>
lit.struct.decl @Box<T: trait<@Destructable>>  {
  lit.struct.field x : !kgen.paramref<:trait<@Destructable> T>
}

lit.struct.decl @Thing {
  lit.struct.field x : index
  lit.struct.field y : index
  lit.struct.field z : index
  lit.func @get(%self[self]: !kgen.pointer<!Thing> borrow_in_mem) {
    kgen.return
  }
}

lit.func @top(%c[c]: !kgen.pointer<@Box<:trait<@Destructable> !Thing>> borrow_in_mem) {
  %0 = lit.struct.gep %c[x] : <!Thing> from <@Box<:trait<@Destructable> !Thing>>
  lit.call @Thing::@get(%0) : !lit.signature<("self": !kgen.pointer<!Thing> borrow_in_mem) -> ()>
  kgen.return
}
