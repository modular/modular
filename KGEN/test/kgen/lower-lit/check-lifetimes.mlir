// RUN: kgen-opt %s -split-input-file -check-lifetimes -verify-diagnostics | FileCheck %s


// struct Struct:
lit.struct.decl @Struct attributes {
  destructor = #kgen.symbol.constant<@Struct::@__del__> : !lit.signature<[1](!lit.ref<@Struct, mut *[0,0]> owned_in_mem) -> !kgen.none>}
{
  //   var a: __mlir_type.index
  lit.struct.field a : index

  //   fn __init__(inout self: Self):
  //     self.a = 1
  lit.func @__init__[mut selflife](%self: !lit.ref<@Struct, mut selflife> init_self) -> !kgen.none attributes {isStatic} {
    %0 = lit.ref.struct.ger %self[a] : <index, mut selflife> from @Struct
    %idx1 = index.constant 1
    lit.ref.store %idx1, %0 : !lit.ref<index, mut selflife>

    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }

  // fn __copyinit__(inout self, existing: Self):
  lit.func @__copyinit__[mut selflife, imm existinglife](
      %self: !lit.ref<@Struct, mut selflife> init_self,
      %existing: !lit.ref<@Struct, imm existinglife> borrow_in_mem) -> !kgen.none {
    %0 = lit.ref.struct.ger %existing[a] : <index, imm existinglife> from @Struct
    %1 = lit.ref.load %0 : !lit.ref<index, imm existinglife>
    %2 = lit.ref.struct.ger %self[a] : <index, mut selflife> from @Struct
    lit.ref.store %1, %2 : !lit.ref<index, mut selflife>
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }

  // fn __del__(owned self): pass
  lit.func @__del__[mut dellife](%self: !lit.ref<@Struct, mut dellife> owned_in_mem) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

// fn useDtor(a: Struct, owned b: Struct):

// CHECK-LABEL: lit.func @useDtor
lit.func @useDtor(
  %a: !lit.ref<@Struct, imm #lit.lifetime> borrow_in_mem,
  %b: !lit.ref<@Struct, mut #lit.lifetime> owned_in_mem) -> !kgen.none {

  // b.a = 42
  // CHECK-NEXT: %0 = lit.ref.struct.ger %b[a]
  %b_a = lit.ref.struct.ger %b[a] : <index, mut #lit.lifetime> from @Struct
  %idx42 = index.constant 42
  lit.ref.store %idx42, %b_a : !lit.ref<index, mut #lit.lifetime>


  // var c = Struct()
  %c = lit.var.decl "c" var : !lit.ref<@Struct, mut *"life">
  %0 = lit.call @Struct::@__init__[mut life](%c) : !lit.signature<[1](!lit.ref<@Struct, mut *[0,0]> init_self) -> !kgen.none>

  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// fn indirectCall(a: Struct):
lit.func @indirectCall(%a: !lit.ref<@Struct, imm #lit.lifetime> borrow_in_mem) {
  // @noncapturing fn byrefResultFn(x: Struct) -> Struct:
  lit.func byrefResultFn(
      %x: !lit.ref<@Struct, imm #lit.lifetime> borrow_in_mem,
      %result: !lit.ref<@Struct, mut *"life"> byref_result) {
    lit.call @Struct::@__copyinit__(%result, %x)
        : !lit.signature<(!lit.ref<@Struct, mut *"life"> init_self,
                          !lit.ref<@Struct, imm #lit.lifetime> borrow_in_mem) -> !kgen.none>
    kgen.return
  }

  // var c = byrefResultFn(x)
  %callee = kgen.create_closure[!lit.signature<(
      !lit.ref<@Struct, imm #lit.lifetime> borrow_in_mem,
      !lit.ref<@Struct, mut *"life"> byref_result) -> !kgen.none>: byrefResultFn]()
  %c = lit.var.decl "c" var : !lit.ref<@Struct, mut *"life">
  lit.call_indirect %callee(%a, %c) :
      !lit.signature<(
        !lit.ref<@Struct, imm #lit.lifetime> borrow_in_mem,
        !lit.ref<@Struct, mut *"life"> byref_result) -> !kgen.none>

  %0 = lit.ref.struct.ger %c[a] : <index, mut life> from @Struct
  lit.ref.load %0 : !lit.ref<index, mut *"life">

  kgen.return
}

// Some tests with non-trivial lifetimes references
// CHECK-LABEL: lit.func @references1
// CHECK-NOT: __del__
lit.func @references1[mut alife](%a: !lit.ref<@Struct, mut alife> owned_in_mem,
                             %cond: i1 borrow) {
  %x = lit.var.decl "x" var : !lit.ref<@Struct, mut xlife>
   // CHECK: lit.call @Struct::@__init__[mut xlife](%x)
  lit.call @Struct::@__init__[mut xlife](%x) : !lit.signature<[1](!lit.ref<@Struct, mut *[0,0]> init_self) -> !kgen.none>

  %x1 = kgen.rebind %x : !lit.ref<@Struct, mut xlife> to !lit.ref<@Struct, mut {xlife,alife}>
  %a1 = kgen.rebind %a : !lit.ref<@Struct, mut alife> to !lit.ref<@Struct, mut {xlife,alife}>

  %z = pop.select %cond, %x1, %a1 : !lit.ref<@Struct, mut {xlife,alife}>

  // This load is a use of both x and a, so their lifetimes are extended.
  // CHECK: lit.ref.load
  %result = lit.ref.load %z : !lit.ref<@Struct, mut {xlife,alife}>
  // CHECK-NEXT: lit.call @Struct::@__del__[mut alife](%a)
  // CHECK-NEXT: lit.call @Struct::@__del__[mut xlife](%x)
  kgen.return
}

// -----

lit.struct.decl @Error {
  lit.struct.field a : index
}

lit.struct.decl @S attributes {destructor =
#kgen.symbol.constant<@S::@"__del__" > : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
  lit.func @__init__(%self: !lit.ref<@S, mut #lit.lifetime> init_self) -> !kgen.none {
    %0 = lit.ref.struct.ger %self[a] : <index, mut #lit.lifetime> from @S
    %idx1 = index.constant 1
    lit.ref.store %idx1, %0 : <index, mut #lit.lifetime>
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
  lit.func @__del__[mut dellife](%self: !lit.ref<@S, mut dellife> owned_in_mem) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

lit.func @verify_destructor_post_throw() -> !kgen.none {
  lit.try {
    %x = lit.var.decl "x" var : !lit.ref<@S, mut life>
    // CHECK: [[V:%.*]] = lit.call @foo(%x)
    %1 = lit.call @foo(%x) : !lit.signature<(!lit.ref<@S, mut *"life"> byref_result) throws|ownedresult -> !kgen.variant<@Error, none>>
    // CHECK: [[VAR0:%.*]] = lit.handle_variant [[V]], %x : (!kgen.variant<@Error, none>, !lit.ref<@S, mut life>) -> !kgen.none {
    // CHECK: [[VAR1:%.*]] = kgen.variant.take [[V]], 1 : <@Error, none>
    // CHECK: lit.yield [[VAR1]]
    // CHECK: } else {
    // CHECK: [[VAR2:%.*]] = kgen.variant.take [[V]], 0 : <@Error, none>
    // CHECK: lit.try.raise [[VAR2]]
    // CHECK: }
    %2 = lit.handle_variant %1, %x: (!kgen.variant<@Error, none>, !lit.ref<@S, mut *"life">) -> !kgen.none {
      %4 = kgen.variant.take %1, 1 : <@Error, none>
      lit.yield %4 : !kgen.none
    } else {
      %4 = kgen.variant.take %1, 0 : <@Error, none>
      lit.try.raise %4 : !lit.declref<@Error>
    }
    // CHECK: lit.call @S::@__del__[mut life](%x) : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem)
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  } except (%arg0: !lit.declref<@Error>) {
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
  %s = lit.var.decl "s" var : !lit.ref<@S, mut *"SLife">
  // CHECK: lit.call @S::@__init__(%s)
  %2 = lit.call @S::@__init__(%s) : !lit.signature<(!lit.ref<@S, mut *"SLife"> init_self) -> !kgen.none>
  lit.try {
    hlcf.if %c {
      %5 = lit.call @mightThrow() : !lit.signature<() throws|ownedresult -> !kgen.variant<@Error, none>>
  	  %6 = lit.handle_variant %5 : (!kgen.variant<@Error, none>) -> !kgen.none {
        %10 = kgen.variant.take %5, 1 : <@Error, none>
        lit.yield %10 : !kgen.none
  	  } else {
  	    // CHECK: [[VAR0:%.*]] = lit.call @S::@__del__[mut SLife](%s)
  	    // CHECK-NEXT: [[VAR1:%.*]] = kgen.variant.take
        %10 = kgen.variant.take %5, 0 : <@Error, none>
        lit.try.raise %10 : !lit.declref<@Error>
      }
      %7 = lit.ref.struct.ger %s[a] : <index, mut *"SLife"> from @S
      // CHECK:  = lit.ref.load
      // CHECK-NEXT: lit.call @S::@__del__[mut SLife](%s)
      %8 = lit.ref.load %7 : !lit.ref<index, mut *"SLife">
      %9 = lit.call @print(%8) : !lit.signature<(index owned) -> !kgen.none>
  	  hlcf.yield
    } else {
  	  hlcf.yield
	  }
    lit.try.yield
  } except (%arg0: !lit.declref<@Error>) {
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
    #kgen.symbol.constant<@S::@__del__> : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

// CHECK-LABEL: lit.struct.decl @DestructSome

lit.struct.decl @DestructSome attributes {
  destructor = #kgen.symbol.constant<@DestructSome::@__del__> : !lit.signature<[1](!lit.ref<@DestructSome, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : !lit.declref<@S>
  lit.struct.field stole : !lit.declref<@S>
  lit.struct.field uninitialized : !lit.declref<@S>
  lit.struct.field byinit: !lit.declref<@S>
  lit.struct.field register : index

  lit.func @__init__(%self: !lit.ref<@DestructSome, mut #lit.lifetime> init_self, %cond: i1,
                     %x: !lit.ref<@S, imm #lit.lifetime> borrow_in_mem,
                     %y: !lit.ref<@S, imm #lit.lifetime> borrow_in_mem,
                     %takeMe: !lit.ref<@S, mut #lit.lifetime> owned_in_mem,
                     %reg: index borrow
                     ) throws|ownedresult -> !kgen.variant<@Error, none> {
    %0 = lit.ref.struct.ger %self[a] : <@S, mut #lit.lifetime> from @DestructSome
    %1 = lit.call @S::@__copyinit__(%0, %x) : !lit.signature<(!lit.ref<@S, mut #lit.lifetime> init_self, !lit.ref<@S, imm #lit.lifetime> borrow_in_mem) -> !kgen.none>

    %100 = lit.ref.struct.ger %self[register] : <index, mut #lit.lifetime> from @DestructSome
    lit.ref.store %reg, %100 : !lit.ref<index, mut #lit.lifetime>

    %103 = lit.ref.struct.ger %self[stole] : <@S, mut #lit.lifetime> from @DestructSome
    %104 = lit.call @S::@__moveinit__(%103, %takeMe) : !lit.signature<(!lit.ref<@S, mut #lit.lifetime> init_self, !lit.ref<@S, mut #lit.lifetime> owned_in_mem) -> !kgen.none>

    %105 = lit.ref.struct.ger %self[byinit] : <@S, mut #lit.lifetime> from @DestructSome
    %106 = lit.call @S::@__init__(%105) : !lit.signature<(!lit.ref<@S, mut #lit.lifetime> init_self) -> !kgen.none>
    // CHECK: hlcf.if %cond {
    // CHECK-NEXT: [[VAR2:%.*]] = lit.ref.struct.ger %self[a]
    // CHECK-NEXT: [[VAR3:%.*]] = lit.call @S::@__del__{{.*}}([[VAR2]])
    // CHECK-NEXT: [[VAR4:%.*]] = lit.ref.struct.ger %self[stole]
    // CHECK-NEXT: [[VAR5:%.*]] = lit.call @S::@__del__{{.*}}([[VAR4]])
    // CHECK-NEXT: [[VAR6:%.*]] = lit.ref.struct.ger %self[byinit]
    // CHECK-NEXT: [[VAR7:%.*]] = lit.call @S::@__del__{{.*}}([[VAR6]])
    // CHECK-NEXT: [[VAR0:%.*]] = lit.call @Error::@__init__()
    // CHECK-NEXT: [[VAR1:%.*]] = kgen.variant.create [[VAR0]], 0 : <@Error, none>
    // CHECK-NEXT: lit.error_return [[VAR1]] : <@Error, none>
    // CHECK-NEXT: } else {
    // CHECK-NEXT: hlcf.yield
    // CHECK-NEXT: }
    hlcf.if %cond {
      %12 = lit.call @Error::@__init__() : !lit.signature<() ownedresult -> !lit.declref<@Error>>
      %13 = kgen.variant.create %12, 0 : <@Error, none>
      lit.error_return %13 : !kgen.variant<@Error, none>
    } else {
      hlcf.yield
    }
    %2 = lit.ref.struct.ger %self[uninitialized] : <@S, mut #lit.lifetime> from @DestructSome
    %3 = lit.call @S::@"__copyinit__"(%2, %y) : !lit.signature<(!lit.ref<@S, mut #lit.lifetime> init_self, !lit.ref<@S, imm #lit.lifetime> borrow_in_mem) -> !kgen.none>
    %none = kgen.param.constant: none = <#kgen.none>
    %14 = kgen.variant.create %none, 1 : <@Error, none>
    kgen.return %14 : !kgen.variant<@Error, none>
  }
}

// CHECK-LABEL: lit.struct.decl @DestructNone
lit.struct.decl @DestructNone attributes {
    destructor = #kgen.symbol.constant<@DestructNone::@__del__> : !lit.signature<[1](!lit.ref<@DestructNone, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : !lit.declref<@S>
  lit.struct.field stole : !lit.declref<@S>
  lit.struct.field uninitialized : !lit.declref<@S>
  lit.struct.field register : index

  lit.func @__init__(%self: !lit.ref<@DestructNone, mut #lit.lifetime> init_self, %cond: i1,
                     %x: !lit.ref<@S, imm #lit.lifetime> borrow_in_mem,
                     %y: !lit.ref<@S, imm #lit.lifetime> borrow_in_mem,
                     %takeMe: !lit.ref<@S, mut #lit.lifetime> owned_in_mem,
                     %reg: index
                     ) throws|ownedresult -> !kgen.variant<@Error, none> {
    // CHECK: hlcf.if %cond {
    // CHECK-NEXT: lit.call @S::@__del__[mut #lit.lifetime](%takeMe)
    // CHECK-NEXT: %[[VAR0:.*]] = lit.call @Error::@__init__()
    // CHECK-NEXT: %[[VAR1:.*]] = kgen.variant.create %[[VAR0]], 0 : <@Error, none>
    // CHECK-NEXT: lit.error_return %[[VAR1]] : <@Error, none>
    // CHECK-NEXT: } else {
    // CHECK-NEXT: hlcf.yield
    // CHECK-NEXT: }
    hlcf.if %cond {
      %12 = lit.call @Error::@__init__() : !lit.signature<() ownedresult -> !lit.declref<@Error>>
      %13 = kgen.variant.create %12, 0 : <@Error, none>
      lit.error_return %13 : !kgen.variant<@Error, none>
    } else {
        hlcf.yield
    }
    %0 = lit.ref.struct.ger %self[a] : <@S, mut #lit.lifetime> from @DestructNone
    %1 = lit.call @S::@__copyinit__(%0, %x) : !lit.signature<(!lit.ref<@S, mut #lit.lifetime> init_self, !lit.ref<@S, imm #lit.lifetime> borrow_in_mem) -> !kgen.none>

    %100 = lit.ref.struct.ger %self[register] : <index, mut #lit.lifetime> from @DestructNone
    lit.ref.store %reg, %100 : <index, mut #lit.lifetime>

    %103 = lit.ref.struct.ger %self[stole] : <@S, mut #lit.lifetime> from @DestructNone
    %104 = lit.call @S::@__moveinit__(%103, %takeMe) : !lit.signature<(!lit.ref<@S, mut #lit.lifetime> init_self, !lit.ref<@S, mut #lit.lifetime> owned_in_mem) -> !kgen.none>

    %2 = lit.ref.struct.ger %self[uninitialized] : <@S, mut #lit.lifetime> from @DestructNone
    %3 = lit.call @S::@"__copyinit__"(%2, %y) : !lit.signature<(!lit.ref<@S, mut #lit.lifetime> init_self, !lit.ref<@S, imm #lit.lifetime> borrow_in_mem) -> !kgen.none>
    %none = kgen.param.constant: none = <#kgen.none>
    %14 = kgen.variant.create %none, 1 : <@Error, none>
    kgen.return %14 : !kgen.variant<@Error, none>
  }
}

// CHECK-LABEL: lit.struct.decl @DestructFull
lit.struct.decl @DestructFull attributes {destructor = #kgen.symbol.constant<@DestructFull::@__del__> : !lit.signature<[1](!lit.ref<@DestructFull, mut #lit.lifetime> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : !lit.declref<@S>
  lit.struct.field stole : !lit.declref<@S>
  lit.struct.field uninitialized : !lit.declref<@S>
  lit.struct.field register : index

  lit.func @__init__(%self: !lit.ref<@DestructFull, mut #lit.lifetime> init_self, %cond: i1,
                     %x: !lit.ref<@S, imm #lit.lifetime> borrow_in_mem,
                     %y: !lit.ref<@S, imm #lit.lifetime> borrow_in_mem,
                     %takeMe: !lit.ref<@S, mut #lit.lifetime> owned_in_mem,
                     %reg: index
                     ) throws|ownedresult -> !kgen.variant<@Error, none> {

    %0 = lit.ref.struct.ger %self[a] : <@S, mut #lit.lifetime> from @DestructFull
    %1 = lit.call @S::@__copyinit__(%0, %x) : !lit.signature<(!lit.ref<@S, mut #lit.lifetime> init_self, !lit.ref<@S, imm #lit.lifetime> borrow_in_mem) -> !kgen.none>

    %100 = lit.ref.struct.ger %self[register] : <index, mut #lit.lifetime> from @DestructFull
    lit.ref.store %reg, %100 : !lit.ref<index, mut #lit.lifetime>

    %103 = lit.ref.struct.ger %self[stole] : <@S, mut #lit.lifetime> from @DestructFull
    %104 = lit.call @S::@__moveinit__(%103, %takeMe) : !lit.signature<(!lit.ref<@S, mut #lit.lifetime> init_self, !lit.ref<@S, mut #lit.lifetime> owned_in_mem) -> !kgen.none>

    %2 = lit.ref.struct.ger %self[uninitialized] : <@S, mut #lit.lifetime> from @DestructFull
    %3 = lit.call @S::@"__copyinit__"(%2, %y) : !lit.signature<(!lit.ref<@S, mut #lit.lifetime> init_self, !lit.ref<@S, imm #lit.lifetime> borrow_in_mem) -> !kgen.none>
    // CHECK: hlcf.if %cond {
    hlcf.if %cond {
      // CHECK-NEXT: [[VAR2:%.*]] = lit.ref.struct.ger %self[a]
      // CHECK-NEXT: lit.call @S::@__del__{{.*}}([[VAR2]])
      // CHECK-NEXT: [[VAR4:%.*]] = lit.ref.struct.ger %self[stole]
      // CHECK-NEXT: lit.call @S::@__del__{{.*}}([[VAR4]])
      // CHECK-NEXT: [[VAR6:%.*]] = lit.ref.struct.ger %self[uninitialized]
      // CHECK-NEXT: lit.call @S::@__del__{{.*}}([[VAR6]])
      // CHECK-NEXT: lit.call @Error::@__init__()
      %12 = lit.call @Error::@__init__() : !lit.signature<() ownedresult -> !lit.declref<@Error>>
      // CHECK-NEXT: kgen.variant.create
      %13 = kgen.variant.create %12, 0 : <@Error, none>
      // CHECK-NEXT: lit.error_return
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

lit.struct.decl @S attributes {destructor = #kgen.symbol.constant<@S::@__del__> : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

// CHECK-LABEL: lit.struct.decl @HasMemFields
lit.struct.decl @HasMemFields attributes {destructor = #kgen.symbol.constant<@HasMemFields::@__del__> : !lit.signature<[1](!lit.ref<@HasMemFields, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : !lit.declref<@S>
  lit.struct.field stole : !lit.declref<@S>
  lit.struct.field uninitialized : !lit.declref<@S>
  lit.struct.field register : index

  lit.func @__del__[mut dellife](%self: !lit.ref<@HasMemFields, mut dellife> owned_in_mem) -> !kgen.none {
    // CHECK: %[[VAR0:.*]] = lit.ref.struct.ger %self[a]
    // CHECK: %[[VAR1:.*]] = lit.call @S::@__del__[mut dellife](%[[VAR0]])
    // CHECK: %[[VAR2:.*]] = lit.ref.struct.ger %self[stole]
    // CHECK: %[[VAR3:.*]] = lit.call @S::@__del__[mut dellife](%[[VAR2]])
    // CHECK: %[[VAR4:.*]] = lit.ref.struct.ger %self[uninitialized]
    // CHECK: %[[VAR5:.*]] = lit.call @S::@__del__[mut dellife](%[[VAR4]])
    // CHECK-NOT: lit.call @HasMemFields::@__del__{{.*}}(%self)
    lit.ownership.mark_destroyed %self : !lit.ref<@HasMemFields, mut dellife>
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

// CHECK-LABEL: lit.func @mark_initialized
lit.func @mark_initialized[mut lt](%arg: !lit.ref<@HasMemFields, mut lt> byref_result) {
  // CHECK-NEXT: lit.ownership.mark_initialized %arg
  lit.ownership.mark_initialized %arg : <@HasMemFields, mut lt>
  kgen.return
}

// -----

// COM: Verify that initialized values are masked out of the function value set.

lit.struct.decl @MyStruct attributes {destructor = #kgen.symbol.constant<@MyStruct::@__del__> : !lit.signature<[1](!lit.ref<@MyStruct, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}


// CHECK-LABEL: lit.func @nestedLocalValueThatNeedsDestruct
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
    %anonymous2A = lit.var.decl "anonymous*" synth : !lit.ref<@MyStruct, mut *"life">
    %3 = lit.call @MyStruct::@__init__(%anonymous2A) : !lit.signature<(!lit.ref<@MyStruct, mut *"life"> init_self) -> !kgen.none>
    // CHECK: lit.call @use(
    %6 = lit.call @use(%anonymous2A) : !lit.signature<(!lit.ref<@MyStruct, mut *"life"> borrow_in_mem) -> !kgen.none>
    // CHECK: lit.call @MyStruct::@__del__[mut life](%anonymous2A)
    hlcf.yield
  }
  kgen.return %1 : !kgen.none
}

lit.globalvar.decl @x : !lit.declref<@MyStruct> {}, {}

// CHECK-LABEL: lit.func @byref_result_global_ref
lit.func @byref_result_global_ref() {
  // CHECK-NEXT: %0 = lit.globalvar.ref @x
  %0 = lit.globalvar.ref @x : <@MyStruct, mut #lit.lifetime>
  // CHECK-NEXT: lit.call @MyStruct::@__del__{{.*}}(%0)
  // CHECK-NEXT: call @memory_result
  lit.call @memory_result(%0) : !lit.signature<(!lit.ref<@MyStruct, mut #lit.lifetime> byref_result) -> ()>
  kgen.return
}

// CHECK-LABEL: lit.func @global_ref_no_use
lit.func @global_ref_no_use() {
  // CHECK-NOT: call @MyStruct::@__del__
  %0 = lit.globalvar.ref @x : <@MyStruct, mut #lit.lifetime>
  kgen.return
}

// -----

lit.struct.decl @MyRegStruct attributes {destructor = #kgen.symbol.constant<@MyRegStruct::@__del__> : !lit.signature<(!lit.declref<@MyRegStruct>) -> !kgen.none>} {
  lit.struct.field a : index
}

lit.globalvar.decl @y : !lit.declref<@MyRegStruct> {}, {}

// CHECK-LABEL: lit.func @global_ref_reg_store
lit.func @global_ref_reg_store(%x: !lit.declref<@MyRegStruct> owned) {
  // CHECK-NEXT: %0 = lit.globalvar.ref @y
  %0 = lit.globalvar.ref @y : <@MyRegStruct, mut #lit.lifetime>
  // CHECK-NEXT: %1 = lit.ref.load %0
  // CHECK-NEXT: call @MyRegStruct::@__del__(%1)
  // CHECK-NEXT: lit.ref.store %x, %0
  lit.ref.store %x, %0 :  <@MyRegStruct, mut #lit.lifetime>
  kgen.return
}

// -----

// COM: Verify that we don't traverse external functions.

lit.struct.decl @MyStruct attributes {destructor = #kgen.symbol.constant<@MyStruct::@__del__> : !lit.signature<[1](!lit.ref<@MyStruct, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

// CHECK-LABEL: @external_func
// CHECK-NEXT: lit.extern_func
lit.func @external_func(%arg: !lit.ref<@MyStruct, mut #lit.lifetime> owned_in_mem) attributes {preCompiledModuleRef = @package, preElaborationName = "external_func"} {
  lit.extern_func
}

// -----

// COM: debuginfo.value ops may reference values that are not initialized (e.g.
// COM: init_self arguments in __init__ functions). We check here that this does
// COM: not cause an error in the pass.

#file = #debuginfo.file<"foo.c" in "/mlir/">
#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<!lit.ref<@MyStruct, mut #lit.lifetime>>

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

lit.struct.decl @SomeData {
}

lit.struct.decl @MyStruct attributes {destructor = #kgen.symbol.constant<@MyStruct::@__del__ > : !lit.signature<[1](!lit.ref<@MyStruct, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field str : !lit.declref<@SomeData>
}

// CHECK-LABEL: lit.func @init
lit.func @init(%self: !lit.ref<@MyStruct, mut #lit.lifetime> init_self) {
  // CHECK-NEXT: debuginfo.value #local_variable
  debuginfo.value #local_variable = %self : !lit.ref<@MyStruct, mut #lit.lifetime> loc(#loc)
  // CHECK-NOT: __del__
  %2 = lit.call @bar(%self) : !lit.signature<(!lit.ref<@MyStruct, mut #lit.lifetime> init_self) -> !kgen.none> loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)

// -----

// COM: Test that destructors are inserted for error instances.

!Error = !lit.declref<@Error>

// CHECK-LABEL: lit.struct.decl @Error
lit.struct.decl @Error register_passable attributes {destructor = #kgen.symbol.constant<@Error::@__del__ > : !lit.signature<(!Error) -> !kgen.none>} {
  lit.struct.field a : index
  lit.func @__init__() ownedresult -> !Error {
     %idx0 = index.constant 0
     %0 = lit.struct.create(a=%idx0) : (index) -> !Error
     kgen.return %0 : !Error
  }
}

lit.func @doSomething(%e: !Error borrow) {
   kgen.return
}

lit.func @i_raise() throws|ownedresult -> !kgen.variant<!Error, index> {
  %0 = lit.call @Error::@__init__() : !lit.signature<() ownedresult -> !Error>
  %1 = kgen.variant.create %0, 0 : <!Error, index>
  lit.error_return %1 : <!Error, index>
}

// CHECK-LABEL: lit.func @eatErrorNoRef
lit.func @eatErrorNoRef() {
  lit.try {
    %3 = lit.call @i_raise() : !lit.signature<() throws|ownedresult -> !kgen.variant<!Error, index>>
    %4 = lit.handle_variant %3 : (!kgen.variant<!Error, index>) -> index {
      %6 = kgen.variant.take %3, 1 : <!Error, index>
      lit.yield %6 : index
    } else {
      %6 = kgen.variant.take %3, 0 : <!Error, index>
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
    %3 = lit.call @i_raise() : !lit.signature<() throws|ownedresult -> !kgen.variant<!Error, index>>
    %4 = lit.handle_variant %3 : (!kgen.variant<!Error, index>) -> index {
      %6 = kgen.variant.take %3, 1 : <!Error, index>
      lit.yield %6 : index
    } else {
      %6 = kgen.variant.take %3, 0 : <!Error, index>
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

lit.func @consume_err(%value: !Error) {
  kgen.return
}

// CHECK-LABEL: lit.func @conditional_consumption_1
// Issue#34320: https://github.com/modularml/modular/issues/34320
lit.func @conditional_consumption_1(%c: i1, %value: !Error) {
  // CHECK-NOT: @Error::@__del__
  hlcf.loop {
    hlcf.if %c {
      lit.call @consume_err(%value) : !lit.signature<(!Error) -> ()>
      hlcf.break
    } else {
      hlcf.yield
    }
    lit.call @consume_err(%value) : !lit.signature<(!Error) -> ()>
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: lit.func @conditional_consumption_2
lit.func @conditional_consumption_2(%c: i1, %value: !Error) {
  // CHECK-NOT: @Error::@__del__
  hlcf.loop {
    hlcf.if %c {
      hlcf.yield
    } else {
      lit.call @consume_err(%value) : !lit.signature<(!Error) -> ()>
      hlcf.break
    }
    lit.call @consume_err(%value) : !lit.signature<(!Error) -> ()>
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: lit.func @conditional_consumption_3
lit.func @conditional_consumption_3(%c: i1, %value: !Error) {
  // CHECK-NOT: @Error::@__del__
  hlcf.loop {
    lit.try {
      hlcf.if %c {
        lit.try.raise %c : i1
      } else {
        hlcf.yield
      }
      lit.try.yield
    } except (%e: i1) {
      lit.call @consume_err(%value) : !lit.signature<(!Error) -> ()>
      hlcf.break
    } else {
      lit.try.yield
    }
    lit.call @consume_err(%value) : !lit.signature<(!Error) -> ()>
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: lit.func @conditional_consumption_4
lit.func @conditional_consumption_4(%c: i1, %value: !Error) {
  // CHECK-NOT: @Error::@__del__
  hlcf.loop {
    lit.try {
      hlcf.if %c {
        lit.try.raise %c : i1
      } else {
        hlcf.yield
      }
      lit.try.yield
    } except (%e: i1) {
      lit.try.yield
    } else {
      lit.call @consume_err(%value) : !lit.signature<(!Error) -> ()>
      hlcf.break
    }
    lit.call @consume_err(%value) : !lit.signature<(!Error) -> ()>
    hlcf.break
  }
  kgen.return
}

// -----

// COM: Copy-del elision of register-passable value, where the argument is an
// COM: owned register-passable letreg decl.

!Reg = !lit.declref<@Reg>
lit.struct.decl @Reg register_passable attributes {
    copyInit = #kgen.symbol.constant<@Reg::@__copyinit__> : !lit.signature<(!Reg borrow) ownedresult -> !Reg>,
    destructor = #kgen.symbol.constant<@Reg::@__del__> : !lit.signature<(!Reg) -> !kgen.none>
} {
  lit.func @__del__(%self: !Reg, |) {
    kgen.return
  }
  // FIXME: Wrong copyinit signature.
  lit.func @__copyinit__(%other: !Reg owned) ownedresult -> !Reg attributes {specialFnKind = 7 : i8} {
    kgen.return %other : !Reg
  }
}

// CHECK-LABEL: lit.func @copy_del_reg_value
lit.func @copy_del_reg_value() {
  %0 = kgen.param.materialize: !Reg = <#lit.struct<{}>>

  %x = lit.var.decl "x" var : !lit.ref<!Reg, mut a>
  lit.ref.store %0, %x : !lit.ref<!Reg, mut a>
  %load = lit.ref.load %x : !lit.ref<!Reg, mut a>
  // CHECK: lit.ref.store
  // CHECK: [[LOAD:%.*]] = lit.ref.load %x
  // CHECK: [[COPY:%.*]] = lit.call @Reg::@__copyinit__([[LOAD]])
  %1 = lit.call @Reg::@__copyinit__(%load) : !lit.signature<(!Reg borrow, |) ownedresult -> !Reg>
  // CHECK: [[ORIG:%.*]] = lit.ref.load %x
  // CHECK: call @Reg::@__del__([[ORIG]])
  // CHECK: call @Reg::@__del__([[COPY]])
  kgen.return
}

// -----

!Thing = !lit.declref<@Thing>
lit.struct.decl @Box<T: trait<@AnyType>>  {
  lit.struct.field x : !kgen.paramref<:trait<@AnyType> T>
}

lit.struct.decl @Thing {
  lit.struct.field x : index
  lit.struct.field y : index
  lit.struct.field z : index
  lit.func @get(%self: !lit.ref<!Thing, imm #lit.lifetime> borrow_in_mem) {
    kgen.return
  }
}

lit.func @top(%c: !lit.ref<@Box<:trait<@AnyType> !Thing>, mut #lit.lifetime> borrow_in_mem) {
  %0 = lit.ref.struct.ger %c[x] : <!Thing, mut #lit.lifetime> from @Box<:trait<@AnyType> !Thing>
  lit.call @Thing::@get(%0) : !lit.signature<("self": !lit.ref<!Thing, mut #lit.lifetime> borrow_in_mem) -> ()>
  kgen.return
}
