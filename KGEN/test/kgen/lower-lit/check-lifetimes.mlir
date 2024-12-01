// RUN: kgen-opt %s -split-input-file -check-lifetimes -verify-diagnostics | FileCheck %s


// struct Struct:
lit.struct.decl @Struct attributes {
  destructor = #kgen.symbol.constant<@Struct::@__del__> : !lit.signature<[1](!lit.ref<@Struct, mut *[0,0]> owned_in_mem) -> !kgen.none>}
{
  //   var a: __mlir_type.index
  lit.struct.field a : index

  //   fn __init__(out self: Self):
  //     self.a = 1
  lit.func @__init__[mut selflife](%self: !lit.ref<@Struct, mut selflife> init_self) -> !kgen.none attributes {isStatic} {
    %0 = lit.ref.struct.ger %self[a] : <@Struct, mut selflife> -> index
    %idx1 = index.constant 1
    lit.ref.store %idx1, %0 : !lit.ref<index, mut selflife->a>

    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }

  // fn __copyinit__(out self, existing: Self):
  lit.func @__copyinit__[mut selflife, imm existinglife](
      %self: !lit.ref<@Struct, mut selflife> init_self,
      %existing: !lit.ref<@Struct, imm existinglife> read_mem) -> !kgen.none {
    %0 = lit.ref.struct.ger %existing[a] : <@Struct, imm existinglife> -> index
    %1 = lit.ref.load %0 : !lit.ref<index, imm existinglife->a>
    %2 = lit.ref.struct.ger %self[a] : <@Struct, mut selflife> -> index
    lit.ref.store %1, %2 : !lit.ref<index, mut selflife->a>
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
  %a: !lit.ref<@Struct, imm #lit.any.origin> read_mem,
  %b: !lit.ref<@Struct, mut #lit.any.origin> owned_in_mem) -> !kgen.none {

  // b.a = 42
  // CHECK-NEXT: %0 = lit.ref.struct.ger %b[a]
  %b_a = lit.ref.struct.ger %b[a] : <@Struct, mut #lit.any.origin> -> index
  %idx42 = index.constant 42
  lit.ref.store %idx42, %b_a : !lit.ref<index, mut #lit.any.origin>


  // var c = Struct()
  %c = lit.var.decl "c" var : !lit.ref<@Struct, mut *"life">
  %0 = lit.call @Struct::@__init__[mut life](%c) : !lit.signature<[1](!lit.ref<@Struct, mut *[0,0]> init_self) -> !kgen.none>

  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// fn indirectCall(a: Struct):
lit.func @indirectCall(%a: !lit.ref<@Struct, imm #lit.any.origin> read_mem) {
  // @noncapturing fn byrefResultFn(x: Struct) -> Struct:
  lit.func byrefResultFn(
      %x: !lit.ref<@Struct, imm #lit.any.origin> read_mem,
      %result: !lit.ref<@Struct, mut *"life"> byref_result) {
    lit.call @Struct::@__copyinit__(%result, %x)
        : !lit.signature<(!lit.ref<@Struct, mut *"life"> init_self,
                          !lit.ref<@Struct, imm #lit.any.origin> read_mem) -> !kgen.none>
    kgen.return
  }

  // var c = byrefResultFn(x)
  %callee = kgen.create_closure[!lit.signature<(
      !lit.ref<@Struct, imm #lit.any.origin> read_mem,
      !lit.ref<@Struct, mut *"life"> byref_result) -> !kgen.none>: byrefResultFn]()
  %c = lit.var.decl "c" var : !lit.ref<@Struct, mut *"life">
  lit.call_indirect %callee(%a, %c) :
      !lit.signature<(
        !lit.ref<@Struct, imm #lit.any.origin> read_mem,
        !lit.ref<@Struct, mut *"life"> byref_result) -> !kgen.none>

  %0 = lit.ref.struct.ger %c[a] : <@Struct, mut life> -> index
  lit.ref.load %0 : !lit.ref<index, mut *"life"->a>

  kgen.return
}

// Some tests with non-trivial lifetimes references
// CHECK-LABEL: lit.func @references1
// CHECK-NOT: __del__
lit.func @references1[mut alife](%a: !lit.ref<@Struct, mut alife> owned_in_mem,
                             %cond: i1) {
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


// @__nonimpldestructible struct Linear:
// CHECK-LABEL: lit.struct.decl @Linear
lit.struct.decl @Linear attributes {
  linearTypeErrorMsg = "'Linear' isn't implicit destructible, call the 'close' or 'explode' methods to explicitly destroy it"
}{
  // fn close(owned self): pass
  // CHECK: lit.func @close
  lit.func @close[mut dellife](%self: !lit.ref<@Linear, mut dellife> owned_in_mem) {
    // CHECK-NEXT: lit.ownership.mark_destroyed %self
    lit.ownership.mark_destroyed %self : !lit.ref<@Linear, mut dellife>
    // CHECK-NEXT: kgen.return
    kgen.return
  }
}

// CHECK-LABEL: lit.func @useLinear
lit.func @useLinear(
  %b: !lit.ref<@Linear, mut #lit.any.origin> owned_in_mem) {
  lit.call @Linear::@close[mut #lit.any.origin](%b) : !lit.signature<[1](!lit.ref<@Linear, mut *[0,0]> owned_in_mem) -> ()>
  kgen.return
}


// -----

// COM: Test all fields are destroyed in object destructor

lit.struct.decl @S attributes {destructor = #kgen.symbol.constant<@S::@__del__> : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

// CHECK-LABEL: lit.struct.decl @HasMemFields
lit.struct.decl @HasMemFields attributes {destructor = #kgen.symbol.constant<@HasMemFields::@__del__> : !lit.signature<[1](!lit.ref<@HasMemFields, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : !lit.struct<@S>
  lit.struct.field stole : !lit.struct<@S>
  lit.struct.field uninitialized : !lit.struct<@S>
  lit.struct.field register : index

  lit.func @__del__[mut dellife](%self: !lit.ref<@HasMemFields, mut dellife> owned_in_mem) -> !kgen.none {
    // CHECK: %[[VAR0:.*]] = lit.ref.struct.ger %self[a]
    // CHECK: %[[VAR1:.*]] = lit.call @S::@__del__[mut dellife->a](%[[VAR0]])
    // CHECK: %[[VAR2:.*]] = lit.ref.struct.ger %self[stole]
    // CHECK: %[[VAR3:.*]] = lit.call @S::@__del__[mut dellife->stole](%[[VAR2]])
    // CHECK: %[[VAR4:.*]] = lit.ref.struct.ger %self[uninitialized]
    // CHECK: %[[VAR5:.*]] = lit.call @S::@__del__[mut dellife->uninitialized](%[[VAR4]])
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
    %6 = lit.call @use(%anonymous2A) : !lit.signature<(!lit.ref<@MyStruct, mut *"life"> read_mem) -> !kgen.none>
    // CHECK: lit.call @MyStruct::@__del__[mut life](%anonymous2A)
    hlcf.yield
  }
  kgen.return %1 : !kgen.none
}

lit.globalvar.decl @x : !lit.struct<@MyStruct> {}, {}

// CHECK-LABEL: lit.func @byref_result_global_ref
lit.func @byref_result_global_ref() {
  // CHECK-NEXT: %0 = lit.globalvar.ref @x
  %0 = lit.globalvar.ref @x : <@MyStruct, mut #lit.any.origin>
  // CHECK-NEXT: lit.call @MyStruct::@__del__{{.*}}(%0)
  // CHECK-NEXT: call @memory_result
  lit.call @memory_result(%0) : !lit.signature<(!lit.ref<@MyStruct, mut #lit.any.origin> byref_result) -> ()>
  kgen.return
}

// CHECK-LABEL: lit.func @global_ref_no_use
lit.func @global_ref_no_use() {
  // CHECK-NOT: call @MyStruct::@__del__
  %0 = lit.globalvar.ref @x : <@MyStruct, mut #lit.any.origin>
  kgen.return
}

// -----

// COM: debuginfo.value ops may reference values that are not initialized (e.g.
// COM: init_self arguments in __init__ functions). We check here that this does
// COM: not cause an error in the pass.

#file = #debuginfo.file<"foo.c" in "/mlir/">
#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<!lit.ref<@MyStruct, mut #lit.any.origin>>

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

lit.struct.decl @SomeData {
}

lit.struct.decl @MyStruct attributes {destructor = #kgen.symbol.constant<@MyStruct::@__del__ > : !lit.signature<[1](!lit.ref<@MyStruct, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field str : !lit.struct<@SomeData>
}

// CHECK-LABEL: lit.func @init
lit.func @init(%self: !lit.ref<@MyStruct, mut #lit.any.origin> init_self) {
  // CHECK-NEXT: debuginfo.value #local_variable
  debuginfo.value #local_variable = %self : !lit.ref<@MyStruct, mut #lit.any.origin> loc(#loc)
  // CHECK-NOT: __del__
  %2 = lit.call @bar(%self) : !lit.signature<(!lit.ref<@MyStruct, mut #lit.any.origin> init_self) -> !kgen.none> loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)

// -----

// COM: Test that destructors are inserted for error instances.

!Error = !lit.struct<@Error>

// CHECK-LABEL: lit.struct.decl @Error
lit.struct.decl @Error register_passable attributes {destructor = #kgen.symbol.constant<@Error::@__del__ > : !lit.signature<(!Error) -> !kgen.none>} {
  lit.struct.field a : index
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

!Thing = !lit.struct<@Thing>
lit.struct.decl @Box<T: trait<@AnyType>>  {
  lit.struct.field x : !kgen.paramref<:trait<@AnyType> T>
}

lit.struct.decl @Thing {
  lit.struct.field x : index
  lit.struct.field y : index
  lit.struct.field z : index
  lit.func @get(%self: !lit.ref<!Thing, imm #lit.any.origin> read_mem) {
    kgen.return
  }
}

lit.func @top(%c: !lit.ref<@Box<:trait<@AnyType> !Thing>, mut #lit.any.origin> read_mem) {
  %0 = lit.ref.struct.ger %c[x] : <@Box<:trait<@AnyType> !Thing>, mut #lit.any.origin> -> !Thing
  lit.call @Thing::@get(%0) : !lit.signature<("self": !lit.ref<!Thing, mut #lit.any.origin> read_mem) -> ()>
  kgen.return
}

// -----

lit.struct.decl @S attributes {
  destructor =
    #kgen.symbol.constant<@S::@__del__> : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
  lit.func @__init__[mut selflife](%self: !lit.ref<@S, mut selflife> init_self, |) -> !kgen.none attributes {isStatic} {
    %0 = lit.ref.struct.ger %self[a] : <@S, mut selflife> -> index
    %idx1 = index.constant 1
    lit.ref.store %idx1, %0 : !lit.ref<index, mut selflife->a>

    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

lit.func @print(%borrowMe: !lit.ref<@S, mut #lit.any.origin> read_mem) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}

lit.func @elifInitCorrect[mut *"__result__`0"](?, %cond: i1, %__result__[__result__]: !lit.ref<@S, mut *"__result__`0"> byref_result) -> !kgen.none {
  // CHECK: hlcf.elif {
  // CHECK-NEXT: hlcf.elif.yield %cond : i1
  // CHECK-NEXT: } then {
  // CHECK-NEXT:   lit.call @S::@__init__[mut *"__result__`0"](%__result__)
  // CHECK-NEXT:   hlcf.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   lit.call @S::@__init__[mut *"__result__`0"](%__result__)
  // CHECK-NEXT:   hlcf.yield
  // CHECK-NEXT: }
  hlcf.elif {
    hlcf.elif.yield %cond : i1
  } then {
    %0 = lit.call @S::@__init__[mut *"__result__`0"](%__result__) : !lit.signature<[1](!lit.ref<@S, mut *"__result__`0"> init_self, |) -> !kgen.none>
    hlcf.yield
  } else {
    %0 = lit.call @S::@__init__[mut *"__result__`0"](%__result__) : !lit.signature<[1](!lit.ref<@S, mut *"__result__`0"> init_self, |) -> !kgen.none>
    hlcf.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: @unreachableInElif
lit.func @unreachableInElif[mut *"range`", mut *"__result__`1"](
  %cond1: i1,
  %cond2: i1, ?,
  %__result__: !lit.ref<@S, mut *"__result__`1"> byref_result) {
  hlcf.elif {
    hlcf.elif.yield %cond1 : i1
  } then {
    hlcf.elif {
      hlcf.elif.yield %cond2 : i1
    } then {
      %1 = lit.call @S::@__init__[mut *"__result__`1"](%__result__) : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> init_self) -> !kgen.none>
      kgen.return
    } else {
      %1 = lit.call @S::@__init__[mut *"__result__`1"](%__result__) : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> init_self) -> !kgen.none>
      kgen.return
    }
    kgen.unreachable
  } else {
    %1 = lit.call @S::@__init__[mut *"__result__`1"](%__result__) : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> init_self) -> !kgen.none>
    kgen.return
  }
  kgen.unreachable
}

// -----

!S = !lit.struct<@S>
lit.struct.decl @S attributes {
  destructor =
    #kgen.symbol.constant<@S::@__del__> : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
  lit.func @__init__[mut selflife](%self: !lit.ref<@S, mut selflife> init_self, |) -> !kgen.none attributes {isStatic} {
    %0 = lit.ref.struct.ger %self[a] : <@S, mut selflife> -> index
    %idx1 = index.constant 1
    lit.ref.store %idx1, %0 : !lit.ref<index, mut selflife->a>

    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

!HasMemFields = !lit.struct<@HasMemFields>

lit.struct.decl @HasMemFields attributes {destructor = #kgen.symbol.constant<@HasMemFields::@__del__> : !lit.signature<[1](!lit.ref<@HasMemFields, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field x : !S
}

// CHECK-LABEL: lit.func @destroyField
lit.func @destroyField(%s2: i1, %A: !lit.ref<@HasMemFields, mut #lit.any.origin> owned_in_mem) -> !kgen.none {
  // CHECK:  hlcf.elif {
  // CHECK-NEXT:   hlcf.elif.yield %s2 : i1
  // CHECK-NEXT:  } then {
  // CHECK-NEXT:    [[V0:%*.]] = lit.ref.struct.ger %A[x]
  // CHECK-NEXT:    lit.call @S::@__del__[mut #lit.any.origin]([[V0]])
  // CHECK-NEXT:    [[V2:%*.]] = lit.ref.struct.ger %A[x] : <@HasMemFields, mut #lit.any.origin> -> @S
  // CHECK-NEXT:    lit.call @S::@__init__[mut #lit.any.origin]([[V2]])
  // CHECK-NEXT:    lit.call @HasMemFields::@__del__[mut #lit.any.origin](%A)
  // CHECK-NEXT:    hlcf.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:  lit.call @HasMemFields::@__del__[mut #lit.any.origin](%A)
  // CHECK-NEXT:  hlcf.yield
  // CHECK-NEXT:  }
  hlcf.elif {
    hlcf.elif.yield %s2 : i1
  } then {
    %0 = lit.ref.struct.ger %A[x] : <!HasMemFields, mut #lit.any.origin> -> !S
    %1 = lit.call @S::@__init__[mut #lit.any.origin](%0) : !lit.signature<[1](!lit.ref<!S, mut *[0,0]> init_self, |) -> !kgen.none>
    hlcf.yield
  } else {
    hlcf.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// -----

// COM: Test Error/Success Region in Fully Initialized Value

!Error = !lit.struct<@Error>
lit.struct.decl @Error register_passable attributes {destructor = #kgen.symbol.constant<@Error::@__del__> : !lit.signature<[1](!lit.ref<@Error, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

!S = !lit.struct<@S>
lit.struct.decl @S attributes {
  destructor =
    #kgen.symbol.constant<@S::@__del__> : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
  lit.func @__init__[mut selflife](%self: !lit.ref<@S, mut selflife> init_self, |) -> !kgen.none attributes {isStatic} {
    %0 = lit.ref.struct.ger %self[a] : <@S, mut selflife> -> index
    %idx1 = index.constant 1
    lit.ref.store %idx1, %0 : !lit.ref<index, mut selflife->a>

    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

!ThrowingSelfInit = !lit.struct<@ThrowingSelfInit>

lit.struct.decl @ThrowingSelfInit attributes {destructor = #kgen.symbol.constant<@ThrowingSelfInit::@__del__> : !lit.signature<[1](!lit.ref<@ThrowingSelfInit, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field x : !S
  // CHECK-LABEL: lit.func @__init__
  lit.func @__init__1[mut self, mut err](%self: !lit.ref<!ThrowingSelfInit, mut self> init_self, |, ?, %__error__: !lit.ref<!Error, mut err> byref_error) throws -> i1 attributes {sourceName = "__init__", specialFnKind = 2 : i8} {
    %0 = lit.ref.struct.ger %self[x] : <!ThrowingSelfInit, mut self> -> !S
    %1 = lit.call @S::@__init__[mut self->x](%0) : !lit.signature<[1](!lit.ref<!S, mut *[0,0]> init_self, |) -> !kgen.none>
    %2 = kgen.param.constant: i1 = <0>
    kgen.return %2 : i1
  }

  // Ensure that destructor is not inserted after calling other throwing initializer
  // CHECK-LABEL: lit.func @__init__2
  lit.func @__init__2[mut self, mut err](%self: !lit.ref<!ThrowingSelfInit, mut self> init_self, |, %x: !S, ?, %__error__: !lit.ref<!Error, mut err> byref_error) throws -> i1 attributes {sourceName = "__init__", specialFnKind = 2 : i8} {
    %0 = lit.call @throwing_init::@ThrowingSelfInit::@__init__1[mut self, mut err](%self, %__error__) : !lit.signature<[2]("self": !lit.ref<!ThrowingSelfInit, mut *[0,0]> init_self, |, ?, "__error__": !lit.ref<!Error, mut *[0,1]> byref_error) throws -> i1>
    // CHECK-NEXT: [[IS_ERR:%.*]] = lit.call {{.*}}@ThrowingSelfInit::@__init__1{{.*}}(%self, %__error__)
    // CHECK-NEXT: if [[IS_ERR]]
    // CHECK-NEXT:   mark_consumed %self
    // CHECK-NEXT:   [[TRUE:%.*]] = kgen.param.constant
    // CHECK-NEXT:   lit.error_return [[TRUE]]
    // CHECK-NEXT: else
    // CHECK-NEXT:   mark_consumed %__error__
    // CHECK-NEXT:   yield
    hlcf.if %0 {
      lit.ownership.mark_consumed %self : <!ThrowingSelfInit, mut self>
      %2 = kgen.param.constant: i1 = <1>
      lit.error_return %2 : i1
    } else {
      lit.ownership.mark_consumed %__error__ : <!Error, mut err>
      hlcf.yield
    }
    %1 = kgen.param.constant: i1 = <0>
    kgen.return %1 : i1
  }

  // Ensure that destructor is not inserted after calling other throwing initializer
  // CHECK-LABEL: lit.func @__init__3
  lit.func @__init__3[mut self, mut err](%self: !lit.ref<!ThrowingSelfInit, mut self> init_self, |, %cond: i1, %x: !S, ?, %__error__: !lit.ref<!Error, mut err> byref_error) throws -> i1 attributes {sourceName = "__init__", specialFnKind = 2 : i8} {
    %0 = lit.call @throwing_init::@ThrowingSelfInit::@__init__1[mut self, mut err](%self, %__error__) : !lit.signature<[2]("self": !lit.ref<!ThrowingSelfInit, mut *[0,0]> init_self, |, ?, "__error__": !lit.ref<!Error, mut *[0,1]> byref_error) throws -> i1>
    // CHECK-NEXT: [[IS_ERR:%.*]] = lit.call {{.*}}@ThrowingSelfInit::@__init__1{{.*}}(%self, %__error__)
    // CHECK-NEXT: if [[IS_ERR]]
    // CHECK-NEXT:   mark_consumed %self
    // CHECK-NEXT:   [[TRUE:%.*]] = kgen.param.constant
    // CHECK-NEXT:   lit.error_return [[TRUE]]
    // CHECK-NEXT: else
    // CHECK-NEXT:   mark_consumed %__error__
    // CHECK-NEXT:   yield
    hlcf.if %0 {
      lit.ownership.mark_consumed %self : <!ThrowingSelfInit, mut self>
      %2 = kgen.param.constant: i1 = <1>
      lit.error_return %2 : i1
    } else {
      lit.ownership.mark_consumed %__error__ : <!Error, mut err>
      hlcf.yield
    }
    %1 = kgen.param.constant: i1 = <0>
    kgen.return %1 : i1
  }
}

// -----

// COM: Track Result References

!Int = !lit.struct<@Int>
lit.struct.decl @Int register_passable_trivial {
  lit.struct.field value : index
}

!Node = !lit.struct<@Node>
lit.struct.decl @Node attributes {
  destructor =
    #kgen.symbol.constant<@Node::@__del__> : !lit.signature<[1](!lit.ref<@Node, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

!Container = !lit.struct<@Container>
lit.struct.decl @Container attributes {
  destructor =
    #kgen.symbol.constant<@Container::@__del__> : !lit.signature<[1](!lit.ref<@Container, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field z : !Node
}

!Wrapper = !lit.struct<@Wrapper>
lit.struct.decl @Wrapper attributes {destructor = #kgen.symbol.constant<@Wrapper::@__del__> : !lit.signature<[1](!lit.ref<@Wrapper, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field tail : !Int
  lit.struct.field y : !kgen.pointer<!Container>
}

// CHECK-LABEL: lit.func @indirectReferences
lit.func @indirectReferences[mut mylife](%s2: i1, %wrapper: !lit.ref<@Wrapper, mut mylife> read_mem) -> !kgen.none {
  // CHECK-NEXT:  hlcf.if %s2 {
  // CHECK-NEXT:    %[[V4:.*]] = lit.call @Wrapper::@__get_ref[mut mylife](%wrapper)
  // CHECK-NEXT:    %[[V5:.*]] = lit.ref.struct.ger %[[V4]][z] : <@Container, mut mylife> -> @Node
  // CHECK-NEXT:    %[[V6:.*]] = lit.call @Node::@__del__[mut mylife->z](%[[V5]])
  // CHECK-NEXT:    lit.call @Node::@__init__[mut mylife->z](%[[V5]])
  // CHECK-NEXT:    hlcf.yield
  hlcf.if %s2 {
    %trackedByLifetime0 = lit.call @Wrapper::@__get_ref[mut mylife](%wrapper) : !lit.signature<[1](!lit.ref<!Wrapper, mut *[0,0]> read_mem) -> !lit.ref<!Container, mut mylife>>
    %trackedByLifetime1 = lit.ref.struct.ger %trackedByLifetime0[z] : <!Container, mut mylife> -> !Node
    %8 = lit.call @Node::@__init__[mut mylife->z](%trackedByLifetime1) : !lit.signature<[1](!lit.ref<@Node, mut *[0,0]> init_self) -> !kgen.none>
    hlcf.yield
  } else {
    hlcf.yield
  }

  // overwrite 'tail' to trigger the reset of some bits within Wrapper
  %tail = lit.ref.struct.ger %wrapper[tail] : <!Wrapper, mut mylife> -> !Int
  %6 = kgen.param.constant: !Int = <{0}>
  lit.ref.store %6, %tail : <!Int, mut mylife->tail>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// -----

// COM: Test Error/Success Region in Partially Initialized Value

!DestructSome = !lit.struct<@DestructSome>
!Error = !lit.struct<@Error>
!Field = !lit.struct<@Field>


lit.struct.decl @Error register_passable attributes {destructor = #kgen.symbol.constant<@Error::@"__del__"> : !lit.signature<("self": !Error, |) -> !kgen.none>} {
}

lit.func @somethingThatRaises[mut *"__error__`", mut *"__result__`1"](?, %__error__: !lit.ref<!Error, mut *"__error__`"> byref_error, %__result__: !lit.ref<none, mut *"__result__`1"> byref_result) throws -> i1  {
  %none = kgen.param.constant: none = <#kgen.none>
  lit.ref.store %none, %__result__ : <none, mut *"__result__`1">
  %0 = kgen.param.constant: i1 = <0>
  kgen.return %0 : i1
}

lit.struct.decl @Field  attributes {destructor = #kgen.symbol.constant<@Field::@"__del__"> : !lit.signature<[1]("self": !lit.ref<!Field, mut *[0,0]> owned_in_mem, |) -> !kgen.none>,
copy = #kgen.symbol.constant<@Field::@"__copyinit__"> : !lit.signature<[2]("self": !lit.ref<!Field, mut *[0,0]> init_self, |, "existing": !lit.ref<!Field, imm *[0,1]> read_mem) -> !kgen.none>} {

}

lit.struct.decl @DestructSome  attributes {destructor = #kgen.symbol.constant<@DestructSome::@"__del__"> : !lit.signature<[1]("self": !lit.ref<!DestructSome, mut *[0,0]> owned_in_mem, |) -> !kgen.none>} {
  lit.struct.field b : !Field
  lit.struct.field a : !Field
  // CHECK-LABEL: lit.func @__init__
  lit.func @__init__[mut self, imm a, imm b, mut err](%self: !lit.ref<!DestructSome, mut self> init_self, |, %a: !lit.ref<!Field, imm a> read_mem, %b: !lit.ref<!Field, imm b> read_mem, ?, %__error__: !lit.ref<!Error, mut err> byref_error) throws -> i1 attributes {sourceName = "__init__", specialFnKind = 2 : i8} {
    // CHECK-NEXT:  [[V0:%.*]] = lit.ref.struct.ger %self[b]
    // CHECK-NEXT:  %1 = lit.call @Field::@__copyinit__{{.*}}([[V0]], %b)
    // CHECK-NEXT:  %tmp = lit.var.decl "tmp" synth : !lit.ref<none, mut tmp>
    // CHECK-NEXT:  lifetime.start %tmp
    // CHECK-NEXT:  %2 = lit.call @somethingThatRaises{{.*}}(%__error__, %tmp)
    // CHECK-NEXT:  if %2
    // CHECK-NEXT:    %[[V4:.*]] = lit.ref.struct.ger %self[b] : <@DestructSome, mut self> -> @Field
    // CHECK-NEXT:    lit.call @Field::@__del__[mut self->b](%[[V4]])
    // CHECK-NEXT:    mark_consumed %tmp
    // CHECK-NEXT:    lifetime.end %tmp
    // CHECK-NEXT:    kgen.param.constant
    // CHECK-NEXT:    lit.error_return
    // CHECK-NEXT:  } else {
    // CHECK-NEXT:    lifetime.end %tmp
    // CHECK-NEXT:    mark_consumed %__error__
    // CHECK-NEXT:    yield
    // CHECK-NEXT:  }
    %4 = lit.ref.struct.ger %self[b] : <!DestructSome, mut self> -> !Field
    %5 = lit.call @Field::@__copyinit__[mut self->b, imm b](%4, %b) : !lit.signature<[2]("self": !lit.ref<!Field, mut *[0,0]> init_self, |, "existing": !lit.ref<!Field, imm *[0,1]> read_mem) -> !kgen.none>
    %tmp = lit.var.decl "tmp" synth : !lit.ref<none, mut tmp>
    %6 = lit.call @somethingThatRaises[mut err, mut tmp](%__error__, %tmp) : !lit.signature<[2](?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<none, mut *[0,1]> byref_result) throws -> i1>
    hlcf.if %6 {
      lit.ownership.mark_consumed %tmp : <none, mut tmp>
      %8 = kgen.param.constant: i1 = <1>
      lit.error_return %8 : i1
    } else {
      lit.ownership.mark_consumed %__error__ : <!Error, mut err>
      hlcf.yield
    }
    %7 = lit.ref.struct.ger %self[a] : <!DestructSome, mut self> -> !Field
    %8 = lit.call @Field::@"__copyinit__"[mut self->a, imm a](%7, %a) : !lit.signature<[2]("self": !lit.ref<!Field, mut *[0,0]> init_self, |, "existing": !lit.ref<!Field, imm *[0,1]> read_mem) -> !kgen.none>
    %9 = kgen.param.constant: i1 = <0>
    kgen.return %9 : i1
  }
}

// -----

// COM: Verify that unreachable code is ignored.

!Error = !lit.struct<@Error>
!FileHandle = !lit.struct<@FileHandle>
!GGUFFile = !lit.struct<@GGUFFile>
!Int = !lit.struct<@Int>
!iter = !lit.struct<@my_iter>

lit.struct.decl @Error register_passable
  destructor :!lit.signature<("self": !Error, |) -> !kgen.none> @stdlib::@builtin::@stubs::@Error::@"__del__(stdlib::builtin::stubs::Error)"{
}

lit.struct.decl @Int register_passable_trivial {
  lit.struct.field value : index
}

lit.struct.decl @my_iter
  destructor :!lit.signature<[1]("self": !lit.ref<!iter, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @small::@my_iter::@"__del__(small::my_iter)"
  copy :!lit.signature<[2]("self": !lit.ref<!iter, mut *[0,0]> init_self, |, "existing": !lit.ref<!iter, imm *[0,1]> read_mem) -> !kgen.none> @small::@my_iter::@"__copyinit__(small::my_iter=&,small::my_iter)"{
  lit.struct.field start : !Int
}

lit.struct.decl @FileHandle
  destructor :!lit.signature<[1]("self": !lit.ref<!FileHandle, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @small::@FileHandle::@"__del__(small::FileHandle)"{
  lit.struct.field str : !Int
}

lit.struct.decl @GGUFFile
  destructor :!lit.signature<[1]("self": !lit.ref<!GGUFFile, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @GGUFFile::@__del__{
  lit.struct.field size : !Int
  lit.struct.field fp : !FileHandle
  // CHECK-LABEL: lit.func @__init__
  lit.func @__init__[mut self, imm *"model_path`2x1", mut *"__error__`2x2", mut *"$RANGE`2x5"](
    %self: !lit.ref<!GGUFFile, mut self> init_self, |,
    %iter: !lit.ref<!iter, mut *"$RANGE`2x5"> read_mem, ?,
    %__error__: !lit.ref<!Error, mut *"__error__`2x2"> byref_error) throws -> i1 attributes {sourceName = "__init__", specialFnKind = 2 : i8} {
    %0 = lit.ref.struct.ger %self[size] : <!GGUFFile, mut self> -> !Int
    %1 = kgen.param.constant: !Int = <{0}>
    lit.ref.store %1, %0 : <!Int, mut self->size>

    hlcf.loop "_loop_0" {
      %9 = lit.call @my_iter::@__len__[mut *"$RANGE`2x5"](%iter) : !lit.signature<[1]("self": !lit.ref<!iter, mut *[0,0]> mut) -> index>
      %idx0 = index.constant 0
      // CHECK: [[V6:%.*]] = index.cmp sgt(%5, %idx0)
      // CHECK-NEXT:  hlcf.if [[V6]] {
      // CHECK-NEXT:    hlcf.yield
      // CHECK-NEXT:  } else {
      // CHECK-NEXT:    hlcf.break "_loop_0"
      // CHECK-NEXT:  }
      %11 = index.cmp sgt(%9, %idx0)
      hlcf.if %11 {
        hlcf.yield
      } else {
        hlcf.break "_loop_0"
      }
      %12 = lit.call @my_iter::@__next__[mut *"$RANGE`2x5"](%iter) : !lit.signature<[1]("self": !lit.ref<!iter, mut *[0,0]> mut) -> !Int>
      %13 = lit.ref.struct.ger %self[size] : <!GGUFFile, mut self> -> !Int
      %14 = kgen.param.constant: !Int = <{1}>

      // Results in self bits getting set but because its unreachable it should not affect the upward consume set of the if.
      %15 = lit.call @Int::@__iadd__[mut self->size](%13, %14) : !lit.signature<[1]("self": !lit.ref<!Int, mut *[0,0]> mut, "rhs": !Int) -> !kgen.none>
      hlcf.continue
    }
    // Causes bits in the self to be reset, which will trigger erroneous destructors if unreachable code is not ignored.
    %6 = lit.ref.struct.ger %self[fp] : <!GGUFFile, mut self> -> !FileHandle
    %7 = lit.call @FileHandle::@__init__[mut self->fp](%6) : !lit.signature<[1]("self": !lit.ref<!FileHandle, mut *[0,0]> init_self) -> !kgen.none>
    %8 = kgen.param.constant: i1 = <0>
    kgen.return %8 : i1
  }
}

// -----


// COM: Verify that only fields destroyed in the region are corrected for full object initialization.

!Error = !lit.struct<@Error>
!Int = !lit.struct<@Int>
!FileHandle = !lit.struct<@FileHandle>
!iter = !lit.struct<@my_iter>
!GGUFFile = !lit.struct<@GGUFFile>

lit.struct.decl @GGUFFile
 destructor :!lit.signature<[1]("self": !lit.ref<!GGUFFile, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @GGUFFile::@__del__{
  lit.struct.field fp : !FileHandle
  lit.func @"__init__"[mut selfLife, mut consumeMeLife, mut iterLife, mut errorLife](
    %self: !lit.ref<!GGUFFile, mut selfLife> init_self, |,
    %consumeMe: !lit.ref<!GGUFFile, mut consumeMeLife> owned_in_mem,
    %iter: !lit.ref<!iter, mut iterLife> mut,
    %x: i1,
    %version: !Int, ?,
    %__error__: !lit.ref<!Error, mut errorLife> byref_error) throws -> i1 {
    %15 = kgen.param.constant: i1 = <1>
    hlcf.loop "_loop_0" {
      %10 = lit.call @my_iter::@__len__[mut iterLife](%iter) : !lit.signature<[1]("self": !lit.ref<!iter, mut *[0,0]> mut) -> index>
      %idx0 = index.constant 0
      %12 = index.cmp sgt(%10, %idx0)
      // CHECK:     hlcf.if %5 {
      // CHECK-NEXT:  hlcf.yield
      // CHECK-NEXT: } else {
      // CHECK-NEXT:  lit.call @GGUFFile::@__del__[mut consumeMeLife](%consumeMe)
      // CHECK-NEXT:  hlcf.break "_loop_0"
      // CHECK-NEXT: }
      hlcf.if %12 {
        hlcf.yield
      } else {
        hlcf.break "_loop_0"
      }
      // CHECK-NEXT: lit.call @my_iter
      %13 = lit.call @my_iter::@"__next__"[mut iterLife](%iter) : !lit.signature<[1]("self": !lit.ref<!iter, mut *[0,0]> mut) -> !Int>

      // CHECK-NEXT: hlcf.elif {
      // CHECK-NEXT:   lit.call @print[mut consumeMeLife](%consumeMe) : !lit.signature<[1](!lit.ref<@GGUFFile, mut *[0,0]>, |) -> !kgen.none>
      // CHECK-NEXT:   hlcf.elif.yield %x : i1
      // CHECK-NEXT: } then {
      // CHECK-NEXT:   lit.call @GGUFFile::@__del__[mut consumeMeLife](%consumeMe)
      // CHECK-NEXT:   lit.call @Error::@__init__
      // CHECK-NEXT:   lit.error_return %0 : i1
      // CHECK-NEXT: } else {
      // CHECK-NEXT:   hlcf.yield
      // CHECK-NEXT: }
      hlcf.elif {
        // consume something here so that the consume set of this block is different from its successors
        %3 = lit.call @print[mut consumeMeLife](%consumeMe) : !lit.signature<[1](!lit.ref<!GGUFFile, mut *[0,0]>, |) -> !kgen.none>
        hlcf.elif.yield %x : i1
      } then {
        %14 = lit.call @Error::@"__init__"[mut errorLife](%__error__) : !lit.signature<[1]("self": !lit.ref<!Error, mut *[0,0]> init_self, |) -> !kgen.none>
        lit.error_return %15 : i1
      } else {
        hlcf.yield
      }
      hlcf.continue
    }

    %7 = lit.ref.struct.ger %self[fp] : <!GGUFFile, mut selfLife> -> !FileHandle
    %8 = lit.call @FileHandle::@__init__[mut selfLife->fp](%7) : !lit.signature<[1]("self": !lit.ref<!FileHandle, mut *[0,0]> init_self) -> !kgen.none>
    %9 = kgen.param.constant: i1 = <0>
    kgen.return %9 : i1
  }
}

lit.struct.decl @Error register_passable
 destructor :!lit.signature<("self": !Error, |) -> !kgen.none> @Error::@"__del__"{
}

lit.struct.decl @Int register_passable_trivial {
  lit.struct.field value : index
}

lit.struct.decl @FileHandle
 destructor :!lit.signature<[1]("self": !lit.ref<!FileHandle, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @FileHandle::@"__del__"{
  lit.struct.field x : !Int
}

lit.struct.decl @my_iter
 destructor :!lit.signature<[1]("self": !lit.ref<!iter, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @my_iter::@"__del__"
copy :!lit.signature<[2]("self": !lit.ref<!iter, mut *[0,0]> init_self, |, "existing": !lit.ref<!iter, imm *[0,1]> read_mem) -> !kgen.none> @my_iter::@"__copyinit__"{
  lit.struct.field start : !Int
  lit.struct.field end : !Int
}

// -----

// COM: Verify that lifetimes are respected.

!Int = !lit.struct<@Int>
lit.struct.decl @Int register_passable_trivial {
  lit.struct.field value : index
}
!Reference = !lit.struct<@Reference>
lit.struct.decl @Reference register_passable_trivial {
  lit.struct.field value : !kgen.pointer<index>
}

!MyStruct = !lit.struct<@MyStruct>
lit.struct.decl @MyStruct attributes {
  destructor =
    #kgen.symbol.constant<@MyStruct::@__del__> : !lit.signature<[1](!lit.ref<@MyStruct, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : !Int
}

!Wrapper = !lit.struct<@Wrapper>
lit.struct.decl @Wrapper attributes {destructor = #kgen.symbol.constant<@Wrapper::@__del__> : !lit.signature<[1](!lit.ref<@Wrapper, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field y : !kgen.pointer<!Int>
}

// CHECK-LABEL: lit.func @respectLifetimes
lit.func @respectLifetimes[mut mylife](%s2: i1) -> !kgen.none {
  // CHECK-NEXT: %v = lit.var.decl "v" var : !lit.ref<@Wrapper, mut *"v`5">
  // CHECK-NEXT: lifetime.start %v
  // CHECK-NEXT: lit.call @Wrapper::@__init__[mut *"v`5"](%v)
  // CHECK-NEXT: %[[V1:.*]] = lit.call @Wrapper::@__get_ref[mut *"v`5"](%v)
  // CHECK-NEXT: %[[V2:.*]] = lit.call @Reference::@__get_ref[mut *"v`5"](%[[V1]])
  // CHECK-NEXT: %[[V3:.*]] = lit.ref.struct.ger %[[V2]][a] : <@MyStruct, mut *"v`5"> -> @Int
  // CHECK-NEXT: %[[V4:.*]] = lit.ref.immut %[[V3]] : <@Int, mut *"v`5"->a>
  // CHECK-NEXT: lit.call @print[muttoimm *"v`5"->a](%[[V4]])
  // CHECK-NEXT: lit.call @Wrapper::@__del__[mut *"v`5"](%v)

  %v = lit.var.decl "v" var : !lit.ref<@Wrapper, mut *"v`5">
  %1 = lit.call @Wrapper::@__init__[mut *"v`5"](%v) : !lit.signature<[1]("self" : !lit.ref<@Wrapper, mut *[0,0]> init_self, |) -> !kgen.none>
  %refWrapper = lit.call @Wrapper::@__get_ref[mut *"v`5"](%v) : !lit.signature<[1](!lit.ref<!Wrapper, mut *[0,0]> read_mem) -> !Reference>
  %ref = lit.call @Reference::@__get_ref[mut *"v`5"](%refWrapper) : !lit.signature<[1](!Reference) -> !lit.ref<!MyStruct, mut *[0,0]>>
  %35 = lit.ref.struct.ger %ref[a] : <!MyStruct, mut *"v`5"> -> !Int
  %36 = lit.ref.immut %35 : <!Int, mut *"v`5"->a>
  %41 = lit.call @print[muttoimm *"v`5"->a](%36) : !lit.signature<[1]("first": !lit.ref<!Int, imm *[0,0]> read_mem) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// -----

!Error = !lit.struct<@Error>
lit.struct.decl @Error {
  lit.struct.field a : index
}

!PythonObject = !lit.struct<@PythonObject>
lit.struct.decl @PythonObject attributes {
  destructor =
    #kgen.symbol.constant<@PythonObject::@__del__> : !lit.signature<[1](!lit.ref<@PythonObject, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

!Context = !lit.struct<@Context>
lit.struct.decl @Context attributes {destructor = #kgen.symbol.constant<@Context::@__del__> : !lit.signature<[1](!lit.ref<@Context, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field __new_repl_var : !kgen.pointer<pointer<!PythonObject>>
  lit.struct.field __new_repl_var2 : !kgen.pointer<pointer<!PythonObject>>
}

// CHECK-LABEL: lit.func @createConditionallyInitializedImmortalReferenceInRepl
lit.func @createConditionallyInitializedImmortalReferenceInRepl[mut topArg, mut localError, mut localResult](
  %__mojo_repl_arg : !lit.ref<!Context, mut topArg> mut,?,
  %__error__: !lit.ref<!Error, mut localError> byref_error,
  %__result__: !lit.ref<none, mut localResult> byref_result) throws|capturing -> i1 {

  %2 = lit.ref.struct.ger %__mojo_repl_arg[__new_repl_var] : <!Context, mut topArg> -> pointer<pointer<!PythonObject>>
  %3 = lit.ref.load %2 : <pointer<pointer<!PythonObject>>, mut topArg->__new_repl_var>
  %int_3 = kgen.param.constant: !kgen.int_literal = <get_sizeof(!PythonObject, current_target())>
  %index_3 = kgen.int_literal.convert %int_3 : to index
  %int_4 = kgen.param.constant: !kgen.int_literal = <get_alignof(!PythonObject, current_target())>
  %index_4 = kgen.int_literal.convert %int_4 : to index
  %4 = pop.aligned_alloc %index_4, %index_3 : <!PythonObject>
  pop.store %4, %3 : !kgen.pointer<pointer<!PythonObject>>

  // CHECK:  kgen.param.declare LOCAL_LIFETIME2: origin<1> = <#lit.any.origin>
  // CHECK-NEXT:  %[[V3:.*]] = lit.ref.from_pointer.repl {{.*}} : <@PythonObject, mut LOCAL_LIFETIME2> {name = "np"}
  // CHECK-NEXT:  [[V4:%*.]] = lit.call @import_module[mut localError, mut LOCAL_LIFETIME2](%__error__, %[[V3]])
  // CHECK-NEXT:  hlcf.if [[V4]]
  // CHECK-NEXT:    mark_consumed %[[V3]]
  // CHECK-NEXT:    kgen.param.constant: i1 = <1>
  // CHECK-NEXT:    lit.error_return
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    mark_consumed %__error__
  // CHECK-NEXT:    yield
  // CHECK-NEXT:  }
  kgen.param.declare LOCAL_LIFETIME2: origin<1> = <#lit.any.origin>
  %5 = lit.ref.from_pointer.repl %4 : <!PythonObject, mut LOCAL_LIFETIME2> {name = "np"}
  %6 = lit.call @import_module[mut localError, mut LOCAL_LIFETIME2](%__error__, %5) : !lit.signature<[2](?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<!PythonObject, mut *[0,1]> byref_result) throws -> i1>
  hlcf.if %6 {
    lit.ownership.mark_consumed %5 : <!PythonObject, mut LOCAL_LIFETIME2>
    %7 = kgen.param.constant: i1 = <1>
    lit.error_return %7 : i1
  } else {
    lit.ownership.mark_consumed %__error__ : <!Error, mut localError>
    hlcf.yield
  }

  %12 = lit.ref.struct.ger %__mojo_repl_arg[__new_repl_var2] : <!Context, mut topArg> -> pointer<pointer<!PythonObject>>
  %13 = lit.ref.load %12 : <pointer<pointer<!PythonObject>>, mut topArg->__new_repl_var2>
  %14 = pop.aligned_alloc %index_4, %index_3 : <!PythonObject>
  pop.store %14, %13 : !kgen.pointer<pointer<!PythonObject>>
  // CHECK:  kgen.param.declare LOCAL_LIFETIME3: origin<1> = <#lit.any.origin>
  // CHECK-NEXT:  %[[V8:.*]] = lit.ref.from_pointer.repl {{.*}} : <@PythonObject, mut LOCAL_LIFETIME3> {name = "np2"}
  // CHECK-NEXT:  %[[V9:.*]] = lit.call @import_module[mut localError, mut LOCAL_LIFETIME3](%__error__, %[[V8]])
  // CHECK-NEXT:  hlcf.if %[[V9]]
  // CHECK-NEXT:    lit.call @PythonObject::@__del__[mut LOCAL_LIFETIME2](%[[V3]])
  // CHECK-NEXT:    mark_consumed %[[V8]]
  // CHECK-NEXT:    kgen.param.constant: i1 = <1>
  // CHECK-NEXT:    lit.error_return
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    mark_consumed %__error__
  // CHECK-NEXT:    yield
  // CHECK-NEXT:  }
  kgen.param.declare LOCAL_LIFETIME3: origin<1> = <#lit.any.origin>
  %15 = lit.ref.from_pointer.repl %14 : <!PythonObject, mut LOCAL_LIFETIME3> {name = "np2"}
  %16 = lit.call @import_module[mut localError, mut LOCAL_LIFETIME3](%__error__, %15) : !lit.signature<[2](?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<!PythonObject, mut *[0,1]> byref_result) throws -> i1>
  hlcf.if %16 {
    lit.ownership.mark_consumed %15 : <!PythonObject, mut LOCAL_LIFETIME3>
    %7 = kgen.param.constant: i1 = <1>
    lit.error_return %7 : i1
  } else {
    lit.ownership.mark_consumed %__error__ : <!Error, mut localError>
    hlcf.yield
  }

  %none_5 = kgen.param.constant: none = <#kgen.none>
  lit.ref.store %none_5, %__result__ : <none, mut localResult>
  %17 = kgen.param.constant: i1 = <0>
  kgen.return %17 : i1
}

// -----

// COM: Verify that unreachable code is ignored in the context of error handling.

!Error = !lit.struct<@Error>
!FileHandle = !lit.struct<@FileHandle>
!LegacyPointer = !lit.struct<@LegacyPointer>
!GGUFFile = !lit.struct<@GGUFFile>
!Int = !lit.struct<@Int>
!iter = !lit.struct<@my_iter>

lit.struct.decl @Error register_passable
  destructor :!lit.signature<("self": !Error, |) -> !kgen.none> @stdlib::@builtin::@stubs::@Error::@"__del__(stdlib::builtin::stubs::Error)"{
}

lit.struct.decl @Int register_passable_trivial {
  lit.struct.field value : index
}

lit.struct.decl @my_iter
  destructor :!lit.signature<[1]("self": !lit.ref<!iter, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @small::@my_iter::@"__del__(small::my_iter)"
  copy :!lit.signature<[2]("self": !lit.ref<!iter, mut *[0,0]> init_self, |, "existing": !lit.ref<!iter, imm *[0,1]> read_mem) -> !kgen.none> @small::@my_iter::@"__copyinit__(small::my_iter=&,small::my_iter)"{
  lit.struct.field start : !Int
}

lit.struct.decl @FileHandle
  destructor :!lit.signature<[1]("self": !lit.ref<!FileHandle, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @small::@FileHandle::@"__del__(small::FileHandle)"{
  lit.struct.field str : !Int
}
lit.struct.decl @LegacyPointer register_passable_trivial {
  lit.struct.field address : !Int
}

lit.struct.decl @GGUFFile
  destructor :!lit.signature<[1]("self": !lit.ref<!GGUFFile, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @GGUFFile::@__del__{
  lit.struct.field fp : !FileHandle
  lit.struct.field infos : !LegacyPointer
  // CHECK-LABEL: lit.func @__init__
  lit.func @__init__[mut selfLife, mut errorLife, mut rangeLife](
    %self: !lit.ref<!GGUFFile, mut selfLife> init_self, |,
    %iter: !lit.ref<!iter, mut rangeLife> read_mem, ?,
    %__error__: !lit.ref<!Error, mut errorLife> byref_error) throws -> i1 {


    %3 = lit.call @LegacyPointer::@alloc() : !lit.signature<() -> !LegacyPointer>
    %4 = lit.ref.struct.ger %self[infos] : <!GGUFFile, mut selfLife> -> @LegacyPointer
    lit.ref.store %3, %4 : <@LegacyPointer, mut selfLife->infos>

    %i = lit.var.decl "i" imp : !lit.ref<!Int, mut iLife>
    hlcf.loop "_loop_0" {
      %9 = lit.call @my_iter::@__len__[mut rangeLife](%iter) : !lit.signature<[1]("self": !lit.ref<!iter, mut *[0,0]> mut) -> index>
      %idx0 = index.constant 0
      %11 = index.cmp sgt(%9, %idx0)
      hlcf.if %11 {
        hlcf.yield
      } else {
        hlcf.break "_loop_0"
      }
      %12 = lit.call @my_iter::@__next__[mut rangeLife](%iter) : !lit.signature<[1]("self": !lit.ref<!iter, mut *[0,0]> mut) -> !Int>
      lit.ref.store %12, %i : <!Int, mut iLife>

      // Conditionally set use ref method
      %20 = lit.ref.struct.ger %self[infos] : <!GGUFFile, mut selfLife> -> @LegacyPointer
      %21 = lit.ref.load %20 : <@LegacyPointer, mut selfLife->infos>
      %22 = lit.ref.immut %i : <!Int, mut iLife>
      %23 = lit.call @LegacyPointer::@__getitem__[muttoimm iLife](%21, %22) : !lit.signature<[1]("self": !LegacyPointer, "offset": !lit.ref<!Int, imm *[0,0]> read_mem) -> !lit.ref<!Int, mut #lit.any.origin>>
      %__call_result_tmp__ = lit.var.decl "__call_result_tmp__" synth : !lit.ref<!Int, mut resultLife>
      %24 = lit.call @raising_function[mut errorLife, mut resultLife](%__error__, %__call_result_tmp__) : !lit.signature<[2]("__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<!Int, mut *[0,1]> byref_result) throws -> i1>
      // CHECK: %[[V0:.*]] = lit.call @raising_function[mut errorLife, mut resultLife](%__error__, %__call_result_tmp__)
      // CHECK-NEXT:  if %[[V0]]
      // CHECK-NEXT:    mark_consumed %__call_result_tmp__
      // CHECK-NEXT:    lifetime.end %__call_result_tmp__
      // CHECK-NEXT:    kgen.param.constant: i1 = <1>
      // CHECK-NEXT:    lit.error_return
      // CHECK-NEXT:  } else {
      // CHECK-NEXT:    mark_consumed %__error__
      // CHECK-NEXT:    yield
      // CHECK-NEXT:  }
      hlcf.if %24 {
        lit.ownership.mark_consumed %__call_result_tmp__ : <!Int, mut resultLife>
        %26 = kgen.param.constant: i1 = <1>
        lit.error_return %26 : i1
      } else {
        lit.ownership.mark_consumed %__error__ : <!Error, mut errorLife>
        hlcf.yield
      }
      // CHECK: lit.load.consume %__call_result_tmp__
      %25 = lit.load.consume %__call_result_tmp__ : !lit.ref<!Int, mut resultLife>
      // CHECK-NEXT: lifetime.end %__call_result_tmp__
      lit.ref.store %25, %23 : <!Int, mut #lit.any.origin>
      hlcf.continue
    }
    // Causes bits in the self to be reset, which will trigger erroneous destructors if unreachable code is not ignored.
    %6 = lit.ref.struct.ger %self[fp] : <!GGUFFile, mut selfLife> -> !FileHandle
    %7 = lit.call @FileHandle::@__init__[mut selfLife->fp](%6) : !lit.signature<[1]("self": !lit.ref<!FileHandle, mut *[0,0]> init_self) -> !kgen.none>
    %8 = kgen.param.constant: i1 = <0>
    kgen.return %8 : i1
  }
}
