// RUN: kgen-opt %s -split-input-file -check-lifetimes -verify-diagnostics | FileCheck %s


// COM: Test all fields are destroyed in object destructor

lit.struct.decl @S attributes {destructor = #kgen.symbol.constant<@S::@__del__> : !lit.generator<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

// CHECK-LABEL: lit.struct.decl @HasMemFields
lit.struct.decl @HasMemFields attributes {destructor = #kgen.symbol.constant<@HasMemFields::@__del__> : !lit.generator<[1](!lit.ref<@HasMemFields, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : !lit.struct<@S>
  lit.struct.field stole : !lit.struct<@S>
  lit.struct.field uninitialized : !lit.struct<@S>
  lit.struct.field register : index

  lit.fn @__del__[mut dellife](%self: !lit.ref<@HasMemFields, mut dellife> owned_in_mem) -> !kgen.none {
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

// CHECK-LABEL: lit.fn @mark_initialized
lit.fn @mark_initialized[mut lt](%arg: !lit.ref<@HasMemFields, mut lt> byref_result) {
  // CHECK-NEXT: lit.ownership.mark_initialized %arg
  lit.ownership.mark_initialized %arg : <@HasMemFields, mut lt>
  kgen.return
}


// CHECK-LABEL: lit.fn @resolved_fn
// CHECK-NEXT:  lit.call @HasMemFields::@__del__{{.*}}(%arg0)
// CHECK-NEXT:  kgen.return
lit.fn @resolved_fn(%arg0: !lit.ref<@HasMemFields, mut dellife> owned_in_mem) {
  kgen.return
}

// CHECK-LABEL: lit.fn @unresolved_fn
// CHECK-NEXT: lit.end_fn unresolved
lit.fn @unresolved_fn(%arg0: !lit.ref<@HasMemFields, mut dellife> owned_in_mem) {
  // Don't process this function or insert the destructor call.  The structdecl
  // might not be resolved.
  lit.end_fn unresolved
}

// -----

// COM: Test that destructors are inserted for error instances.

!Error = !lit.struct<@Error>

// CHECK-LABEL: lit.struct.decl @Error
lit.struct.decl @Error register_passable attributes {destructor = #kgen.symbol.constant<@Error::@__del__ > : !lit.generator<(!Error) -> !kgen.none>} {
  lit.struct.field a : index
}

lit.fn @consume_err(%value: !Error) {
  kgen.return
}

// CHECK-LABEL: lit.fn @conditional_consumption_1
// Issue#34320: https://github.com/modularml/modular/issues/34320
lit.fn @conditional_consumption_1(%c: i1, %value: !Error) {
  // CHECK-NOT: @Error::@__del__
  hlcf.loop {
    hlcf.if %c {
      lit.call @consume_err(%value) : !lit.generator<(!Error) -> ()>
      hlcf.break
    } else {
      hlcf.yield
    }
    lit.call @consume_err(%value) : !lit.generator<(!Error) -> ()>
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: lit.fn @conditional_consumption_2
lit.fn @conditional_consumption_2(%c: i1, %value: !Error) {
  // CHECK-NOT: @Error::@__del__
  hlcf.loop {
    hlcf.if %c {
      hlcf.yield
    } else {
      lit.call @consume_err(%value) : !lit.generator<(!Error) -> ()>
      hlcf.break
    }
    lit.call @consume_err(%value) : !lit.generator<(!Error) -> ()>
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: lit.fn @conditional_consumption_3
lit.fn @conditional_consumption_3(%c: i1, %value: !Error) {
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
      lit.call @consume_err(%value) : !lit.generator<(!Error) -> ()>
      hlcf.break
    } else {
      lit.try.yield
    }
    lit.call @consume_err(%value) : !lit.generator<(!Error) -> ()>
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: lit.fn @conditional_consumption_4
lit.fn @conditional_consumption_4(%c: i1, %value: !Error) {
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
      lit.call @consume_err(%value) : !lit.generator<(!Error) -> ()>
      hlcf.break
    }
    lit.call @consume_err(%value) : !lit.generator<(!Error) -> ()>
    hlcf.break
  }
  kgen.return
}

// -----

!Thing = !lit.struct<@Thing>
lit.struct.decl @Box<T: trait<@AnyType>>  {
  lit.struct.field x : !kgen.param<:trait<@AnyType> T>
}

lit.struct.decl @Thing {
  lit.struct.field x : index
  lit.struct.field y : index
  lit.struct.field z : index
  lit.fn @get(%self: !lit.ref<!Thing, imm #lit.any.origin> read_mem) {
    kgen.return
  }
}

lit.fn @top(%c: !lit.ref<@Box<:trait<@AnyType> !Thing>, mut #lit.any.origin> read_mem) {
  %0 = lit.ref.struct.ger %c[x] : <@Box<:trait<@AnyType> !Thing>, mut #lit.any.origin> -> !Thing
  lit.call @Thing::@get(%0) : !lit.generator<("self": !lit.ref<!Thing, mut #lit.any.origin> read_mem) -> ()>
  kgen.return
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
    #kgen.symbol.constant<@Node::@__del__> : !lit.generator<[1](!lit.ref<@Node, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

!Container = !lit.struct<@Container>
lit.struct.decl @Container attributes {
  destructor =
    #kgen.symbol.constant<@Container::@__del__> : !lit.generator<[1](!lit.ref<@Container, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field z : !Node
}

!Wrapper = !lit.struct<@Wrapper>
lit.struct.decl @Wrapper attributes {destructor = #kgen.symbol.constant<@Wrapper::@__del__> : !lit.generator<[1](!lit.ref<@Wrapper, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field tail : !Int
  lit.struct.field y : !kgen.pointer<!Container>
}

// -----

!Error = !lit.struct<@Error>
lit.struct.decl @Error {
  lit.struct.field a : index
}

!PythonObject = !lit.struct<@PythonObject>
lit.struct.decl @PythonObject attributes {
  destructor =
    #kgen.symbol.constant<@PythonObject::@__del__> : !lit.generator<[1](!lit.ref<@PythonObject, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

!Context = !lit.struct<@Context>
lit.struct.decl @Context attributes {destructor = #kgen.symbol.constant<@Context::@__del__> : !lit.generator<[1](!lit.ref<@Context, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field __new_repl_var : !kgen.pointer<pointer<!PythonObject>>
  lit.struct.field __new_repl_var2 : !kgen.pointer<pointer<!PythonObject>>
}

// CHECK-LABEL: lit.fn @createConditionallyInitializedImmortalReferenceInRepl
lit.fn @createConditionallyInitializedImmortalReferenceInRepl[mut topArg, mut localError, mut localResult](
  %__mojo_repl_arg : !lit.ref<!Context, mut topArg> mut,?,
  %__error__: !lit.ref<!Error, mut localError> byref_error,
  %__result__: !lit.ref<none, mut localResult> byref_result) throws|capturing -> i1 {

  %2 = lit.ref.struct.ger %__mojo_repl_arg[__new_repl_var] : <!Context, mut topArg> -> pointer<pointer<!PythonObject>>
  %3 = lit.ref.load %2 : <pointer<pointer<!PythonObject>>, mut topArg->__new_repl_var>
  %index_3 = kgen.param.constant: index = <get_sizeof(!PythonObject, current_target())>
  %index_4 = kgen.param.constant: index = <get_alignof(!PythonObject, current_target())>
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
  %6 = lit.call @import_module[mut localError, mut LOCAL_LIFETIME2](%__error__, %5) : !lit.generator<[2](?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<!PythonObject, mut *[0,1]> byref_result) throws -> i1>
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
  %16 = lit.call @import_module[mut localError, mut LOCAL_LIFETIME3](%__error__, %15) : !lit.generator<[2](?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<!PythonObject, mut *[0,1]> byref_result) throws -> i1>
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
