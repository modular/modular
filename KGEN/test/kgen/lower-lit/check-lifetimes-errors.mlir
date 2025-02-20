// RUN: kgen-opt %s -split-input-file -check-lifetimes -verify-diagnostics

lit.struct.decl @S attributes {
  destructor =
    #kgen.symbol.constant<@S::@__del__> : !lit.generator<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
  lit.fn @__init__[mut selflife](%self: !lit.ref<@S, mut selflife> byref_result, |) -> !kgen.none attributes {sourceName = "__init__", specialFnKind = 2 : i8} {
    %0 = lit.ref.struct.ger %self[a] : <@S, mut selflife> -> index
    %idx1 = index.constant 1
    lit.ref.store %idx1, %0 : !lit.ref<index, mut selflife->a>

    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

lit.fn @print(%borrowMe: !lit.ref<@S, mut #lit.any.origin> read_mem) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}

// expected-note @below {{'__result__' declared here}}
lit.fn @elifInitError[mut *"__result__`0"](?, %cond: i1, %__result__[__result__]: !lit.ref<@S, mut *"__result__`0"> byref_result) -> !kgen.none {
  hlcf.elif {
    hlcf.elif.yield %cond
  } then {
    %0 = lit.call @S::@__init__[mut *"__result__`0"](%__result__) : !lit.generator<[1](!lit.ref<@S, mut *"__result__`0"> byref_result, |) -> !kgen.none>
    hlcf.yield
  } else {
    hlcf.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  // expected-error @below {{'__result__' is uninitialized at return from this function}}
  kgen.return %none : !kgen.none
}

// @__nonimpldestructible struct Linear:
lit.struct.decl @Linear attributes {
  linearTypeErrorMsg = "'Linear' isn't implicit destructible, call the 'close' or 'explode' methods to explicitly destroy it"
}{
  // fn close(owned self): pass
  lit.fn @close[mut dellife](%self: !lit.ref<@Linear, mut dellife> owned_in_mem) {

    // expected-error @below {{'Linear' isn't implicit destructible, call the 'close' or 'explode' methods to explicitly destroy it}}
    lit.ownership.use %self: !lit.ref<@Linear, mut dellife>
    kgen.return
  }
}

// CHECK-LABEL: lit.fn @useLinear
lit.fn @useLinear(
  %b: !lit.ref<@Linear, mut #lit.any.origin> owned_in_mem) {

  // expected-error @below {{'Linear' isn't implicit destructible, call the 'close' or 'explode' methods to explicitly destroy it}}
  lit.ownership.use %b: !lit.ref<@Linear, mut #lit.any.origin>
  kgen.return
}
