// RUN: kgen-opt %s -split-input-file -check-lifetimes -verify-diagnostics

lit.struct.decl @S attributes {
  destructor =
    #kgen.symbol.constant<@S::@__del__> : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
  lit.func @__init__[mut selflife](%self: !lit.ref<@S, mut selflife> init_self, |) -> !kgen.none attributes {isStatic} {
    %0 = lit.ref.struct.ger %self[a] : <index, mut selflife> from @S
    %idx1 = index.constant 1
    lit.ref.store %idx1, %0 : !lit.ref<index, mut selflife>

    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

lit.func @print(%borrowMe: !lit.ref<@S, mut #lit.lifetime> borrow_in_mem) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}

// expected-note @below {{'__result__' declared here}}
lit.func @elifInitError[mut *"__result__`0"](?, %cond: i1, %__result__[__result__]: !lit.ref<@S, mut *"__result__`0"> byref_result) -> !kgen.none {
  hlcf.elif {
    hlcf.elif.yield %cond : i1
  } then {
    %0 = lit.call @S::@__init__[mut *"__result__`0"](%__result__) : !lit.signature<[1](!lit.ref<@S, mut *"__result__`0"> init_self, |) -> !kgen.none>
    hlcf.yield
  } else {
    hlcf.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  // expected-error @below {{'__result__' is uninitialized at return from this function}}
  kgen.return %none : !kgen.none
}
