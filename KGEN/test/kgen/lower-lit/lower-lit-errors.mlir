// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file -verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// -----
// expected-error @-2 {{cyclic dependencies between global variables in 'lower-lit' pass}}

lit.globalvar.decl @foo : index {
  lit.globalvar.ref @bar : <index, mut #lit.any.origin>
}, {
}

lit.globalvar.decl @bar : index {
  lit.globalvar.ref @foo : <index, mut #lit.any.origin>
}, {
}

//===----------------------------------------------------------------------===//
// Recursive type via parameter
//===----------------------------------------------------------------------===//

// -----

lit.struct.decl @foo<T: type> register_passable {
  lit.struct.field field : !kgen.struct<(T)>
}

// expected-error @below {{struct has recursive reference to itself}}
lit.struct.decl @bar register_passable {
  lit.struct.field address : !lit.struct<@foo<:type @bar>>
}
