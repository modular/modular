// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file -verify-diagnostics %s

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
