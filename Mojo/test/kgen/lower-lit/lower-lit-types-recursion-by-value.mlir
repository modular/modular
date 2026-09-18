// RUN: kgen-opt %s -lower-lit -split-input-file -verify-parameters \
// RUN:   -verify-diagnostics

// Struct layouts that genuinely contain themselves, so they have no finite
// size. This is the only struct recursion that is illegal, and it is decided
// before lowering: a parameter position counts only when building the layout
// of the struct requires building the layout of the argument in it.

//===----------------------------------------------------------------------===//
// Directly by value.
//===----------------------------------------------------------------------===//

// expected-error @below {{'A' must not contain itself by value}}
lit.struct.decl @A {
  lit.struct.field a : !lit.struct<@A>
}

// -----

//===----------------------------------------------------------------------===//
// Through a wrapper that holds its parameter by value. The pointer-wrapper
// version of this shape is finite and lowers - see
// lower-lit-types-recursion-through-pointer.mlir.
//===----------------------------------------------------------------------===//

lit.struct.decl @Box<ty: type> register_passable {
  lit.struct.field val : !kgen.param<ty>
}

// expected-error @below {{'Node' must not contain itself by value}}
lit.struct.decl @Node {
  lit.struct.field next : !lit.struct<@Box<:type !lit.struct<@Node>>>
}

// -----

//===----------------------------------------------------------------------===//
// An unbounded family of distinct instantiations, each holding the next by
// value. Unboundedness is not what makes this illegal - the same shape through
// a pointer wrapper has a finite layout - so the check is by-value-ness, over
// declarations rather than instantiations.
//===----------------------------------------------------------------------===//

lit.struct.decl @Box<ty: type> register_passable {
  lit.struct.field val : !kgen.param<ty>
}

// expected-error @below {{'Rec' must not contain itself by value}}
lit.struct.decl @Rec<ty: type> {
  lit.struct.field next : !kgen.struct<(!lit.struct<@Box<:type
      !lit.struct<@Rec<:type !lit.struct<@Box<:type ty>>>>>>)>
}
