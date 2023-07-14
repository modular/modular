# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics -split-input-file %s

struct Crash1[XXX: __mlir_type.index]:
  fn __init__(a: __mlir_type.float32)
  # expected-error @-1 {{expected ':' in function definition}}
    pass

  fn f(self): pass

// -----

# Forward reference needs to get resolved.
fn printFloat32(x: Float32):
  pass

struct Float32:
  var value : __mlir_type.`!pop.scalar<f32>`

// -----
# Crashed at top level.

# expected-error @below {{MLIR attribute is not a TypedAttr}}
__mlir_attr.`#index<cmp_predicate eq>`

// -----
# MLIR Symbol Redefinition crash

struct Foo: pass # expected-note {{previous definition here}}
struct Foo: pass # expected-error {{invalid redefinition of 'Foo'}}

fn x(a: Foo): pass

// -----

fn test():
  alias x: __mlir_type.index

// -----

# Issue #6874: Cannot use aliased variables as return value
struct XDType[_type: __mlir_type.`!kgen.dtype`]:
    alias type = _type

    fn getType(self) -> __mlir_type.`!kgen.dtype`:
      return Self.type

// -----

# The octal escape sequence in string literals \ooo can have variable length.
fn testOctal():
  var x = "A\0"
  x = "A\01"
  x = "A\012"

// -----

fn testTripleQuote():
  # expected-error @below {{invalid escape sequence}}
  var x = """$\s$"""
