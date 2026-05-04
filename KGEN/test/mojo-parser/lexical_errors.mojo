# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics

##===----------------------------------------------------------------------===##
# Lexical Issues
##===----------------------------------------------------------------------===##

# https://github.com/modularml/modular/issues/4181
struct Issue4181IndentWeirdness[dt: DType]:
  var b : Int
    # expected-error @+1 {{definition isn't on its own line at the correct indentation}}
    def f() raises:
      pass

# Failed to parse due to indentation.
def issue_6291(
    val: __mlir_type.index
) -> __mlir_type.index:
    return val

def testIndentation6291[index: __mlir_type.index](
    ptr: __mlir_type.`!kgen.pointer<!kgen.scalar<index>>`):
  var result = __mlir_op.`pop.load`[
            alignment=__mlir_attr.`1: index`,
            _type=__mlir_type.`!kgen.scalar<index>`
](ptr)

# This file contains parsing related bugs.

def bracketError1():
  _ = ] # expected-error {{unexpected token in expression}}

def bracketError2():
  _ = [[1, 2], }# expected-error {{unexpected token in expression}}


# Indentation errors
def nothing(): pass

def test_indentation1():
  nothing()
    nothing() # expected-error {{statement indentation must match the rest of the block; adjust to align}}

def test_indentation2(p: Bool):
  nothing()
  if p:
      nothing()
   nothing() # expected-error {{statement indentation must match the rest of the block; adjust to align}}

# Decorator processing.
# https://github.com/modular/mojo/issues/1655
@ : # expected-error {{unexpected token in expression}}
    def a  # expected-error {{expected '(' for argument list}}

# https://github.com/modular/mojo/issues/1230
# Parser crashes on incomplete decorator
@ # expected-error {{found stray '@'; '@' must be followed by a decorator name}}
def m # expected-error {{expected '(' for argument list}}

# Issue #6909
# expected-error @below {{invalid comptime declaration: expected an identifier or '_'}}
comptime True = 42

