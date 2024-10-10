# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Lexical Issues
##===----------------------------------------------------------------------===##

# https://github.com/modularml/modular/issues/4181
struct Issue4181IndentWeirdness[dt: DType]:
  var b : Int
    # expected-error @+1 {{definition isn't on its own line at the correct indentation}}
    def f():
      pass

# Failed to parse due to indentation.
fn issue_6291(
    val: __mlir_type.index
) -> __mlir_type.index:
    return val

fn testIndentation6291[index: __mlir_type.index](
    ptr: __mlir_type.`!kgen.pointer<!pop.scalar<index>>`):
  var result = __mlir_op.`pop.load`[
            alignment=__mlir_attr.`1: index`,
            _type=__mlir_type.`!pop.scalar<index>`
](ptr)

# This file contains parsing related bugs.

fn bracketError1():
  _ = ] # expected-error {{unexpected token in expression}}

fn bracketError2():
  _ = [[1, 2], }# expected-error {{unexpected token in expression}}


# Indentation errors
fn nothing(): pass

fn test_indentation1():
  nothing()   # expected-note {{indentation should match previous statement}}
    nothing() # expected-error {{statement has excess indentation}}

fn test_indentation2(p: Bool):
  nothing()
  if p:   # expected-note {{indentation should match previous statement}}
      nothing()
   nothing() # expected-error {{statement has excess indentation}}


# Decorator processing.
# https://github.com/modularml/mojo/issues/1655
@ : # expected-error {{unexpected token in expression}}
    fn a  # expected-error {{expected '(' for argument list}}

# https://github.com/modularml/mojo/issues/1230
# Parser crashes on incomplete decorator
@ # expected-error {{missing decorator expression after '@'}}
fn m # expected-error {{expected '(' for argument list}}

# Issue #6909
# expected-error @below {{expected name for 'alias' declaration}}
# expected-note @below {{escape keyword 'True' with backticks to use it as an identifier}}
alias True = 42
