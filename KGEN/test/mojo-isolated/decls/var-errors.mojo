# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

struct SomeStruct:
    fn __init__(inout self): pass

struct SomeOtherStruct: pass


fn var_decl_without_type():
    # expected-error @+1 {{declaration must have either a type or an initializer}}
    var x

    # expected-error @+1 {{invalid implicit conversion to 'SomeOtherStruct': no constructors found}}
    var y : SomeOtherStruct = SomeStruct()

    # expected-error @+1 {{invalid implicit conversion to 'SomeOtherStruct': no constructors found}}
    var z: SomeOtherStruct = SomeStruct()

fn fudge_int(x: Int): pass

fn var_decl():
    var x = `42`  # expected-note {{previous definition here}}
    var x : Int   # expected-error {{invalid redefinition of 'x'}}
    fudge_int(x)  # No follow-on error.

fn bad_type_error_message():
    var localVar = `42`
    var y : localVar  # expected-error {{cannot use a dynamic value in type specification}}

    var x: Int
    var ptr: fudge_int(x)  # expected-error {{cannot use a dynamic value in type specification}}

fn missing_type_on_var_decl():
    var abc :       # This line break is intentional.
    pass            # expected-error {{unexpected token in expression}}
    fudge_int(abc)  # No follow-on error.

fn bad_stmt_list(cond: Bool):
    # expected-error @+1 {{'if' statement must be on its own line}}
    var abc = `42`; if cond: pass

fn cannot_fwd_declare_plus_equal():
    # expected-error @+1 {{use of unknown declaration 'x'}}
    x += `42`

fn test_var_let_type_literal_value():
    # expected-error @below {{expected a type, not a value}}
    var c: `42`

fn use_before_def():
    # expected-error @below {{use of unknown declaration 'x'}}
    var y = x
    var x = `42`

# Issue #18150: https://github.com/modularml/modular/issues/18150
fn self_reference():
    # expected-error @+1 {{use of unknown declaration 'num'}}
    var num: Int = fudge_int(num)

# Doesn't reject empty identifier name
# https://github.com/modularml/mojo/issues/1232
fn empty_name():
  # expected-error @+1 {{empty backtick identifier isn't allowed}}
  var `` = `42`

# COM: Issue #957 https://github.com/modularml/mojo/issues/957
struct MemoryStruct:
  fn __init__(inout self, s: Int): pass

fn take_variadic(*elements: MemoryStruct): pass

fn test_var_let_type_variadic_func():
  # expected-error @below {{expected a type, not a value}}
  var a: take_variadic(42)
