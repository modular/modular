# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that control flow statements in struct/trait/extension bodies emit
# errors instead of crashing the compiler (MOCO-3870).

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Control flow statements in struct body
##===----------------------------------------------------------------------===##

struct StructWithIf:
    var x: Int
    # expected-error @below {{'if' must be contained in a function}}
    if True:
        pass

struct StructWithFor:
    var x: Int
    # expected-error @below {{'for' must be contained in a function}}
    for i in range(10):
        pass

struct StructWithWhile:
    var x: Int
    # expected-error @below {{'while' must be contained in a function}}
    while True:
        pass

struct StructWithComptimeIf:
    var x: Int
    # expected-error @below {{'comptime if' must be contained in a function}}
    comptime if True:
        pass

struct StructWithComptimeFor:
    var x: Int
    # expected-error @below {{'comptime for' must be contained in a function}}
    comptime for i in range(10):
        pass

struct StructWithWith:
    var x: Int
    # expected-error @below {{'with' must be contained in a function}}
    with foo:
        pass

##===----------------------------------------------------------------------===##
# Control flow statements in trait body
##===----------------------------------------------------------------------===##

trait TraitWithIf:
    # expected-error @below {{'if' must be contained in a function}}
    if True:
        pass

trait TraitWithFor:
    # expected-error @below {{'for' must be contained in a function}}
    for i in range(10):
        pass

trait TraitWithWhile:
    # expected-error @below {{'while' must be contained in a function}}
    while True:
        pass

trait TraitWithWith:
    # expected-error @below {{'with' must be contained in a function}}
    with foo:
        pass

##===----------------------------------------------------------------------===##
# Control flow statements in extension body
##===----------------------------------------------------------------------===##

struct ExtendedStruct:
    var x: Int

__extension ExtendedStruct:
    # expected-error @below {{'if' must be contained in a function}}
    if True:
        pass

struct ExtendedStruct2:
    var x: Int

__extension ExtendedStruct2:
    # expected-error @below {{'for' must be contained in a function}}
    for i in range(10):
        pass

struct ExtendedStruct3:
    var x: Int

__extension ExtendedStruct3:
    # expected-error @below {{'while' must be contained in a function}}
    while True:
        pass

struct ExtendedStruct4:
    var x: Int

__extension ExtendedStruct4:
    # expected-error @below {{'comptime if' must be contained in a function}}
    comptime if True:
        pass

struct ExtendedStruct5:
    var x: Int

__extension ExtendedStruct5:
    # expected-error @below {{'comptime for' must be contained in a function}}
    comptime for i in range(10):
        pass

struct ExtendedStruct6:
    var x: Int

__extension ExtendedStruct6:
    # expected-error @below {{'with' must be contained in a function}}
    with foo:
        pass
