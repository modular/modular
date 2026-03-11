# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --mojo-disable-builtins -I %S/inputs %s -split-input-file -verify-diagnostics

# Test suite for ambiguous import detection (MOCO-3264).
# Each section is an independent compilation unit.
#
# Input module structure: each test scenario uses a dedicated pair of input
# packages (e.g. struct_a/struct_b, fn_a/fn_b). This looks redundant — a
# single pair of packages with multiple differently-named exports and aliased
# imports (`from mod_a import StructFoo as Foo`) could in principle cover all
# cases. However, -verify-diagnostics checks @expected-note annotations in ALL
# SourceMgr buffers, including imported modules. Because different test sections
# trigger notes at different declaration sites within the same shared module,
# a shared module cannot satisfy all sections simultaneously without spurious
# "expected note not found" failures. Dedicated per-scenario packages avoid
# this constraint cleanly.

# ---------------------------------------------------------------------------
# Structs: importing two different structs with the same name produces an
# error at the second import statement, not at the use site.

from struct_a import Foo
# @expected-error @below {{import of 'Foo' is ambiguous}}
from struct_b import Foo

# After the import error the first struct is still usable.
fn use_it[x: Foo]():
    pass

# // -----

# ---------------------------------------------------------------------------
# Type aliases (comptime): same rule applies — ambiguous import detected at
# the second import statement.

from comptime_a import Foo
# @expected-error @below {{import of 'Foo' is ambiguous}}
from comptime_b import Foo

fn use_alias[x: Foo]():
    pass

# // -----

# ---------------------------------------------------------------------------
# Functions: importing functions with the same name from different modules is
# legal at import time — they form an overload set. No import-time error.

from fn_a import Foo
from fn_b import Foo

# Foo() unambiguously calls fn_a.Foo (no args); fn_b.Foo requires an i1 arg.
fn call_it():
    Foo()

# // -----

# ---------------------------------------------------------------------------
# Raw MLIR types (comptime __mlir_type): same rule applies — ambiguous import
# detected at the second import statement.

from mlirtype_a import Foo
# @expected-error @below {{import of 'Foo' is ambiguous}}
from mlirtype_b import Foo

fn use_mlirtype[x: Foo]():
    pass

# // -----

# ---------------------------------------------------------------------------
# Mixed struct and function: importing a struct and a function with the same
# name is an error — functions and non-functions cannot share a name, matching
# the rule for local declarations.

from struct_c import Foo
# @expected-error @below {{import of 'Foo' is ambiguous}}
from fn_c import Foo

# // -----

# ---------------------------------------------------------------------------
# Mixed function and struct: same rule as above but with the order reversed —
# function imported first, then a struct with the same name. Exercises the
# incomingNonFn+existingFn branch of checkImportNamingConflict.

from fn_d import Foo
# @expected-error @below {{import of 'Foo' is ambiguous}}
from struct_d import Foo

# // -----

# ---------------------------------------------------------------------------
# Duplicate import: importing the same name from the same module twice is
# legal and must not be treated as a conflict — both import statements
# resolve to the same ASTDecl, which is excluded from the existing-set scan
# via the skip parameter in checkImportNamingConflict.

from struct_dup import Foo
from struct_dup import Foo  # no error

fn use_dup[x: Foo]():
    pass
