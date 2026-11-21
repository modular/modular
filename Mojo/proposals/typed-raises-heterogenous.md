# Heterogeneous error handling

Status: Draft; November 14, 2026

Mojo can catch different error types in separate `except` clauses. This
proposal adopts Python's typed-handler spelling, adds `except as error:` for a
type-inferred catch-all, and phases out `except error:`.

## Introduction

Mojo functions have typed raises: a raising function identifies the type of the
value it can raise, and a `try` statement binds the caught value with that
type. This works naturally when every raising operation in a `try` body
produces the same type:

```mojo
def parse_header() raises ParseError:
    ...

def parse_body() raises ParseError:
    ...

try:
    parse_header()
    parse_body()
except error:
    handle(error)
```

The Mojo 1.0 inference model did not generalize to a body containing different
error types. The first raising operation encountered fixed the type of `error`,
and later operations had to raise that same type or convert to it:

```mojo
def parse_input() raises ParseError:
    ...

def read_input() raises IOError:
    ...

try:
    read_input()
    parse_input()
except error:
    handle(error)
```

In November 2026, the compiler started merging the error types in a `try` body
and falling back to `Error` when every error can convert to it. This is useful
because all `Writable` types convert to `Error`.

This proposal extends that model to:

1. Allow handlers for multiple error types.
2. Align the syntax with Python.
3. Migrate existing code gradually instead of making a hard break.

## Detailed design

Mojo currently has two `except` forms:

- `except:` catches every error without binding its value.
- `except e:` catches every error and binds it to `e`.

The second form conflicts with Python, where `except StopIteration:` catches an
error of type `StopIteration`, while `except StopIteration as e:` catches that
type and binds the value to `e`. See the
[Python `try` statement grammar](https://docs.python.org/3/reference/compound_stmts.html#the-try-statement).

### Step 1: Accept `except Type as name:`

Mojo accepts Python's published syntax for filtering on an error type and
naming the bound value:

```mojo
try:
    read_input()
except IOError as error:
    recover_io(error)
```

### Step 2: Add `except as name:`

The current `except error:` spelling infers the caught error type, but Python
compatibility requires a type name before an identifier in an `except` header.
Requiring `except Error as error:` would erase useful type information.

Mojo therefore adds `except as error:`. It catches every remaining error,
merges their types using the existing `try` error-type rules, and binds the
result to `error`.

```mojo
try:
    parse_header()
    parse_body()
except as error:
    handle(error)
```

### Step 3: Add multiple handlers

A `try` statement may have multiple `except` clauses:

```mojo
try:
    read_input()
    parse_input()
except IOError as error:
    recover_io(error)
except ParseError as error:
    report_parse_error(error)
except as error:
    report_other_error(error)
```

The compiler tracks the error types that can escape the `try` body and examines
handlers in source order:

- A typed handler receives an error whose type exactly matches, or implicitly
  converts to, the type written in its header. Its binding has that written
  type.
- `except as error:` catches every error not handled by an earlier clause. The
  compiler merges the remaining types to determine the type of `error`.
- `except:` catches every remaining error without binding it.

Any error that can reach the end of the handler list is a compile-time error:

```mojo
try:
    read_input()
    parse_input()
except IOError as error:
    recover_io(error)
```

The example leaves `ParseError` unhandled. The program can explicitly
propagate it:

```mojo
try:
    read_input()
    parse_input()
except IOError as error:
    recover_io(error)
except:
    raise
```

The compiler also diagnoses handlers that cannot receive a remaining error:

```mojo
try:
    read_input()
    parse_input()
except Error:
    # Handles both errors because they implicitly convert to Error.
    ...
except IOError:
    # Warning: unreachable `except` clause.
    ...
```

Implicit convertibility keeps typed handlers consistent with the current
`try`-body merge rules. It also prepares the model for dynamic types and
subtyping: when Mojo supports classes, `except BaseException:` can handle a
`DerivedException` just as it does in Python.

### Step 4: Migrate `except name:`

Existing Mojo code uses `except error:` to bind the caught value. Changing its
meaning immediately would be source incompatible.

During the migration, the compiler resolves a single identifier in an `except`
header as follows:

1. If it resolves to a type, the clause is a typed handler.
2. Otherwise, the clause uses the legacy catch-all binding syntax.

Type names are normally capitalized and value names are normally lowercase, so
these forms are unlikely to conflict in practice. A future release can warn on
the legacy form, then make it an error after an extended migration period.

## Rationale

Multiple handlers preserve useful error types instead of forcing all errors in
a `try` body into one representation. A handler can recover from `IOError`
without losing the structure of `ParseError`.

Python's type-first spelling clearly distinguishes a type filter from a value
binding. `except as error:` retains Mojo's useful inferred catch-all without
forcing an imprecise `Error` type.

The staged migration keeps existing `except error:` code working while giving
developers time to adopt the Python-compatible syntax.
