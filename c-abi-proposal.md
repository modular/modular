# Mojo Calling Conventions: The `abi` Effect

## TL;DR

Mojo currently has no way to annotate a function — or a function pointer type —
with a specific calling convention. This creates silent ABI mismatches when
interfacing with C, and leaves `@export` semantically incomplete. This proposal
introduces `abi` as a first-class function effect keyword, analogous to `raises`
or `async`, and tightens the `@export` decorator to require an explicit ABI
declaration.

The syntax follows the pattern established by Rust's `extern "C"` and Swift's
`@convention(c)`, but fits naturally into Mojo's existing effect system. A
function that wants to be callable from C says so explicitly:

```mojo
def add(a: Int32, b: Int32) abi("C") -> Int32:
    return a + b
```

The same annotation works on function pointer types, which is the primary reason
`abi` must be a syntactic effect rather than a decorator argument: decorators
cannot appear inside type expressions.

---

## Current State and Pain Points

### `@export(abi="C")` is syntactically accepted but semantically wrong

`@export("my_func", abi="C")` is supposed to produce a symbol that any C program
can call without a special header. In practice, `processCExportedFunction`
implements a custom struct-flattening scheme instead of following the platform C
ABI (System V x86-64 / ARM64 AAPCS). Concretely:

- A `{Int, Int}` return is lowered to two output pointer parameters instead
  of being returned in `RAX:RDX` (x86-64) or `X0:X1` (ARM64) as C expects.
- A `{Float32, Float32}` argument becomes two `XMM` registers instead of one
  packed `Float64` register on x86-64.

The generated header is self-consistent with the implementation, so code
compiled *against the generated header* works. But any C code that writes a
natural struct declaration for the same function will silently get the wrong
calling convention. PR [#78709](https://github.com/modularml/modular/pull/78709)
fixes the lowering; this proposal provides the language-level contract that
makes the fix meaningful.

### There is no way to mark a function as C-ABI without exporting it

If a Mojo function needs to be passed as a callback into a C library — but
should not be a public export symbol — there is currently no way to express
that. The function pointer will use Mojo's calling convention, which diverges
from C ABI on structs, causing silent value corruption at the call site.

### There is no way to annotate a function pointer type as C-compatible

`DLHandle` and any user code that loads symbols from dynamic libraries freely
bitcasts a "pointer to C function" to a "pointer to Mojo function". Since the
two calling conventions diverge on struct arguments and return values, this
assumption is wrong and will silently corrupt values. There is no type-level
mechanism to distinguish a `fn(Int32) -> Int32` using Mojo CC from one using
C CC, so the compiler cannot catch the mismatch.

---

## Proposal

### 1. The `abi` effect on function declarations

`abi` is a new soft keyword that appears in the function effect position,
between the parameter list and the return type arrow. This is the same
syntactic slot used by `raises` and `async`:

```mojo
# C calling convention — struct args/returns follow platform C ABI rules
def add(a: Int32, b: Int32) abi("C") -> Int32:
    return a + b

# Explicit Mojo calling convention (default; rarely needs to be written)
def internal_helper(x: Int) abi("Mojo") -> Int:
    return x + 1

# Effects compose naturally
async def fetch(url: String) raises abi("C") -> Int32:
    ...
```

**Semantics:**

- `abi("C")` — the function uses the platform C calling convention (System V
  x86-64 ABI on Linux/macOS x86-64, ARM64 AAPCS on Apple Silicon / AArch64
  Linux). Struct arguments and return values are classified and passed exactly
  as a C compiler would.
- `abi("Mojo")` — the function uses Mojo's native calling convention. This is
  the default when no `abi` effect is written.
- The ABI string is case-insensitive (`"c"` and `"C"` are equivalent).
- Additional ABI strings may be added in the future (e.g. `"C++"`, SPIR-V
  shader types).

**Implementation note:** `abi` is added as a bit field in `FnEffects`, similar
to `raises`. The MLIR lowering uses the ABI annotation to drive LLVM calling
convention selection and struct-coercion logic.

### 2. The `abi` effect on function pointer types

The `abi` effect appears in the same syntactic position inside a function type
expression:

```mojo
# A function pointer that points to a C function
var c_callback: def(Int32) abi("C") -> Int32

# Used as a parameter type
def register_callback(cb: def(Int32) abi("C") -> Void):
    ...

# A higher-order function that returns a C-compatible callback
def make_handler() -> def(Int32, Int32) abi("C") -> Int32:
    ...
```

This is the key reason `abi` must be a syntactic effect rather than a
decorator argument: decorators cannot appear inside type expressions. Rust
solves this identically with `extern "C" fn(i32) -> i32`; Swift uses
`@convention(c) (Int32) -> Int32`. Mojo's `abi("C")` in the effect slot
follows the same pattern while staying consistent with the language's existing
effect syntax.

The compiler enforces ABI compatibility at assignment and call sites:
a `def(Int32) abi("C") -> Int32` and a `def(Int32) abi("Mojo") -> Int32` are
distinct types and are not implicitly convertible to each other.

### 3. Changes to `@export`

**3.1 Remove the `abi=` argument from `@export`**

The `abi=` keyword argument on `@export` is removed. ABI is now part of the
function declaration itself, not the export annotation. This separation of
concerns mirrors the Rust / Swift model:

- Rust: `#[no_mangle]` controls symbol visibility; `extern "C"` controls ABI.
- Swift: `@cdecl("name")` controls symbol name; `@convention(c)` or the
  calling convention of the function itself controls ABI.

**3.2 `@export` requires an explicit `abi` effect**

Any function decorated with `@export` must carry an explicit `abi` effect. The
compiler rejects an export without one:

```mojo
# ERROR: exported function must declare an ABI
@export("my_func")
def my_func(a: Int32) -> Int32:
    return a

# OK
@export("my_func")
def my_func(a: Int32) abi("C") -> Int32:
    return a
```

The rationale: exporting a function is a contract with the outside world. The
ABI is the most important part of that contract. Requiring it to be explicit
prevents accidental exposure of a Mojo-CC function as if it were C-CC — the
class of bug this entire proposal is designed to eliminate.

**3.3 The `@export` name argument remains optional**

`@export` without a name argument continues to use the Mojo identifier as the
symbol name, and `abi("C")` guarantees tha there is no name mangling.
This is unchanged:

```mojo
@export
def my_func(a: Int32) abi("C") -> Int32:
    return a
```

### 4. The `@extern` decorator and ABI

`@extern("symbol_name")` declares a Mojo function stub backed by an external
implementation. Today it is used exclusively for LLVM bitcode compiled from
Mojo (via `@export`), and it silently applies the Mojo calling convention on
both sides. This is self-consistent and correct for that use case.

However, `@extern` has no way to express the calling convention, which creates
two problems:

1. **C library calls via `@extern` are silently broken.** If a user writes
   `@extern("strlen")` expecting to call `libc`, the call goes through Mojo's
   struct-flattening convention rather than C ABI. No error is reported. The
   correct mechanism for calling C library functions today is
   `pop.external_call` (exposed as `external_call` in the standard library),
   which applies proper C ABI coercion. `@extern` should not be used for this
   purpose until it gains explicit ABI support.

2. **The Mojo-CC assumption on bitcode is implicit and unverified.** The
   compiler trusts that the bitcode was produced by the Mojo compiler and uses
   the same convention. There is no check.

With this proposal, `@extern` accepts the same `abi` effect as function
declarations, and will eventually require it:

```mojo
# Bitcode / Mojo-compiled interop — Mojo calling convention
@extern("my_add_one")
def my_add_one(x: UnsafePointer[Int32]) abi("Mojo"):
    ...

# C library function — C calling convention, proper ABI coercion
@extern("strlen")
def strlen(s: UnsafePointer[UInt8]) abi("C") -> Int:
    ...
```

There is no need for a separate `abi("llvm")` string. LLVM's default calling
convention is Mojo's native convention for non-exported functions; `"Mojo"` is
the correct name for it from the user's perspective.

### 5. `DLHandle` and dynamic symbol loading

`DLHandle.get_function[F]()` returns a function pointer of type `F`. With this
proposal, callers are expected to include the `abi` effect in `F`:

```mojo
var f = handle.get_function[def(Int32) abi("C") -> Int32]("sqrt32")
```

`DLHandle` can be updated to require that `F` carries an `abi("C")` effect,
enforcing the invariant that dynamically-loaded symbols are always called with
C ABI. A deprecation warning for bare `fn` types in `DLHandle` is appropriate
during the transition period.

---

## Prior Art

| Language                 | Function definition      | Function pointer type        | Export with C name            |
|--------------------------|--------------------------|------------------------------|-------------------------------|
| **Mojo** (this proposal) | `def f() abi("C") -> T`  | `def() abi("C") -> T`        | `@export` + `abi("C")` effect |
| **Rust**                 | `extern "C" fn f() -> T` | `extern "C" fn() -> T`       | `#[no_mangle] extern "C" fn`  |
| **Swift**                | *(not on `func` decl)*   | `@convention(c) () -> T`     | `@_cdecl("f") func f()`       |
| **Zig**                  | `fn f() callconv(.C) T`  | `*const fn() callconv(.C) T` | `export fn f()`               |
| **C++**                  | `extern "C" T f()`       | `T (*)(…)` *(implicit)*      | `extern "C"` linkage          |

Mojo's design is closest to Rust's: the ABI annotation is part of the function
signature in both declaration and type position, and is orthogonal to the
export/visibility mechanism. The choice of `abi("C")` over `extern "C"` avoids
confusion with Mojo's existing `@extern` decorator (which declares a function
stub backed by an external implementation) while remaining self-documenting.

---

## Summary of Changes

| Location                     | Before                          | After                                   |
|------------------------------|---------------------------------|-----------------------------------------|
| Function declaration         | *(no ABI annotation)*           | `def f() abi("C") -> T`                 |
| Function pointer type        | `fn() -> T`                     | `def() abi("C") -> T`                   |
| Export with C ABI            | `@export("f", abi="C") def f()` | `@export("f") def f() abi("C") -> T`    |
| Export without explicit ABI  | *(silently uses Mojo CC)*       | **compiler error**                      |
| Extern C library function    | *(broken — wrong CC)*           | `@extern("f") def f() abi("C") -> T`    |
| Extern Mojo bitcode function | *(implicit Mojo CC)*            | `@extern("f") def f() abi("Mojo") -> T` |
| DLHandle                     | returns bare `fn` ptr           | requires `abi("C")` in type param       |

---

## FAQ

**Q: Does `@export` always require a name argument?**

No. `@export` without a name argument uses the Mojo identifier as the exported
symbol name. Name mangling is suppressed by `abi("C")` — a C-ABI function must
be addressable by a plain identifier, so the compiler never mangles its name.
The name argument to `@export` is only needed when you want the exported symbol
to differ from the Mojo identifier:

```mojo
# Exports symbol "add" — no name argument needed
@export
def add(a: Int32, b: Int32) abi("C") -> Int32:
    return a + b

# Exports symbol "MyLib_add" — name argument overrides the identifier
@export("MyLib_add")
def add(a: Int32, b: Int32) abi("C") -> Int32:
    return a + b
```

---

**Q: Can a closure with captured state be used as an `abi("C")` function
pointer?**

No. `abi("C")` implies a *thin* function pointer — a plain machine code address
with no accompanying context. The C calling convention has no mechanism for
passing a closure's captured environment, so there is nowhere to put the
captured state. The compiler rejects the assignment:

```mojo
var multiplier: Int32 = 3

# ERROR: cannot convert capturing closure to abi("C") function pointer
var f: def(Int32) abi("C") -> Int32 = lambda x: x * multiplier

# OK: non-capturing closure (or a bare function reference)
var g: fn(Int32) abi("C") -> Int32 = lambda x: x * 3
```

This is the same constraint imposed by Rust (`extern "C" fn` pointers cannot
capture) and Swift (`@convention(c)` closure types must be non-capturing). If
you need to pass stateful callbacks through a C API, the conventional approach
is to use a `void*` context parameter alongside the function pointer and recover
the state via an unsafe cast on the Mojo side.
