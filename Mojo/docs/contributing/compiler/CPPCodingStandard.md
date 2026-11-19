# C++ coding standard

This document is the single source of truth for C++ style and practice at
Modular. It consolidates the LLVM Coding Standards, the LLVM Programmer's
Manual, the MLIR Developer Guide, and Modular-specific overrides into one
self-contained reference.

Where standards conflict, the authority hierarchy is:
**Modular > MLIR > LLVM.**

> [!TIP]
> Automated enforcement: `clang-format` (LLVM base style with Modular
> overrides) and `clang-tidy` run automatically on changed files through Bazel.
> Run `./bazelw run //:format` to format and `./bazelw run //:lint` to lint.

## Language standard

Use **C++20**. Restrict code to features available in Clang, GCC, and MSVC.

### Preferred C++20 features

- **Concepts** — prefer `template <std::integral T>` over `std::enable_if_t`.
  Abbreviated function templates (`std::integral auto i`) are also acceptable.
- **Designated initializers** — prefer `.field = value` syntax for POD structs
  over positional or comment-annotated initialization.
- **`[[likely]]` / `[[unlikely]]`** — use sparingly in hot paths.

### C++20 features to avoid

The following C++20 features have existing equivalents or are not yet ready for
general use:

- **Coroutines** — may be integrated into AsyncRT in the future; avoid ad-hoc
  usage.
- **Ranges** — partially available through LLVM ADT; `std::ranges` headers can
  hurt compile times.
- **`std::format`** — use `llvm::formatv` instead. `std::format` may be
  reconsidered for its compile-time type checking.
- **`std::span`** — use `llvm::ArrayRef` instead. `ArrayRef` has broader
  implicit conversions.
- **Modules** — tooling is not mature enough and they require invasive build
  changes.

## Naming conventions

The following table lists the naming convention for each entity. These are
enforced by `clang-tidy` where possible.

| Entity                                      | Style                                  | Examples                            |
|---------------------------------------------|----------------------------------------|-------------------------------------|
| Types (classes, structs, typedefs, aliases) | UpperCamelCase                         | `TextFileReader`, `TensorShape`     |
| Functions and methods                       | lowerCamelCase                         | `openFile()`, `isFoo()`             |
| Local variables                             | camelBack                              | `leader`, `numBoats`                |
| Member variables                            | camelBack                              | `value`, `numElements`              |
| Namespaces                                  | CamelCase                              | `MLRT`, `Detail`, `KGEN`            |
| Enumerations (the type)                     | UpperCamelCase                         | `TypeKind`, `Signedness`            |
| Enumerators (values)                        | lowercase `k` prefix; `m` for bitmasks | `kSigned`, `mComplex`               |
| Template parameters                         | UpperCamelCase                         | `ElementType`, `N`                  |
| Macros                                      | UPPER_SNAKE_CASE                       | `LLVM_DEBUG`, `MY_MACRO`            |
| STL-mimic methods                           | lowercase_with_underscores (exception) | `begin()`, `push_back()`, `empty()` |

### Naming rules

- Avoid abbreviations unless universally known.
- Avoid single-letter names except for loop counters (`i`, `j`, `k`).
- Use descriptive names that reflect semantics and role, not temporary status
  (avoid `Tmp1`, `Tmp2`).
- Enum types used as discriminators or subclass indicators should use a `Kind`
  suffix (for example, `ValueKind`).

### Parser method naming

Recursive descent parsers (for example, `Mojo/lib/MojoParser/`) follow these
verb-prefix conventions:

- `parseXxx()` — parse entry points, return `ParseResult` with output
  parameters.
- `consumeXxx()` — token consumption (`consumeToken()`, `consumeIdentifier()`,
  `consumeIf()`).
- `skipXxx()` — advance past tokens without building AST (for example,
  `skipUntilIndentation()`).
- `emitXxx()` — IR emission from parsed AST.
- `isXxx()` / `hasXxx()` — boolean predicates (`isTokenInCurrentStatement()`,
  `isPrimaryExprToken()`).

### Enum guidelines

Use `enum class` for enumerators unless they are tightly scoped inside a
structure that makes their values obvious. Use a lowercase `k` prefix for
individual values and `m` for bitmasks:

```cpp
enum class TypeKind : uint8_t {
  kInvalid = 0,
  mComplex = 1,       // bitmask
  mSigned = 1 << 1,   // bitmask
  kS8 = mSigned,      // concrete value
};
```

A tightly scoped enum within a struct does not need `enum class`:

```cpp
struct MyIntegerType {
  enum {
    kSigned, kUnsigned
  } signedness;
};
```

## Formatting

Formatting is enforced by `clang-format` using the LLVM base style with
Modular-specific overrides:

```yaml
BasedOnStyle: LLVM
AlwaysBreakTemplateDeclarations: Yes
InsertNewlineAtEOF: true
RemoveSemicolon: true
```

### Core rules

- **Indentation:** 2 spaces. Never tabs.
- **Line length:** 80 columns maximum.
- **Trailing whitespace:** prohibited.
- **Line endings:** Unix (LF).
- **Preincrement:** prefer `++i` over `i++` (avoids unnecessary copies for
  non-trivial iterators).

### Braces

- Omit braces when the body is a single simple statement with no preceding
  comment, and the `if`/`else` chain is consistent.
- Use braces when: the body has multiple statements, the body has comments,
  any branch in an `if`/`else` chain uses braces, or dangling-`else` ambiguity
  exists.

### Namespaces

Do not indent inside namespaces. Add a closing comment:

```cpp
namespace MLRT {

void foo();

} // namespace MLRT
```

### Spaces

Place a space before `(` for control-flow keywords (`if`, `for`, `while`). Do
not place a space before `(` for function calls:

```cpp
if (X) ...          // space before (
somefunc(42);       // no space before (
```

### Lambda formatting

Format multi-line lambdas like blocks with 2-space indent. A single trailing
lambda uses standard block indent from the opening `[`:

```cpp
std::sort(V.begin(), V.end(), [](int a, int b) {
  return a > b;
});
```

### Braced initializer lists

Use `=` before the brace for variable initialization:

```cpp
int data[] = {0, 1, 2, 3};
```

## File organization

### File header boilerplate

Use a separator line only. Do not put a description or file name in the first
line. Do not use the `-*- C++ -*-` marker:

```cpp
//===----------------------------------------------------------------------===//
```

Good:

```cpp
//===----------------------------------------------------------------------===//
```

Avoid:

```cpp
//===- ABC.td - ABC dialect definition ------------------------------------===//
```

Run `./bazelw run //utils/lint-copyright-notices.py` to check and enforce
copyright headers.

### Module directory structure

Each top-level C++ module follows this layout:

- `ModuleName/include/ModuleName/` — public API headers
- `ModuleName/lib/` — implementation libraries
- `ModuleName/tools/` — command-line tools
- `ModuleName/test/` — lit and integration tests
- `ModuleName/unittests/` — unit tests
- `ModuleName/docs/` — developer documentation

Public API headers live under `ModuleName/include/ModuleName/` so code can
`#include "ModuleName/Foo.h"` in the LLVM style. No library within a module
may include internal implementation files from another module.

### Public API headers

- Keep headers minimal: use forward declarations and the pIMPL idiom.
- Reduce the number of `#include`s in headers.
- Internal header files may live in the `lib` directory for intra-module
  sharing.

### `#include` ordering

Order includes lexicographically within each group, separated by blank lines:

1. Main module header (the `.h` for this `.cpp` — always first)
2. Local/private headers
3. LLVM/MLIR project headers (most specific to least specific)
4. System `#include`s

### Header guards

Use ALL_CAPS path-based guards with `_` replacing `/` and `.`:

```cpp
#ifndef MODULENAME_SUBDIR_FILENAME_H
#define MODULENAME_SUBDIR_FILENAME_H
// ...
#endif // MODULENAME_SUBDIR_FILENAME_H
```

Every header must be self-contained and compile on its own.

### Non-header includes

Files meant for textual inclusion (not standalone headers) use the `.inc`
extension. They may lack guards and are used sparingly.

## Headers and includes

### Minimal includes

Include as little as possible, especially in headers. Forward-declare classes
when you only need pointers or references. Do not include headers just for
return types in prototyped functions.

### Banned includes

`#include <iostream>` is **banned** in library code. It injects a static
constructor into every translation unit that includes it. Use `raw_ostream`
instead. Other stream headers such as `<sstream>` are permitted.

### `using namespace` rules

- **Never** use `using namespace std;` anywhere.
- **Never** use `using namespace` in header files.
- In `.cpp` files, `using namespace llvm;` after all includes is acceptable.
- Always use the explicit `std::` prefix.

### Anonymous namespaces vs `static`

Keep anonymous namespaces as small as possible — use them only for class
declarations. Prefer `static` for functions, which maintains locality of
reference:

```cpp
namespace {
class StringSort {
  // ...
};
} // namespace

static void runHelper() { /* ... */ }
```

### Out-of-line definitions

Use namespace qualifiers, not reopened namespace blocks in `.cpp` files. This
catches declaration/definition mismatches at compile time:

```cpp
// Good:
int MLRT::fooVal;
int MLRT::foo(const char *s) { /* ... */ }

// Avoid:
namespace MLRT {
int fooVal;
int foo(const char *s) { /* ... */ }
} // namespace MLRT
```

## Error handling

### `assert`

Use aggressively for preconditions. Always include a descriptive message:

```cpp
assert(I < Operands.size() && "getOperand() out of range!");
```

Use `[[maybe_unused]]` for variables whose only use is in an assertion:

```cpp
[[maybe_unused]] bool newToSet = mySet.insert(value);
assert(newToSet && "The value shouldn't be in the set yet");
```

### `llvm_unreachable`

Use for code paths that should never be reached due to internal bugs (not user
input). It replaces `assert(0)` and hints the compiler to skip code generation
when assertions are disabled:

```cpp
switch (kind) {
case A: return handleA();
case B: return handleB();
}
llvm_unreachable("unhandled kind");
```

### `Error` and `Expected<T>`

Use the LLVM recoverable error mechanism for functions that can fail. All
`Error` instances **must** be checked or moved before destruction; unchecked
errors abort the program.

**Returning errors:**

```cpp
Error processFile(StringRef path) {
  if (auto err = openFile(path))
    return err;
  // ...
  return Error::success();
}
```

**Returning values or errors:**

```cpp
Expected<FormattedFile> openFormattedFile(StringRef path) {
  if (auto err = checkFormat(path))
    return std::move(err);
  return FormattedFile(path);
}

// Caller:
auto fileOrErr = openFormattedFile(path);
if (auto err = fileOrErr.takeError())
  return err;
auto &file = *fileOrErr;
```

**Type-based dispatch:**

```cpp
handleErrors(processFile(path),
  [](const BadFileFormat &e) { report(e); },
  [](const FileNotFound &e) { report(e); });
```

Unhandled error types are propagated. Prefer `handleErrors` with propagation
over `handleAllErrors`.

**Other error utilities:**

- `cantFail()` — asserts success; use sparingly in library code.
- `ExitOnError` — wraps `Error`/`Expected` calls to exit on failure; use in
  tool code.
- `joinErrors()` — concatenates multiple errors into an `ErrorList`.
- `report_fatal_error()` — for user-triggered errors when `Error`/`Expected`
  is impractical.

### Error recovery in parsers

When recovering from parse errors:

- **Suppress redundant diagnostics** — if the lexer already reported an error
  token, abandon the parser diagnostic to avoid double-reporting:

```cpp
auto diag = emitError(diagLoc, message);
if (getToken().is(Token::error))
  diag.abandon();
```

- **Indentation-aware skipping** — use `skipUntilIndentation()` to advance
  past malformed constructs while respecting bracket nesting. Track open
  brackets with a `SmallVector` to avoid skipping inside nested expressions.
- **Location adjustment** — when a missing token should have appeared on the
  previous line, adjust the error location:

```cpp
SMLoc loc = getToken().isStartOfLine()
                ? lexer.findEndOfPreviousLine(getToken().getLoc())
                : getToken().getLoc();
```

## Data structures

Prefer LLVM ADT containers over standard library equivalents. The following
tables list the recommended container for each use case.

### Sequential containers

| Container            | When to use                                                               |
|----------------------|---------------------------------------------------------------------------|
| `ArrayRef<T>`        | Read-only interface accepting any contiguous sequence. Pass by value.     |
| `SmallVector<T, N>`  | Vectors that are usually small. `N` elements inline (avoids malloc).      |
| `SmallVectorImpl<T>` | **As function parameters** — not `SmallVector<T, N>` with a specific `N`. |
| `std::vector<T>`     | Only when more than `UINT32_MAX` elements are needed.                     |
| `TinyPtrVector<T*>`  | Pointer-only; avoids allocation for 0–1 elements.                         |

> [!CAUTION]
> **Never** use `SmallVector<T, N>` with a specific `N` as a function
> parameter type. Use `SmallVectorImpl<T>` instead, which accepts any
> `SmallVector` variant.

**Avoid `std::list`** — it heap-allocates every element and is extremely
inefficient. Use `ilist<T>` (intrusive list) for polymorphic objects like
`Instruction` or `BasicBlock`.

### String types

| Type             | When to use                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `StringRef`      | Passing existing live strings. Pointer+length, no ownership. Pass by value. |
| `Twine`          | Concatenated strings as function arguments **only**. Always `const Twine&`. |
| `SmallString<N>` | Temporary scratch buffers on the stack.                                     |
| `std::string`    | Persisting results, heap storage, general mutation.                         |

> [!CAUTION]
> **Never** store `StringRef` or `Twine` — they contain external pointers and
> easily lead to dangling references. Never return `StringRef` for computed
> strings — return `std::string` instead.

### Set-like containers

| Container                                          | When to use                                    |
|----------------------------------------------------|------------------------------------------------|
| Sorted `std::vector` + `std::sort` + `std::unique` | Distinct insert/query phases.                  |
| `SmallSet<T, N>`                                   | Usually-small sets.                            |
| `SmallPtrSet<T*, N>`                               | Pointer sets with constant-time operations.    |
| `DenseSet<T>`                                      | Arbitrary types with a good hash.              |
| `StringSet`                                        | String uniquing.                               |
| `SetVector<T>`                                     | Fast lookup **and** insertion-order iteration. |

### Map-like containers

| Container          | When to use                                            |
|--------------------|--------------------------------------------------------|
| `DenseMap<K, V>`   | General key-value with good hash. Low constant factor. |
| `StringMap<V>`     | String keys.                                           |
| `MapVector<K, V>`  | Fast lookup **and** insertion-order iteration.         |
| `IndexedMap<V, I>` | Dense enum-like keys. Vector-backed, constant-time.    |

> [!CAUTION]
> **Avoid pointer-keyed unordered containers** — iteration order is undefined
> and causes non-deterministic output. Sort before iteration or use
> `MapVector`/`SetVector`.

### Sorting

Default to `llvm::sort` instead of `std::sort`. It randomly shuffles elements
when `EXPENSIVE_CHECKS` is enabled, which catches non-determinism bugs.

### Callbacks and function objects

| Type                 | When to use                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------------|
| `function_ref<Sig>`  | Non-owning callable reference. Pass by value. More efficient than `std::function`. Never store. |
| `std::function<Sig>` | When you need to store a callable.                                                              |

### Memory management

- **`BumpPtrAllocator`** — stack-like allocation, very fast, no per-object
  overhead. Cannot free individual objects.
- **`SpecificBumpPtrAllocator<T>`** — type-specific variant with individual
  deallocation.
- **`ManagedStatic<T>`** — lazy-initialized global. Constructed on first use,
  destructed at `llvm_shutdown()`.

### Hashing

Use LLVM hashing utilities rather than `std::hash`:

- **BLAKE3** (`llvm::BLAKE3`) — deterministic, cryptographic-strength hashing
  for cache keys and content addressing.
- **XXH3** (`llvm::xxh3_128bits`) — fast non-cryptographic hashing for content
  integrity checks.

For key composition, use a decorator pattern that chains hash transformations:

```cpp
llvm::BLAKE3 hashState{};
hashState.update(data);
auto hash = hashState.final();
```

### Binary serialization

Use `llvm::support::endian` for platform-independent binary I/O:

```cpp
// Writing:
llvm::support::endian::Writer writer(os, llvm::endianness::little);
writer.write((uint64_t)value);

// Reading (advances the pointer):
uint64_t value =
    llvm::support::endian::readNext<uint64_t, llvm::endianness::little,
                                    llvm::support::unaligned>(dataIt);
```

Use a pointer-passing pattern (`const char *&dataIt`) for stream position
tracking during deserialization.

### Container reuse in loops

Reuse containers across loop iterations to avoid repeated allocation:

```cpp
// Good: reuse allocation.
std::vector<Foo> v;
for (...) {
  v.clear();
  // ...
}

// Bad: allocates every iteration.
for (...) {
  std::vector<Foo> v;
  // ...
}
```

## Casting and RTTI

C++ RTTI (`dynamic_cast`, `typeid`) is **prohibited**. C++ exceptions are
**prohibited**. Use LLVM's hand-rolled casting infrastructure:

| Function                      | Behavior                                                          |
|-------------------------------|-------------------------------------------------------------------|
| `isa<T>(val)`                 | Returns `bool`. Like `instanceof`. Variadic: `isa<A, B, C>(val)`. |
| `cast<T>(val)`                | Checked cast. Asserts if wrong type. Use when certain of type.    |
| `dyn_cast<T>(val)`            | Returns `T*` or `nullptr`. Combines `isa` + `cast`.               |
| `isa_and_present<T>(val)`     | Like `isa<>` but allows null (returns `false`).                   |
| `cast_if_present<T>(val)`     | Like `cast<>` but propagates null.                                |
| `dyn_cast_if_present<T>(val)` | Like `dyn_cast<>` but propagates null.                            |

**Anti-pattern:** do not use `isa<>` followed by `cast<>` — use `dyn_cast<>`
instead.

## Class design

### `class` vs `struct`

- Use `struct` when **all** members are public.
- Use `class` when there are private members.
- All declarations and definitions of a given type must use the same keyword
  consistently.

### Virtual method anchor

Every class defined in a header with a vtable must have at least one
out-of-line virtual method. Without this, the compiler copies the vtable into
every `.o` file that includes the header.

### Move-only pervasive types

Pervasive types such as `RCRef`, `Error`, and `TensorShape` are designed to be
**move-only** by default, with an explicit `copy()` method for duplication.
This avoids accidental implicit allocations or reference count bumps and makes
additional allocations explicit:

```cpp
auto shapeCopy = shape.copy();  // explicit allocation
auto shape2 = std::move(shape); // cheap move
```

### Pass efficiently movable types by value

When a function sinks a parameter (stores it internally), pass efficiently
movable types by value instead of rvalue reference:

```cpp
// Good:
struct Foo {
  explicit Foo(std::string arg) : value(std::move(arg)) {}
private:
  std::string value;
};

// Avoid:
struct Foo {
  explicit Foo(std::string &&arg) : value(std::move(arg)) {}
private:
  std::string value;
};
```

Pass-by-value handles both lvalue and rvalue arguments cleanly. However, do not
pass expensive-to-copy types (such as `std::array<int, 100000>`) by value — use
`const&` instead.

### Template metaprogramming conventions

Prefer lightweight compile-time dispatch over heavy SFINAE:

- Use **`if constexpr`** with `constexpr bool` type traits for branching on
  callable signatures or type properties:

```cpp
template <typename FnT>
constexpr bool return_buffer_v =
    std::is_constructible_v<ReturnBufferTransformFn, FnT>;

template <typename FnT>
void run(FnT fn) {
  if constexpr (return_buffer_v<FnT>)
    handleReturnBuffer(std::move(fn));
  else
    handleWriteToBuffer(std::move(fn));
}
```

- Use **`llvm::is_detected`** for detecting member functions or nested types at
  compile time.
- Use **fold expressions** for variadic pack expansion:

```cpp
// Apply all wrappers in sequence:
([&]() { hashedKey = Wrappers::wrapKey(hashedKey); return true; }() && ...);
```

### Public APIs

Public header files should:

- Use forward declarations wherever possible.
- Use the pIMPL idiom to hide implementation details.
- Minimize `#include`s.

## Comments and documentation

### Comment style

- Prefer C++ style `//` for normal comments.
- Use `///` for Doxygen documentation on all top-level and class member
  definitions.
- Use C style `/* */` only for: constant parameter significance
  (`Object.emit(/*Prefix=*/nullptr)`), C89-compatible code, or headers
  included by C.
- Write comments as proper English prose.

### Doxygen conventions

- Use `///` for all doc comments.
- Use `\file` in file headers for Doxygen.
- Use `\brief` for the abstract, or let the first sentence serve as the brief.
- Use `\p name` to reference parameters in prose.
- Use `\param`, `\returns`, `\code ... \endcode` as needed.
- Do not duplicate the function or class name at the start of the comment.
- Document public APIs in headers only; implementation files may have
  additional internal comments.

### Grammar documentation in parsers

In recursive descent parsers, document the grammar rule being parsed in a
`///` comment above the method. Use BNF-like notation:

```cpp
/// Parse a suite, which is either a series of comma separated simple_stmt's
/// on one line, or an indented block of statements.
///
/// suite ::= [stmt_list NEWLINE] | NEWLINE INDENT statement+ DEDENT
ParseResult StmtParser::parseSuite(/* ... */);
```

### Error and warning message style

- Start the first sentence with a **lowercase letter**.
- End the last sentence **without a period** (unless other punctuation such as
  `?` is needed).
- Use succinct, correct English prose.

```text
error: file.o: section header 3 is corrupt. Size is 10 when it should be 20
```

## Control flow

### Early exits and `continue`

Prefer early returns, `break`, and `continue` to reduce nesting:

```cpp
// Good:
if (!isValid(x))
  return failure();
doWork(x);

// Avoid:
if (isValid(x)) {
  doWork(x);
} else {
  return failure();
}
```

### No `else` after `return`

Do not use `else` after `return`, `break`, `continue`, or `goto`:

```cpp
// Good:
if (condition)
  return earlyResult;
return normalResult;

// Avoid:
if (condition)
  return earlyResult;
else
  return normalResult;
```

Exception: `if constexpr` where `else` prevents template instantiation of the
other branch.

### Range-based `for`

Use range-based `for` wherever possible:

```cpp
for (const auto &val : container)
  observe(val);
```

### `auto` usage

Use `auto` only when it makes code more readable. Do **not** follow "almost
always auto."

Good uses: `cast<Foo>(...)`, iterator types, type-erased containers.

Default to `auto &` for values and `auto *` for pointers to avoid copies:

```cpp
for (const auto &val : container) observe(val);
for (auto *ptr : container) ptr->change();
```

### `switch` over enums

Do not use a `default` case in fully-covered switches over enums — it prevents
`-Wswitch` from catching missing cases. Place `llvm_unreachable` after the
switch instead:

```cpp
switch (kind) {
case Kind::A: return handleA();
case Kind::B: return handleB();
}
llvm_unreachable("unhandled Kind");
```

### `std::optional` conventions

Use `std::optional` to distinguish "absent" from "present with a value." Do
not use it as a general-purpose error type — use `ErrorOr<T>` for that.

Common patterns:

- Distinguish "not found" (`std::nullopt`) from "error during lookup"
  (`ErrorOr::isError()`). A cache lookup returns
  `ErrorOr<std::optional<BufferRef>>` — the outer layer is the error, the
  inner layer is "found or not."
- Use for values that can be legitimately absent (resolved types, indentation
  levels, optional attributes).
- Prefer `has_value()` over implicit bool conversion when clarity matters.
- Use `llvm::SaveAndRestore<std::optional<size_t>>` to temporarily override
  optional state.

### Lambda capture conventions

- **`[&]` capture** — use for short-lived lambdas that mutate local state
  (parser callbacks, `llvm::all_of` predicates). Never escape the enclosing
  scope.
- **`RCRef` captures** — use `.copy()` in the capture list for references that
  must outlive the lambda. Use `std::move()` for ownership transfer:

```cpp
std::move(chain).andThenSync(
    [result = result.copy(),                // refcount increment
     obj = std::move(obj)](/* ... */) {     // ownership transfer
      // ...
    });
```

- **`function_ref<Sig>`** — use for non-owning callback parameters. Pass by
  value. Never store. More efficient than `std::function`.
- **`llvm::unique_function<Sig>`** — use when the callable must be stored and
  is move-only (captures `std::unique_ptr` or similar).
- **`llvm::scope_exit`** — use for guaranteed cleanup in lambda form:

```cpp
auto restore = llvm::scope_exit([&] { savedCursor.restore(lexer); });
```

## MLIR-specific guidelines

### Pass naming

Prepend the dialect name to pass names and options:

- Pass: `-affine-tile`
- Option: `-affine-tile-size`

### Pass options

Use the MLIR pass options mechanism, **not** `cl::opt`. This allows
serialization in pipeline descriptions and running multiple instances with
different options.

### Pass threading safety

Passes that operate on nested regions run in parallel. Observe these rules:

1. Do not inspect or modify sibling operations.
2. Do not modify operations outside the nested scope.
3. May freely modify attributes of the current operation.
4. No mutable state across `runOnOperation()` calls.
5. No global mutable state (no static variables).
6. Passes must be copy-constructible.

### IR validity contract

- Passes **must assume** input IR is verifier-valid.
- Passes **must not** re-check invariants the verifier guarantees.
- Output IR **must** be verifier-valid.
- Pure optimization passes should safely no-op if their preconditions are
  missing.
- Lowering passes must emit diagnostics if the transformation is impossible.

### Verifier rules

- Only verify **local** aspects — do not follow def-use chains, do not examine
  producers or result users.
- Verifiers must fire only on definitively broken invariants, not "possibly
  invalid" cases.

### Dependent dialects

Override `getDependentDialects()` if a pass creates entities from dialects not
guaranteed to be loaded.

### No unbounded recursion

Avoid unbounded recursion. The only accepted exceptions are IR nesting
traversal and type nesting traversal.

## Testing

### Test types (in preference order)

1. **lit + FileCheck tests** — primary method for compiler and tool testing.
2. **Diagnostic tests** — for error message verification.
3. **C++ unit tests** — when the above are insufficient (data structures,
   internal APIs).

All commits must include tests. NFC (no functional change) commits are the only
exception.

### FileCheck conventions

- Use `CHECK-LABEL` to mark logical sections.
- Use `CHECK-NEXT` for consecutive lines.
- Use `CHECK-DAG` for unordered matches.
- Use `CHECK-NOT` to assert absence.
- Use named capture variables: `%[[VAR:.*]]`.
- Test only what is essential to the transformation — do not over-specify
  types, formatting, or SSA names.

### Diagnostic tests

```mlir
// RUN: mlir-opt %s -split-input-file -verify-diagnostics
cf.br ^missing  // expected-error {{reference to an undefined block}}
```

Every `op.emitError()` must have a corresponding diagnostic test.

### Test naming

Use descriptive names encoding what is tested. Prefix negative tests with
`negative_` or `no_`:

```text
@maskedload_to_load_static_i32_all_true     // positive test
@negative_maskedload_to_load_mixed           // negative test
```

Use consistent variable names from operation documentation (for example,
`%base`, `%mask`), not `%arg0`.

> [!TIP]
> The existing compiler lit tests show these conventions in practice. For
> example,
> [`pop-canonicalize.mlir`](https://github.com/modular/modular/blob/main/Mojo/test/kgen/pop-ir/pop-canonicalize.mlir)
> tests a pass with `CHECK-LABEL` and `CHECK-DAG`, and
> [`pop-errors.mlir`](https://github.com/modular/modular/blob/main/Mojo/test/kgen/pop-ir/pop-errors.mlir)
> tests diagnostics. Where the existing tests differ from MLIR/LLVM style,
> follow the existing tests.

## Prohibited features

This is the consolidated list of features that are banned or strongly
discouraged:

- **C++ RTTI** (`dynamic_cast`, `typeid`) — use `isa<>`/`cast<>`/`dyn_cast<>`.
- **C++ exceptions** (`throw`, `try`/`catch`).
- **Static constructors/destructors** — no global variables with constructors.
  Causes arbitrary initialization order and hurts launch time.
- **`#include <iostream>`** in library code — use `raw_ostream`.
- **`std::endl`** — use `'\n'` (avoids unnecessary flush).
- **`inline`** keyword on methods defined in class — already implicitly inline.
- **`std::list`** — heap-allocates every element.
- **`std::sort`** — use `llvm::sort` (shuffles on `EXPENSIVE_CHECKS`).
- **C++20 modules** — tooling is not mature.
- **C++20 coroutines, ranges, `std::format`, `std::span`** — see
  [language standard](#language-standard).
- **`using namespace std;`** — anywhere.
- **`using namespace`** in headers.
- **Storing `StringRef`, `Twine`, or `function_ref`** — they reference
  external memory and easily dangle.
- **`SmallVector<T, N>` as a parameter type** — use `SmallVectorImpl<T>`.
- **`cl::opt` for MLIR pass options** — use the pass options mechanism.
- **`default` in fully-covered enum switches** — prevents `-Wswitch` from
  catching missing cases.

## Debugging

### `LLVM_DEBUG` macro

Guards debug code blocks. Compiles to nothing in non-assert builds:

```cpp
LLVM_DEBUG({
  dbgs() << "Processing: " << name << "\n";
});
```

### `LDBG` macro

Convenient debug output with automatic filename, line number, and newline:

```cpp
LDBG() << "I am here!";
LDBG(2) << "Level 2 info";
```

Output: `[debug-type Filename.cpp:123 2] Message`.

### `DEBUG_TYPE`

Set once per file, after all includes. Controls `-debug-only=typename`
filtering:

```cpp
#define DEBUG_TYPE "my-pass"
```

### `STATISTIC` macro

Track optimization statistics, viewable with the `-stats` flag:

```cpp
#define DEBUG_TYPE "my-pass"
STATISTIC(NumTransformed, "Number of things transformed");

void runPass() {
  // ...
  ++NumTransformed;
}
```

## Modular error types

Modular defines its own error types in `Support/` that are distinct from LLVM's
`Error`/`Expected`. Use Modular error types for Modular code; use LLVM error
types only when interfacing directly with LLVM APIs.

### `Error` and `ErrorOr<T>`

`M::Error` is a lightweight, move-only error holding a `const char*` with
static string optimization (no heap allocation for string literals). Copying
requires an explicit `copy()` call. All error types are `[[nodiscard]]`.

`M::ErrorOr<T>` is a two-state result type (value XOR error). Use it for
functions that return a value or fail:

```cpp
ErrorOr<Attribute> result = state.readAttributeFromPointer(ptr, type);
if (result.isError())
  return ErrorTree(getLoc(), result.takeError());
state.mapResults(result.takeValue());
```

`M::ErrorOrSuccess` is the value-free variant for functions that either succeed
or fail with no return value.

### `ErrorTree` and `ErrorTreeOrSuccess`

`ErrorTree` is a hierarchical error type for compiler diagnostics. It carries a
location and supports nested causes, making it ideal for reporting chains of
errors with IR context:

```cpp
ErrorTreeOrSuccess validateSIMDConstruction(SIMDType simdType,
                                            const Location &loc) {
  if (!simdType.getResolvedSize())
    return ErrorTree(loc, "simd size must be known");
  if (!simdType.getResolvedDType())
    return ErrorTree(loc, "simd DType must be known");
  return success();
}
```

Chain causes with `addCause()`:

```cpp
ErrorTree(loc, "failed to lower operation").addCause(innerError);
```

### When to use which

- **`ErrorTreeOrSuccess`** — compiler passes, op interpreters, verifiers
  (anything with IR locations).
- **`ErrorOr<T>` / `ErrorOrSuccess`** — runtime code, utilities, and functions
  without IR locations.
- **LLVM `Error` / `Expected<T>`** — only when directly interfacing with LLVM
  APIs.

## Reference counting and ownership

### `RCRef<T>`

Modular uses intrusive reference counting via `RCRef<T>` for shared ownership.
Objects must inherit from `ReferenceCounted<T>` (CRTP). `RCRef` is **move-only
by default** with an explicit `copy()` method:

```cpp
auto ref = RCRef<MyObj>::create(args...);  // construct, refcount=1
auto ref2 = std::move(ref);               // cheap move
auto ref3 = ref2.copy();                  // explicit refcount increment
```

Factory methods:

- `RCRef::create(args...)` — construct in-place.
- `RCRef::take(ptr)` — take ownership without incrementing refcount.
- `ref.copy()` — explicit copy with refcount increment.

### Ownership hierarchy

In order of preference:

1. **Value types / move semantics** — default for local data.
2. **`std::unique_ptr<T>`** — exclusive ownership.
3. **`RCRef<T>`** — shared ownership (preferred over `std::shared_ptr`).
4. **`GenericUniquePtr` / `GenericRCRef`** — type-erased variants for
   heterogeneous containers.

Avoid raw `new`/`delete`. Avoid `std::shared_ptr` — use `RCRef` instead.

## Thread safety

### Reader-writer locks

Use `std::shared_mutex` for data structures with concurrent readers and
exclusive writers. Use `std::shared_lock` for reads and `std::lock_guard` for
writes:

```cpp
struct Cache {
  ErrorOr<bool> contains(StringRef key) const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return data.count(key);
  }

  ErrorOrSuccess insert(StringRef key, BufferRef obj) {
    std::lock_guard<std::shared_mutex> lock(mutex);
    data[key] = std::move(obj);
    return success();
  }

private:
  llvm::StringMap<BufferRef> data;
  mutable std::shared_mutex mutex;
};
```

Mark the mutex `mutable` so `const` methods can acquire shared locks.

### Process-level locking

Use `llvm::LockFileManager` for inter-process synchronization (for example,
cache directory cleanup):

```cpp
llvm::LockFileManager locker(path.string());
if (locker.tryLock()) {
  // Safe to modify shared resource; destructor releases lock.
}
```

## RAII state management

Beyond smart pointers, use RAII guards for any state that must be restored on
scope exit.

### `llvm::SaveAndRestore`

Temporarily overrides a variable and restores it when the guard goes out of
scope. Use for parser state, builder state, or any scoped override:

```cpp
llvm::SaveAndRestore<ASTDecl *> keepDecl(curDeclScope);
llvm::SaveAndRestore<std::optional<size_t>> savedIndent(stmtIndent,
                                                        std::nullopt);
// curDeclScope and stmtIndent restored when guards destruct.
```

### `llvm::scope_exit`

Guaranteed cleanup via lambda. Use when no dedicated RAII type exists:

```cpp
auto restoreLexer = llvm::scope_exit([&] { savedCursor.restore(lexer); });
```

### Coordinating multiple guards

When entering a compound scope (for example, a parser scope with debug info
and declaration tracking), pass RAII guards by reference and initialize them
together:

```cpp
void pushChildScope(DebugInfo::DIBuilder::ScopeGuard &scopeGuard,
                    llvm::SaveAndRestore<ASTDecl *> &keepDecl) {
  // Initialize both guards; caller's scope manages lifetimes.
}
```

### Other RAII utilities

- **`TimeTraceScope`** — automatic performance profiling. Destructor logs
  elapsed time.
- **`ScopedDiagnosticHandler`** — captures diagnostics during a scoped
  operation. Destructor pops the handler.
- **Lexer cursor checkpoints** — save lexer state with `getCursor()`, restore
  on backtrack. Enables arbitrary look-ahead:

```cpp
auto cursor = lexer.getCursor();
// ... try parsing ...
if (backtrackNeeded)
  cursor.restore(lexer);
```

## MLIR operation implementation patterns

This section covers conventions for implementing MLIR operations, drawn from
how ops are structured across the Modular codebase (for example,
`Mojo/lib/POPDialect/`, `Mojo/lib/KGENDialect/`, `GraphCompiler/lib/`).

### File splitting convention

Split large dialect implementations into separate files by concern:

- `*Ops.cpp` — op definitions, builders, parsers, printers.
- `*OpsFolders.cpp` — `fold()` method implementations.
- `*OpsCanonicalizers.cpp` or `*Canonicalizers.cpp` — `canonicalize()` methods
  and `OpRewritePattern` implementations.
- `*OpsVerifiers.cpp` or `*Verifiers.cpp` — `verify()` method implementations.

### Section separators within files

Use `//===---...---===//` separators to group related operations. Use a
top-level separator for each op or category, with sub-separators for
subcategories:

```cpp
//===----------------------------------------------------------------------===//
// Arithmetic operation folders
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Unary operations

OpFoldResult NegOp::fold(FoldAdaptor adaptor) { /* ... */ }
OpFoldResult FloorOp::fold(FoldAdaptor adaptor) { /* ... */ }

//===----------------------------------------------------------------------===//
// Binary operations

OpFoldResult AddOp::fold(FoldAdaptor adaptor) { /* ... */ }
```

### Fold methods

Return `OpFoldResult` (which is either an `Attribute` for constant results or a
`Value` for identity folds). Return `{}` (empty `OpFoldResult`) when folding is
not possible:

```cpp
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  // Try constant folding first.
  if (auto res = foldSIMDOp(
          adaptor.getOperands(),
          [](APSInt lhs, APSInt rhs) { return lhs + rhs; },
          [](APFloat lhs, APFloat rhs) { return lhs + rhs; }))
    return res;

  // Identity: X + 0 -> X (integers only).
  if (hasIntLikeType(this) && isIntZero(getRhs()))
    return getLhs();

  return {};
}
```

**Conventions:**

- Try constant folding first, then identity/simplification folds.
- Comment each fold pattern (for example, `// X + 0 -> X`).
- Use `adaptor.getOperands()` for constant operands, `getLhs()`/`getRhs()` for
  non-constant `Value` operands.
- Use `lookupTargetInfo(*this)` to retrieve target information in fold context.

### Canonicalize methods

Return `LogicalResult`. Use `notifyMatchFailure()` with a terse, lowercase
message to explain why the pattern does not apply:

```cpp
LogicalResult DivOp::canonicalize(DivOp op, PatternRewriter &b) {
  auto dtype = op.getType().getResolvedDType();
  if (!dtype)
    return b.notifyMatchFailure(op, "result type isn't resolved");

  if (!dtype->isIntLike())
    return b.notifyMatchFailure(op, "result type isn't int-like");

  // Canonicalize "x / 2^n" into "x >> n".
  SIMDAttr rhsAttr;
  if (!mlir::matchPattern(op.getRhs(), mlir::m_Constant(&rhsAttr)))
    return b.notifyMatchFailure(op, "rhs is not a constant");

  b.replaceOpWithNewOp<ShrOp>(op, op.getType(), op.getLhs(), shiftAmount);
  return success();
}
```

**Conventions:**

- Validate preconditions first with early `notifyMatchFailure` returns.
- Use `b.replaceOpWithNewOp<T>(...)` to replace and create in one step.
- Use `b.replaceOp(op, values)` to replace with existing values.
- Use `b.eraseOp(cast)` to clean up intermediate ops whose uses were removed.
- Add a `///` Doxygen comment above the method describing the transformation
  (for example, `/// Canonicalize bitcast(bitcast(x)) -> bitcast(x)`).

### `notifyMatchFailure` message style

Messages should be terse, lowercase, present-tense descriptions of why the
pattern does not match. Describe the condition that failed:

```cpp
return b.notifyMatchFailure(op, "result type isn't resolved");
return b.notifyMatchFailure(op, "rhs is not a constant");
return b.notifyMatchFailure(op, "intermediate cast has multiple uses");
return b.notifyMatchFailure(op, "target isn't resolved");
```

### Interpreter methods

Operations that support compile-time interpretation implement `interpret()` and
`parametric_interpret()`. These share logic with fold via `static` helper
functions, but use `state.getTarget()` instead of `lookupTargetInfo()`:

```cpp
static OpFoldResult foldDivOp(ArrayRef<Attribute> operands,
                              TargetInfoAttr target) {
  // shared fold logic
}

OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  return foldDivOp(adaptor.getOperands(), lookupTargetInfo(*this));
}

ErrorTreeOrSuccess DivOp::interpret(ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  if (OpFoldResult result = foldDivOp(operands, state.getTarget())) {
    if (auto attr = dyn_cast<Attribute>(result)) {
      state.mapResults(attr);
      return success();
    }
  }
  return ErrorTree(getLoc(), "failed to interpret POP::DivOp");
}
```

**Conventions:**

- Extract shared logic into a `static` helper (for example,
  `foldDivOp(operands, target)`).
- The fold method calls the helper with `lookupTargetInfo(*this)`.
- The interpreter calls the helper with `state.getTarget()`.
- Return `ErrorTree(getLoc(), "failed to interpret POP::XxxOp")` on failure.

### `OpRewritePattern` structure

```cpp
struct EmptyIfToSelect : public OpRewritePattern<HLCF::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(HLCF::IfOp op,
                                PatternRewriter &b) const override {
    if (!isEmpty(op.getThenRegion()) || !isEmpty(op.getElseRegion()))
      return b.notifyMatchFailure(op.getLoc(), "bodies aren't empty");

    // Perform rewrite.
    b.replaceOp(op, replacements);
    return success();
  }
};
```

**Conventions:**

- Inherit constructor with `using OpRewritePattern::OpRewritePattern;`.
- Name the `PatternRewriter` parameter `b` (short for builder).
- Validate preconditions before modifying IR.

### Pattern matching

Use `mlir::matchPattern` with MLIR matchers for constant extraction:

```cpp
SIMDAttr rhsAttr;
if (!mlir::matchPattern(op.getRhs(), mlir::m_Constant(&rhsAttr)))
  return b.notifyMatchFailure(op, "rhs is not a constant");
```

Common matchers:

- `mlir::m_Constant(&attr)` — match and extract a constant attribute.
- `mlir::m_ConstantInt(&apInt)` — match and extract a constant integer.
- `mlir::m_One()` — match the constant 1.
- `mlir::m_Zero()` — match the constant 0.

### Op creation

Use `OpType::create(builder, loc, args...)` or
`builder.create<OpType>(loc, args...)`:

```cpp
auto newOp = ParamConstantOp::create(b, op.getLoc(),
                                     SIMDAttr::get(values, op.getType()));
b.replaceOpWithNewOp<ShrOp>(op, op.getType(), op.getLhs(), newOp);
```

### Pass structure

Passes use the ODS-generated base class via CRTP:

```cpp
namespace M::KGEN {
#define GEN_PASS_DEF_MYPASS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct MyPass : public impl::MyPassBase<MyPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    op->walk([&](SomeOp someOp) {
      // transform
    });

    if (failed)
      signalPassFailure();
  }
};
} // namespace
```

### Verifier methods

Return `LogicalResult`. Use `emitOpError()` with stream insertion for
messages. Check multiple conditions before returning:

```cpp
LogicalResult MyOp::verify() {
  if (getNumOperands() != 2)
    return emitOpError() << "expected 2 operands, but given "
                         << getNumOperands();
  return success();
}
```

### Static helper conventions

Place `static` helper functions before the op methods that use them. Use
descriptive names reflecting the operation:

- `static OpFoldResult foldXxxOp(...)` — fold helpers.
- `static ErrorTreeOrSuccess validateXxx(...)` — validation helpers.
- `static bool hasIntLikeType(OpT op)` — type query helpers.
- `static std::optional<APSInt> getIntVal(Value val)` — value extraction
  helpers.

Overloading is common for different type variations of the same validation:

```cpp
static ErrorTreeOrSuccess validateSIMDConstruction(int64_t size, DType dtype,
                                                   const Location &loc);
static ErrorTreeOrSuccess validateSIMDConstruction(SIMDType simdType,
                                                   const Location &loc);
```

## Third-party code

### Integration methods

- **Submodule** (in `third-party/`): for actively developed projects such as
  LLVM/MLIR. Allows tracking upstream changes.
- **Vendored copy**: for complete/dead projects or code requiring major
  architectural changes. Discuss with the team first.

### Licensing

Generally compatible with Apache 2.0 and MIT/BSD. Always discuss with the team
before incorporating new third-party code.

After pulling in external code, verify copyright headers:

```bash
./bazelw run //utils:lint-copyright-notices
```

## The golden rule

If you are extending, enhancing, or bug-fixing already-implemented code, **use
the style that is already being used** so that the source is uniform and easy
to follow. Local consistency overrides all specific rules in this document.
