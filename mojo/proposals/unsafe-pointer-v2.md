# `UnsafePointer` v2

## Motivation

As Mojo’s standard library matures, a few foundational types need to be
stabilized — one of the most important being `UnsafePointer`. It is a
fundamental building block of many low-level abstractions and data structures,
however its current API has several flaws.

## Current Issues with `UnsafePointer`

1. **Unsafe implicit conversions**
   - `immutable` → `mutable` ([GitHub issue #4386](https://github.com/modular/modular/issues/4386))
   - `__origin_of(a)` → `__origin_of(b)`
   - `AnyOrigin` → `__origin_of(a)`

2. **Defaulted origin (`AnyOrigin`)**
   When an `UnsafePointer` is introduced with its defaulted `AnyOrigin`, any use
   of it extends *all* lifetimes and bypasses Mojo’s ASAP destruction rules.
   While sometimes desirable, such “escape hatches” should always be explicit.

3. **Defaulted mutability**
   Combining a defaulted `mut=True` with implicit casting from immutable to
   mutable has spread unsafe conversions throughout the codebase, especially in
   C FFI and kernel code.

Overall, Mojo’s current `UnsafePointer` is arguably *less safe* than C++ raw
pointers.

### Example Comparison

**C++ (errors on unsafe cast):**

```cpp
void foo(int* ptr) {}

int main() {
    const int y = 42;
    foo(&y); // Error: invalid conversion from 'const int*' to 'int*'
}
```

**Mojo (currently compiles):**

```mojo
fn foo(ptr: UnsafePointer[mut=True, Int]): pass

def main():
    var y = 42
    foo(UnsafePointer(to=y).as_immutable())
    # ^^ Compiles without an error :(
```

## Path Forward

Two main fixes are needed:

1. Prevent unsafe implicit conversions.
2. Remove defaulted parameters for mutability and origin, aligning with other
types (`Span`, `LayoutTensor`, etc.).

The proposal introduces a new `UnsafePointerV2` type that corrects these issues
and provides a migration path. During transition, `v1` and `v2` pointers will
support implicit conversions to avoid breaking existing code.

## `UnsafePointer` API (current)

```mojo
@register_passable("trivial")
struct UnsafePointer[
    type: AnyType,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    mut: Bool = True,  # ⚠️ Defaulted to mutable
    origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin],  # ⚠️ Defaulted to AnyOrigin
]:
    ...
```

**Issues:**

- `mut` defaults to `True`, making pointers mutable by default
- `origin` defaults to `MutableAnyOrigin`, bypassing lifetime tracking
- Allows unsafe implicit conversions (immutable → mutable, origin casts)

## `UnsafePointerV2` API

```mojo
struct UnsafePointerV2[
    mut: Bool, //, # ✅ Inferred mutability, no default
    type: AnyType,
    origin: Origin[mut], # ✅ Non-defaulted origin, must be explicit
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    ...

alias UnsafeMutPointer[...] = UnsafePointerV2[mut=True, ...]
alias UnsafeImmutPointer[...] = UnsafePointerV2[mut=False, ...]
```

**Improvements:**

- `mut` is inferred (using `//` marker) and has no default value
- `origin` must be explicitly specified or parameterized
- Prevents unsafe implicit conversions (compile-time errors)

### Cross-language Comparison

| Mojo | C++ | Rust |
| --- | --- | --- |
| `UnsafeImmutPointer[T]` | `const T*` | `*const T` |
| `UnsafeMutPointer[T]` | `T*` | `*mut T` |

---

## Why `V2`?

`UnsafePointer` is deeply integrated across the codebase.
Changing its interface directly would break a large amount of code, both
internally and in the community. `UnsafePointerV2` provides a transition path,
allowing incremental migration and validation before replacing `UnsafePointer`
entirely.

---

## Tentative Migration Timeline

### **Nightly (current)**

- Introduce `UnsafePointerV2`.
- Begin migrating `stdlib` and kernel code.

### **25.7 (late November 2025)**

- Mark `UnsafePointer` as deprecated.
- Promote `UnsafePointerV2` for general use.

### **26.1 (Jan 2026)**

- Remove old `UnsafePointer`.
- Rename `UnsafePointerV2` -> `UnsafePointer`.
- Deprecate the temporary `UnsafePointerV2` name.

---

## Migration Guidelines

Developers should start updating their code to the v2 API using the new
`UnsafeMutPointer` and `UnsafeImmutPointer` aliases where possible.
Use `UnsafePointerV2` directly only when these aliases don’t apply.

These aliases will remain stable across the migration, while `UnsafePointerV2`
itself will eventually be renamed back to `UnsafePointer`.

### Example Migration Steps

#### **25.7**

- Replace `UnsafePointer` with `UnsafePointerV2` or the new aliases.
- Fix any unsafe implicit casts.

#### **26.1**

- Rename `UnsafePointerV2` to `UnsafePointer`.

---

## Alternative Approaches Considered

1. **Using `@implicit(deprecated=True)`**
   - Not feasible due to the complexity of `UnsafePointer` constructors.
   - The current conversion constructor would require multiple overloads, some
   deprecated, some not, to separate safe and unsafe conversions.
   - Managing ~7+ overloads while avoiding ambiguity would be error-prone and
   still wouldn’t address inferred mutability or defaulted origins.
   - Starting fresh with `UnsafePointerV2` ensures correctness and clarity.

2. **Using `alias UnsafePointerV2 = ...`**
   - Would only help reorder parameters, not change constructor behavior.
   - Since the problem extends beyond API shape to semantics (implicit casting
   and defaults), an alias alone isn’t sufficient.
   - Mojo’s alias system also prevents initializer syntax for aliases, making
   this option impractical.
