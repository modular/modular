# Swappable Abstract Data Types (ADTs) in the Mojo Standard Library

**Status**: Draft

Author: Anish Kanthamneni

Date: January 2026

---

## Summary

This proposal introduces a small set of **base abstract data type (ADT) traits** in the Mojo standard library. These traits define *semantic categories of data structures* with stable core operations, allowing users to **swap implementations without refactoring code**.

The goal is not to expose every possible capability, but to establish a **common, minimal vocabulary** for the most widely used ADT classes (maps, sets, sequences, etc.).

---

## Scope

This proposal is scoped to:

- **Base ADT classes only**
- **Static (compile-time) dispatch by default**

It explicitly does **not** attempt to:

- Define all possible collection operations
- Replace concrete collection types
- Specify performance characteristics
- Introduce capability or extension traits

Those may be layered on later.

---

## Motivation

In practice, developers frequently experiment with different data structure implementations to improve performance or memory behavior:
- hash-based vs tree-based maps (ex, rust's HashMap vs BTreeMap)
- standard library structure vs a third part tuned implementation (ex, rust's HashMap vs hashbrown::HashMap)
- typical data structure vs thread safe variants (ex rust's HashMap vs dashmap::DashMap)

In Mojo today (as in Rust and many other languages), switching between implementations often requires **non-trivial refactoring** because APIs differ slightly in naming, return values, or iteration behavior.

This discourages experimentation and leads users to prematurely lock in choices.

The proposed solution is to define **ADT Traits** that capture *semantic intent* or *sets of supported operations* rather than implementation details.

---

## Design principles
1. **Strict Decoupling of Functionality and Implementation**
    - Each base ADT trait represents a distinct kind of data structure with different invariants and operations.
    - Base ADT Traits are not differentiated by performance, only by behavior.

2. **Explicit ordering guarantees**
   - Ordering semantics are part of the ADT’s identity, not optional behavior.

3. **Minimalist Hierarchical Base APIs**
    - In order for library authors to create functions that take ADT traits rather than concrete types, the ADT traits need to have all the functionality they want. Otherwise, they start using concrete types for their function signatures which subtracts from the value of this proposal. For this reason, a pure minimalist API base API (where the base ADT trait only requires the smallest subset of operations all implementations can perform) doesn't work. 
    - A better approach is to define a hierarchy of ADT traits. For example, a `BTreeMap` can implement `.range_scan()` while a `HashMap` cannot. Rather than grouping them into the same ADT trait and leaving out the range scan, we will make `HashMap` implement the `Map` ADT trait and make `BTreeMap` implement the `SortedMap` ADT trait. Since `SortedMap` is a strict superset of `Map`, we can create a default trait implementation for `Map` for all types that implement `SortedMap`. 
    - Since we also don't want an explosion of ADT trait hierarchies, we can allow functions without an observable effect (like `.reserve_capacity()`) to be implemented as no ops. This allows us to use this function on sequence types without having `ReservableSequence`, `ShrinkableSequence`, etc. etc. 

## Vocabulary
- **ADT Trait:** A trait that defines all operations that a category of ADTs should be able to implement. For example, a map ADT trait may define get, set, and delete operations. 
- **ADT Trait Extensions:** An ADT trait that is a strict superset of another underlying ADT trait and creates a default implementation of the underlying ADT trait for every structure that implements it. 
- **Unobservable Functions:** These are functions that do not affect the logical state of the data structure. For example, calling `.reserve_capacity()` on a sequence may perform an heap allocation but does not change the logical state of our sequence, so its considered to be an un observable function for the purposes of ADT Traits. Unobservable functions should have a no op default trait implementation. 

---

## Proposed ADT Traits

### ADT Traits

#### `Map[K, V]`
Unordered key–value association.

**Semantics**
- Each key maps to at most one value
- No guarantees about iteration order

**Base operations**
- `__init__() -> Self`
- `__get__(k: K) -> Optional[ref V]`
- `__set__(k: K, v: V)`
- `delete(k: K) -> Bool`         # Return True if it deleted a value
- `iter() -> Iterator[(ref K, ref V)]` (order unspecified)
- `len() -> UInt` 

**Default Trait Implementations**
- `__init__(a: (K, V), b: (K, V), ...) -> Self`
- `get_or_insert(k: K, v: V) -> ref V`
- `get_or_insert_with(k: K, f: fn() -> V) -> ref V`
- `contains(k: K) -> Bool`
- `is_empty() -> Bool`
- `update(k, f: fn(ref mut V) -> None) -> Bool`   # Returns True if a value was updated
- `map[NewVType](f: Fn(k: ref K, v: ref V) -> NewVType) -> Self[K, NewVType]`
- `map_inplace(f: Fn(k: ref K, v: ref mut V))`
- `filter(f: Fn(k: ref K, v: ref V) -> Bool) -> Self[K, V]`
- `filter_inplace(f: Fn(k: ref K, v: ref V) -> Bool)`
- `reserve_capacity(cap: UInt)`
- `shrink_to_fit()`

#### `Set[T]`

Unordered membership collection.

**Semantics**
- Each element appears at most once
- No iteration order guarantees

**Base operations**
- `__init__() -> Self`
- `add(item: T) -> Bool`      # Return True if item already exists
- `delete(item: T) -> Bool`   # Return True if it deleted a value
- `contains(item: T) -> Bool`
- `iter() -> Iterator[ref T]`

**Default Trait Implementations**
- `filter(f: Fn(f: ref T) -> Bool) -> Self[T]`
- `filter_inplace(f: Fn(f: ref T) -> Bool)`
- `reserve_capacity(cap: UInt)`
- `shrink_to_fit()`

#### `Sequence[T]`
Index-based sequence.

**Semantics**
- Elements are stored in a linear order
- Supports indexed insertion
- Implemented by List, LinkedList, Deque

**Base operations**
- `__init__() -> Self`
- `get_safe(idx: UInt) -> Optional[ref T]`
- `set_safe(idx: UInt, item: T) -> Bool`  # True if the operation was a success
- `insert(item: T, idx: UInt)`
- `extent_at(slice: Slice[T], idx: Uint)`
- `pop_at(idx: Uint) -> T`
- `iter() -> Iterator[ref T]`
- `len() -> Uint`

**Default Trait Implementations**
- `__init__(a: T, b: T, ...) -> Self`
- `__get__(idx: UInt) -> ref T`
- `__set__(idx: UInt, item: T)`
- `push(item: T)`
- `push_front(item: T)`
- `pop() -> T`
- `first() -> Optional[ref T]`
- `last() -> Optional[ref T]`
- `map[NewType](f: Fn(t: T) -> NewType) -> Self[NewType]`
- `map_inplace(f: Fn(f: ref mut T))`
- `filter(f: Fn(f: ref T) -> Bool) -> Self[T]`
- `filter_inplace(f: Fn(f: ref T) -> Bool)`
- `extend_front(slice: Slice[T])`
- `extend(slice: Slice[T])`
- `is_empty() -> Bool`
- `reserve_capacity(cap: UInt)`
- `shrink_to_fit()`

## ADT Trait Extensions

#### `InsertionOrderedMap[K, V]`
Insertion-ordered key–value association.

**Semantics**
- Preserves insertion order during iteration
- Matches Python-style dict behavior
- Extends `Map[K, V]`

**Additional operations**
- No additional operations. But by implementing this, the library author guarantees that their `.iter()` function returns elements in the order they were inserted. 

---

#### `SortedMap[K, V]`
Key-ordered key–value association.

**Semantics**
- Iteration order is sorted by key
- Ordering defined by `K: Comparable`
- Extends `Map[K, V]`

**Additional operations**
- `range_scan(start: K, end: K) -> Iterator[(ref K, ref V)]`
- By implementing this, the library author guarantees that their `.iter()` function returns elements in sorted order as is defined by `K: Comparable`

---

#### `InsertionOrderedSet[T]`
Insertion-ordered membership collection.

**Semantics**
- Preserves insertion order during iteration
- Extends `Set[T]`

**Additional operations**
- No additional operations. But by implementing this, the library author guarantees that their `.iter()` function returns elements in the order they were inserted. 


---

#### `SortedSet[T]`
Ordered membership collection.

**Semantics**
- Iteration order is sorted
- Ordering defined by `T: Comparable`
- Extends `Set[T]`

**Additional operations**
- `range_scan(start: T, end: T) -> Iterator[ref T]`
- By implementing this, the library author guarantees that their `.iter()` function returns elements in sorted order. 

---

#### `ContiguousSequence[T]`
Index-based sequence whose elements are laid out contiguously in memory. 

**Semantics**
- Extends `Sequence[T]`

**Additional Operations**
- `as_slice() -> Slice[T]`
- `to_unsafe_pointer() -> UnsafePointer[T]`


## Interaction with concrete implementations
Concrete types in `std.collections` (or equivalent) are expected to:
- Implement one or more base ADT traits
- Clearly document any extra guarantees

Users are encouraged to write APIs generic over the base ADT rather than a specific implementation:
```mojo
fn foo[M: Map[String, Int]](m: M):
    """
    This function needs a type that has basic get and set operations
    """

fn bar[M: SortedMap[String, Int] + Sync](m: M):
    """
    This function needs a type that has get, set, and range_scan operations. 
    It also needs the type to be thread safe. 
    """
```

---

## Why this belongs in the standard library
Without a standardized base each library defines its own pseudo-Map trait and APIs fragment quickly. More importantly, a proposal like this is only feasible in the standard library. 

---

## Conclusion
This proposal establishes a **small, stable foundation** for collection abstractions in Mojo. It deliberately prioritizes **semantic clarity and swappability** over completeness. There is also some precedent for this: Zig did something very similar with allocators, and it was so successful for them they decided to do it again with IO!

There are three main benefits to this
- **Reduces API fragmentation:** Adopting a third party data structure becomes easy as you already know what all the functions do. 
- **Increases Performance:** Since this makes experimenting with ADT implementations trivial, users are much more likely to choose the correct implementation for their use case and see a performance boost (as opposed to blindly reaching for HashMap every time). 
- **Increases Productivity:** Whether you're a library author, or are just making a data structure for your own application, you can implement a few functions and the rest of the functions with default trait implementations get implemented for you. 


## Next Steps
If this proposal is implemented it could potentially open the door to some interesting features
- Firstly, if mojo decides to include generic allocators like Zig, then this means we can choose to construct a map with a tuned implementation (one from std, a third party one, or our own) and with a specialized allocator (like an arena or fixed buffer allocator if our use case allows), and this hand-crafted type 
will "just work" with any mojo function anywhere that deals with map types. 
- This one is *very* speculative, but we could create a "PGO on steroids" feature into the compiler that not only chooses between low level compilation decisions, (should I inline this function or not) but also between different implementations of ADTs. 
