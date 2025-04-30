# Upgrading “Trivial”

April 2025. Status: Proposed, not implemented.

The `@register_passable("trivial")` decorator on struct declarations were added
early in the evolution of Mojo as a way to bring up the language, when
memory-only types were added. Given other progress, it makes sense to rework
this to generalize them.

Here are some simple examples of both forms of `@register_passable` :

```mojo
@register_passable
struct RP:
  var v: Int
  var p: ArcPointer[String]
  
  # Can choose to provide __copyinit__ and __del__ if it wants.
  fn __copyinit__(out self, existing: Self): ...

  # Can't define __moveinit__.
  
@register_passable("trivial")
struct RPTrivial:
  var x: Int
  var y: Float64
  
  # Can't define __copyinit__ or __moveinit__ or __del__
```

## What `@register_passable` does today

The `@register_passable` decorator has a few effects, because it changes the
fundamental nature of how values of that types are passed and returned from
functions: instead of being passed around by reference (i.e. a hidden pointer)
it is passed in an MLIR register. This is an important optimization for small
types like `Int` - this document doesn’t explore this behavior and doesn’t
recommend changing it.

As a consequence of being `@register_passable` :

- the value doesn’t have “identity” - you can’t take the address of `self` on
  `read` convention methods, this is because they are in a register, not in
  memory.

- You aren’t allowed to define an `__moveinit__`: Mojo needs to be able to move
  around values of your type by loading and storing them, so it doesn’t allow
  you to have custom logic in your. move constructor.

- Mojo makes your type implicitly conform to `Movable` and will synthesize a
  trivial move constructor for you.

- Mojo checks that any stored member (`var` ’s) are also `@register_passable` -
  it wouldn’t be possible to provide identity for a contained member if the
  container doesn’t have identity.

- The type can choose whether it wants to be `Copyable` or not, use the
  `@value` decorator etc.

I'm quite happy with all of this, I feel like it works well and provides
important performance optimizations that differentiate Mojo from languages like
C++ and Rust.

## What `@register_passable("trivial")` does today

This is an extension of the `@register_passable` decorator that indicates the
type is “trivial” which means "movable + copyable by memcpy" and "has no
destructor”. In addition to the behavior of the `@register_passable` decorator,
marking something trivial provides the following behavior:

- The type implicitly conforms to `Copyable` and is given a synthesized
  `__copyinit__` that does a memcpy. It is given a trivial `__del__` member as
  well (so can’t be a linear type).

- All declared members are required to also be trivial, since you can’t memcpy
  or trivially destroy a container if one of its stored members has a
  non-trivial copy constructor.

- You are not allowed to define a custom `__copyinit__` or `__del__`.

- Types that are both `@register_passable` and trivial have some internal IR
  generation benefits to reduce IR bloat and improve compile time, but these
  aren’t visible to users of Mojo.

This behavior has been very effective and valuable: both as a way to make it
more convenient to define a common class of types (not having to write a
moveinit explicitly) and as a performance optimization for very low level code.

## Desired improvements to `“trivial”`

Trivial has worked well, but it can be better.  Three observations:

- The semantics of ”copyable with memcpy” and “no behavior in the destructor”
  really have nothing to do with register pass-ability, we would like for this
  to work with memory-only types too.

- We would like for algorithms (e.g. `List.resize` or `List.__del__`) to be
  able to notice when a generic type is trivial, and use bulk memory operations
  (`memcpy`) instead of a for loop or skip calling destructors for elements.

- Some day, we would like to be able to define that a type is trivial if its
  elements are trivial, even if one of the element types is generic.

For all these reasons, we need to upgrade “trivial” and decouple it from the
`@register_passable` decorator.

## Proposed approach: introduce a `Trivial` trait + type checking

To up-level the design, let’s introduce a new (compiler-known) `Trivial` trait
and type checking to enforce that it is used correctly.

### Defining the Semantics of “Trivial”

First, what does `Trivial` mean? We define it as two things:

1. A destructor with no side effects, this allows calls to it to be completely
   elided.

2. The value can be copied and moved with `memcpy`.

These two things cover a range of important types and enable the optimizations
(e.g. bulk memcpy) that we are looking for.

### Add the trait to the standard library

```mojo
trait Trivial(Copyable, Movable):
    alias __trivial_trait_magic : ()
```

This says that all `Trivial` types are required to be both copyable and
movable. It has a `__trivial_trait_magic` because Mojo has implicit
conformance, and we don’t want “everything” `Copyable` and `Movable` to be
inferred to be `Trivial` .

When Mojo gets rid of implicit conformance to traits, this can be removed. This
work is in flight.

### Add type checking for types that conform to it

We can now declare that types conform to the `Trivial` trait, some examples:

```mojo
@register_passable
struct RPTrivial(Trivial):
  var x: Int
  var y: Float64
  
  # Can't define copyinit or moveinit
  

struct MemTrivial(Trivial):
  var x: Int
  var y: Float64
  var z: OtherMemTrivial
  
  # Can't define copyinit or moveinit
```

In order to verify correctness, we should port over the existing semantic
analysis for “trivial”:

- Reject attempts to explicitly declare `__copyinit__`  `__moveinit__`  and
  `__del__` methods, (synthesizing them instead).

- Inject the special `alias __trivial_trait_magic : () = ()` alias. Optional,
  but for correctness we should also change alias declarations to prevent
  explicitly declared aliases with this name.

- Verify that the members all conform to `Trivial` as well.

- Depending on what happens with the `@value` decorator, we should look at its
  intersection with this trait and make them orthogonal.

## Optimizations made possible by Trivial

Now that `Trivial` is a trait, it automatically participates in the type system
in an expected way. Right now Mojo’s type system isn’t particularly powerful,
but with the introduction of require’s clauses and correct conditional
conformance and improvements to metatype representation, it will all dovetail
correctly. For example, I would like to be able to write something like this
someday:

```mojo
# No more hint_trivial_type!
struct List[T: CollectionElement]:

    fn __del__(owned self):
        @parameter
        if not __instanceof(T, Trivial): # Pick concrete syntax sometime later
            for i in range(len(self)):
                (self.data + i).destroy_pointee()
        self.data.free()

    fn _realloc(mut self, new_capacity: Int):
        var new_data = UnsafePointer[T].alloc(new_capacity)

        @parameter
        if not __instanceof(T, Trivial):  # Paint this later :-)
            memcpy(new_data, self.data, len(self))
        else:
            for i in range(len(self)):
                (self.data + i).move_pointee_into(new_data + i)
        ...
```

Implementing these features is out of scope for this proposal, but rearranging
the deck chairs here will make them compose together properly, and eliminate
some hacks like `hint_trivial_type`.
