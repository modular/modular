# Collection Literal Design

Date: May 18, 2025
Status: Partially Implemented

This doc describes the design and behavior of collection literals in Mojo. It is
intended to explain how types can work with collection literals and how corner
cases are handled in one end-to-end place. It does not go into the design
details of specific collections like `List` or `Dict`.

## Background

Mojo inherits ergonomic collection literals from Python, for example, we want
things like this to work as a python programmer would expect:

```mojo
    mylist = [1, 2, 3]
    assert(mylist[1] == 2)

    myset = {1, 2, 3}
    assert(2 in myset)

    mydict = {"foo": 1, "bar": 2}
    assert(mydict["foo"] == 1)
```

Mojo has `List` and `Set` and `Dict` collection types, but we need to wire up
the collection literal syntax to work with these properly.

While we want collection literals to default to the standard library types, we
also want collection literals to work with other types as well. This is because
we want to enable specialized collections, e.g. lists that contain elements
inline, dictionaries specialized for string keys, and we also want to
interoperate well with Python.

```mojo
fn take_inline_array(arr: InlineArray[Int, 18]): ...
fn take_python(obj: PythonObject): ...

fn example():
    # This should form an InlineArray directly. It shouldn't create a List and
    # copy it.
    take_inline_array([1, 2, 3]) 

    # This should pass a python list with python integers, string, set and dict
    # inside of it.
    take_python([1, "str", {2, 3, 4}, {"foo": 1, "bar": 2}])
```

This gives us the power of easy of use of ergonomic defaults, while also
enabling the construction of efficient datatypes and interoperability with
existing APIs.

## List literals

In order to support list literal syntax, a type implements an initializer that
takes a `__list_literal__: ()` argument - an argument of empty-tuple type.  The
empty tuple ensures that this overload won't be picked accidentally, because
some types may need to support multiple literals (e.g. `PythonObject` needs to
support set literals and list literals, and the initializers do different
things.  For example, `List` has this initializer:

```mojo
struct List[T: Copyable & Movable, hint_trivial_type: Bool = False](...):
     fn __init__(out self, owned *values: T, __list_literal__: () = ()):
        """Constructs a list from the given values.
        ...
```

This allows `List` to be built from a list literal, and because the
`__list_literal__` has a default value, you can also use the same initializer
with explicit syntax like `List[Int](1, 2, 3)` if you'd like.

While it is conventional to define this initializer with a variadic, this isn't
required:

```mojo
# A type that allows initializers with 2 or 3 integers only.
struct TwoAndThreeList:
   fn __init__(out self, a: Int, b: Int, __list_literal__: ()): pass
   fn __init__(out self, a: Int, b: Int, c: Int, __list_literal__: ()): pass

fn test():
  # These are ok.
  var a: TwoAndThreeList = [1, 2]
  var b: TwoAndThreeList = [1, 2, 3]

  # Error: no matching function in initialization
  var c: TwoAndThreeList = [1, 2, 3, 4]
```

Similarly, you can also define things that take heterogenous types:

```mojo
struct StringIntListLiteral:
   fn __init__(out self, a: String, b: Int, __list_literal__: ()): pass

fn test():
  # Weird but accepted.
  var a: StringIntListLiteral = ["foo", 2]
```

While this is possible, this isn't something you should actually do.  The reason
that this is important to support is `PythonObject` which takes a heterogenous
collection of values that conform to `PythonConvertible`.

### Defaulting rule

If Mojo knows the contextual type for a literal, it will impose use that type to
construct the literal.  Otherwise it will default to the standard library `List`
type defined in the collections package.

## Set literals

TODO.

## Dictionary literals

TODO.

## Initializer lists

TODO.
