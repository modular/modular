---
title: '@fieldwise_init'
description: Generates fieldwise constructor for a struct.
codeTitle: true

---

You can add the `@fieldwise_init` decorator on a struct to generate the
field-wise `__init__()` constructor.

For example, consider a simple struct like this:

```mojo
@fieldwise_init
struct MyPet:
    var name: String
    var age: Int
```

Mojo sees the `@fieldwise_init` decorator and synthesizes a fieldwise
constructor, the result being as if you had actually written this:

```mojo
struct MyPet:
    var name: String
    var age: Int

    fn __init__(out self, owned name: String, age: Int):
        self.name = name^
        self.age = age
```

This decorator replaces one function of the `@value` decorator. To synthesize
the copy constructor and move constructor, add the `Copyable` and `Movable`
traits to your struct.

For more information about these lifecycle methods, read
[Life of a value](/mojo/manual/lifecycle/life).
