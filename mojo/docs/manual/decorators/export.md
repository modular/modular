---
title: '@export'
description: Marks a function for export.
codeTitle: true

---

You can add the `@export` decorator on any function to make it publicly
available as an exported symbol in the compiled artifact, allowing it to be
called from external code, such as
[Python](/mojo/manual/python/mojo-from-python#import-a-mojo-module-in-python).

```mojo
# This function is internal - not callable from Python
fn internal_helper():
    print("Internal")

# This function can be called from Python as "my_exported_function"
@export
fn my_exported_function():
    print("Exported!")
    internal_helper()

# This function can be called from Python as "my_renamed_function"
@export("my_renamed_function")
fn my_other_function():
    print("Another function.")
```

The `@export` decorator can take two optional arguments:

- An alternate name to export the function under, as shown above.
- An ABI specifier. Currently only the the C ABI is supported.

Use the name and ABI specifier to export a function that complies with the C
calling conventions. You must also supply a function name that is a valid C
identifier. For example:

```mojo
@export("my_func", ABI="C")
fn my_function(name: StaticString, ptr: UnsafePointer[NoneType]) -> None:
    pass
```
