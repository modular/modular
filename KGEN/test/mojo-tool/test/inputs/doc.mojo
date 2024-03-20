# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This is the module.

```mojo
print("doc test 0")
```

```mojo
print("doc test 1")
```
"""


fn doc_fn(arg: Int) -> Int:
    """This is a function.

    ```mojo
    print("doc test 0")
    ```
    """
    return arg


struct Struct:
    fn doc_fn(self, arg: Int) -> Int:
        """This is a function.

        ```mojo
        print("doc test 0")
        ```
        """
        return arg
