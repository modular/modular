# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-doc %s | FileCheck %s

"""Module Summary.

Handle `%#`:

```mojo
# Simple statements.
%# var value = 5
print(value)

# Multi-line/complex statements.
%# def return_value() -> Int:
%#   return 10
print(return_value())
```
"""

# CHECK: Handle `%#`:
# Check that the hidden lines are not displayed in the documentation.
# CHECK-SAME: # Simple statements.\nprint(value)
# CHECK-SAME: # Multi-line/complex statements.\nprint(return_value())
