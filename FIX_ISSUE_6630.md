# Fix for Issue #6630: Hard Keywords Should Not Be Allowed as Function Names

## Problem Summary

Currently, Mojo allows hard keywords (`match`, `class`, `yield`, `del`) to be used
as function names without backtick escaping. However, these functions cannot be called,
leading to confusing error messages that only appear at the call site rather than at
the declaration site.

## Current Behavior (Bug)

```mojo
def match():  # No error here
    pass

def main():
    match()  # Error: unexpected token in expression
```

## Expected Behavior (Fix)

```mojo
def match():  # Should error: 'match' is a reserved keyword
    pass
```

## Correct Usage with Escaping

Hard keywords CAN be used as function names when escaped with backticks:

```mojo
def `match`():  # OK - escaped keyword
    pass

def main():
    `match`()  # OK - can be called
```

## Exception: Struct Methods

Hard keywords SHOULD be allowed as struct method names because the dot operator
disambiguates them from keyword usage:

```mojo
struct Foo:
    def match(self):  # OK - method name
        pass

def main():
    var f = Foo()
    f.match()  # OK - clearly a method call, not a keyword
```

## Implementation Notes

The fix should be implemented in the Mojo parser/frontend, specifically in the
function declaration parsing logic. The validation should:

1. Check if a function name (without backticks) is a hard keyword
2. If yes, emit an error: "'{name}' is a reserved keyword and cannot be used as a function name. Use backticks to escape it: `{name}`"
3. Allow the same keyword as a struct/class method name (since it's accessed via dot notation)
4. Continue to allow escaped keywords (backtick-wrapped) as function names

## Hard Keywords to Check

The following keywords should be rejected as unescaped function names:
- `match`
- `class`
- `yield`
- `del`

Note: Other keywords like `def`, `fn`, `struct`, `if`, `else`, `for`, `while`, etc.
may already be properly rejected, but should be verified.

## Test Cases

See `test_hard_keywords_function_names.mojo` for comprehensive test cases that verify:
1. Unescaped hard keywords are rejected as function names
2. Escaped hard keywords work as function names
3. Hard keywords work as struct method names

## Related Files

- Parser/Frontend: (Mojo compiler source - not in this repo)
- Test file: `mojo/docs/code/reference/keywords/test_hard_keywords_function_names.mojo`
- Related example: `mojo/docs/code/reference/function-declarations/tests.mojo` (line 54 shows escaped `import` keyword)

## Benefits of This Fix

1. **Earlier error detection**: Errors are caught at declaration time, not call time
2. **Clearer error messages**: Users immediately understand they need to escape the keyword
3. **Consistency**: Aligns with how keywords are handled for variable names
4. **Better developer experience**: Reduces confusion and debugging time
