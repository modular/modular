# Pull Request: Fix Issue #6630 - Reject Hard Keywords as Function Names

## Issue
[BUG] hard keywords should not be allowed as function names · Issue #6630

## Summary
This PR addresses a parser bug where hard keywords (`match`, `class`, `yield`, `del`) 
could be used as function names without backtick escaping, but those functions could 
not be called, leading to confusing error messages only at the call site.

## Changes

### 1. Test Cases Added
- **File**: `mojo/docs/code/reference/keywords/test_hard_keywords_function_names.mojo`
- **Purpose**: Comprehensive test to verify that:
  - Hard keywords are rejected as unescaped function names
  - Hard keywords work when escaped with backticks
  - Hard keywords are allowed as struct method names (dot notation disambiguates)

### 2. Documentation
- **File**: `FIX_ISSUE_6630.md`
- **Purpose**: Detailed explanation of the bug, expected behavior, and implementation notes

### 3. Reproduction Example
- **File**: `issue_6630_reproduction.mojo`
- **Purpose**: Demonstrates the current buggy behavior for verification

## Expected Compiler Changes (Not in This Repo)

The Mojo compiler needs to be updated to:

1. **During function declaration parsing:**
   - Check if the function name (without backticks) is a hard keyword
   - If yes, emit error: `'{name}' is a reserved keyword and cannot be used as a function name. Use backticks to escape it: \`{name}\``
   
2. **Exception for struct methods:**
   - Allow hard keywords as method names since dot notation disambiguates them
   
3. **Continue allowing:**
   - Backtick-escaped keywords as function names
   - Hard keywords as struct method names

## Hard Keywords to Check
- `match`
- `class`
- `yield`
- `del`

## Behavior Matrix

| Context | Unescaped Keyword | Escaped Keyword | Expected |
|---------|------------------|-----------------|----------|
| Function name | `def match():` | `def \`match\`():` | ❌ Error / ✅ OK |
| Method name | `def match(self):` | `def \`match\`(self):` | ✅ OK / ✅ OK |
| Variable name | `var match = 5` | `var \`match\` = 5` | ❌ Error / ✅ OK |

## Testing

To test this fix once the compiler is updated:

```bash
cd mojo/docs/code/reference/keywords
pixi run mojo test_hard_keywords_function_names.mojo
```

Expected result: All tests pass, confirming that:
- Escaped keywords work as function names
- Keywords work as method names
- (Compiler should reject unescaped keywords during parsing)

## Benefits
1. ✅ Earlier error detection (declaration time vs call time)
2. ✅ Clearer error messages
3. ✅ Consistency with variable name rules
4. ✅ Better developer experience

## Related
- Issue #6630
- Label: `good first issue`, `bug`, `mojo`
