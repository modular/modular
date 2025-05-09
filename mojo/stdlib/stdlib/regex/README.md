# Regex Module

The regex module provides functionality for pattern matching and manipulation
of strings using regular expressions. This is a simple implementation that
supports basic regex operations.

## Usage

```mojo
from stdlib.regex import search, match, fullmatch, split, sub

# Search for a pattern in a string
result = search("world", "hello world")
if result:
    print("Found at position", result.start)  # Found at position 6
    print("Matched text:", result.group())    # Matched text: world

# Match at the beginning of a string
if match("hello", "hello world"):
    print("String starts with 'hello'")

# Match the entire string
if fullmatch("hello world", "hello world"):
    print("Exact match")

# Split a string by a pattern
parts = split(",", "apple,banana,orange")
for part in parts:
    print(part)  # Prints: apple, banana, orange

# Replace occurrences of a pattern
new_text = sub("hello", "hi", "hello world, hello universe")
print(new_text)  # Prints: hi world, hi universe
```

## API Reference

{{< /*<!-- markdownlint-disable MD013 -->*/ >}}

### Functions

- `compile(pattern: String, case_sensitive: Bool = True) -> Pattern`

  Compile a regular expression pattern into a Pattern object.

- `search(pattern: String, string: String, case_sensitive: Bool = True)
   -> Optional[Match]`

  Search for the pattern anywhere in the string.

- `match(pattern: String, string: String, case_sensitive: Bool = True)
   -> Optional[Match]`

  Match the pattern at the beginning of the string.

- `fullmatch(pattern: String, string: String, case_sensitive: Bool = True)
   -> Optional[Match]`

  Match the entire string to the pattern.

- `split(pattern: String, string: String, maxsplit: Int = 0,
   case_sensitive: Bool = True) -> List[String]`

  Split a string by occurrences of the pattern.

- `sub(pattern: String, replacement: String, string: String, count: Int = 0,
   case_sensitive: Bool = True) -> String`

  Replace occurrences of the pattern in the string.

### Structs

#### Pattern

A compiled regular expression pattern.

- `search(string: String, start_pos: Int = 0) -> Optional[Match]`

  Search for the pattern anywhere in the string, starting at the given
  position.

- `match(string: String) -> Optional[Match]`

  Match the pattern at the beginning of the string.

#### Match

Represents the result of a successful regex match operation.

- `group(index: Int = 0) -> String`

  Returns the substring matched by the regular expression. Index 0 represents
  the entire match.

- `start: Int`

  The index where the match starts in the original string.

- `end: Int`

  The index where the match ends in the original string.

- `string: String`

  The original string that was searched.

- `groups: List[String]`

  List of matched groups in the pattern.

## Current Limitations

This is a simple implementation with several limitations:

1. Limited support for regex syntax - only handles basic pattern matching
2. No support for capturing groups with parentheses
3. Limited support for character classes (only through escaped characters)
4. No support for quantifiers like `*`, `+`, `?`, or `{n,m}`
5. No support for anchors like `^` and `$` (except through the match/fullmatch
   functions)

Future versions will expand these capabilities.
