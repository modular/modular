# Mojo docstring style guide

This is a language style guide for Mojo API docs (code comments known as
“docstrings”). The Mojo docstring style is based on the
[Google Python docstring style](https://google.github.io/styleguide/pyguide.html#381-docstrings),
with the addition of the Mojo-specific section headings, `Parameters:` and
`Constraints:`.

This is a brief set of guidelines that cover most situations. If you have
questions that are not answered here, refer to the more comprehensive [Google
Style Guide for API reference code
comments](https://developers.google.com/style/api-reference-comments).

For information on validating docstrings, see
[API docstrings](style-guide.md#api-docstrings) in the Coding standards and
style guide.

## Basics

- Docstrings support Markdown formatting.

- End all sentences with a period (including sentence fragments).

  - As you’ll see, most API descriptions are sentence fragments (they are
    often missing a subject because we don’t repeat the struct, function, or
    argument name in the first sentence).

  - Similarly, items in a bulleted list should end with a period, unless they
    are single words, or items that would go in code font.

- Use code font for all API names (structs, functions, attributes, argument and
  parameter names, etc.).

  - Create code font with backticks (\`Int\`).

  - When writing function/method names in text, add empty parentheses after the
    name, regardless of argument length. Don't include square brackets (even if
    the function takes parameters) If it's crucial to identify a specific
    function overload, add argument names, and/or a parameter list.

    For example:

    - Call the `erase()` method.

    - Use `pop(index)` to pop a specific element from the list.

    - If you know the power at compile time, you can use the `pow[n](x)` version
      of this function.

  - Follow the existing style of the source file. (For example, wrap text at
    80 columns.)

## Functions/Methods

### Description

- The first sentence is a brief description of what the *function* *does*. The
  first word should be a present tense verb ("Gets", "Sets", "Checks",
  "Converts", "Performs", "Adds", etc.).

- Follow the first sentence with a blank line.

- If you’re unsure how to phrase a description, just answer the question,
  “What does this function do?” Your answer should complete the sentence, “This
  function ____” (but without saying “this function”).

- If there are any prerequisites, specify them with the second
  sentence. Then provide a more detailed description, if necessary.

### Parameters and arguments

Since Mojo supports compile-time parameters, Mojo docstrings can include a
`Parameters:` section, as well as the `Args:` section described in the Google
Python docstring style guide. These both have the same format, with a parameter
name followed by a description.

```plaintext
Parameters:
    size: The static capacity for this array.

Args:
    value: The value to fill the array with. In the case of an extremely long
        description, wrap to another line and indent subsequent lines relative to
```

In the description for each parameter or argument:

- Use a noun phrase to describe what the *argument or parameter is.* This
  description should be formatted as a sentence (capitalize the first word, add
  a period at the end), even though it’s usually a sentence fragment. It should
  not be necessary to list the type, since this is added by the API doc
  generator. Add additional sentences for further description, as appropriate.

- Should usually begin with “The” or “A.”

### Return values

As with Python, use the `Returns:` section to document return values.

### Errors

Use the `Raises:` section to describe error conditions for a function.
Note that this isn’t currently supported by the Mojo API doc tooling, and will
render as regular text in the function description, not as a separate section.

```plaintext
Raises:
  An error if the named file doesn't exist.
```

### Constraints

Mojo functions can have compile-time *constraints,* defined using the
[`constrained()`](https://docs.modular.com/mojo/stdlib/builtin/constrained#constrained)
function. If the constraint isn’t met, compilation fails. Constraints can be
based on anything known at compile time, like a parameter value. You can't
create a constraint on an *argument*, because argument values are only known at
runtime.

Document constraints using the `Constraints:` section. Use a bulleted list if
there are multiple constraints.

Example:

```plaintext
Constraints:
    - The system must be x86.
    - `x.type` must be floating point.
```

If the only constraints are simple limits on single parameters, they should be
documented as part of the parameter description:

Example:

```plaintext
Parameters:
    size: The size of the SIMD vector. Constraints: Must be positive and a
          power of two.
```

For consistency, use the plural `Constraints:` even when documenting the
constraint inline in the parameter description. When describing a constraint on
 a single parameter, use a sentence fragment omitting the subject:

```plaintext
# AVOID
    type: The DType of the data. Constraints: This type must be integral.

# PREFER
    type: The DType of the data. Constraints: Must be integral.
```

Always use the standalone `Constraints:` section if the constraint doesn’t
neatly fit into the description of a single parameter. For example, the
constraints on a struct method may be based on parameters on the struct itself,
or on the machine architecture the code is compiled for.

**Don’t** use the term “constraints” for runtime limitations or error
conditions. Wherever possible, be specific about what happens when a runtime
value is out of range (error, undefined behavior, etc.).

```plaintext
# AVOID
Args:
    value: The input value. Constraints: Must be non-negative.

# PREFER
Raises:
    An error if `value` is negative.

# OR

Returns:
    The factorial of `value`. Results are undefined if `value`
    is negative.
```

## Structs/Traits

- As with a function, the first sentence forms the summary. Follow it with a
  blank line to mark the end of the summary.

- Do not repeat the name in the first sentence.

- Use a noun phrase to describe what the type *is* (”An unordered collection of
  items.”).

- Or, similar to function descriptions, use a present tense verb (when possible)
  to describe what an instance does or what the data represents (“Specifies,”
  “Provides,” “Configures,” etc.).

- Optionally include code examples, as with functions.

- Docstrings for traits follow the same rules as docstrings for structs, except
  that traits can't have parameters or fields—only method definitions and
  aliases.

Example:

```mojo
struct RuntimeConfig:
"""Specifies the Inference Engine configuration.

Configuration properties include the number threads, enabling telemetry,
logging level, etc.
"""
```

## Fields and aliases

Document fields or aliases by adding a docstring after the field or alias
declaration. Be descriptive even when the name seems obvious.

Example:

```mojo
alias FORWARD = 0
"""Perform the forward operation."""

alias REVERSE = 1
"""Perform the reverse operation."""

var label: Int
"""The class label ID."""

var score: Float64
"""The prediction score."""
```

### Parameters

Structs can have parameters, which follow the same rules as function parameters.

## Aliases

Aliases can be defined at the module level. See the description of Fields and
aliases above.

## Freeform sections

When documenting functions, structs, traits, or methods you can add other
sections, such as `Examples:` and `Notes:`. (For consistency with the other
section titles, we recommend using plurals nouns, even when you only have a
single note or a single example.)

These sections do not receive
any special treatment—for example, the section heading does not display in bold.
These "freeform" sections are displayed at the end of the description, in the
order in which they appear in the source file.

Add a blank line after the section header and write the section contents at the
same indent level as the header. *Don't* indent the contents of a freeform
section.

By convention these sections are added after the defined sections
(`Constraints:`, `Parameters:`, `Args:`, `Returns:`, and `Raises:`), but this
isn't enforced.

Example:

```mojo
fn noop():
    """Does nothing.

    Notes:

    - Efficiently distributes nothing to all available cores.
    - Takes advantage of GPU acceleration to do nothing faster.
    """
```

## Code examples

Add code examples using markdown-fenced code blocks, specifying the language
name.

You can include code examples in the body of the description or add an
`Examples:` freeform section. The examples section should go after
the defined sections (`Parameters:`,`Args:`, `Constraints:` `Returns:`,
`Raises:`).

**Don't** indent the code example.

Note that the code fence starts at the same indent level as the `Examples:`
header—unlike the `Parameters:` or `Returns:` sections, the contents of the
examples section aren't indented.

Example:

```mojo
fn select[
    result_type: DType
](
    self,
    true_case: SIMD[result_type, size],
    false_case: SIMD[result_type, size],
) -> SIMD[result_type, size]:
    """Produces a new vector by selecting values from the input vectors based on
    the current boolean values of this SIMD vector.

    Parameters:
        result_type: The element type of the input and output SIMD vectors.

    Args:
        true_case: The values selected if the positional value is True.
        false_case: The values selected if the positional value is False.

    Returns:
        A new vector of the form
        `[true_case[i] if elem else false_case[i] for i, elem in enumerate(self)]`.

    Examples:

    ```mojo
    v1 = SIMD[DType.bool, 4](0, 1, 0, 1)
    true_case =  SIMD[DType.int32, 4](1, 2, 3, 4)
    false_case = SIMD[DType.int32, 4](0, 0, 0, 0)
    output = v1.select[DType.int32](true_case, false_case)
    print(output)
    ```
    """
```
