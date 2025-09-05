# Requires

This doc goes over how the "requires" clause works in the various contexts that
supports it.

## Key Concepts

- ConstraintAttr: A user-provided constraint. It contains a proposition (a
parameter expression whose type is i1), a error message for when the proposition
does not hold, and a source location for where the constraint was declared.

## Sorting Constraints for Function Name Mangling (RASCFNM)

Constraints can be ordered by sorting on their respective propositions. This
provides a consistent ordering when serializing it as part of a function name
when used to constrain the input parameters of a function.

Constraints are serialized into a function's mangled name so that functions
overloaded on differing constraints have a more readable name. It also serves as
a first step in catching duplicate function declarations.
