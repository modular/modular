# Lightning ⚡️ Notes

Lightning is intended to evolve into a superset of Python, which adds
first-class support for static types, "structs" with zero-cost abstraction
features, and support for kgen-parameters and search.

That said, it is still in early development and is missing many features.  This
document is intended to track notes about its ongoing development.

## Intentional differences from Python

Lightning is generally a superset, but here are some intentional differences in
our current implementation.  These are subject to discussion and re-evaluation
over time.  Not all of these are implemented.

1) Lightning supports structs, not just classes.  Design TBD.

2) In addition to the builtin dynamic Python object types like "int" and "dict",
   we will have library-defined static versions named "Int" and "Dict" etc that
   are defined as Lightning structs in the library.  These implementations will
   be very similar to the Python types in surface syntax, but will have type
   parameters and may have different behavior in some cases (TBD).

3) We support type parameters being declared on function definitions, ala
   `def method[size: Int]`.  This may be standardized into Python as [PEP
   695](https://peps.python.org/pep-0695/).

4) We have more generous indentation rules, not requiring `\` at end of line
   in most cases, due to a more sophisticated lexer rule that allows expression
   continuation so long as the continuation is more indented than the start of
   the expression.

5) While we allow forward references to values defined in outer scopes, we do
   not allow forward references within the same scope.  We will need to decide
   what to do about things like:

```
   if cond:
     x = 42
   else
     x = 17
   use(x)
```
   We will probably want to allow variable definitions anyway (with let/var like
   syntax), so that can probably be the solution.  We can also use dataflow
   analysis to reason about this.  Immediate-term workaround: use `x = 0` ahead
   of the `if` in cases like this.
