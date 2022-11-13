# Lightning ⚡️ Notes

Lightning is intended to evolve into a superset of Python, which adds
first-class support for static types, "structs" with zero-cost abstraction
features, and support for kgen-parameters and search.

That said, it is still in early development and is missing many features.  This
document is intended to track notes about its ongoing development.

## Topics to come back to later.

Here are a few things we should reconsider as we get further along.

1) Should `var` declarations be lexically scoped to their containing
   if/while/etc statement, or should they be whole-function like Python handles
   things.  Lexical scoping has some advantages and using explicit syntax to
   designate this makes it opt-in and teachable.

## Intentional differences from Python

Lightning is generally a superset of Python, but here are some intentional
differences in our current implementation as well as extensions.  These are
subject to discussion and re-evaluation over time.  Not all of these are
implemented.

### Differences with features that Python already supports

1) We have more generous indentation rules, not requiring `\` at end of line
   in most cases, due to a more sophisticated lexer rule that allows expression
   continuation so long as the continuation is more indented than the start of
   the expression.

2) While we allow forward references to values defined in outer scopes, we do
   not allow forward references within the same scope.  TODO: we should support
   things like:

```
   if cond:
     x = 42
   else
     x = 17
   use(x)
```

   by hoisting the implicit variable declaration to the top of the function.

3) We do not support all the deprecated features of the Python lexer.

4) We treat "soft" keywords like `case` as "hard" keywords in Lightning.
   Rationale is that we don't want things like "case = 42" to be supported.
   Python supports this for backwards compatibility reasons, but we can handle
   this in the future Python -> Lightning translator tool by (e.g.) adding an
   underscore to the keywords.

### New Features / Extensions that Python doesn't have

In addition to specific differences, we support the following extensions:

1) `struct` definitions, not just classes.  Structs are stored inline in their
   containing type/frame instead of indirectly with a reference count.  They
   don't support inheritance, and therefore are suitable for lots of
   optimization and things built with zero-cost abstractions.

2) We support type parameters being declared on function definitions, ala
   `def method[size: Int]`.  This may be standardized into Python as [PEP
   695](https://peps.python.org/pep-0695/).

3) In addition to the builtin dynamic Python object types like "int" and "dict",
   we will have library-defined static versions named "Int" and "Dict" etc that
   are defined as Lightning structs in the library.  These implementations will
   be very similar to the Python types in surface syntax, but will have type
   parameters and may have different behavior in some cases (TBD).

4) `var` definitions for local variables and instance properties in structs:
   We support implicit local variable definitions, but allow them to be
   optionally explicitly declared as well.

5) In addition to loosely typed `def` statements, we support a more strict `fn`
   statement, described below.

6) We support "backtick" identifiers ([like Swift](https://docs.swift.org/swift-book/ReferenceManual/LexicalStructure.html#:~:text=To%20use%20a%20reserved%20word,x%20have%20the%20same%20meaning.))
   which allow using keywords and lexically invalid strings of text as a single
   identifier.  This is useful when migrating code from Python that uses new
   keywords (e.g. fn), and is a generally useful escape hatch.

## Various Design notes

### New `fn` introducer

Lit supports a new `fn` introducer that can be used in place of `def`.  Both
form of declaration have the same capabilities - a `fn` can include dynamic
operations and interact with Python objects directly.

`fn` is effectively a more strict `def` that has different defaults.  Notably
a `fn` disables the following behavior that `def` maintains for Python
compatibility / familiarity, and to make it easier to migrate large swaths of
Python code to Lit someday.

More specific differences:

1) `fn` definitions require type annotations on the function signature.  In a
   `def`, they are optional and default to `object`.
2) Formal arguments in a `def` are mutable l-values, in a `fn` they are
   immutable.
3) `def` definitions allow local variables to be implicitly declared on their
   first use, a `fn` definition requires `let` and `var` introducers.

The choice of `fn` for this is arbitrary - we could have picked `func` (like
Swift/Go) or `function` like Javascript.  `fn` aligns with Rust/Zig which are
cool kids on the block these days.

### Expression parsing happens in two phases

Python uses its expression grammar for value expressions and for types.  This is
quite convenient for Lightning ⚡️ given we want types to be parameter values!
That said, there are some annoyances to deal with in terms of how to handle
this, for example, Python allows:

1) Values may be lexically used before they are defined, e.g. in expressions
   like `[x*x for x in range(42)]`
2) Code generation does not follow order of emission, e.g. in expressions like
   `x() if cond() else y()` where the `cond()`
   expression is evaluated first, then x/y are evaluated conditionally based on
   that.
3) As mentioned above, the expression grammar may resolve into a type or value
   depending on context.  Usually this doesn't matter, but we want to resolve
   expressions like `()` into different things in type and expression contexts.
4) As described below, we need to be able to parse the structure of a file
   before resolving types.

To handle all these problems we have a two phase resolution of expressions: we
first parse them into a bump pointer allocated tree data structure (defined
in `LitExprNodes.h`) and we can then "codegen" them into SSA expressions or into
a type.  This second phase is what performs name lookup etc, which means we can
parse the expression (and then ignore it) even before name binding.

### Structure of parsing + name binding + type checking

Python supports forward references to declarations in a file and/or module.
It handles this by making everything be dynamically executable (including `def`s
which are "executed" to install them in the dictionary for a class) and does not
actually type expressions statically.  This works for Python, but won't work for
Lightning ⚡️, and we can't give up support for forward references.

As such, we currently handle this by parsing the source file in three phases:

1) Declaration structure parsing.
2) Name binding + resolution of type expressions in declarations.
3) Parsing of values within those declarations (notably, function bodies and
   default initializers).

Let's take a look at an example to illustrate how this works:

```
struct Int:   # Eventually defined by stdlib.
  pass

def frolick(d: Doggie[42, Color(0, 255, 0)])
  print(d.furColor, d.numSpots*2)

struct Doggie[NumSpots: Int, FavoriteColor: Color]
  var furColor : Color
  cst numSpots : Int = NumSpots

struct Color:
  var r, g, b : Int

```

This example shows forward references of the `Color` type from the declaration
of the FavoriteColor parameter and `furColor` instance variable, as well as the
initializer expression for the parameter in the type list of `frolick` and
from the `d.color` usage in the `print`.  The reference to `Doggie` first occurs
in the argument list for `frolick` etc.  To support this, the first pass just
resolves the top level declaration names and structures, deferring parsing and
resolution of types and value expressions.

It parses and builds IR for these declarations:

```
struct Int:
  SKIPPED

def frolick(d: SKIPPED)
  SKIPPED

struct Doggie[NumSpots: SKIPPED, FavoriteColor: SKIPPED]
  var furColor : SKIPPED
  cst numSpots : SKIPPED = SKIPPED

struct Color:
  var r, g, b : SKIPPED

```

In the second pass we reparse the type expressions (which is nicely efficient
given how our parser works), allowing us to "see" this much of the example:

```
struct Int:
  SKIPPED

def frolick(d: Doggie[42, Color(0, 255, 0)])
  SKIPPED

struct Doggie[NumSpots: Int, FavoriteColor: Color]
  var furColor : Color
  cst numSpots : Int = SKIPPED

struct Color:
  var r, g, b : Int

```

Note that the deferred parsing and resolution of types cannot proceed in lexical
order: we need to resolve the type of `r/g/b` in the Color struct before we can
resolve the initializer expression in the parameter list of `d` in `frolick`.
This means 1) We need to do this in a worklist order, and 2) we can have cycles
which we need to identify and reject.

Once this is completed, we can parse and type check the remaining initializer
expressions / bodies which are all self contained in different scopes.  This can
be done in parallel.
