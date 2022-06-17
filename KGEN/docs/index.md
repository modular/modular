# KGEN Kernel Generator

[TOC]

TODO: Need to write this.


## Some notes about parameter syntax

Parameters work differently than SSA values in a variety of ways and have their
own little mini-language.  To delineate they are special and different, we keep
them in the `<...>` syntax, which gives them a corner of the lexical world that
we know is theirs.  This section describes a bit of how they work and why.
Nothing is precious here, we can change this, this just reflects the current
approach.

Individual parameters:

0) First, it is important to understand that MLIR doesn't allow us to do
   contextual lookups to determine the type of a name.  Parameters can be
   declared after they are used, and we have a one pass parser.

1) Parameters can have many different MLIR types for future proofness (we might
   want to have string parameters etc) but there will be a high bias towards
   simple integer values (secondarily dtypes will occur, then there will be a
   longer tail).  We use the builtin MLIR `index` type as a convenient
   `ssize_t` type for math.
   
2) We want to reduce syntactic verbosity where reasonably possible, because
   syntactic noise makes it more difficult to write and read IR dumps.  In some
   cases, we "know" the type of a parameter expression, for example, in a buffer
   type like `!meta.buffer<a, b>` we "know" the type of `a` is `index` and the
   type of `b` is `!kgen.dtype`, as such, we don't require their type specifiers
   at all.

3) In cases with take arbitrary types (e.g. the input list to `kgen.call`, the
   parameter expression in `kgen.param.value` etc) we allow specifying a type
   with `: type = value` syntax which provides full generality for dtypes,
   strings etc.  However, because almost all parameters are of type 'index', 
   we allow omitting a type with `= value` syntax.  Note that an omitted type
   defaults to type `index` - it is not inferred from the initializer value
   (we can't do this for parameter references because of the forward reference
   issue mentioned above).

4) We will eventually have an expression evaluator that does constant folding
   etc, and that will need to have an integer width for the `index`
   computations.  We should use the width of the target's pointer size for this
   math, and overflows should be trapped as errors.

Parameter list syntax:

1) In practice, we expect almost all generator parameters to be input
   parameters, not result parameters.  As such, it is nice to have ceremony
   free syntax like `<height, width, p1, cacheSize>` for this common case.  We
   shouldn't require a result parameter specifier for no reason.  We do *allow*
   you to write `<height, width, p1, cacheSize -> ()>` for generality, but the
   IR printer won't generate it.

2) Return parameters follow the argument list and are separated from it with an
   arrow.  Like with arguments, we don't need to have parens in the normal case,
   we just use `<vecLen, unrollFactor -> outTileWidth, outTileHeight>` syntax.

3) We need a way to specify cases that use return parameters without arguments,
   and it "looks weird" to have an empty argument list (like
   `< -> outTileWidth, outTileHeight>`).  To solve this, we specify an empty
   argument list with empty parentheses, ala
   `<() -> outTileWidth, outTileHeight>`.

This design is a consequence of why you only see parens for empty argument
lists, and why (if you're working on the compiler parser itself) we should
support parens in the result type parser.
