# KGEN ⚜️: Design Rationale

This file contains design notes and other details about parts of the KGEN, along
with rationale for their design.  This is an evolving document that may be
turned into better structured documentation at some point.

## KGEN Parameter design

### Parameter syntax notes

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
   type like `!zap.buffer<a, b>` we "know" the type of `a` is `index` and the
   type of `b` is `!kgen.dtype`, as such, we don't require their type specifiers
   at all.

3) In cases with take arbitrary types (for example the input list to `kgen.call`, the
   parameter expression in `kgen.param.constant` etc) we allow specifying a type
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

## Structure of parameter definitions and uses.

The kgen dialect and system is defined in a way that makes it moderately open
for extension, but for that to work, operations need to follow some conventions
for their parameter declarations and uses.

Any operation is allowed to declare new parameters with a `ParamDeclAttr`.  This
node contains the `StringAttr` name for the parameter as well as its type.  The
key requirement is that `ParamDeclAttr`s may only occur in one place on an
operation: the operation must have them in a `paramDecls` attribute: if present,
that attribute must be an `ArrayAttr` of `ParamDeclAttr`s.  This means the
`paramDecls` attribute name is reserved for this purpose in kgen compatible
dialects.

Parameter uses, on the other hand, are far more flexible.  Parameters
expressions may occur anywhere in an operation -- including in types of values
referred to or returned by an operation.  This allows parameterized types,
allows an open and expressive set of operators that use parameters (for example to
pass to invoked generators, to materialize as SSA values, to return from the
function) etc.  There are no limitations on where they occur.

Parameter definitions and uses do not follow the standard dominance structure of
SSA or the MLIR region tree.  Instead, their requirement is that operations
that define and use parameter must have *some DAG ordering* that respects the
parameters definitions and uses within a kernel or kernel generator context.  By
convention, the location of the operation in the MLIR graph typically
represents an insertion point, not the order of execution of the metaprogram.

## Meta dialect types

### Support for dynamic shapes in `!zap.buffer` et al

The kgen infrastructure natively supports kernels that work with dynamic shapes
and dynamic dtypes, currently with the `!zap.buffer<?, ?>` type.  This allows
extracting the size/dtype as SSA values, which can then be switched over, or
have other calculations done at runtime.  When kgen supports Nd-arrays (tensors)
we will have the equivalent for that.  In order to work with dynamic shapes,
we need to be able to extract the only-known-at-runtime values with some
operations that produce SSA values.  These are:

```mlir
kgen.generator @algo(%dest: !zap.buffer<?, ?>) {
  // This returns a SSA value of type `!kgen.dtype`.
  %dtype = zap.buffer.dtype %dest: !zap.buffer<?, ?>

  // This returns a SSA value of type `index`.
  %size = zap.buffer.size %dest: !zap.buffer<?, ?>
  ...
}
```

Note that we do *not* support dynamic shapes or dtypes for the `!pop.scalar` or
`!pop.simd` types.  These may be *parameterized* with arithmetic that
determines the vector length or element, but it may not be dynamic (that is, there
is no `?` allowed) - parameters are always resolved to static values as part of
the code generation process.
This is because these are register-equivalent types, not memory-equivalent
types.  In the case of the runtime representation of a buffer, the size and
dtype doesn't affect how the buffer value itself is codegen'd: it is always a
tuple of `{void*, numElements, dtype}` at runtime.

Because the SIMD/scalar types do not support dynamic shapes or dtypes, they also
do not need operations like `pop.simd.size`. For any SIMD type, you either have
an integer constant in the IR or a parameter expression.  You can materialize
either of these into an SSA value with `kgen.param.constant`:

```mlir
kgen.generator @algo<veclen, dt: dtype>(%src: !pop.simd<mul(veclen,veclen), dt>) {
  // These do not need to exist!
  %dtypeSSAValue = pop.simd.dtype %src: !pop.simd<mul(veclen,veclen), dt>
  %veclenSSAValue = pop.simd.size %src: !pop.simd<mul(veclen,veclen), dt>

  // Use this instead:
  %dtypeSSAValue = kgen.param.constant : dtype = <dt>
  %veclenSSAValue = kgen.param.constant = <mul(veclen,veclen)>
}
```

## "pop" dialect design

The `pop` dialect solves two problems for KGEN:

1) It enables the definion of parametric operations (pre-elaboration) that can
   be generated by a front end parser.  The elaborator then resolves these to
   concrete values that exist post-elaboration.
2) The post-elaboration IR is serialized to IR and can be used as a distributed
   code IR (e.g. sent over a wire and executed remotely) and to enable tooling.
   This requires it to be sufficiently high level that doesn't expose target
   specific information (e.g. ABIs) unnecessarily, and also means that we want
   to capture information needed by tooling (e.g. header file printing wants to
   know whether integers are signed or not).

This section captures other specific design points that may be surprising about
its design and why.

### `pop.constant` allows its attribute value to mismatch the result type

The `pop.constant` operation allows things like this, which have mismatched
attribute types and the result type:

```
  %c1 = pop.constant(1.5 : f64) : !pop.scalar<f32>  // wrong fp type
  %c2 = pop.constant(12 : i8) : !pop.scalar<f32> // means 12.0f
  %c3 = pop.constant(256) : scalar<si8>   // too big for si8
```

The rationale stems from pop being a parametric dialect which allows the
concrete type to be resolved late.  The first code could very reasonably have
started out as:

```
  %0 = pop.constant(1.5 : f64) : !pop.scalar<someFPType>
```

and `someFPType` got resolve to `f32` by the elaborator, which generically
resolves type parameters without making other adjustments to the operation.

Some amount of type fluidity makes sense, but there are other designs we could
investigate:

1) We could reject `%c3` with an error or warning, for the same reason that
   using `T x = 256;` in C++ and resolving T to char would.
2) We could allow operations to adopt an operation interface with a "resolve
   yourself" method that would be invoked after parameters are resolved in the
   operation.  The elaborator would invoke this, allowing the `pop.constant`
   operation to fix itself after elaboration or report an error.

The benefit of #2 is that it would lead to a simpler IR.

## `zap` dialect design

The `zap` dialect is a substitute for language-level features and libraries
until those can exist. The `zap` dialect exists only pre-elaboration and is
lowered to the `pop` dialect before elaboration.

For example, `zap.buffer` is a substitute for a user-defined/library type.
Operations on buffers reside in the `zap` dialect.
