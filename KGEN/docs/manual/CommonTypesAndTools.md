---
title: Common Types and Tools - Mojo 🔥 Compiler Dev Manual
markdown-notebook-data-directory: mdnb-data/manual-common-types-tools/
---

There are two main kinds of data types in Mojo:

- Memory types - Most structs are memory-types. Memory types can have references
  point at them.
- Register-passable - Primitives like `int64`, `float32`, and any struct marked
  `@register_passable` or `@register_passable("trivial")`. These are passed
  around (by value) in SSA registers, and don't ever have references pointing at
  them (there might be some rare exceptions to that rule). See
  [Life of Mojo reg-passable arguments](KGEN/docs/overviews/LifeOfMojoRegPassableArgs.md)
  for everything you could ever need to know about register-passable types.

With that in mind, here are all the various kinds of values you’ll encounter
(from `IRValues.h`):

<wolfram-cell cexpr="ImportIRValuesSnippet.wl" />

```wolfram,cell:Output
"AnyValue       <- Expr emitted to MLIR...
  UValue         <- unresolved value that cannot be materialized
    OverloadSetUValue  <- with an unresolved overload set
    InitializerUValue  <- constructor operands for an unknown type
  CValue        <- Concrete value: something with a known type.
    LValue         <- mutable reference to storage
      MLValue        <- value is in memory with a mutable reference
      DLValue        <- with dynamic get/set accessors
    BValue         <- with a borrowed value
      SBValue        <- value is register-passable and in an SSA \
register
      MBValue        <- value is in memory with a reference (may be \
mutable)
      MBPValue       <- reference with parametric mutability
      PValue         <- value is a parameter expression.
    RValue         <- with an owned value
      SRValue        <- with a register-passable value in an SSA \
register
      MRValue        <- value is in memory with a mutable reference
      PValue         <- with a parameter value"
```

## MLIR

We define our MLIR attributes and operations in `.td` files, which are
[Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
files that are transformed by tablegen into C++ code.

## Parameter Operator Code

Parameter Operator Code’s are named operations that are variants of the `POC`
enum. POC defines the names of operations supported by the
`#kgen.param.expr<op, args...>` MLIR attribute. This mechanism provides a way
for Mojo code to query values from the compiler at compile time.

This is used for a variety of reasons, from “simple” operations like `sizeof()`,
to innovative use-cases like `compile_assembly` (a way to compile a Mojo
function to assembly that is then embedded in the resulting binary).

<wolfram-cell ctext="Input07.wl" />

```td
def KGEN_POCAttr : I32EnumAttr<"POC", "Parameter Operator Code", [
  /// Fully associative variadic expressions.
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Mul", 1, "mul">,
  I32EnumAttrCase<"MulNuw", 2, "mul_nuw">,
  I32EnumAttrCase<"And", 3, "and">,
  I32EnumAttrCase<"Or",  4, "or">,
  I32EnumAttrCase<"Xor", 5, "xor">,
  I32EnumAttrCase<"Max", 6, "max">,
  I32EnumAttrCase<"Min", 7, "min">,
```
