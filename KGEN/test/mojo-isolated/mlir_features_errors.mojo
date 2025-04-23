# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

alias `42` = __mlir_attr.`42 : index`

fn test_mlir():
  var x: __mlir_type.index
  x += x # expected-error {{'index' does not implement the '__iadd__' method}}

  # expected-error @+1 {{'index.add' op expected 2 operands, but found 0}}
  __mlir_op.`index.add`()

  # expected-error @+1 {{MLIR type 'index' has no attributes}}
  __mlir_op.`index.add`(x, x).x

  # expected-error @+1 {{operation already has attributes}}
  __mlir_op.`op`[value1=`42`][value2=`42`]

  __mlir_op.`op`[
    value=`42`,  # expected-note {{previously specified here}}
    value=`42`,  # expected-error {{duplicate keyword parameter 'value'}}
  ]

fn test_mlir2():
  # expected-error @below {{invalid MLIR type: kgen.dtype}}
  # expected-note @below {{MLIR error: expected non-function type}}
  var y : __mlir_type.`kgen.dtype`  # should be !kgen.dtype

  var a : __mlir_type
  var x: __mlir_type.index

  # expected-error @+1 {{unable to infer result type from MLIR operation 'index.castu'}}
  __mlir_op.`index.castu`(x, a)
  # expected-error @+1 {{unable to infer result type from MLIR operation 'index.castu'}}
  __mlir_op.`index.castu`(x)
  # expected-error @+1 {{'index.castu' op result #0 must be integer or index, but got 'f32'}}
  __mlir_op.`index.castu`[_type=__mlir_type.f32](x)

  # expected-error @below {{failed properties conversion while building index.constant with `{value = 4.200000e+01 : f32}`: Invalid attribute `value` in property conversion: 4.200000e+01 : f32}}
  # expected-error @below {{unable to infer result type from MLIR operation 'index.constant'}}
  var c42e = __mlir_op.`index.constant`[value=__mlir_attr.`42.0 : f32`]()
  var c42 = __mlir_op.`index.constant`[value=`42`]() # Good

  # expected-error @below {{invalid MLIR attribute:}}
  # expected-note @below {{attempting to parse: '#index<cmp_predicate xeq>'}}
  __mlir_attr.`#index<cmp_predicate xeq>`

  # expected-warning @below {{'!kgen.deferred' value is unused}}
  __mlir_attr.`#index<cmp_predicate eq>`

  # expected-error @below {{expected name in attribute reference}}
  # expected-note @below {{escape keyword '_' with backticks to use it as an identifier}}
  __mlir_attr.

  # expected-error @below {{expected name in attribute reference}}
  # expected-error @below {{attribute spec requires a keyword parameter}}
  _ = __mlir_op.`test.op`[__mlir_attr.]

  # expected-error @+1 {{cannot use initializer syntax on MLIR type 'index'}}
  _ = __mlir_type.index(`42`)

fn colon_instead_of_equal():
  # expected-error @below {{attribute spec requires a keyword parameter; did you mean 'value=...'?}}
  _ = __mlir_op.`lit.crazy`[value:`42`]()

@register_passable("trivial")
struct Int:
  var value : __mlir_type.index

# Issue #7307: Error message can be improved when a user accidentally uses = instead of :
fn equal_instead_of_colon():
  var someInt : Int
  # expected-error @+1 {{unable to infer result type from MLIR operation 'pop.array.gep'}}
  var ptr = __mlir_op.`pop.array.gep`((((someInt))), `42`)

fn crash_on_invalid():
  # expected-error @+1 {{use of unregistered MLIR operation 'invalid_op'}}
  _ = __mlir_op.`invalid_op`[_type=__mlir_type.i16]()


# expected-error @below {{invalid MLIR type}}
# expected-note @below {{argument #0 with convention 'mut' in func type should be a `!kgen.pointer`}}
fn bad_signature_type[func: __mlir_type[`!kgen.func<(index mut) -> !kgen.none>`]]():
    pass


fn mlir_magic_keyword_param():
    # expected-error @below {{only positional operands allowed in mlir magic}}
    alias a = __mlir_type[a=`!pop.scalar<bool>`]


fn mlir_properties(arg0: __mlir_type.i64, arg1: __mlir_type.i64):
    _ = __mlir_op.`llvm.add`[
        _type = __mlir_type.i64,
        _properties = __mlir_attr.`#llvm.overflow<nsw>`,
    ](arg0, arg1)
    # expected-error @above {{cannot set property}}
    # expected-error @above {{expected DictionaryAttr to set properties}}


fn mlir_illegal_op():
    # Intentional typo for `is_zero_poison`:
    # expected-error @below {{attribute 'is_zero_poson' is not an inherent attribute of 'llvm.intr.ctlz'}}
    __mlir_op.`llvm.intr.ctlz`[_type=Int, is_zero_poson=__mlir_attr.`0: i1`](1)

    # expected-error @below {{MLIR verification error: 'llvm.intr.ctlz' op requires attribute 'is_zero_poison'}}
    __mlir_op.`llvm.intr.ctlz`[_type=Int](1)
