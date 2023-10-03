# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s

fn test_mlir():
  var x: __mlir_type.index
  x += x # expected-error {{'index' does not implement the '__iadd__' method}}

  # expected-error @+1 {{'index.add' op expected 2 operands, but found 0}}
  __mlir_op.`index.add`()

  # expected-error @+1 {{MLIR type 'index' has no attributes}}
  __mlir_op.`index.add`(x, x).x

  # expected-error @+1 {{operation already has attributes}}
  __mlir_op.`op`[value1=42][value2=42]

  # expected-error @+1 {{duplicate keyword parameter 'value'}}
  __mlir_op.`op`[value=42, value=42]

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

  # expected-error @+1 {{'index.constant' op MLIR verification error: 'index.constant' op requires attribute 'value'}}
  var c42e = __mlir_op.`index.constant`[value=42.0]()
  var c42 = __mlir_op.`index.constant`[value=Int(42).value]() # Good

  # expected-error @+1 {{invalid MLIR attribute:}}
  __mlir_attr.`#index<cmp_predicate xeq>`

  # expected-error @+1 {{MLIR attribute is not a TypedAttr: #index<cmp_predicate eq>}}
  __mlir_attr.`#index<cmp_predicate eq>`

  # expected-error @below {{expected name in attribute reference}}
  # expected-note @below {{escape keyword '_' with backticks to use it as an identifier}}
  __mlir_attr.

  # expected-error @below {{expected name in attribute reference}}
  # expected-error @below {{attribute spec requires a keyword parameter}}
  _ = __mlir_op.`test.op`[__mlir_attr.]

  # expected-error @+1 {{cannot use initializer syntax on MLIR type 'index'}}
  _ = __mlir_type.index(42)

fn colon_instead_of_equal():
  # expected-error @below {{attribute spec requires a keyword parameter; did you mean 'value=...'?}}
  _ = __mlir_op.`lit.crazy`[value:42]()

# Issue #7307: Error message can be improved when a user accidentally uses = instead of :
fn equal_instead_of_colon():
  # expected-error @+1 {{expected ':' after dictionary key, not '='}}
  _ = Int{value = Int(42).value}

  var someInt : Int
  # expected-error @+1 {{unable to infer result type from MLIR operation 'pop.array.gep'}}
  let ptr = __mlir_op.`pop.array.gep`((((someInt))), 4)

fn crash_on_invalid():
  # expected-error @+1 {{use of unregistered MLIR operation 'invalid_op'}}
  _ = __mlir_op.`invalid_op`[_type=__mlir_type.i16]()


# expected-error @below {{invalid MLIR type}}
# expected-note @below {{argument #0 with convention 'byref' in signature type should be a `!kgen.pointer`}}
fn bad_signature_type[func: __mlir_type[`!kgen.signature<(`, Int, ` byref) -> !kgen.none>`]]():
    pass


fn mlir_magic_keyword_param():
    # TODO(#21618): improve test when keyword parameters are enabled.
    # expected-error @below {{keyword parameters in indirect calls not supported yet}}
    alias a = __mlir_type[a=`!pop.scalar<bool>`]
