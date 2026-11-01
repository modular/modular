//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file implements the IR Value classes.
//
//===----------------------------------------------------------------------===//

#include "DLValues.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/MojoParser/ASTDecl.h"

#include "Mojo/LITDialect/LITOps.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// DLValue / BaseDLValue
//===----------------------------------------------------------------------===//

DLValue::~DLValue() = default;

DLValue &DLValue::operator=(const DLValue &existing) {
  storage = existing.storage.copy();
  return *this;
}

BaseDLValue::~BaseDLValue() = default; // vtable anchor.

//===----------------------------------------------------------------------===//
// DiscardDLValue
//===----------------------------------------------------------------------===//

DiscardDLValue::DiscardDLValue(ASTType elementType, const ExprNode *expr)
    : BaseDLValue(elementType), expr(expr) {}

void DiscardDLValue::print(raw_ostream &os) const {
  os << "discard pattern, type=" << elementType;
}

CValue DiscardDLValue::emitLoad(ExprDest &dest, IREmitter &emitter) const {
  emitter.emitError(expr->getLoc(), "cannot read from discard pattern '_'")
      << expr->getRange();
  return {};
}

CValue DiscardDLValue::emitStore(ASTExprAnd<CValue> value,
                                 IREmitter &emitter) const {
  // If the source already has an MLIR representation (thus is an MLValue,
  // SBValue etc) just returned the input value unmodified. This ensures we
  // don't load/copy lvalues in expressions like "_ = someVariable".  We do emit
  // an ownership use to extend the origin of the value to here and disable
  // "unused value" warnings.
  value.ir = maybeEmitRefinementRebind(value, emitter);
  if (auto mlirValue = value.ir.getMlirValue()) {
    OwnershipUseOp::create(*emitter.builder, value.expr->getLocation(emitter),
                           mlirValue);
    return value.ir;
  }

  // If 'value' is a DLValue like a subscript index, emit it as an RValue to
  // load to fully evaluate it.  It would be weird for "_ = dict[i]" to not
  // call the getitem method. This also handles PValues (which are already
  // RValues, so no materialization happens here.
  return emitter.emitRValue(value, EC_Assignment, elementType);
}

//===----------------------------------------------------------------------===//
// StoredAttributeRefDLValue
//===----------------------------------------------------------------------===//

StoredAttributeRefDLValue::StoredAttributeRefDLValue(
    ASTExprAnd<DLValue> baseVal, StructFieldOp fieldOp, ASTType elementType,
    const ExprNode *expr)
    : BaseDLValue(elementType), expr(expr), baseVal(baseVal), fieldOp(fieldOp) {
}

StructFieldOp StoredAttributeRefDLValue::getField() const {
  return cast<StructFieldOp>(fieldOp);
}

void StoredAttributeRefDLValue::print(raw_ostream &os) const {
  os << "stored attr '" << getField().getName() << " : ";
  baseVal.ir->print(os);
}

CValue StoredAttributeRefDLValue::emitLoad(ExprDest &dest,
                                           IREmitter &emitter) const {
  // To load x.y, we load x, then then load y out of it.
  ExprDest baseDest(dest.getContext());
  auto base = baseVal.ir->emitLoad(baseDest, emitter);
  if (!base)
    return {};
  // Use the AttributeRefNode logic to load the subfield or address the memory.
  return AttributeRefNode::emitStoredFieldRef({base, baseVal.expr}, getField(),
                                              expr, dest, emitter);
}

CValue StoredAttributeRefDLValue::emitStore(ASTExprAnd<CValue> value,
                                            IREmitter &emitter) const {

  if (!emitter.builder) {
    emitter.emitErrorForDynamicValueInParameter(expr);
    return CValue();
  }

  auto loc = expr->getLocation(emitter);

  // To store to "base().x" we need to first emit a load of the base LValue.
  ExprDest tmpValueDest(EC_AttributeRefBase);
  auto loadVal = baseVal.ir->emitLoad(tmpValueDest, emitter);
  if (!loadVal)
    return CValue();

  // If the result is a mutable ref (e.g. returned by a getter call), then we
  // can use that directly.
  if (auto baseRef = loadVal.getIfMLValue()) {
    auto fieldPtr =
        RefStructGEROp::create(*emitter.builder, loc, baseRef, getField());
    emitter.emitStoreToLValue(value, MLValue(fieldPtr), EC_AttributeRefBase);
    // Done: no tmp, no __setitem__, in-place mutation via the ref.
    return MBValue(baseRef);
  }

  // Otherwise, the getter will have returned an owned value, which we need to
  // mutate and write-back.
  //
  //   tmp = load(base)
  //   tmp.field = value
  //   store(tmp -> base)
  //
  // It is either in a memory temporary, or in an SSA register.  If it is in a
  // memory temp, we can commander it, but a register needs to be dropped into
  // memory for us to mutate it.
  MRValue loadMR = emitter.emitMRValue({loadVal, expr}, EC_AttributeRefBase);
  if (!loadMR)
    return BValue();

  // Store into the field.
  auto fieldPtr =
      RefStructGEROp::create(*emitter.builder, loc, loadMR, getField());
  emitter.emitStoreToLValue(value, MLValue(fieldPtr), EC_AttributeRefBase);

  // Store the whole result back, transferring ownership as an MRValue.
  return baseVal.ir->emitStore({loadMR, expr}, emitter);
}

//===----------------------------------------------------------------------===//
// SubscriptDLValue
//===----------------------------------------------------------------------===//

SubscriptDLValue::SubscriptDLValue(PValue getter, StringAttr setterValueName,
                                   CallOperands &&operands, ASTType elementType)
    : BaseDLValue(elementType), getter(getter),
      setterValueName(setterValueName), operands(std::move(operands)) {}

/// Return true if this is a subscript, false if this is an attribute access.
bool SubscriptDLValue::isSubscript() const {
  return operands.callExpr->kind == ExprNode::kSubscript;
}

void SubscriptDLValue::print(raw_ostream &os) const {
  os << (isSubscript() ? "(subscript): " : "(attribute): ") << elementType
     << " ";
}

CValue SubscriptDLValue::emitLoad(ExprDest &dest, IREmitter &emitter) const {
  // We got an elementType, so we know it has at least a getter or a setter.
  if (!getter) {
    emitter.emitError(operands.callExpr->getLoc(),
                      "cannot read from set-only value of type ")
        << elementType << operands.callExpr->getRange();
    return {};
  }

  return emitter.emitIndirectCall(getter,
                                  CallOperands(operands, std::move(dest)));
}

CValue SubscriptDLValue::emitStore(ASTExprAnd<CValue> value,
                                   IREmitter &emitter) const {
  // Add the set value to the keyword arguments list.  Semantic analysis already
  // checked that there can't be a duplicate.
  CallOperands operandsWithValue(operands, EC_Assignment);
  operandsWithValue.add(setterValueName, value, ArgUnpackStyle::kKeyword);

  // We got an elementType, so we know it has at least a setter, so if we
  // couldn't resolve a setter, emit it to the named method so we can balk
  // with something more specific.
  StringRef setterName = isSubscript() ? "__setitem__" : "__setattr__";
  return emitter.emitNamedMethodCall(setterName, std::move(operandsWithValue));
}

//===----------------------------------------------------------------------===//
// TupleDLValue
//===----------------------------------------------------------------------===//

TupleDLValue::TupleDLValue(ArrayRef<ASTExprAnd<AnyValue>> eltLValues,
                           ASTType tupleType, const ExprNode *expr)
    : BaseDLValue(tupleType), expr(expr),
      eltLValues(eltLValues.begin(), eltLValues.end()) {
  for ([[maybe_unused]] auto &elt : eltLValues)
    assert(elt.ir.getIfLValue() && "element must be an lvalue");
}

void TupleDLValue::print(raw_ostream &os) const {
  os << "(tuple lvalue): " << elementType << " ";
}

/// Loading a tuple RValue loads all the elements and returns a tuple instance.
CValue TupleDLValue::emitLoad(ExprDest &dest, IREmitter &emitter) const {
  // Emit a call to the tuple type constructor as an explicit construction.
  return emitter.emitConstructorCall(
      elementType,
      CallOperands(CallSyntax::kTypeCall, expr, std::move(dest), eltLValues));
}

// TODO: Move this somewhere common like IREmitter
AnyValue emitGetterSetterAccess(const ExprNode *node, ASTExprAnd<CValue> base,
                                ArrayRef<Operand> exprOperands, ExprDest &dest,
                                IREmitter &emitter);

/// Storing to a tuple LValue extracts the elements out of the provided value
/// stores them into each component LValue.
CValue TupleDLValue::emitStore(ASTExprAnd<CValue> value,
                               IREmitter &emitter) const {
  auto emitError = [&]() -> MojoInflightDiag {
    return emitter.emitError(expr->getLoc())
           << value.expr->getRange() << expr->getRange();
  };

  // If the value is a type with a statically known length, check that it agrees
  // with the # of lvalues being assigned into.  Maybe we could generalize this
  // to invoke a new static get_static_len method or something?
  // TODO(generalize): Need @__parameter fn's for methods
  // https://github.com/modularml/modular/issues/14945
  ASTDecl &tupleLiteralDecl = *elementType.getDecl(emitter.shared);
  ASTType srcRValueType = value.ir.getRValueType();

  // TODO: We need to support storing anything into a tuple that can be
  // extracted from, even things with dynamic length.  For example, Python
  // allows "(a, b) = [1, 2]", we need to support PythonObject.  The correct
  // sequence is to check the len(x) of the argument and see if it is exactly
  // right, CPython produces these errors at runtime:
  //   ValueError: too many values to unpack (expected 2)
  //   ValueError: not enough values to unpack (expected 2, got 1)
  //
  // We currently require the input be a Tuple.
  if (srcRValueType.getDecl(emitter.shared) != &tupleLiteralDecl) {
    if (!isa<TypeCheckErrorType>(srcRValueType))
      emitError() << "cannot unpack value of type " << srcRValueType
                  << " into a tuple";
    return BValue();
  }

  assert(srcRValueType.getParamBindings().size() == 2 &&
         "Tuple has one param_list of types and one TypeList parameter");
  TypedAttr packAttr = srcRValueType.getParamBindings()[0];
  auto packVariadic = dyn_cast<ParamListAttr>(packAttr);
  if (!packVariadic) {
    emitError() << "cannot unpack value of parametric tuple type "
                << srcRValueType << " into a fixed arity";
    return BValue();
  }
  if (packVariadic.getValues().size() != eltLValues.size()) {
    emitError() << "cannot unpack tuple value with "
                << packVariadic.getValues().size()
                << " elements into tuple binding with " << eltLValues.size()
                << " elements";
    return BValue();
  }

  // Emit the input value to a BValue, loading it if it is an LValue and
  // decaying from an RValue. We need to do this because each of the tuple
  // subscript operations we generate below will operate on this same IR value
  // multiple times: we don't want each of them to load the LValue redundantly
  // and do not want them to consume an RValue multiple times.
  auto bvalue = emitter.emitBValue(value, EC_TupleElement);
  if (!bvalue)
    return BValue();

  // Ok, we have a tuple with the right number of elements, extract each element
  // and store into the corresponding lvalue.
  for (auto [index, lvalue] : llvm::enumerate(eltLValues)) {
    // Get the item from the tuple into the corresponding LValue.
    LValue lv = lvalue.ir.getIfLValue();
    assert(lv && "Each dest is known to be an lvalue");
    ExprDest eltDest(lv, EC_TupleElement);

    CValue intIndexCValue = emitter.emitInt(index, value.expr->getLoc(),
                                            ExprContext::EC_CallParamValue);
    if (!intIndexCValue) {
      eltDest.resetForError(emitter);
      return BValue();
    }
    PValue intIndex = intIndexCValue.getIfPValue();
    assert(intIndex && "Int must be PValue when constructed from int attr");

    SyntheticNode indexExpr(expr->getLoc(), intIndex);
    Operand exprOperand(&indexExpr, expr->getLoc(),
                        ArgUnpackStyle::kPositional);
    SubscriptNode subscript(expr, expr->getLoc(), {}, expr->getLoc());

    // Emit the extraction from the tuple as a synthesized subscript with
    // this value as an index.
    if (!emitGetterSetterAccess(&subscript, {bvalue, value.expr}, exprOperand,
                                eltDest, emitter)) {
      eltDest.resetForError(emitter);
      return BValue();
    }
  }

  return bvalue;
}

//===----------------------------------------------------------------------===//
// ListPatternDLValue
//===----------------------------------------------------------------------===//

ListPatternDLValue::ListPatternDLValue(ArrayRef<ExprNode *> eltExprs,
                                       ASTType listType,
                                       PatternDeclKind patternDeclKind,
                                       const ListLiteralNode *expr)
    : BaseDLValue(listType), expr(expr),
      eltExprs(eltExprs.begin(), eltExprs.end()),
      patternDeclKind(patternDeclKind) {}

void ListPatternDLValue::print(raw_ostream &os) const {
  os << "(list pattern lvalue): " << elementType << " ";
}

CValue ListPatternDLValue::emitLoad(ExprDest &dest, IREmitter &emitter) const {
  // ListPatternDLValue should only be formed in destination context.
  emitter.emitError(expr->getLoc(), "cannot load from list pattern")
      << expr->getRange();
  return {};
}

CValue ListPatternDLValue::emitStore(ASTExprAnd<CValue> value,
                                     IREmitter &emitter) const {
  ASTType valueType = value.ir.getRValueType();
  auto sizedConformance =
      valueType
          .conformsToBuiltinTrait(
              "Sized", value.expr->getLoc(), emitter.shared,
              ASTDecl::getAssumptionsFromScope(&emitter.getDeclScope()))
          .first;
  if (!sizedConformance.isTrue()) {
    auto diag = emitter.emitError(value.expr->getLoc());
    if (sizedConformance.isUnknown())
      diag << "cannot prove that value of type " << valueType
           << " conforms to 'Sized'";
    else
      diag << "value of type " << valueType << " does not conform to 'Sized'";
    diag << value.expr->getRange();
    return {};
  }

  // Emit the input to a BValue so the length check and each __getitem__ can
  // reuse it without reloading an LValue or consuming an RValue multiple times.
  SMLoc loc = expr->getLoc();
  BValue bvalue = emitter.emitBValue(value, EC_CollectionLiteral);
  if (!bvalue)
    return {};

  // The minimum is the # of patterns in the list literal: the * pattern can
  // be empty though, so it doesn't count.
  CValue minimumLength = emitter.emitInt(
      eltExprs.size() - expr->starIndex.has_value(), loc, EC_CallParamValue);
  if (!minimumLength)
    return {};

  CValue isExact = emitter.emitBool(
      ASTExprAnd<PValue>{
          BoolAttr::get(emitter.getContext(), !expr->starIndex.has_value()),
          expr},
      EC_CallParamValue);
  if (!isExact)
    return {};

  // Look up check_list_length the same way paramfor_has_next is resolved.
  ArrayRef<ASTDecl *> checkFns = emitter.shared.getBuiltinFunction(
      emitter.getDeclScope(), {"std", "builtin", "_stubs"}, "check_list_length",
      loc);
  if (checkFns.empty())
    return {};

  CallOperands checkOperands(CallSyntax::kDirectCall, expr,
                             EC_CollectionLiteral);
  checkOperands.add({bvalue, expr});
  checkOperands.add({minimumLength, expr});
  checkOperands.add({isExact, expr});

  ParamBindings bindings(emitter.getDeclScope(), expr);
  OverloadSet checkCall("check_list_length", checkFns, std::move(bindings),
                        CallSyntax::kDirectCall);
  if (!checkCall.emitCall(std::move(checkOperands), emitter))
    return {};

  // If there is a * pattern e.g. [a, b, *c, d, e], we need to compute the index
  // that d and e will live at as well as the range that c will cover.  We do
  // this by calling `collection.__len__()` to get the actual length and doing
  // simple math.  We only need this if the * pattern is present and not the
  // last element in the list.
  CValue suffixStart;
  if (expr->starIndex && *expr->starIndex + 1 != eltExprs.size()) {
    // Emit the call to collection.__len__() to get the actual length.
    SyntheticNode valueNode(loc, bvalue);
    AttributeRefNode lenAttr(&valueNode, loc, "__len__");
    CallNode lenCall(&lenAttr, loc, /*operands=*/{}, loc);
    CValue actualLength =
        emitter.emitExprCValue(&lenCall, EC_CollectionLiteral);
    if (!actualLength)
      return {};

    // Start of the suffix is length-#trailing.
    size_t suffixLength = eltExprs.size() - *expr->starIndex - 1;
    CValue suffixLengthValue =
        emitter.emitInt(suffixLength, loc, EC_CallParamValue);
    if (!suffixLengthValue)
      return {};
    SyntheticNode actualLengthNode(loc, actualLength);
    SyntheticNode suffixLengthNode(loc, suffixLengthValue);
    BinOpNode subtract(ExprNode::kSub, &actualLengthNode, loc,
                       &suffixLengthNode);
    suffixStart = emitter.emitExprCValue(&subtract, EC_CollectionLiteral);
    if (!suffixStart)
      return {};
  }

  // Now that we know the length is correct, extract and store each element.
  SyntheticNode baseNode(loc, bvalue);
  for (auto [index, eltExpr] : llvm::enumerate(eltExprs)) {
    // If this is the * pattern, emit a slice operation to extract the suffix.
    if (expr->starIndex && index == *expr->starIndex) {
      CValue startValue = emitter.emitInt(index, loc, EC_CallParamValue);
      if (!startValue)
        return {};

      SyntheticNode startNode(loc, startValue);
      std::optional<SyntheticNode> endNode;
      if (suffixStart)
        endNode.emplace(loc, suffixStart);
      SliceLiteralNode slice(&startNode, loc, endNode ? &*endNode : nullptr, {},
                             nullptr);
      Operand sliceOperand(&slice, loc, ArgUnpackStyle::kPositional);
      SubscriptNode subscript(&baseNode, loc, sliceOperand, loc);

      ExprDest eltDest(eltExpr, EC_CollectionLiteral);
      eltDest.setPatternDeclKind(patternDeclKind);
      if (!emitter.emitExpr(&subscript, eltDest))
        return {};
      continue;
    }

    // Otherwise, we have a single pattern like "x".  Figure out its offset in
    // the collection based on whether it is before or after the * pattern.
    CValue indexValue;
    if (!expr->starIndex || index < *expr->starIndex) {
      indexValue = emitter.emitInt(index, loc, EC_CallParamValue);
    } else {
      size_t suffixOffset = index - *expr->starIndex - 1;
      if (suffixOffset == 0) {
        indexValue = suffixStart;
      } else {
        CValue offsetValue =
            emitter.emitInt(suffixOffset, loc, EC_CallParamValue);
        if (!offsetValue)
          return {};
        SyntheticNode suffixStartNode(loc, suffixStart);
        SyntheticNode offsetNode(loc, offsetValue);
        BinOpNode add(ExprNode::kAdd, &suffixStartNode, loc, &offsetNode);
        indexValue = emitter.emitExprCValue(&add, EC_CollectionLiteral);
      }
    }
    if (!indexValue)
      return {};

    // Build `base.__getitem__(42)` syntactically.
    SyntheticNode indexNode(loc, indexValue);
    Operand indexOperand(&indexNode, loc, ArgUnpackStyle::kPositional);
    AttributeRefNode getItemAttr(&baseNode, loc, "__getitem__");
    CallNode getItemCall(&getItemAttr, loc, indexOperand, loc);
    ExprDest eltDest(eltExpr, EC_CollectionLiteral);
    eltDest.setPatternDeclKind(patternDeclKind);
    if (!emitter.emitExpr(&getItemCall, eltDest))
      return {};
  }

  return bvalue;
}
