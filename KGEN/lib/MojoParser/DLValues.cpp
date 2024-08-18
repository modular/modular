//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Value classes.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/DLValues.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"

#include "KGEN/LITDialect/LITOps.h"

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

// This hook is called before an argument is passed inout.
LValue BaseDLValue::prepareForInoutAccess(SMLoc loc,
                                          ExprEmitter &emitter) const {
  return DLValue(RCRef<BaseDLValue>::copy(const_cast<BaseDLValue *>(this)));
}

/// If this is a def argument shadow, resolve the underlying ref type for the
/// def argument.
RefType BaseDLValue::getMBValueTypeFromDefArgument() const { return {}; }

//===----------------------------------------------------------------------===//
// DiscardDLValue
//===----------------------------------------------------------------------===//

DiscardDLValue::DiscardDLValue(ASTType elementType, const ExprNode *expr)
    : BaseDLValue(elementType), expr(expr) {}

void DiscardDLValue::print(raw_ostream &os) const { os << "discard pattern"; }

CValue DiscardDLValue::emitLoad(ValueDest &dest, ExprEmitter &emitter) const {
  emitter.emitError(expr->getLoc(), "cannot read from discard pattern '_'")
      << expr->getRange();
  return {};
}

BValue DiscardDLValue::emitStore(ASTExprAnd<CValue> value,
                                 ExprEmitter &emitter) const {
  // Convert to an RValue to fully evaluate it.
  auto rvalue = emitter.emitRValue(value, EC_Assignment, elementType);
  // Promote to a BValue to return.
  return emitter.emitBValue({rvalue, value.expr}, EC_Assignment);
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

CValue StoredAttributeRefDLValue::emitLoad(ValueDest &dest,
                                           ExprEmitter &emitter) const {
  // To load x.y, we load x, then then load y out of it.
  ValueDest baseDest(dest.getContext());
  auto base = baseVal.ir->emitLoad(baseDest, emitter);
  if (!base)
    return {};
  return AttributeRefNode::emitStoredFieldRef({base, baseVal.expr}, getField(),
                                              expr, dest, emitter);
}

BValue StoredAttributeRefDLValue::emitStore(ASTExprAnd<CValue> value,
                                            ExprEmitter &emitter) const {

  if (!emitter.builder) {
    emitter.emitErrorForDynamicValueInParameter(expr);
    return BValue();
  }

  // tmp = load(base)
  // tmp.field = value
  // store(tmp -> base)
  auto loc = expr->getLocation(emitter);
  ASTType rvalueType = baseVal.ir->elementType;
  Value tmpDecl = emitter.emitVarDecl("__store_tmp__", rvalueType, loc,
                                      VarDeclKind::Synthesized);

  // Load the entire base LValue into tmpDecl.
  ValueDest tmpValueDest(MLValue(tmpDecl), EC_AttributeRefBase);
  auto base = baseVal.ir->emitLoad(tmpValueDest, emitter);
  if (!base) {
    tmpValueDest.resetForError();
    return BValue();
  }

  // Store into the field.
  auto fieldPtr =
      emitter.builder->create<RefStructGEROp>(loc, tmpDecl, getField());
  emitter.emitStoreToLValue(value, MLValue(fieldPtr), EC_AttributeRefBase);

  // Store the whole result back, transferring ownership as an MRValue.
  return baseVal.ir->emitStore({MRValue(tmpDecl), expr}, emitter);
}

MBValue StoredAttributeRefDLValue::emitMBValueFromDefArgument(
    ExprEmitter &emitter) const {
  auto baseRef = baseVal.ir->emitMBValueFromDefArgument(emitter);
  if (!baseRef)
    return {};

  auto fieldRef = emitter.builder->create<RefStructGEROp>(
      expr->getLocation(emitter), baseRef, cast<StructFieldOp>(fieldOp));
  return MBValue(fieldRef);
}

/// If this is a def argument shadow, resolve the underlying ref type for the
/// def argument.
RefType StoredAttributeRefDLValue::getMBValueTypeFromDefArgument() const {
  if (auto baseRef = baseVal.ir->getMBValueTypeFromDefArgument())
    return RefStructGEROp::getFieldType(baseRef, cast<StructFieldOp>(fieldOp));
  return {};
}

//===----------------------------------------------------------------------===//
// SubscriptDLValue
//===----------------------------------------------------------------------===//

SubscriptDLValue::SubscriptDLValue(PValue getter, StringAttr setterValueName,
                                   CallOperands &&operands, ASTType elementType,
                                   const ExprNode *expr)
    : BaseDLValue(elementType), getter(getter),
      setterValueName(setterValueName), operands(std::move(operands)),
      expr(expr) {}

/// Return true if this is a subscript, false if this is an attribute access.
bool SubscriptDLValue::isSubscript() const {
  return expr->kind == ExprNode::kSubscript;
}

void SubscriptDLValue::print(raw_ostream &os) const {
  os << (isSubscript() ? "(subscript): " : "(attribute): ") << elementType
     << " ";
}

CValue SubscriptDLValue::emitLoad(ValueDest &dest, ExprEmitter &emitter) const {
  // We got an elementType, so we know it has at least a getter or a setter.
  if (!getter) {
    emitter.emitError(expr->getLoc(),
                      "cannot read from set-only value of type ")
        << elementType << expr->getRange();
    return {};
  }

  return emitter.emitIndirectCall(getter, CallOperands(operands), dest,
                                  CallSyntax::kMethodCall, expr);
}

BValue SubscriptDLValue::emitStore(ASTExprAnd<CValue> value,
                                   ExprEmitter &emitter) const {
  // Add the set value to the keyword arguments list.  Semantic analysis already
  // checked that there can't be a duplicate.
  CallOperands operandsWithValue(operands);
  operandsWithValue.add(setterValueName, value);

  ValueDest storeDest(EC_Assignment);

  // We got an elementType, so we know it has at least a setter, so if we
  // couldn't resolve a setter, emit it to the named method so we can balk
  // with something more specific.
  // if (!setter) {
  StringRef setterName = isSubscript() ? "__setitem__" : "__setattr__";

  auto result =
      emitter.emitNamedMethodCall(setterName, std::move(operandsWithValue),
                                  storeDest, CallSyntax::kMethodCall, expr);
  return emitter.emitBValue({result, value.expr}, EC_Subscript);
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
CValue TupleDLValue::emitLoad(ValueDest &dest, ExprEmitter &emitter) const {
  // Emit a call to the tuple type constructor as an implicit conversion.
  return emitter.emitConstructorCall(elementType, CallOperands(eltLValues),
                                     expr, CallSyntax::kImplicitConvert, dest);
}

// TODO: Move this somewhere common like ExprEmitter
AnyValue emitGetterSetterAccess(const ExprNode *node, ASTExprAnd<CValue> base,
                                ArrayRef<Operand> exprOperands, ValueDest &dest,
                                ExprEmitter &emitter);

/// Storing to a tuple LValue extracts the elements out of the provided value
/// stores them into each component LValue.
BValue TupleDLValue::emitStore(ASTExprAnd<CValue> value,
                               ExprEmitter &emitter) const {
  auto emitError = [&]() -> InflightDiag {
    return emitter.emitError(expr->getLoc())
           << value.expr->getRange() << expr->getRange();
  };

  // If the value is a type with a staticly known length, check that it agrees
  // with the # of lvalues being assigned into.  Maybe we could generalize this
  // to invoke a new static get_static_len method or something?
  // TODO(generalize): Need @parameter fn's for methods
  // https://github.com/modularml/modular/issues/14945
  ASTDecl &tupleLiteralDecl = *elementType.getDecl(emitter.shared);
  ASTType srcRValueType = value.ir.getRValueType();

  // TODO: We need to support storing anything into a tuple that can be
  // extracted from, even things with dynamic length.  For example, Python
  // allows "(a, b) = [1, 2]", we need to support PythonObject.  The correct
  // sequence is to check the len(x) of the argument and see if it is exactly
  // right, CPython produces these errors at runtime:
  //   ValueRrror: too many values to unpack (expected 2)
  //   ValueError: not enough values to unpack (expected 2, got 1)
  //
  // We currently require the input be a Tuple.
  if (srcRValueType.getDecl(emitter.shared) != &tupleLiteralDecl) {
    emitError() << "cannot unpack value of type " << srcRValueType
                << " into a tuple";
    return BValue();
  }

  assert(srcRValueType.getParamBindings().size() == 1 &&
         "Tuple has one pack parameter");
  TypedAttr packAttr = srcRValueType.getParamBindings()[0];
  auto packVariadic = dyn_cast<VariadicAttr>(packAttr);
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
    ValueDest eltDest(lv, EC_TupleElement);

    // Bind the i parameters.  Int implicitly constructs from index type.
    TypedAttr iParam =
        IntegerAttr::get(IndexType::get(emitter.getContext()), index);

    SyntheticNode indexExpr(expr->getLoc(), PValue(iParam));
    Operand exprOperand(&indexExpr, expr->getLoc(),
                        Operand::PassKind::kPositional);
    SubscriptNode subscript(expr, expr->getLoc(), {}, expr->getLoc());

    // We emit the extraction from the tuple as a synthesized subscript with
    // this value as an index.
    if (!emitGetterSetterAccess(&subscript, {bvalue, value.expr}, exprOperand,
                                eltDest, emitter)) {
      eltDest.resetForError();
      return BValue();
    }
  }

  return bvalue;
}

//===----------------------------------------------------------------------===//
// DefArgumentWrapperDLValue
//===----------------------------------------------------------------------===//

DefArgumentWrapperDLValue::DefArgumentWrapperDLValue(ASTDecl *argDecl,
                                                     BValue argRef,
                                                     ASTType eltType,
                                                     size_t argIndex)
    : BaseDLValue(eltType), argDecl(argDecl), argRef(argRef),
      argIndex(argIndex) {}

/// If this is a def argument shadow, resolve it to the incoming immutable
/// borrowed value without forming a local copy.  Otherwise return null.
MBValue DefArgumentWrapperDLValue::emitMBValueFromDefArgument(
    ExprEmitter &emitter) const {
  return argRef.getIfMBValue();
}

/// If this is a def argument shadow, resolve the underlying ref type for the
/// def argument.
RefType DefArgumentWrapperDLValue::getMBValueTypeFromDefArgument() const {
  if (auto mb = argRef.getIfMBValue())
    return cast<RefType>(mb.getType());
  return {}; // SBValue.
}

void DefArgumentWrapperDLValue::print(raw_ostream &os) const {
  os << "def argument wrapper of type " << elementType;
}

// This hook is called before an argument is passed inout.
LValue
DefArgumentWrapperDLValue::prepareForInoutAccess(SMLoc loc,
                                                 ExprEmitter &emitter) const {
  // Okay, if the def argument is mutated, we need to snap into action and
  // lazily build a shadow in the function entry.
  auto func = cast<FuncOp>(argDecl->getParentDecl());
  ExprEmitter entryEmitter(emitter.shared, *argDecl->getParentDecl(),
                           OpBuilder::atBlockBegin(func.getBody()));
  StringAttr argName = func.getSignature().getArgName(argIndex);

  // Create the shadow box and copy the argument into it.  This will emit an
  // error at the specified location if the underlying type isn't copyable.
  VarDeclOp declOp = entryEmitter.makeArgLValueVarSlot(argRef, argName, loc);

  // Emission can fail when the type is non-copyable.
  if (!declOp) {
    argDecl->setErroneous();
    return LValue();
  }

  declOp.setArgShadowIndex(argIndex);

  // Update the representation so we don't do this again.
  argDecl->setIRValue(MLValue(declOp));
  return MLValue(declOp);
}

CValue DefArgumentWrapperDLValue::emitLoad(ValueDest &dest,
                                           ExprEmitter &emitter) const {
  // Loads of the def argument wrapper are simple enough.
  SyntheticNode expr(argDecl->getLoc());
  return emitter.emitCResult(argRef, &expr, dest);
}

BValue DefArgumentWrapperDLValue::emitStore(ASTExprAnd<CValue> value,
                                            ExprEmitter &emitter) const {
  // Okay, if the def argument is mutated, we need to snap into action and
  // lazily build a shadow in the function entry.
  LValue newVal = prepareForInoutAccess(value.expr->getLoc(), emitter);
  if (!newVal)
    return BValue();

  // Ok, now emit a normal store.
  return emitter.emitStoreToLValue(value, newVal, ExprContext::EC_Assignment);
}
