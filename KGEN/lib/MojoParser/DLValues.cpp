//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Value classes.
//
//===----------------------------------------------------------------------===//

#include "DLValues.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MojoParser/ASTDecl.h"

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

// This hook is called before an argument is passed mut.
LValue BaseDLValue::prepareForMutAccess(SMLoc loc, IREmitter &emitter) const {
  return DLValue(RCRef<BaseDLValue>::copy(const_cast<BaseDLValue *>(this)));
}

// This hook is called if the DLValue needs to be resolved to a physical ref.
// This emits an error and returns null on failure.
Value BaseDLValue::emitAsRefValue(llvm::SMLoc loc, IREmitter &emitter) const {
  emitter.emitError(loc)
      << "cannot convert computed lvalue to a stored reference";
  return {};
}

//===----------------------------------------------------------------------===//
// DiscardDLValue
//===----------------------------------------------------------------------===//

DiscardDLValue::DiscardDLValue(ASTType elementType, const ExprNode *expr)
    : BaseDLValue(elementType), expr(expr) {}

void DiscardDLValue::print(raw_ostream &os) const {
  os << "discard pattern, type=" << elementType;
}

CValue DiscardDLValue::emitLoad(ValueDest &dest, IREmitter &emitter) const {
  emitter.emitError(expr->getLoc(), "cannot read from discard pattern '_'")
      << expr->getRange();
  return {};
}

CValue DiscardDLValue::emitStore(ASTExprAnd<CValue> value,
                                 IREmitter &emitter) const {
  // Convert to an RValue to fully evaluate it.
  auto rvalue = emitter.emitRValue(value, EC_Assignment, elementType);
  // Promote to a CValue to return.
  return emitter.emitCValue({rvalue, value.expr}, EC_Assignment);
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
                                           IREmitter &emitter) const {
  // To load x.y, we load x, then then load y out of it.
  ValueDest baseDest(dest.getContext());
  auto base = baseVal.ir->emitLoad(baseDest, emitter);
  if (!base)
    return {};
  return AttributeRefNode::emitStoredFieldRef({base, baseVal.expr}, getField(),
                                              expr, dest, emitter);
}

CValue StoredAttributeRefDLValue::emitStore(ASTExprAnd<CValue> value,
                                            IREmitter &emitter) const {

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
    IREmitter &emitter) const {
  auto baseRef = baseVal.ir->emitMBValueFromDefArgument(emitter);
  if (!baseRef)
    return {};

  auto fieldRef = emitter.builder->create<RefStructGEROp>(
      expr->getLocation(emitter), baseRef, cast<StructFieldOp>(fieldOp));
  return MBValue(fieldRef);
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

CValue SubscriptDLValue::emitLoad(ValueDest &dest, IREmitter &emitter) const {
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

CValue SubscriptDLValue::emitStore(ASTExprAnd<CValue> value,
                                   IREmitter &emitter) const {
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

  return emitter.emitNamedMethodCall(setterName, std::move(operandsWithValue),
                                     storeDest, CallSyntax::kMethodCall, expr);
}

// Some subscripts, notably Dict, are defined with both a getter and a setter
// but the getter returns a ref (and throws).  If we need to bind the dict entry
// into a ref, call the getter.
Value SubscriptDLValue::emitAsRefValue(llvm::SMLoc loc,
                                       IREmitter &emitter) const {
  // If there is no getter, then this just fails like other computed lvalues.
  if (getter) {
    // Call the getter to get the ref.
    ValueDest storeDest(EC_RefBinding);
    auto ref = emitLoad(storeDest, emitter);
    if (ref && ref.isMValue())
      return ref.getMValueReference();
    if (!ref)
      return {}; // Error emitted by emitLoad.
  }

  return BaseDLValue::emitAsRefValue(loc, emitter);
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
CValue TupleDLValue::emitLoad(ValueDest &dest, IREmitter &emitter) const {
  // Emit a call to the tuple type constructor as an explicit construction.
  return emitter.emitConstructorCall(elementType, CallOperands(eltLValues),
                                     expr, CallSyntax::kTypeCall, dest);
}

// TODO: Move this somewhere common like IREmitter
AnyValue emitGetterSetterAccess(const ExprNode *node, ASTExprAnd<CValue> base,
                                ArrayRef<Operand> exprOperands, ValueDest &dest,
                                IREmitter &emitter);

/// Storing to a tuple LValue extracts the elements out of the provided value
/// stores them into each component LValue.
CValue TupleDLValue::emitStore(ASTExprAnd<CValue> value,
                               IREmitter &emitter) const {
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
  //   ValueError: too many values to unpack (expected 2)
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
                                                     CValue argRef,
                                                     ASTType eltType,
                                                     size_t argIndex)
    : BaseDLValue(eltType), argDecl(argDecl), argRef(argRef),
      argIndex(argIndex) {}

/// If this is a def argument shadow, resolve it to the incoming immutable
/// borrowed value without forming a local copy.  Otherwise return null.
MBValue DefArgumentWrapperDLValue::emitMBValueFromDefArgument(
    IREmitter &emitter) const {
  return argRef.getIfMBValue();
}

void DefArgumentWrapperDLValue::print(raw_ostream &os) const {
  os << "def argument wrapper of type " << elementType;
}

// This hook is called before an argument is passed mut.
LValue
DefArgumentWrapperDLValue::prepareForMutAccess(SMLoc loc,
                                               IREmitter &emitter) const {
  // Okay, if the by-reg def argument is mutated, we need to snap into action
  // and lazily build a shadow in the function entry.
  auto func = cast<FnOp>(argDecl->getParentDecl());

  // We may have already emitted read-only accesses that use the argument, and
  // they need to be revectored to the new vardecl.  Collect them so we can
  // update them later and not get confused by the access we're about to
  // generate. This can matter on things like:
  //    for ...:
  //      use(arg)  # Emitted as a direct use of the arg.
  //      arg += 1  # Forces mutation after the use was emitted.
  BlockArgument bbArg = func.getArgument(argIndex);
  SmallVector<OpOperand *> argUses;
  for (auto &use : bbArg.getUses())
    argUses.push_back(&use);

  IREmitter entryEmitter(*argDecl->getParentDecl(),
                         OpBuilder::atBlockBegin(func.getBody()));
  StringAttr argName = func.getFuncTypeGenerator().getArgName(argIndex);

  // Create the shadow box that has an address and copy the argument into it.
  VarDeclOp varDecl = entryEmitter.emitVarDecl(argName, argRef.getRValueType(),
                                               emitter.translateLocation(loc),
                                               VarDeclKind::Arg);
  if (!entryEmitter.emitStoreToLValue({argRef, SyntheticNode(loc)},
                                      MLValue(varDecl), EC_OwnedRegArgShadow)) {
    // This can fail if not copyable/movable.
    argDecl->setErroneous();
    return LValue();
  }

  // Now that we've got the new representation as an MLValue, we need to update
  // the previous uses.  The arg was either an MBValue (for normal types) or
  // SRValue for trivial types.
  bool isTrivial = argRef.getIfSRValue() != SRValue();
  for (OpOperand *use : argUses) {
    Value valueToUse = varDecl.getResult();
    auto *user = use->getOwner();
    OpBuilder builder(user);

    if (isTrivial) {
      // Trivial values need a load from the vardecl at the point of the use.
      valueToUse = builder.create<RefLoadOp>(user->getLoc(), valueToUse);
    } else {
      // Non-trivial need an adjustment of the reference type because the
      // origins mismatch.
      // FIXME: This won't propagate the origin of the vardecl correctly for
      // ref-returning values.
      valueToUse = builder.create<RebindOp>(user->getLoc(),
                                            use->get().getType(), valueToUse);
    }
    use->set(valueToUse);
  }

  // This helps debug info and QoI.
  varDecl.setArgShadowIndex(argIndex);
  // Update the representation so we don't do this again.
  argDecl->setIRValue(MLValue(varDecl));

  return MLValue(varDecl);
}

CValue DefArgumentWrapperDLValue::emitLoad(ValueDest &dest,
                                           IREmitter &emitter) const {
  // Loads of the def argument wrapper are simple enough.
  SyntheticNode expr(argDecl->getLoc());
  return emitter.emitCResult(argRef, &expr, dest);
}

CValue DefArgumentWrapperDLValue::emitStore(ASTExprAnd<CValue> value,
                                            IREmitter &emitter) const {
  // Okay, if the def argument is mutated, we need to snap into action and
  // lazily build a shadow in the function entry.
  LValue newVal = prepareForMutAccess(value.expr->getLoc(), emitter);
  if (!newVal)
    return BValue();

  // Ok, now emit a normal store.
  return emitter.emitStoreToLValue(value, newVal, ExprContext::EC_Assignment);
}
