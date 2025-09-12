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

// This hook is called if the DLValue needs to be resolved to a physical ref.
// This emits an error and returns null on failure.
Value BaseDLValue::emitAsRefValue(llvm::SMLoc loc, IREmitter &emitter) const {
  emitter.emitError(loc)
      << "cannot convert computed lvalue to a stored reference";
  return {};
}

Value BaseDLValue::emitAsRefValueIfOwned(llvm::SMLoc loc,
                                         IREmitter &emitter) const {
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
  // If the source already has an MLIR representation (thus is an MLValue,
  // SBValue etc) just returned the input value unmodified. This ensures we
  // don't load/copy lvalues in expressions like "_ = someVariable".  We do emit
  // an ownership use to extend the origin of the value to here and disable
  // "unused value" warnings.
  if (auto mlirValue = value.ir.getMlirValue()) {
    emitter.builder->create<OwnershipUseOp>(value.expr->getLocation(emitter),
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
  auto loc = expr->getLocation(emitter);
  // Try to bind the base (e.g., dict[key]) as a stored ref.
  if (Value baseRef =
          baseVal.ir->emitAsRefValueIfOwned(expr->getLoc(), emitter)) {
    auto fieldPtr =
        emitter.builder->create<RefStructGEROp>(loc, baseRef, getField());
    emitter.emitStoreToLValue(value, MLValue(fieldPtr), EC_AttributeRefBase);
    // Done: no tmp, no __setitem__, in-place mutation via the ref.
    return MLValue(baseRef);
  }
  // tmp = load(base)
  // tmp.field = value
  // store(tmp -> base)
  ASTType rvalueType = baseVal.ir->elementType;
  Value tmpDecl = emitter.emitVarDecl("__store_tmp__", rvalueType, loc,
                                      VarDeclKind::Synthesized);

  // Load the entire base LValue into tmpDecl.
  ValueDest tmpValueDest(MLValue(tmpDecl), EC_AttributeRefBase);
  auto base = baseVal.ir->emitLoad(tmpValueDest, emitter);
  if (!base) {
    tmpValueDest.resetForError(emitter);
    return BValue();
  }

  // Store into the field.
  auto fieldPtr =
      emitter.builder->create<RefStructGEROp>(loc, tmpDecl, getField());
  emitter.emitStoreToLValue(value, MLValue(fieldPtr), EC_AttributeRefBase);

  // Store the whole result back, transferring ownership as an MRValue.
  return baseVal.ir->emitStore({MRValue(tmpDecl), expr}, emitter);
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

Value SubscriptDLValue::emitRef(IREmitter &emitter) const {
  // Call the getter to get the ref.
  ValueDest storeDest(EC_RefBinding);
  auto ref = emitLoad(storeDest, emitter);
  if (!ref)
    return {}; // Error emitted by emitLoad.
  if (ref.isMValue())
    return ref.getMValueReference();
  // getitem returned something that isn't a ref.
  return {};
}

// Some subscripts, notably Dict, are defined with both a getter and a setter
// but the getter returns a ref (and throws).  If we need to bind the dict entry
// into a ref, call the getter.
Value SubscriptDLValue::emitAsRefValue(llvm::SMLoc loc,
                                       IREmitter &emitter) const {
  // If there is no getter, then this just fails like other computed lvalues.
  if (!getter)
    return BaseDLValue::emitAsRefValue(loc, emitter);
  if (Value value = emitRef(emitter))
    return value;
  return BaseDLValue::emitAsRefValue(loc, emitter);
}

Value SubscriptDLValue::emitAsRefValueIfOwned(llvm::SMLoc loc,
                                              IREmitter &emitter) const {
  // If there is no getter, then this just fails like other computed lvalues.
  if (getter) {
    // We only want to return a reference owned by the dlvalue. If the getter
    // populates its result via a byref_result then we don't want to return a
    // reference to the temporary.
    if (!cast<FnTypeGeneratorType>(getter.getType()).isRefResult())
      return {};
    return emitRef(emitter);
  }
  return BaseDLValue::emitAsRefValueIfOwned(loc, emitter);
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

  // If the value is a type with a statically known length, check that it agrees
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
    if (!isa<TypeCheckErrorType>(srcRValueType))
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
      eltDest.resetForError(emitter);
      return BValue();
    }
  }

  return bvalue;
}
