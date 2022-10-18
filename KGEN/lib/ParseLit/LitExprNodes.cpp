//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements logic related to the expression nodes for the Lightning
// language.
//
//===----------------------------------------------------------------------===//

#include "LitExprNodes.h"
#include "LitLexer.h"
#include "LitScope.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/IndexDialect/IndexOps.h"

using namespace M;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// MLIRValueRep Implementation
//===----------------------------------------------------------------------===//

/// If this contains an Attribute, it is known to be a TypedAttr.  This helper
/// performs the conversion.  This returns null if this contains a value.
TypedAttr MLIRValueRep::dyn_castTypedAttr() const {
  if (auto attr = (*this).dyn_cast<Attribute>())
    return attr.cast<TypedAttr>();
  return {};
}

Type MLIRValueRep::getType() const {
  if (!*this)
    return Type();
  if (TypedAttr attr = dyn_castTypedAttr())
    return attr.getType();
  return cast<Value>(*this).getType();
}

/// This helper emits this MLIRValueRep as an SSA value, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
Value MLIRValueRep::getAsValue(Location loc, OpBuilder &builder) const {
  if (!*this)
    return {};

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  if (auto attr = this->dyn_castTypedAttr()) {
    // Materialize index integer constants as a special case.
    if (auto intAttr = attr.dyn_cast<IntegerAttr>())
      if (intAttr.getType().isIndex())
        // TODO: This shouldn't require passing in the type.
        return builder.create<index::ConstantOp>(loc, intAttr.getType(),
                                                 intAttr);

    // Otherwise, emit a generalized parameter constant.
    return builder.create<ParamConstantOp>(loc, attr);
  }

  return cast<Value>(*this);
}

//===----------------------------------------------------------------------===//
// EmitterState Implementation
//===----------------------------------------------------------------------===//

/// This helper emits the specified value rep as an SSA value, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
Value EmitterState::emitAsValue(MLIRValueRep rep, SMLoc loc) {
  return rep.getAsValue(translateLocation(loc), builder);
}

/// This helper emits the specified value rep as an SSA value, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
Value EmitterState::emitAsValue(const ExprNode *node) {
  assert(node && "cannot emit a null node");
  return emitAsValue(node->emit(*this), node->getLoc());
}

//===----------------------------------------------------------------------===//
// ExprNode implementations
//===----------------------------------------------------------------------===//

ExprNode::~ExprNode() { llvm_unreachable("never called"); }

/// Error nodes cannot be emitted.
MLIRValueRep ErrorNode::emit(EmitterState &state) const {
  return MLIRValueRep();
}

MLIRValueRep IntLiteralNode::emit(EmitterState &state) const {
  // TODO: Handle contextual types.
  APInt value = LitLexer::getIntegerLiteralValue(spelling);

  // Make sure the value fits in 64-bits.  There are no negative values here.
  // TODO: Detect overflow errors.
  value = value.zextOrTrunc(64);
  return IntegerAttr::get(state.builder.getIndexType(), value);
}

MLIRValueRep FloatLiteralNode::emit(EmitterState &state) const {
  // TODO: this assumes float literal are always doubles
  APFloat value = LitLexer::getFloatLiteralValue(spelling);
  return state.builder.getF64FloatAttr(value.convertToDouble());
}

MLIRValueRep StringLiteralNode::emit(EmitterState &state) const {
  std::string value = LitLexer::getStringLiteralValue(spelling);
  return state.builder.getStringAttr(value);
}

MLIRValueRep DeclRefNode::emit(EmitterState &state) const {
  Optional<Scope::ScopeValue> declOrValue = state.scope.lookup(spelling);
  if (!declOrValue) {
    state.emitError(getLoc(), "use of unknown declaration \"")
        << spelling << '"';
    return {};
  }

  // Attributes always resolve to their known value.
  if (std::holds_alternative<Scope::MetaParameterValue>(declOrValue.value()))
    return std::get<Scope::MetaParameterValue>(declOrValue.value()).getAttr();

  // Variable references resolve to load from the variable.
  auto var = std::get<VarDeclOp>(declOrValue.value());
  return state.builder
      .create<POP::LoadOp>(state.translateLocation(getLoc()), var,
                           /*alignment*/ None)
      .getResult();
}

MLIRValueRep CallNode::emit(EmitterState &state) const {
  auto calleeVal = callee->emit(state);
  if (!calleeVal)
    return {};

  // Emit all the arguments. TODO: Handle contextual types.
  for (auto arg : args) {
    auto argVal = arg->emit(state);
    if (!argVal)
      return {};
  }
  // TODO: Pass arguments.

  auto calleeParam = calleeVal.dyn_castTypedAttr();
  if (!calleeParam || !args.empty()) {
    state.emitError(getLoc(), "TODO: value call not supported yet");
    return {};
  }

  state.builder.create<CallParamOp>(state.translateLocation(getLoc()),
                                    /*resultTypes*/ ArrayRef<Type>(),
                                    calleeParam,
                                    /*inputParams*/ ArrayRef<ParamBindAttr>(),
                                    /*resultParams*/ ArrayRef<ParamDeclAttr>(),
                                    /*operands*/ ArrayRef<Value>());

  // FIXME: Need a correct representation for a non-error void return.
  return {};
}

MLIRValueRep ParenExprNode::emit(EmitterState &state) const {
  return subExpr->emit(state);
}

MLIRValueRep BinOpNode::emit(EmitterState &state) const {
  auto lhsRep = lhs->emit(state);
  auto rhsRep = rhs->emit(state);
  if (!lhsRep || !rhsRep)
    return {};

  auto lhsType = lhsRep.getType();
  if (lhsType != rhsRep.getType() || !lhsType.isIndex()) {
    state.emitError(getLoc(),
                    "binary operator with interesting types not implemented");
    return {};
  }

  // If these are both parameter values, we can fold them using parameter
  // expressions.
  if (auto lhsParam = lhsRep.dyn_castTypedAttr()) {
    if (auto rhsParam = rhsRep.dyn_castTypedAttr()) {
      POC opcode;
      switch (kind) {
      default:
        llvm_unreachable("unknown binary operator");
      case kAdd:
        opcode = POC::Add;
        break;
      case kMul:
        opcode = POC::Mul;
        break;
      }
      return ParamOperatorAttr::get(opcode, lhsParam, rhsParam);
    }
  }

  auto lhsVal = state.emitAsValue(lhsRep, lhs->getLoc());
  auto rhsVal = state.emitAsValue(rhsRep, rhs->getLoc());

  switch (kind) {
  default:
    llvm_unreachable("unknown binary operator");
  case kAdd:
    return (Value)state.builder.create<index::AddOp>(
        state.translateLocation(getLoc()), lhsVal, rhsVal);

  case kMul:
    return (Value)state.builder.create<index::MulOp>(
        state.translateLocation(getLoc()), lhsVal, rhsVal);
  }
}
