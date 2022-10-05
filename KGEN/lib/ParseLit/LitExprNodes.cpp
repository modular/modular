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
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/IndexDialect/IndexOps.h"
#include "llvm/ADT/PointerUnion.h"

using namespace M;
using namespace M::KGEN::LIT;

ExprNode::~ExprNode() { assert(0 && "never called"); }

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

  // FIXME: Should actually lower these as attributes so they can be used as
  // parameter expressions, not as SSA values, they should be converted to value
  // when used in that context.
  Value result = state.builder.create<index::ConstantOp>(
      state.mapLocation(getLoc()), value.getZExtValue());
  return result;
}

MLIRValueRep DeclRefNode::emit(EmitterState &state) const {
  Operation *decl = state.scope->lookup(spelling);
  if (!decl) {
    state.emitError(getLoc(), "use of unknown declaration \"")
        << spelling << '"';
    return {};
  }

  // Function references resolve to attributes.
  if (auto ref = dyn_cast<LITFuncOp>(decl))
    return (TypedAttr)SymbolConstantAttr::get(
        FlatSymbolRefAttr::get(ref.getSymNameAttr()), ref.getSignature());

  // Variable references resolve to load from the variable.
  if (auto var = dyn_cast<VarDeclOp>(decl))
    return state.builder
        .create<POP::LoadOp>(state.mapLocation(getLoc()), var,
                             /*alignment*/ None)
        .getResult();

  state.emitError(getLoc(), "cannot emit reference to decl yet");
  return {};
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

  auto calleeParam = dyn_cast<Attribute>(calleeVal);
  if (!calleeParam || !args.empty()) {
    state.emitError(getLoc(), "call not supported yet");
    return {};
  }

  state.builder.create<CallParamOp>(state.mapLocation(getLoc()),
                                    /*resultTypes*/ ArrayRef<Type>(),
                                    cast<TypedAttr>(calleeParam),
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

  // TODO: Add support for parameters exprs.
  auto lhsVal = dyn_cast<Value>(lhsRep);
  auto rhsVal = dyn_cast<Value>(rhsRep);
  if (!lhsVal || !rhsVal) {
    state.emitError(getLoc(),
                    "binary operator with parameters not implemented yet");
    return {};
  }

  auto lhsType = lhsVal.getType();
  if (lhsType != rhsVal.getType() || !lhsType.isIndex()) {
    state.emitError(getLoc(),
                    "binary operator with interesting types not implemented");
    return {};
  }

  switch (kind) {
  default:
    assert(0 && "unknown binary operator");
  case kAdd:
    return (Value)state.builder.create<index::AddOp>(
        state.mapLocation(getLoc()), lhsType, lhsVal, rhsVal);

  case kMul:
    return (Value)state.builder.create<index::MulOp>(
        state.mapLocation(getLoc()), lhsType, lhsVal, rhsVal);
  }
}
