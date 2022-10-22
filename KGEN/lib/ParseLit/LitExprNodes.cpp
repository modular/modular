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
#include "LitDecls.h"
#include "LitParserBase.h"
#include "LitScope.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITTypes.h"
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
    return cast<TypedAttr>(attr);
  return {};
}

Type MLIRValueRep::getType() const {
  if (!*this)
    return Type();
  if (TypedAttr attr = dyn_castTypedAttr())
    return attr.getType();
  return cast<Value>(*this).getType();
}

//===----------------------------------------------------------------------===//
// IREmitter Implementation
//===----------------------------------------------------------------------===//

/// This helper emits the specified value rep as an SSA value, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
Value IREmitter::emitAsValue(MLIRValueRep rep, SMLoc loc) {
  if (!rep)
    return {};

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  if (auto attr = rep.dyn_castTypedAttr()) {
    auto location = translateLocation(loc);
    // Materialize index integer constants as a special case.
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      if (intAttr.getType().isIndex())
        return builder.create<index::ConstantOp>(
            location, intAttr.getValue().getSExtValue());

    // Otherwise, emit a generalized parameter constant.
    return builder.create<ParamConstantOp>(location, attr);
  }

  return cast<Value>(rep);
}

/// This helper emits the specified value rep as an SSA value, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
Value IREmitter::emitAsValue(const ExprNode *node) {
  assert(node && "cannot emit a null node");
  return emitAsValue(node->emit(*this), node->getLoc());
}

//===----------------------------------------------------------------------===//
// Type Parsing implementations
//===----------------------------------------------------------------------===//

/// Given a cursor location for a type expression that correctly parsed in the
/// first pass, reparse it into an expression and resolve it into a type by
/// performing name lookup and other resolution.  This can produce errors, but
/// always returns a non-null type.
ParseResult LitParserBase::parseType(Type &result, Scope &scope) {
  ExprNode *typeExpr = nullptr;
  if (parseExpression(typeExpr))
    return failure();

  auto emitError = [&](const Twine &message) -> ParseResult {
    result = TypeCheckErrorType::get(getContext());
    this->emitError(typeExpr->getLoc(), message);
    return success(); // Semantic error, but the parse succeeded.
  };

  // TODO: Make this a recursive walk when we have more interesting types.
  if (auto dre = dyn_cast<DeclRefNode>(typeExpr)) {
    // TODO(types): This is a hack to unblock tests in the interim.
    if (dre->spelling == "index") {
      result = IndexType::get(getContext());
      return success();
    }

    // Lookup the identifier.
    Optional<Scope::NameEntry> lookup =
        scope.lookup(StringAttr::get(getContext(), dre->spelling));
    if (!lookup)
      return emitError("unknown type name '" + dre->spelling + "'");
    if (!std::holds_alternative<Scope *>(*lookup))
      return emitError("'" + dre->spelling + "' names a value, not a type");

    Scope &scope = *std::get<Scope *>(*lookup);
    auto typeDecl = dyn_cast<LITStructDeclOp>(scope.getDecl());
    if (!typeDecl)
      return emitError("'" + dre->spelling + "' names a value, not a type");

    // We need the signature for the struct to be resolved in order to know how
    // to refer to it.
    auto resolveResult = getDeclResolver().resolve(
        scope, DeclResolvedness::signatureResolved, dre->getLoc());

    // If the decl was erroneous somehow, then don't form a reference to it.
    // Just return an TypeCheckError instead so we don't get downstream errors.
    if (failed(resolveResult)) {
      result = TypeCheckErrorType::get(getContext());
      return success();
    }

    // TODO: Handle type parameters!
    result = RefType::get(FlatSymbolRefAttr::get(typeDecl.getNameAttr()),
                          ParamBindArrayAttr::get(getContext(), {}));
    return success();
  }

  // Parse errors always resolve as TypeCheckErrorType since they have already
  // been diagnosed.
  if (isa<ErrorNode>(typeExpr)) {
    result = TypeCheckErrorType::get(getContext());
    return success();
  }

  return emitError("FIXME: Unsupported type kind!");
}

//===----------------------------------------------------------------------===//
// ExprNode implementations
//===----------------------------------------------------------------------===//

ExprNode::~ExprNode() { llvm_unreachable("never called"); }

/// Error nodes cannot be emitted.
MLIRValueRep ErrorNode::emit(IREmitter &state) const { return MLIRValueRep(); }

MLIRValueRep IntLiteralNode::emit(IREmitter &state) const {
  // TODO: Handle contextual types.
  APInt value = LitLexer::getIntegerLiteralValue(spelling);

  // Make sure the value fits in 64-bits.  There are no negative values here.
  // TODO: Detect overflow errors.
  value = value.zextOrTrunc(64);
  return IntegerAttr::get(state.builder.getIndexType(), value);
}

MLIRValueRep FloatLiteralNode::emit(IREmitter &state) const {
  // TODO: this assumes float literal are always doubles
  APFloat value = LitLexer::getFloatLiteralValue(spelling);
  return state.builder.getF64FloatAttr(value.convertToDouble());
}

MLIRValueRep StringLiteralNode::emit(IREmitter &state) const {
  std::string value = LitLexer::getStringLiteralValue(spelling);
  return state.builder.getStringAttr(value);
}

MLIRValueRep DeclRefNode::emit(IREmitter &state) const {
  Optional<Scope::NameEntry> declOrValue =
      state.scope.lookup(state.builder.getStringAttr(spelling));
  if (!declOrValue) {
    state.emitError(getLoc(), "use of unknown declaration \"")
        << spelling << '"';
    return {};
  }

  // Attributes always resolve to their known value.
  if (std::holds_alternative<Scope::MetaParameterValue>(declOrValue.value()))
    return std::get<Scope::MetaParameterValue>(declOrValue.value()).getAttr();

  // References to decls have different access paths.
  Scope &scope = *std::get<Scope *>(declOrValue.value());

  // We need the signature for the struct to be resolved in order to know how
  // to refer to it.
  auto resolveResult = state.shared.declResolver->resolve(
      scope, DeclResolvedness::signatureResolved, getLoc());

  // If the decl was erroneous somehow, then don't form a reference to it.
  if (failed(resolveResult))
    return {};

  // Variable references resolve to load from the variable.
  if (auto var = dyn_cast<VarDeclOp>(scope.getDecl())) {
    return state.builder
        .create<POP::LoadOp>(state.translateLocation(getLoc()), var,
                             /*alignment*/ None)
        .getResult();
  }

  // Functions form an address.
  if (auto fnDecl = dyn_cast<LITFuncOp>(scope.getDecl()))
    return SymbolConstantAttr::get(FlatSymbolRefAttr::get(fnDecl.getNameAttr()),
                                   fnDecl.getSignature());

  state.emitError(getLoc(), "use of declaration \"")
      << spelling << "\" as a value isn't supported yet";
  return {};
}

MLIRValueRep CallNode::emit(IREmitter &state) const {
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

MLIRValueRep ParenExprNode::emit(IREmitter &state) const {
  return subExpr->emit(state);
}

MLIRValueRep BinOpNode::emit(IREmitter &state) const {
  auto lhsRep = lhs->emit(state);
  auto rhsRep = rhs->emit(state);
  if (!lhsRep || !rhsRep)
    return {};
  auto lhsType = lhsRep.getType(), rhsType = rhsRep.getType();
  if (lhsType != rhsType || !lhsType.isIndex()) {
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
  auto loc = state.translateLocation(getLoc());

  switch (kind) {
  default:
    llvm_unreachable("unknown binary operator");
  case kAdd:
    return (Value)state.builder.create<index::AddOp>(loc, lhsVal, rhsVal);
  case kMul:
    return (Value)state.builder.create<index::MulOp>(loc, lhsVal, rhsVal);
  }
}
