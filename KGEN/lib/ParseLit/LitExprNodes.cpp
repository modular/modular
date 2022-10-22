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
#include "LitScope.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/IndexDialect/IndexOps.h"

using namespace M;
using namespace M::KGEN::LIT;

static const char *plural(size_t value) { return value == 1 ? "" : "s"; }

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
  if (!rep) // Already diagnosed error.
    return {};

  if (!builder) {
    emitError(loc, "context only permits a meta value, not a dynamic one");
    return {};
  }

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  if (auto attr = rep.dyn_castTypedAttr()) {
    auto location = translateLocation(loc);
    // Materialize index integer constants as a special case.
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      if (intAttr.getType().isIndex())
        return builder->create<index::ConstantOp>(
            location, intAttr.getValue().getSExtValue());

    // Otherwise, emit a generalized parameter constant.
    return builder->create<ParamConstantOp>(location, attr);
  }

  return cast<Value>(rep);
}

/// This helper emits the specified value rep as an SSA value, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
Value IREmitter::emitAsValue(const ExprNode *node) {
  assert(node && "cannot emit a null node");
  return emitAsValue(node->emitIR(*this), node->getLoc());
}

/// This helper emits the specified value rep as a meta value, diagnosing the
/// problem if the expression is only valid as a runtime value.  This returns
/// null if emission fails.
TypedAttr IREmitter::emitAsMetaValue(const ExprNode *node) {
  auto valueRep = node->emitIR(*this);
  if (!valueRep)
    return {};

  // If this is a parameter, return it.
  if (auto attr = valueRep.dyn_castTypedAttr())
    return attr;

  emitError(node->getLoc(),
            "context only permits a meta value, not a dynamic one");
  return {};
}

/// This helper emits the specified expression tree as a type, e.g. turning
/// "Int" into the type for it.  This never returns null - if the expression
/// is erroneous, it is diagnosed and a TypeCheckErrorType is returned.
Type IREmitter::emitAsType(const ExprNode *node) {
  Type result = node->emitType(*this);
  // The emitType methods return null on failure, we return a TypeCheckErrorType
  // to simplify clients.
  return result ? result : TypeCheckErrorType::get(getContext());
}

/// Perform a name lookup in the current scope and return the named
/// declaration.  This emits an error and returns null on error.
Scope *IREmitter::lookupDecl(StringRef name, SMLoc loc) {
  Optional<Scope::NameEntry> lookupResult =
      scope.lookup(StringAttr::get(getContext(), name));
  if (!lookupResult)
    return emitError(loc, "unknown type name '" + name + "'"), nullptr;
  if (!std::holds_alternative<Scope *>(*lookupResult))
    return emitError(loc, "'" + name + "' names a value, not a type"), nullptr;

  Scope &scope = *std::get<Scope *>(*lookupResult);

  // We need the signature for the struct to be resolved in order to know how
  // to refer to it.
  auto resolveResult = shared.declResolver->resolve(
      scope, DeclResolvedness::signatureResolved, loc);

  // If the decl was erroneous somehow, then don't form a reference to it, the
  // error has already been diagnosed.
  if (failed(resolveResult))
    return nullptr;
  return &scope;
}

//===----------------------------------------------------------------------===//
// ExprNode implementations
//===----------------------------------------------------------------------===//

ExprNode::~ExprNode() { llvm_unreachable("never called"); }

/// Error nodes cannot be emitted and have already been diagnosed.
MLIRValueRep ErrorNode::emitIR(IREmitter &state) const {
  return MLIRValueRep();
}

Type ErrorNode::emitType(IREmitter &state) const { return Type(); }

MLIRValueRep IntLiteralNode::emitIR(IREmitter &state) const {
  // TODO: Handle contextual types.
  APInt value = LitLexer::getIntegerLiteralValue(spelling);

  // Make sure the value fits in 64-bits.  There are no negative values here.
  // TODO: Detect overflow errors.
  value = value.zextOrTrunc(64);
  return IntegerAttr::get(IndexType::get(state.getContext()), value);
}

Type IntLiteralNode::emitType(IREmitter &state) const {
  state.emitError(getLoc(), "cannot emit this expression as a type");
  return Type();
}

MLIRValueRep FloatLiteralNode::emitIR(IREmitter &state) const {
  // TODO: this assumes float literal are always doubles
  APFloat value = LitLexer::getFloatLiteralValue(spelling);
  return FloatAttr::get(FloatType::getF64(state.getContext()),
                        APFloat(value.convertToDouble()));
}

Type FloatLiteralNode::emitType(IREmitter &state) const {
  state.emitError(getLoc(), "cannot emit this expression as a type");
  return Type();
}

MLIRValueRep StringLiteralNode::emitIR(IREmitter &state) const {
  std::string value = LitLexer::getStringLiteralValue(spelling);
  return StringAttr::get(state.getContext(), value);
}

Type StringLiteralNode::emitType(IREmitter &state) const {
  state.emitError(getLoc(), "cannot emit this expression as a type");
  return Type();
}

MLIRValueRep DeclRefNode::emitIR(IREmitter &state) const {
  Optional<Scope::NameEntry> declOrValue =
      state.scope.lookup(StringAttr::get(state.getContext(), spelling));
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
    if (!state.builder) {
      state.emitError(getLoc(),
                      "cannot load dynamic value in meta value context");
      return {};
    }

    return state.builder
        ->create<POP::LoadOp>(state.translateLocation(getLoc()), var,
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

Type DeclRefNode::emitType(IREmitter &state) const {
  auto *context = state.getContext();

  // TODO(types): This is a hack to unblock tests in the interim.
  if (spelling == "index")
    return IndexType::get(context);

  // Lookup the identifier.
  Scope *declScope = state.lookupDecl(spelling, getLoc());
  if (!declScope)
    return Type();
  auto typeDecl = dyn_cast<LITStructDeclOp>(declScope->getDecl());
  if (!typeDecl) {
    state.emitError(getLoc(), "'" + spelling + "' names a value, not a type");
    return Type();
  }

  auto numParams = typeDecl.getParamDecls().size();
  if (numParams != 0) {
    state.emitError(getLoc(), "'" + spelling + "' requires ")
        << numParams << " meta parameter" << plural(numParams);
    return Type();
  }

  return RefType::get(FlatSymbolRefAttr::get(typeDecl.getNameAttr()));
}

MLIRValueRep CallNode::emitIR(IREmitter &state) const {
  auto calleeVal = callee->emitIR(state);
  if (!calleeVal)
    return {};

  // Emit all the arguments. TODO: Handle contextual types.
  for (auto arg : args) {
    auto argVal = arg->emitIR(state);
    if (!argVal)
      return {};
  }
  // TODO: Pass arguments.

  auto calleeParam = calleeVal.dyn_castTypedAttr();
  if (!calleeParam || !args.empty()) {
    state.emitError(getLoc(), "TODO: value call not supported yet");
    return {};
  }

  if (!state.builder) {
    state.emitError(getLoc(),
                    "TODO: cannot call function in parameter context");
    return {};
  }

  state.builder->create<CallParamOp>(state.translateLocation(getLoc()),
                                     /*resultTypes*/ ArrayRef<Type>(),
                                     calleeParam,
                                     /*inputParams*/ ArrayRef<ParamBindAttr>(),
                                     /*resultParams*/ ArrayRef<ParamDeclAttr>(),
                                     /*operands*/ ArrayRef<Value>());

  // FIXME: Need a correct representation for a non-error void return.
  return {};
}

Type CallNode::emitType(IREmitter &state) const {
  state.emitError(getLoc(), "cannot emit this expression as a type");
  return Type();
}

MLIRValueRep SubscriptNode::emitIR(IREmitter &state) const {
  state.emitError(getLoc(), "TODO: Subscript irgen not implemented yet");
  return {};
}

Type SubscriptNode::emitType(IREmitter &state) const {
  // KGEN doesn't support unbound parametric types (e.g. "SIMD" with unbound
  // size/dtype) as a stand-alone type, so we handle name resolution here.
  auto baseDRE = dyn_cast<DeclRefNode>(base);
  if (!baseDRE) {
    if (Type baseType = base->emitType(state))
      state.emitError(getLoc(), "unknown parameterized type ") << baseType;
    return Type();
  }

  // Lookup the identifier.
  Scope *declScope = state.lookupDecl(baseDRE->spelling, getLoc());
  if (!declScope)
    return Type();

  auto typeDecl = dyn_cast<LITStructDeclOp>(declScope->getDecl());
  if (!typeDecl) {
    state.emitError(getLoc(),
                    "'" + baseDRE->spelling + "' names a value, not a type");
    return Type();
  }

  auto numParams = typeDecl.getParamDecls().size();
  if (numParams != indices.size()) {
    state.emitError(getLoc(), "'" + baseDRE->spelling + "' requires ")
        << numParams << " meta parameter" << plural(numParams) << " but "
        << indices.size() << " were specified";
    return Type();
  }

  // Emit each of the indices as parameter expressions.
  SmallVector<ParamBindAttr> exprs;
  for (auto [indexExpr, decl] : llvm::zip(indices, typeDecl.getParamDecls())) {
    // TODO: Slice syntax is the obvious way to support named parameter
    // arguments.
    auto value = state.emitAsMetaValue(indexExpr);
    if (!value)
      return {};

    // TODO: Support conversions.
    if (value.getType() != decl.getType()) {
      state.emitError(indexExpr->getLoc(), "parameter of type ")
          << value.getType() << " cannot be converted to expected type "
          << decl.getType();
      return {};
    }

    exprs.push_back(ParamBindAttr::get(decl, value));
  }

  return RefType::get(FlatSymbolRefAttr::get(typeDecl.getNameAttr()),
                      ParamBindArrayAttr::get(state.getContext(), exprs));
}

MLIRValueRep ParenExprNode::emitIR(IREmitter &state) const {
  return subExpr->emitIR(state);
}

Type ParenExprNode::emitType(IREmitter &state) const {
  return subExpr->emitType(state);
}

MLIRValueRep BinOpNode::emitIR(IREmitter &state) const {
  auto lhsRep = lhs->emitIR(state);
  auto rhsRep = rhs->emitIR(state);
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

  assert(state.builder && "cannot have dynamic values without a builder");

  auto lhsVal = state.emitAsValue(lhsRep, lhs->getLoc());
  auto rhsVal = state.emitAsValue(rhsRep, rhs->getLoc());
  auto loc = state.translateLocation(getLoc());

  switch (kind) {
  default:
    llvm_unreachable("unknown binary operator");
  case kAdd:
    return (Value)state.builder->create<index::AddOp>(loc, lhsVal, rhsVal);
  case kMul:
    return (Value)state.builder->create<index::MulOp>(loc, lhsVal, rhsVal);
  }
}

Type BinOpNode::emitType(IREmitter &state) const {
  state.emitError(getLoc(), "cannot emit this expression as a type");
  return Type();
}
