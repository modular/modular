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
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/IndexDialect/IndexOps.h"

using namespace M;
using namespace M::KGEN::LIT;

static const char *plural(size_t value) { return value == 1 ? "" : "s"; }

//===----------------------------------------------------------------------===//
// RValue / AnyValue Implementation
//===----------------------------------------------------------------------===//

Type RValue::getType() const {
  if (isNull())
    return Type();
  if (TypedAttr attr = getIfMValue())
    return attr.getType();
  return getIfDRValue().getType();
}

Type AnyValue::getType() const {
  if (isNull())
    return Type();
  if (RValue rvalue = getIfRValue())
    return rvalue.getType();

  LValue lvalue = getIfLValue();
  assert(lvalue && "Unknown type");
  return lvalue.getType();
}

//===----------------------------------------------------------------------===//
// ExprEmitter Implementation
//===----------------------------------------------------------------------===//

/// This helper emits the specified value rep as an SSA value, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
RValue ExprEmitter::emitRValue(AnyValue rep, SMLoc loc) {
  if (!rep) // Already diagnosed error.
    return {};

  if (!builder) {
    emitError(loc, "context only permits a meta value, not a dynamic one");
    return {};
  }

  if (auto rvRep = rep.getIfRValue())
    return rvRep;

  auto pointer = rep.getIfLValue();
  assert(pointer);

  // Finally, if this is an LValue, emit a load.
  return builder
      ->create<POP::LoadOp>(translateLocation(loc), pointer,
                            /*alignment*/ None)
      .getResult();
}

Value ExprEmitter::emitDRValue(RValue rep, SMLoc loc) {
  if (!rep)
    return {};
  // If this is already an DRValue, emit this.
  if (auto rvalue = rep.getIfDRValue())
    return rvalue;

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  auto attr = rep.getIfMValue();
  assert(attr);
  auto location = translateLocation(loc);
  // Materialize index integer constants as a special case.
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    if (intAttr.getType().isIndex())
      return builder->create<index::ConstantOp>(
          location, intAttr.getValue().getSExtValue());

  // Otherwise, emit a generalized parameter constant.
  return builder->create<ParamConstantOp>(location, attr);
}

/// This helper emits the specified value rep as a meta value, diagnosing the
/// problem if the expression is only valid as a runtime value.  This returns
/// null if emission fails.
TypedAttr ExprEmitter::emitMValue(const ExprNode *node, const Twine &message) {
  auto valueRep = node->emitIR(*this);
  if (!valueRep)
    return {};

  // If this is a parameter, return it.
  if (auto attr = valueRep.getIfMValue())
    return attr;

  emitError(node->getLoc(), message);
  return {};
}

/// This helper emits the specified expression tree as a type, e.g. turning
/// "Int" into the type for it.  This never returns null - if the expression
/// is erroneous, it is diagnosed and a TypeCheckErrorType is returned.
Type ExprEmitter::emitType(const ExprNode *node) {
  Type result = node->emitType(*this);
  // The emitType methods return null on failure, we return a
  // TypeCheckErrorType to simplify clients.
  return result ? result : TypeCheckErrorType::get(getContext());
}

/// Perform a name lookup in the current scope and return the named
/// declaration.  This emits an error and returns null on error.
Scope *ExprEmitter::lookupDecl(StringRef name, SMLoc loc) {
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
AnyValue ErrorNode::emitIR(ExprEmitter &emitter) const { return AnyValue(); }

Type ErrorNode::emitType(ExprEmitter &emitter) const { return Type(); }

AnyValue IntLiteralNode::emitIR(ExprEmitter &emitter) const {
  // TODO: Handle contextual types.
  APInt value = LitLexer::getIntegerLiteralValue(spelling);

  // Make sure the value fits in 64-bits.  There are no negative values here.
  // TODO: Detect overflow errors.
  value = value.zextOrTrunc(64);
  return IntegerAttr::get(IndexType::get(emitter.getContext()), value);
}

Type IntLiteralNode::emitType(ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return Type();
}

AnyValue FloatLiteralNode::emitIR(ExprEmitter &emitter) const {
  // TODO: this assumes float literal are always doubles
  APFloat value = LitLexer::getFloatLiteralValue(spelling);
  return FloatAttr::get(FloatType::getF64(emitter.getContext()),
                        APFloat(value.convertToDouble()));
}

Type FloatLiteralNode::emitType(ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return Type();
}

AnyValue StringLiteralNode::emitIR(ExprEmitter &emitter) const {
  std::string value = LitLexer::getStringLiteralValue(spelling);
  return StringAttr::get(emitter.getContext(), value);
}

Type StringLiteralNode::emitType(ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return Type();
}

AnyValue DeclRefNode::emitIR(ExprEmitter &emitter) const {
  Optional<Scope::NameEntry> declOrValue =
      emitter.scope.lookup(StringAttr::get(emitter.getContext(), spelling));
  if (!declOrValue) {
    emitter.emitError(getLoc(), "use of unknown declaration \"")
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
  auto resolveResult = emitter.shared.declResolver->resolve(
      scope, DeclResolvedness::signatureResolved, getLoc());

  // If the decl was erroneous somehow, then don't form a reference to it.
  if (failed(resolveResult))
    return {};

  // Variable references resolve to an lvalue addressing the variable.
  if (auto var = dyn_cast<VarDeclOp>(scope.getDecl()))
    return LValue(var.getResult());

  // Functions form an address.
  if (auto fnDecl = dyn_cast<LITFuncOp>(scope.getDecl()))
    return SymbolConstantAttr::get(FlatSymbolRefAttr::get(fnDecl.getNameAttr()),
                                   fnDecl.getSignature());

  emitter.emitError(getLoc(), "use of declaration \"")
      << spelling << "\" as a value isn't supported yet";
  return {};
}

Type DeclRefNode::emitType(ExprEmitter &emitter) const {
  auto *context = emitter.getContext();

  // TODO(types): This is a hack to unblock tests in the interim.
  if (spelling == "index")
    return IndexType::get(context);

  // Lookup the identifier.
  Scope *declScope = emitter.lookupDecl(spelling, getLoc());
  if (!declScope)
    return Type();
  auto typeDecl = dyn_cast<LITStructDeclOp>(declScope->getDecl());
  if (!typeDecl) {
    emitter.emitError(getLoc(), "'" + spelling + "' names a value, not a type");
    return Type();
  }

  auto numParams = typeDecl.getParamDecls().size();
  if (numParams != 0) {
    emitter.emitError(getLoc(), "'" + spelling + "' requires ")
        << numParams << " meta parameter" << plural(numParams);
    return Type();
  }

  return RefType::get(FlatSymbolRefAttr::get(typeDecl.getNameAttr()));
}

AnyValue CallNode::emitIR(ExprEmitter &emitter) const {
  auto calleeVal = emitter.emitRValue(callee);
  if (!calleeVal || isa<TypeCheckErrorType>(calleeVal.getType()))
    return {};

  // The only callable thing we have right now are functions.
  // TODO: Support struct initialization.
  auto calleeType = dyn_cast<SignatureType>(calleeVal.getType());
  if (!calleeType) {
    emitter.emitError(getLoc(), "unable to call value of type ")
        << calleeVal.getType();
    return {};
  }

  // If there are any unbound parameters then we cannot call it.
  // TODO: infer the parameters from the types of the operands.
  if (!calleeType.getInputParams().empty()) {
    emitter.emitError(getLoc(),
                      "unable to call parameterized value that expects ")
        << calleeType.getInputParams().size() << " bound parameters";
    return {};
  }

  assert(calleeType.getResultParamTypes().empty() &&
         "TODO: meta results not implemented yet");

  size_t numArgs = calleeType.getValues().getNumInputs();
  if (numArgs != args.size()) {
    emitter.emitError(getLoc(), "callee expects ")
        << numArgs << " argument" << plural(numArgs);
    return {};
  }

  // Emit all the arguments.
  SmallVector<Value> valueArguments;
  for (auto [arg, expectedType] :
       llvm::zip(args, calleeType.getValues().getInputs())) {
    auto argVal = emitter.emitDRValue(arg);
    if (!argVal)
      return {};

    if (argVal.getType() != expectedType) {
      // TODO: Handle implicit conversions.
      emitter.emitError(arg->getLoc(), "value of type ")
          << argVal.getType() << " cannot be converted to expected type "
          << expectedType;
      return {};
    }
    valueArguments.push_back(argVal);
  }

  auto calleeParam = calleeVal.getIfMValue();
  if (!calleeParam) {
    emitter.emitError(getLoc(), "TODO: indirect value call not supported yet");
    return {};
  }

  if (!emitter.builder) {
    emitter.emitError(getLoc(),
                      "TODO: cannot call function in parameter context");
    return {};
  }

  auto call = emitter.builder->create<CallParamOp>(
      emitter.translateLocation(getLoc()),
      /*resultTypes*/ calleeType.getValues().getResults(), calleeParam,
      /*inputParams*/ ArrayRef<ParamBindAttr>(),
      /*resultParams*/ ArrayRef<ParamDeclAttr>(),
      /*operands*/ valueArguments);

  // Value returning call returns its result.
  if (calleeType.getValues().getNumResults())
    return call.getResult(0);

  // On null return we return a UnitAttr to represent that this function
  // returned void - returning a nullptr indicates that there was an error on
  // emission.
  // TODO: We should replace this with an empty tuple when we support tuples.
  return VoidAttr::get(emitter.getContext(),
                       mlir::TupleType::get(emitter.getContext()));
}

Type CallNode::emitType(ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return Type();
}

AnyValue SubscriptNode::emitIR(ExprEmitter &emitter) const {
  // Subscripting a generic function binds the parameter expressions.
  auto subValue = base->emitIR(emitter);
  if (!subValue)
    return {};

  // If we have a value of signature type, we can bind parameters to it.
  if (auto signature = dyn_cast<SignatureType>(subValue.getType())) {
    size_t numParams = signature.getInputParams().size();
    if (numParams != indices.size()) {
      emitter.emitError(getLoc(), "signature expects ")
          << numParams << " parameter value" << plural(numParams);
      return {};
    }

    auto declParam = subValue.getIfMValue();
    if (!declParam) {
      emitter.emitError(getLoc(), "cannot parameterize dynamic value");
      return {};
    }

    // Emit each index as a meta value and type check it.
    SmallVector<TypedAttr> bindOperands;
    bindOperands.push_back(declParam);
    for (auto [idx, decl] : llvm::zip(indices, signature.getInputParams())) {
      auto val = emitter.emitMValue(
          idx, "declaration parameters may not be a run-time value");
      if (!val)
        return {};

      // Check the type matches what is expected.
      // TODO: Do implicit conversions.
      // TODO: Handle signatures like (T, scalar<T>) where early bound
      // parameters changes the types of later ones.
      if (val.getType() != decl.getType()) {
        emitter.emitError(idx->getLoc(), "index has type ")
            << val.getType() << " but declaration expects " << decl.getType();
        return {};
      }
      bindOperands.push_back(val);
    }
    // Okay, everything checks out, form the bind operation.
    return ParamOperatorAttr::get(POC::BindSignature, bindOperands);
  }

  // Emit each of the index values.
  SmallVector<RValue> indexValues;
  for (ExprNode *index : indices) {
    indexValues.push_back(emitter.emitRValue(index));
    if (!indexValues.back())
      return {};
  }

  emitter.emitError(getLoc(), "TODO: Subscript irgen not implemented yet ")
      << subValue.getType();
  return {};
}

Type SubscriptNode::emitType(ExprEmitter &emitter) const {
  // KGEN doesn't support unbound parametric types (e.g. "SIMD" with unbound
  // size/dtype) as a stand-alone type, so we handle name resolution here.
  auto baseDRE = dyn_cast<DeclRefNode>(base);
  if (!baseDRE) {
    if (Type baseType = base->emitType(emitter))
      emitter.emitError(getLoc(), "unknown parameterized type ") << baseType;
    return Type();
  }

  // Lookup the identifier.
  Scope *declScope = emitter.lookupDecl(baseDRE->spelling, getLoc());
  if (!declScope)
    return Type();

  auto typeDecl = dyn_cast<LITStructDeclOp>(declScope->getDecl());
  if (!typeDecl) {
    emitter.emitError(getLoc(),
                      "'" + baseDRE->spelling + "' names a value, not a type");
    return Type();
  }

  auto numParams = typeDecl.getParamDecls().size();
  if (numParams != indices.size()) {
    emitter.emitError(getLoc(), "'" + baseDRE->spelling + "' requires ")
        << numParams << " meta parameter" << plural(numParams) << " but "
        << indices.size() << " were specified";
    return Type();
  }

  // Emit each of the indices as parameter expressions.
  SmallVector<ParamBindAttr> exprs;
  for (auto [indexExpr, decl] : llvm::zip(indices, typeDecl.getParamDecls())) {
    // TODO: Slice syntax is the obvious way to support named parameter
    // arguments.
    auto value = emitter.emitMValue(
        indexExpr, "type parameters may not be a run-time value");
    if (!value)
      return {};

    // TODO: Support conversions.
    if (value.getType() != decl.getType()) {
      emitter.emitError(indexExpr->getLoc(), "parameter of type ")
          << value.getType() << " cannot be converted to expected type "
          << decl.getType();
      return {};
    }

    exprs.push_back(ParamBindAttr::get(decl, value));
  }

  return RefType::get(FlatSymbolRefAttr::get(typeDecl.getNameAttr()),
                      ParamBindArrayAttr::get(emitter.getContext(), exprs));
}

AnyValue ParenExprNode::emitIR(ExprEmitter &emitter) const {
  return subExpr->emitIR(emitter);
}

Type ParenExprNode::emitType(ExprEmitter &emitter) const {
  return subExpr->emitType(emitter);
}

AnyValue BinOpNode::emitIR(ExprEmitter &emitter) const {
  auto lhsRep = emitter.emitRValue(lhs);
  auto rhsRep = emitter.emitRValue(rhs);
  if (!lhsRep || !rhsRep)
    return {};
  auto lhsType = lhsRep.getType(), rhsType = rhsRep.getType();
  if (lhsType != rhsType || !lhsType.isIndex()) {
    emitter.emitError(getLoc(),
                      "binary operator with interesting types not implemented");
    return {};
  }

  // If these are both parameter values, we can fold them using parameter
  // expressions.
  if (auto lhsParam = lhsRep.getIfMValue()) {
    if (auto rhsParam = rhsRep.getIfMValue()) {
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

  assert(emitter.builder && "cannot have dynamic values without a builder");

  auto lhsVal = emitter.emitDRValue(lhsRep, lhs->getLoc());
  auto rhsVal = emitter.emitDRValue(rhsRep, rhs->getLoc());
  auto loc = emitter.translateLocation(getLoc());

  switch (kind) {
  default:
    llvm_unreachable("unknown binary operator");
  case kAdd:
    return (Value)emitter.builder->create<index::AddOp>(loc, lhsVal, rhsVal);
  case kMul:
    return (Value)emitter.builder->create<index::MulOp>(loc, lhsVal, rhsVal);
  }
}

Type BinOpNode::emitType(ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return Type();
}
