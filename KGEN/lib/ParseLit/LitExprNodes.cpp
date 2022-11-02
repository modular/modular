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
#include "LitASTDecl.h"
#include "LitDecls.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LitSharedState.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace M;
using namespace M::KGEN::LIT;
namespace scf = mlir::scf;

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
ASTTypeAnd<RValue> ExprEmitter::emitRValue(ASTTypeAnd<AnyValue> rep,
                                           SMLoc loc) {
  if (!rep) // Already diagnosed error.
    return {};

  if (!builder) {
    emitError(loc, "context only permits a meta value, not a dynamic one");
    return {};
  }

  if (auto rvRep = rep.ir.getIfRValue())
    return {rvRep, rep.type};

  auto pointer = rep.ir.getIfLValue();
  assert(pointer);

  // Finally, if this is an LValue, emit a load.
  Value load = builder->create<POP::LoadOp>(translateLocation(loc), pointer,
                                            /*alignment*/ None);
  return {DRValue(load), rep.type};
}

ASTTypeAnd<DRValue> ExprEmitter::emitDRValue(ASTTypeAnd<RValue> rep,
                                             SMLoc loc) {
  if (!rep)
    return {};
  // If this is already an DRValue, emit this.
  if (auto rvalue = rep.ir.getIfDRValue())
    return {rvalue, rep.type};

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  auto attr = rep.ir.getIfMValue();
  assert(attr);
  auto location = translateLocation(loc);
  // Materialize index integer constants as a special case.
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    if (intAttr.getType().isIndex()) {
      auto cst = builder->create<mlir::index::ConstantOp>(
          location, intAttr.getValue().getSExtValue());
      return {DRValue(cst), rep.type};
    }

  // Otherwise, emit a generalized parameter constant.
  return {DRValue(builder->create<ParamConstantOp>(location, attr)), rep.type};
}

/// This helper emits the specified expression as a meta value, diagnosing the
/// problem if the expression is only valid as a runtime value.  This returns
/// null if emission fails.
ASTTypeAnd<MValue> ExprEmitter::emitMValue(const ExprNode *node,
                                           const Twine &message) {
  auto rep = node->emitIR(*this);
  if (!rep)
    return {};

  // If this is a parameter, return it.
  if (auto attr = rep.ir.getIfMValue())
    return {attr, rep.type};

  emitError(node->getLoc(), message);
  return {};
}

/// Emit the specified expression as an LValue which can be loaded and stored.
/// If contextualType is non-null, then an implicitly declared LValue will
/// that that type.
///
/// This diagnoses the expression with the specified message if it isn't a
/// valid LValue.
ASTTypeAnd<LValue> ExprEmitter::emitLValue(const ExprNode *node,
                                           FullType contextualType,
                                           const Twine &message) {
  ASTTypeAnd<AnyValue> anyValue = node->emitIR(*this, contextualType);
  if (!anyValue)
    return {}; // Error already diagnosed.
  if (LValue lValue = anyValue.ir.getIfLValue())
    return {lValue, anyValue.type};
  emitError(node->getLoc(), message);
  return {};
}

/// This helper emits the specified expression tree as a type, e.g. turning
/// "Int" into the type for it.  This never returns null - if the expression
/// is erroneous, it is diagnosed and a TypeCheckErrorType is returned.
FullType ExprEmitter::emitType(const ExprNode *node) {
  FullType result = node->emitType(*this);

  // The emitType methods return null on failure, we return a
  // TypeCheckErrorType to simplify clients.
  if (!result.first) {
    result.first = TypeCheckErrorType::get(getContext());
    result.second = shared.getTypeCheckErrorType();
  }
  // TODO: Should this return an magic decl marked erroneous?
  return result;
}

/// Perform a name lookup in the current scope and return the named
/// declaration.  This emits an error and returns null on error.
ASTDecl *ExprEmitter::lookupDecl(StringRef name, SMLoc loc, ASTDecl &scope,
                                 Twine errorMessage) {
  ASTDecl *lookupResult = scope.lookup(StringAttr::get(getContext(), name));
  if (!lookupResult)
    return emitError(loc, errorMessage + " '" + name + "'"), nullptr;

  // We need the signature for the struct to be resolved in order to know how
  // to refer to it.
  auto resolveResult = shared.declResolver->resolve(
      *lookupResult, DeclResolvedness::signatureResolved, loc);

  // If the decl was erroneous somehow, then don't form a reference to it, the
  // error has already been diagnosed.
  if (failed(resolveResult))
    return nullptr;
  return lookupResult;
}

//===----------------------------------------------------------------------===//
// ExprNode implementations
//===----------------------------------------------------------------------===//

ExprNode::~ExprNode() { llvm_unreachable("never called"); }

ASTTypeAnd<AnyValue> IntLiteralNode::emitIR(ExprEmitter &emitter,
                                            FullType contextualType) const {
  // TODO: Handle contextual types.
  APInt value = LitLexer::getIntegerLiteralValue(spelling);

  // Make sure the value fits in 64-bits.  There are no negative values here.
  // TODO: Detect overflow errors.
  value = value.zextOrTrunc(64);
  auto attr = IntegerAttr::get(IndexType::get(emitter.getContext()), value);
  return {MValue(attr), emitter.shared.getIndexType()};
}

FullType IntLiteralNode::emitType(ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return {};
}

ASTTypeAnd<AnyValue> FloatLiteralNode::emitIR(ExprEmitter &emitter,
                                              FullType contextualType) const {
  // TODO: this assumes float literal are always doubles
  APFloat value = LitLexer::getFloatLiteralValue(spelling);
  auto attr = FloatAttr::get(FloatType::getF64(emitter.getContext()),
                             APFloat(value.convertToDouble()));
  return {MValue(attr),
          // FIXME: Wrong type!
          emitter.shared.getNoneType()};
}

FullType FloatLiteralNode::emitType(ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return {};
}

ASTTypeAnd<AnyValue> StringLiteralNode::emitIR(ExprEmitter &emitter,
                                               FullType contextualType) const {
  std::string value = LitLexer::getStringLiteralValue(spelling);
  return {MValue(StringAttr::get(emitter.getContext(), value)),
          // FIXME: Wrong type!
          emitter.shared.getNoneType()};
}

FullType StringLiteralNode::emitType(ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return {};
}

ASTTypeAnd<AnyValue> NoneLiteralNode::emitIR(ExprEmitter &emitter,
                                             FullType contextualType) const {
  // FIXME (Issue #4315): None should be emitted as an attribute (not a dynamic
  // value), but KGEN doesn't allow unknown parameters. This should work:
  //
  // return NoneAttr::get(emitter.getContext(), emitType(emitter));
  if (!emitter.builder) {
    emitter.emitError(
        getLoc(), "TODO(Issue #4315) we need a builder to emit None values");
    return {};
  }

  auto loc = emitter.translateLocation(getLoc());
  auto type = KGEN::NoneType::get(emitter.getContext());
  return {DRValue(emitter.builder->create<NoneValueOp>(loc, type).getResult()),
          emitter.shared.getNoneType()};
}

FullType NoneLiteralNode::emitType(ExprEmitter &emitter) const {
  return {KGEN::NoneType::get(emitter.getContext()),
          emitter.shared.getNoneType()};
}

ASTTypeAnd<AnyValue> DeclRefNode::emitIR(ExprEmitter &emitter,
                                         FullType contextualType) const {
  // Look up the name.
  auto nameAttr = StringAttr::get(emitter.getContext(), spelling);
  ASTDecl *decl = emitter.declScope.lookup(nameAttr);

  // Handle the case where lookup fails.
  if (!decl) {
    // If there is a contextual type available then this is an implicit variable
    // definition, otherwise it is an error.
    if (!contextualType.first || !emitter.varDeclCursor) {
      emitter.emitError(getLoc(), "use of unknown declaration \"")
          << spelling << '"';
      return {};
    }

    // Otherwise, introduce a new lit.var.decl node whose type matches the
    // initializer expression.
    //
    // TODO(autopromotions): turn infinite integers into concrete ones as
    // needed.
    auto declType = POP::PointerType::get(contextualType.first);

    // Use this builder to place any VarDeclOps. In Python there is only one
    // scope per function and all variables belong to that scope, so builders
    // should reflect that.
    auto varDecl = OpBuilder(emitter.varDeclCursor)
                       .create<VarDeclOp>(emitter.translateLocation(getLoc()),
                                          declType, nameAttr);
    decl = &emitter.shared.declResolver->addFullyResolvedDecl(
        varDecl, contextualType.second, &emitter.declScope);
  }

  // We need the signature for the struct to be resolved in order to know how
  // to refer to it.
  auto resolveResult = emitter.shared.declResolver->resolve(
      *decl, DeclResolvedness::signatureResolved, getLoc());

  // If the decl was erroneous somehow, then don't form a reference to it.
  if (failed(resolveResult))
    return {};

  // Variable references resolve to an lvalue addressing the variable.
  if (auto var = dyn_cast<VarDeclOp>(*decl))
    return {LValue(var.getResult()), decl->getResolvedType()};

  // Functions form an address.
  if (auto fnDecl = dyn_cast<LITFuncOp>(*decl)) {
    auto attr = SymbolConstantAttr::get(
        FlatSymbolRefAttr::get(fnDecl.getNameAttr()), fnDecl.getSignature());
    return {MValue(attr),
            // TODO: Correct signature type.
            emitter.shared.getSignatureType()};
  }

  // Attributes always resolve to their known value.
  if (auto param = decl->getParamDecl())
    return {MValue(ParamDeclRefAttr::get(param.getName(), param.getType())),
            decl->getResolvedType()};

  emitter.emitError(getLoc(), "use of declaration \"")
      << spelling << "\" as a value isn't supported yet";
  return {};
}

FullType DeclRefNode::emitType(ExprEmitter &emitter) const {
  auto *context = emitter.getContext();

  // Lookup the identifier.
  ASTDecl *decl = emitter.lookupDecl(spelling, getLoc(), emitter.declScope,
                                     "unknown type name");
  if (!decl)
    return {};
  auto typeDecl = dyn_cast<LITStructDeclOp>(*decl);
  if (!typeDecl) {
    if (decl->isMagic()) {
      Type mlirType;
      switch (decl->magicKind) {
      case MagicDeclKind::kNormal:
        llvm_unreachable("not a magic declaration?");
      case MagicDeclKind::kIndexType:
        // TODO(types): This is a hack to unblock tests in the interim.
        mlirType = IndexType::get(context);
        break;
      case MagicDeclKind::kNoneType:
        mlirType = KGEN::NoneType::get(context);
        break;
      case MagicDeclKind::kTypeCheckErrorType:
        mlirType = TypeCheckErrorType::get(context);
        break;
      case MagicDeclKind::kPointerType:
      case MagicDeclKind::kSignatureType:
        emitter.emitError(getLoc(),
                          "TODO: Cannot emit this until it is parameterized");
        return {};
      }
      return {mlirType, decl->getResolvedType()};
    }

    emitter.emitError(getLoc(), "'" + spelling + "' names a value, not a type");
    return {};
  }

  auto numParams = typeDecl.getParamDecls().size();
  if (numParams != 0) {
    emitter.emitError(getLoc(), "'" + spelling + "' requires ")
        << numParams << " meta parameter" << plural(numParams);
    return {};
  }

  return {RefType::get(FlatSymbolRefAttr::get(typeDecl.getNameAttr())),
          decl->getResolvedType()};
}

ASTTypeAnd<AnyValue> AttributeRefNode::emitIR(ExprEmitter &emitter,
                                              FullType contextualType) const {
  auto baseVal = base->emitIR(emitter);

  if (LValue baseLV = baseVal.ir.getIfLValue()) {
    if (!emitter.builder) {
      emitter.emitError(getLoc(),
                        "TODO: cannot call function in parameter context");
      return {};
    }

    ASTDecl *typeDecl = baseVal.type.getDecl();
    auto typeParams = baseVal.type.getParamValues();
    if (!typeParams.empty()) {
      emitter.emitError(getLoc(), "TODO: Cannot handle parameterized types ")
          << baseVal.type;
      return {};
    }

    if (!isa<LITStructDeclOp>(*typeDecl)) {
      emitter.emitError(getLoc(), "cannot access fields in type ")
          << baseVal.type;
      return {};
    }

    // Find the field.
    ASTDecl *fieldDecl = emitter.lookupDecl(attrSpelling, getLoc(), *typeDecl,
                                            "object has no attribute");
    if (!fieldDecl)
      return {};

    // TODO: Support method references some day.
    auto varOp = dyn_cast_or_null<VarDeclOp>(fieldDecl->getOperation());
    if (!varOp) {
      emitter.emitError(getLoc(), "'" + attrSpelling + "' is not a field");
      return {};
    }

    // TODO(Issue #4321): Perform parameter substitution
    Value resultGEP = emitter.builder->create<LITStructGEPOp>(
        emitter.translateLocation(getLoc()), varOp.getType(),
        varOp.getNameAttr(), baseLV);
    return {LValue(resultGEP), fieldDecl->getResolvedType()};
  }

  // TODO: Handle parameter member references.
  emitter.emitError(getLoc(), "cannot emit members of rvalues yet");
  return {};
}

FullType AttributeRefNode::emitType(ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return {};
}

ASTTypeAnd<AnyValue> CallNode::emitIR(ExprEmitter &emitter,
                                      FullType contextualType) const {
  auto calleeVal = emitter.emitRValue(callee).ir;
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
    if (!argVal.ir)
      return {};

    if (argVal.ir.getType() != expectedType) {
      // TODO: Handle implicit conversions.
      emitter.emitError(arg->getLoc(), "value of type ")
          << argVal.type
          << " cannot be converted to expected type "
          // TODO: Print pretty expected type.
          << expectedType;
      return {};
    }
    valueArguments.push_back(argVal.ir);
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
  // FIXME: This is a completely wrong result type from the call!
  return {DRValue(call.getResult(0)), ASTType()};
}

FullType CallNode::emitType(ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return {};
}

ASTTypeAnd<AnyValue> SubscriptNode::emitIR(ExprEmitter &emitter,
                                           FullType contextualType) const {
  // Subscripting a generic function binds the parameter expressions.
  auto subValue = base->emitIR(emitter);
  if (!subValue)
    return {};

  // If we have a value of signature type, we can bind parameters to it.
  if (auto signature = dyn_cast<SignatureType>(subValue.ir.getType())) {
    size_t numParams = signature.getInputParams().size();
    if (numParams != indices.size()) {
      emitter.emitError(getLoc(), "signature expects ")
          << numParams << " parameter value" << plural(numParams);
      return {};
    }

    auto declParam = subValue.ir.getIfMValue();
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
      if (!val.ir)
        return {};

      // Check the type matches what is expected.
      // TODO: Do implicit conversions.
      // TODO: Handle signatures like (T, scalar<T>) where early bound
      // parameters changes the types of later ones.
      if (val.ir.getType() != decl.getType()) {
        emitter.emitError(idx->getLoc(), "index has type ")
            // TODO: Print pretty decl type.
            << val.type << " but declaration expects " << decl.getType();
        return {};
      }
      bindOperands.push_back(val.ir);
    }
    // Okay, everything checks out, form the bind operation.
    return {MValue(ParamOperatorAttr::get(POC::BindSignature, bindOperands)),
            // TODO: Correct signature type.
            emitter.shared.getSignatureType()};
  }

  // Emit each of the index values.
  SmallVector<RValue> indexValues;
  for (ExprNode *index : indices) {
    indexValues.push_back(emitter.emitRValue(index).ir);
    if (!indexValues.back())
      return {};
  }

  emitter.emitError(getLoc(), "TODO: Subscript irgen not implemented yet ")
      << subValue.type;
  return {};
}

FullType SubscriptNode::emitType(ExprEmitter &emitter) const {
  // KGEN doesn't support unbound parametric types (e.g. "SIMD" with unbound
  // size/dtype) as a stand-alone type, so we handle name resolution here.
  auto baseDRE = dyn_cast<DeclRefNode>(base);
  if (!baseDRE) {
    auto baseType = base->emitType(emitter);
    if (baseType.first)
      emitter.emitError(getLoc(), "unknown parameterized type ")
          << baseType.second;
    return {};
  }

  // Lookup the identifier.
  ASTDecl *decl = emitter.lookupDecl(baseDRE->spelling, getLoc(),
                                     emitter.declScope, "unknown type name");
  if (!decl)
    return {};

  auto structOp = dyn_cast<LITStructDeclOp>(*decl);
  if (!structOp) {
    emitter.emitError(getLoc(),
                      "'" + baseDRE->spelling + "' names a value, not a type");
    return {};
  }

  auto numParams = structOp.getParamDecls().size();
  if (numParams != indices.size()) {
    emitter.emitError(getLoc(), "'" + baseDRE->spelling + "' requires ")
        << numParams << " meta parameter" << plural(numParams) << " but "
        << indices.size() << " were specified";
    return {};
  }

  // Emit each of the indices as parameter expressions.
  SmallVector<ParamBindAttr> exprs;
  for (auto [indexExpr, decl] : llvm::zip(indices, structOp.getParamDecls())) {
    // TODO: Slice syntax is the obvious way to support named parameter
    // arguments.
    auto indexVal = emitter.emitMValue(
        indexExpr, "type parameters may not be a run-time value");
    if (!indexVal.ir)
      return {};

    // TODO: Support conversions.
    if (indexVal.ir.getType() != decl.getType()) {
      emitter.emitError(indexExpr->getLoc(), "parameter of type ")
          << indexVal.type
          << " cannot be converted to expected type "
          // TODO: Pretty type.
          << decl.getType();
      return {};
    }

    exprs.push_back(ParamBindAttr::get(decl, indexVal.ir));
  }

  auto typeParams = ParamBindArrayAttr::get(emitter.getContext(), exprs);
  return {
      RefType::get(FlatSymbolRefAttr::get(structOp.getNameAttr()), typeParams),
      ASTType(decl, typeParams)};
}

ASTTypeAnd<AnyValue> ParenExprNode::emitIR(ExprEmitter &emitter,
                                           FullType contextualType) const {
  return subExpr->emitIR(emitter, contextualType);
}

FullType ParenExprNode::emitType(ExprEmitter &emitter) const {
  return subExpr->emitType(emitter);
}

ASTTypeAnd<AnyValue> BinOpNode::emitIR(ExprEmitter &emitter,
                                       FullType contextualType) const {
  auto lhsRep = emitter.emitRValue(lhs);
  auto rhsRep = emitter.emitRValue(rhs);
  if (!lhsRep.ir || !rhsRep.ir)
    return {};
  auto lhsType = lhsRep.ir.getType(), rhsType = rhsRep.ir.getType();
  if (lhsType != rhsType || !lhsType.isIndex()) {
    emitter.emitError(getLoc(),
                      "binary operator with interesting types not implemented");
    return {};
  }

  // If these are both parameter values, we can fold them using parameter
  // expressions.
  if (auto lhsParam = lhsRep.ir.getIfMValue()) {
    if (auto rhsParam = rhsRep.ir.getIfMValue()) {
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
      return {MValue(ParamOperatorAttr::get(opcode, lhsParam, rhsParam)),
              emitter.shared.getIndexType()};
    }
  }

  assert(emitter.builder && "cannot have dynamic values without a builder");

  auto lhsVal = emitter.emitDRValue(lhsRep, lhs->getLoc()).ir;
  auto rhsVal = emitter.emitDRValue(rhsRep, rhs->getLoc()).ir;
  auto loc = emitter.translateLocation(getLoc());

  // TODO: implement properly these operations once we have a real type system
  //       also, logical operators should implement short circuiting of the
  //       operands.
  Value result;
  switch (kind) {
  default:
    llvm_unreachable("unknown binary operator");
  case kAdd:
    result = emitter.builder->create<mlir::index::AddOp>(loc, lhsVal, rhsVal);
    break;
  case kSub:
    result = emitter.builder->create<mlir::index::SubOp>(loc, lhsVal, rhsVal);
    break;
  case kBoolAnd:
  case kBitwiseAnd:
    result = emitter.builder->create<mlir::index::AddOp>(loc, lhsVal, rhsVal);
    break;
  case kBoolOr:
  case kBitwiseOr:
    result = emitter.builder->create<mlir::index::SubOp>(loc, lhsVal, rhsVal);
    break;
  case kBitwiseXor:
    result = emitter.builder->create<mlir::index::DivSOp>(loc, lhsVal, rhsVal);
    break;
  case kMul:
  case kMatrixMul:
    result = emitter.builder->create<mlir::index::MulOp>(loc, lhsVal, rhsVal);
    break;
  case kDiv:
  case kFloorDiv:
    // TODO(types): kDiv should be floating point division
    result = emitter.builder->create<mlir::index::DivSOp>(loc, lhsVal, rhsVal);
    break;
  case kCmpEqual:
    result = emitter.builder->create<mlir::index::RemUOp>(loc, lhsVal, rhsVal);
    break;
  case kModulo:
    result = emitter.builder->create<mlir::index::RemSOp>(loc, lhsVal, rhsVal);
    break;
  case kExp:
    // TODO(types): this should be an exponentiation op
    // eventually we should call object.__pow__(self, other[, modulo])
    result = emitter.builder->create<mlir::index::RemSOp>(loc, lhsVal, rhsVal);
    break;
  }

  return {DRValue(result), emitter.shared.getIndexType()};
}

FullType BinOpNode::emitType(ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return {};
}

ASTTypeAnd<AnyValue> UnaryOpNode::emitIR(ExprEmitter &emitter,
                                         FullType contextualType) const {
  auto exprRep = emitter.emitRValue(subExpr);
  if (!exprRep)
    return {};
  auto exprType = exprRep.ir.getType();
  if (!exprType.isIndex()) {
    emitter.emitError(getLoc(),
                      "unary operator with interesting types not implemented");
    return {};
  }

  assert(emitter.builder && "cannot have dynamic values without a builder");

  auto exprVal = emitter.emitDRValue(exprRep, subExpr->getLoc()).ir;
  auto loc = emitter.translateLocation(getLoc());
  DRValue result;
  switch (kind) {
  default:
    emitter.emitError(getLoc(), "TODO: cannot emit this operator yet");
    return {};
  case kUnaryPlus: {
    // TODO:  this should eventually implement a call to object.__pos__(self)
    auto zero = emitter.builder->create<mlir::index::ConstantOp>(loc, 0);
    result = emitter.builder->create<mlir::index::AddOp>(loc, zero, exprVal);
    break;
  }
  case kBoolNot:
  case kUnaryMinus: {
    // TODO:  this should eventually implement a call to object.__neg__(self)
    auto zero = emitter.builder->create<mlir::index::ConstantOp>(loc, 0);
    result = emitter.builder->create<mlir::index::SubOp>(loc, zero, exprVal);
    break;
  }
  }
  return {result, emitter.shared.getIndexType()};
}

FullType UnaryOpNode::emitType(ExprEmitter &emitter) const {
  auto eltType = subExpr->emitType(emitter);
  if (!eltType.first)
    return {};

  // FIXME: This should be a declared type in the standard library parameterized
  // by an element type.
  if (kind == kUnaryAmp)
    return {POP::PointerType::get(eltType.first),
            emitter.shared.getPointerType()};

  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return {};
}

ASTTypeAnd<AnyValue> TernaryOpNode::emitIR(ExprEmitter &emitter,
                                           FullType contextualType) const {
  Value cond = emitter.emitDRValue(condExpr).ir;
  if (!cond)
    return {};

  // TODO(types): we only support 'index' values as a hack right now.
  if (!cond.getType().isIndex()) {
    emitter.emitError(condExpr->getLoc(), "value of type ")
        << cond.getType() << " isn't convertible to Bool";
    return {};
  }
  // TODO(types)
  Type resType = mlir::IndexType::get(emitter.getContext());
  Location ifLoc = emitter.translateLocation(getLoc());
  auto one = emitter.builder->create<mlir::index::ConstantOp>(cond.getLoc(), 1);
  Value condValue = emitter.builder->create<mlir::index::CmpOp>(
      cond.getLoc(), mlir::index::IndexCmpPredicate::EQ, cond, one);
  auto ifOp = emitter.builder->create<scf::IfOp>(ifLoc, TypeRange{resType},
                                                 condValue, /*withElse=*/true);
  emitter.builder = ifOp.getThenBodyBuilder();
  ASTTypeAnd<DRValue> trueVal = emitter.emitDRValue(trueExpr);
  if (!trueVal.ir)
    return {};
  emitter.builder->create<scf::YieldOp>(ifLoc, trueVal.ir);
  emitter.builder = ifOp.getElseBodyBuilder();
  ASTTypeAnd<DRValue> falseVal = emitter.emitDRValue(falseExpr);
  if (!falseVal.ir)
    return {};
  emitter.builder->create<scf::YieldOp>(ifLoc, falseVal.ir);
  emitter.builder->setInsertionPointAfter(ifOp);
  return {(DRValue)ifOp.getResult(0), emitter.shared.getIndexType()};
}

std::pair<Type, ASTType> TernaryOpNode::emitType(ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "cannot emit this expression as a type");
  return {};
}
