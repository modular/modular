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
#include "ASTDecl.h"
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
using namespace M::KGEN;
using namespace M::KGEN::LIT;
namespace scf = mlir::scf;

static const char *plural(size_t value) { return value == 1 ? "" : "s"; }

//===----------------------------------------------------------------------===//
// IRValues Implementation Logic.
//===----------------------------------------------------------------------===//

using VariantStorage = PointerUnion<MAValue, ASTType, DRValue, LValue>;

static raw_ostream &printStorage(raw_ostream &os, VariantStorage storage,
                                 bool isDump = false) {
  if (storage.isNull()) {
    os << "<NULL IR Value>\n";
  } else if (auto val = dyn_cast<MAValue>(storage)) {
    if (isDump)
      os << "MA: ";
    os << val.get();
  } else if (auto val = dyn_cast<ASTType>(storage)) {
    if (isDump)
      os << "MT: ";
    os << val;
  } else if (auto val = dyn_cast<DRValue>(storage)) {
    if (isDump)
      os << "DR: ";
    os << val;
  } else if (auto val = dyn_cast<LValue>(storage)) {
    if (isDump)
      os << "LV: ";
    os << val;
  } else {
    os << "<UNKNOWN IRVALUE>";
  }
  return os;
}

raw_ostream &LIT::operator<<(raw_ostream &os, MValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &LIT::operator<<(raw_ostream &os, RValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &LIT::operator<<(raw_ostream &os, AnyValue value) {
  return printStorage(os, value.getStorage());
}

void MValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}
void RValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}
void AnyValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}

static std::string getStorageAsString(VariantStorage storage) {
  std::string result;
  llvm::raw_string_ostream os(result);
  printStorage(os, storage);
  return os.str();
}

mlir::Diagnostic &LIT::operator<<(mlir::Diagnostic &diag, MValue value) {
  return diag << getStorageAsString(value.getStorage());
}
mlir::Diagnostic &LIT::operator<<(mlir::Diagnostic &diag, RValue value) {
  return diag << getStorageAsString(value.getStorage());
}
mlir::Diagnostic &LIT::operator<<(mlir::Diagnostic &diag, AnyValue value) {
  return diag << getStorageAsString(value.getStorage());
}

static Type getTypeFrom(VariantStorage storage, MLIRContext *context) {
  if (storage.isNull())
    return Type();
  if (auto attr = dyn_cast<MAValue>(storage))
    return attr.get().getType();
  if (auto value = dyn_cast<DRValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<LValue>(storage))
    return value.getType();

  if (isa<ASTType>(storage))
    return MLIRTypeType::get(context);

  // TODO: Handle ASTType.
  llvm_unreachable("unhandled case ASTType");
}

Type MValue::getType(MLIRContext *context) const {
  return getTypeFrom(storage, context);
}
Type RValue::getType(MLIRContext *context) const {
  return getTypeFrom(storage, context);
}
Type AnyValue::getType(MLIRContext *context) const {
  return getTypeFrom(storage, context);
}

/// Lower this MValue to a TypedAttr.  If this contains an ASTType, it is
/// lowered to an MLIRType and wrapped in a ParameteredTypeConstantAttr.
TypedAttr MValue::lowerToAttribute(LitSharedState &shared, Location loc) const {
  assert(!isNull() && "Cannot emit null attribute");

  // If this is already an attribute, return it.
  if (auto attr = getIfMAValue())
    return attr;

  // If this is a type, convert it.
  auto astType = getIfMTValue();
  assert(astType && "Unknown MValue kind");
  return ParameterizedTypeConstantAttr::get(shared.getMLIRType(astType, loc));
}

/// Lower this MValue to a TypedAttr.  If this contains an ASTType, it is
/// lowered to an MLIRType and wrapped in a ParameteredTypeConstantAttr.
TypedAttr MValue::lowerToAttribute(LitSharedState &shared, SMLoc loc) const {
  assert(!isNull() && "Cannot emit null attribute");

  // If this is already an attribute, return it.
  if (auto attr = getIfMAValue())
    return attr;

  // If this is a type, convert it.
  auto astType = getIfMTValue();
  assert(astType && "Unknown MValue kind");
  return ParameterizedTypeConstantAttr::get(shared.getMLIRType(astType, loc));
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

  if (auto rvRep = rep.ir.getIfRValue())
    return {rvRep, rep.type};

  auto pointer = rep.ir.getIfLValue();
  assert(pointer);

  if (!builder) {
    emitError(loc, "context only permits a meta value, not a dynamic one");
    return {};
  }

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
  auto attr = rep.ir.getIfMAValue();
  assert(attr);

  auto location = translateLocation(loc);
  // Materialize index integer constants as a special case.
  if (auto intAttr = dyn_cast<IntegerAttr>(attr.get()))
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
  if (auto value = rep.ir.getIfMValue())
    return {value, rep.type};

  emitError(node->getLoc(), message);
  return {};
}

/// Emit the specified expression as an LValue which can be loaded and stored.
/// If contextualType is non-null, then an implicitly declared LValue will be
/// assigned that type.
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
  auto value = emitMValue(node, "expected a type");
  if (!value)
    return {TypeCheckErrorType::get(getContext()),
            shared.getTypeCheckErrorType()};

  // If this emitted a type, we can lower it.
  if (auto astType = value.ir.getIfMTValue()) {
    Type mlirType = shared.getMLIRType(astType, node->getLoc());
    return {mlirType, astType};
  }

  // If we emitted a NoneAttr then convert it to a NoneType.  This is a special
  // case because "None" is both a value and a type, and defaults to a value.
  if (isa<KGEN::NoneAttr>(value.ir.getIfMAValue().get()))
    return {KGEN::NoneType::get(getContext()), shared.getNoneType()};

  emitError(node->getLoc(), "expected a type, not a value");
  return {TypeCheckErrorType::get(getContext()),
          shared.getTypeCheckErrorType()};
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

  // TODO: Switch to builtin.IntegerLiteralType.
  return {AnyValue(attr), emitter.shared.getIndexType()};
}

ASTTypeAnd<AnyValue> FloatLiteralNode::emitIR(ExprEmitter &emitter,
                                              FullType contextualType) const {
  // TODO: this assumes float literal are always doubles
  APFloat value = LitLexer::getFloatLiteralValue(spelling);
  auto attr = FloatAttr::get(FloatType::getF64(emitter.getContext()),
                             APFloat(value.convertToDouble()));
  return {AnyValue(attr), emitter.shared.getFloatLiteralType()};
}

ASTTypeAnd<AnyValue> StringLiteralNode::emitIR(ExprEmitter &emitter,
                                               FullType contextualType) const {
  std::string value = LitLexer::getStringLiteralValue(spelling);
  return {AnyValue(StringAttr::get(emitter.getContext(), value)),
          emitter.shared.getStringLiteralType()};
}

ASTTypeAnd<AnyValue> NoneLiteralNode::emitIR(ExprEmitter &emitter,
                                             FullType contextualType) const {
  auto noneMLIRType = KGEN::NoneType::get(emitter.getContext());
  return {MAValue(NoneAttr::get(emitter.getContext(), noneMLIRType)),
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
            // TODO: Correct argument/parameter type.
            emitter.shared.getFunctionType(decl->getResolvedType())};
  }

  // Attributes always resolve to their known value.
  if (auto param = decl->getParamDecl())
    return {MValue(ParamDeclRefAttr::get(param.getName(), param.getType())),
            decl->getResolvedType()};

  if (isa<LITStructDeclOp>(*decl) || decl->isMagic()) {
    auto astType = emitter.shared.getASTType(*decl, {});
    return {MValue(astType), emitter.shared.getTypeType()};
  }

  emitter.emitError(getLoc(), "use of declaration \"")
      << spelling << "\" as a value isn't supported yet";
  return {};
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

    ASTDecl &typeDecl = baseVal.type.getDecl();
    auto typeParams = baseVal.type.getParamValues();
    if (!typeParams.empty()) {
      emitter.emitError(getLoc(), "TODO: Cannot handle parameterized types ")
          << baseVal.type;
      return {};
    }

    if (!isa<LITStructDeclOp>(typeDecl)) {
      emitter.emitError(getLoc(), "cannot access fields in type ")
          << baseVal.type;
      return {};
    }

    // Find the field.
    ASTDecl *fieldDecl = emitter.lookupDecl(attrSpelling, getLoc(), typeDecl,
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

ASTTypeAnd<AnyValue> CallNode::emitIR(ExprEmitter &emitter,
                                      FullType contextualType) const {
  auto calleeVal = emitter.emitRValue(callee);
  if (!calleeVal)
    return {};

  // The only callable thing we have right now are functions.
  // TODO: Support struct initialization.
  auto calleeAnyType = calleeVal.ir.getType(emitter.getContext());
  auto calleeType = dyn_cast<SignatureType>(calleeAnyType);
  if (!calleeType) {
    emitter.emitError(getLoc(), "unable to call value of type ")
        << calleeAnyType;
    return {};
  }

  // The ASTType of calleeVal must be a magic function type, for the IR to have
  // signature type.  We cannot have error types or anything else here.
  // TODO: Switch to key off the AST type when it carries everything we need.
  assert(calleeVal.type.getDecl().magicKind == MagicDeclKind::kFunctionType);
  assert(calleeVal.type.getParamValues().size() == 1 &&
         "FunctionType should have one (result) parameter");
  auto resultASTTypeVal = calleeVal.type.getParamValues()[0].second;

  auto resultASTType = resultASTTypeVal.getIfMTValue();
  if (!resultASTType) {
    // TODO: We have no way to represent a symbolic value of ASTType.  The best
    // we can do is
    emitter.emitError(
        getLoc(), "unable to call function value with parametric result type");
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

  auto calleeParam = calleeVal.ir.getIfMAValue();
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
  return {DRValue(call.getResult(0)), resultASTType};
}

ASTTypeAnd<AnyValue> SubscriptNode::emitIR(ExprEmitter &emitter,
                                           FullType contextualType) const {
  // Subscripting a generic function binds the parameter expressions.
  auto subValue = base->emitIR(emitter);
  if (!subValue)
    return {};

  // If the sub-value is an unbound ASTType, try binding things to it!
  if (auto astType = subValue.ir.getIfMTValue()) {
    // If already parameterized, give up.
    if (!astType.getParamValues().empty()) {
      emitter.emitError(
          getLoc(),
          "cannot apply more parameters to an already parameterized type ")
          << astType;
      return {};
    }

    auto structOp = dyn_cast<LITStructDeclOp>(astType.getDecl());
    if (!structOp) {
      emitter.emitError(getLoc(), "unknown parameterized type ") << astType;
      return {};
    }

    auto numParams = structOp.getParamDecls().size();
    if (numParams != indices.size()) {
      emitter.emitError(getLoc(), "")
          << astType << " requires " << numParams << " meta parameter"
          << plural(numParams) << " but " << indices.size()
          << " were specified";
      return {};
    }

    // Emit each of the indices as parameter expressions.
    SmallVector<LitSharedState::ParamBinding> paramBindings;
    for (auto [indexExpr, decl] :
         llvm::zip(indices, structOp.getParamDecls())) {
      // TODO: Slice syntax is the obvious way to support named parameter
      // arguments.
      auto indexVal = emitter.emitMValue(
          indexExpr, "type parameters may not be a run-time value");
      if (!indexVal.ir)
        return {};

      // TODO: Support conversions.
      if (indexVal.ir.getType(emitter.getContext()) != decl.getType()) {
        emitter.emitError(indexExpr->getLoc(), "parameter of type ")
            << indexVal.type
            << " cannot be converted to expected type "
            // TODO: Pretty type.
            << decl.getType();
        return {};
      }
      paramBindings.push_back({decl, indexVal.ir});
    }

    // Ok, we succeeded at reparameterizing the type.
    auto result = emitter.shared.getASTType(astType.getDecl(), paramBindings);
    return {MValue(result), emitter.shared.getTypeType()};
  }

  // If we have a value of signature type, we can bind parameters to it.
  // TODO(SignatureASTTypes): Use subValue.type when we have signatures.
  if (auto signature =
          dyn_cast<SignatureType>(subValue.ir.getType(emitter.getContext()))) {

    // The ASTType of subValue must be a magic function type, for the IR to have
    // signature type.  We cannot have error types or anything else here.
    // TODO: Switch to key off the AST type when it carries everything we need.
    assert(subValue.type.getDecl().magicKind == MagicDeclKind::kFunctionType);
    assert(subValue.type.getParamValues().size() == 1 &&
           "FunctionType should have one (result) parameter");
    auto resultASTType = subValue.type.getParamValues()[0].second;

    size_t numParams = signature.getInputParams().size();
    if (numParams != indices.size()) {
      emitter.emitError(getLoc(), "signature expects ")
          << numParams << " parameter value" << plural(numParams);
      return {};
    }

    auto declParam = subValue.ir.getIfMAValue();
    if (!declParam) {
      emitter.emitError(getLoc(), "cannot parameterize dynamic value");
      return {};
    }

    // Emit each index as a meta value and type check it.
    SmallVector<TypedAttr> bindOperands;
    bindOperands.push_back(declParam);
    for (auto [idx, decl] : llvm::zip(indices, signature.getInputParams())) {
      auto val = emitter.emitMAValue(
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
            emitter.shared.getFunctionType(resultASTType)};
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

ASTTypeAnd<AnyValue> ParenExprNode::emitIR(ExprEmitter &emitter,
                                           FullType contextualType) const {
  return subExpr->emitIR(emitter, contextualType);
}

ASTTypeAnd<AnyValue> BinOpNode::emitIR(ExprEmitter &emitter,
                                       FullType contextualType) const {
  auto lhsRep = emitter.emitRValue(lhs);
  auto rhsRep = emitter.emitRValue(rhs);
  if (!lhsRep.ir || !rhsRep.ir)
    return {};
  if (lhsRep.type.getDecl().magicKind != MagicDeclKind::kIndexType ||
      rhsRep.type.getDecl().magicKind != MagicDeclKind::kIndexType) {
    emitter.emitError(getLoc(),
                      "binary operator with interesting types not implemented");
    return {};
  }

  // If these are both parameter values, we can fold them using parameter
  // expressions.
  if (auto lhsParam = lhsRep.ir.getIfMAValue()) {
    if (auto rhsParam = rhsRep.ir.getIfMAValue()) {
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

ASTTypeAnd<AnyValue> UnaryOpNode::emitIR(ExprEmitter &emitter,
                                         FullType contextualType) const {
  auto exprRep = emitter.emitRValue(subExpr);
  if (!exprRep)
    return {};

  // If the sub-value is an ASTType, apply type sugar.
  if (auto astType = exprRep.ir.getIfMTValue()) {
    // TODO: Shouldn't be MTValue check: use "getMValue()" + check for typetype.
    if (kind == kUnaryAmp)
      return {MValue(emitter.shared.getPointerType(astType)), exprRep.type};

    emitter.emitError(getLoc(), "cannot emit this expression as a type");
    return {};
  }

  // Otherwise we just have our hard coded expression stuff going on.
  if (exprRep.type.getDecl().magicKind != MagicDeclKind::kIndexType) {
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
