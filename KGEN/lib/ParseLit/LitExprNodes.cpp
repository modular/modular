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
#include "IRValues.h"
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
                                           ASTType contextualType,
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
ASTType ExprEmitter::emitType(const ExprNode *node) {
  auto value = emitMValue(node, "expected a type");
  if (!value)
    return shared.getTypeCheckErrorType();

  // If this emitted a type, we can lower it.
  if (auto astType = value.ir.getIfMTValue())
    return astType;

  // If we emitted a NoneAttr then convert it to a NoneType.  This is a special
  // case because "None" is both a value and a type, and defaults to a value.
  if (isa<KGEN::NoneAttr>(value.ir.getIfMAValue().get()))
    return shared.getNoneType();

  emitError(node->getLoc(), "expected a type, not a value");
  return shared.getTypeCheckErrorType();
}

/// Perform a name lookup in the specified scope and return the named
/// declaration.  This emits an error and returns null on error.
ASTDecl *
ExprEmitter::lookupDecl(StringRef name, SMLoc loc, ASTDecl &scope,
                        std::function<void(InFlightDiagnostic)> errorFn,
                        ASTType implicitDeclType) {

  // Look up the name.
  auto nameAttr = StringAttr::get(getContext(), name);
  ASTDecl *lookupResult = scope.lookup(nameAttr);

  // Handle the case where lookup fails.
  if (!lookupResult) {
    // If there is a contextual type available then this is an implicit variable
    // definition, otherwise it is an error.  There will never be a contextual
    // type in a `fn`, only a `def`.
    if (!implicitDeclType || !varDeclCursor) {
      errorFn(emitError(loc, ""));
      return nullptr;
    }

    // Otherwise, introduce a new lit.var.decl node whose type matches the
    // implicitDeclType.
    //
    // TODO(autopromotions): turn infinite integers into concrete ones as
    // needed.
    Type declIRType = shared.getMLIRType(implicitDeclType, loc);
    declIRType = POP::PointerType::get(declIRType);

    // Use this builder to place any VarDeclOps. In Python there is only one
    // scope per function and all variables belong to that scope, so builders
    // should reflect that.
    auto varDecl =
        OpBuilder(varDeclCursor)
            .create<VarDeclOp>(translateLocation(loc), declIRType, nameAttr);
    lookupResult = &shared.declResolver->addFullyResolvedDecl(
        varDecl, nameAttr, implicitDeclType, &scope);
  }

  // If the lookup succeeded, make sure the signature for the referenced decl is
  // understood.
  auto resolveResult = shared.declResolver->resolve(
      *lookupResult, DeclResolvedness::signatureResolved, loc);

  // If the decl was erroneous somehow, then don't form a reference to it, the
  // error has already been diagnosed.
  if (failed(resolveResult))
    return nullptr;
  return lookupResult;
}

//===----------------------------------------------------------------------===//
// IR Emission helpers
//===----------------------------------------------------------------------===//

static ASTTypeAnd<AnyValue>
emitFunctionCall(const CallNode &call, ASTTypeAnd<RValue> calleeValAndType,
                 ExprEmitter &emitter) {
  if (!calleeValAndType)
    return {};

  auto emitError = [&](const Twine &message) {
    return emitter.emitError(call.getLoc(), message);
  };

  RValue calleeVal = calleeValAndType.ir;
  ASTType calleeType = calleeValAndType.type;

  auto calleeAnyType = calleeVal.getType(emitter.getContext());
  SignatureType calleeIRType = dyn_cast<SignatureType>(calleeAnyType);
  if (!calleeIRType) {
    emitError("invalid function to call");
    return {};
  }

  // The ASTType of calleeVal must be a magic function type, for the IR to
  // have signature type.  We cannot have error types or anything else here.
  // TODO: Switch to key off the AST type when it carries everything we need.
  assert(calleeType.getDecl().magicKind == MagicDeclKind::kFunctionType);
  assert(calleeType.getParamValues().size() == 1 &&
         "FunctionType should have one (result) parameter");
  auto resultASTTypeVal = calleeType.getParamValues()[0].second;

  auto resultASTType = resultASTTypeVal.getIfMTValue();
  if (!resultASTType) {
    // TODO: We have no way to represent a symbolic value of ASTType.
    emitError("unable to call function value with parametric result type");
    return {};
  }

  // If there are any unbound parameters then we cannot call it.
  // TODO: infer the parameters from the types of the operands.
  if (!calleeIRType.getInputParams().empty()) {
    emitError("unable to call parameterized value that expects ")
        << calleeIRType.getInputParams().size() << " bound parameters";
    return {};
  }

  assert(calleeIRType.getResultParamTypes().empty() &&
         "TODO: meta results not implemented yet");

  size_t numArgs = calleeIRType.getValues().getNumInputs();
  if (numArgs != call.args.size()) {
    emitError("callee expects ") << numArgs << " argument" << plural(numArgs);
    return {};
  }

  // Emit all the arguments.
  SmallVector<Value> valueArguments;
  for (auto [arg, expectedType] :
       llvm::zip(call.args, calleeIRType.getValues().getInputs())) {
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

  if (!emitter.builder) {
    emitError("TODO: cannot call function in parameter context");
    return {};
  }

  // If this is a call to something representable as an attribute, we can use
  // a kgen.call_param.
  Value resultVal;
  auto loc = emitter.translateLocation(call.getLoc());
  auto resultTypes = calleeIRType.getValues().getResults();
  if (auto calleeParam = calleeVal.getIfMAValue()) {
    resultVal =
        emitter.builder
            ->create<CallParamOp>(loc, resultTypes, calleeParam,
                                  /*inputParams*/ ArrayRef<ParamBindAttr>(),
                                  /*resultParams*/ ArrayRef<ParamDeclAttr>(),
                                  /*operands*/ valueArguments)
            .getResult(0);
  } else {
    // Otherwise emit calls to SSA values with call_indirect.
    auto calleeDRVal =
        emitter.emitDRValue({AnyValue(calleeVal), calleeType}, call.getLoc());
    if (!calleeDRVal)
      return {};
    resultVal = emitter.builder
                    ->create<CallIndirectOp>(loc, resultTypes, calleeDRVal.ir,
                                             /*operands*/ valueArguments)
                    .getResult(0);
  }

  // Value returning call returns its result.
  return {DRValue(resultVal), resultASTType};
}

/// Get a symbol for a direct reference to the specified function in its
/// enclosing context.  This does not bind any values to arguments.
static ASTTypeAnd<MValue> emitFuncReference(LITFuncOp fnOp, ASTDecl &decl,
                                            ExprEmitter &emitter) {
  // Generate a nested symbol ref if we are a method in a struct.
  SymbolRefAttr symbolRef = FlatSymbolRefAttr::get(fnOp.getNameAttr());
  if (auto parentStruct = dyn_cast<LITStructDeclOp>(*decl.getParentDecl()))
    symbolRef = SymbolRefAttr::get(parentStruct.getNameAttr(),
                                   cast<FlatSymbolRefAttr>(symbolRef));
  auto fnAttr = SymbolConstantAttr::get(symbolRef, fnOp.getSignature());

  // TODO: Correct argument/parameter type.
  ASTType astType = emitter.shared.getFunctionType(decl.getResolvedType());
  return {MValue(fnAttr), astType};
}

/// Given an ASTType 'containingType', look up a named member of it and return
/// the reference to its symbol as an RValue.
/// TODO: This should take the parameters on the enclosing decl being referenced
/// to support things like SomeType[42].member()
static ASTTypeAnd<AnyValue>
emitDeclMemberReference(ASTDecl &container, StringRef memberName, SMLoc loc,
                        ExprEmitter &emitter, ASTType implicitDeclType = {}) {
  ASTDecl *decl = emitter.lookupDecl(
      memberName, loc, container,
      [&](InFlightDiagnostic diag) {
        if (auto structDecl = dyn_cast<LITStructDeclOp>(container)) {
          diag << structDecl.getName() << " has no '" << memberName
               << "' member";
        } else {
          diag << "use of unknown declaration \"" << memberName << '"';
        }
      },
      implicitDeclType);
  if (!decl)
    return {};

  // Variable references resolve to an lvalue addressing the variable.
  if (auto var = dyn_cast<VarDeclOp>(*decl))
    return {LValue(var.getResult()), decl->getResolvedType()};

  // Functions form an address.
  if (auto fnOp = dyn_cast<LITFuncOp>(*decl)) {
    ASTTypeAnd<MValue> mv = emitFuncReference(fnOp, *decl, emitter);
    return {mv.ir, mv.type};
  }

  // RValue's and LValues always resolve to their known value.
  if (auto rvalue = decl->getIfRValue())
    return {rvalue, decl->getResolvedType()};
  if (auto lvalue = decl->getIfLValue())
    return {lvalue, decl->getResolvedType()};

  if (isa<LITStructDeclOp>(*decl) || decl->isMagic()) {
    auto astType = emitter.shared.getASTType(*decl, {});
    return {MValue(astType), emitter.shared.getTypeType()};
  }

  emitter.emitError(loc, "use of declaration \"")
      << memberName << "\" as a value isn't supported yet";
  return {};
}

//===----------------------------------------------------------------------===//
// ExprNode implementations
//===----------------------------------------------------------------------===//

ExprNode::~ExprNode() { llvm_unreachable("never called"); }

ASTTypeAnd<AnyValue> IntLiteralNode::emitIR(ExprEmitter &emitter,
                                            ASTType contextualType) const {
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
                                              ASTType contextualType) const {
  // TODO: this assumes float literal are always doubles
  APFloat value = LitLexer::getFloatLiteralValue(spelling);
  auto attr = FloatAttr::get(FloatType::getF64(emitter.getContext()),
                             APFloat(value.convertToDouble()));
  return {AnyValue(attr), emitter.shared.getFloatLiteralType()};
}

ASTTypeAnd<AnyValue> StringLiteralNode::emitIR(ExprEmitter &emitter,
                                               ASTType contextualType) const {
  std::string value = LitLexer::getStringLiteralValue(spelling);
  return {AnyValue(StringAttr::get(emitter.getContext(), value)),
          emitter.shared.getStringLiteralType()};
}

ASTTypeAnd<AnyValue> NoneLiteralNode::emitIR(ExprEmitter &emitter,
                                             ASTType contextualType) const {
  auto noneMLIRType = KGEN::NoneType::get(emitter.getContext());
  return {MAValue(NoneAttr::get(emitter.getContext(), noneMLIRType)),
          emitter.shared.getNoneType()};
}

ASTTypeAnd<AnyValue> DeclRefNode::emitIR(ExprEmitter &emitter,
                                         ASTType contextualType) const {
  return emitDeclMemberReference(emitter.declScope, spelling, getLoc(), emitter,
                                 contextualType);
}

ASTTypeAnd<AnyValue> AttributeRefNode::emitIR(ExprEmitter &emitter,
                                              ASTType contextualType) const {
  auto baseVal = base->emitIR(emitter);
  if (!baseVal)
    return {};

  // Handle member references on types.
  if (ASTType baseType = baseVal.ir.getIfMTValue()) {
    auto rValueAnd = emitDeclMemberReference(baseType.getDecl(), attrSpelling,
                                             getLoc(), emitter);
    return {rValueAnd.ir, rValueAnd.type};
  }

  // Otherwise, it must be an access to a field of a value.
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

  // Find the member being accessed.
  ASTDecl *memberDecl = emitter.lookupDecl(
      attrSpelling, getLoc(), typeDecl, [&](InFlightDiagnostic diag) {
        diag << "object has no attribute '" << attrSpelling << "'";
      });
  if (!memberDecl)
    return {};

  // If the field is a variable, emit a reference to it.
  if (auto varOp = dyn_cast<VarDeclOp>(*memberDecl)) {
    auto varASTType = memberDecl->getResolvedType();

    // If the base is an lvalue, then we can return an lvalue to the field.
    if (LValue baseLV = baseVal.ir.getIfLValue()) {
      if (!emitter.builder) {
        emitter.emitError(
            getLoc(), "TODO: cannot access lvalue member in parameter context");
        return {};
      }
      // TODO(Issue #4321): Perform parameter substitution
      Value resultGEP = emitter.builder->create<LITStructGEPOp>(
          emitter.translateLocation(getLoc()), varOp.getType(),
          varOp.getNameAttr(), baseLV);
      return {LValue(resultGEP), varASTType};
    }

    // Otherwise, it must be an rvalue.
    // TODO: If this is an MValue, emit as a parameter field access, this would
    // enable `size.value` in things like:
    //
    // fn f[size: Int](a: SomeType[size.value])
    ASTTypeAnd<DRValue> baseRV = emitter.emitDRValue(baseVal, getLoc());
    if (!baseRV)
      return {};

    if (!emitter.builder) {
      emitter.emitError(getLoc(),
                        "TODO: cannot access member in parameter context");
      return {};
    }

    // TODO(Issue #4321): Perform parameter substitution
    Value resultVal = emitter.builder->create<LITStructExtractOp>(
        emitter.translateLocation(getLoc()),
        emitter.shared.getMLIRType(varASTType, getLoc()), varOp.getNameAttr(),
        baseRV.ir);
    return {DRValue(resultVal), varASTType};
  }

  // Handle method references.
  if (auto fnOp = dyn_cast<LITFuncOp>(*memberDecl)) {
    // Get a symbol for the underlying function.
    ASTTypeAnd<MValue> fnRef = emitFuncReference(fnOp, *memberDecl, emitter);
    assert(fnRef.ir && "always succeeds");

    // If the callee is a static method, we can directly reference it without
    // binding a self parameter.
    if (fnOp.getIsStatic())
      return {fnRef.ir, fnRef.type};

    // If this is an instance method, we partially apply the base value to the
    // function as the first self argument.  Handle the case of a mutating
    // method first since that requires an lvalue.
    // TODO: Move this to ASTType checking when it can represent parameter
    // types.
    auto symbolIRType =
        cast<SignatureType>(fnRef.ir.getType(emitter.getContext()));
    Type firstArgIRType = symbolIRType.getValues().getInputs()[0];
    Value firstArgValue;
    if (isa<POP::PointerType>(firstArgIRType)) {
      LValue baseLV = baseVal.ir.getIfLValue();
      if (!baseLV) {
        emitter.emitError(getLoc(),
                          "invalid use of mutating method on rvalue of type ")
            << baseVal.type;
        return {};
      }

      // TODO: Using partial application over an lvalue like this isn't
      // technically safe.  We need to extend the lifetime of the pointer
      // captured for as long as the partial application thunk is alive.  This
      // will require some sort of borrow model.  In practice, this will be fine
      // in the short term of Lit bringup because the thunk cannot be emitted
      // independently anyway, it must always be canonicalized into another
      // call.
      firstArgValue = baseLV;
    } else {
      // Otherwise we can have either an lvalue or rvalue, but we need to
      // convert to an rvalue if we have an lvalue.
      auto drValue = emitter.emitDRValue(baseVal, getLoc());
      if (!drValue)
        return {};
      firstArgValue = drValue.ir;
    }

    if (!emitter.builder) {
      emitter.emitError(getLoc(),
                        "TODO: cannot access method in parameter context");
      return {};
    }

    assert(firstArgIRType == firstArgValue.getType() &&
           "base types should always structurally line up");

    // PartialApply takes the callee as a Value.
    auto calleeDRVal =
        emitter.emitDRValue({AnyValue(fnRef.ir), fnRef.type}, getLoc());

    // Partial apply wants to know what operands to bind, we always bind the
    // first one.
    auto zeroAttr = emitter.builder->getAttr<mlir::DenseI64ArrayAttr>(0);

    // The result type will be a signature type with one fewer value argument.
    auto resultFnType = emitter.builder->getFunctionType(
        symbolIRType.getValues().getInputs().drop_front(),
        symbolIRType.getValues().getResults());
    auto resultSigType =
        SignatureType::get(symbolIRType.getInputParams(),
                           symbolIRType.getResultParamTypes(), resultFnType);

    // TODO(Issue #4321): Perform parameter substitution
    Value result = emitter.builder->create<PartialApplyOp>(
        emitter.translateLocation(getLoc()), resultSigType, calleeDRVal.ir,
        mlir::ValueRange(firstArgValue), zeroAttr);

    // TODO: We should have proper function argument types.
    return {DRValue(result), calleeDRVal.type};
  }

  // TODO: Handle parameter member references.
  emitter.emitError(getLoc(), "cannot emit members of ") << baseVal.type;
  return {};
}

/// Given a call of a type T value, lower it into a call of 'T.__new__'.
static ASTTypeAnd<AnyValue> emitInitializerCall(const CallNode &call,
                                                ASTType calledType,
                                                ExprEmitter &emitter) {
  // Ensure the type specified is fully resolved, so all its members are known.
  if (failed(emitter.shared.declResolver->resolve(
          calledType.getDecl(), DeclResolvedness::fullyResolved,
          call.getLoc())))
    return {};

  auto newMemberVal = emitDeclMemberReference(calledType.getDecl(), "__new__",
                                              call.getLoc(), emitter);
  return emitFunctionCall(call, emitter.emitRValue(newMemberVal, call.getLoc()),
                          emitter);
}

ASTTypeAnd<AnyValue> CallNode::emitIR(ExprEmitter &emitter,
                                      ASTType contextualType) const {
  auto calleeVal = emitter.emitRValue(callee);
  if (!calleeVal)
    return {};

  // Invoking a type is a call to an initialize for the type.
  if (ASTType calledType = calleeVal.ir.getIfMTValue())
    return emitInitializerCall(*this, calledType, emitter);

  // Otherwise, handle callable functions.
  if (isa<SignatureType>(calleeVal.ir.getType(emitter.getContext())))
    return emitFunctionCall(*this, calleeVal, emitter);

  emitter.emitError(getLoc(), "unable to call value of type ")
      << calleeVal.type;
  return {};
}

ASTTypeAnd<AnyValue> SubscriptNode::emitIR(ExprEmitter &emitter,
                                           ASTType contextualType) const {
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
                                           ASTType contextualType) const {
  return subExpr->emitIR(emitter, contextualType);
}

ASTTypeAnd<AnyValue> ListExprNode::emitIR(ExprEmitter &emitter,
                                          ASTType contextualType) const {
  // TODO: here we return the last expression, we should return a list object
  // instead.
  DRValue last;
  for (ExprNode *expr : exprs) {
    auto exprRep = emitter.emitRValue(expr);
    if (!exprRep)
      return {};

    // TODO(types): allow all types.
    if (exprRep.type.getDecl().magicKind != MagicDeclKind::kIndexType) {
      emitter.emitError(
          getLoc(), "List expression with interesting types not implemented");
      return {};
    }
    assert(emitter.builder && "cannot have dynamic values without a builder");
    last = emitter.emitDRValue(exprRep, expr->getLoc()).ir;
  }
  if (exprs.empty()) {
    auto loc = emitter.translateLocation(getLoc());
    last = DRValue(emitter.builder->create<mlir::index::ConstantOp>(loc, 0));
  }
  return {last, emitter.shared.getIndexType()};
}

ASTTypeAnd<AnyValue> BinOpNode::emitIR(ExprEmitter &emitter,
                                       ASTType contextualType) const {
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
                                         ASTType contextualType) const {
  auto exprRep = emitter.emitRValue(subExpr);
  if (!exprRep)
    return {};

  // If the sub-value is an ASTType, apply type sugar.
  if (auto astType = exprRep.ir.getIfMTValue()) {
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
                                           ASTType contextualType) const {
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
