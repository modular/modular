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
#include "LitExprs.h"
#include "LitSharedState.h"
#include "SpecialFunctions.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;
namespace scf = mlir::scf;

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

/// When a lookup in __mlir_type fails for a named field, this method tries to
/// resolve it.  On success, it lazily creates a resolved declaration.  On
/// failure, it bails out.
static ASTDecl *synthesizeMLIRTypeDeclEntry(StringRef name, SMLoc loc,
                                            ASTDecl &scope,
                                            LitSharedState &shared) {
  Type result;
  {
    // Capture errors thrown by parseType and ignore them.
    // FIXME: This doesn't silence errors!
    mlir::ScopedDiagnosticHandler handler(shared.getContext(),
                                          [](Diagnostic &diag) {});

    // FIXME(https://github.com/llvm/llvm-project/issues/58964)
    // Copy the string into a temporary smallvector so we can make sure it is
    // nul terminated for the MLIR asmparser.
    SmallString<64> tmpBuf(name.begin(), name.end());
    tmpBuf.push_back(0);
    result =
        mlir::parseType(StringRef(tmpBuf).drop_back(), shared.getContext());
  }
  if (!result) {
    shared.emitError(loc, "unknown MLIR type: ") << name;
    return nullptr;
  }

  return &shared.declResolver->addFullyResolvedDecl(
      result, StringAttr::get(shared.getContext(), name),
      shared.translateLocation(loc), shared.getTypeType(), &scope);
}

/// When a lookup in __mlir_op fails for a named field, this method tries to
/// resolve it.  On success, it lazily creates a resolved declaration.  On
/// failure, it bails out.
static ASTDecl *synthesizeMLIROpDeclEntry(StringRef name, SMLoc loc,
                                          ASTDecl &scope,
                                          ExprEmitter &emitter) {
  auto &shared = emitter.shared;
  auto nameStr = StringAttr::get(shared.getContext(), name);

  auto result = UnboundMLIROperationAttr::get(emitter.getContext(),
                                              nameStr.getType(), nameStr);
  return &shared.declResolver->addFullyResolvedDecl(
      MAValue(result), nameStr, emitter.translateLocation(loc),
      shared.getUnboundMLIROperatorType(), &scope);
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
    // If this is a lookup in __mlir_type or __mlir_op, then try to lazily
    // synthesize the element in question.
    if (scope.magicKind == MagicDeclKind::k__mlir_type)
      return synthesizeMLIRTypeDeclEntry(name, loc, scope, shared);
    if (scope.magicKind == MagicDeclKind::k__mlir_op)
      return synthesizeMLIROpDeclEntry(name, loc, scope, *this);

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

using ArgumentValueType = std::pair<ASTTypeAnd<AnyValue>, SMLoc>;

/// Emit a function call to the specified callee with the specified operand
/// values.
static ASTTypeAnd<AnyValue>
emitFunctionCall(ASTTypeAnd<RValue> calleeValAndType,
                 ArrayRef<ArgumentValueType> operands, SMLoc callLoc,
                 ExprEmitter &emitter) {
  if (!calleeValAndType)
    return {};

  auto emitError = [&](const Twine &message) {
    return emitter.emitError(callLoc, message);
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
  if (numArgs != operands.size()) {
    emitError("callee expects ") << numArgs << " argument" << plural(numArgs);
    return {};
  }

  // Emit all the arguments.
  SmallVector<Value> valueArguments;
  for (auto [argAnyValueTypeAndLoc, expectedType] :
       llvm::zip(operands, calleeIRType.getValues().getInputs())) {
    // If the callee takes the operand as a by-ref argument, we require an
    // lvalue.
    Value argVal;
    if (isa<POP::PointerType>(expectedType)) {
      argVal = argAnyValueTypeAndLoc.first.ir.getIfLValue();
      if (!argVal) {
        emitter.emitError(
            argAnyValueTypeAndLoc.second,
            "operand must be mutable in order to pass as a by-ref argument");
        return {};
      }
    } else {
      // Otherwise, we pass as an r-value.
      argVal = emitter
                   .emitDRValue(argAnyValueTypeAndLoc.first,
                                argAnyValueTypeAndLoc.second)
                   .ir;
    }

    if (!argVal)
      return {};

    // TODO: Handle implicit conversions.
    if (argVal.getType() != expectedType) {
      emitter.emitError(argAnyValueTypeAndLoc.second, "value of type ")
          << argAnyValueTypeAndLoc.first.type
          << " cannot be converted to expected type "
          // TODO: Print pretty expected type when we have it.
          << expectedType;
      return {};
    }
    valueArguments.push_back(argVal);
  }

  if (!emitter.builder) {
    emitError("TODO: cannot call function in parameter context");
    return {};
  }

  // If this is a call to something representable as an attribute, we can use
  // a kgen.call_param.
  Value resultVal;
  auto loc = emitter.translateLocation(callLoc);
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
        emitter.emitDRValue({AnyValue(calleeVal), calleeType}, callLoc);
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

/// Emit a function call for a call node with the specified operands.
static ASTTypeAnd<AnyValue>
emitFunctionCall(const CallNode &call, ASTTypeAnd<RValue> calleeValAndType,
                 ExprEmitter &emitter) {
  SmallVector<ArgumentValueType> operands;
  for (ExprNode *arg : call.args) {
    operands.push_back({arg->emitIR(emitter), arg->getLoc()});
    if (!operands.back().first)
      return {};
  }
  return emitFunctionCall(calleeValAndType, operands, call.getLoc(), emitter);
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
    assert(mv.ir && "always succeeds");
    return {mv.ir, mv.type};
  }

  // RValue's and LValues always resolve to their known value.
  if (auto rvalue = decl->getIfRValue())
    return {rvalue, decl->getResolvedType()};
  if (auto lvalue = decl->getIfLValue())
    return {lvalue, decl->getResolvedType()};

  // If this is a type declaration, return it as a type.
  if (isa<LITStructDeclOp>(*decl) || decl->isMagic() || decl->getIfMLIRType()) {
    auto astType = emitter.shared.getASTType(*decl, {});
    return {MValue(astType), emitter.shared.getTypeType()};
  }

  emitter.emitError(loc, "use of declaration \"")
      << memberName << "\" as a value isn't supported yet";
  return {};
}

/// Given an MLIR type, return an ASTType that we can use for type system
/// processing.  This should only be used for low level operations touching
/// MLIR, it isn't efficient and shouldn't be used for general user defined
/// types.
static ASTType getASTTypeForMLIRType(Type mlirType, SMLoc loc,
                                     LitSharedState &shared) {

  // To get an ASTType from an MLIR type, we stringify the MLIR type and look it
  // up on the __mlir_type declaration.
  std::string typeStr;
  llvm::raw_string_ostream(typeStr) << mlirType;

  // See if we already have this declaration.
  auto &mlirTypeScope = shared.getMLIRTypeScope();
  ASTDecl *typeDecl =
      mlirTypeScope.lookup(StringAttr::get(shared.getContext(), typeStr));

  // If not, synthesize it.
  if (!typeDecl) {
    typeDecl = synthesizeMLIRTypeDeclEntry(typeStr, loc, mlirTypeScope, shared);
    if (!typeDecl)
      return {};
  }

  return shared.getASTType(*typeDecl, {});
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

/// Given a call to an UnboundMLIROperator, generate an MLIR operation with
/// the operands as SSA values.
static ASTTypeAnd<AnyValue> emitMLIROperatorCall(const CallNode &call,
                                                 RValue calleeVal,
                                                 ExprEmitter &emitter) {
  if (!emitter.builder) {
    emitter.emitError(call.getLoc(), "cannot emit operation in this context");
    return {};
  }
  auto maVal = calleeVal.getIfMAValue();
  if (!maVal) {
    emitter.emitError(call.getLoc(), "unknown unbound MLIR operator");
    return {};
  }
  auto unboundOp = dyn_cast<UnboundMLIROperationAttr>(maVal.get());
  if (!unboundOp) {
    emitter.emitError(call.getLoc(), "unknown unbound MLIR operator");
    return {};
  }

  // Emit all the arguments so we can encode them as SSA values.
  SmallVector<Value> opOperands;
  for (auto operand : call.args) {
    opOperands.push_back(emitter.emitDRValue(operand).ir);
    if (!opOperands.back())
      return {};
  }

  OperationState state(emitter.translateLocation(call.getLoc()),
                       unboundOp.getName());
  state.addOperands(opOperands);
  // TODO: Translate attributes from opAttr when it can hold them.

  // Finally, figure out the return types using InferTypeOpInterface if the
  // operation is registered and if it is present.
  auto *context = emitter.getContext();
  bool inferredTypes = false;
  if (auto opNameInfo =
          mlir::RegisteredOperationName::lookup(unboundOp.getName(), context)) {
    if (auto inferTypesItf =
            opNameInfo->getInterface<mlir::InferTypeOpInterface>()) {
      if (failed(inferTypesItf->inferReturnTypes(
              context, state.location, state.operands,
              DictionaryAttr::get(context, state.attributes), state.regions,
              state.types))) {
        emitter.emitError(call.getLoc(),
                          "unable to infer result type from MLIR operation ")
            << unboundOp.getName();
        return {};
      }

      if (state.types.size() > 1) {
        emitter.emitError(call.getLoc(),
                          "cannot use operations with multiple results (yet) ")
            << unboundOp.getName();
      }

      inferredTypes = true;
    }
  }
  // If a result type wasn't specified, it must be set as an attribute.
  if (!inferredTypes) {
    emitter.emitError(call.getLoc(),
                      "unable to infer result type from MLIR operation ")
        << unboundOp.getName();
    return {};
  }

  Operation *resultOp = emitter.builder->create(state);

  // Explicitly run the verifier on the new operation so we make sure to catch
  // problems early.
  std::string errorMessage;
  bool verificationError;
  {
    // FIXME: This doesn't silence errors!
    mlir::ScopedDiagnosticHandler handler(
        context, [&](Diagnostic &diag) { errorMessage = diag.str(); });
    // Verify that the resulting op is correctly constructed.  If not, we fail.
    verificationError = failed(mlir::verify(resultOp));
  }
  if (verificationError) {
    resultOp->emitOpError("MLIR verification error: ") << errorMessage;
    return {};
  }

  // If we succeeded and have no types, then install a None type.
  if (resultOp->getNumResults() == 0) {
    auto noneMLIRType = KGEN::NoneType::get(emitter.getContext());
    return {MAValue(NoneAttr::get(emitter.getContext(), noneMLIRType)),
            emitter.shared.getNoneType()};
  }

  auto astType = getASTTypeForMLIRType(resultOp->getResult(0).getType(),
                                       call.getLoc(), emitter.shared);
  return {DRValue(resultOp->getResult(0)), astType};
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

  // If this is the invocation of an unbound MLIR operator, bind it into an
  // actual operator!
  if (calleeVal.type.getDecl().magicKind ==
      MagicDeclKind::kUnboundMLIROperatorType)
    return emitMLIROperatorCall(*this, calleeVal.ir, emitter);

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
    ASTTypeAnd<RValue> exprRep = emitter.emitRValue(expr);
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
    Location loc = emitter.translateLocation(getLoc());
    last = DRValue(emitter.builder->create<mlir::index::ConstantOp>(loc, 0));
  }
  return {last, emitter.shared.getIndexType()};
}

/// Given an operator, return the SpecialFunction that implements it.
/// TODO: Expand this to support multiple results, e.g. add/radd.
static SpecialFunctionKind getOpSpecialFunctions(ExprNode::Kind kind) {
  switch (kind) {
  default:
    // TODO: Add support for more of these.
    return SpecialFunctionKind::kNormal;
  case ExprNode::Kind::kUnaryPlus:
    return SpecialFunctionKind::kPos;
  case ExprNode::Kind::kUnaryMinus:
    return SpecialFunctionKind::kNeg;
  case ExprNode::Kind::kUnaryTilde:
    return SpecialFunctionKind::kInvert;
  case ExprNode::kAdd:
    return SpecialFunctionKind::kAdd;
  case ExprNode::kSub:
    return SpecialFunctionKind::kSub;
  case ExprNode::kMul:
    return SpecialFunctionKind::kMul;
  case ExprNode::kMatrixMul:
    return SpecialFunctionKind::kMatmul;
  case ExprNode::kDiv:
    return SpecialFunctionKind::kTrueDiv;
  case ExprNode::kModulo:
    return SpecialFunctionKind::kMod;
  case ExprNode::kBitwiseAnd:
    return SpecialFunctionKind::kAnd;
  case ExprNode::kBitwiseOr:
    return SpecialFunctionKind::kOr;
  case ExprNode::kBitwiseXor:
    return SpecialFunctionKind::kXor;
  case ExprNode::kLeftShift:
    return SpecialFunctionKind::kLshift;
  case ExprNode::kRightShift:
    return SpecialFunctionKind::kRshift;
  case ExprNode::kExp:
    return SpecialFunctionKind::kPow;
  case ExprNode::kFloorDiv:
    return SpecialFunctionKind::kFloorDiv;
  case ExprNode::kCmpLess:
    return SpecialFunctionKind::kCmpLess;
  case ExprNode::kCmpLessEqual:
    return SpecialFunctionKind::kCmpLessEqual;
  case ExprNode::kCmpEqual:
    return SpecialFunctionKind::kCmpEqual;
  case ExprNode::kCmpNotEqual:
    return SpecialFunctionKind::kCmpNotEqual;
  case ExprNode::kCmpGreater:
    return SpecialFunctionKind::kCmpGreater;
  case ExprNode::kCmpGreaterEqual:
    return SpecialFunctionKind::kCmpGreaterEqual;
  case ExprNode::kPlusAssign:
    return SpecialFunctionKind::kIAdd;
  case ExprNode::kMinusAssign:
    return SpecialFunctionKind::kISub;
  case ExprNode::kMulAssign:
    return SpecialFunctionKind::kIMul;
  case ExprNode::kMatMulAssign:
    return SpecialFunctionKind::kIMatmul;
  case ExprNode::kDivAssign:
    return SpecialFunctionKind::kITrueDiv;
  case ExprNode::kModuloAssign:
    return SpecialFunctionKind::kIMod;
  case ExprNode::kBitwiseAndAssign:
    return SpecialFunctionKind::kIAnd;
  case ExprNode::kBitwiseOrAssign:
    return SpecialFunctionKind::kIOr;
  case ExprNode::kBitwiseXorAssign:
    return SpecialFunctionKind::kIXor;
  case ExprNode::kLeftShiftAssign:
    return SpecialFunctionKind::kILshift;
  case ExprNode::kRightShiftAssign:
    return SpecialFunctionKind::kIRshift;
  case ExprNode::kExpAssign:
    return SpecialFunctionKind::kIPow;
  case ExprNode::kFloorDivAssign:
    return SpecialFunctionKind::kIFloorDiv;
  }
}

ASTTypeAnd<AnyValue> BinOpNode::emitIR(ExprEmitter &emitter,
                                       ASTType contextualType) const {
  ASTTypeAnd<AnyValue> lhsRep, rhsRep;

  // We generally emit the LHS before the RHS, but need to do special things
  // for an assignment statement.
  if (!isAssignmentStmt()) {
    lhsRep = lhs->emitIR(emitter);
    rhsRep = rhs->emitIR(emitter);
    if (!lhsRep || !rhsRep)
      return {};
  } else {
    // In an assignment, we emit the RHS first as a value and the LHS as an
    // lvalue with a contextual type.  This is required to enable the 'implicit
    // declaration' behavior in a def.
    rhsRep = rhs->emitIR(emitter);
    if (!rhsRep)
      return {};

    // If this variable is being declared in a `def` definition, then we allow
    // implicit declarations of variables.  In `fn` and top level, we do not.
    ASTType lhsContextualType;
    if (emitter.declScope.isDef)
      lhsContextualType = rhsRep.type;

    // Emit the LHS pattern as an lvalue.
    auto lhsPat = emitter.emitLValue(lhs, lhsContextualType,
                                     "cannot assign to immutable expression");
    if (!lhsPat)
      return {};

    // Assignment expression (`=`) turns into a store, not into a method call.
    if (kind == kAssign) {
      auto rv = emitter.emitDRValue(rhsRep, rhs->getLoc());
      if (!rv)
        return {};

      // Check to see if the destination type and the source type are
      // compatible.
      // TODO: Implement implicit conversions.
      if (!lhsPat.type.isEqualCanon(rv.type)) {
        emitter.emitError(rhs->getLoc(), "cannot convert value of type ")
            << rv.type << " to " << lhsPat.type;
        return {};
      }

      // If everything worked out, store the resultant value into the lvalue for
      // the destination.  If things didn't work, just drop this on the floor.
      emitter.builder->create<POP::StoreOp>(emitter.translateLocation(getLoc()),
                                            rv.ir, lhsPat.ir,
                                            /*alignment*/ None);
      return {rv.ir, rv.type};
    }

    // Otherwise, handle as a normal binary operator.
    lhsRep = {lhsPat.ir, lhsPat.type};
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnKind = getOpSpecialFunctions(kind);

  // FIXME: We currently hack in index type support as transition to proper
  // expression support.
  if (lhsRep.type.getDecl().magicKind == MagicDeclKind::kIndexType ||
      rhsRep.type.getDecl().magicKind == MagicDeclKind::kIndexType)
    specialFnKind = SpecialFunctionKind::kNormal;

  if (specialFnKind != SpecialFunctionKind::kNormal) {
    // Get metadata about the special function that backs this expression.  This
    // allows us to look up information about whether the operands implement
    // support for it.
    auto specialFnInfo = SpecialFunctionInfo::get(specialFnKind);

    // Look up the normal function on the LHS type.
    // TODO: Add support for radd, looking up on the RHS.
    auto nameAttr = StringAttr::get(emitter.getContext(), specialFnInfo.name);
    ASTDecl *lookupResult = lhsRep.type.getDecl().lookup(nameAttr);
    if (!lookupResult) {
      // TODO: Add support for radd, looking up on the RHS.  On a hit, notice
      // its result and swap lhs/rhs rep values.
      emitter.emitError(getLoc(), "")
          << lhsRep.type << " does not implement the " << nameAttr
          << " special method";
      return {};
    }

    // Make sure the signature is resolved.
    if (failed(emitter.shared.declResolver->resolve(
            *lookupResult, DeclResolvedness::signatureResolved, getLoc())))
      return {};

    ASTTypeAnd<MValue> callee = emitFuncReference(
        cast<LITFuncOp>(*lookupResult), *lookupResult, emitter);
    assert(callee.ir && "always succeeds");
    ArgumentValueType argValues[] = {{lhsRep, lhs->getLoc()},
                                     {rhsRep, rhs->getLoc()}};

    return emitFunctionCall({RValue(callee.ir), callee.type}, argValues,
                            getLoc(), emitter);
  }

  // TODO: Remove all this legacy code.

  auto lhsRVal = emitter.emitRValue(lhsRep, lhs->getLoc());
  auto rhsRVal = emitter.emitRValue(rhsRep, rhs->getLoc());
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
  if (auto lhsParam = lhsRVal.ir.getIfMAValue()) {
    if (auto rhsParam = rhsRVal.ir.getIfMAValue()) {
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

  auto lhsVal = emitter.emitDRValue(lhsRVal, lhs->getLoc()).ir;
  auto rhsVal = emitter.emitDRValue(rhsRVal, rhs->getLoc()).ir;
  auto loc = emitter.translateLocation(getLoc());

  // TODO: implement properly these operations once we have a real type system
  //       also, logical operators should implement short circuiting of the
  //       operands.
  Value result;
  switch (kind) {
  default:
    emitter.emitError(getLoc(),
                      "cannot emit binary operator on this index value yet");
    return {};

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
  auto exprRep = subExpr->emitIR(emitter);
  if (!exprRep)
    return {};

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnKind = getOpSpecialFunctions(kind);

  assert(specialFnKind != SpecialFunctionKind::kNormal &&
         "Unary operators are implemented via special methods");
  // Get metadata about the special function that backs this expression.  This
  // allows us to look up information about whether the operands implement
  // support for it.
  auto specialFnInfo = SpecialFunctionInfo::get(specialFnKind);

  // Look up the normal function on the expr type.
  auto nameAttr = StringAttr::get(emitter.getContext(), specialFnInfo.name);
  ASTDecl *lookupResult = exprRep.type.getDecl().lookup(nameAttr);
  if (!lookupResult) {
    emitter.emitError(getLoc(), "")
        << exprRep.type << " does not implement the " << nameAttr
        << " special method";
    return {};
  }

  // Make sure the signature is resolved.
  if (failed(emitter.shared.declResolver->resolve(
          *lookupResult, DeclResolvedness::signatureResolved, getLoc())))
    return {};

  ASTTypeAnd<MValue> callee =
      emitFuncReference(cast<LITFuncOp>(*lookupResult), *lookupResult, emitter);
  assert(callee.ir && "always succeeds");
  ArgumentValueType argValue = {exprRep, subExpr->getLoc()};

  return emitFunctionCall({RValue(callee.ir), callee.type}, argValue, getLoc(),
                          emitter);
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
