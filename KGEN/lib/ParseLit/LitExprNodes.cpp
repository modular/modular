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
#include "mlir/IR/Matchers.h"
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
  if (!attr) {
    auto type = rep.ir.getIfMTValue();
    assert(type && "unknown rvalue kind");
    attr = ParameterizedTypeConstantAttr::get(shared.getMLIRType(type, loc));
  }

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

/// Given a StringRef for an MLIR attribute, invoke the MLIR parser to resolve
/// it into an Attribute (which may not be a TypedAttr) and return it.  On
/// error, emit a diagnostic and return null.
static Attribute parseMLIRAttrFromString(StringRef name, SMLoc loc,
                                         ExprEmitter &emitter) {
  Attribute result;
  std::string errorMsg;
  {
    // Capture errors thrown by parseAttribute and ignore them.
    // FIXME: This doesn't silence errors!
    mlir::ScopedDiagnosticHandler handler(emitter.shared.getContext(),
                                          [&](Diagnostic &diag) {
                                            errorMsg = diag.str();
                                            printf("hello\n");
                                          });

    // FIXME(https://github.com/llvm/llvm-project/issues/58964)
    // Copy the string into a temporary smallvector so we can make sure it is
    // nul terminated for the MLIR asmparser.
    SmallString<64> tmpBuf(name.begin(), name.end());
    tmpBuf.push_back(0);
    result = mlir::parseAttribute(StringRef(tmpBuf).drop_back(),
                                  emitter.shared.getContext());
  }
  if (!result) {
    emitter.shared.emitError(loc, "invalid MLIR attribute: ") << errorMsg;
    return {};
  }
  return result;
}

/// This implements __mlir_attr.x lookup, synthesizing a MAValue for the
/// attribute on demand.
static ASTTypeAnd<AnyValue>
synthesizeMLIRAttrFromString(StringRef name, SMLoc loc, ExprEmitter &emitter) {
  auto attr = parseMLIRAttrFromString(name, loc, emitter);
  if (!attr)
    return {};

  auto typedAttr = dyn_cast<TypedAttr>(attr);
  if (!typedAttr) {
    emitter.shared.emitError(loc, "MLIR attribute has no type: ") << attr;
    return {};
  }

  auto astType = emitter.shared.getASTTypeForMLIRType(typedAttr.getType(), loc);
  return {MAValue(typedAttr), astType};
}

/// When a lookup in __mlir_op fails for a named field, this method tries to
/// resolve it.  On success, it lazily creates a resolved declaration.  On
/// failure, it bails out.
static ASTTypeAnd<AnyValue> synthesizeMLIROpFromString(StringRef name,
                                                       ExprEmitter &emitter) {
  auto &shared = emitter.shared;
  auto *context = shared.getContext();
  auto nameStr = StringAttr::get(context, name);

  auto result = UnboundMLIROperationAttr::get(
      context, nameStr.getType(), nameStr, DictionaryAttr::get(context));

  return {MAValue(result), shared.getUnboundMLIROperatorType()};
}

/// Calculate the result of an __mlir_op.`thing`[attributes], applying the
/// attributes list to the operation specification.
static ASTTypeAnd<AnyValue>
bindAttributesToMLIROperatorCall(const SubscriptNode &subscript,
                                 TypedAttr opInfo, ExprEmitter &emitter) {
  auto unboundOp = cast<UnboundMLIROperationAttr>(opInfo);
  auto loc = subscript.getLoc();
  auto *context = emitter.getContext();

  // Only allow applying attributes to something without them.
  if (!unboundOp.getAttrs().empty()) {
    emitter.shared.emitError(loc, "operation already has attributes");
    return {};
  }

  // Given an expression, try to resolve it into an Attribute that we can
  // install on this operation.
  auto getAttrFromExpr = [&](StringRef name, ExprNode *node) -> Attribute {
    // Special case handling of __mlir_attr.`xxx` directly in this parser,
    // because we want to be able to install arbitrary attributes into an
    // operation's attribute list, and emitMAValue only supports TypedAttrs.
    if (auto attrRef = dyn_cast<AttributeRefNode>(node)) {
      auto mlirAttr = dyn_cast<DeclRefNode>(attrRef->base);
      if (mlirAttr && mlirAttr->spelling == "__mlir_attr")
        return parseMLIRAttrFromString(attrRef->attrSpelling, attrRef->getLoc(),
                                       emitter);
    }

    // Otherwise emit the value as an MAValue.  This allows references to
    // parameter expressions.
    auto value = emitter.emitMAValue(
        node, "attribute value for '" + Twine(name) + "' must be constant");
    if (!value)
      return {};
    return value.ir.get();
  };

  SmallVector<NamedAttribute> attrValues;

  // Each element of the subscript must have a name identifier and a value as an
  // MAValue.
  for (auto *subscriptIdx : subscript.indices) {
    auto *slice = dyn_cast<SliceNode>(subscriptIdx);
    if (!slice || slice->colon2Loc.isValid() || !slice->lower ||
        !slice->upper || !isa<DeclRefNode>(slice->lower)) {
      emitter.shared.emitError(
          loc, "attribute spec requires an attribute name and attr value");
      return {};
    }

    auto name = cast<DeclRefNode>(slice->lower)->spelling;
    auto value = getAttrFromExpr(name, slice->upper);
    if (!value)
      return {};
    attrValues.push_back(NamedAttribute(StringAttr::get(context, name), value));
  }

  // Check for duplicate attribute specifications.
  if (auto duplicate = DictionaryAttr::findDuplicate(attrValues, false)) {
    emitter.shared.emitError(loc, "attribute ")
        << duplicate->getName() << " redundantly specified";
    return {};
  }

  // Return it.
  auto attrs = DictionaryAttr::get(context, attrValues);
  auto result = UnboundMLIROperationAttr::get(context, unboundOp.getType(),
                                              unboundOp.getName(), attrs);
  return {MAValue(result), emitter.shared.getUnboundMLIROperatorType()};
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
    // If this is a lookup in __mlir_type, then try to lazily synthesize the
    // element in question.
    if (scope.magicKind == MagicDeclKind::k__mlir_type)
      return shared.synthesizeMLIRTypeDeclEntry(name, loc, scope);

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

  // If the lookup succeeded, make sure the signature for the referenced decl
  // is understood.
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

/// This helper emits a method call to a special function (`kind`) on the
/// `caller` object with the provided `operands`. If the special function
/// is not implemented by the caller it emits an error.
/// This returns null if emission fails.
ASTTypeAnd<AnyValue> ExprEmitter::emitSpecialFunctionCall(
    ASTTypeAnd<DRValue> caller, SpecialFunctionKind kind,
    ArrayRef<ArgumentValueType> operands, SMLoc callLoc) {

  auto specialFnInfo = SpecialFunctionInfo::get(kind);
  // Look up the special function on the expr type.
  auto nameAttr = StringAttr::get(getContext(), specialFnInfo.name);
  ASTDecl *lookupResult = caller.type.getDecl().lookup(nameAttr);
  if (!lookupResult) {
    emitError(callLoc, "") << caller.type << " does not implement the "
                           << nameAttr << " special method";
    return {};
  }

  // Make sure the signature is resolved.
  if (failed(shared.declResolver->resolve(
          *lookupResult, DeclResolvedness::signatureResolved, callLoc)))
    return {};
  ASTTypeAnd<MValue> callee =
      emitFuncReference(cast<LITFuncOp>(*lookupResult), *lookupResult, *this);
  assert(callee.ir && "always succeeds");
  return emitFunctionCall({RValue(callee.ir), callee.type}, operands, callLoc,
                          *this);
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
    // Handle __mlir_op.`xxx` references.
    if (baseType.getDecl().magicKind == MagicDeclKind::k__mlir_op)
      return synthesizeMLIROpFromString(attrSpelling, emitter);
    // Handle __mlir_attr.`xxx` references.
    if (baseType.getDecl().magicKind == MagicDeclKind::k__mlir_attr)
      return synthesizeMLIRAttrFromString(attrSpelling, getLoc(), emitter);
    // Normal member reference.
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
                                                 TypedAttr calleeVal,
                                                 ExprEmitter &emitter) {
  auto unboundOp = cast<UnboundMLIROperationAttr>(calleeVal);
  auto *context = emitter.getContext();

  if (!emitter.builder) {
    emitter.emitError(call.getLoc(), "cannot emit operation in this context");
    return {};
  }

  // Emit all the arguments so we can encode them as SSA values.
  SmallVector<Value> opOperands;
  for (auto operand : call.args) {
    opOperands.push_back(emitter.emitDRValue(operand).ir);
    if (!opOperands.back())
      return {};
  }

  // Set up the OperationState for the thing we're building.
  OperationState state(emitter.translateLocation(call.getLoc()),
                       unboundOp.getName());
  state.addOperands(opOperands);

  // Process the attributes and figure out the result type if specified.
  for (auto &attr : unboundOp.getAttrs()) {
    if (attr.getName() == "_type") {
      // The value must be a concrete or parametric type.
      if (auto type = dyn_cast<ConcreteTypeConstantAttr>(attr.getValue())) {
        state.types.push_back(type.getValue());
      } else if (auto type =
                     dyn_cast<ParameterizedTypeConstantAttr>(attr.getValue())) {
        state.types.push_back(type.getValue());
      } else {
        emitter.emitError(call.getLoc(), "unknown _type value");
        return {};
      }
      continue;
    }
    state.addAttributes(attr);
  }

  // Finally, if we don't already have a type, figure out the return types using
  // InferTypeOpInterface if the operation is registered and if it is present.
  auto inferType = [&]() -> LogicalResult {
    auto opNameInfo =
        mlir::RegisteredOperationName::lookup(unboundOp.getName(), context);
    if (!opNameInfo)
      return failure();
    auto inferTypesItf = opNameInfo->getInterface<mlir::InferTypeOpInterface>();
    if (!inferTypesItf)
      return failure();
    return inferTypesItf->inferReturnTypes(
        context, state.location, state.operands,
        DictionaryAttr::get(context, state.attributes), state.regions,
        state.types);
  };

  if (state.types.empty()) {
    if (failed(inferType())) {
      emitter.emitError(call.getLoc(),
                        "unable to infer result type from MLIR operation ")
          << unboundOp.getName();
      return {};
    }
    if (state.types.size() > 1) {
      emitter.emitError(call.getLoc(),
                        "cannot use operations with multiple results (yet) ")
          << unboundOp.getName();
      return {};
    }
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
    // Verify that the resulting op is correctly constructed.  If not, we
    // fail.
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

  assert(resultOp->getNumResults() == 1 &&
         "Only support single result ops so far");

  auto astType = emitter.shared.getASTTypeForMLIRType(
      resultOp->getResult(0).getType(), call.getLoc());

  // Check to see if we can fold this operation.  This enables use of __mlir_op
  // to produce meta-values without forcing them into the dynamic value domain.
  SmallVector<Attribute, 4> constOperands(resultOp->getNumOperands());
  for (unsigned i = 0, e = constOperands.size(); i != e; ++i)
    matchPattern(resultOp->getOperand(i), mlir::m_Constant(&constOperands[i]));
  SmallVector<OpFoldResult, 4> foldResults;
  if (succeeded(resultOp->fold(constOperands, foldResults)) &&
      foldResults.size() == 1) {
    auto folded = PointerUnion<Attribute, Value>(foldResults[0]);
    // If the result was some other value that already exists, use it.
    if (auto val = dyn_cast<Value>(folded)) {
      resultOp->erase();
      return {DRValue(val), astType};
    }

    if (auto attr = dyn_cast<TypedAttr>(cast<Attribute>(folded))) {
      // If it is a constant, make an MAValue result.
      resultOp->erase();
      return {MAValue(attr), astType};
    }
  }

  // If folding failed, return the operation normally.
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
    return emitMLIROperatorCall(*this, calleeVal.ir.getIfMAValue(), emitter);

  emitter.emitError(getLoc(), "unable to call value of type ")
      << calleeVal.type;
  return {};
}

ASTTypeAnd<AnyValue> SliceNode::emitIR(ExprEmitter &emitter,
                                       ASTType contextualType) const {
  emitter.emitError(getLoc(), "slice values not implemented yet");
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

  if (subValue.type.getDecl().magicKind ==
      MagicDeclKind::kUnboundMLIROperatorType)
    return bindAttributesToMLIROperatorCall(*this, subValue.ir.getIfMAValue(),
                                            emitter);

  // Emit each of the index values to generate error messages.
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
      rhsRep.type.getDecl().magicKind == MagicDeclKind::kIndexType) {
    auto lhsParam =
        emitter.emitMAValue(lhs, "expecting parameter values as operands");
    auto rhsParam =
        emitter.emitMAValue(rhs, "expecting parameter values as operands");
    // If these are both parameter values, we can fold them using parameter
    // expressions.
    if (!lhsParam || !rhsParam) {
      emitter.emitError(getLoc(), "expecting parameter values as operands");
      return {};
    }
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
    return {MValue(ParamOperatorAttr::get(opcode, lhsParam.ir, rhsParam.ir)),
            emitter.shared.getIndexType()};
  }

  assert(specialFnKind != SpecialFunctionKind::kNormal);
  // TODO: Add support for radd, looking up on the RHS.
  ArgumentValueType argValues[] = {{lhsRep, lhs->getLoc()},
                                   {rhsRep, rhs->getLoc()}};
  return emitter.emitSpecialFunctionCall(
      {lhsRep.ir.getIfDRValue(), lhsRep.type}, specialFnKind, argValues,
      getLoc());
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

  ArgumentValueType argValue = {exprRep, subExpr->getLoc()};

  return emitter.emitSpecialFunctionCall(
      {exprRep.ir.getIfDRValue(), exprRep.type}, specialFnKind, argValue,
      getLoc());
}

ASTTypeAnd<AnyValue> TernaryOpNode::emitIR(ExprEmitter &emitter,
                                           ASTType contextualType) const {
  ASTTypeAnd<DRValue> cond = emitter.emitDRValue(condExpr);
  if (!cond)
    return {};

  SMLoc condLoc = condExpr->getLoc();
  ArgumentValueType argValue = {{cond.ir, cond.type}, condLoc};
  ASTTypeAnd<AnyValue> boolCall = emitter.emitSpecialFunctionCall(
      cond, SpecialFunctionKind::kBool, argValue, condLoc);
  if (!boolCall)
    return {};

  argValue = {boolCall, condLoc};
  ASTTypeAnd<AnyValue> litBoolCall = emitter.emitSpecialFunctionCall(
      {boolCall.ir.getIfDRValue(), boolCall.type},
      SpecialFunctionKind::kLitBool, argValue, condLoc);
  if (!litBoolCall || !litBoolCall.ir.getIfDRValue())
    return {};

  Value condValue = static_cast<Value>(litBoolCall.ir.getIfDRValue());

  Type dummyType = mlir::IndexType::get(emitter.getContext());
  Location ifLoc = emitter.translateLocation(getLoc());
  // At this point we don't know the type of trueExpr / falseExpr, use
  // a dummy one.
  auto ifOp = emitter.builder->create<scf::IfOp>(ifLoc, TypeRange{dummyType},
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
  if (!trueVal.type.isEqualCanon(falseVal.type)) {
    emitter.emitError(
        getLoc(), "the types of a conditional expression must be compatible:  ")
        << trueVal.type << " is not compatible with " << falseVal.type;
    return {};
  }
  Type resultType = emitter.shared.getMLIRType(trueVal.type, ifLoc);
  // Ensure the correct type is used.
  ifOp->getResult(0).setType(resultType);
  return {(DRValue)ifOp.getResult(0), trueVal.type};
}
