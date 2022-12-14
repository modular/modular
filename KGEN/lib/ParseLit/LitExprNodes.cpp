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

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LitExprEmitter.h"
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

  // Check to see if the

  return result;
}

/// This implements __mlir_attr.x lookup, synthesizing a MAValue for the
/// attribute on demand.
static AnyValue synthesizeMLIRAttrFromString(StringRef name, SMLoc loc,
                                             ExprEmitter &emitter) {
  auto attr = parseMLIRAttrFromString(name, loc, emitter);
  if (!attr)
    return {};

  auto typedAttr = dyn_cast<TypedAttr>(attr);
  if (!typedAttr) {
    emitter.shared.emitError(loc, "MLIR attribute has no type: ") << attr;
    return {};
  }
  return MValue(typedAttr);
}

/// When a lookup in __mlir_op fails for a named field, this method tries to
/// resolve it.  On success, it lazily creates a resolved declaration.  On
/// failure, it bails out.
static AnyValue synthesizeMLIROpFromString(StringRef name,
                                           ExprEmitter &emitter) {
  auto &shared = emitter.shared;
  auto *context = shared.getContext();
  auto nameStr = StringAttr::get(context, name);

  auto result = UnboundMLIROperationAttr::get(
      context, nameStr.getType(), nameStr, DictionaryAttr::get(context));
  return MValue(result);
}

/// Calculate the result of an __mlir_op.`thing`[attributes], applying the
/// attributes list to the operation specification.
static AnyValue
bindAttributesToMLIROperatorCall(const SubscriptNode &subscript,
                                 UnboundMLIROperationAttr unboundOp,
                                 ExprEmitter &emitter) {
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
    auto value = emitter.emitMValue(
        node, "attribute value for '" + Twine(name) + "' must be constant");
    if (!value)
      return {};
    return value.get();
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
  return MValue(UnboundMLIROperationAttr::get(context, unboundOp.getType(),
                                              unboundOp.getName(), attrs));
}

/// Given an ASTType 'containingType', look up a named member of it and return
/// the reference to its symbol as an RValue.
static CallableValue
emitDeclMemberAsCallable(ASTDecl &container, ArrayRef<ParamBindAttr> bindings,
                         StringRef memberName, const ExprNode *node,
                         ExprEmitter &emitter, ASTType contextualType = {}) {
  // Perform a lookup of the specified decl in the current container.
  ExprEmitter::LookupResult lookup =
      emitter.lookupAndResolveDecl(memberName, node->getLoc(), container);

  // If that lookup failed, but we can synthesize a variable declaration in this
  // scope, do that.  We can only do this if there is a contextual type
  // available and an insertion point.
  if (lookup.isFailure() && contextualType && emitter.varDeclCursor) {
    // Introduce a new lit.var.decl node whose type matches the
    // implicitDeclType.
    // TODO(autopromotions): turn infinite integers into concrete ones as
    // needed.
    Type declIRType = POP::PointerType::get(contextualType);

    // Use this builder to place any VarDeclOps. In Python there is only one
    // scope per function and all variables belong to that scope, so builders
    // should reflect that.
    auto loc = emitter.translateLocation(node->getLoc());
    auto nameAttr = StringAttr::get(loc.getContext(), memberName);
    auto varDecl = OpBuilder(emitter.varDeclCursor)
                       .create<VarDeclOp>(loc, declIRType, nameAttr);

    // If the unresolved name is `_`, then we have a discard pattern.  Python
    // supports this by just implicitly declaring a variable named _ and
    // allowing rewrites, but we cannot take this approach because each discard
    // could have a different type.  Handle this specially by not inserting the
    // `_` variable into the name table, so we'll get a new instance on every
    // use.
    if (memberName == "_") {
      // Move it right before the use, like a var decl, instead of leaving it at
      // the entrypoint of the function.  It won't get reused.
      varDecl->moveBefore(emitter.builder->getInsertionBlock(),
                          emitter.builder->getInsertionPoint());
      return {{AnyValue(LValue(varDecl.getResult())), node}};
    }

    // By policy in order to produce a more predictable programming model,
    // implicit declarations of variables are only allowed in `def` contexts,
    // not in `fn`, structs, or top level.  We could re-evaluate this in the
    // future if we'd like.
    auto funcContext =
        dyn_cast_or_null<LIT::FuncOp>(emitter.declScope.getIfOperation());
    if (!funcContext || !funcContext.getIsDef()) {
      auto diag = emitter.emitError(node->getLoc())
                  << "use of unknown declaration \"" << memberName << '"';
      if (funcContext)
        diag << ", `fn` declarations require explicit variable declarations";
      return {};
    }

    // In a normal implicit declaration, we add it to the name table so
    // subsequent uses find this one.
    auto *decl = &emitter.shared.declResolver->addFullyResolvedDecl(
        varDecl, node->getLoc(), nameAttr, &container);
    lookup = ExprEmitter::LookupResult::getSuccess(decl);
  }

  ASTDecl *decl = lookup.getIfSuccess();
  if (!decl) {
    if (lookup.isFailure()) {
      auto diag = emitter.emitError(node->getLoc());
      if (auto structDecl = dyn_cast<StructDeclOp>(container))
        diag << structDecl.getName() << " has no '" << memberName << "' member";
      else
        diag << "use of unknown declaration \"" << memberName << '"';
    }
    return {};
  }

  // Variable references resolve to an lvalue addressing the variable.
  if (auto var = dyn_cast<VarDeclOp>(*decl))
    return {{AnyValue(LValue(var.getResult())), node}};

  // Functions form an address.
  if (isa<LIT::FuncOp>(*decl))
    return CallableValue(node->getLoc(), *decl, bindings);

  // Parameters form an meta-value.
  if (auto param = dyn_cast<ParamDeclareOp>(*decl))
    return {{MValue(ParamDeclRefAttr::get(param.getName(), param.getType())),
             node}};

  // RValue's and LValues always resolve to their known value.
  if (auto rvalue = decl->getIfRValue())
    return {{rvalue, node}};
  if (auto lvalue = decl->getIfLValue())
    return {{lvalue, node}};

  // If this is a type declaration, return it as a type.
  if (isa<StructDeclOp>(*decl))
    return {{MValue(DeclRefType::get(decl->getSymbolRef())), node}};

  emitter.emitError(node->getLoc(), "use of declaration \"")
      << memberName << "\" as a value isn't supported yet";
  return {};
}

/// Given a call to a Type value, figure out what 'T.__new__' initializer to
/// call.
static CallableValue emitInitializerCallable(ASTType calledType,
                                             const ExprNode *node,
                                             ExprEmitter &emitter) {
  ASTDecl *calledDecl = calledType.getDecl(emitter.shared);
  if (!calledDecl) {
    emitter.emitError(node->getLoc(), "cannot create instance of MLIR type ")
        << calledType;
    return {};
  }

  return emitDeclMemberAsCallable(*calledDecl, calledType.getParamBindings(),
                                  "__new__", node, emitter);
}

//===----------------------------------------------------------------------===//
// ExprNode Implementation
//===----------------------------------------------------------------------===//

ExprNode::~ExprNode() { llvm_unreachable("never called"); }

/// Emit this expression to MLIR as a CallableValue.  On error, emit an error
/// and return a null value.
CallableValue ExprNode::emitCallable(ExprEmitter &emitter,
                                     ASTType contextualType) const {
  // The default implementation of this returns the expression as an RValue.
  auto calleeVal = emitter.emitRValue(this);
  if (!calleeVal)
    return {};

  return CallableValue({calleeVal, this});
}

//===----------------------------------------------------------------------===//
// CallableValue Implementation
//===----------------------------------------------------------------------===//

SymbolConstantAttr
DirectCallable::getBoundConstantAttr(ExprEmitter &emitter) const {
  SignatureType resultType = type;

  // SymbolConstantAttr provides a type for the SymbolRefAttr with the
  // parameters substituted in.  The function reference binds any parameter
  // bindings present on the access (in bindings), which typically concretizes
  // the signature.
  if (bindings.empty()) {
    resultType = type;
  } else {
    resultType = resultType.getSpecializedSignature(
        bindings,
        [&]() -> InFlightDiagnostic { return emitter.emitError(loc, ""); });
    if (!resultType)
      return {};
  }

  return SymbolConstantAttr::get(symbol, bindings, resultType);
}

/// Get a symbol for a direct reference to the specified function in its
/// enclosing context.  This does not bind any values to arguments.
CallableValue::CallableValue(SMLoc loc, ASTDecl &fnDecl,
                             ArrayRef<ParamBindAttr> bindings)
    : CallableValue(loc, fnDecl.getSymbolRef(),
                    cast<LIT::FuncOp>(fnDecl).getFullSignature(), bindings) {}

/// Emit this as a flattened RValue or LValue.  This returns null on failure.
AnyValue CallableValue::emitAsValue(ExprEmitter &emitter) const {
  // If we have no bound symbol, return the normal lvalue or rvalue we
  // represent.
  if (!direct)
    return baseVal.ir;

  auto directSymbolAttr = direct->getBoundConstantAttr(emitter);
  if (!directSymbolAttr)
    return {};

  // If we have no base value, then we are just a symbol, return it.
  if (!baseVal)
    return MValue(directSymbolAttr);

  auto loc = baseVal.expr->getLoc();

  // Otherwise, we have a base symbol for an instance method /and/ a self
  // value to apply to it.  Partially apply it to form a result closure.
  SignatureType calleeSignature = directSymbolAttr.getType();
  Type firstArgIRType = calleeSignature.getValueInputs()[0];
  Value firstArgValue;
  switch (calleeSignature.getInputConvention(0)) {
  case ValueInputConvention::ByRef: {
    LValue baseLV = baseVal.ir.getIfLValue();
    if (!baseLV) {
      emitter.emitError(loc,
                        "invalid use of mutating method on rvalue of type ")
          << ASTType(baseVal.ir.getType());
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
    break;
  }
  case ValueInputConvention::ByVal:
    // Otherwise we can have either an lvalue or rvalue, but we need to
    // convert to an rvalue if we have an lvalue.
    firstArgValue = emitter.emitDRValue(baseVal.ir, loc);
    if (!firstArgValue)
      return {};
    break;
  }

  assert(firstArgIRType == firstArgValue.getType() &&
         "base types should always structurally line up");

  // For an instance value, we have to partially apply the callee to the first
  // argument of the reference.  Materialize callee as a DRValue for
  // partial_apply.
  auto calleeDRVal = emitter.emitDRValue(AnyValue(directSymbolAttr), loc);

  // Partial apply wants to know what operands to bind, we always bind the
  // first one.
  auto zeroAttr = emitter.builder->getAttr<mlir::DenseI64ArrayAttr>(0);
  return DRValue(emitter.builder->create<PartialApplyOp>(
      emitter.translateLocation(loc), calleeDRVal,
      mlir::ValueRange(firstArgValue), zeroAttr));
}

//===----------------------------------------------------------------------===//
// ExprNode implementations
//===----------------------------------------------------------------------===//

AnyValue IntLiteralNode::emitIR(ExprEmitter &emitter,
                                ASTType contextualType) const {
  // TODO: Handle contextual types.
  APInt value = LitLexer::getIntegerLiteralValue(spelling);

  // Make sure the value fits in 64-bits.  There are no negative values here.
  // TODO: Detect overflow errors.
  value = value.zextOrTrunc(64);
  auto attr = IntegerAttr::get(IndexType::get(emitter.getContext()), value);

  // TODO: Switch to builtin.IntegerLiteralType.
  return AnyValue(attr);
}

AnyValue FloatLiteralNode::emitIR(ExprEmitter &emitter,
                                  ASTType contextualType) const {
  // TODO: this assumes float literal are always doubles
  APFloat value = LitLexer::getFloatLiteralValue(spelling);
  auto attr = FloatAttr::get(FloatType::getF64(emitter.getContext()),
                             APFloat(value.convertToDouble()));
  // FIXME: This should eventually use emitter.shared.getFloatLiteralType()
  // when we support conversions.
  return AnyValue(attr);
}

AnyValue BoolLiteralNode::emitIR(ExprEmitter &emitter,
                                 ASTType contextualType) const {
  return AnyValue(BoolAttr::get(emitter.getContext(), value));
}

AnyValue StringLiteralNode::emitIR(ExprEmitter &emitter,
                                   ASTType contextualType) const {
  std::string value = LitLexer::getStringLiteralValue(spelling);
  auto attr = StringAttr::get(emitter.getContext(), value);
  return AnyValue(attr);
}

AnyValue NoneLiteralNode::emitIR(ExprEmitter &emitter,
                                 ASTType contextualType) const {
  // auto noneMLIRType = KGEN::NoneType::get(emitter.getContext());
  return MValue(NoneAttr::get(emitter.getContext()));
}

AnyValue DeclRefNode::emitIR(ExprEmitter &emitter,
                             ASTType contextualType) const {
  return emitCallable(emitter, contextualType).emitAsValue(emitter);
}

/// Emit this expression to MLIR as a CallableValue.  On error, emit an error
/// and return a null value.
CallableValue DeclRefNode::emitCallable(ExprEmitter &emitter,
                                        ASTType contextualType) const {
  return emitDeclMemberAsCallable(emitter.declScope, /*no param bindings*/ {},
                                  spelling, this, emitter, contextualType);
}

/// This uses the MLIR parser to turn the specified MLIR type name into an MLIR
/// type.
static Type parseMLIRType(StringRef name, SMLoc loc, LitSharedState &shared) {
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
  if (!result)
    shared.emitError(loc, "unknown MLIR type: ") << name;
  return result;
}

/// Emit an expression 'x.y' where x is known to be a Type value.
static CallableValue emitTypeAttributeRef(ASTType baseType,
                                          const AttributeRefNode *node,
                                          ExprEmitter &emitter) {
  auto attrSpelling = node->attrSpelling;
  auto loc = node->getLoc();
  ASTDecl *typeDecl = baseType.getDecl(emitter.shared);
  if (!typeDecl) {
    emitter.emitError(loc, "MLIR type ") << baseType << " has no attributes";
    return {};
  }

  // Handle __mlir_op.`xxx` references, lazily synthesizing values when
  // they are referenced.
  if (typeDecl->resolvedness == DeclResolvedness::fullyResolved &&
      isa<StructDeclOp>(*typeDecl)) {
    auto resolvedMLIRType = typeDecl->getSelfType().mlirType;
    if (isa<MagicMLIRAttrType>(resolvedMLIRType))
      return {{synthesizeMLIRAttrFromString(attrSpelling, loc, emitter), node}};
    if (isa<MagicMLIROpType>(resolvedMLIRType))
      return {{synthesizeMLIROpFromString(attrSpelling, emitter), node}};
    if (isa<MagicMLIRTypeType>(resolvedMLIRType)) {
      Type result = parseMLIRType(attrSpelling, loc, emitter.shared);
      return {{result ? AnyValue(result) : AnyValue(), node}};
    }
  }

  // Normal member reference.
  return emitDeclMemberAsCallable(*typeDecl, baseType.getParamBindings(),
                                  attrSpelling, node, emitter);
}

AnyValue AttributeRefNode::emitIR(ExprEmitter &emitter,
                                  ASTType contextualType) const {
  return emitCallable(emitter, contextualType).emitAsValue(emitter);
}

/// Emit this expression to MLIR as a CallableValue.  On error, emit an error
/// and return a null value.
CallableValue AttributeRefNode::emitCallable(ExprEmitter &emitter,
                                             ASTType contextualType) const {

  auto baseVal = base->emitIR(emitter);
  if (!baseVal)
    return {};

  // Handle member references on types, like Int.member.
  if (ASTType baseType = baseVal.getIfTypeValue())
    return emitTypeAttributeRef(baseType, this, emitter);

  // Otherwise, it must be an access to a field of a value.  Emit the value as
  // an rvalue.
  ASTType baseRVType = baseVal.getRValueType();
  ASTDecl *typeDecl = baseRVType.getDecl(emitter.shared);
  if (!typeDecl) {
    emitter.emitError(getLoc(), "MLIR type ")
        << ASTType(baseVal.getType()) << " has no attributes";
    return {};
  }

  if (!isa<StructDeclOp>(*typeDecl)) {
    emitter.emitError(getLoc(), "cannot access fields in type ")
        << ASTType(baseVal.getType());
    return {};
  }

  // Find the member being accessed.
  ExprEmitter::LookupResult lookup =
      emitter.lookupAndResolveDecl(attrSpelling, getLoc(), *typeDecl);
  ASTDecl *memberDecl = lookup.getIfSuccess();
  if (!memberDecl) {
    // If the error hasn't been diagnosed, handle it now.
    if (lookup.isFailure())
      emitter.emitError(getLoc(), "object has no attribute '")
          << attrSpelling << "'";

    return {};
  }

  // Handle method references.
  if (auto fnOp = dyn_cast<LIT::FuncOp>(*memberDecl)) {
    // Get a symbol for the underlying function.
    CallableValue fnRef(getLoc(), *memberDecl, baseRVType.getParamBindings());

    // If the callee is a static method, we can directly reference it without
    // binding a self parameter.
    if (fnOp.getIsStatic())
      return fnRef;

    // If this is an instance method, we bind the base value and the symbol
    // together into a callable.
    fnRef.baseVal = {baseVal, base};
    return fnRef;
  }

  if (!emitter.builder) {
    emitter.emitError(getLoc(),
                      "TODO: cannot access member in parameter context");
    return {};
  }

  auto mlirLoc = emitter.translateLocation(getLoc());

  // If the field is a variable, emit a reference to it.
  if (auto fieldOp = dyn_cast<StructFieldOp>(*memberDecl)) {
    // If the base is an lvalue, then we can return an lvalue to the field.
    if (LValue baseLV = baseVal.getIfLValue()) {
      auto fieldPtr =
          emitter.builder->create<StructGEPOp>(mlirLoc, baseLV, fieldOp);
      return {{LValue(fieldPtr), this}};
    }

    // Otherwise, it must be an rvalue.
    // TODO: If this is an MValue, emit as a parameter field access, this
    // would enable `size.value` in things like:
    //
    // fn f[size: Int](a: SomeType[size.value])
    DRValue baseRV = emitter.emitDRValue(baseVal, getLoc());
    if (!baseRV)
      return {};

    return {{DRValue(emitter.builder->create<StructExtractOp>(mlirLoc, baseRV,
                                                              fieldOp)),
             this}};
  }

  emitter.emitError(getLoc(), "reference to unknown member");
  return {};
}

/// Given a call to an UnboundMLIROperator, generate an MLIR operation with
/// the operands as SSA values.
static AnyValue emitMLIROperatorCall(const CallNode &call,
                                     UnboundMLIROperationAttr unboundOp,
                                     ExprEmitter &emitter) {
  auto *context = emitter.getContext();

  if (!emitter.builder) {
    emitter.emitError(call.getLoc(), "cannot emit operation in this context");
    return {};
  }

  // Emit all the arguments so we can encode them as SSA values.
  SmallVector<Value> opOperands;
  for (auto operand : call.args) {
    opOperands.push_back(emitter.emitDRValue(operand));
    if (!opOperands.back())
      return {};
  }

  // Set up the OperationState for the thing we're building.
  OperationState state(emitter.translateLocation(call.getLoc()),
                       unboundOp.getName());
  state.addOperands(opOperands);

  // Process the attributes and figure out the result type if specified.
  bool hadTypeSpec = false;
  for (auto &attr : unboundOp.getAttrs()) {
    if (attr.getName() == "_type") {
      // The value must be a type value, or array thereof.
      if (isa<NoneAttr>(attr.getValue())) {
        // TODO: We don't currently have array attrs for lists, but we use
        // NoneAttr to mark an empty list for operations with no result.
      } else if (auto typedAttr = dyn_cast<TypedAttr>(attr.getValue())) {
        state.types.push_back(MValue(typedAttr).getIfTypeValue());
      } else {
        emitter.emitError(call.getLoc(), "unknown _type value");
        return {};
      }
      hadTypeSpec = true;
      continue;
    }
    state.addAttributes(attr);
  }

  // Finally, if we don't already have a type, figure out the return types
  // using InferTypeOpInterface if the operation is registered and if it is
  // present.
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

  if (!hadTypeSpec) {
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

  // Explicitly run the verifier on the new operation so we make sure to
  // catch problems early.
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
    auto noneMLIRType = LIT::NoneType::get(emitter.getContext());
    return MValue(NoneAttr::get(emitter.getContext(), noneMLIRType));
  }

  assert(resultOp->getNumResults() == 1 &&
         "Only support single result ops so far");

  // Check to see if we can fold this operation.  This enables use of
  // __mlir_op to produce meta-values without forcing them into the dynamic
  // value domain.
  SmallVector<Attribute, 4> constOperands(resultOp->getNumOperands());
  for (unsigned i = 0, e = constOperands.size(); i != e; ++i)
    matchPattern(resultOp->getOperand(i), mlir::m_Constant(&constOperands[i]));
  SmallVector<OpFoldResult, 4> foldResults;
  if (succeeded(resultOp->fold(constOperands, foldResults)) &&
      foldResults.size() == 1) {
    auto folded = PointerUnion<Attribute, Value>(foldResults[0]);
    // If the result was some other value that already exists, use it.
    if (auto val = dyn_cast<Value>(folded)) {
      // FIXME: This should be an assert but pop seems broken:
      // https://github.com/modularml/modular/issues/5162
      if (val.getType() == resultOp->getResult(0).getType()) {
        resultOp->erase();
        return DRValue(val);
      }
    }

    if (auto attr = dyn_cast<TypedAttr>(cast<Attribute>(folded))) {
      // FIXME: This should be an assert but pop seems broken:
      // https://github.com/modularml/modular/issues/5162
      if (attr.getType() == resultOp->getResult(0).getType()) {
        // If it is a constant, make an MAValue result.
        resultOp->erase();
        return MValue(attr);
      }
    }
  }

  // If folding failed, return the operation normally.
  return DRValue(resultOp->getResult(0));
}

AnyValue CallNode::emitIR(ExprEmitter &emitter, ASTType contextualType) const {
  auto calleeVal = callee->emitCallable(emitter, {});
  if (!calleeVal)
    return {};

  // If this is the invocation of an unbound MLIR operator, bind it into an
  // actual operator!
  if (calleeVal.baseVal) {
    if (auto mValue = calleeVal.baseVal.ir.getIfMValue()) {
      if (auto unboundOperator =
              dyn_cast<UnboundMLIROperationAttr>(mValue.get()))
        return emitMLIROperatorCall(*this, unboundOperator, emitter);
    }
  }

  // If the returned RValue is a type value (as in `T()` or `T[123]()`), then
  // this is an invocation of the initializer for the type.
  if (!calleeVal.direct)
    if (ASTType calledType = calleeVal.baseVal.ir.getIfTypeValue())
      calleeVal = emitInitializerCallable(calledType, this, emitter);

  /// Emit a function call for a call node with the specified operands.
  SmallVector<ASTExprAnd<AnyValue>> operands;
  for (ExprNode *arg : args) {
    operands.push_back({arg->emitIR(emitter), arg});
    if (!operands.back())
      return {};
  }
  return emitter.emitFunctionCall(calleeVal, operands, getLoc());
}

AnyValue SliceNode::emitIR(ExprEmitter &emitter, ASTType contextualType) const {
  emitter.emitError(getLoc(), "slice values not implemented yet");
  return {};
}

/// Given a value of type type, substitute parameters into the type, producing
/// a more concrete type.  This syntax is `SomeType[1, 4, Int]`.
static CallableValue substituteParametersIntoUserDefinedType(
    DeclRefType declRef, const SubscriptNode &subscript, ExprEmitter &emitter) {
  // If already parameterized, give up.
  // TODO: Why not allow multiple partial type applications?
  if (!declRef.getParamValues().empty()) {
    emitter.emitError(
        subscript.getLoc(),
        "cannot apply more parameters to an already parameterized type ")
        << ASTType(declRef);
    return {};
  }

  ASTDecl &typeDecl = emitter.shared.getDeclForSymbol(declRef.getSymbol());
  auto structOp = dyn_cast<StructDeclOp>(typeDecl);
  if (!structOp) {
    emitter.emitError(subscript.getLoc(), "unknown parameterized type ")
        << ASTType(declRef);
    return {};
  }

  auto numParams = structOp.getInputParamDecls().size();
  if (numParams != subscript.indices.size()) {
    emitter.emitError(subscript.getLoc(), "")
        << ASTType(declRef) << " requires " << numParams << " meta parameter"
        << plural(numParams) << " but " << subscript.indices.size()
        << " were specified";
    return {};
  }

  // Emit each of the indices as parameter expressions.
  SmallVector<ParamBindAttr> paramBindings;
  for (auto [indexExpr, decl] :
       llvm::zip(subscript.indices, structOp.getInputParamDecls())) {
    // TODO: Slice syntax is the obvious way to support named parameter
    // arguments.
    auto indexVal = emitter.emitMValue(
        indexExpr, "type parameters may not be a run-time value");
    if (!indexVal)
      return {};

    // TODO: Support conversions.
    if (indexVal.getType() != decl.getType()) {
      emitter.emitError(indexExpr->getLoc(), "parameter of type ")
          << ASTType(indexVal.getType())
          << " cannot be converted to expected type "
          << ASTType(decl.getType());
      return {};
    }
    paramBindings.push_back(ParamBindAttr::get(decl, indexVal));
  }

  // Ok, we succeeded at reparameterizing the type.
  return {{MValue(DeclRefType::get(typeDecl.getSymbolRef(), paramBindings)),
           &subscript}};
}

/// Given a set of decomposed types and attributes from an MLIR attribute or
/// type, calculate the post-substitution set of types and attributes that
/// should be used to rebuild the entity.  On failure, this emits an error and
/// returns failure.
static LogicalResult performPlaceholderSubstitution(
    ArrayRef<PointerUnion<Type, Attribute>> params,
    SmallVectorImpl<Attribute> &newAttrs, SmallVectorImpl<Type> &newTypes,
    const SubscriptNode &subscript, ExprEmitter &emitter) {
  unsigned nextIdx = 0;
  for (auto &elt : params) {
    if (auto type = dyn_cast<Type>(elt)) {
      // Types aren't replacable with attributes.
      newTypes.push_back(type);
      continue;
    }

    auto attr = dyn_cast<Attribute>(elt);
    if (!attr) {
      emitter.emitError(subscript.getLoc(),
                        "MLIR substitution has unknown parameter: ")
          << attr;
      return failure();
    }

    auto placeholder = dyn_cast<PlaceholderAttr>(attr);
    if (!placeholder || nextIdx >= subscript.indices.size()) {
      newAttrs.push_back(attr);
      continue;
    }

    ExprNode *indexVal = subscript.indices[nextIdx++];
    TypedAttr newVal = emitter.emitMValue(
        indexVal, "expected meta value in type substitution list");
    if (!newVal)
      return failure();

    // TODO: Support conversions.
    auto expectedType = cast<PlaceholderAttr>(attr).getType();
    if (newVal.getType() != expectedType) {
      emitter.emitError(indexVal->getLoc(), "parameter of type ")
          << ASTType(newVal.getType())
          << " cannot be converted to expected type " << ASTType(expectedType);
      return failure();
    }
    newAttrs.push_back(newVal);
  }

  // Reject extraneous subscript indices.
  if (nextIdx != subscript.indices.size()) {
    emitter.emitError(subscript.indices[nextIdx]->getLoc(),
                      "unused parameter substitution");
    return failure();
  }
  return success();
}

/// Given a value of type type, substitute parameters into the type, producing
/// a more concrete type.  This syntax is `SomeType[1, 4, Int]`.
static CallableValue
substituteParametersIntoMLIRType(Type type, const SubscriptNode &subscript,
                                 ExprEmitter &emitter) {
  auto itf = dyn_cast_or_null<mlir::SubElementTypeInterface>(type);
  if (!itf) {
    emitter.emitError(subscript.getLoc(), "MLIR type ")
        << ASTType(type) << " has no parameters";
    return {};
  }

  // Collect all the attributes and types out of this type.
  SmallVector<PointerUnion<Type, Attribute>> params;
  itf.walkImmediateSubElements([&](Attribute attr) { params.push_back(attr); },
                               [&](Type type) { params.push_back(type); });

  // Figure out the replacements.
  SmallVector<Attribute> newAttrs;
  SmallVector<Type> newTypes;
  if (failed(performPlaceholderSubstitution(params, newAttrs, newTypes,
                                            subscript, emitter)))
    return {};

  // Rewrite the type with the substitutions.
  Type result = itf.replaceImmediateSubElements(newAttrs, newTypes);
  if (!result) {
    emitter.emitError(subscript.getLoc(),
                      "failed to substitute parameters into ")
        << ASTType(type);
    return {};
  }
  return CallableValue({result, &subscript});
}

/// Given an MValue that is being subscripted with a type that cannot be
/// subscripted, check to see if it contains placeholder attributes, and if so
/// substitute new values in for them.  If not, it is not an error, just
/// silently return null.
static TypedAttr substituteParametersIntoMLIRAttr(
    TypedAttr origAttr, const SubscriptNode &subscript, ExprEmitter &emitter) {

  // We can only replace placeholders in iterable attributes.
  auto itf = dyn_cast_or_null<mlir::SubElementAttrInterface>(origAttr);
  if (!itf)
    return {};

  // Collect all the attributes and types out of this type.
  SmallVector<PointerUnion<Type, Attribute>> params;
  bool havePlaceholder = false;
  itf.walkImmediateSubElements(
      [&](Attribute attr) {
        havePlaceholder |= isa<PlaceholderAttr>(attr);
        params.push_back(attr);
      },
      [&](Type type) { params.push_back(type); });

  // If there are no placeholders in the attribute, then we're done.
  if (!havePlaceholder)
    return {};

  // Figure out the replacements.
  SmallVector<Attribute> newAttrs;
  SmallVector<Type> newTypes;
  if (failed(performPlaceholderSubstitution(params, newAttrs, newTypes,
                                            subscript, emitter)))
    return {};

  // Rewrite the type with the substitutions.
  Attribute result = itf.replaceImmediateSubElements(newAttrs, newTypes);
  if (!result || !isa<TypedAttr>(result)) {
    emitter.emitError(subscript.getLoc(),
                      "failed to substitute parameters into ")
        << origAttr;
    return {};
  }
  return cast<TypedAttr>(result);
}

AnyValue SubscriptNode::emitIR(ExprEmitter &emitter,
                               ASTType contextualType) const {
  return emitCallable(emitter, contextualType).emitAsValue(emitter);
}

/// Emit this expression to MLIR as a CallableValue.  On error, emit an error
/// and return a null value.
CallableValue SubscriptNode::emitCallable(ExprEmitter &emitter,
                                          ASTType contextualType) const {

  // Subscripting a generic function binds the parameter expressions.
  auto subValue = base->emitCallable(emitter, {});
  if (!subValue)
    return {};

  // If the subValue has a bound callable symbol, then this is applying (more)
  // meta values to bind its parameters.
  if (subValue.direct) {
    // We can bind additional parameters to a signature.
    SignatureType signature = subValue.direct->type;

    // TODO: For now we just support positional arguments, we could support
    // named arguments in the future.
    SmallVector<ParamBindAttr> bindings(subValue.direct->bindings.getValue());
    size_t numParams = signature.getInputParams().size();

    // Process each subscript entry as a binding.
    for (auto idx : indices) {
      if (bindings.size() >= numParams) {
        emitter.emitError(idx->getLoc(),
                          "too many parameters bound, signature expects ")
            << numParams << " parameter value" << plural(numParams);
        return {};
      }

      ParamDeclAttr decl = signature.getInputParams()[bindings.size()];
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
            << ASTType(val.getType()) << " but declaration expects "
            << ASTType(decl.getType());
        return {};
      }
      bindings.push_back(ParamBindAttr::get(decl.getName(), val));
    }

    // Okay, everything checks out, form the new binding array.
    subValue.direct->bindings =
        ParamBindArrayAttr::get(emitter.getContext(), bindings);
    return subValue;
  }

  // Otherwise, if there is no symbol, it is just an LValue or RValue being
  // subscript.

  // If the sub-value is an unbound Type, try binding things to it!
  if (auto typeValue = subValue.baseVal.ir.getIfTypeValue()) {
    // Handle user-defined types.
    if (auto declRef = dyn_cast<DeclRefType>(typeValue))
      return substituteParametersIntoUserDefinedType(declRef, *this, emitter);

    // Handle __mlir_type types.
    return substituteParametersIntoMLIRType(typeValue, *this, emitter);
  }

  if (auto mValue = subValue.baseVal.ir.getIfMValue()) {
    if (auto unboundOperator = dyn_cast<UnboundMLIROperationAttr>(mValue.get()))
      return {
          {bindAttributesToMLIROperatorCall(*this, unboundOperator, emitter),
           this}};

    if (auto boundAttr =
            substituteParametersIntoMLIRAttr(mValue.get(), *this, emitter))
      return {{MValue(boundAttr), this}};
  }

  // Emit each of the index values to generate error messages.
  SmallVector<RValue> indexValues;
  for (ExprNode *index : indices) {
    indexValues.push_back(emitter.emitRValue(index));
    if (!indexValues.back())
      return {};
  }

  emitter.emitError(getLoc(), "TODO: Subscript irgen not implemented yet ")
      << ASTType(subValue.baseVal.ir.getType());
  return {};
}

AnyValue ParenExprNode::emitIR(ExprEmitter &emitter,
                               ASTType contextualType) const {
  return subExpr->emitIR(emitter, contextualType);
}

AnyValue ListExprNode::emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const {
  SmallVector<RValue> elements;
  for (ExprNode *expr : exprs) {
    elements.push_back(emitter.emitRValue(expr));
    if (!elements.back())
      return {};
  }

  // TODO: If all of these are meta values, produce some typed array constant.
  // We cannot use ArrayAttr here though, because it isn't a TypedAttr.

  // TODO: Form a dynamic array value instead of returning the last element.
  if (!elements.empty())
    return elements.back();

  // TODO: None is the wrong thing, but is useful for now for referring to type
  // arrays used by __mlir_op.
  auto noneType = emitter.shared.getNoneType();
  // TODO: NoneAttr should have a nicer builder.
  auto noneAttr = NoneAttr::get(emitter.getContext(), noneType);
  return MValue(noneAttr);
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

AnyValue BinOpNode::emitIR(ExprEmitter &emitter, ASTType contextualType) const {
  AnyValue lhsRep;
  RValue rhsRep;

  // We generally emit the LHS before the RHS, but need to do special things
  // for an assignment statement.
  if (!isAssignmentStmt()) {
    auto lhsRV = emitter.emitRValue(lhs);
    lhsRep = lhsRV;
    rhsRep = emitter.emitRValue(rhs);
    if (!lhsRep || !rhsRep)
      return {};
  } else {
    // In an assignment, we emit the RHS first as a value and the LHS as an
    // lvalue with a contextual type.  This is required to enable the 'implicit
    // declaration' behavior in a def.
    rhsRep = emitter.emitRValue(rhs);
    if (!rhsRep)
      return {};

    // Emit the LHS pattern as an lvalue.  Pass in the RHS's type as the
    // contextual type in case we need to implicitly declare a variable.
    auto lhsLV = emitter.emitLValue(lhs, rhsRep.getType(),
                                    "cannot assign to immutable expression");
    if (!lhsLV)
      return {};

    // Assignment expression (`=`) turns into a store, not into a method call.
    if (kind == kAssign) {
      auto rv = emitter.emitDRValue(rhsRep, rhs->getLoc());
      if (!rv)
        return {};

      // Check to see if the destination type and the source type are
      // compatible.
      // TODO: Implement implicit conversions.
      if (!lhsLV.getRValueType().isEqualCanon(rv.getType())) {
        emitter.emitError(rhs->getLoc(), "cannot convert value of type ")
            << ASTType(rv.getType()) << " to " << lhsLV.getRValueType();
        return {};
      }

      // If everything worked out, store the resultant value into the lvalue for
      // the destination.  If things didn't work, just drop this on the floor.
      emitter.builder->create<POP::StoreOp>(emitter.translateLocation(getLoc()),
                                            rv, lhsLV,
                                            /*alignment*/ None);
      return rv;
    }

    // Otherwise, handle as a normal binary operator.
    lhsRep = lhsLV;
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnKind = getOpSpecialFunctions(kind);

  // FIXME: We currently hack in index type support as transition to proper
  // expression support.
  if (lhsRep.getType().isIndex() && rhsRep.getType().isIndex()) {
    auto lhsParam =
        emitter.emitMValue(lhs, "expecting parameter values as operands");
    auto rhsParam =
        emitter.emitMValue(rhs, "expecting parameter values as operands");
    // If these are both parameter values, we can fold them using parameter
    // expressions.
    if (!lhsParam || !rhsParam) {
      emitter.emitError(getLoc(), "expecting parameter values as operands");
      return {};
    }
    uint32_t opcode;
    bool needsInvert = false;
    switch (kind) {
    default:
      llvm_unreachable("unknown binary operator");
    case kAdd:
      opcode = (uint32_t)POC::Add;
      break;
    case kMul:
      opcode = (uint32_t)POC::Mul;
      break;
    case kBitwiseAnd:
      opcode = (uint32_t)POC::And;
      break;
    case kBitwiseOr:
      opcode = (uint32_t)POC::Or;
      break;
    case kBitwiseXor:
      opcode = (uint32_t)POC::Xor;
      break;
    case kLeftShift:
      opcode = (uint32_t)POC::Shl;
      break;
    case kRightShift:
      opcode = (uint32_t)POC::Shr;
      break;
    case kFloorDiv:
      opcode = (uint32_t)POC::Div;
      break;
    case kModulo:
      opcode = (uint32_t)POC::Mod;
      break;
    case kCmpEqual:
      opcode = (uint32_t)POC::EQ;
      break;
    case kCmpNotEqual:
      opcode = (uint32_t)POC::EQ;
      needsInvert = true;
      break;
    case kCmpGreaterEqual:
      opcode = (uint32_t)POC::LT;
      needsInvert = true;
      break;
    case kCmpGreater:
      opcode = (uint32_t)POC::LE;
      needsInvert = true;
      break;
    case kCmpLess:
      opcode = (uint32_t)POC::LT;
      break;
    case kCmpLessEqual:
      opcode = (uint32_t)POC::LE;
      break;
    }
    auto value = ParamOperatorAttr::get((POC)opcode, lhsParam, rhsParam);
    if (needsInvert)
      value = ParamOperatorAttr::getNot(value);
    return value;
  }

  assert(specialFnKind != SpecialFunctionKind::kNormal);
  // TODO: Add support for radd, looking up on the RHS.
  ASTExprAnd<AnyValue> argValues[] = {{lhsRep, lhs}, {rhsRep, rhs}};
  return emitter.emitSpecialMethodCall(lhsRep.getRValueType(), specialFnKind,
                                       argValues, getLoc());
}

AnyValue UnaryOpNode::emitIR(ExprEmitter &emitter,
                             ASTType contextualType) const {
  auto exprRep = subExpr->emitIR(emitter);
  if (!exprRep)
    return {};

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnKind = getOpSpecialFunctions(kind);

  if (exprRep.getType().isIndex()) {
    auto exprParam =
        emitter.emitMValue(subExpr, "expecting parameter values as operands");
    if (!exprParam) {
      emitter.emitError(getLoc(), "expecting parameter values as operands");
      return {};
    }
    switch (kind) {
    default:
      llvm_unreachable("unknown binary operator");
    case ExprNode::kUnaryMinus: {
      IntegerAttr minusOne = emitter.builder->getIndexAttr(-1);
      return ParamOperatorAttr::get(POC::Mul, exprParam, minusOne);
    }
    case ExprNode::kUnaryPlus:
      return exprParam;
    }
  }

  assert(specialFnKind != SpecialFunctionKind::kNormal &&
         "Unary operators are implemented via special methods");

  ASTExprAnd<AnyValue> argValue = {exprRep, subExpr};
  return emitter.emitSpecialMethodCall(exprRep.getType(), specialFnKind,
                                       argValue, getLoc());
}

AnyValue IfElseOpNode::emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const {
  auto condValue = emitter.emitConditionValueAsI1(condExpr);
  if (!condValue)
    return {};

  Type dummyType = mlir::IndexType::get(emitter.getContext());
  Location ifLoc = emitter.translateLocation(getLoc());
  // At this point we don't know the type of trueExpr / falseExpr, use
  // a dummy one.
  auto ifOp = emitter.builder->create<scf::IfOp>(ifLoc, TypeRange{dummyType},
                                                 condValue, /*withElse=*/true);
  emitter.builder = ifOp.getThenBodyBuilder();
  DRValue trueVal = emitter.emitDRValue(trueExpr);
  if (!trueVal)
    return {};
  emitter.builder->create<scf::YieldOp>(ifLoc, trueVal);
  emitter.builder = ifOp.getElseBodyBuilder();
  DRValue falseVal = emitter.emitDRValue(falseExpr);
  if (!falseVal)
    return {};
  emitter.builder->create<scf::YieldOp>(ifLoc, falseVal);
  emitter.builder->setInsertionPointAfter(ifOp);
  if (!ASTType(trueVal.getType()).isEqualCanon(falseVal.getType())) {
    emitter.emitError(
        getLoc(), "the types of a conditional expression must be compatible:  ")
        << ASTType(trueVal.getType()) << " is not compatible with "
        << ASTType(falseVal.getType());
    return {};
  }
  // Ensure the correct type is used.
  ifOp->getResult(0).setType(trueVal.getType());
  return DRValue(ifOp.getResult(0));
}
