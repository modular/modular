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
#include "LitExprCalls.h"

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
emitDeclMemberAsCallable(ASTDecl &container, ParamBindArrayAttr bindings,
                         StringRef memberName, const ExprNode *node,
                         ExprEmitter &emitter, ASTType contextualType = {}) {
  // Perform a lookup of the specified decl in the current container.
  LookupResult lookup = emitter.shared.lookupAndResolveDecl(
      memberName, node->getLoc(), container);

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
    emitter.shared.declResolver->addFullyResolvedDecl(varDecl, node->getLoc(),
                                                      nameAttr, &container);
    // Re-do lookup, making sure we form a uniqued vector that we can reference.
    lookup = emitter.shared.lookupAndResolveDecl(memberName, node->getLoc(),
                                                 container);
  }

  ArrayRef<ASTDecl *> decls = lookup.getIfSuccess();
  if (decls.empty()) {
    if (lookup.isFailure()) {
      auto diag = emitter.emitError(node->getLoc());
      if (auto structDecl = dyn_cast<StructDeclOp>(container))
        diag << structDecl.getName() << " has no '" << memberName << "' member";
      else
        diag << "use of unknown declaration \"" << memberName << '"';
    }
    return {};
  }

  // Functions form an address, and may be overloaded.
  if (isa<LIT::FuncOp>(*decls[0]))
    return CallableValue(node->getLoc(), decls, bindings);

  assert(decls.size() == 1 && "Only functions may be overloaded");
  ASTDecl &decl = *decls[0];

  // Variable references resolve to an lvalue addressing the variable.
  if (auto var = dyn_cast<VarDeclOp>(decl))
    return {{AnyValue(LValue(var.getResult())), node}};

  // Parameters form an meta-value.
  if (auto param = dyn_cast<ParamDeclareOp>(decl))
    return {{MValue(ParamDeclRefAttr::get(param.getName(), param.getType())),
             node}};

  // RValue's and LValues always resolve to their known value.
  if (auto rvalue = decl.getIfRValue())
    return {{rvalue, node}};
  if (auto lvalue = decl.getIfLValue())
    return {{lvalue, node}};

  // If this is a type declaration, return it as a type.
  if (isa<StructDeclOp>(decl))
    return {{MValue(DeclRefType::get(decl.getSymbolRef())), node}};

  emitter.emitError(node->getLoc(), "use of declaration \"")
      << memberName << "\" as a value isn't supported yet";
  return {};
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
  return emitDeclMemberAsCallable(
      emitter.declScope,
      /*no param bindings*/ ParamBindArrayAttr::get(emitter.getContext(), {}),
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

  auto baseVal = base->emitIR(emitter, /*No Contextual Type*/ {});
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
  LookupResult lookup =
      emitter.shared.lookupAndResolveDecl(attrSpelling, getLoc(), *typeDecl);
  ArrayRef<ASTDecl *> memberDecls = lookup.getIfSuccess();
  if (memberDecls.empty()) {
    // If the error hasn't been diagnosed, handle it now.
    if (lookup.isFailure())
      emitter.emitError(getLoc(), "object has no attribute '")
          << attrSpelling << "'";

    return {};
  }

  // Handle method references, which might be overloaded.
  if (auto fnOp = dyn_cast<LIT::FuncOp>(*memberDecls[0])) {
    // Get a symbol for the underlying function.
    CallableValue fnRef(getLoc(), memberDecls, baseRVType.getParamBindings());

    // If the callee is a static method, we can directly reference it without
    // binding a self parameter.  If this is an instance method, we bind the
    // base value and the symbol together into a callable.
    // FIXME: This isn't handling overloaded static/non-static methods
    // correctly.  What is the actual behavior we want for static methods?
    // Maybe we don't allow overloading static and non-static methods with the
    // same name?
    if (!fnOp.getIsStatic())
      fnRef.baseVal = {baseVal, base};
    return fnRef;
  }
  assert(memberDecls.size() == 1 && "only methods may be overloaded");
  ASTDecl &memberDecl = *memberDecls[0];

  if (!emitter.builder) {
    emitter.emitError(getLoc(),
                      "TODO: cannot access member in parameter context");
    return {};
  }

  auto mlirLoc = emitter.translateLocation(getLoc());

  // If the field is a variable, emit a reference to it.
  if (auto fieldOp = dyn_cast<StructFieldOp>(memberDecl)) {
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
      // FIXME(Issue#5162): This should be an assert but pop seems broken:
      // https://github.com/modularml/modular/issues/5162
      if (val.getType() == resultOp->getResult(0).getType()) {
        resultOp->erase();
        return DRValue(val);
      }
    }

    if (auto attr = dyn_cast<TypedAttr>(cast<Attribute>(folded))) {
      // FIXME(Issue#5162): This should be an assert but pop seems broken:
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
    if (ASTType calledType = calleeVal.baseVal.ir.getIfTypeValue()) {
      bool isErroneousDecl = false;
      calleeVal = CallableValue(calledType, "__new__", getLoc(),
                                /*emitErrorOnFailure=*/true, isErroneousDecl,
                                emitter.shared);
      if (calleeVal.isNull())
        return {};
    }

  /// Emit a function call for a call node with the specified operands.
  SmallVector<ASTExprAnd<AnyValue>> operands;
  for (ExprNode *arg : args) {
    operands.push_back({arg->emitIR(emitter, /*No Contextual Type*/ {}), arg});
    if (!operands.back())
      return {};
  }
  return calleeVal.emitFunctionCall(operands, getLoc(), emitter);
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

  ASTDecl &typeDecl =
      emitter.shared.declResolver->getDeclForTypeSymbol(declRef.getSymbol());
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
  return CallableValue(
      {MValue(DeclRefType::get(typeDecl.getSymbolRef(), paramBindings)),
       &subscript});
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

  // If the subValue has a bound callable symbol, then this is applying (more?)
  // meta values to bind its parameters.
  if (subValue.direct) {
    // Process each subscript entry as a binding.
    // TODO: Support named bindings in addition to positional ones: `A[x: 42]`.
    for (auto idx : indices) {
      auto val = emitter.emitMValue(
          idx, "declaration parameters may not be a run-time value");
      if (!val)
        return {};

      // We don't do any checking to see if the value is compatible with the
      // expected type - this is deferred until when the symbol is actually
      // emitted for something.  This allow us to use the provided parameters to
      // filter down the overload set.
      //
      // Note: we're being a bit abusive here by making a ParamBindAttr with a
      // null name for positional attributes.
      subValue.direct->bindings.push_back(
          {idx->getLoc(), Attribute(val.get())});
    }
    // The bindings will be checked for validity when a reference is formed.
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

CallableValue ParenExprNode::emitCallable(ExprEmitter &emitter,
                                          ASTType contextualType) const {
  return subExpr->emitCallable(emitter, contextualType);
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
static SpecialFunctionKind getOpSpecialFunctions(ExprNode::Kind kind,
                                                 bool isReversed) {

  // Use an if chain to find the right match.  We can't use switch here because
  // multiple special functions may implement the same kind, e.g. __add__ and
  // __radd__ special methods both implement kAdd.
#define SF(ENUM, NAME, NUMOPERANDS, EXPRNODE, FLAGS)                           \
  if (kind == ExprNode::Kind::EXPRNODE &&                                      \
      SpecialFunctionInfo::get(SpecialFunctionKind::ENUM).isReversed() ==      \
          isReversed)                                                          \
    return SpecialFunctionKind::ENUM;                                          \
  else
#include "SpecialFunctions.def"
  // If everything fails we should return "normal".
  return SpecialFunctionKind::kNormal;
}

AnyValue BinOpNode::emitIR(ExprEmitter &emitter, ASTType contextualType) const {
  AnyValue lhsRep;
  RValue rhsRep;

  // We generally emit the LHS before the RHS, but need to do special things
  // for short-circuiting and assignment statements.
  if (kind == kBoolAnd || kind == kBoolOr) // `x and y`, `x or y`
    return emitAndOr(emitter);

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
      // Emit the RHS and coerce to the LHS type.
      auto rv = emitter.emitDRValue(rhsRep, rhs->getLoc());
      rv = emitter.getAsExpectedType(rv, rhs, lhsLV.getRValueType());
      if (!rv)
        return {};

      // If everything worked out, store the resultant value into the lvalue for
      // the destination.  If things didn't work, just drop this on the floor.
      emitter.builder->create<POP::StoreOp>(emitter.translateLocation(getLoc()),
                                            rv, lhsLV,
                                            /*alignment=*/std::nullopt);
      return rv;
    }

    // Otherwise, handle as a normal binary operator.
    lhsRep = lhsLV;
  }

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
    case kAnd:
      opcode = (uint32_t)POC::And;
      break;
    case kOr:
      opcode = (uint32_t)POC::Or;
      break;
    case kXor:
      opcode = (uint32_t)POC::Xor;
      break;
    case kLShift:
      opcode = (uint32_t)POC::Shl;
      break;
    case kRShift:
      opcode = (uint32_t)POC::Shr;
      break;
    case kFloorDiv:
      opcode = (uint32_t)POC::Div;
      break;
    case kMod:
      opcode = (uint32_t)POC::Mod;
      break;
    case kCmpEQ:
      opcode = (uint32_t)POC::EQ;
      break;
    case kCmpNE:
      opcode = (uint32_t)POC::EQ;
      needsInvert = true;
      break;
    case kCmpGE:
      opcode = (uint32_t)POC::LT;
      needsInvert = true;
      break;
    case kCmpGT:
      opcode = (uint32_t)POC::LE;
      needsInvert = true;
      break;
    case kCmpLT:
      opcode = (uint32_t)POC::LT;
      break;
    case kCmpLE:
      opcode = (uint32_t)POC::LE;
      break;
    }
    auto value = ParamOperatorAttr::get((POC)opcode, lhsParam, rhsParam);
    if (needsInvert)
      value = ParamOperatorAttr::getNot(value);
    return value;
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnKind = getOpSpecialFunctions(kind, /*isReversed=*/false);
  assert(specialFnKind != SpecialFunctionKind::kNormal);
  ASTExprAnd<AnyValue> argValues[] = {{lhsRep, lhs}, {rhsRep, rhs}};

  // Check to see if we have a forward version of this function on the primary
  // receiver.
  bool isErroneousDecl = false;
  CallableValue callee(lhsRep.getRValueType(),
                       SpecialFunctionInfo::get(specialFnKind).name, getLoc(),
                       /*emitErrorOnFailure=*/false, isErroneousDecl,
                       emitter.shared);
  if (callee.direct &&
      succeeded(callee.direct->filterOverloadSet(
          argValues, /*isMethodSyntax*/ false,
          /*emitDiagnosticOnFailure=*/false, emitter.shared))) {
    return callee.emitFunctionCall(argValues, getLoc(), emitter);
  }
  if (isErroneousDecl)
    return {};

  // Check to see if we have the reverse version of this operator.
  auto reversedFnKind = getOpSpecialFunctions(kind, /*isReversed=*/true);
  if (reversedFnKind != SpecialFunctionKind::kNormal) {
    // Swap the operand order.
    std::swap(argValues[0], argValues[1]);
    callee = CallableValue(
        rhsRep.getType(), SpecialFunctionInfo::get(reversedFnKind).name,
        getLoc(),
        /*emitErrorOnFailure=*/false, isErroneousDecl, emitter.shared);
    if (callee.direct &&
        succeeded(callee.direct->filterOverloadSet(
            argValues, /*isMethodSyntax*/ false,
            /*emitDiagnosticOnFailure=*/false, emitter.shared))) {
      return callee.emitFunctionCall(argValues, getLoc(), emitter);
    }

    // Swap these back so we emit the right error.
    std::swap(argValues[0], argValues[1]);
  }

  // Emit an error complaining about the forward version of the operator.
  return emitter.emitSpecialMethodCall(lhsRep.getRValueType(), specialFnKind,
                                       argValues, getLoc());
}

/// This method emits the `x and y`, `x or y` operators.  These are interesting
/// in Python:
///
///   "Note that neither `and` nor `or` restrict the value and type they return
///   to False and True, but rather return the last evaluated argument. This is
///   sometimes useful, e.g., if `s` is a string that should be replaced by a
///   default value if it is empty, the expression `s or 'foo'` yields the
///   desired value.
///
/// Unlike Python, we have static types that could disagree.  Our policy on this
/// is to either return the pre-Bool'ified value when their types agree, or to
/// return the common Bool type if they don't.
///
/// TODO(subtyping): With subtypes, we can find intersection types, e.g. a
/// common superclass.
///
AnyValue BinOpNode::emitAndOr(ExprEmitter &emitter) const {
  Location ifLoc = emitter.translateLocation(getLoc());

  if (!emitter.builder) {
    emitter.emitError(getLoc(), "cannot emit operation in this context");
    return {};
  }

  // Emit the LHS value and capture the result of calling __bool__ in case we
  // need it.
  AnyValue lhsBool;
  DRValue lhsRV = emitter.emitDRValue(lhs);
  auto lhsI1Value = emitter.emitConditionValueAsI1({lhsRV, lhs}, lhsBool);
  if (!lhsI1Value)
    return {};

  auto ifOp = emitter.builder->create<scf::IfOp>(
      ifLoc, TypeRange{lhsBool.getType()}, lhsI1Value, /*withElse=*/true);

  OpBuilder trueBuilder = ifOp.getThenBodyBuilder();
  OpBuilder falseBuilder = ifOp.getElseBodyBuilder();
  if (kind == kBoolOr) // and/or just treat the bool differently.
    std::swap(trueBuilder, falseBuilder);

  emitter.builder = trueBuilder;
  DRValue rhsRV = emitter.emitDRValue(rhs);
  if (!rhsRV)
    return {};

  // Now that we know lhsRV and rhsRV we can tell if they have common types.
  // If so, we use that as the result of the 'if'.
  if (ASTType(lhsRV.getType()).isEqualCanon(rhsRV.getType())) {
    emitter.builder->create<scf::YieldOp>(ifLoc, rhsRV);
    // Emit the false side.
    emitter.builder = falseBuilder;
    emitter.builder->create<scf::YieldOp>(ifLoc, lhsRV);
    ifOp->getResult(0).setType(lhsRV.getType());
  } else {
    // Otherwise, check to see if their boolean versions are compatible.
    auto rhsBool = emitter.emitSpecialMethodCall(rhsRV.getType(),
                                                 SpecialFunctionKind::kBool,
                                                 {{rhsRV, rhs}}, rhs->getLoc());
    if (!ASTType(lhsBool.getType()).isEqualCanon(rhsBool.getType())) {
      emitter.emitError(getLoc(), "cannot find common type between ")
          << ASTType(lhsRV.getType()) << " and " << ASTType(rhsRV.getType());
      return {};
    }
    auto rhsBoolDRVal = emitter.emitDRValue(rhsBool, rhs->getLoc());
    if (!rhsBoolDRVal)
      return {};
    emitter.builder->create<scf::YieldOp>(ifLoc, rhsBoolDRVal);
    // Emit the false side.
    emitter.builder = falseBuilder;
    auto lhsBoolDRVal = emitter.emitDRValue(lhsBool, lhs->getLoc());
    if (!lhsBoolDRVal)
      return {};
    emitter.builder->create<scf::YieldOp>(ifLoc, lhsBoolDRVal);
    ifOp->getResult(0).setType(lhsBool.getType());
  }

  emitter.builder->setInsertionPointAfter(ifOp);
  return DRValue(ifOp.getResult(0));
}

AnyValue UnaryOpNode::emitIR(ExprEmitter &emitter,
                             ASTType contextualType) const {
  auto exprRep = subExpr->emitIR(emitter, /*No Contextual Type*/ {});
  if (!exprRep)
    return {};

  // Special case some things for literals.
  // TODO: Fix literal representation.
  if (exprRep.getType().isIndex() || exprRep.getType().isF64()) {
    auto exprParam =
        emitter.emitMValue(subExpr, "expecting parameter values as operands");
    if (!exprParam) {
      emitter.emitError(getLoc(), "expecting parameter values as operands");
      return {};
    }
    switch (kind) {
    default:
      break;
    case ExprNode::kNeg:
      if (auto constantFP = dyn_cast<FloatAttr>(exprParam.get()))
        return MValue(
            FloatAttr::get(constantFP.getType(), -constantFP.getValue()));

      // Support general integer parameter exprss.
      if (exprRep.getType().isIndex()) {
        IntegerAttr minusOne = emitter.builder->getIndexAttr(-1);
        return ParamOperatorAttr::get(POC::Mul, exprParam, minusOne);
      }
      break;
    case ExprNode::kPos:
      return exprParam;
    }
  }

  ASTExprAnd<AnyValue> argValue = {exprRep, subExpr};
  Kind kindToEmit = kind;

  // Handle special cases that don't correspond to special function, "not x".
  if (kindToEmit == kBoolNot) {
    // Turn this into a call to __bool__.
    argValue.ir = emitter.emitSpecialMethodCall(
        exprRep.getType(), SpecialFunctionKind::kBool, argValue, getLoc());
    if (!argValue.ir)
      return {};
    // Now that we know we bool-ized the expression, invert it with ~.
    kindToEmit = kInvert;
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnKind = getOpSpecialFunctions(kindToEmit, /*isReversed=*/false);
  assert(specialFnKind != SpecialFunctionKind::kNormal &&
         "Unary operators are implemented via special methods");

  return emitter.emitSpecialMethodCall(argValue.ir.getType(), specialFnKind,
                                       argValue, getLoc());
}

AnyValue IfElseOpNode::emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const {
  auto condValue = emitter.emitConditionValueAsI1(condExpr);
  if (!condValue)
    return {};

  if (!emitter.builder) {
    emitter.emitError(getLoc(), "cannot emit operation in this context");
    return {};
  }

  Location ifLoc = emitter.translateLocation(getLoc());
  // At this point we don't know the type of trueExpr / falseExpr, use
  // a dummy one and fix it later.
  auto ifOp = emitter.builder->create<scf::IfOp>(
      ifLoc, TypeRange{condValue.getType()}, condValue, /*withElse=*/true);
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

  /// TODO(subtyping): With subtypes, we can find intersection types, e.g. a
  /// common superclass.
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
