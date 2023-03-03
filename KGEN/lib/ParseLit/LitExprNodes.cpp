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
#include "LitExprEmitter.h"
#include "LitParameterEvaluator.h"
#include "LitSharedState.h"
#include "SpecialFunctions.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Given a StringRef for an MLIR attribute, invoke the MLIR parser to resolve
/// it into an Attribute (which may not be a TypedAttr) and return it.  On
/// error, emit a diagnostic and return null.
static Attribute parseMLIRAttrFromString(StringRef name, SMLoc loc,
                                         LitSharedState &shared) {
  Attribute result;
  std::string errorMsg;
  {
    // Capture errors thrown by parseAttribute and ignore them.
    // FIXME: This doesn't silence errors!
    mlir::ScopedDiagnosticHandler handler(shared.getContext(),
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
                                  shared.getContext());
  }
  if (!result) {
    shared.emitError(loc, "invalid MLIR attribute: ") << errorMsg;
    return {};
  }

  return result;
}

/// This implements __mlir_attr.x lookup, synthesizing a MAValue for the
/// attribute on demand.
static PRValue synthesizeMLIRAttrFromString(StringRef name, SMLoc loc,
                                            LitSharedState &shared) {
  auto attr = parseMLIRAttrFromString(name, loc, shared);
  if (!attr)
    return {};

  auto typedAttr = dyn_cast<TypedAttr>(attr);
  if (!typedAttr) {
    SmallString<128> str;
    llvm::raw_svector_ostream os(str);
    attr.print(os);
    shared.emitError(loc, "MLIR attribute is not a TypedAttr: ") << os.str();
    return {};
  }
  return PRValue(typedAttr);
}

/// Given an __mlir_type[a,b,c] or __mlir_attr[a,b,c] usage, stringize the
/// indices and return the result.  On error, emit an error and return an empty
/// string.
static std::string substituteMLIRMagic(const SubscriptNode &node,
                                       ExprEmitter &emitter) {
  std::string result;
  llvm::raw_string_ostream os(result);

  for (auto *indexExpr : node.indices) {
    // If the index is an identifer, and if it is a backtick identifier, we
    // treat it as an interpolated literal string.  Otherwise we look it up as
    // an expression.  Rationale: this allows using strings attributes, which
    // could be useful someday, and keeps __mlir_attr.`thing` more consistent
    // with __mlir_attr[`thing`].
    if (auto *dre = dyn_cast<DeclRefNode>(indexExpr))
      if (dre->spelling.data()[dre->spelling.size()] == '`') {
        os << dre->spelling;
        continue;
      }

    // As a very special hack, we treat a unary plus as a marker that the type
    // should not be printed when the attribute is stringized.
    bool elideType = false;
    if (indexExpr->kind == ExprNode::kPos) {
      elideType = true;
      indexExpr = cast<UnaryOpNode>(indexExpr)->subExpr;
    }

    auto indexVal =
        emitter.emitExprPRValue(indexExpr, ASTType(), " in MLIR magic");
    if (!indexVal)
      return "";

    // If this is a wrapper for a type, print it as such.
    if (auto typeVal = indexVal.getIfTypeValue())
      os << typeVal.mlirType;
    else // Otherwise print it as an attribute.
      indexVal.get().print(os, elideType);
  }

  if (result.empty())
    emitter.emitError(node.getLoc(), "mlir magic expanded to an empty string");
  return result;
}

/// When a lookup in __mlir_op fails for a named field, this method tries to
/// resolve it.  On success, it lazily creates a resolved declaration.  On
/// failure, it bails out.
static PRValue synthesizeMLIROpFromString(StringRef name,
                                          ExprEmitter &emitter) {
  auto *context = emitter.getContext();
  auto nameStr = StringAttr::get(context, name);

  auto result = UnboundMLIROperationAttr::get(
      context, nameStr.getType(), nameStr, DictionaryAttr::get(context));
  return PRValue(result);
}

/// Calculate the result of an __mlir_op.`thing`[attributes], applying the
/// attributes list to the operation specification.
static PRValue
bindAttributesToMLIROperatorCall(const SubscriptNode &subscript,
                                 UnboundMLIROperationAttr unboundOp,
                                 ExprEmitter &emitter) {
  auto loc = subscript.getLoc();
  auto *context = emitter.getContext();

  // Only allow applying attributes to something without them.
  if (!unboundOp.getAttrs().empty()) {
    emitter.emitError(loc, "operation already has attributes")
        << subscript.getRange();
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
                                       emitter.shared);
    }

    // Likewise, special case the __mlir_attr[a,b,c] syntax to support
    // attributes without types.
    if (auto subscript = dyn_cast<SubscriptNode>(node)) {
      auto mlirAttr = dyn_cast<DeclRefNode>(subscript->base);
      if (mlirAttr && mlirAttr->spelling == "__mlir_attr") {
        std::string result = substituteMLIRMagic(*subscript, emitter);
        if (result.empty())
          return {};
        return parseMLIRAttrFromString(result, subscript->getLoc(),
                                       emitter.shared);
      }
    }

    // Otherwise emit the value as an MAValue.  This allows references to
    // parameter expressions.
    PRValue value = emitter.emitExprPRValue(
        node, ASTType(), " in value for '" + Twine(name) + "' attribute");
    return value.get();
  };

  SmallVector<NamedAttribute> attrValues;

  // Each element of the subscript must have a name identifier and a value as an
  // MAValue.
  for (auto *subscriptIdx : subscript.indices) {
    auto *slice = dyn_cast<SliceNode>(subscriptIdx);
    if (!slice || slice->colon2Loc.isValid() || !slice->lower ||
        !slice->upper || !isa<DeclRefNode>(slice->lower)) {
      emitter.emitError(
          loc, "attribute spec requires an attribute name and attr value")
          << subscriptIdx->getRange();
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
    emitter.emitError(loc, "attribute ")
        << duplicate->getName() << " redundantly specified"
        << subscript.getIndexRange();
    return {};
  }

  // Return it.
  auto attrs = DictionaryAttr::get(context, attrValues);
  return PRValue(UnboundMLIROperationAttr::get(context, unboundOp.getType(),
                                               unboundOp.getName(), attrs));
}

/// Given a ParamDeclareOp, return the value that should be used in a reference
/// to it.  This currently fully substitutes members unless they are in a
/// function definition.
static PRValue resolveParamDeclareValue(ParamDeclareOp param,
                                        ParamBindArrayAttr bindings,
                                        LitSharedState &shared) {
  // If the param is declared in a function, then just directly use it.
  Operation *parent = param->getParentOp();
  while (1) {
    // If this reference is within a function then keep it symbolic.
    if (parent && isa<LIT::FuncOp>(parent))
      return PRValue(ParamDeclRefAttr::get(param.getName(), param.getType()));
    // If this is at file scope, inline it.
    if (!parent || isa<FileModuleOp>(parent))
      return param.getValue();

    // If this is in a struct, then the value may refer to parameters declared
    // on the struct, whose values come through 'bindings'.  Remap.
    if (auto structDecl = dyn_cast<StructDeclOp>(parent)) {
      // If the reference is to a member of the struct that has bindings, remap
      // them.  This allows things like `SomeType[a,b].someAlias` to substitute
      // the a/b values into the body of `someAlias`.  If we have no bindings,
      // then we know we're in a context where the body of the alias is still
      // valid.
      if (!bindings)
        return param.getValue();

      assert(structDecl.getInputParamDecls().size() == bindings.size() &&
             "mismatch in # struct parameters and # bindings");

      LitParameterEvaluator evaluator(*shared.declResolver, bindings);
      auto result = evaluator.getReboundAttribute(param.getValue());
      return PRValue(cast<TypedAttr>(result));
    }

    // Ignore if and other control flow things.
    parent = parent->getParentOp();
  }

  return PRValue(ParamDeclRefAttr::get(param.getName(), param.getType()));
}

//===----------------------------------------------------------------------===//
// ExprNode Implementation
//===----------------------------------------------------------------------===//

ExprNode::~ExprNode() { llvm_unreachable("never called"); }

/// Return the start or end of the source range.
llvm::SMLoc ExprNode::getRangeStart() const { return getRange().getStart(); }
llvm::SMLoc ExprNode::getRangeEnd() const { return getRange().getEnd(); }

AnyValue ExprNode::emitExprResultIntoPattern(ASTExprAnd<AnyValue> value,
                                             ExprEmitter &emitter) const {
  // Emit this node to see if it is a general LValue.
  AnyValue aValue = emitIR(emitter, ValueDest());
  if (!aValue)
    return {};
  LValue lValue = aValue.getIfLValue();
  if (!lValue) {
    emitter.emitError(getLoc(), "cannot assign to immutable expression")
        << getRange();
    return {};
  }

  // If we got an lvalue, we can try to emit into it.
  return emitter.emitExprResultIntoLValue(value, lValue);
}

/// Return the 'loc' for this node translated to an MLIR location.
Location ExprNode::getLocation(ExprEmitter &emitter) const {
  return emitter.translateLocation(getLoc());
}

//===----------------------------------------------------------------------===//
// ExprNode implementations
//===----------------------------------------------------------------------===//

AnyValue IntLiteralNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  // TODO: Handle contextual types.
  APInt value = LitLexer::getIntegerLiteralValue(spelling);

  // Make sure the value fits in 64-bits.  There are no negative values here.
  // TODO: Detect overflow errors.
  // TODO: Switch to builtin.IntegerLiteralType.
  value = value.zextOrTrunc(64);
  auto attr = IntegerAttr::get(IndexType::get(emitter.getContext()), value);
  return emitter.emitResult(attr, this, dest);
}

AnyValue FloatLiteralNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  // TODO: this assumes float literal are always doubles
  APFloat value = LitLexer::getFloatLiteralValue(spelling);
  auto attr = FloatAttr::get(FloatType::getF64(emitter.getContext()),
                             APFloat(value.convertToDouble()));
  // FIXME: This should eventually use a float literal type.
  // when we support conversions.
  return emitter.emitResult(attr, this, dest);
}

AnyValue BoolLiteralNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  ASTDecl &compilerBuiltinDecl = emitter.shared.getCompilerBuiltInDecl();
  LookupResult lookup = emitter.shared.lookupAndResolveDecl(
      "BoolLiteral", getLoc(), compilerBuiltinDecl,
      /*searchParentScopes=*/true);

  //  BoolLiteral must be in scope since it is auto-imported.
  assert(!lookup.isFailure() && !lookup.getIfSuccess().empty());
  ASTDecl &decl = *lookup.getIfSuccess()[0];
  assert(isa<StructDeclOp>(decl));
  mlir::MLIRContext *ctx = emitter.getContext();
  auto boolLiteralDeclType = DeclRefType::get(decl.getSymbolRef());
  bool isErroneousDecl = false;
  OverloadSet newVal(boolLiteralDeclType, "__new__", this, isErroneousDecl,
                     emitter.shared);
  assert(!newVal.isNull() && "__new__ should be always there by construction");
  auto boolDType = DTypeConstantAttr::get(ctx, DType::kBool);
  auto boolAttr = POP::SIMDAttr::get({value, KGENDType::kBool},
                                     POP::SIMDType::get(1, boolDType));
  return newVal.emitCall(ASTExprAnd<AnyValue>{AnyValue(boolAttr), this}, dest,
                         this, CallSyntax::kTypeCall, emitter);
}

AnyValue SelfLiteralNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  // Self resolves to the type of the enclosing structure type.
  ASTDecl *structDecl = &emitter.declScope;
  while (!isa<StructDeclOp>(*structDecl)) {
    structDecl = structDecl->getParentDecl();
    if (!structDecl) {
      emitter.emitError(getLoc(), "'Self' type may only be used inside a type");
      return {};
    }
  }

  // FIXME(Issue#5975): Verify the struct's signature is resolved.  This should
  // go away when the FIXME at the top of lookupAndResolveDecl is resolved.
  if (failed(emitter.getDeclResolver().resolve(
          *structDecl, DeclResolvedness::signature, getLoc())))
    return {};

  // Once we have the type in question we can just return its Self type as an
  // PRValue.  This already includes bound parameters etc.
  return emitter.emitResult(structDecl->getSelfType(), this, dest);
}

AnyValue StringLiteralNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  std::string value = LitLexer::getStringLiteralValue(spelling);
  auto attr =
      StringAttr::get(value, KGEN::StringType::get(emitter.getContext()));
  return emitter.emitResult(attr, this, dest);
}

AnyValue NoneLiteralNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  return emitter.emitResult(NoneAttr::get(emitter.getContext()), this, dest);
}

/// This handles emission of a value into a DeclRefNode as a pattern.
AnyValue DeclRefNode::emitExprResultIntoPattern(ASTExprAnd<AnyValue> value,
                                                ExprEmitter &emitter) const {
  ASTDecl &container = emitter.declScope;

  // Perform a lookup of the specified decl in the current container.
  LookupResult lookup = emitter.shared.lookupAndResolveDecl(
      spelling, getLoc(), container, /*searchParentScopes=*/true);

  auto createVarDeclWithValueType = [&](OpBuilder &builder) -> VarDeclOp {
    Type declIRType = POP::PointerType::get(value.ir.getRValueType());
    auto loc = getLocation(emitter);
    auto nameAttr = StringAttr::get(loc.getContext(), spelling);
    return builder.create<VarDeclOp>(loc, declIRType, nameAttr);
  };

  auto finishLValue = [&](LValue lvalue) -> AnyValue {
    return emitter.emitExprResultIntoLValue(value, lvalue);
  };

  // If the unresolved name is `_`, then we have a discard pattern.  Just
  // materialize the value into a dynamic representation and return that value
  // without storing into the discard.
  if (lookup.isFailure() && spelling == "_" && emitter.builder)
    // TODO(memory-primary): don't force into SSA value.
    return emitter.emitSRValue(value);

  // If that lookup failed, but we can synthesize a variable declaration in this
  // scope, do that.  We can only do this if there is a varDeclCursor,
  // indicating that we're in a `def` node.
  if (lookup.isFailure() && emitter.varDeclCursor) {
    // Use this builder to place any VarDeclOps. In Python there is only one
    // scope per function and all variables belong to that scope, so builders
    // should reflect that.
    OpBuilder varDeclBuilder(emitter.varDeclCursor);
    auto varDecl = createVarDeclWithValueType(varDeclBuilder);

    // In a normal implicit declaration, we add it to the name table so
    // subsequent uses find this one.
    emitter.getDeclResolver().addFullyResolvedDecl(
        varDecl, getLoc(), varDecl.getNameAttr(), &container);
    return finishLValue(varDecl.getResult());
  }

  ArrayRef<ASTDecl *> decls = lookup.getIfSuccess();
  if (decls.empty()) {
    if (lookup.isErroneous())
      return {}; // Error already diagnosed.

    // By policy in order to produce a more predictable programming model,
    // implicit declarations of variables are only allowed in `def` contexts,
    // not in `fn`, structs, or top level.
    auto funcContext =
        dyn_cast_or_null<LIT::FuncOp>(emitter.declScope.getIfOperation());
    if (!funcContext || !funcContext.getIsDef()) {
      auto diag = emitter.emitError(getLoc()) << "use of unknown declaration '"
                                              << spelling << "'" << getRange();
      if (funcContext)
        diag << ", `fn` declarations require explicit variable declarations";
      return {};
    }

    auto diag = emitter.emitError(getLoc()) << getRange();
    diag << "use of unknown declaration '" << spelling << "'";
    return {};
  }

  ASTDecl &decl = *decls[0];

  // Variable references resolve to an lvalue addressing the variable.
  if (auto var = dyn_cast<VarDeclOp>(decl))
    return finishLValue(var.getResult());

  if (auto lvalue = decl.getIfLValue())
    return finishLValue(lvalue);

  // Reject unqualified struct field references.
  if (auto fieldOp = dyn_cast<StructFieldOp>(decl)) {
    emitter.emitError(getLoc(), "cannot access instance field '")
        << spelling << "' directly; did you mean `self.`?" << getRange();
    return {};
  }

  emitter.emitError(getLoc(), "cannot assign to declaration '")
      << spelling << "', it isn't a mutable value" << getRange();
  return {};
}

/// Emit IR for an unqualified declaration reference "x" looked up in current
/// context.
AnyValue DeclRefNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  ASTDecl &container = emitter.declScope;

  // Perform a lookup of the specified decl in the current container.
  LookupResult lookup = emitter.shared.lookupAndResolveDecl(
      spelling, getLoc(), container, /*searchParentScopes=*/true);

  ArrayRef<ASTDecl *> decls = lookup.getIfSuccess();
  if (decls.empty()) {
    if (lookup.isErroneous())
      return {}; // Error already diagnosed.

    // By policy in order to produce a more predictable programming model,
    // implicit declarations of variables are only allowed in `def` contexts,
    // not in `fn`, structs, or top level.
    auto funcContext =
        dyn_cast_or_null<LIT::FuncOp>(emitter.declScope.getIfOperation());
    if (!funcContext || !funcContext.getIsDef()) {
      auto diag = emitter.emitError(getLoc()) << "use of unknown declaration '"
                                              << spelling << "'" << getRange();
      if (funcContext)
        diag << ", `fn` declarations require explicit variable declarations";
      return {};
    }

    auto diag = emitter.emitError(getLoc()) << getRange();
    if (auto structDecl = dyn_cast<StructDeclOp>(container))
      diag << structDecl.getName() << " has no '" << spelling << "' member";
    else
      diag << "use of unknown declaration '" << spelling << "'";
    return {};
  }

  // Functions form an address, and may be overloaded.
  if (isa<LIT::FuncOp>(*decls[0])) {
    // When unqualified name lookup finds a method on a struct, we bind in the
    // parameters from the enclosing struct.
    ParamBindArrayAttr paramBindings;
    if (isa_and_nonnull<StructDeclOp>(*decls[0]->getParentDecl())) {
      auto structDeclType = decls[0]->getParentDecl()->getSelfType();
      paramBindings = structDeclType.getParamBindings();
    }

    // Form an overload set value with all the candidates.
    auto result = ORValue::create(spelling, decls, paramBindings);
    return emitter.emitResult(std::move(result), this, dest);
  }

  assert(decls.size() == 1 && "Only functions may be overloaded");
  ASTDecl &decl = *decls[0];

  // Let declarations resolve to an rvalue.
  if (auto letDecl = dyn_cast<LetDeclOp>(decl)) {
    // TODO: Loading a 'let' value into an RValue is a semantic copy of the
    // underlying value.  Unfortunately, we don't have a proper notion of
    // rvalue references / borrows yet (they will come with a more baked out
    // ownership model).  Thus the __clone__ operation takes a mutable
    // reference as input, which cannot bind to a let value.
    //
    // As a stop-gap-for-now, just check to see if the value is copyable.  We
    // cannot actually invoke the __clone__ operation if it exists, but at
    // least we can ban copying of non-copyable values.
    ASTType letType(letDecl.getType());
    if (ASTDecl *letTypeDecl = letType.getDecl(emitter.shared)) {
      bool isErroneousDecl = false;
      // TODO: Unify this with the logic in emitRValue when __clone__ moves to
      // taking a borrow instead of a mutable byref argument.
      if (!OverloadSet(letType, "__clone__", this, isErroneousDecl,
                       emitter.shared)) {
        auto diag = emitter.emitError(getLoc(), "cannot clone this value: ")
                    << letType << " doesn't implement '__clone__'"
                    << getRange();
        diag.attachNote(letTypeDecl->getLoc()) << "type declared here";
        return {};
      }
    }

    return emitter.emitResult(SRValue(letDecl.getResult()), this, dest);
  }

  // Variable references resolve to an lvalue addressing the variable.
  if (auto var = dyn_cast<VarDeclOp>(decl))
    return emitter.emitResult(LValue(var.getResult()), this, dest);

  // Parameters form a meta-value.
  if (auto param = dyn_cast<ParamDeclareOp>(decl)) {
    PRValue result =
        resolveParamDeclareValue(param, /*bindings=*/{}, emitter.shared);
    return emitter.emitResult(result, this, dest);
  }

  // Use of forward references.
  if (auto param = dyn_cast<AliasForwardDeclOp>(decl)) {
    PRValue result(ParamDeclRefAttr::get(param.getName(), param.getType()));
    return emitter.emitResult(result, this, dest);
  }

  // RValue's and LValues always resolve to their known value.
  if (auto rvalue = decl.getIfRValue())
    return emitter.emitResult(rvalue, this, dest);
  if (auto lvalue = decl.getIfLValue())
    return emitter.emitResult(lvalue, this, dest);

  // If this is a type declaration, return it as a type.
  if (isa<StructDeclOp>(decl)) {
    PRValue result(DeclRefType::get(decl.getSymbolRef()));
    return emitter.emitResult(result, this, dest);
  }

  // Reject unqualified struct field references.
  if (auto fieldOp = dyn_cast<StructFieldOp>(decl)) {
    emitter.emitError(getLoc(), "cannot access instance field '")
        << spelling << "' directly; did you mean `self.`?" << getRange();
    return {};
  }

  emitter.emitError(getLoc(), "use of declaration \"")
      << spelling << "\" as a value isn't supported yet" << getRange();
  return {};
}

/// This uses the MLIR parser to turn the specified MLIR type name into an MLIR
/// type.
static ASTType parseMLIRType(StringRef name, const ExprNode *node,
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
  if (!result)
    shared.emitError(node->getLoc(), "unknown MLIR type: ")
        << name << node->getRange();
  return result;
}

/// Perform subsitutions of the specified bindings into the symbol, returning,
/// in symConstAttrs, the resultant SymbolConstant attr for each adaptive
/// function overload.
/// On failure it produces an error message and returns failure.
AnyValue AttributeRefNode::emitAdaptiveSet(ORValue overloads,
                                           ExprEmitter &emitter,
                                           ValueDest dest) const {
  SmallVector<TypedAttr> symConstAttrs;
  for (ASTDecl *fnDecl : overloads->fnDecls) {
    auto funcOp = cast<LIT::FuncOp>(*fnDecl);
    if (!funcOp.getIsAdaptive()) {
      auto diag = emitter.emitError(getLoc(),
                                    "cannot form a reference to non @adaptive "
                                    "declaration of '")
                  << overloads->baseName << "'" << getRange();
      diag.attachNote(funcOp.getLoc()) << "declared here";
      return {};
    }
    TypedAttr symbolAttr =
        overloads->getBoundConstAttrFor(this, funcOp, emitter);
    if (!symbolAttr)
      return {};
    symConstAttrs.push_back(symbolAttr);
  }

  auto attr =
      VariadicAttr::get(emitter.getContext(), symConstAttrs,
                        VariadicType::get(symConstAttrs.front().getType()));
  return emitter.emitResult(attr, this, dest);
}

/// Emit a qualified attribute reference to MLIR.  On error, emit an error and
/// return a null value.
AnyValue AttributeRefNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  auto baseVal = base->emitIR(emitter, ValueDest());
  if (!baseVal)
    return {};

  // Handle __adaptive_set.
  if (auto overloads = baseVal.getIfORValue())
    if (attrSpelling == "__adaptive_set")
      return emitAdaptiveSet(overloads, emitter, dest);

  // Figure out what type is being accessed.  'hasTypeBase' is when the base
  // expression is itself a type, e.g. `Int.__add__`.
  ASTType baseRVType;
  bool hasTypeBase = false;

  // Handle member references on types, like Int.member.
  if (ASTType baseType = baseVal.getIfTypeValue()) {
    baseRVType = baseType;
    hasTypeBase = true;
  } else {
    // Otherwise, it must be an access to a field of a value.  Look up in the
    // RValueType of the value.
    baseRVType = baseVal.getRValueType();
  }

  // Find the decl for the type we're looking up into.
  ASTDecl *typeDecl = baseRVType.getDecl(emitter.shared);
  if (!typeDecl) {
    // If there is no decl, the type is an MLIR type.
    Type baseMLIRType = baseRVType.mlirType;

    // Handle __mlir_op.`xxx` references, lazily synthesizing values when
    // they are referenced.
    if (isa<MagicMLIRAttrType>(baseMLIRType)) {
      PRValue result =
          synthesizeMLIRAttrFromString(attrSpelling, getLoc(), emitter.shared);
      return emitter.emitResult(result, this, dest);
    }
    if (isa<MagicMLIROpType>(baseMLIRType)) {
      PRValue result = synthesizeMLIROpFromString(attrSpelling, emitter);
      return emitter.emitResult(result, this, dest);
    }
    if (isa<MagicMLIRTypeType>(baseMLIRType)) {
      ASTType result = parseMLIRType(attrSpelling, this, emitter.shared);
      return emitter.emitResult(result, this, dest);
    }

    emitter.emitError(getLoc(), "MLIR type ")
        << baseRVType << " has no attributes" << base->getRange();
    return {};
  }

  if (!isa<StructDeclOp>(*typeDecl)) {
    emitter.emitError(getLoc(), "cannot access attribute in type ")
        << ASTType(baseVal.getType()) << base->getRange();
    return {};
  }

  // Find the member being accessed.
  LookupResult lookup =
      emitter.shared.lookupAndResolveDecl(attrSpelling, getLoc(), *typeDecl,
                                          /*searchParentScopes=*/false);
  ArrayRef<ASTDecl *> memberDecls = lookup.getIfSuccess();
  if (memberDecls.empty()) {
    // If the error hasn't been diagnosed, handle it now.
    if (lookup.isFailure())
      emitter.emitError(getLoc(), "")
          << baseRVType << " value has no attribute '" << attrSpelling << "'"
          << getRange();
    return {};
  }

  // Handle method references, which might be overloaded.
  if (auto fnOp = dyn_cast<LIT::FuncOp>(*memberDecls[0])) {
    // Get a symbol for the underlying function.
    auto result = ORValue::create(attrSpelling, memberDecls,
                                  baseRVType.getParamBindings());

    // If the callee is a static method, we can directly reference it
    // without binding a self parameter.  If this is an instance method, we
    // bind the base value and the symbol together into a callable.
    // FIXME: This isn't handling overloaded static/non-static methods
    // correctly.  What is the actual behavior we want for static methods?
    // Maybe we don't allow overloading static and non-static methods with
    // the same name?
    if (!fnOp.getIsStatic() && !hasTypeBase)
      result->baseValue = {baseVal, base};
    return emitter.emitResult(std::move(result), this, dest);
  }

  assert(memberDecls.size() == 1 && "only methods may be overloaded");
  ASTDecl &memberDecl = *memberDecls[0];

  // Parameters form a meta-value.
  if (auto param = dyn_cast<ParamDeclareOp>(memberDecl)) {
    PRValue result = resolveParamDeclareValue(
        param, baseRVType.getParamBindings(), emitter.shared);
    return emitter.emitResult(result, this, dest);
  }

  auto mlirLoc = getLocation(emitter);

  // If the field is a variable, emit a reference to it.
  if (auto fieldOp = dyn_cast<StructFieldOp>(memberDecl)) {
    if (hasTypeBase) {
      emitter.emitError(getLoc(), "cannot access instance field '")
          << attrSpelling << "' without an instance of " << baseRVType
          << getRange();
      return {};
    }

    // If the base is an lvalue, then we can return an lvalue to the field.
    if (LValue baseLV = baseVal.getIfLValue()) {
      auto fieldPtr =
          emitter.builder->create<StructGEPOp>(mlirLoc, baseLV, fieldOp);
      return emitter.emitResult(LValue(fieldPtr), this, dest);
    }

    // If the base is an PRValue, emit a field extract as an PRValue.
    if (PRValue baseMV = baseVal.getIfPRValue()) {
      auto extractVal = LIT::StructExtractAttr::get(baseMV.get(), fieldOp);
      return emitter.emitResult(PRValue(extractVal), this, dest);
    }

    // Otherwise, it must be an rvalue.
    // TODO(memory_primary): Handle memory-only rvalues by gep'ing into them.
    SRValue baseRV = emitter.emitSRValue({baseVal, base});
    if (!baseRV)
      return {};

    auto extractVal =
        emitter.builder->create<StructExtractOp>(mlirLoc, baseRV, fieldOp);
    return emitter.emitResult(SRValue(extractVal), this, dest);
  }

  // Reference to some non-function/struct member of the type.
  emitter.emitError(getLoc(), "reference to unknown member '")
      << attrSpelling << "'" << getRange();
  return {};
}

/// Given a call to an UnboundMLIROperator, generate an MLIR operation with
/// the operands as SSA values.
static AnyValue emitMLIROperatorCall(const CallNode &call,
                                     UnboundMLIROperationAttr unboundOp,
                                     ExprEmitter &emitter) {
  auto *context = emitter.getContext();

  if (!emitter.builder) {
    emitter.emitError(
        call.getLoc(),
        "MLIR operation cannot be used directly in parameter expressions")
        << call.getRange();
    return {};
  }

  // Emit all the arguments so we can encode them as SSA values.
  SmallVector<Value> opOperands;
  for (ExprNode *operand : call.args) {
    // We allow clients to use three nested paren expressions to get access to
    // the address of an lvalue.  This is a gross hack, we might want to use
    // keyword arguments for this when we have them, e.g.
    // __mlir_op.`thing`(addr_of = expression)
    size_t numParens = 0;
    while (auto *paren = dyn_cast<ParenNode>(operand)) {
      operand = paren->subExpr;
      ++numParens;
    }

    Value value;
    if (numParens >= 3)
      value =
          emitter.emitExprLValue(call.getLoc(), operand, /*contextualType=*/{},
                                 "((())) operand must be an lvalue");
    else
      value = emitter.emitExprSRValue(operand);
    if (!value)
      return {};
    opOperands.push_back(value);
  }

  // Set up the OperationState for the thing we're building.
  OperationState state(call.getLocation(emitter), unboundOp.getName());
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
        state.types.push_back(PRValue(typedAttr).getIfTypeValue());
      } else {
        emitter.emitError(call.getLoc(), "unknown _type value");
        return {};
      }
      hadTypeSpec = true;
      continue;
    }
    if (attr.getName() == "_region") {
      // A region is specified for an MLIR operation by using the `_region`
      // special attribute to refer to a function declaration.
      auto bodyRef = dyn_cast<StringAttr>(attr.getValue());
      if (!bodyRef) {
        emitter.emitError(call.getLoc(),
                          "MLIR operation region must be a region reference");
        return {};
      }
      // Lookup the operation body.
      LookupResult result = emitter.shared.lookupAndResolveDecl(
          bodyRef, call.getLoc(), emitter.declScope,
          /*searchParentScopes=*/false);
      ArrayRef<ASTDecl *> results = result.getIfSuccess();
      if (result.isFailure() || results.size() != 1 ||
          !isa<LIT::UnboundRegionOp>(*results.front())) {
        emitter.emitError(call.getLoc(), "MLIR operation region reference did "
                                         "not resolve to a region body");
        return {};
      }
      auto unboundRegion = cast<LIT::UnboundRegionOp>(*results.front());
      auto region = std::make_unique<Region>();
      region->takeBody(unboundRegion.getRegion());
      unboundRegion.erase();
      results.front()->setIRValue(PRValue(BoolAttr::get(context, false)));
      state.addRegion(std::move(region));
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

    // Check to see if this implements the ZeroResults trait.
    if (opNameInfo->hasTrait<mlir::OpTrait::ZeroResults>())
      return success(); // We know there are zero results.

    // Otherwise, check for InferTypeOpInterface.
    auto inferTypesItf = opNameInfo->getInterface<mlir::InferTypeOpInterface>();
    if (!inferTypesItf)
      return failure();
    if (failed(inferTypesItf->inferReturnTypes(
            context, state.location, state.operands,
            DictionaryAttr::get(context, state.attributes), state.regions,
            state.types)))
      return failure();
    return success(
        llvm::all_of(state.types, [](Type t) { return t != Type(); }));
  };

  if (!hadTypeSpec) {
    if (failed(inferType())) {
      emitter.emitError(call.getLoc(),
                        "unable to infer result type from MLIR operation ")
          << unboundOp.getName() << call.getRange();
      return {};
    }
    if (state.types.size() > 1) {
      emitter.emitError(call.getLoc(),
                        "cannot use operations with multiple results (yet) ")
          << unboundOp.getName() << call.getRange();
      return {};
    }
  }

  Operation *resultOp = emitter.builder->create(state);

  // Explicitly run the verifier on the new operation so we make sure to
  // catch problems early.
  std::string errorMessage;
  bool verificationError = false;
  // FIXME: Terminators expect certain parent operations and are only valid when
  // inlined into an operation's region. Don't verify them.
  if (!resultOp->hasTrait<OpTrait::IsTerminator>()) {
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
    return PRValue(NoneAttr::get(emitter.getContext(), noneMLIRType));
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
      assert(val.getType() == resultOp->getResult(0).getType());
      resultOp->erase();
      return SRValue(val);
    }

    if (auto attr = dyn_cast<TypedAttr>(cast<Attribute>(folded))) {
      assert(attr.getType() == resultOp->getResult(0).getType());
      // If it is a constant, make an MAValue result.
      resultOp->erase();
      return PRValue(attr);
    }
  }

  // If folding failed, return the operation normally.
  return SRValue(resultOp->getResult(0));
}

AnyValue CallNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  RValue calleeVal = emitter.emitExprRValue(callee, ValueDest());
  if (!calleeVal)
    return {};

  // If this is the invocation of an unbound MLIR operator, bind it into an
  // actual operator!
  if (auto mValue = calleeVal.getIfPRValue()) {
    if (auto unboundOp = dyn_cast<UnboundMLIROperationAttr>(mValue.get())) {
      AnyValue result = emitMLIROperatorCall(*this, unboundOp, emitter);
      return emitter.emitResult(result, this, dest);
    }
  }

  /// Emit all the operands that we'll need.
  SmallVector<ASTExprAnd<AnyValue>> operands;
  for (ExprNode *arg : args) {
    operands.push_back({arg->emitIR(emitter, ValueDest()), arg});
    if (!operands.back())
      return {};
  }

  // If the callee is a type value (as in `T()` or `T[123]()`), then this is an
  // invocation of the initializer for the type.
  if (ASTType calledType = calleeVal.getIfTypeValue()) {
    bool isErroneousDecl = false;
    OverloadSet overloads(calledType, "__new__", this, isErroneousDecl,
                          emitter.shared);
    if (overloads.isNull()) {
      if (isErroneousDecl)
        return {};

      if (calledType.getDecl(emitter.shared)) {
        emitter.emitError(getLoc(), "")
            << calledType << " does not have any `__new__` methods"
            << callee->getRange();
      } else {
        emitter.emitError(getLoc(),
                          "cannot use initializer syntax on MLIR type ")
            << calledType << callee->getRange();
      }
      return {};
    }

    return overloads.emitCall(operands, dest, this, CallSyntax::kTypeCall,
                              emitter);
  }

  // If this is an overloaded operand, resolve it and call the result.
  if (auto overloads = calleeVal.getIfORValue()) {
    // Figure out how this was spelled.
    CallSyntax syntax = overloads->baseValue ? CallSyntax::kMethodCall
                                             : CallSyntax::kDirectCall;
    return overloads->emitCall(operands, dest, this, syntax, emitter);
  }

  // Otherwise, we must have a concrete RValue, emit an indirect call.
  auto crVal = calleeVal.getIfCRValue();
  return OverloadSet::emitIndirectCall(crVal, operands, dest, this, emitter);
}

AnyValue SliceNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  auto diag =
      emitter.emitError(getLoc(), "TODO: SliceNode::emitIR not implemented yet")
      << getRange();
  diag.attachNote(getLocation(emitter))
      << "keyword arguments aren't supported yet";
  return {};
}

/// Given a value of type type, substitute parameters into the type, producing
/// a more concrete type.  This syntax is `SomeType[1, 4, Int]`.
static PRValue substituteParametersIntoUserDefinedType(
    DeclRefType declRef, const SubscriptNode &subscript, ExprEmitter &emitter) {
  // If already parameterized, give up.
  // TODO: Why not allow multiple partial type applications?
  if (!declRef.getParamValues().empty()) {
    emitter.emitError(
        subscript.getLoc(),
        "cannot apply more parameters to an already parameterized type ")
        << ASTType(declRef) << subscript.getIndexRange();
    return {};
  }

  ASTDecl &typeDecl =
      emitter.shared.declResolver->getDeclForTypeSymbol(declRef.getSymbol());
  auto structOp = dyn_cast<StructDeclOp>(typeDecl);
  if (!structOp) {
    emitter.emitError(subscript.getLoc(), "unknown parameterized type ")
        << ASTType(declRef) << subscript.base->getRange();
    return {};
  }

  // Build up a InputParamBindings set to validate and check the bindings.
  InputParamBindings paramBindings;
  for (ExprNode *indexExpr : subscript.indices) {
    // TODO: Slice syntax is the obvious way to support named parameter
    // arguments.
    auto indexVal =
        emitter.emitExprPRValue(indexExpr, ASTType(), " in type parameter");
    if (!indexVal)
      return {};
    paramBindings.add(indexExpr, indexVal.get());
  }

  // Check the bindings.
  ssize_t incorrectBindingNo = 0;
  ASTType incorrectBindingExpectedType;
  auto bindingAttr = paramBindings.verifyBindings(
      structOp.getInputParamDeclsAttr(), structOp.getName(), subscript.getLoc(),
      incorrectBindingNo, incorrectBindingExpectedType, emitter, structOp,
      structOp.getParamVarargs());
  if (!bindingAttr)
    return {};

  // Ok, we succeeded at reparameterizing the type.
  return PRValue(DeclRefType::get(typeDecl.getSymbolRef(), bindingAttr));
}

/// When subscripting a callable with a bound symbol (i.e. a direct method call
/// or call to a method), apply parameter bindings to it.
static ORValue bindParamValuesToDirectCall(ORValue value,
                                           ArrayRef<ExprNode *> indices,
                                           ExprEmitter &emitter) {
  // If the indices are a single () expression, then we treat this as having
  // no parameters.  This is used with arrow expressions to allow `f[() -> x]`.
  if (indices.size() == 1) {
    if (auto *tuple = dyn_cast<TupleNode>(indices[0]))
      if (tuple->exprs.empty())
        return value;
  }

  // Process each subscript entry as a binding.
  // TODO: Support named bindings in addition to positional ones: `A[x: 42]`.
  for (auto idx : indices) {
    // If all entries in this overload set take a parameter with a common type,
    // use it for parameter type inference.
    ASTType paramType;
    if (!value->fnDecls.empty()) {
      auto getCandidateParamType = [&](ASTDecl *fnDecl) -> Type {
        auto signature = cast<LIT::FuncOp>(*fnDecl).getFullSignature();
        return value->inputParamBindings.getNextExpectedBindingType(signature,
                                                                    emitter);
      };
      paramType = getCandidateParamType(value->fnDecls[0]);
      if (paramType && value->fnDecls.size() != 1 &&
          !llvm::all_of(value->fnDecls, [&](ASTDecl *decl) {
            return ASTType(paramType).isEqualCanon(getCandidateParamType(decl));
          }))
        paramType = ASTType();
    }

    auto val = emitter.emitExprPRValue(idx, paramType, " in parameter binding");
    if (!val)
      return {};

    // We don't do any checking to see if the value is compatible with the
    // expected type - this is deferred until when the symbol is actually
    // emitted for something.  This allow us to use the provided parameters to
    // filter down the overload set.
    //
    // Note: we're being a bit abusive here by making a ParamBindAttr with a
    // null name for positional attributes.
    value->inputParamBindings.add(idx, val.get());
  }
  // The bindings will be checked for validity when a reference is formed.
  return value;
}

AnyValue SubscriptNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  // Subscripting a generic function binds the parameter expressions.
  auto subValue = base->emitIR(emitter, ValueDest());
  if (!subValue)
    return {};

  // If the subValue has a bound callable symbol, then this is applying (more?)
  // parameter expressions to bind its parameters.
  if (auto overloads = subValue.getIfORValue()) {
    auto result = bindParamValuesToDirectCall(overloads, indices, emitter);
    return emitter.emitResult(result, this, dest);
  }

  if (auto callableMVal = subValue.getIfPRValue()) {
    if (auto sig = dyn_cast<SignatureType>(callableMVal.getType())) {
      // If this is a signature-type PRValue callable, this is binding parameter
      // values to a call.
      SmallVector<TypedAttr> bindOperands({callableMVal.get()});
      if (indices.size() != sig.getInputParams().size()) {
        emitter.emitError(getLoc(), "parametric callable expected ")
            << sig.getInputParams().size() << " parameter"
            << plural(sig.getInputParams().size()) << getIndexRange();
        return {};
      }
      for (auto [idx, type] : llvm::zip(indices, sig.getInputParams())) {
        bindOperands.push_back(emitter.emitExprPRValue(
            idx, type.getType(), " in call parameter binding"));
        if (!bindOperands.back())
          return {};
      }

      PRValue result(ParamOperatorAttr::get(POC::BindSignature, bindOperands));
      return emitter.emitResult(result, this, dest);
    }
  }

  // If the sub-value is an unbound Type, try binding things to it!
  if (Type typeValue = subValue.getIfTypeValue()) {
    // Handle user-defined types.
    if (auto declRef = dyn_cast<DeclRefType>(typeValue)) {
      PRValue result =
          substituteParametersIntoUserDefinedType(declRef, *this, emitter);
      return emitter.emitResult(result, this, dest);
    }

    // Handle __mlir_type["foo"] and __mlir_attr["foo"].
    if (isa<MagicMLIRTypeType>(typeValue)) {
      std::string result = substituteMLIRMagic(*this, emitter);
      if (result.empty())
        return {};
      ASTType type = parseMLIRType(result, this, emitter.shared);
      return emitter.emitResult(type, this, dest);
    }
    if (isa<MagicMLIRAttrType>(typeValue)) {
      std::string result = substituteMLIRMagic(*this, emitter);
      if (result.empty())
        return {};
      PRValue attr =
          synthesizeMLIRAttrFromString(result, getLoc(), emitter.shared);
      return emitter.emitResult(attr, this, dest);
    }
  }

  // Otherwise, if there is no symbol, it is just an LValue or RValue being
  // subscript.
  if (auto value = subValue.getIfPRValue()) {
    if (auto unboundOperator =
            dyn_cast<UnboundMLIROperationAttr>(value.get())) {
      PRValue result =
          bindAttributesToMLIROperatorCall(*this, unboundOperator, emitter);
      return emitter.emitResult(result, this, dest);
    }
  }

  // Emit each of the index values, which will be passed to the __getitem__ and
  // __setitem__ calls.
  SmallVector<ASTExprAnd<AnyValue>> indexValues;
  indexValues.push_back({subValue, base});
  for (ExprNode *index : indices) {
    indexValues.push_back({index->emitIR(emitter, ValueDest()), index});
    if (!indexValues.back())
      return {};
  }

  // Okay, we're doing a normal value subscript.  We expect at least a
  // __getitem__ method.
  auto baseType = subValue.getRValueType();
  bool isErroneousDecl = false;
  OverloadSet getItem(baseType, "__getitem__", this, isErroneousDecl,
                      emitter.shared);
  // If there is no __getitem__ at all, then this is not a subscriptable type.
  if (getItem.isNull()) {
    if (isErroneousDecl)
      return {};
    emitter.emitError(getLoc(), "")
        << baseType << " does not implement the `__getitem__` method"
        << base->getRange();
    return {};
  }

  // Okay, we have one, that's a positive sign. In the case of multiple index
  // values, we could either pass this as a Python style tuple, or could pass as
  // multiple arguments.

  // TODO: If we have multiple indexes, package up the values in a tuple value
  // and try to see if this works.
  if (indexValues.size() > 2) {
    // TODO(Tuples). need tuples :-)
  }

  // Next, check the multiple argument path.
  if (getItem && succeeded(getItem.filterOverloadSet(
                     indexValues, CallSyntax::kSubscript, this,
                     /*allowImplicitConversions=*/true,
                     /*emitDiagnosticOnFailure=*/false, emitter))) {
    // Ok, this looks like it will work.
    // TODO(Computed LValues): We need to look up __setitem__ and have a better
    // model for computed LValues.
  }

  // Finally, just emit the call to __getitem__.
  return getItem.emitCall(indexValues, dest, this, CallSyntax::kSubscript,
                          emitter);
}

AnyValue SubscriptArrowNode::emitIR(ExprEmitter &emitter,
                                    ValueDest dest) const {
  // Subscripting a generic function binds the parameter expressions.
  auto subValue = base->emitIR(emitter, ValueDest());
  if (!subValue)
    return {};

  // If the subValue has a bound callable symbol, then this is applying (more?)
  // meta values to bind its parameters.
  auto overloads = subValue.getIfORValue();
  if (!overloads) {
    emitter.emitError(arrowLoc, "invalid '->' when subscripting type ")
        << ASTType(subValue.getType()) << getRange();
    return {};
  }

  // The only use of SubscriptArrow nodes right now is to bind parameter
  // input values and results to a call.  Start by binding the input values.
  overloads = bindParamValuesToDirectCall(overloads, indices, emitter);
  if (!overloads)
    return {};

  // Next, bind the results.  The grammar allows any expression, but we only
  // accept identifiers.
  for (ExprNode *dest : arrowExprs) {
    auto *drn = dyn_cast<DeclRefNode>(dest);
    if (!drn) {
      emitter.emitError(drn->getLoc(),
                        "expected identifier for parameter result")
          << dest->getRange();
      return {};
    }
    StringRef resultName = drn->spelling;

    // Lookup the name.  We must find a forward declared alias that isn't
    // already completed.
    auto result = emitter.shared.lookupAndResolveDecl(
        resultName, drn->getLoc(), emitter.declScope,
        /*searchParentScopes=*/false);

    // Reject the code if nothing was found.
    ArrayRef<ASTDecl *> resultDecls = result.getIfSuccess();
    if (resultDecls.empty()) {
      if (result.isFailure())
        emitter.emitError(drn->getLoc(),
                          "unable to find forward-declared alias named '")
            << resultName << "'" << drn->getRange();
      return {};
    }

    // Reject non-alias results.
    auto aliasDecl = dyn_cast<AliasForwardDeclOp>(*resultDecls[0]);
    if (!aliasDecl || resultDecls.size() > 1) {
      auto diag = emitter.emitError(drn->getLoc(), "'")
                  << resultName << "' is not a forward declared alias"
                  << drn->getRange();
      for (auto *decl : resultDecls)
        diag.attachNote(decl->getLoc())
            << "'" << resultName << "' declared here";
      return {};
    }

    // Verify the decl isn't already defined.
    if (aliasDecl.getResultParamLoc().has_value()) {
      auto diag = emitter.emitError(drn->getLoc(), "'")
                  << resultName << "' alias was defined by another result"
                  << drn->getRange();
      diag.attachNote(*aliasDecl.getResultParamLoc())
          << "previously defined here";
      return {};
    }

    // Set the location for this definition so we can know it was defined
    // correctly, and diagnose subsequent attempts to redefine it.
    aliasDecl.setResultParamLocAttr(drn->getLocation(emitter));
    overloads->resultParams.push_back({resultDecls[0], drn->getLoc()});
  }

  return emitter.emitResult(overloads, this, dest);
}

AnyValue ParenNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  return subExpr->emitIR(emitter, dest);
}

AnyValue ParenNode::emitExprResultIntoPattern(ASTExprAnd<AnyValue> value,
                                              ExprEmitter &emitter) const {
  return subExpr->emitExprResultIntoPattern(value, emitter);
}

AnyValue TupleNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  // Emit each of the index values to generate error messages.
  SmallVector<RValue> exprValues;
  for (ExprNode *expr : exprs) {
    exprValues.push_back(emitter.emitExprRValue(expr, ValueDest()));
    if (!exprValues.back())
      return {};
  }

  emitter.emitError(getLoc(), "FIXME: Cannot emit tuple expressions yet")
      << getRange();
  return {};
}

AnyValue ListNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  SmallVector<RValue> elements;
  for (ExprNode *expr : exprs) {
    elements.push_back(emitter.emitExprRValue(expr, ValueDest()));
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
  return PRValue(noneAttr);
}

AnyValue DictionaryNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  emitter.emitError(getLoc(), "TODO: cannot emit dictionary literals yet")
      << getRange();
  return {};
}

/// Emit a DictSubscriptNode when the base is a Type expression.
AnyValue DictSubscriptNode::emitTypeSubscriptIR(ASTType initType,
                                                ExprEmitter &emitter) const {
  auto *decl = initType.getDecl(emitter.shared);
  if (!decl) {
    emitter.emitError(getLoc(),
                      "MLIR types may not be initialized with this syntax")
        << base->getRange();
    return {};
  }

  // The type must be fully parsed to understand what fields it contains.
  if (failed(emitter.getDeclResolver().resolveFully(*decl, base->getLoc())))
    return {};

  auto structOp = dyn_cast<StructDeclOp>(*decl);
  if (!structOp) {
    emitter.emitError(getLoc(),
                      "can only initialize struct types with this syntax")
        << base->getRange();
    return {};
  }

  // While we use general dictionary syntax, the keys are syntactically
  // limited to being keywords.  The values may be arbitrary RValues
  // though, and are emitted in lexical order.
  DenseMap<StringAttr, ASTExprAnd<RValue>> fieldMapping;
  for (auto &keyValue : indices->values) {
    // We don't support `**dict` syntax.
    if (!keyValue.first) {
      emitter.emitError(keyValue.second->getLoc(),
                        "cannot expand into initializer list")
          << keyValue.second->getRange();
      return {};
    }

    auto fieldName = dyn_cast<DeclRefNode>(keyValue.first);
    if (!fieldName) {
      emitter.emitError(keyValue.first->getLoc(),
                        "type initializer requires keys to be bare field names")
          << keyValue.first->getRange() << base->getRange();
      return {};
    }
    StringAttr fieldNameAttr =
        StringAttr::get(emitter.getContext(), fieldName->spelling);

    auto value = emitter.emitExprRValue(keyValue.second, ValueDest());
    if (!value)
      return {};

    auto mapResult =
        fieldMapping.insert({fieldNameAttr, {value, keyValue.second}});
    if (!mapResult.second) {
      emitter.emitError(keyValue.first->getLoc(), "field ")
          << fieldNameAttr << " specified multiple times"
          << keyValue.first->getRange() << base->getRange()
          << mapResult.first->second.expr->getRange();
      return {};
    }
  }

  // Now that we have all the values, generate the initializers for
  // StructCreate.
  if (!emitter.builder) {
    emitter.emitError(getLoc(), "TODO: Don't have #lit.struct.attr yet");
    return {};
  }

  // Perform parameter substitution if there are input parameters.
  LitParameterEvaluator paramEvaluator(emitter.getDeclResolver(),
                                       initType.getParamBindings());

  SmallVector<StringAttr> fieldNames;
  SmallVector<Value> fieldValues;
  for (StructFieldOp field : structOp.getFieldDecls()) {
    ASTExprAnd<RValue> fieldVal = fieldMapping[field.getNameAttr()];
    if (!fieldVal) {
      emitter.emitError(indices->rbraceLoc, "no value for field ")
          << field.getNameAttr() << " specified";
      return {};
    }

    // The field must be fully parsed to understand its type etc.  Do a lookup
    // to find its decl and resolve it.
    for (ASTDecl *fieldDecl :
         *decl->lookupInCurrentScope(field.getNameAttr())) {
      if (failed(emitter.getDeclResolver().resolveFully(
              *fieldDecl, fieldVal.expr->getLoc())))
        return {};
    }

    auto value = emitter.getAsExpectedType(
        fieldVal, paramEvaluator.getReboundType(field.getType()),
        // TODO(memory_primary)
        ValueDest(), " in field initialization");
    // TODO(memory_primary): Handle memory-only values by direct initializing.
    auto drValue = emitter.emitSRValue({value, fieldVal.expr});
    if (!drValue)
      return {};
    fieldNames.push_back(field.getNameAttr());
    fieldValues.push_back(drValue);
  }

  return SRValue(emitter.builder->create<StructCreateOp>(
      getLocation(emitter), initType.mlirType, fieldValues,
      StringArrayAttr::get(emitter.getContext(), fieldNames)));
}

AnyValue DictSubscriptNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {

  auto baseValue = base->emitIR(emitter, /*contextualType*/ {});
  if (!baseValue)
    return {};

  // Subscripting a type constructs it with lit.struct.create.
  if (ASTType typeValue = baseValue.getIfTypeValue())
    return emitTypeSubscriptIR(typeValue, emitter);

  emitter.emitError(getLoc(), "TODO: cannot emit dictionary subscripts yet")
      << getRange();
  return {};
}

/// Given an operator, return the SpecialFunctionInfo that implements it.
static SpecialFunctionInfo getOpSpecialFunctions(ExprNode::Kind kind,
                                                 bool isReversed) {

  // Use an if chain to find the right match.  We can't use switch here because
  // multiple special functions may implement the same kind, e.g. __add__ and
  // __radd__ special methods both implement kAdd.
#define SF(ENUM, NAME, NUMOPERANDS, EXPRNODE, FLAGS)                           \
  if (kind == ExprNode::Kind::EXPRNODE) {                                      \
    auto info = SpecialFunctionInfo::get(SpecialFunctionKind::ENUM);           \
    if (info.isReversed() == isReversed)                                       \
      return info;                                                             \
  }
#include "SpecialFunctions.def"
  // If everything fails we should return "normal".
  return SpecialFunctionInfo::get(SpecialFunctionKind::kNormal);
}

/// Emit the binary operation (with a `lhs`, `rhs` and `kind`) as a special
/// function call.
/// A special function call is one where the` kind` must corresponds to a valid
/// SpecialFunctionInfo when we invoke getOpSpecialFunctions(kind).
/// `callNode` is the call like expression that results in the call.
//
/// This is an utility function to share code between BinOpNone and
/// ChainedCmpOpNode since the latter is a sequence of binary operations.
static AnyValue emitBinOpCall(ASTExprAnd<AnyValue> lhs,
                              ASTExprAnd<AnyValue> rhs, ExprNode::Kind kind,
                              ValueDest dest, const ExprNode *callNode,
                              ExprEmitter &emitter) {

  // FIXME: We currently hack in index type support as transition to proper
  // expression support.
  if ((lhs.ir.getType().isIndex() && rhs.ir.getType().isIndex()) &&
      lhs.ir.getIfPRValue() && rhs.ir.getIfPRValue()) {
    auto lhsParam = lhs.ir.getIfPRValue();
    auto rhsParam = rhs.ir.getIfPRValue();
    POC opcode;
    bool needsInvert = false;
    switch (kind) {
    default:
      emitter.emitError(
          callNode->getLoc(),
          "cannot emit this binary operator in parameter context yet")
          << callNode->getRange();
      return {};
    case ExprNode::kSub:
      return ParamOperatorAttr::getSub(lhsParam, rhsParam);
    case ExprNode::kAdd:
      opcode = POC::Add;
      break;
    case ExprNode::kMul:
      opcode = POC::Mul;
      break;
    case ExprNode::kAnd:
      opcode = POC::And;
      break;
    case ExprNode::kOr:
      opcode = POC::Or;
      break;
    case ExprNode::kXor:
      opcode = POC::Xor;
      break;
    case ExprNode::kLShift:
      opcode = POC::Shl;
      break;
    case ExprNode::kRShift:
      opcode = POC::Shr;
      break;
    case ExprNode::kFloorDiv:
      opcode = POC::Div;
      break;
    case ExprNode::kMod:
      opcode = POC::Mod;
      break;
    case ExprNode::kCmpEQ:
      opcode = POC::EQ;
      break;
    case ExprNode::kCmpNE:
      opcode = POC::EQ;
      needsInvert = true;
      break;
    case ExprNode::kCmpGE:
      opcode = POC::LT;
      needsInvert = true;
      break;
    case ExprNode::kCmpGT:
      opcode = POC::LE;
      needsInvert = true;
      break;
    case ExprNode::kCmpLT:
      opcode = POC::LT;
      break;
    case ExprNode::kCmpLE:
      opcode = POC::LE;
      break;
    }
    auto value = ParamOperatorAttr::get((POC)opcode, lhsParam, rhsParam);
    if (needsInvert)
      value = ParamOperatorAttr::getNot(value);
    return emitter.emitResult(value, callNode, dest);
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnInfo = getOpSpecialFunctions(kind, /*isReversed=*/false);
  assert(specialFnInfo.kind != SpecialFunctionKind::kNormal);
  ASTExprAnd<AnyValue> argValues[] = {lhs, rhs};

  // Check to see if we have a forward version of this function on the primary
  // receiver.
  bool isErroneousDecl = false;
  OverloadSet callee(lhs.ir.getRValueType(), specialFnInfo.name, callNode,
                     isErroneousDecl, emitter.shared);
  if (isErroneousDecl)
    return {};

  if (callee && succeeded(callee.filterOverloadSet(
                    argValues, CallSyntax::kOperator, callNode,
                    /*allowImplicitConversions=*/true,
                    /*emitDiagnosticOnFailure=*/false, emitter))) {
    return callee.emitCall(argValues, dest, callNode, CallSyntax::kOperator,
                           emitter);
  }

  // Check to see if we have the reverse version of this operator.
  auto reversedFnInfo = getOpSpecialFunctions(kind, /*isReversed=*/true);
  if (reversedFnInfo.kind != SpecialFunctionKind::kNormal) {
    // Swap the operand order.
    std::swap(argValues[0], argValues[1]);
    callee = OverloadSet(rhs.ir.getType(), reversedFnInfo.name, callNode,
                         isErroneousDecl, emitter.shared);
    if (callee && succeeded(callee.filterOverloadSet(
                      argValues, CallSyntax::kReversedOperator, callNode,
                      /*allowImplicitConversions=*/true,
                      /*emitDiagnosticOnFailure=*/false, emitter))) {
      return callee.emitCall(argValues, dest, callNode,
                             CallSyntax::kReversedOperator, emitter);
    }

    // Swap these back so we emit the right error.
    std::swap(argValues[0], argValues[1]);
  }

  // Emit an error complaining about the forward version of the operator.
  return emitter.emitNamedMethodCall(specialFnInfo.name, argValues, dest,
                                     CallSyntax::kOperator, callNode);
}

/// Emit a simple assignment statement. Python evaluates the RHS of an
/// assignment before the LHS, as seen in things like:
///    def test1(): print("test1"); return 0
///    def test2(): print("test2"); return 1
///    a[test1()] = test2()
///  ==> test2; test1
AnyValue BinOpNode::emitAssign(ValueDest dest, ExprEmitter &emitter) const {
  // In an assignment, we emit the RHS into the LHS as its context.  This is
  // required to enable the 'implicit declaration' behavior in a def and to
  // support patterns.
  RValue rhsRep = emitter.emitExprRValue(rhs, ValueDest(lhs));
  if (!rhsRep)
    return {};

  // Assignments are not actually expressions in Python.  We treat them this
  // way for consistency, but model them as returning None.
  return emitter.emitResult(NoneAttr::get(emitter.getContext()), this, dest);
}

/// Emit a inplace assignment statement like `x += y`. Python evaluates the RHS
/// of an assignment before the LHS, as seen in things like:
///    def test1(): print("test1"); return 0
///    def test2(): print("test2"); return 1
///    a[test1()] += test2()
///  ==> test1; test2
AnyValue BinOpNode::emitInplace(ValueDest dest, ExprEmitter &emitter) const {
  AnyValue lhsRep;
  RValue rhsRep;

  // Inplace operations evaluate the LHS first, so emit the LHS pattern as an
  // lvalue.
  LValue lhsLV =
      emitter.emitExprLValue(getLoc(), lhs, /*contextualType=*/{},
                             "cannot assign to immutable expression");
  if (!lhsLV)
    return {};

  // Then emit the right side.
  RValue rhsRV = emitter.emitExprRValue(rhs, ValueDest());
  if (!rhsRV)
    return {};

  // Emit the call to the operator function like `__iadd__`.
  return emitBinOpCall({lhsLV, lhs}, {rhsRV, rhs}, kind, dest, this, emitter);
}

AnyValue BinOpNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  // Handle weird binary operators specially if we have them.
  if (kind == kBoolAnd || kind == kBoolOr) // `x and y`, `x or y`
    return emitAndOr(dest, emitter);
  if (kind == kAssign) // `x = y`
    return emitAssign(dest, emitter);
  if (isAssignmentStmt()) // `x += y`
    return emitInplace(dest, emitter);

  // Othewise we emit the LHS followed by the RHS.
  RValue lhsRV = emitter.emitExprRValue(lhs, ValueDest());
  RValue rhsRV = emitter.emitExprRValue(rhs, ValueDest());
  if (!lhsRV || !rhsRV)
    return {};

  return emitBinOpCall({lhsRV, lhs}, {rhsRV, rhs}, kind, dest, this, emitter);
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
AnyValue BinOpNode::emitAndOr(ValueDest dest, ExprEmitter &emitter) const {
  Location ifLoc = getLocation(emitter);

  if (!emitter.builder) {
    emitter.emitError(getLoc(), "TODO(#6626): cannot emit short-circuit and/or "
                                "in a parameter context")
        << lhs->getRange() << rhs->getRange();
    return {};
  }

  // Emit the LHS value and capture the result of calling __bool__ in case we
  // need it.
  AnyValue lhsBool;
  SRValue lhsRV = emitter.emitExprSRValue(lhs);
  RValue lhsI1Value = emitter.emitConditionValueAsI1({lhsRV, lhs}, lhsBool);
  Value lhsI1SRValue = emitter.emitSRValue({AnyValue(lhsI1Value), lhs});
  if (!lhsI1SRValue)
    return {};

  auto ifOp = emitter.builder->create<HLCF::IfOp>(
      ifLoc, TypeRange{lhsBool.getType()}, lhsI1SRValue);
  emitter.builder->createBlock(&ifOp.getThenRegion());
  emitter.builder->createBlock(&ifOp.getElseRegion());

  OpBuilder trueBuilder = ifOp.getThenBodyBuilder();
  OpBuilder falseBuilder = ifOp.getElseBodyBuilder();
  if (kind == kBoolOr) // and/or just treat the bool differently.
    std::swap(trueBuilder, falseBuilder);

  emitter.builder = trueBuilder;
  SRValue rhsRV = emitter.emitExprSRValue(rhs);
  if (!rhsRV)
    return {};

  // Now that we know lhsRV and rhsRV we can tell if they have common types.
  // If so, we use that as the result of the 'if'.
  if (ASTType(lhsRV.getType()).isEqualCanon(rhsRV.getType())) {
    emitter.builder->create<HLCF::YieldOp>(ifLoc, rhsRV);
    // Emit the false side.
    emitter.builder = falseBuilder;
    emitter.builder->create<HLCF::YieldOp>(ifLoc, lhsRV);
    ifOp->getResult(0).setType(lhsRV.getType());
  } else {
    // Otherwise, check to see if their boolean versions are compatible.
    auto rhsBool =
        emitter.emitNamedMethodCall("__bool__", {{rhsRV, rhs}}, ValueDest(),
                                    CallSyntax::kImplicitConvert, this);
    if (!rhsBool)
      return {};
    if (!ASTType(lhsBool.getType()).isEqualCanon(rhsBool.getType())) {
      emitter.emitError(getLoc(), "cannot find common type between ")
          << ASTType(lhsRV.getType()) << " and " << ASTType(rhsRV.getType())
          << lhs->getRange() << rhs->getRange();
      return {};
    }
    auto rhsBoolDRVal = emitter.emitSRValue({rhsBool, rhs});
    if (!rhsBoolDRVal)
      return {};
    emitter.builder->create<HLCF::YieldOp>(ifLoc, rhsBoolDRVal);
    // Emit the false side.
    emitter.builder = falseBuilder;
    auto lhsBoolDRVal = emitter.emitSRValue({lhsBool, lhs});
    if (!lhsBoolDRVal)
      return {};
    emitter.builder->create<HLCF::YieldOp>(ifLoc, lhsBoolDRVal);
    ifOp->getResult(0).setType(lhsBool.getType());
  }

  emitter.builder->setInsertionPointAfter(ifOp);
  return emitter.emitResult(SRValue(ifOp.getResult(0)), this, dest);
}

AnyValue UnaryOpNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  auto exprRep = subExpr->emitIR(emitter, /*No Contextual Type*/ {});
  if (!exprRep)
    return {};

  // Special case some things for literals.
  // TODO: Fix literal representation.
  if ((exprRep.getType().isIndex() || exprRep.getType().isF64()) &&
      exprRep.getIfPRValue()) {
    auto exprParam = exprRep.getIfPRValue();
    switch (kind) {
    default:
      break;
    case ExprNode::kNeg:
      if (auto constantFP = dyn_cast<FloatAttr>(exprParam.get()))
        return PRValue(
            FloatAttr::get(constantFP.getType(), -constantFP.getValue()));

      // Support general integer parameter exprs.
      if (exprRep.getType().isIndex())
        return ParamOperatorAttr::getNeg(exprParam);

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
    argValue.ir = emitter.emitNamedMethodCall(
        "__bool__", argValue, ValueDest(), CallSyntax::kImplicitConvert, this);
    if (!argValue.ir)
      return {};
    // Now that we know we bool-ized the expression, invert it with ~.
    kindToEmit = kInvert;
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnInfo = getOpSpecialFunctions(kindToEmit, /*isReversed=*/false);
  assert(specialFnInfo.kind != SpecialFunctionKind::kNormal &&
         "Unary operators are implemented via special methods");

  return emitter.emitNamedMethodCall(specialFnInfo.name, argValue, dest,
                                     CallSyntax::kOperator, this);
}

AnyValue IfElseOpNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  RValue condRVal = emitter.emitExprConditionValueAsI1(condExpr);
  Value condValue = emitter.emitSRValue({AnyValue(condRVal), condExpr});

  if (!condValue)
    return {};

  if (!emitter.builder) {
    emitter.emitError(
        getLoc(),
        "TODO(#6626): cannot emit if/else in parameter expression yet")
        << trueExpr->getRange();
    return {};
  }

  Location ifLoc = getLocation(emitter);
  // At this point we don't know the type of trueExpr / falseExpr, use
  // a dummy one and fix it later.
  auto ifOp = emitter.builder->create<HLCF::IfOp>(
      ifLoc, TypeRange{condValue.getType()}, condValue);
  emitter.builder->createBlock(&ifOp.getThenRegion());
  emitter.builder->createBlock(&ifOp.getElseRegion());

  emitter.builder = ifOp.getThenBodyBuilder();
  SRValue trueVal = emitter.emitExprSRValue(trueExpr);
  if (!trueVal)
    return {};
  emitter.builder->create<HLCF::YieldOp>(ifLoc, trueVal);
  emitter.builder = ifOp.getElseBodyBuilder();
  SRValue falseVal = emitter.emitExprSRValue(falseExpr);
  if (!falseVal)
    return {};
  emitter.builder->create<HLCF::YieldOp>(ifLoc, falseVal);
  emitter.builder->setInsertionPointAfter(ifOp);

  /// TODO(subtyping): With subtypes, we can find intersection types, e.g. a
  /// common superclass.
  if (!ASTType(trueVal.getType()).isEqualCanon(falseVal.getType())) {
    emitter.emitError(getLoc(), "true value of type ")
        << ASTType(trueVal.getType()) << " is not compatible with false value "
        << ASTType(falseVal.getType()) << " in conditional"
        << trueExpr->getRange() << falseExpr->getRange();
    return {};
  }
  // Ensure the correct type is used.
  ifOp->getResult(0).setType(trueVal.getType());
  return SRValue(ifOp.getResult(0));
}

/// Emit the comparison expression with operator ops[opIdx] and operands:
///  1. lastExpr: the SSA value of the last expression in the chain emitted
///     so far.
///  2. The next expression node in the chain to be emitted: expr[opIdx + 1].
///
///  lastCmpExpr is the SSA value of the previous comparison expression.
///  Example
///  If the whole chained expression is a < b < c, and hence
///  a < b and b < c,  emitNextCmp will emit b < c, using the SSA
///  value of b in the previous comparison (a < b). lastCmpExpr is the value
///  of a < b.
///  Note that a < b  is handled by ChainedCmpOpNode::emitIR.
AnyValue ChainedCmpOpNode::emitNextCmp(ExprEmitter &emitter, size_t opIdx,
                                       SRValue lastCmpExpr,
                                       SRValue lastExpr) const {
  Location ifLoc = lastCmpExpr.getLoc();
  AnyValue boolResult;
  OpBuilder lastBuilder = emitter.builder.value();
  RValue lastCmpI1Value =
      emitter.emitConditionValueAsI1({lastCmpExpr, this}, boolResult);
  SRValue lastCmpI1RValue =
      emitter.emitSRValue({AnyValue(lastCmpI1Value), this});
  if (!lastCmpI1RValue)
    return {};
  auto ifOp = emitter.builder->create<HLCF::IfOp>(ifLoc, boolResult.getType(),
                                                  lastCmpI1RValue);
  emitter.builder->createBlock(&ifOp.getThenRegion());
  SRValue exprValue = emitter.emitExprSRValue(exprs[opIdx + 1]);
  if (!exprValue)
    return {};
  AnyValue lastBinOp =
      emitBinOpCall({lastExpr, exprs[opIdx]}, {exprValue, exprs[opIdx + 1]},
                    ops[opIdx], ValueDest(), this, emitter);
  SRValue lastRV = emitter.emitSRValue({lastBinOp, exprs[opIdx + 1]});
  if (!lastRV)
    return {};

  if (opIdx + 1 == ops.size())
    emitter.builder->create<HLCF::YieldOp>(ifLoc, lastRV);
  else if (!emitNextCmp(emitter, opIdx + 1, lastRV, exprValue))
    return {};

  emitter.builder->createBlock(&ifOp.getElseRegion());
  ifOp->getResult(0).setType(lastCmpExpr.getType());
  emitter.builder->create<HLCF::YieldOp>(ifLoc, lastCmpExpr);
  emitter.builder = lastBuilder;
  if (opIdx > 1)
    emitter.builder->create<HLCF::YieldOp>(ifLoc, ifOp->getResult(0));
  return SRValue(ifOp->getResult(0));
}

AnyValue ChainedCmpOpNode::emitIR(ExprEmitter &emitter, ValueDest dest) const {
  RValue e0Rep = emitter.emitExprRValue(exprs[0], ValueDest());
  RValue e1Rep = emitter.emitExprRValue(exprs[1], ValueDest());
  if (!e0Rep || !e1Rep)
    return {};

  AnyValue cmpe0e1RV =
      emitBinOpCall({e0Rep, exprs[0]}, {e1Rep, exprs[1]}, ops[0],
                    exprs.size() == 2 ? dest : ValueDest(), this, emitter);
  if (exprs.size() == 2)
    return cmpe0e1RV;

  SRValue lastCmpExpr = emitter.emitSRValue({cmpe0e1RV, exprs[1]});
  SRValue e1RV = emitter.emitSRValue(ASTExprAnd<RValue>{e1Rep, exprs[1]});
  if (!lastCmpExpr || !e1RV)
    return {};
  return emitter.emitResult(emitNextCmp(emitter, 1, lastCmpExpr, e1RV), this,
                            dest);
}
