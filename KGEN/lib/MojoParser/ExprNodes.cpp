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

#include "ExprNodes.h"
#include "ASTDecl.h"
#include "CallEmission.h"
#include "ClosureEmitter.h"
#include "ExprEmitter.h"
#include "IRValues.h"
#include "ParserParamEvaluator.h"
#include "SharedState.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LifetimeTrackable.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/MojoParser.h"
#include "KGEN/MojoParser/ASTDeclRef.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Given a StringRef for an MLIR attribute, invoke the MLIR parser to resolve
/// it into an Attribute (which may not be a TypedAttr) and return it.  On
/// error, emit a diagnostic and return null.
static Attribute parseMLIRAttrFromString(StringRef name, SMLoc loc,
                                         SharedState &shared) {
  Attribute result;
  std::string errorMsg;
  {
    // Capture errors thrown by parseAttribute and ignore them.
    // FIXME: This doesn't silence errors!
    mlir::ScopedDiagnosticHandler handler(
        shared.getContext(), [&](Diagnostic &diag) { errorMsg = diag.str(); });

    // FIXME(https://github.com/llvm/llvm-project/issues/58964)
    // Copy the string into a temporary smallvector so we can make sure it is
    // nul terminated for the MLIR asmparser.
    SmallString<64> tmpBuf(name.begin(), name.end());
    tmpBuf.push_back(0);

    // FIXME(#9621): Need to track the number of bytes read because we pass in
    // more than just the attribute we actually want to parse. This avoids
    // returning an error but is actually just masking the real problem.
    size_t bytesRead;
    result = mlir::parseAttribute(StringRef(tmpBuf).drop_back(),
                                  shared.getContext(), Type(), &bytesRead);
  }
  if (!result) {
    shared.emitError(loc, "invalid MLIR attribute: ") << errorMsg;
    return {};
  }

  return result;
}

/// This implements __mlir_attr.x lookup, synthesizing a PValue for the
/// attribute on demand.
static PValue synthesizeMLIRAttrFromString(StringRef name, SMLoc loc,
                                           SharedState &shared) {
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
  return PValue(typedAttr);
}

/// Given an __mlir_type[a,b,c] or __mlir_attr[a,b,c] usage, stringize the
/// indices and return the result.  On error, emit an error and return an empty
/// string.
static std::string substituteMLIRMagic(const SubscriptNode &node,
                                       ExprEmitter &emitter) {
  std::string result;
  llvm::raw_string_ostream os(result);

  for (auto *indexExpr : node.indices) {
    // If the index is an identifier, and if it is a backtick identifier, we
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

    auto indexVal = emitter.emitExprPValue(indexExpr, EC_MLIRMagic);
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
static PValue synthesizeMLIROpFromString(StringRef name, ExprEmitter &emitter) {
  auto *context = emitter.getContext();
  auto nameStr = StringAttr::get(context, name);

  auto result =
      UnboundMLIROperationAttr::get(nameStr, DictionaryAttr::get(context));
  return PValue(result);
}

/// Calculate the result of an __mlir_op.`thing`[attributes], applying the
/// attributes list to the operation specification.
static PValue
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
    // operation's attribute list, and emitPValue only supports TypedAttrs.
    if (auto attrRef = dyn_cast<AttributeRefNode>(node)) {
      auto mlirAttr = dyn_cast<DeclRefNode>(attrRef->base);
      if (mlirAttr && mlirAttr->spelling == "__mlir_attr") {
        if (attrRef->attrSpelling.empty())
          return {};
        return parseMLIRAttrFromString(attrRef->attrSpelling, attrRef->getLoc(),
                                       emitter.shared);
      }
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

    // Otherwise emit the value as an PValue.
    return emitter.emitExprPValue(node, EC_MLIRMagic);
  };

  SmallVector<NamedAttribute> attrValues;

  // Each element of the subscript must have a name identifier and a value as an
  // PValue.
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
  return UnboundMLIROperationAttr::get(unboundOp.getName(), attrs);
}

/// Given a AliasDeclOp, return the value that should be used in a reference
/// to it.  This currently fully substitutes members unless they are in a
/// function definition.
static PValue resolveAliasDeclareValue(AliasDeclOp param,
                                       ParamBindArrayAttr bindings,
                                       SharedState &shared, SMLoc errLoc) {
  // If the param is declared in a function, then just directly use it.
  Operation *parent = param->getParentOp();
  while (1) {
    // If this reference is within a function then keep it symbolic.
    if (parent && isa<LIT::FuncOp>(parent))
      return ParamDeclRefAttr::get(param.getName(), param.getType());

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

      if (structDecl.getInputParams().size() != bindings.size()) {
        shared.emitError(errLoc,
                         "incorrect number of struct parameters, expected:")
            << structDecl.getInputParams().size() << " got: " << bindings.size()
            << ".";
        return PValue();
      }

      ParserParamEvaluator evaluator(*shared.declResolver, bindings);
      return PValue(evaluator.getReboundAttribute(param.getValue()));
    }

    // Ignore if and other control flow things.
    parent = parent->getParentOp();
  }

  return ParamDeclRefAttr::get(param.getName(), param.getType());
}

//===----------------------------------------------------------------------===//
// ExprNode Implementation
//===----------------------------------------------------------------------===//

ExprNode::~ExprNode() {}

/// Return the start or end of the source range.
llvm::SMLoc ExprNode::getRangeStart() const { return getRange().getStart(); }
llvm::SMLoc ExprNode::getRangeEnd() const { return getRange().getEnd(); }

/// Return the 'loc' for this node translated to an MLIR location.
Location ExprNode::getLocation(ExprEmitter &emitter) const {
  return emitter.translateLocation(getLoc());
}
/// Recursively dig through noop paren nodes (if present) to find what is
/// inside of them.
ExprNode *ExprNode::getWithoutParens() {
  if (auto *paren = dyn_cast<ParenNode>(this))
    return paren->subExpr->getWithoutParens();
  return this;
}

/// Return true if this is a TupleNode with no subexpressions.
bool ExprNode::isEmptyTuple() const {
  if (auto *tuple = dyn_cast<TupleNode>(this))
    return tuple->exprs.empty();
  return false;
}

//===----------------------------------------------------------------------===//
// ExprNode implementations
//===----------------------------------------------------------------------===//

AnyValue IntLiteralNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // TODO: Handle contextual types.
  APInt value = Lexer::getIntegerLiteralValue(spelling);

  // Make sure the value fits in 64-bits.  There are no negative values here.
  // TODO: Detect overflow errors.
  value = value.zextOrTrunc(64);
  auto attr = IntegerAttr::get(IndexType::get(emitter.getContext()), value);

  // Convert this to an instance of Int. Int must be in scope since it is
  // auto-imported.
  ASTType type = emitter.shared.getBuiltinIntType(emitter.declScope, getLoc());
  return emitter.emitConstructorCall(type, {{AnyValue(attr), this}}, this,
                                     CallSyntax::kImplicitConvert, dest);
}

AnyValue FloatLiteralNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // TODO: this assumes float literal are always doubles
  APFloat value = Lexer::getFloatLiteralValue(spelling);
  auto attr = FloatAttr::get(FloatType::getF64(emitter.getContext()),
                             APFloat(value.convertToDouble()));

  // Convert this to an instance of Double.
  ASTType type =
      emitter.shared.getBuiltinDoubleType(emitter.declScope, getLoc());
  return emitter.emitConstructorCall(type, {{AnyValue(attr), this}}, this,
                                     CallSyntax::kImplicitConvert, dest);
}

AnyValue BoolLiteralNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // Create the SIMDAttr to represent the constant.
  auto boolDType = DTypeConstantAttr::get(emitter.getContext(), DType::kBool);
  auto boolAttr = POP::SIMDAttr::get({value, KGENDType::kBool},
                                     POP::SIMDType::get(1, boolDType));

  // Convert this to an instance of Bool. Bool must be in scope since it is
  // auto-imported.
  ASTType type = emitter.shared.getBuiltinBoolType(emitter.declScope, getLoc());
  return emitter.emitConstructorCall(type, {{AnyValue(boolAttr), this}}, this,
                                     CallSyntax::kImplicitConvert, dest);
}

AnyValue SimpleLiteralNode::emitIR(ValueDest &dest,
                                   ExprEmitter &emitter) const {
  if (kind == kNoneLiteral)
    return emitter.emitResult(emitter.shared.getNoneAttr(), this, dest);

  // Discard pattern is a DLValue.
  if (kind == kDiscardLiteral) {
    // We can only create an implicitly declared value if we have a contextual
    // type to infer from.
    ASTType initializerType = dest.getIfLValueInitializerType();
    if (!initializerType) {
      emitter.emitError(getLoc(),
                        "discard pattern requires an initializing expression");
      return {};
    }
    DLValue result(LLCL::RCRef<DiscardDLValue>::create(initializerType, this));
    return emitter.emitResult(std::move(result), this, dest);
  }

  assert(kind == kSelfLiteral && "Unknown simple literal kind");
  // Self resolves to the type of the enclosing structure type.
  ASTDecl *structDecl = &emitter.declScope;
  while (!isa<StructDeclOp>(*structDecl)) {
    structDecl = structDecl->getParentDecl();
    if (!structDecl) {
      emitter.emitError(getLoc(), "'Self' type may only be used inside a type");
      return {};
    }
  }

  // Once we have the type in question we can just return its Self type as an
  // PValue.  This already includes bound parameters etc.
  assert(structDecl->resolvedness >= DeclResolvedness::signature);
  return emitter.emitResult(structDecl->getSelfType(), this, dest);
}

/// The value of a string is the concatenated value with escapes and quotes
/// removed.
std::string StringLiteralNode::getValue() const {
  std::string result;
  for (auto spelling : spellings)
    result += Lexer::getStringLiteralValue(spelling);
  return result;
}

AnyValue StringLiteralNode::emitIR(ValueDest &dest,
                                   ExprEmitter &emitter) const {
  std::string value = getValue();
  auto attr = StringAttr::get(value, StringType::get(emitter.getContext()));

  // Convert this to an instance of StringLiteral.
  ASTType type =
      emitter.shared.getBuiltinStringLiteralType(emitter.declScope, getLoc());
  return emitter.emitConstructorCall(type, {{AnyValue(attr), this}}, this,
                                     CallSyntax::kImplicitConvert, dest);
}

/// Return true if this is a positional argument with a string literal
/// containing the specified string.
bool CallArgument::isPositionalStringLiteral(StringRef str) const {
  auto *strExpr = dyn_cast<StringLiteralNode>(expr);
  return kind == kPositional && strExpr && strExpr->getValue() == str;
}

/// Emit a reference to a declaration to an AnyValue. If the value is concrete
/// and has a runtime value, `mlirValue` is populated with the corresponding SSA
/// value.
/// FIXME: The `mlirValue` is a hack for closures and should be removed.
static AnyValue emitDeclReference(StringRef spelling, ExprEmitter &emitter,
                                  ArrayRef<ASTDecl *> decls,
                                  const ExprNode *expr, ValueDest &dest,
                                  Capture &capture) {
  emitter.shared.notifyListenerOnRef(decls, spelling, expr);

  // Functions form an address, and may be overloaded.
  if (auto firstCandidate = dyn_cast<LIT::FuncOp>(*decls[0])) {
    ParamBindArrayAttr paramBindings = {};
    // Form an overload set value with all the candidates.
    auto result = ORValue::create(spelling, decls, paramBindings, expr,
                                  CallSyntax::kDirectCall);
    return emitter.emitResult(std::move(result), expr, dest);
  }

  assert(decls.size() == 1 && "Only functions may be overloaded");
  ASTDecl &decl = *decls[0];

  // Aliases form a PValue.
  if (auto param = dyn_cast<AliasDeclOp>(decl)) {
    PValue result = resolveAliasDeclareValue(param, /*bindings=*/{},
                                             emitter.shared, expr->getLoc());
    return emitter.emitResult(result.get(), expr, dest);
  }

  // Use of forward alias references.
  if (auto param = dyn_cast<AliasForwardDeclOp>(decl)) {
    PValue result(ParamDeclRefAttr::get(param.getName(), param.getType()));
    return emitter.emitResult(result, expr, dest);
  }

  // If this is a type declaration, return it as a type.
  if (isa<StructDeclOp>(decl)) {
    PValue result(DeclRefType::get(decl.getSymbolRef()));
    return emitter.emitResult(result, expr, dest);
  }

  // If this is a module or package declaration, form a module reference.
  if (isa<FileModuleOp, PackageOp>(decl)) {
    PValue result(ModuleAttr::get(MetaTypeType::get(decl.getSymbolRef())));
    return emitter.emitResult(result, expr, dest);
  }

  if (auto pvalue = decl.getIfPValue())
    return emitter.emitResult(pvalue, expr, dest);

  // All the declarations below require resolving a dynamic value.
  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(expr);

  // Narrow the decl to a CValue, and dig out the underlying MLIR value so we
  // can check if it is captured in a function.
  CValue value;
  Value mlirValue;
  // 'let' declarations resolve to an SBvalue when they are register_passable.
  if (auto letDecl = dyn_cast<LetRegDeclOp>(decl)) {
    mlirValue = letDecl.getResult();
    value = SBValue(mlirValue);

    // Variable references resolve to an MBValue or LValue addressing the
    // memory.
  } else if (auto var = dyn_cast<VarLetDeclOp>(decl)) {
    // We handle both var and let's as mutable lvalues and let check lifetimes
    // diagnose any problems.  This allows us to handle late-initialized lets.
    mlirValue = var.getResult();
    value = LValue(mlirValue);

    // RValue's and LValues always resolve to their known value.
  } else if (auto rvalue = decl.getIfRValue()) {
    if (auto mrValue = rvalue.getIfMRValue())
      mlirValue = mrValue;
    else
      mlirValue = rvalue.getIfSRValue();
    value = rvalue;
  } else if (auto bvalue = decl.getIfBValue()) {
    if (auto mbValue = bvalue.getIfMBValue())
      mlirValue = mbValue;
    else
      mlirValue = bvalue.getIfSBValue();
    value = bvalue;
  } else if (auto lvalue = decl.getIfLValue()) {
    mlirValue = lvalue;
    value = lvalue;
  } else if (auto globalOp = dyn_cast<GlobalVarDeclOp>(decl)) {
    auto ref = emitter.builder->create<GlobalVarRefOp>(
        emitter.translateLocation(expr->getLoc()), globalOp);
    mlirValue = ref;
    if (globalOp.getIsVar())
      value = SLValue(mlirValue);
    else
      value = MBValue(mlirValue);
  } else {
    emitter.emitError(expr->getLoc(), "use of declaration \"")
        << spelling << "\" as a value isn't supported yet" << expr->getRange();
    return {};
  }
  if (auto slValue = value.getIfSLValue()) {
    if (ASTType(slValue.getRValueType())
            .isRegisterPassable(expr->getLoc(), emitter.shared)) {
      capture =
          Capture(mlirValue, value.getRValueType(), value.getRValueType());
      return value;
    }
  }
  capture = Capture(mlirValue, value.getRValueType(), value.getType());
  return value;
}

AnyValue SyntheticNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  llvm_unreachable("emitIR is undefined for synthetic nodes.");
}

/// Emit IR for an unqualified declaration reference "x" looked up in current
/// context.
AnyValue DeclRefNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  ASTDecl &container = emitter.declScope;

  // Perform a lookup of the specified decl in the current container.
  LookupResult lookup = emitter.shared.lookupAndResolveDecl(
      spelling, getLoc(), container, /*searchParentScopes=*/true);

  // This creates an untyped VarLetDeclOp which is then inferred from its
  // initializer.  `isVar` indicates whether this should be considered mutable.
  auto createVarDecl = [&](OpBuilder &builder, bool isVar,
                           bool isSynth) -> VarLetDeclOp {
    auto contextualType = dest.getIfLValueInitializerType();
    assert(contextualType && "must have contextual type");
    auto loc = getLocation(emitter);
    Type declIRType = POP::PointerType::get(contextualType);
    auto nameAttr = StringAttr::get(loc.getContext(), spelling);
    return builder.create<VarLetDeclOp>(loc, declIRType, nameAttr, isVar,
                                        isSynth);
  };

  // If that lookup failed, but we can synthesize a variable declaration in this
  // scope, do that.  We can only do this if there is a varDeclCursor,
  // indicating that we're in a `def` node, and if we have a contextual type
  // (which tells us we need to emit an LValue).
  if (lookup.isFailure() && emitter.varDeclCursor &&
      dest.getIfLValueInitializerType()) {
    // Use this builder to place any VarLetDeclOps. In Python there is only one
    // scope per function and all variables belong to that scope, so builders
    // should reflect that.
    OpBuilder varDeclBuilder(
        emitter.varDeclCursor->getInsertionBlock(),
        std::next(emitter.varDeclCursor->getInsertionPoint()));
    VarLetDeclOp varDecl = // Marked isSynth to disable warnings.
        createVarDecl(varDeclBuilder, /*isVar=*/true, /*isSynth=*/true);

    // In a normal implicit declaration, we add it to the name table so
    // subsequent uses find this one.
    emitter.getDeclResolver().addFullyResolvedDecl(
        DeclIRValue(varDecl), varDecl.getNameAttr(), getLoc(), &container);
    return emitter.emitResult(SLValue(varDecl), this, dest);
  }

  ArrayRef<ASTDecl *> decls = lookup.getIfSuccess();
  if (decls.empty()) {
    if (lookup.isErroneous())
      return {}; // Error already diagnosed.
    ArrayRef<ASTDecl *> failureDecls = lookup.getIfFailure();
    if (!failureDecls.empty()) {
      // Reject unqualified struct field references.
      if (auto fieldOp = dyn_cast<StructFieldOp>(*failureDecls[0])) {
        emitter.emitError(getLoc(), "cannot access instance field '")
            << spelling << "' directly; did you mean 'self.'?" << getRange()
            << FixIt::insertBeforeToken(getLoc(), "self.");
        return {};
        // Rejected unqualified struct method references.
      } else if (isa<StructDeclOp>(*failureDecls[0]->getParentDecl())) {
        const char *replacement = "self.";
        // References to static methods can always use capital Self.
        if (auto firstCandidate = dyn_cast<FuncOp>(*failureDecls[0]))
          if (firstCandidate.getIsStatic())
            replacement = "Self.";

        // References /from/ static methods can only use capital Self.
        if (auto curFn = dyn_cast<FuncOp>(container))
          if (curFn.getIsStatic())
            replacement = "Self.";

        emitter.emitError(getLoc(), "cannot access method '")
            << spelling << "' directly; did you mean '" << replacement << "'?"
            << getRange() << FixIt::insertBeforeToken(getLoc(), replacement);
        return {};
      }
    }

    // By policy in order to produce a more predictable programming model,
    // implicit declarations of variables are only allowed in `def` contexts,
    // not in `fn`, structs, or top level.
    auto funcContext =
        dyn_cast_or_null<LIT::FuncOp>(emitter.declScope.getIfOperation());
    if (!funcContext || !funcContext.getIsDef()) {
      auto diag = emitter.emitError(getLoc()) << "use of unknown declaration '"
                                              << spelling << "'" << getRange();
      if (funcContext)
        diag << ", 'fn' declarations require explicit variable declarations";
      return {};
    }

    auto diag = emitter.emitError(getLoc()) << getRange();
    if (auto structDecl = dyn_cast<StructDeclOp>(container))
      diag << structDecl.getName() << " has no '" << spelling << "' member";
    else
      diag << "use of unknown declaration '" << spelling << "'";
    return {};
  }

  Capture capture;
  AnyValue result =
      emitDeclReference(spelling, emitter, decls, this, dest, capture);

  if (!capture)
    return result;
  auto nestedFunc =
      getBlockParentOfType<FuncOp>(emitter.builder->getInsertionBlock());
  bool livesInsideNestedFunc = nestedFunc && !nestedFunc.getIsParametric() &&
                               nestedFunc.getParamDeclAttr();
  if (livesInsideNestedFunc) {
    // if we have referenced a function that is not a closure, then there is no
    // state and this is not considered a capture.
    if (!isa<LIT::FuncOp>(*decls[0])) {
      assert(decls.size() == 1 && "Only functions may be overloaded");
      ASTDecl *decl = decls[0];
      if (decl->getParentDecl() && (decl->getParentDecl() != &container))
        emitter.shared.addCaptureToScope(container, decl, capture);
    }
  }
  CValue value = result.getIfCValue();
  Value mlirValue = capture.getMlirValue();

  // If this is a capture inside a nonparametric function, emit a copy.
  if (livesInsideNestedFunc && !nestedFunc.getSignature().isEscaping()) {
    assert(mlirValue && "unexpected PValue");
    if (mlirValue.getParentRegion()->isProperAncestor(
            &nestedFunc.getBodyRegion())) {
      // This is a captured value. Emit a copy and bind the name within the
      // function to the copied value.
      FuncOp parentFunc = nestedFunc->getParentOfType<FuncOp>();
      OpBuilder::InsertionGuard guard(*emitter.builder);
      emitter.builder->setInsertionPoint(nestedFunc);
      // Emit a raw stack allocation.
      auto ptrType = POP::PointerType::get(value.getRValueType());
      Value tmp = emitter.builder->create<POP::StackAllocationOp>(
          parentFunc.getLoc(), ptrType, 1);
      ValueDest copyDest(SLValue(tmp), EC_CaptureCopy);
      DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
      if (DebugInfo::DIScopeAttr funcSpAttr = parentFunc.getLocScope())
        diScopeGuard = emitter.shared.diBuilder->pushScopeGuard(funcSpAttr);
      if (!emitter.emitCopyOfValue({value, this}, copyDest)) {
        copyDest.resetForError();
        return {};
      }
      // Rig the closure formation into emitting a memcpy of the raw value
      // by causing the whole value to cross the closure boundary.
      Value rawBytes =
          emitter.builder->create<POP::LoadOp>(parentFunc.getLoc(), tmp);

      // Redeclare the value inside the closure region using a raw stack
      // allocation. We want the lifetime tracker to ignore this: the object
      // will live inside the closure.
      emitter.builder->setInsertionPointToStart(nestedFunc.getBody());
      Value localDecl = emitter.builder->create<POP::StackAllocationOp>(
          nestedFunc.getLoc(), ptrType, 1);
      // Copy the raw bytes in.
      emitter.builder->create<POP::StoreOp>(nestedFunc.getLoc(), rawBytes,
                                            localDecl);

      // If the parent function was malformed somehow, it may not get added
      // to the symbol table.
      ASTDecl *parentDecl = emitter.getDeclResolver().getDeclForFuncSymbol(
          getFullyResolvedSymbolRef(nestedFunc));
      if (!parentDecl)
        return {};

      // Bind the copy to the name.
      emitter.getDeclResolver().addFullyResolvedDecl(
          DeclIRValue(MBValue(localDecl)), spelling, getLoc(), parentDecl);
      value = MBValue(localDecl);
    }
  }

  return emitter.emitResult(value, this, dest);
}

/// This uses the MLIR parser to turn the specified MLIR type name into an MLIR
/// type.
static ASTType parseMLIRType(StringRef name, const ExprNode *node,
                             SharedState &shared) {
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
    // FIXME(#9621): Need to track the number of bytes read because we pass in
    // more than just the attribute we actually want to parse. This avoids
    // returning an error but is actually just masking the real problem.
    size_t bytesRead;
    result = mlir::parseType(StringRef(tmpBuf).drop_back(), shared.getContext(),
                             &bytesRead);
  }
  if (!result)
    shared.emitError(node->getLoc(), "unknown MLIR type: ")
        << name << node->getRange();

  // The parser is sensitive to certain "builtin" types and expects them to
  // follow certain invariants. For handwritten MLIR types, this is not always
  // guaranteed, so verify them here.
  if (auto sig = dyn_cast_or_null<SignatureType>(result)) {
    // Verify argument conventions.
    for (auto [i, argType, conv] : llvm::enumerate(
             sig.getValueInputs(), sig.getValueInputConventions())) {
      Type type = argType;
      if (sig.isVararg(i)) {
        auto variadic = dyn_cast<VariadicType>(type);
        if (!variadic) {
          shared.emitError(node->getLoc(), "argument #")
              << i
              << " in manually specified signature type should be a "
                 "`!kgen.variadic`";
          return {};
        }
        type = ASTType(variadic.getElementType());
      }
      switch (conv) {
      default:
        break;
      case ValueInputConvention::BorrowedInMem:
      case ValueInputConvention::ByRef:
      case ValueInputConvention::ByRefResult:
      case ValueInputConvention::OwnedInMem:
        if (!isa<POP::PointerType>(type)) {
          shared.emitError(node->getLoc(), "argument #")
              << i
              << " in manually specified signature type should be a "
                 "`!pop.pointer`";
          return {};
        }
        break;
      }
    }
  }
  return result;
}

/// Emit a reference to a stored field with a base that is known not to be a
/// dynamic lvalue.
CValue AttributeRefNode::emitStoredFieldRef(ASTExprAnd<CValue> base,
                                            StructFieldOp fieldOp,
                                            const ExprNode *expr,
                                            ValueDest &dest,
                                            ExprEmitter &emitter) {
  assert(!base.ir.getIfDLValue() &&
         "Dynamic lvalues should already be handled");
  auto mlirLoc = expr->getLocation(emitter);

  // If the base is an stored lvalue, then we can return an lvalue to the
  // field.
  if (SLValue baseLV = base.ir.getIfSLValue()) {
    assert(emitter.builder && "Must have a builder given dynamic base value");
    auto fieldPtr =
        emitter.builder->create<StructGEPOp>(mlirLoc, baseLV, fieldOp);
    return emitter.emitCResult(SLValue(fieldPtr), expr, dest);
  }

  // We know the base.ir is a BValue or CRValue, decay to BValue.
  BValue baseBVal = emitter.emitBValue(base, ValueDest::none());
  if (!baseBVal)
    return {};

  // Keep things in the parameter expression domain if we can.
  if (PValue baseMV = baseBVal.getIfPValue()) {
    auto extractVal = LIT::StructExtractAttr::get(baseMV.get(), fieldOp);
    return emitter.emitCResult(PValue(extractVal), expr, dest);
  }

  // Okay, handle dynamic field references.
  assert(emitter.builder && "Must have a builder given dynamic base value");

  // If the base is an MRValue or MBValue, reference the field as an
  // MBValue so we lazy copy only the piece that is needed in the case of
  // `x.y.z.w`
  if (MBValue baseMBV = baseBVal.getIfMBValue()) {
    auto fieldPtr =
        emitter.builder->create<StructGEPOp>(mlirLoc, baseMBV, fieldOp);
    return emitter.emitCResult(MBValue(fieldPtr), expr, dest);
  }

  // Otherwise, we have an SSA register for the base, which must be an SRValue
  // or SBValue.
  SBValue baseSB = baseBVal.getIfSBValue();
  assert(baseSB && "All cases handled above");
  auto extractVal =
      emitter.builder->create<StructExtractOp>(mlirLoc, baseSB, fieldOp);
  return emitter.emitCResult(SBValue(extractVal), expr, dest);
}

/// Given a base value, emit access to a base value element using getter and
/// setter methods and the provided arguments. If a getter is present on the
/// base type but a setter is not, this method immediately emits a getter call.
/// Otherwise, it returns a SubscriptDLValue for later materializing calls to
/// the getter or setter as appropriate.
static AnyValue
emitGetterSetterAccess(const ExprNode *node, const ExprNode *base,
                       ValueDest &dest, ExprEmitter &emitter, ASTType baseType,
                       StringRef getterName, StringRef setterName,
                       CallSyntax syntax, function_ref<void()> lookupError,
                       ArrayRef<ASTExprAnd<AnyValue>> callArgs) {
  // If there is no getter at all, then this is not a subscriptable type.
  OverloadSet getter(baseType, getterName, node, syntax, emitter.shared,
                     /*no error on failure*/ {});

  // Check for the presence of a setitem but don't provide index values because
  // we don't know what the ultimate element type is.  It may be overloaded and
  // we don't know which candidate to pick until it is actually invoked.
  OverloadSet setter(baseType, setterName, node, syntax, emitter.shared,
                     /*no error on failure*/ {});

  if (getter.isNull() && setter.isNull()) {
    lookupError();
    return {};
  }

  // Ok, we have a getter and/or setter.  Check to see if there is one
  // specific known element type.
  ASTType elementType;
  if (!getter.isNull()) {
    // If we have at least one getter implementation then filter it based on the
    // indices we have.  This will ensure we treat its presence of indication
    // that the type was intended to be subscriptable, but whine about index
    // values and base type if they aren't actually compatible at this usage
    // site.
    PValue getterCallee =
        getter.filterOverloadSet(callArgs, /*allowImplicitConversions=*/true,
                                 /*emitDiagnosticOnFailure=*/true, emitter);
    if (!getterCallee)
      return {};

    // If we /just/ have a getter, emit this as a call to the getter
    // immediately. The getter is allowed to return a reference if it has a
    // physical lvalue.
    if (setter.isNull())
      return emitter.emitIndirectCall(getterCallee, callArgs, dest, node);

    elementType = getterCallee.getType().getSignatureUserResultType();
  } else {
    // If we don't have a getter then check to see if we have a setter.  This is
    // a bit tricky in that the setter candidate set is completely unfiltered.

    // Cannot support overloaded setter.  We could make this more flexible in
    // the future if needed, eg if they have common set values but different
    // indices.
    if (setter.fnDecls.size() != 1) {
      auto diag = emitter.emitError(node->getLoc())
                  << baseType << " has overloaded " << setterName
                  << " implementations, which isn't supported"
                  << node->getRange();
      for (auto candidate : setter.fnDecls)
        diag.attachNote(candidate->getLoc()) << "candidate declared here";
      return {};
    }

    // TODO: This won't handle parameterized setters right, inferring the
    // parameter types.  We should use something like
    // `filterOverloadSetForValueType` or use a dummy value to filter the
    // overload set.
    auto directSymbolAttr = setter.getBoundConstantAttr(emitter);
    if (!directSymbolAttr)
      return {}; // Getter invalid.
    auto sigType = cast<SignatureType>(directSymbolAttr.getType());
    // Check basic sanity.
    size_t setValueIdx = callArgs.size() + sigType.hasMemoryOnlyResult();
    if (sigType.getNumInputs() <= setValueIdx) {
      auto diag = emitter.emitError(node->getLoc())
                  << setterName << " has too few arguments";
      diag.attachNote(setter.fnDecls[0]->getLoc())
          << setterName << " declared here";
      return {};
    }
    elementType = sigType.getValueInputs()[setValueIdx];
    auto setValueConvention = sigType.getInputConvention(setValueIdx);
    if (setValueConvention != ValueInputConvention::OwnedInReg &&
        setValueConvention != ValueInputConvention::BorrowedInReg)
      elementType = elementType.getPointerElementType();
  }

  DLValue result(
      LLCL::RCRef<SubscriptDLValue>::create(callArgs, elementType, node));
  return emitter.emitResult(std::move(result), node, dest);
}

/// Emit a qualified attribute reference to MLIR.  On error, emit an error and
/// return a null value.
AnyValue AttributeRefNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  AnyValue baseAnyVal = base->emitIR(ValueDest::none(), emitter);
  if (!baseAnyVal)
    return {};

  // Handle __adaptive_set.
  if (auto overloads = baseAnyVal.getIfORValue())
    if (attrSpelling == "__adaptive_set")
      return emitter.emitResult(overloads->getAdaptiveSet(emitter), this, dest);

  // Otherwise must have a concrete type.
  CValue baseVal = emitter.emitCValue({baseAnyVal, this}, ValueDest::none());
  if (!baseVal)
    return {};

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
    // If the attribute spelling is empty, we couldn't find a name to look up.
    // This was already diagnosed during initial parsing, so we can just bail
    // here.
    if (attrSpelling.empty())
      return {};

    // If there is no decl, the type is an MLIR type.
    Type baseMLIRType = baseRVType.mlirType;

    // Handle __mlir_op.`xxx` references, lazily synthesizing values when
    // they are referenced.
    if (isa<MagicMLIRAttrType>(baseMLIRType)) {
      PValue result =
          synthesizeMLIRAttrFromString(attrSpelling, getLoc(), emitter.shared);
      return emitter.emitResult(result, this, dest);
    }
    if (isa<MagicMLIROpType>(baseMLIRType)) {
      PValue result = synthesizeMLIROpFromString(attrSpelling, emitter);
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

  // Notify the listener of a member lookup.
  emitter.shared.notifyListenerOnMemberLookup(*typeDecl, getAttributeNameLoc());

  // If the attribute spelling is empty, we couldn't find a name to look up.
  // This was already diagnosed during initial parsing, so we can just bail
  // here.
  if (attrSpelling.empty())
    return {};

  // Handle module or package references.
  if (isa<PackageOp, FileModuleOp>(*typeDecl)) {
    FailureOr<ArrayRef<ASTDecl *>> decls =
        emitter.getDeclResolver().lookupDeclInModule(
            *typeDecl, StringAttr::get(emitter.getContext(), attrSpelling),
            getLoc());
    if (failed(decls))
      return {};
    Capture unused;
    return emitDeclReference(attrSpelling, emitter, *decls, this, dest, unused);
  }

  if (!isa<StructDeclOp>(*typeDecl)) {
    emitter.emitError(getLoc(), "cannot access attribute in type ")
        << baseVal.getType() << base->getRange();
    return {};
  }

  // Find the member being accessed.
  LookupResult lookup =
      emitter.shared.lookupAndResolveDecl(attrSpelling, getLoc(), *typeDecl,
                                          /*searchParentScopes=*/false);
  ArrayRef<ASTDecl *> memberDecls = lookup.getIfSuccess();
  if (memberDecls.empty()) {
    // The struct has no static member of the required name, but try to look for
    // dynamic lookup attribute methods on the type.
    auto lookupError = [&] {
      // If the error hasn't been diagnosed, handle it now.
      if (lookup.isFailure())
        emitter.emitError(getLoc()) << baseRVType << " value has no attribute '"
                                    << attrSpelling << "'" << getRange();
    };

    // Emit the value as a StringLiteral.
    ASTType type =
        emitter.shared.getBuiltinStringLiteralType(emitter.declScope, getLoc());

    auto attr =
        StringAttr::get(attrSpelling, StringType::get(emitter.getContext()));
    ValueDest keyDest(EC_AttributeRefBase);
    AnyValue key =
        emitter.emitConstructorCall(type, {{AnyValue(attr), this}}, this,
                                    CallSyntax::kImplicitConvert, keyDest);
    if (!key)
      return {};

    SmallVector<ASTExprAnd<AnyValue>> callArgs = {{baseVal, base}, {key, base}};
    return emitGetterSetterAccess(
        this, base, dest, emitter, baseRVType, "__getattr__", "__setattr__",
        CallSyntax::kAttribute, lookupError, callArgs);
  }
  emitter.shared.notifyListenerOnRef(memberDecls, attrSpelling, this);

  // Handle method references, which might be overloaded.
  if (auto fnOp = dyn_cast<LIT::FuncOp>(*memberDecls[0])) {
    // Get a symbol for the underlying function.
    auto result = ORValue::create(attrSpelling, memberDecls,
                                  baseRVType.getParamBindings(), this,
                                  CallSyntax::kDirectCall);

    // If the callee is a static method, we can directly reference it
    // without binding a self parameter.  If this is an instance method, we
    // bind the base value and the symbol together into a callable.
    // FIXME: This isn't handling overloaded static/non-static methods
    // correctly.  What is the actual behavior we want for static methods?
    // Maybe we don't allow overloading static and non-static methods with
    // the same name?
    if (!fnOp.getIsStatic() && !hasTypeBase) {
      result->baseValue = {baseVal, base};
      result->syntax = CallSyntax::kMethodCall;
    }
    return emitter.emitResult(std::move(result), this, dest);
  }

  assert(memberDecls.size() == 1 && "only methods may be overloaded");
  ASTDecl &memberDecl = *memberDecls[0];

  // Parameters form a meta-value.
  if (auto param = dyn_cast<AliasDeclOp>(memberDecl)) {
    PValue result = resolveAliasDeclareValue(
        param, baseRVType.getParamBindings(), emitter.shared, getLoc());
    return emitter.emitResult(result.get(), this, dest);
  }

  // If the field is a variable, emit a reference to it.
  if (auto fieldOp = dyn_cast<StructFieldOp>(memberDecl)) {
    if (hasTypeBase) {
      emitter.emitError(getLoc(), "cannot access instance field '")
          << attrSpelling << "' without an instance of " << baseRVType
          << getRange();
      return {};
    }

    // We know that baseVal is a CValue, so handle all the cases.

    // If the base is a DLValue, we need to emit this as a projected DLValue.
    // This allows to emit a get and/or set as needed.
    if (DLValue baseLV = baseVal.getIfDLValue()) {
      // The base is a known DeclRefType because we got the ASTDecl from it.
      ASTType elementType =
          fieldOp.getReboundType(cast<DeclRefType>(baseRVType.mlirType));
      DLValue result(LLCL::RCRef<StoredAttributeRefDLValue>::create(
          ASTExprAnd<DLValue>{baseLV, base}, fieldOp, elementType, this));
      return emitter.emitResult(std::move(result), this, dest);
    }

    // Otherwise, emit the stored field reference.
    return emitStoredFieldRef({baseVal, base}, fieldOp, this, dest, emitter);
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
  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(&call);

  // Emit all the arguments so we can encode them as SSA values.
  SmallVector<Value> opOperands;
  for (CallArgument argument : call.args) {
    if (argument.kind != CallArgument::kPositional) {
      emitter.emitError(argument.getLoc(),
                        "MLIR operators only support position arguments");
      return {};
    }
    Value value = emitter.emitExprSRValue(argument.expr, EC_MLIRMagic);
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
      // We expect either a single type or `None`.
      if (isa<NoneAttr>(attr.getValue())) {
      } else if (auto value = dyn_cast<TypedAttr>(attr.getValue())) {
        if (!isa<MLIRTypeType>(value.getType())) {
          emitter.emitError(call.getLoc(), "_type value is not a type");
          return {};
        }
        state.types.push_back(ASTType(value));
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
      results.front()->setIRValue(PValue(BoolAttr::get(context, false)));
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
            DictionaryAttr::get(context, state.attributes),
            state.getRawProperties(), state.regions, state.types)))
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

  for (auto type : state.types)
    if (!ASTType(type).isRegisterPassable(call.getLoc(), emitter.shared)) {
      emitter.emitError(call.getLoc())
          << ASTType(type)
          << " cannot be returned directly from __mlir_op as it is not a "
             "'@register_passable' types";
      return {};
    }

  // Check for an unregistered operation, because otherwise MLIR will crash when
  // assertions are enabled.
  if (!state.name.getDialect() &&
      !emitter.getContext()->allowsUnregisteredDialects()) {
    emitter.emitError(call.getLoc(), "use of unregistered MLIR operation ")
        << unboundOp.getName() << call.getRange();
    return {};
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
  if (resultOp->getNumResults() == 0)
    return PValue(emitter.shared.getNoneAttr());

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
    Type foldedType;
    // If the result was some other value that already exists, use it.
    if (auto val = dyn_cast<Value>(folded)) {
      if (val.getType() == resultOp->getResult(0).getType()) {
        resultOp->erase();
        return SRValue(val);
      }
      foldedType = val.getType();
    } else {
      // If it is a constant, make an PValue result.
      auto attr = cast<TypedAttr>(cast<Attribute>(folded));
      if (attr.getType() == resultOp->getResult(0).getType()) {
        resultOp->erase();
        return PValue(attr);
      }
      foldedType = val.getType();
    }
    emitter.emitError(call.getLoc())
        << unboundOp.getName() << " operation folded to result type "
        << ASTType(foldedType) << " but we expected it to be "
        << ASTType(resultOp->getResult(0).getType()) << call.getRange();
    return {};
  }

  // If folding failed, return the operation normally.
  return SRValue(resultOp->getResult(0));
}

AnyValue CallNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  AnyValue calleeVal = emitter.emitExpr(callee, EC_CallCalleeValue);
  if (!calleeVal)
    return {};

  // If this is the invocation of an unbound MLIR operator, bind it into an
  // actual operator!
  if (auto mValue = calleeVal.getIfPValue()) {
    if (auto unboundOp = dyn_cast<UnboundMLIROperationAttr>(mValue.get())) {
      AnyValue result = emitMLIROperatorCall(*this, unboundOp, emitter);
      return emitter.emitResult(result, this, dest);
    }
  }

  /// Emit all the operands that we'll need.
  SmallVector<ASTExprAnd<AnyValue>> operands;
  for (CallArgument arg : args) {
    if (arg.kind == CallArgument::kKeyword) {
      emitter.emitError(arg.getLoc(),
                        "keyword arguments are not supported yet");
      return {};
    }
    if (arg.kind != CallArgument::kPositional) {
      emitter.emitError(arg.getLoc(),
                        "unpacked arguments are not supported yet");
      return {};
    }
    ExprNode *expr = arg.expr;
    operands.push_back({expr->emitIR(ValueDest::none(), emitter), expr});
    if (!operands.back())
      return {};
  }

  // If the callee is a type value (as in `T()` or `T[123]()`), then this is an
  // invocation of the initializer for the type.
  if (ASTType calledType = calleeVal.getIfTypeValue()) {
    if (!calledType.getDecl(emitter.shared)) {
      emitter.emitError(getLoc(), "cannot use initializer syntax on MLIR type ")
          << calledType << callee->getRange();
      return {};
    }

    return emitter.emitConstructorCall(calledType, operands, this,
                                       CallSyntax::kTypeCall, dest);
  }

  // If this is an overloaded operand, resolve it and call the result.
  if (auto overloads = calleeVal.getIfORValue()) {
    overloads->expr = this;
    return overloads->emitCall(operands, dest, emitter);
  }

  // Otherwise, we must have a concrete RValue, emit an indirect call.
  auto crVal = calleeVal.getIfCValue();
  return emitter.emitIndirectCall(crVal, operands, dest, this);
}

AnyValue SliceNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // Emit the slice index expressions as values. Any one of them could be null
  // if they were not present. Emit them as `None` in that case.
  SmallVector<ASTExprAnd<AnyValue>> ctorArgs;
  auto emitOrNone = [&](ExprNode *const expr, llvm::SMLoc loc) -> ParseResult {
    if (!expr) {
      ctorArgs.push_back({NoneAttr::get(emitter.getContext()), this});
      return success();
    }
    AnyValue value = emitter.emitExpr(expr, EC_SliceIndex);
    if (!value)
      return failure();
    ctorArgs.push_back({value, expr});
    return success();
  };

  // The location of the first colon is always set. The location of the second
  // colon is set if there is a stride.
  if (emitOrNone(lower, getLoc()) || emitOrNone(upper, colon1Loc) ||
      emitOrNone(stride, stride ? colon2Loc : colon1Loc))
    return {};

  // Lookup the builtin slice type and emit a constructor call.
  ASTType type =
      emitter.shared.getBuiltinSliceType(emitter.declScope, getLoc());
  return emitter.emitConstructorCall(type, ctorArgs, this,
                                     CallSyntax::kImplicitConvert, dest);
}

/// Given a value of type type, substitute parameters into the type, producing
/// a more concrete type.  This syntax is `SomeType[1, 4, Int]`.
static PValue substituteParametersIntoUserDefinedType(
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
    auto indexVal = emitter.emitExprPValue(indexExpr, EC_TypeParamValue);
    if (!indexVal)
      return {};
    paramBindings.add(indexExpr, indexVal.get());
  }

  // Check the bindings.
  ssize_t incorrectBindingNo = 0;
  ASTType incorrectBindingExpectedType;
  SmallVector<Type> paramTypes;
  for (ParamDeclAttr decl : structOp.getInputParams())
    paramTypes.push_back(decl.getType());
  auto [bindingValuesAttr, _] = paramBindings.verifyBindings(
      paramTypes, structOp.getInputParamsAttr(), structOp.getName(),
      subscript.getLoc(), incorrectBindingNo, incorrectBindingExpectedType,
      emitter, structOp, structOp.getParamVarargs());
  if (!bindingValuesAttr)
    return {};

  SmallVector<ParamBindAttr> bindingValues;
  for (auto [decl, value] :
       llvm::zip(structOp.getInputParams(), bindingValuesAttr))
    bindingValues.push_back(ParamBindAttr::get(decl.getName(), value));

  // Ok, we succeeded at reparameterizing the type.
  return PValue(DeclRefType::get(
      typeDecl.getSymbolRef(),
      ParamBindArrayAttr::get(structOp.getContext(), bindingValues)));
}

/// When subscripting a callable with a bound symbol (i.e. a direct method call
/// or call to a method), apply parameter bindings to it.
static ORValue bindParamValuesToDirectCall(ORValue value,
                                           ArrayRef<ExprNode *> indices,
                                           ExprEmitter &emitter) {
  // If the indices are a single () expression, then we treat this as having
  // no parameters.  This is used with arrow expressions to allow `f[() -> x]`.
  if (indices.size() == 1 && indices[0]->getWithoutParens()->isEmptyTuple())
    return value;

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

    auto val = emitter.emitExprPValue(idx, EC_CallParamValue, paramType);
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

AnyValue SubscriptNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // Subscripting a generic function binds the parameter expressions.
  auto baseAnyValue = base->emitIR(ValueDest::none(), emitter);
  if (!baseAnyValue)
    return {};

  // If the baseAnyValue has a bound callable symbol, then this is applying
  // (more?) parameter expressions to bind its parameters.
  if (auto overloads = baseAnyValue.getIfORValue()) {
    auto result = bindParamValuesToDirectCall(overloads, indices, emitter);
    return emitter.emitResult(result, this, dest);
  }

  // Otherwise, this must be a concrete node to be able to further subscript it.
  CValue baseValue =
      emitter.emitCValue({baseAnyValue, base}, ValueDest::none());
  if (!baseValue)
    return {};

  if (auto callableMVal = baseValue.getIfPValue()) {
    if (auto sig = dyn_cast<SignatureType>(callableMVal.getType().mlirType)) {
      // If this is a signature-type PValue callable, this is binding parameter
      // values to a call.
      SmallVector<TypedAttr> bindOperands({callableMVal.get()});
      if (indices.size() != sig.getNumInputParams()) {
        emitter.emitError(getLoc(), "parametric callable expected ")
            << sig.getNumInputParams() << " parameter"
            << plural(sig.getNumInputParams()) << getIndexRange();
        return {};
      }
      for (auto [idx, type] : llvm::zip(indices, sig.getInputParamTypes())) {
        bindOperands.push_back(
            emitter.emitExprPValue(idx, EC_CallParamValue, type));
        if (!bindOperands.back())
          return {};
      }

      PValue result(ParamOperatorAttr::get(POC::BindSignature, bindOperands));
      return emitter.emitResult(result, this, dest);
    }
  }

  // If the sub-value is an unbound Type, try binding things to it!
  if (Type typeValue = baseValue.getIfTypeValue()) {
    // Handle user-defined types.
    if (auto declRef = dyn_cast<DeclRefType>(typeValue)) {
      PValue result =
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
      PValue attr =
          synthesizeMLIRAttrFromString(result, getLoc(), emitter.shared);
      return emitter.emitResult(attr, this, dest);
    }
  }

  // Check for attribute bindings to an MLIR operation.
  if (auto value = baseValue.getIfPValue()) {
    if (auto unboundOperator =
            dyn_cast<UnboundMLIROperationAttr>(value.get())) {
      PValue result =
          bindAttributesToMLIROperatorCall(*this, unboundOperator, emitter);
      return emitter.emitResult(result, this, dest);
    }
  }

  // Otherwise, if there is no symbol, it is just an LValue or RValue being
  // subscript, invoking a dynamic subscript.

  // Emit each of the index values, which will be passed to the __getitem__ and
  // __setitem__ calls.
  SmallVector<ASTExprAnd<AnyValue>> indexValues;
  indexValues.push_back({baseValue, base});
  for (ExprNode *index : indices) {
    indexValues.push_back({index->emitIR(ValueDest::none(), emitter), index});
    if (!indexValues.back())
      return {};
  }

  // TODO: If we have multiple indexes, package up the values in a tuple value
  // and try to see if this works.
  if (indexValues.size() > 2) {
    // TODO(Tuples). need tuples :-)
  }

  // Okay, we're doing a normal value subscript.  Check for compatible
  // __getitem__ and __setitem__ implementations.
  ASTType baseType = baseValue.getRValueType();

  // Check if we are subscripting a variadic. Emit `pop.variadic.get`.
  // FIXME(#13015): We shouldn't need this code. Variadic arguments should emit
  // a standard library type that implements `__getitem__` and `__setitem__`.
  if (auto variadic = dyn_cast<VariadicType>(baseType.mlirType)) {
    // Attempt to convert the index.
    if (indexValues.size() != 2) {
      emitter.emitError(getLoc())
          << "variadic can only be subscripted with a single index";
      return {};
    }
    ValueDest indexDest(EC_Subscript);
    const ExprNode *indexExpr = indexValues.back().expr;
    CValue index =
        emitter.emitNamedMethodCall("__index__", indexValues.back(), indexDest,
                                    CallSyntax::kMethodCall, indexExpr);
    if (!index)
      return {};
    // Convert the index value to an MLIR index type.
    indexDest = {EC_Subscript};
    CValue mlirIndex = emitter.emitNamedMethodCall(
        "__mlir_index__", {{index, indexExpr}}, indexDest,
        CallSyntax::kMethodCall, indexExpr);
    if (!mlirIndex)
      return {};
    // Inside a parameter context, emit a parameter operator.
    if (!emitter.builder) {
      return ParamOperatorAttr::get(
          POC::VariadicGet,
          {emitter.emitPValue(indexValues.front(), EC_Subscript),
           mlirIndex.getIfPValue()});
    }
    // Otherwise, emit an MLIR operation.
    Value value = emitter.builder->create<POP::VariadicGetOp>(
        emitter.translateLocation(getLoc()),
        emitter.emitSRValue(indexValues.front(), EC_Subscript),
        emitter.emitSRValue({mlirIndex, indexExpr}, EC_Subscript));
    // FIXME: Should not be doing a bare `!pop.pointer` type check.
    if (auto ptrType = dyn_cast<POP::PointerType>(
            ASTType(variadic.getElementType()).mlirType)) {
      if (!ASTType(ptrType.getElementType())
               .isRegisterPassable(getLoc(), emitter.shared))
        return emitter.emitResult(MRValue(value), this, dest);
    }
    return emitter.emitResult(SRValue(value), this, dest);
  }

  auto lookupError = [&] {
    emitter.emitError(getLoc())
        << baseType
        << " is not subscriptable, it does not implement the "
           "`__getitem__`/`__setitem__` methods"
        << base->getRange();
  };
  return emitGetterSetterAccess(
      this, base, dest, emitter, baseType, "__getitem__", "__setitem__",
      CallSyntax::kSubscript, lookupError, indexValues);
}

AnyValue SubscriptArrowNode::emitIR(ValueDest &dest,
                                    ExprEmitter &emitter) const {
  // Subscripting a generic function binds the parameter expressions.
  RValue baseValue = emitter.emitExprRValue(base, EC_SubscriptBase);
  if (!baseValue)
    return {};

  // If the baseValue has a bound callable symbol, then this is applying (more?)
  // meta values to bind its parameters.
  auto overloads = baseValue.getIfORValue();
  if (!overloads) {
    assert(baseValue.getIfCRValue() && "Must be CRValue if not ORValue");
    emitter.emitError(arrowLoc, "invalid '->' when subscripting type ")
        << baseValue.getIfCRValue().getRValueType() << getRange();
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

AnyValue ParenNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  return subExpr->emitIR(dest, emitter);
}

/// Both tuple literals and list literals are emitted as heterogenous sequences,
/// with each element type encoded in a variadic type parameter.
static AnyValue emitHeterogenousSequence(ValueDest &dest, ExprEmitter &emitter,
                                         ASTType type, const ExprNode *node,
                                         ArrayRef<ExprNode *> exprs) {
  // If we failed to look up the tuple/list type, fail.
  if (!type || type.isTypeCheckErrorType())
    return {};

  // Emit each of the tuple elements.
  SmallVector<ASTExprAnd<AnyValue>> elements;
  bool allEltsLValue = true;
  bool allEltsTypes = true;
  for (ExprNode *expr : exprs) {
    auto exprVal = emitter.emitExpr(expr, EC_TupleElement);
    if (!exprVal)
      return {};
    allEltsLValue &= !exprVal.getIfLValue().isNull();
    allEltsTypes &= !exprVal.getIfTypeValue().isNull();

    elements.push_back({std::move(exprVal), expr});
  }

  // If this is a tuple with all LValue elements, return a DLValue since we can
  // assign into this expression.
  // TODO: Add support for list LValues as well.
  if (allEltsLValue && isa<TupleNode>(node)) {
    SmallVector<Type> typeElts;
    for (auto elt : elements)
      typeElts.push_back(elt.ir.getIfLValue().getRValueType());
    type = emitter.shared.getBuiltinTupleInstantion(emitter.declScope,
                                                    node->getLoc(), typeElts);
    if (type.isTypeCheckErrorType())
      return {};
    DLValue result(LLCL::RCRef<TupleDLValue>::create(elements, type, node));
    return emitter.emitResult(std::move(result), node, dest);
  }

  // If this tuple has all type elements (and is not empty) then we can form a
  // tuple type.  Note that we do not treat () as a type here, it is considered
  // a value, and the ambiguity is handled in emitExprType.
  if (allEltsTypes && isa<TupleNode>(node) && !elements.empty()) {
    SmallVector<Type> typeElts;
    for (auto elt : elements)
      typeElts.push_back(elt.ir.getIfTypeValue());

    auto result = emitter.shared.getBuiltinTupleInstantion(
        emitter.declScope, node->getLoc(), typeElts);
    if (type.isTypeCheckErrorType())
      return {};
    return emitter.emitResult(PValue(result), node, dest);
  }

  // The ASTType will carry around parameters bound, we want to unbind them so
  // they can be inferred from the elements.
  type = DeclRefType::get(type.getDecl(emitter.shared)->getSymbolRef());

  // Emit a call to the builtin type constructor as an implicit conversion.
  // The type parameters are inferred from the element types.
  return emitter.emitConstructorCall(type, elements, node,
                                     CallSyntax::kImplicitConvert, dest);
}

AnyValue TupleNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // Lookup the builtin Tuple type, in order to call its constructor.
  ASTType type =
      emitter.shared.getBuiltinTupleType(emitter.declScope, getLoc());
  return emitHeterogenousSequence(dest, emitter, type, this, exprs);
}

AnyValue ListNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // Lookup the builtin ListLiteral type, in order to call its constructor.
  ASTType type =
      emitter.shared.getBuiltinListLiteralType(emitter.declScope, getLoc());
  return emitHeterogenousSequence(dest, emitter, type, this, exprs);
}

AnyValue DictionaryNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  emitter.emitError(getLoc(), "TODO: cannot emit dictionary literals yet")
      << getRange();
  return {};
}

/// Emit a DictSubscriptNode when the base is a Type expression.
AnyValue DictSubscriptNode::emitTypeSubscriptIR(ASTType initType,
                                                ValueDest &dest,
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

  // Perform parameter substitution if there are input parameters.
  ParserParamEvaluator paramEvaluator(emitter.getDeclResolver(),
                                      initType.getParamBindings());

  // Build a mapping of field names to field decls for fast lookup.
  SmallDenseMap<StringAttr, StructFieldOp> fieldNameMap;
  for (StructFieldOp field : structOp.getFieldDecls())
    fieldNameMap[field.getNameAttr()] = field;

  // If this is a memory-only struct, initialize the fields into the result
  // buffer.
  if (!structOp.isRegisterPassable()) {
    emitter.emitError(getLoc(),
                      "this initializer syntax may only be used with "
                      "'@register_passable' values; use '__init__' instead")
        << getRange();
    return {};
  }

  DenseMap<StringAttr, ASTExprAnd<AnyValue>> fieldMapping;
  bool allInitializersPValues = true;
  for (auto &keyValue : indices->values) {
    const ExprNode *valueExpr = keyValue.second;
    // We don't support `**dict` syntax.
    if (!keyValue.first) {
      emitter.emitError(valueExpr->getLoc(),
                        "cannot expand into initializer list")
          << valueExpr->getRange();
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
    auto fieldNameDecls = emitter.shared.lookupAndResolveDecl(
        fieldNameAttr, valueExpr->getLoc(), *decl,
        /*searchParentScopes=*/false);
    if (!fieldNameDecls.isSuccess()) {
      if (!fieldNameDecls.isErroneous())
        emitter.emitError(keyValue.first->getLoc())
            << initType << " has no field named " << fieldNameAttr
            << keyValue.first->getRange() << base->getRange();
      return {};
    }

    auto field = fieldNameMap[fieldNameAttr];
    if (!field) {
      emitter.emitError(keyValue.first->getLoc(), "")
          << initType << " has no field named " << fieldNameAttr
          << valueExpr->getRange();
      return {};
    }

    // For register values, make sure we convert to the right dest field type.
    auto fieldType = paramEvaluator.getReboundType(field.getType());
    AnyValue value = emitter.emitExpr(valueExpr, EC_FieldInitValue, fieldType);
    if (!value)
      return {};

    // Keep track of whether everything is a PValue.
    if (allInitializersPValues && !value.getIfPValue())
      allInitializersPValues = false;

    auto mapResult = fieldMapping.insert({fieldNameAttr, {value, valueExpr}});
    if (!mapResult.second) {
      emitter.emitError(keyValue.first->getLoc(), "field ")
          << fieldNameAttr << " specified multiple times"
          << keyValue.first->getRange() << base->getRange()
          << mapResult.first->second.expr->getRange();
      return {};
    }
  }

  // If it is register-passable, we build a list of field values+names.  For
  // memory-only, we just check that each value got emitted.
  SmallVector<StringAttr> fieldNames;
  SmallVector<Value> fieldSRValues;
  SmallVector<std::tuple<StringAttr, TypedAttr>> fieldParamValues;
  for (StructFieldOp field : structOp.getFieldDecls()) {
    ASTExprAnd<AnyValue> fieldVal = fieldMapping[field.getNameAttr()];
    if (!fieldVal) {
      emitter.emitError(indices->rbraceLoc, "no value for field ")
          << field.getNameAttr() << " specified";
      return {};
    }

    // If all the initializers are PValues, we can emit this as a StructAttr.
    if (allInitializersPValues) {
      fieldParamValues.push_back(
          {field.getNameAttr(), fieldVal.ir.getIfPValue()});
      continue;
    }

    // If the any initializers required emitting a load sequence, emit the rest
    auto srValue = emitter.emitSRValue(fieldVal, EC_FieldInitValue);
    if (!srValue)
      return {};
    fieldNames.push_back(field.getNameAttr());
    fieldSRValues.push_back(srValue);
  }

  // If all the fields are PValues, form a new PValue.
  if (allInitializersPValues) {
    auto result = StructAttr::get(emitter.getContext(), fieldParamValues,
                                  cast<DeclRefType>(initType.mlirType));
    return emitter.emitResult(result, this, dest);
  }

  // Now that we have all the values, generate the initializers for
  // StructCreate.
  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(this);

  // For register-passable types, bundle all the values up and return them.
  auto result = SRValue(emitter.builder->create<StructCreateOp>(
      getLocation(emitter), initType.mlirType, fieldSRValues,
      StringArrayAttr::get(emitter.getContext(), fieldNames)));
  return emitter.emitResult(result, this, dest);
}

AnyValue DictSubscriptNode::emitIR(ValueDest &dest,
                                   ExprEmitter &emitter) const {
  // Subscripting a type constructs it with lit.struct.create.
  ASTType typeValue = emitter.emitExprType(base);
  if (!typeValue)
    return {};

  return emitTypeSubscriptIR(typeValue, dest, emitter);
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
#include "KGEN/LITDialect/SpecialFunctions.def"
  // If everything fails we should return "normal".
  return SpecialFunctionInfo::get(SpecialFunctionKind::kNormal);
}

/// Emit the binary operation (with a `lhs`, `rhs` and `kind`) as a special
/// function call.
/// A special function call is one where the` kind` must corresponds to a valid
/// SpecialFunctionInfo when we invoke getOpSpecialFunctions(kind).
/// `callExpr` is the call like expression that results in the call.
//
/// This is an utility function to share code between BinOpNone and
/// ChainedCmpOpNode since the latter is a sequence of binary operations.
static AnyValue emitBinOpCall(ASTExprAnd<AnyValue> lhs,
                              ASTExprAnd<AnyValue> rhs, ExprNode::Kind kind,
                              ValueDest &dest, const ExprNode *callExpr,
                              ExprEmitter &emitter) {

  // FIXME: We currently hack in support for _mlir_type equality comparison
  // until we have proper metatypes.
  if (kind == ExprNode::Kind::kCmpEQ) {
    PValue lhsParam = lhs.ir.getIfPValue(), rhsParam = rhs.ir.getIfPValue();
    if (lhsParam && rhsParam &&
        isa<MLIRTypeType>(lhsParam.getType().mlirType) &&
        isa<MLIRTypeType>(rhsParam.getType().mlirType)) {
      return ParamOperatorAttr::get(POC::EQ, {lhsParam.get(), rhsParam.get()});
    }
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnInfo = getOpSpecialFunctions(kind, /*isReversed=*/false);
  assert(specialFnInfo.kind != SpecialFunctionKind::kNormal);
  ASTExprAnd<AnyValue> argValues[] = {lhs, rhs};

  // Check to see if we have a forward version of this function on the primary
  // receiver.
  if (auto lhsCV = lhs.ir.getIfCValue()) {
    if (PValue callee = OverloadSet::lookup(
            lhsCV.getRValueType(), specialFnInfo.name, argValues, callExpr,
            CallSyntax::kOperator, emitter,
            /*no error*/ {}))
      return emitter.emitIndirectCall(callee, argValues, dest, callExpr);
  }

  // Check to see if we have the reverse version of this operator.
  auto reversedFnInfo = getOpSpecialFunctions(kind, /*isReversed=*/true);
  if (reversedFnInfo.kind != SpecialFunctionKind::kNormal) {
    // Swap the operand order.
    std::swap(argValues[0], argValues[1]);
    if (auto rhsCV = rhs.ir.getIfCValue()) {
      if (PValue callee = OverloadSet::lookup(
              rhsCV.getRValueType(), reversedFnInfo.name, argValues, callExpr,
              CallSyntax::kReversedOperator, emitter,
              /*no error*/ {}))
        return emitter.emitIndirectCall(callee, argValues, dest, callExpr);
    }

    // Swap these back so we emit the right error.
    std::swap(argValues[0], argValues[1]);
  }

  // Emit an error complaining about the forward version of the operator.
  return emitter.emitNamedMethodCall(specialFnInfo.name, argValues, dest,
                                     CallSyntax::kOperator, callExpr);
}

/// Emit a simple assignment statement. Python evaluates the RHS of an
/// assignment before the LHS, as seen in things like:
///    def test1(): print("test1"); return 0
///    def test2(): print("test2"); return 1
///    a[test1()] = test2()
///  ==> test2; test1
///
/// The walrus := operator in Python requires the left side to be a simple
/// identifier, but Mojo allows arbitrary lvalues like the assign stmt.
AnyValue BinOpNode::emitAssign(ValueDest &dest, ExprEmitter &emitter) const {
  // In an assignment, we emit the RHS into the LHS as its context.  This is
  // required to enable the 'implicit declaration' behavior in a def and to
  // support patterns.
  ValueDest assignDest(lhs, EC_Assignment);
  auto resultValue = rhs->emitIR(assignDest, emitter);
  if (!resultValue) {
    assignDest.resetForError();
    return {};
  }

  // To support the walrus operator and chained assignment like `x = y = 1`, the
  /// assignment operation returns a borrowed version of the dest value.
  return emitter.emitResult(resultValue, this, dest);
}

/// Emit a inplace assignment statement like `x += y`. Python evaluates the RHS
/// of an assignment before the LHS, as seen in things like:
///    def test1(): print("test1"); return 0
///    def test2(): print("test2"); return 1
///    a[test1()] += test2()
///  ==> test1; test2
AnyValue BinOpNode::emitInplace(ValueDest &dest, ExprEmitter &emitter) const {
  AnyValue lhsRep;
  RValue rhsRep;

  // Inplace operations evaluate the LHS first, so emit the LHS pattern as an
  // lvalue.
  LValue lhsLV = emitter.emitExprLValue(lhs, EC_InplaceBinOpDest);
  if (!lhsLV)
    return {};

  // Then emit the right side.
  AnyValue rhsV = emitter.emitExpr(rhs, EC_OperatorOperandValue);
  if (!rhsV)
    return {};

  // Emit the call to the operator function like `__iadd__`.
  return emitBinOpCall({lhsLV, lhs}, {rhsV, rhs}, kind, dest, this, emitter);
}

AnyValue BinOpNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // Handle weird binary operators specially if we have them.
  if (kind == kBoolAnd || kind == kBoolOr) // `x and y`, `x or y`
    return emitAndOr(dest, emitter);
  if (kind == kAssign || kind == kWalrus) // `x = y` and `x := y`
    return emitAssign(dest, emitter);
  if (isAssignmentStmt()) // `x += y`
    return emitInplace(dest, emitter);

  // Othewise we emit the LHS followed by the RHS.
  AnyValue lhsRV = emitter.emitExpr(lhs, EC_OperatorOperandValue);
  AnyValue rhsRV = emitter.emitExpr(rhs, EC_OperatorOperandValue);
  if (!lhsRV || !rhsRV)
    return {};

  return emitBinOpCall({lhsRV, lhs}, {rhsRV, rhs}, kind, dest, this, emitter);
}

/// Given two values that need to match, try to coerce one to the other if they
/// disagree on type.  This emits an error (when loc is non-null) and returns
/// failure if the request is ambiguous or impossible.
template <typename ValueType>
static ParseResult
coerceTypesToEachOther(SMLoc loc, ValueType &lhs, const ExprNode *lhsExpr,
                       ValueType &rhs, const ExprNode *rhsExpr,
                       ExprEmitter &emitter,
                       std::function<ValueType(ASTExprAnd<AnyValue> value,
                                               ASTType destType, bool isLHS)>
                           convert) {
  if (!lhs || !rhs)
    return failure();

  ASTType lhsType = lhs.getRValueType(), rhsType = rhs.getRValueType();

  // If the types already match, then we're done.
  if (lhsType.isEqualCanon(rhsType))
    return success();

  bool lhsConvertibleToRHS =
      emitter.canImplicitlyConvertToType({lhs, lhsExpr}, rhsType);
  bool rhsConvertibleToLHS =
      emitter.canImplicitlyConvertToType({rhs, rhsExpr}, lhsType);
  if (lhsConvertibleToRHS && !rhsConvertibleToLHS) {
    lhs = convert({lhs, lhsExpr}, rhsType, /*isLHS*/ true);
    return failure(!lhs);
  }

  if (!lhsConvertibleToRHS && rhsConvertibleToLHS) {
    rhs = convert({rhs, rhsExpr}, lhsType, /*isLHS*/ false);
    return failure(!rhs);
  }

  // Otherwise we have an error.  If we have no source location, we just return
  // failure without returning an error.
  if (!loc.isValid())
    return failure();

  if (!lhsConvertibleToRHS && !rhsConvertibleToLHS) {
    emitter.emitError(loc, "value of type ")
        << lhsType << " is not compatible with value of type " << rhsType
        << lhsExpr->getRange() << rhsExpr->getRange();
    return failure();
  }

  auto diag = emitter.emitError(loc, "ambiguous merge: left value has type ")
              << lhsType << " and right value has type " << rhsType
              << ", and both convert to each other" << lhsExpr->getRange()
              << rhsExpr->getRange();
  diag.attachNote(loc) << "you could disambiguate by casting the left value to "
                       << rhsType << lhsExpr->getRange();
  diag.attachNote(loc) << "or cast the right value to " << lhsType
                       << rhsExpr->getRange();
  return failure();
}

/// This method emits the `x and y`, `x or y` operators.  These are
/// interesting in Python:
///
///   "Note that neither `and` nor `or` restrict the value and type they
///   return to False and True, but rather return the last evaluated argument.
///   This is sometimes useful, e.g., if `s` is a string that should be
///   replaced by a default value if it is empty, the expression `s or 'foo'`
///   yields the desired value.
///
/// Unlike Python, we have static types that could disagree.  Our policy on
/// this is to either return the pre-Bool'ified value when their types agree (or
/// can be converted to each other unambiguously) or to return the common Bool
/// type if they don't.
///
AnyValue BinOpNode::emitAndOr(ValueDest &dest, ExprEmitter &emitter) const {
  Location ifLoc = getLocation(emitter);

  if (!emitter.builder) {
    PValue lhsPVal = emitter.emitExprPValue(lhs, EC_OperatorOperandValue);
    RValue lhsI1Val = emitter.emitExprI1(lhs, EC_BoolCondition);
    PValue lhsI1PVal = emitter.emitPValue({lhsI1Val, lhs}, EC_BoolCondition);
    PValue rhsPVal = emitter.emitExprPValue(rhs, EC_BoolCondition);
    if (!lhsI1PVal)
      return {};

    // Coerce the true/false values into a compatible type if they disagree.
    auto convertValue = [&](ASTExprAnd<AnyValue> value, ASTType type,
                            bool isLHS) -> PValue {
      return emitter.emitPValue(value, EC_OperatorOperandValue, type);
    };
    if (coerceTypesToEachOther<PValue>(getLoc(), lhsPVal, lhs, rhsPVal, rhs,
                                       emitter, convertValue))
      return {};

    if (kind == kBoolOr) // and/or swap true/false operands
      std::swap(lhsPVal, rhsPVal);

    auto value =
        ParamOperatorAttr::get(POC::Cond, {lhsI1PVal, rhsPVal, lhsPVal});
    return emitter.emitResult(value, this, dest);
  }

  // Emit the LHS value as a bool/i1 value.
  CValue lhsV = emitter.emitExprCValue(lhs, EC_OperatorOperandValue);
  RValue lhsI1Value = emitter.emitI1({lhsV, lhs});
  SRValue lhsI1SRValue =
      emitter.emitSRValue({AnyValue(lhsI1Value), lhs}, EC_BoolCondition);
  if (!lhsI1SRValue)
    return {};

  auto ifOp = emitter.builder->create<HLCF::IfOp>(
      ifLoc, TypeRange{lhsV.getType()}, lhsI1SRValue);
  emitter.builder->createBlock(&ifOp.getThenRegion());
  emitter.builder->createBlock(&ifOp.getElseRegion());

  OpBuilder trueBuilder = ifOp.getThenBodyBuilder();
  OpBuilder falseBuilder = ifOp.getElseBodyBuilder();
  if (kind == kBoolOr) // and/or just treat the bool differently.
    std::swap(trueBuilder, falseBuilder);

  emitter.builder = trueBuilder;
  CValue rhsV = emitter.emitExprCValue(rhs, EC_BoolCondition);
  if (!rhsV)
    return {};

  /// If the types disagree, then we need to emit a conversion to a common
  /// type. See if one is convertible to the other, and if so, emit a
  /// conversion to get to a common type.
  auto convertValue = [&](ASTExprAnd<AnyValue> value, ASTType type,
                          bool isLHS) -> CValue {
    emitter.builder = isLHS ? falseBuilder : trueBuilder;
    return emitter.emitCValue(value, EC_OperatorOperandValue, type);
  };
  // Try to find compatibility between the raw values.  Pass in a null SMLoc so
  // that an error isn't diagnosed with an error message.
  if (coerceTypesToEachOther<CValue>(SMLoc(), lhsV, lhs, rhsV, rhs, emitter,
                                     convertValue)) {
    // If the two types are incompatible or ambiguously convertible to each
    // other, then the user wrote something like `if someInt and someString`.
    // This has no common type to return, but the result should still be
    // boolean-ish.  Handle this by extracting the boolean result out of the
    // second argument and converting that to a proper Bool result.
    ASTType boolType =
        emitter.shared.getBuiltinBoolType(emitter.declScope, getLoc());

    // If the RHS is already a Bool, we're good, otherwise convert to i1 then
    // back to Bool with a ctor.
    if (!rhsV.getRValueType().isEqualCanon(boolType)) {
      RValue rhsI1Value = emitter.emitI1({rhsV, rhs});
      rhsV = convertValue({rhsI1Value, rhs}, boolType, /*isLHS=*/false);
    }

    // Similarly, if the LHS was already a Bool then use it, otherwise convert
    // the i1 we already have back to Bool with a ctor.
    if (!lhsV.getRValueType().isEqualCanon(boolType))
      lhsV = convertValue({lhsI1SRValue, lhs}, boolType, /*isLHS=*/true);

    if (!lhsV || !rhsV)
      return {};
  }

  // Now we know they have common types.
  auto resultType = lhsV.getRValueType();
  if (resultType.isRegisterPassable(lhs->getLoc(), emitter.shared)) {
    emitter.builder = trueBuilder;
    auto rhsSR = emitter.emitSRValue({rhsV, rhs}, EC_OperatorOperandValue);
    if (!rhsSR)
      return {};
    emitter.builder->create<HLCF::YieldOp>(ifLoc, rhsSR);
    // Emit the false side.
    emitter.builder = falseBuilder;
    auto lhsSR = emitter.emitSRValue({lhsV, rhs}, EC_OperatorOperandValue);
    if (!lhsSR)
      return {};
    emitter.builder->create<HLCF::YieldOp>(ifLoc, lhsSR);
    ifOp->getResult(0).setType(lhsSR.getType());
    emitter.builder->setInsertionPointAfter(ifOp);
    return emitter.emitResult(SRValue(ifOp.getResult(0)), this, dest);
  }

  // If we have a memory only type, we have to handle the various issues with
  // the ValueDest.  It may specify an SLValue to emit into, it may be
  // ambiguous (like a call argument) or it may even be something like a
  // DLValue.  We handle this by projecting the ValueDest to an SLValue if we
  // can, but otherwise using a scratch buffer if not.
  emitter.builder->setInsertionPoint(ifOp);
  SLValue destBuffer = dest.getSLValueForResult(getLoc(), resultType, emitter);

  emitter.builder = falseBuilder;
  ValueDest falseDest(destBuffer, EC_CondExpr);
  if (!emitter.emitResult(lhsV, lhs, falseDest))
    falseDest.resetForError();
  emitter.builder->create<HLCF::YieldOp>(ifLoc);

  emitter.builder = trueBuilder;
  ValueDest trueDest(destBuffer, EC_CondExpr);
  if (!emitter.emitResult(rhsV, rhs, trueDest))
    trueDest.resetForError();
  emitter.builder->create<HLCF::YieldOp>(ifLoc);

  // MemoryOnly results don't need the 'if' result.  There is no way to remove
  // results after creating it, so we create a new IfOp and move IR over.
  emitter.builder->setInsertionPointAfter(ifOp);
  auto newIfOp =
      emitter.builder->create<HLCF::IfOp>(ifLoc, TypeRange{}, lhsI1SRValue);
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
  ifOp->erase();

  return emitter.emitCResult(MRValue(destBuffer), this, dest);
}

/// Emit the x^ expression.
AnyValue UnaryOpNode::emitTransfer(AnyValue argValue, ValueDest &dest,
                                   ExprEmitter &emitter) const {
  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(
        this, "cannot transfer a value in this context");

  // If this is a value we can track the lifetime of, then we can end that
  // value's lifetime to make a new RValue, otherwise return null.
  auto handleLifetimeEnd = [&](Value v, bool isRegister) -> AnyValue {
    // Lifetime checking needs to understand this value or field.
    if (!LifetimeTrackable::findUnderlyingValueFromField(v))
      return {};

    // If the input is already an owned RValue, then there is no need to
    // transfer from the temporary.
    if (argValue.getIfRValue())
      emitter.emitWarning(getLoc())
          << "transfer from an owned value has no effect and can be removed"
          << FixIt::remove(getLoc());

    auto newVal = emitter.builder->create<OwnershipEndLifetimeOp>(
        getLocation(emitter), v, isRegister);
    if (isRegister)
      return emitter.emitResult(SRValue(newVal), this, dest);
    return emitter.emitResult(MRValue(newVal), this, dest);
  };

  // The transfer expression expects the result to be a ownable value that it
  // can launder into an RValue.
  if (auto sl = argValue.getIfSLValue()) {
    if (auto result = handleLifetimeEnd(sl, /*isRegister=*/false))
      return result;
    // TODO: When we support explicit move operations and have an lvalue, we
    // can invoke it.
  }
  if (auto mb = argValue.getIfMBValue())
    if (auto result = handleLifetimeEnd(mb, /*isRegister=*/false))
      return result;
  if (auto mr = argValue.getIfMRValue())
    if (auto result = handleLifetimeEnd(mr, /*isRegister=*/false))
      return result;
  if (auto sb = argValue.getIfSBValue())
    if (auto result = handleLifetimeEnd(sb, /*isRegister=*/true))
      return result;
  if (auto sr = argValue.getIfSRValue())
    if (auto result = handleLifetimeEnd(sr, /*isRegister=*/true))
      return result;

  emitter.emitError(getLoc(),
                    "expression does not designate a value with a lifetime");
  return {};
}

AnyValue UnaryOpNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  auto exprRep = subExpr->emitIR(ValueDest::none(), emitter);
  if (!exprRep)
    return {};

  // Special case some things for literals.
  // TODO: Fix literal representation.
  if (auto exprParam = exprRep.getIfPValue();
      exprParam && (exprParam.getType().mlirType.isIndex() ||
                    exprParam.getType().mlirType.isF64())) {
    switch (kind) {
    default:
      break;
    case ExprNode::kNeg:
      if (auto constantFP = dyn_cast<FloatAttr>(exprParam.get())) {
        auto result =
            FloatAttr::get(constantFP.getType(), -constantFP.getValue());
        return emitter.emitResult(result, this, dest);
      }

      // Support general integer parameter exprs.
      if (exprParam.getType().mlirType.isIndex())
        return emitter.emitResult(ParamOperatorAttr::getNeg(exprParam), this,
                                  dest);

      break;
    case ExprNode::kPos:
      return emitter.emitResult(exprParam, this, dest);
    }
  }

  // Handle special cases that don't correspond to special functions, such as
  // `not x`, `*args: *Ts`, `x^` etc
  if (kind == kTransfer)
    return emitTransfer(exprRep, dest, emitter);

  ASTExprAnd<AnyValue> argValue = {exprRep, subExpr};
  Kind kindToEmit = kind;
  if (kind == kBoolNot) {
    // Turn this into a call to __bool__.
    argValue.ir =
        emitter.emitNamedMethodCall("__bool__", argValue, ValueDest::none(),
                                    CallSyntax::kImplicitConvert, this);
    if (!argValue.ir)
      return {};
    // Now that we know we bool-ized the expression, invert it with ~.
    kindToEmit = kInvert;
  } else if (kind == kUnpack) {
    if (auto pValue = exprRep.getIfPValue()) {
      // There are two distinct cases of unpacking:
      // 1. Unpacking within an expression list, e.g. `a = [1, 2]; b = (0,
      // *a)`,
      //    with the result being a tuple `b` with 3 elements `0, 1, 2`. This
      //    is handled with the special function `__iter__`.
      // 2. Unpacking in a type annotation, e.g. `*args: *Ts`, with the result
      //    being akin to the types of `Ts` being mapped to the type
      //    annotations for the arguments `args`: `args[0]: Ts[0], args[1]:
      //    Ts[1], ...`. This is not handled with a special function of any
      //    kind, and so is handled here.
      if (!isa<VariadicType>(pValue.get().getType())) {
        emitter.emitError(getLoc(), "only variadic types may be unpacked");
        return {};
      }
      return emitter.emitResult(
          TypeConstantAttr::get(POP::PackType::get(pValue.get())), this, dest);
    }
  } else if (kind == kAwait) {
    // Diagnose errors with 'await'.
    if (!emitter.builder) {
      emitter.emitErrorForDynamicValueInParameter(this, "cannot await");
      return {};
    }
    Operation *func = emitter.builder->getInsertionBlock()->getParentOp();
    while (!isa<FuncOp>(func))
      func = func->getParentOp();
    if (!cast<FuncOp>(func).getSignature().isAsync()) {
      emitter.emitError(getLoc(), "cannot await inside a non-async function")
          << getRange();
      return {};
    }
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnInfo = getOpSpecialFunctions(kindToEmit, /*isReversed=*/false);
  assert(specialFnInfo.kind != SpecialFunctionKind::kNormal &&
         "Unary operators are implemented via special methods");

  return emitter.emitNamedMethodCall(specialFnInfo.name, argValue, dest,
                                     CallSyntax::kOperator, this);
}

AnyValue OwnershipOpNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // Complain if lifetimes are not enabled.
  if (!emitter.shared.useExperimentalLifetimes())
    emitter.emitError(getLoc(), "lifetimes are not enabled yet") << getRange();

  // Get the base type and lifetime specifier.
  PValue lifetimePVal = emitter.emitExprPValue(
      lifetime, EC_LifetimeSpec, emitter.shared.getLifetimeType());
  if (!lifetimePVal)
    return {};
  auto subType = emitter.emitExprType(subExpr);
  if (!subType)
    return {};

  // FIXME: Swap RefType attr order to match syntax order.
  auto result = RefType::get(isMutable, subType, lifetimePVal.get());
  return emitter.emitResult(result, this, dest);
}

AnyValue IfElseOpNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  RValue condRVal = emitter.emitExprI1(condExpr, EC_BoolCondition);

  // Inside a parameter context, emit conditional expression.
  if (!emitter.builder) {
    PValue condPVal =
        emitter.emitPValue({condRVal, condExpr}, EC_BoolCondition);
    if (!condPVal)
      return {};

    PValue trueVal = emitter.emitExprPValue(trueExpr, EC_BoolCondition);
    PValue falseVal = emitter.emitExprPValue(falseExpr, EC_BoolCondition);

    // Coerce the true/false values into a compatible type.
    auto convertValue = [&](ASTExprAnd<AnyValue> value, ASTType type,
                            bool isLHS) -> PValue {
      return emitter.emitPValue(value, EC_CondExpr, type);
    };
    if (coerceTypesToEachOther<PValue>(getLoc(), trueVal, trueExpr, falseVal,
                                       falseExpr, emitter, convertValue))
      return {};

    auto value =
        ParamOperatorAttr::get(POC::Cond, {condPVal, trueVal, falseVal});
    return emitter.emitResult(value, this, dest);
  }

  // Otherwise, emit HLCF::IfOp.
  Value condValue =
      emitter.emitSRValue({AnyValue(condRVal), condExpr}, EC_BoolCondition);

  if (!condValue)
    return {};

  Location ifLoc = getLocation(emitter);
  // At this point since we don't know the type of trueExpr / falseExpr, use a
  // dummy type for the 'if' result.  We'll fix it later.
  auto ifOp = emitter.builder->create<HLCF::IfOp>(
      ifLoc, TypeRange{condValue.getType()}, condValue);

  // Emit the trueVal and falseVal's but don't check for error or emit the
  // yield yet.
  emitter.builder->createBlock(&ifOp.getThenRegion());
  CValue trueVal = emitter.emitExprCValue(trueExpr, EC_CondExpr);

  emitter.builder->createBlock(&ifOp.getElseRegion());
  CValue falseVal = emitter.emitExprCValue(falseExpr, EC_CondExpr);

  if (!trueVal || !falseVal) {
    emitter.builder->setInsertionPointAfter(ifOp);
    return {};
  }

  /// If the types disagree, then we need to emit a conversion to a common
  /// type. See if one is convertible to the other, and if so, emit a
  /// conversion to get to a common type.
  auto convertValue = [&](ASTExprAnd<AnyValue> value, ASTType type,
                          bool isLHS) -> CValue {
    Block &b = isLHS ? ifOp.getThenBlock() : ifOp.getElseBlock();
    emitter.builder->setInsertionPointToEnd(&b);
    return emitter.emitCValue(value, EC_CondExpr, type);
  };
  if (coerceTypesToEachOther<CValue>(getLoc(), trueVal, trueExpr, falseVal,
                                     falseExpr, emitter, convertValue))
    return {};

  // Ok, we now know if the types were register_passable or not, so finish up
  // the logic.  register_passable values get merged together as SSA registers
  // in the 'if' result.
  auto resultType = trueVal.getRValueType();
  if (resultType.isRegisterPassable(trueExpr->getLoc(), emitter.shared)) {
    // Finish false.
    emitter.builder->setInsertionPointToEnd(&ifOp.getElseBlock());
    auto falseSR = emitter.emitSRValue({falseVal, falseExpr}, EC_BoolCondition);
    if (!falseSR)
      return {};
    emitter.builder->create<HLCF::YieldOp>(ifLoc, falseSR);
    // Finish true.
    emitter.builder->setInsertionPointToEnd(&ifOp.getThenBlock());
    auto trueSR = emitter.emitSRValue({trueVal, trueExpr}, EC_BoolCondition);
    if (!trueSR)
      return {};
    emitter.builder->create<HLCF::YieldOp>(ifLoc, trueSR);
    emitter.builder->setInsertionPointAfter(ifOp);
    // Ensure the correct type is used.
    ifOp->getResult(0).setType(trueVal.getType());
    return emitter.emitResult(SRValue(ifOp.getResult(0)), this, dest);
  }

  // If we have a memory only type, we have to handle the various issues with
  // the ValueDest.  It may specify an SLValue to emit into, it may be
  // ambiguous (like a call argument) or it may even be something like a
  // DLValue.  We handle this by projecting the ValueDest to an SLValue if we
  // can, but otherwise using a scratch buffer if not.
  emitter.builder->setInsertionPoint(ifOp);
  SLValue destBuffer = dest.getSLValueForResult(getLoc(), resultType, emitter);

  emitter.builder->setInsertionPointToEnd(&ifOp.getElseBlock());
  ValueDest falseDest(destBuffer, EC_CondExpr);
  if (!emitter.emitResult(falseVal, falseExpr, falseDest))
    falseDest.resetForError();
  emitter.builder->create<HLCF::YieldOp>(ifLoc);

  emitter.builder->setInsertionPointToEnd(&ifOp.getThenBlock());
  ValueDest trueDest(destBuffer, EC_CondExpr);
  if (!emitter.emitResult(trueVal, falseExpr, trueDest))
    trueDest.resetForError();
  emitter.builder->create<HLCF::YieldOp>(ifLoc);

  // MemoryOnly results don't need the 'if' result.  There is no way to remove
  // results after creating it, so we create a new IfOp and move IR over.
  emitter.builder->setInsertionPointAfter(ifOp);
  auto newIfOp =
      emitter.builder->create<HLCF::IfOp>(ifLoc, TypeRange{}, condValue);
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
  ifOp->erase();

  return emitter.emitCResult(MRValue(destBuffer), this, dest);
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
  OpBuilder lastBuilder = emitter.builder.value();
  RValue lastCmpI1Value = emitter.emitI1({lastCmpExpr, this});
  SRValue lastCmpI1RValue =
      emitter.emitSRValue({AnyValue(lastCmpI1Value), this}, EC_BoolCondition);
  if (!lastCmpI1RValue)
    return {};
  auto ifOp = emitter.builder->create<HLCF::IfOp>(
      ifLoc, lastCmpI1RValue.getType().mlirType, lastCmpI1RValue);
  emitter.builder->createBlock(&ifOp.getThenRegion());
  SRValue exprValue =
      emitter.emitExprSRValue(exprs[opIdx + 1], EC_OperatorOperandValue);
  if (!exprValue)
    return {};
  AnyValue lastBinOp =
      emitBinOpCall({lastExpr, exprs[opIdx]}, {exprValue, exprs[opIdx + 1]},
                    ops[opIdx], ValueDest::none(), this, emitter);
  SRValue lastRV =
      emitter.emitSRValue({lastBinOp, exprs[opIdx + 1]}, EC_BoolCondition);
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

AnyValue ChainedCmpOpNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  AnyValue e0Rep = emitter.emitExpr(exprs[0], EC_OperatorOperandValue);
  AnyValue e1Rep = emitter.emitExpr(exprs[1], EC_OperatorOperandValue);
  if (!e0Rep || !e1Rep)
    return {};

  AnyValue cmpe0e1RV = emitBinOpCall(
      {e0Rep, exprs[0]}, {e1Rep, exprs[1]}, ops[0],
      exprs.size() == 2 ? dest : ValueDest::none(), this, emitter);
  if (exprs.size() == 2)
    return cmpe0e1RV;

  SRValue lastCmpExpr =
      emitter.emitSRValue({cmpe0e1RV, exprs[1]}, EC_BoolCondition);
  SRValue e1RV =
      emitter.emitSRValue({e1Rep, exprs[1]}, EC_OperatorOperandValue);
  if (!lastCmpExpr || !e1RV)
    return {};
  return emitter.emitResult(emitNextCmp(emitter, 1, lastCmpExpr, e1RV), this,
                            dest);
}

AnyValue FunctionTypeNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // Parameters declared within the function type must be visible. Create a
  // dummy declaration.
  ASTDecl &dummyScope = emitter.getDeclResolver().addFullyResolvedDecl(
      nullptr, StringAttr(), getLoc(), &emitter.declScope);
  ExprEmitter typeEmitter(emitter.shared, dummyScope, EC_Type);

  bool paramVararg = false;
  SmallVector<ParamDeclAttr> inputParamDecls, resultParamDecls;
  ParsedArgument::processParameterArgs(typeEmitter, dummyScope, inputParams,
                                       inputParamDecls,
                                       /*isResultParams=*/false, paramVararg);
  ParsedArgument::processParameterArgs(typeEmitter, dummyScope, resultParams,
                                       resultParamDecls,
                                       /*isResultParams=*/true, paramVararg);
  FnEffects effects = this->effects;
  if (paramVararg)
    effects = effects | FnEffects::ParamVararg;

  SmallVector<ParsedArgument> args = llvm::to_vector(arguments);
  SmallVector<Type> argTypes;
  SmallVector<TypedAttr> defaults;
  ASTType resultType = ParsedArgument::emitFunctionArgumentsAndResults(
      [&] { return failure(); }, emitter.shared, typeEmitter, resultTypeExpr,
      effects, args, argTypes, defaults, isDef, resultLoc, emitter.declScope);
  if (!resultType)
    return {};

  ParsedArgument::computeArgumentConventions(emitter.shared, args, argTypes,
                                             defaults);

  SmallVector<ValueInputConvention> inputConventions = llvm::map_to_vector(
      args, [](const ParsedArgument &arg) { return arg.kgenConvention; });

  if (bitEnumContainsAny(effects, FnEffects::Throws)) {
    Type errorType =
        emitter.shared.getBuiltinErrorType(emitter.declScope, resultLoc);
    if (!errorType)
      return {};

    resultType = POP::VariantType::get({errorType, resultType});

    // FIXME(#12604): Cannot return Error from raising function.
    if (cast<POP::VariantType>(resultType.mlirType).getNumTypes() == 1) {
      emitter.emitError(
          resultLoc, "cannot return and raise the same type from a function");
      return {};
    }
  }

  // Build the signature type.
  Builder b(emitter.getContext());
  auto inputParamsAttr = b.getAttr<ParamDeclArrayAttr>(inputParamDecls);
  auto resultParamsAttr = b.getAttr<ParamDeclArrayAttr>(resultParamDecls);
  FunctionType functionType =
      b.getFunctionType(argTypes, {resultType.mlirType});

  // Compute the signature of the function.
  auto signature = IndexRefRemapper::remapToSignature(
      inputParamsAttr, resultParamsAttr, functionType,
      b.getAttr<FnMetadataAttr>(inputConventions, defaults, effects),
      [&] { return mlir::emitError(emitter.translateLocation(getLoc())); });
  if (!signature) {
    typeEmitter.emitError(getLoc(), "failed to construct signature type");
    return {};
  }
  if (bitEnumContainsAny(effects, FnEffects::Escaping)) {
    LIT::FileModuleOp fileModuleOp;
    ASTDecl *astDecl = &emitter.declScope;
    for (; !fileModuleOp && astDecl; astDecl = astDecl->getParentDecl()) {
      fileModuleOp = dyn_cast<LIT::FileModuleOp>(*astDecl);
      if (fileModuleOp)
        break;
    }
    assert(fileModuleOp &&
           "It should not be possible for the parser to parse a "
           "type outside a file module op.");
    if (fileModuleOp) {
      StructDeclOp declOp = emitter.shared.getOrGenerateClosureWrapperStruct(
          this->getLoc(), signature, fileModuleOp);
      ASTType result(DeclRefType::get(
          SymbolRefAttr::get(SymbolTable::getSymbolName(declOp))));
      // TODO: uncomment (https://github.com/modularml/modular/issues/17073).
      // emitter.emitResult(result, this, dest);
    }

    // TODO: remove (https://github.com/modularml/modular/issues/17073).
    FnEffects newFn =
        bitEnumSet(bitEnumClear(signature.getFnEffects(), FnEffects::Escaping),
                   FnEffects::Capturing);
    return emitter.emitResult(ASTType(signature.getWithFnEffects(newFn)), this,
                              dest);
  }
  return emitter.emitResult(ASTType(signature), this, dest);
}

AnyValue AddressConvertNode::emitIR(ValueDest &dest,
                                    ExprEmitter &emitter) const {
  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(this);

  // __get_lvalue_as_address(someSLValue) returns a pop.pointer.
  if (kind == kGetLValueAsAddress) {
    LValue result = emitter.emitExprLValue(subExpr, EC_Unknown);
    if (!result)
      return {};
    SLValue resultPtr = result.getIfSLValue();
    if (!resultPtr) {
      emitter.emitError(getLoc(),
                        "cannot use a dynamic LValue in this operator")
          << getRange();
      return {};
    }
    // Emit an intrinsic so the compiler knows the value is mutable.
    emitter.builder->create<OwnershipDefLValueOp>(getLocation(emitter),
                                                  resultPtr);

    // Return the SLValue as an SRValue since the pointer itself is the
    // result.
    return emitter.emitResult(SRValue(resultPtr), this, dest);
  }

  SRValue exprVal = emitter.emitExprSRValue(subExpr, EC_Unknown);
  if (!exprVal)
    return {};
  auto pointerType = dyn_cast<POP::PointerType>(exprVal.getType().mlirType);
  if (!pointerType) {
    emitter.emitError(getLoc(),
                      "operand must have '!pop.pointer<T>' type, not ")
        << exprVal.getType() << getRange();
    return {};
  }

  // If this is a user defined type with ownership, emit lifetime intrinsics
  // for it, if not, we don't need/want them.
  auto pointeeType = ASTType(pointerType).getPointerElementType();
  bool needsLifetime = isa<DeclRefType>(pointeeType.mlirType);

  /// __get_address_as_owned_value(pop_pointer) # returns RValue
  if (kind == ExprNode::kGetAddressAsOwned) {
    // Make sure to take ownership of the address and create a new lifetime
    // tracked value.
    if (needsLifetime)
      exprVal = emitter.builder->create<OwnershipEndLifetimeOp>(
          getLocation(emitter), exprVal, /*isRegister=*/false);
    return emitter.emitResult(MRValue(exprVal), this, dest);
  }

  // These both return an SLValue with different ownership semantics.
  // __get_address_as_lvalue(ptr) & __get_address_as_uninit_lvalue(ptr)
  assert(kind == kGetAddressAsLValue || kind == kGetAddressAsUninitLValue);
  if (needsLifetime)
    exprVal = emitter.builder->create<OwnershipMakePointerLValue>(
        getLocation(emitter), exprVal,
        /*isLiveOnEntry=*/kind == kGetAddressAsLValue, /*isLiveOnExit=*/true);

  return emitter.emitResult(SLValue(exprVal), this, dest);
}
