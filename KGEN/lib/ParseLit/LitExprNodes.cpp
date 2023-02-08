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
#include "KGEN/KGENDialect/ParameterEvaluator.h"
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
#include "mlir/IR/IRMapping.h"
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

  // Check to see if the

  return result;
}

/// This implements __mlir_attr.x lookup, synthesizing a MAValue for the
/// attribute on demand.
static AnyValue synthesizeMLIRAttrFromString(StringRef name, SMLoc loc,
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
  return MValue(typedAttr);
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
        emitter.emitExprMValue(indexExpr, ASTType(), " in MLIR magic");
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
static AnyValue synthesizeMLIROpFromString(StringRef name,
                                           ExprEmitter &emitter) {
  auto *context = emitter.getContext();
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
    auto value = emitter.emitExprMValue(
        node, ASTType(), " in value for '" + Twine(name) + "' attribute");
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
  return MValue(UnboundMLIROperationAttr::get(context, unboundOp.getType(),
                                              unboundOp.getName(), attrs));
}

/// Given a ParamDeclareOp, return the value that should be used in a reference
/// to it.  This currently fully substitutes members unless they are in a
/// function definition.
static MValue resolveParamDeclareValue(ParamDeclareOp param,
                                       ParamBindArrayAttr bindings) {
  // If the param is declared in a function, then just directly use it.
  Operation *parent = param->getParentOp();
  while (1) {
    // If this reference is within a function then keep it symbolic.
    if (parent && isa<LIT::FuncOp>(parent))
      return MValue(ParamDeclRefAttr::get(param.getName(), param.getType()));
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

      ParameterEvaluator evaluator;
      for (ParamBindAttr binding : bindings)
        evaluator.setParameterValue(binding.getName(), binding.getValue());

      auto result = evaluator.getReboundAttribute(param.getValue());
      return MValue(cast<TypedAttr>(result));
    }

    // Ignore if and other control flow things.
    parent = parent->getParentOp();
  }

  return MValue(ParamDeclRefAttr::get(param.getName(), param.getType()));
}

//===----------------------------------------------------------------------===//
// ExprNode Implementation
//===----------------------------------------------------------------------===//

ExprNode::~ExprNode() { llvm_unreachable("never called"); }

/// Return the start or end of the source range.
llvm::SMLoc ExprNode::getRangeStart() const { return getRange().getStart(); }
llvm::SMLoc ExprNode::getRangeEnd() const { return getRange().getEnd(); }

/// Emit this expression to MLIR as a CallableValue.  On error, emit an error
/// and return a null value.
CallableValue ExprNode::emitCallable(ExprEmitter &emitter,
                                     ASTType contextualType) const {
  // The default implementation of this returns the expression as an RValue.
  auto calleeVal = emitter.emitExprRValue(this);
  if (!calleeVal)
    return {};

  return CallableValue({calleeVal, this});
}

/// Return the 'loc' for this node translated to an MLIR location.
Location ExprNode::getLocation(IREmitter &emitter) const {
  return emitter.translateLocation(getLoc());
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
  // FIXME: This should eventually use a float literal type.
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
  auto attr =
      StringAttr::get(value, KGEN::StringType::get(emitter.getContext()));
  return AnyValue(attr);
}

AnyValue NoneLiteralNode::emitIR(ExprEmitter &emitter,
                                 ASTType contextualType) const {
  return MValue(NoneAttr::get(emitter.getContext()));
}

AnyValue DeclRefNode::emitIR(ExprEmitter &emitter,
                             ASTType contextualType) const {
  return emitCallable(emitter, contextualType).emitAsValue(emitter);
}

/// Emit IR for an unqualified declaration reference "x" looked up in current
/// context.
CallableValue DeclRefNode::emitCallable(ExprEmitter &emitter,
                                        ASTType contextualType) const {
  ASTDecl &container = emitter.declScope;

  // Perform a lookup of the specified decl in the current container.
  LookupResult lookup = emitter.shared.lookupAndResolveDecl(
      spelling, getLoc(), container, /*searchParentScopes=*/true);

  auto createVarDeclWithContextualType = [&](OpBuilder &builder) -> VarDeclOp {
    Type declIRType = POP::PointerType::get(contextualType);
    auto loc = getLocation(emitter);
    auto nameAttr = StringAttr::get(loc.getContext(), spelling);
    return builder.create<VarDeclOp>(loc, declIRType, nameAttr);
  };

  // If the unresolved name is `_`, then we have a discard pattern.  Python
  // supports this by just implicitly declaring a variable named _ and
  // allowing rewrites, but we cannot take this approach because each discard
  // could have a different type.  Handle this specially by not inserting the
  // `_` variable into the name table, so we'll get a new instance on every use.
  if (lookup.isFailure() && contextualType && spelling == "_" &&
      emitter.builder) {
    // Introduce a new lit.var.decl node whose type matches the
    // implicitDeclType.
    // TODO(autopromotions): turn infinite integers into concrete ones as
    // needed.
    auto varDecl = createVarDeclWithContextualType(*emitter.builder);
    return {{AnyValue(LValue(varDecl.getResult())), this}};
  }

  // If that lookup failed, but we can synthesize a variable declaration in this
  // scope, do that.  We can only do this if there is a contextual type
  // available and an insertion point.
  if (lookup.isFailure() && contextualType && emitter.varDeclCursor) {
    // Use this builder to place any VarDeclOps. In Python there is only one
    // scope per function and all variables belong to that scope, so builders
    // should reflect that.
    OpBuilder varDeclBuilder(emitter.varDeclCursor);
    auto varDecl = createVarDeclWithContextualType(varDeclBuilder);

    // In a normal implicit declaration, we add it to the name table so
    // subsequent uses find this one.
    emitter.getDeclResolver().addFullyResolvedDecl(
        varDecl, getLoc(), varDecl.getNameAttr(), &container);
    // Re-do lookup, making sure we form a uniqued vector that we can reference.
    lookup = emitter.shared.lookupAndResolveDecl(spelling, getLoc(), container,
                                                 /*searchParentScopes=*/false);
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
    if (auto structDecl = dyn_cast<StructDeclOp>(container))
      diag << structDecl.getName() << " has no '" << spelling << "' member";
    else
      diag << "use of unknown declaration '" << spelling << "'";
    return {};
  }

  // Functions form an address, and may be overloaded.
  if (isa<LIT::FuncOp>(*decls[0]))
    return CallableValue(getLoc(), spelling, decls, /*bindings=*/{});

  assert(decls.size() == 1 && "Only functions may be overloaded");
  ASTDecl &decl = *decls[0];

  // Let declarations resolve to an rvalue.
  if (auto letDecl = dyn_cast<LetDeclOp>(decl))
    return {{AnyValue(RValue(letDecl.getResult())), this}};

  // Variable references resolve to an lvalue addressing the variable.
  if (auto var = dyn_cast<VarDeclOp>(decl))
    return {{AnyValue(LValue(var.getResult())), this}};

  // Parameters form a meta-value.
  if (auto param = dyn_cast<ParamDeclareOp>(decl)) {
    return {{resolveParamDeclareValue(param, /*bindings=*/{}), this}};
  }

  // Use of forward references.
  if (auto param = dyn_cast<AliasForwardDeclOp>(decl))
    return {{MValue(ParamDeclRefAttr::get(param.getName(), param.getType())),
             this}};

  // RValue's and LValues always resolve to their known value.
  if (auto rvalue = decl.getIfRValue())
    return {{rvalue, this}};
  if (auto lvalue = decl.getIfLValue())
    return {{lvalue, this}};

  // If this is a type declaration, return it as a type.
  if (isa<StructDeclOp>(decl))
    return {{MValue(DeclRefType::get(decl.getSymbolRef())), this}};

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
static Type parseMLIRType(StringRef name, const ExprNode *node,
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

AnyValue AttributeRefNode::emitIR(ExprEmitter &emitter,
                                  ASTType contextualType) const {
  return emitCallable(emitter, contextualType).emitAsValue(emitter);
}

/// Emit a qualified attribute reference to MLIR as a CallableValue.  On error,
/// emit an error and return a null value.
CallableValue AttributeRefNode::emitCallable(ExprEmitter &emitter,
                                             ASTType contextualType) const {

  auto baseVal = base->emitIR(emitter, /*No Contextual Type*/ {});
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
    // If there is no decl, the type is an MLIR type.
    Type baseMLIRType = baseRVType.mlirType;

    // Handle __mlir_op.`xxx` references, lazily synthesizing values when
    // they are referenced.
    if (isa<MagicMLIRAttrType>(baseMLIRType)) {
      AnyValue result =
          synthesizeMLIRAttrFromString(attrSpelling, getLoc(), emitter.shared);
      return {{result, this}};
    }
    if (isa<MagicMLIROpType>(baseMLIRType))
      return {{synthesizeMLIROpFromString(attrSpelling, emitter), this}};
    if (isa<MagicMLIRTypeType>(baseMLIRType)) {
      Type result = parseMLIRType(attrSpelling, this, emitter.shared);
      return {{result ? AnyValue(result) : AnyValue(), this}};
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
    CallableValue fnRef(getLoc(), attrSpelling, memberDecls,
                        baseRVType.getParamBindings());

    // If the callee is a static method, we can directly reference it without
    // binding a self parameter.  If this is an instance method, we bind the
    // base value and the symbol together into a callable.
    // FIXME: This isn't handling overloaded static/non-static methods
    // correctly.  What is the actual behavior we want for static methods?
    // Maybe we don't allow overloading static and non-static methods with the
    // same name?
    if (!fnOp.getIsStatic() && !hasTypeBase)
      fnRef.baseVal = {baseVal, base};
    return fnRef;
  }
  assert(memberDecls.size() == 1 && "only methods may be overloaded");
  ASTDecl &memberDecl = *memberDecls[0];

  // Parameters form a meta-value.
  if (auto param = dyn_cast<ParamDeclareOp>(memberDecl)) {
    MValue result =
        resolveParamDeclareValue(param, baseRVType.getParamBindings());
    return {{result, this}};
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
      return {{LValue(fieldPtr), this}};
    }

    // If the base is an MValue, emit a field extract as an MValue.
    if (MValue baseMV = baseVal.getIfMValue()) {
      auto extractVal = LIT::StructExtractAttr::get(baseMV.get(), fieldOp);
      return {{MValue(extractVal), this}};
    }

    // Otherwise, it must be an rvalue.
    DRValue baseRV = emitter.emitDRValue({baseVal, base});
    if (!baseRV)
      return {};

    return {{DRValue(emitter.builder->create<StructExtractOp>(mlirLoc, baseRV,
                                                              fieldOp)),
             this}};
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
        "TODO: cannot emit MLIR operation in parameter expressions yet")
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
      value = emitter.emitExprDRValue(operand);
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
        state.types.push_back(MValue(typedAttr).getIfTypeValue());
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
                          "MLIR operation region must be a function reference");
        return {};
      }
      // Lookup the operation body.
      LookupResult result = emitter.shared.lookupAndResolveDecl(
          bodyRef, call.getLoc(), emitter.declScope,
          /*searchParentScopes=*/false);
      ArrayRef<ASTDecl *> results = result.getIfSuccess();
      if (result.isFailure() || results.size() != 1 ||
          !isa<LIT::FuncOp>(*results.front())) {
        emitter.emitError(call.getLoc(), "MLIR operation region reference did "
                                         "not resolve to a function body");
        return {};
      }
      // Resolve the body before using it.
      ASTDecl &body = *results.front();
      if (failed(
              emitter.shared.declResolver->resolveFully(body, call.getLoc()))) {
        emitter.emitError(body.getLoc(),
                          "failed to immediately resolve MLIR operation region")
                .attachNote(call.getLocation(emitter))
            << "see MLIR operation here";
        return {};
      }
      // SUPER-MEGA-HACK: The body is single-use. Move it in, because otherwise
      // the function will not verify. Make sure to replace the terminator.
      auto func = cast<LIT::FuncOp>(body);
      auto region = std::make_unique<Region>();
      region->takeBody(func.getBodyRegion());
      for (Operation &op : region->front()) {
        if (!op.hasTrait<OpTrait::IsTerminator>())
          continue;
        if (isa<EndFuncOp>(op)) {
          op.erase();
          break;
        }
        op.getBlock()->splitBlock(++Block::iterator(&op))->erase();
      }
      func.erase();
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
      assert(val.getType() == resultOp->getResult(0).getType());
      resultOp->erase();
      return DRValue(val);
    }

    if (auto attr = dyn_cast<TypedAttr>(cast<Attribute>(folded))) {
      assert(attr.getType() == resultOp->getResult(0).getType());
      // If it is a constant, make an MAValue result.
      resultOp->erase();
      return MValue(attr);
    }
  }

  // If folding failed, return the operation normally.
  return DRValue(resultOp->getResult(0));
}

AnyValue CallNode::emitIR(ExprEmitter &emitter, ASTType contextualType) const {
  auto calleeVal = callee->emitCallable(emitter, {});
  if (!calleeVal)
    return {};

  // Figure out how this was spelled.
  auto syntax = CallSyntax::kDirectCall;

  // If there was a base value, then this is a method or indirect call.
  if (calleeVal.baseVal) {
    syntax =
        calleeVal.direct ? CallSyntax::kMethodCall : CallSyntax::kIndirectCall;

    if (auto mValue = calleeVal.baseVal.ir.getIfMValue()) {
      // If this is the invocation of an unbound MLIR operator, bind it into an
      // actual operator!
      if (auto unboundOperator =
              dyn_cast<UnboundMLIROperationAttr>(mValue.get()))
        return emitMLIROperatorCall(*this, unboundOperator, emitter);
    }
  }

  // If the callee is a type value (as in `T()` or `T[123]()`), then this is an
  // invocation of the initializer for the type.
  if (!calleeVal.direct)
    if (ASTType calledType = calleeVal.baseVal.ir.getIfTypeValue()) {
      bool isErroneousDecl = false;
      calleeVal = CallableValue(calledType, "__new__", getLoc(),
                                isErroneousDecl, emitter.shared);
      if (calleeVal.isNull()) {
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

      syntax = CallSyntax::kTypeCall;
    }

  /// Emit a function call for a call node with the specified operands.
  SmallVector<ASTExprAnd<AnyValue>> operands;
  for (ExprNode *arg : args) {
    operands.push_back({arg->emitIR(emitter, /*No Contextual Type*/ {}), arg});
    if (!operands.back())
      return {};
  }

  return calleeVal.emitFunctionCall(operands, syntax, this, emitter);
}

AnyValue SliceNode::emitIR(ExprEmitter &emitter, ASTType contextualType) const {
  auto diag =
      emitter.emitError(getLoc(), "TODO: SliceNode::emitIR not implemented yet")
      << getRange();
  diag.attachNote(getLocation(emitter))
      << "keyword arguments aren't supported yet";
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
        emitter.emitExprMValue(indexExpr, ASTType(), " in type parameter");
    if (!indexVal)
      return {};
    paramBindings.add(indexExpr, indexVal.get());
  }

  // Check the bindings.
  ssize_t incorrectBindingNo = 0;
  ASTType incorrectBindingExpectedType;
  auto bindingAttr = paramBindings.verifyBindings(
      structOp.getInputParamDeclsAttr(), structOp.getName(), subscript.getLoc(),
      incorrectBindingNo, incorrectBindingExpectedType, emitter.shared,
      structOp);
  if (!bindingAttr)
    return {};

  // Ok, we succeeded at reparameterizing the type.
  return CallableValue(
      {MValue(DeclRefType::get(typeDecl.getSymbolRef(), bindingAttr)),
       &subscript});
}

/// When subscripting a callable with a bound symbol (i.e. a direct method call
/// or call to a method), apply parameter bindings to it.
static CallableValue bindAttrValuesToDirectCall(CallableValue &callable,
                                                ArrayRef<ExprNode *> indices,
                                                ExprEmitter &emitter) {
  assert(callable.direct && "only valid on direct call");

  // If the indices are a single () expression, then we treat this as having
  // no parameters.  This is used with arrow expressions to allow `f[() -> x]`.
  if (indices.size() == 1) {
    if (auto *tuple = dyn_cast<TupleNode>(indices[0]))
      if (tuple->exprs.empty())
        return std::move(callable);
  }

  // Process each subscript entry as a binding.
  // TODO: Support named bindings in addition to positional ones: `A[x: 42]`.
  for (auto idx : indices) {
    auto val = emitter.emitExprMValue(idx, ASTType(), " in parameter binding");
    if (!val)
      return {};

    // We don't do any checking to see if the value is compatible with the
    // expected type - this is deferred until when the symbol is actually
    // emitted for something.  This allow us to use the provided parameters to
    // filter down the overload set.
    //
    // Note: we're being a bit abusive here by making a ParamBindAttr with a
    // null name for positional attributes.
    callable.direct->inputParamBindings.add(idx, val.get());
  }
  // The bindings will be checked for validity when a reference is formed.
  return std::move(callable);
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
  if (subValue.direct)
    return bindAttrValuesToDirectCall(subValue, indices, emitter);

  if (auto callableMVal = subValue.baseVal.ir.getIfMValue()) {
    if (auto sig = dyn_cast<SignatureType>(callableMVal.getType())) {
      // If this is a signature-type MValue callable, this is binding parameter
      // values to a call.
      SmallVector<TypedAttr> bindOperands({callableMVal.get()});
      if (indices.size() != sig.getInputParams().size()) {
        emitter.emitError(getLoc(), "parametric callable expected ")
            << sig.getInputParams().size() << " parameter"
            << plural(sig.getInputParams().size()) << getIndexRange();
        return {};
      }
      for (auto [idx, type] : llvm::zip(indices, sig.getInputParams())) {
        bindOperands.push_back(emitter.emitExprMValue(
            idx, type.getType(), " in call parameter binding"));
        if (!bindOperands.back())
          return {};
      }
      return CallableValue(
          {MValue(ParamOperatorAttr::get(POC::BindSignature, bindOperands)),
           this});
    }
  }

  // Otherwise, if there is no symbol, it is just an LValue or RValue being
  // subscript.

  // If the sub-value is an unbound Type, try binding things to it!
  if (Type typeValue = subValue.baseVal.ir.getIfTypeValue()) {
    // Handle user-defined types.
    // TODO: This seems wrong, we won't handle things like Type.AssocType[1] ?
    if (auto declRef = dyn_cast<DeclRefType>(typeValue))
      return substituteParametersIntoUserDefinedType(declRef, *this, emitter);

    // Handle __mlir_type["foo"] and __mlir_attr["foo"].
    if (isa<MagicMLIRTypeType>(typeValue)) {
      std::string result = substituteMLIRMagic(*this, emitter);
      if (result.empty())
        return {};
      auto type = parseMLIRType(result, this, emitter.shared);
      if (!type)
        return {};
      return CallableValue({type, this});
    }
    if (isa<MagicMLIRAttrType>(typeValue)) {
      std::string result = substituteMLIRMagic(*this, emitter);
      if (result.empty())
        return {};
      auto attr =
          synthesizeMLIRAttrFromString(result, getLoc(), emitter.shared);
      if (!attr)
        return {};
      return CallableValue({attr, this});
    }
  }

  if (auto mValue = subValue.baseVal.ir.getIfMValue()) {
    if (auto unboundOperator = dyn_cast<UnboundMLIROperationAttr>(mValue.get()))
      return {
          {bindAttributesToMLIROperatorCall(*this, unboundOperator, emitter),
           this}};
  }

  // Emit each of the index values, which will be passed to the __getitem__ and
  // __setitem__ calls.
  SmallVector<ASTExprAnd<AnyValue>> indexValues;
  indexValues.push_back(subValue.baseVal);
  for (ExprNode *index : indices) {
    indexValues.push_back(
        {index->emitIR(emitter, /*No Contextual Type*/ {}), index});
    if (!indexValues.back())
      return {};
  }

  // Okay, we're doing a normal value subscript.  We expect at least a
  // __getitem__ method.
  auto baseType = subValue.baseVal.ir.getRValueType();
  bool isErroneousDecl = false;
  auto getItem = CallableValue(baseType, "__getitem__", getLoc(),
                               isErroneousDecl, emitter.shared);
  // If there is no __getitem__ at all, then this is not a subscriptable type.
  if (getItem.isNull()) {
    if (isErroneousDecl)
      return {};
    emitter.emitError(getLoc(), "")
        << baseType << " does not implement the `__getitem__` method"
        << subValue.baseVal.expr->getRange();
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
  if (getItem.direct &&
      succeeded(getItem.direct->filterOverloadSet(
          indexValues, CallSyntax::kSubscript,
          /*emitDiagnosticOnFailure=*/false, emitter.shared))) {
    // Ok, this looks like it will work.
    // TODO(Computed LValues): We need to look up __setitem__ and have a better
    // model for computed LValues.
  }

  // Finally, just emit the call to __getitem__.
  auto result = getItem.emitFunctionCall(indexValues, CallSyntax::kSubscript,
                                         this, emitter);
  return CallableValue({result, this});
}

AnyValue SubscriptArrowNode::emitIR(ExprEmitter &emitter,
                                    ASTType contextualType) const {
  return emitCallable(emitter, contextualType).emitAsValue(emitter);
}

/// Emit this expression to MLIR as a CallableValue.  On error, emit an error
/// and return a null value.
CallableValue SubscriptArrowNode::emitCallable(ExprEmitter &emitter,
                                               ASTType contextualType) const {

  // Subscripting a generic function binds the parameter expressions.
  auto subValue = base->emitCallable(emitter, {});
  if (!subValue)
    return {};

  // If the subValue has a bound callable symbol, then this is applying (more?)
  // meta values to bind its parameters.
  if (!subValue.direct) {
    emitter.emitError(arrowLoc, "invalid '->' when subscripting type ")
        << ASTType(subValue.baseVal.ir.getType()) << getRange();
    return {};
  }

  // The only use of SubscriptArrow nodes right now is to bind parameter
  // input values and results to a call.  Start by binding the input values.
  subValue = bindAttrValuesToDirectCall(subValue, indices, emitter);
  if (!subValue)
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
        diag.attachNote(emitter.translateLocation(decl->getLoc()))
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
    subValue.direct->resultParams.push_back({resultDecls[0], drn->getLoc()});
  }

  return subValue;
}

AnyValue ParenNode::emitIR(ExprEmitter &emitter, ASTType contextualType) const {
  return subExpr->emitIR(emitter, contextualType);
}

CallableValue ParenNode::emitCallable(ExprEmitter &emitter,
                                      ASTType contextualType) const {
  return subExpr->emitCallable(emitter, contextualType);
}

AnyValue TupleNode::emitIR(ExprEmitter &emitter, ASTType contextualType) const {
  // Emit each of the index values to generate error messages.
  SmallVector<RValue> exprValues;
  for (ExprNode *expr : exprs) {
    exprValues.push_back(emitter.emitExprRValue(expr));
    if (!exprValues.back())
      return {};
  }

  emitter.emitError(getLoc(), "FIXME: Cannot emit tuple expressions yet")
      << getRange();
  return {};
}

AnyValue ListNode::emitIR(ExprEmitter &emitter, ASTType contextualType) const {
  SmallVector<RValue> elements;
  for (ExprNode *expr : exprs) {
    elements.push_back(emitter.emitExprRValue(expr));
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

AnyValue DictionaryNode::emitIR(ExprEmitter &emitter,
                                ASTType contextualType) const {
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

    auto value = emitter.emitExprRValue(keyValue.second);
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
  ParameterEvaluator paramEvaluator(initType.getParamBindings());

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
        fieldVal.ir, fieldVal.expr,
        paramEvaluator.getReboundType(field.getType()),
        " in field initialization");
    auto drValue = emitter.emitDRValue({value, fieldVal.expr});
    if (!drValue)
      return {};
    fieldNames.push_back(field.getNameAttr());
    fieldValues.push_back(drValue);
  }

  return DRValue(emitter.builder->create<StructCreateOp>(
      getLocation(emitter), initType.mlirType, fieldValues,
      StringArrayAttr::get(emitter.getContext(), fieldNames)));
}

AnyValue DictSubscriptNode::emitIR(ExprEmitter &emitter,
                                   ASTType contextualType) const {

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

AnyValue BinOpNode::emitIR(ExprEmitter &emitter, ASTType contextualType) const {
  AnyValue lhsRep;
  RValue rhsRep;

  // We generally emit the LHS before the RHS, but need to do special things
  // for short-circuiting and assignment statements.
  if (kind == kBoolAnd || kind == kBoolOr) // `x and y`, `x or y`
    return emitAndOr(emitter);

  if (!isAssignmentStmt()) {
    auto lhsRV = emitter.emitExprRValue(lhs);
    lhsRep = lhsRV;
    rhsRep = emitter.emitExprRValue(rhs);
    if (!lhsRep || !rhsRep)
      return {};
  } else {
    // In an assignment, we emit the RHS first as a value and the LHS as an
    // lvalue with a contextual type.  This is required to enable the 'implicit
    // declaration' behavior in a def.
    rhsRep = emitter.emitExprRValue(rhs);
    if (!rhsRep)
      return {};

    // Emit the LHS pattern as an lvalue.  Pass in the RHS's type as the
    // contextual type in case we need to implicitly declare a variable.
    auto lhsLV =
        emitter.emitExprLValue(getLoc(), lhs, rhsRep.getType(),
                               "cannot assign to immutable expression");
    if (!lhsLV)
      return {};

    // Assignment expression (`=`) turns into a store, not into a method call.
    if (kind == kAssign) {
      // Emit the RHS and coerce to the LHS type.
      // auto rv = emitter.emitDRValue(ASTExprAnd<RValue>{rhsRep, rhs});
      auto rv = emitter.emitDRValue(
          {emitter.getAsExpectedType(rhsRep, rhs, lhsLV.getRValueType(),
                                     " in assignment"),
           rhs});
      if (!rv)
        return {};

      // If everything worked out, store the resultant value into the lvalue for
      // the destination.  If things didn't work, just drop this on the floor.
      emitter.builder->create<POP::StoreOp>(getLocation(emitter), rv, lhsLV,
                                            /*alignment=*/std::nullopt);
      // Assignments are not actually expressions in Python.  We treat them this
      // way for consistency, but model them as returning None.
      return MValue(NoneAttr::get(emitter.getContext()));
    }

    // Otherwise, handle as a normal binary operator.
    lhsRep = lhsLV;
  }

  // FIXME: We currently hack in index type support as transition to proper
  // expression support.
  if ((lhsRep.getType().isIndex() && rhsRep.getType().isIndex()) &&
      lhsRep.getIfMValue() && rhsRep.getIfMValue()) {
    auto lhsParam = lhsRep.getIfMValue();
    auto rhsParam = rhsRep.getIfMValue();
    POC opcode;
    bool needsInvert = false;
    switch (kind) {
    default:
      emitter.emitError(
          getLoc(), "cannot emit this binary operator in parameter context yet")
          << getRange();
      return {};
    case kSub:
      return ParamOperatorAttr::getSub(lhsParam, rhsParam);
    case kAdd:
      opcode = POC::Add;
      break;
    case kMul:
      opcode = POC::Mul;
      break;
    case kAnd:
      opcode = POC::And;
      break;
    case kOr:
      opcode = POC::Or;
      break;
    case kXor:
      opcode = POC::Xor;
      break;
    case kLShift:
      opcode = POC::Shl;
      break;
    case kRShift:
      opcode = POC::Shr;
      break;
    case kFloorDiv:
      opcode = POC::Div;
      break;
    case kMod:
      opcode = POC::Mod;
      break;
    case kCmpEQ:
      opcode = POC::EQ;
      break;
    case kCmpNE:
      opcode = POC::EQ;
      needsInvert = true;
      break;
    case kCmpGE:
      opcode = POC::LT;
      needsInvert = true;
      break;
    case kCmpGT:
      opcode = POC::LE;
      needsInvert = true;
      break;
    case kCmpLT:
      opcode = POC::LT;
      break;
    case kCmpLE:
      opcode = POC::LE;
      break;
    }
    auto value = ParamOperatorAttr::get((POC)opcode, lhsParam, rhsParam);
    if (needsInvert)
      value = ParamOperatorAttr::getNot(value);
    return value;
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnInfo = getOpSpecialFunctions(kind, /*isReversed=*/false);
  assert(specialFnInfo.kind != SpecialFunctionKind::kNormal);
  ASTExprAnd<AnyValue> argValues[] = {{lhsRep, lhs}, {rhsRep, rhs}};

  // Check to see if we have a forward version of this function on the primary
  // receiver.
  bool isErroneousDecl = false;
  CallableValue callee(lhsRep.getRValueType(), specialFnInfo.name, getLoc(),
                       isErroneousDecl, emitter.shared);
  if (isErroneousDecl)
    return {};
  if (callee.direct &&
      succeeded(callee.direct->filterOverloadSet(
          argValues, CallSyntax::kOperator,
          /*emitDiagnosticOnFailure=*/false, emitter.shared))) {
    return callee.emitFunctionCall(argValues, CallSyntax::kOperator, this,
                                   emitter);
  }

  // Check to see if we have the reverse version of this operator.
  auto reversedFnInfo = getOpSpecialFunctions(kind, /*isReversed=*/true);
  if (reversedFnInfo.kind != SpecialFunctionKind::kNormal) {
    // Swap the operand order.
    std::swap(argValues[0], argValues[1]);
    callee = CallableValue(rhsRep.getType(), reversedFnInfo.name, getLoc(),
                           isErroneousDecl, emitter.shared);
    if (callee.direct &&
        succeeded(callee.direct->filterOverloadSet(
            argValues, CallSyntax::kReversedOperator,
            /*emitDiagnosticOnFailure=*/false, emitter.shared))) {
      return callee.emitFunctionCall(argValues, CallSyntax::kReversedOperator,
                                     this, emitter);
    }

    // Swap these back so we emit the right error.
    std::swap(argValues[0], argValues[1]);
  }

  // Emit an error complaining about the forward version of the operator.
  return emitter.emitNamedMethodCall(specialFnInfo.name, argValues,
                                     CallSyntax::kOperator, this);
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
  Location ifLoc = getLocation(emitter);

  if (!emitter.builder) {
    emitter.emitError(
        getLoc(),
        "TODO: cannot emit short-circuit and/or in this parameter context")
        << lhs->getRange() << rhs->getRange();
    return {};
  }

  // Emit the LHS value and capture the result of calling __bool__ in case we
  // need it.
  AnyValue lhsBool;
  DRValue lhsRV = emitter.emitExprDRValue(lhs);
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
  DRValue rhsRV = emitter.emitExprDRValue(rhs);
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
    auto rhsBool = emitter.emitNamedMethodCall(
        "__bool__", {{rhsRV, rhs}}, CallSyntax::kImplicitConvert, this);
    if (!rhsBool)
      return {};
    if (!ASTType(lhsBool.getType()).isEqualCanon(rhsBool.getType())) {
      emitter.emitError(getLoc(), "cannot find common type between ")
          << ASTType(lhsRV.getType()) << " and " << ASTType(rhsRV.getType())
          << lhs->getRange() << rhs->getRange();
      return {};
    }
    auto rhsBoolDRVal = emitter.emitDRValue({rhsBool, rhs});
    if (!rhsBoolDRVal)
      return {};
    emitter.builder->create<scf::YieldOp>(ifLoc, rhsBoolDRVal);
    // Emit the false side.
    emitter.builder = falseBuilder;
    auto lhsBoolDRVal = emitter.emitDRValue({lhsBool, lhs});
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
  if ((exprRep.getType().isIndex() || exprRep.getType().isF64()) &&
      exprRep.getIfMValue()) {
    auto exprParam = exprRep.getIfMValue();
    switch (kind) {
    default:
      break;
    case ExprNode::kNeg:
      if (auto constantFP = dyn_cast<FloatAttr>(exprParam.get()))
        return MValue(
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
        "__bool__", argValue, CallSyntax::kImplicitConvert, this);
    if (!argValue.ir)
      return {};
    // Now that we know we bool-ized the expression, invert it with ~.
    kindToEmit = kInvert;
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnInfo = getOpSpecialFunctions(kindToEmit, /*isReversed=*/false);
  assert(specialFnInfo.kind != SpecialFunctionKind::kNormal &&
         "Unary operators are implemented via special methods");

  return emitter.emitNamedMethodCall(specialFnInfo.name, argValue,
                                     CallSyntax::kOperator, this);
}

AnyValue IfElseOpNode::emitIR(ExprEmitter &emitter,
                              ASTType contextualType) const {
  auto condValue = emitter.emitExprConditionValueAsI1(condExpr);
  if (!condValue)
    return {};

  if (!emitter.builder) {
    emitter.emitError(getLoc(),
                      "TODO: cannot emit if/else in parameter expression yet")
        << trueExpr->getRange();
    return {};
  }

  Location ifLoc = getLocation(emitter);
  // At this point we don't know the type of trueExpr / falseExpr, use
  // a dummy one and fix it later.
  auto ifOp = emitter.builder->create<scf::IfOp>(
      ifLoc, TypeRange{condValue.getType()}, condValue, /*withElse=*/true);
  emitter.builder = ifOp.getThenBodyBuilder();
  DRValue trueVal = emitter.emitExprDRValue(trueExpr);
  if (!trueVal)
    return {};
  emitter.builder->create<scf::YieldOp>(ifLoc, trueVal);
  emitter.builder = ifOp.getElseBodyBuilder();
  DRValue falseVal = emitter.emitExprDRValue(falseExpr);
  if (!falseVal)
    return {};
  emitter.builder->create<scf::YieldOp>(ifLoc, falseVal);
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
  return DRValue(ifOp.getResult(0));
}
