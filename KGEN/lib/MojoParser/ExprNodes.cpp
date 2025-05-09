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

#include "CallEmission.h"
#include "DLValues.h"
#include "ExprEmitter.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/SharedState.h"
#include "ParserEvaluationContext.h"

#include "ExprNodes.h"
#include "MojoUtils.h"
#include "Signatures.h"
#include "Traits.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Given a StringRef for an MLIR attribute, invoke the MLIR parser to resolve
/// it into an Attribute (which may not be a TypedAttr) and return it.  On
/// error, emit a diagnostic and return null.
static ErrorOr<Attribute> parseMLIRAttrFromString(StringRef name, SMLoc loc,
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

  if (!result)
    return Error(errorMsg);

  return result;
}

static Attribute parseMLIRAttrFromStringWithError(StringRef name, SMLoc loc,
                                                  SharedState &shared) {
  auto result = parseMLIRAttrFromString(name, loc, shared);
  if (result.isError()) {
    auto diag = shared.emitError(loc, "invalid MLIR attribute: ")
                << result.takeError().get();
    diag.attachNote(loc) << "attempting to parse: '" << name << "'";
    return {};
  }
  return result.takeValue();
}

/// This implements __mlir_attr.x lookup, synthesizing a PValue for the
/// attribute on demand.
static PValue synthesizeMLIRAttrFromString(StringRef name, SMLoc loc,
                                           SharedState &shared,
                                           bool reportError = true) {
  Attribute attr;
  if (reportError) {
    attr = parseMLIRAttrFromStringWithError(name, loc, shared);
  } else {
    // If we were not able to build a string right now, we will wrap this
    // attribute with `#kgen.deferred` and let elaborator try to build attribute
    // again.
    auto res = parseMLIRAttrFromString(name, loc, shared);
    attr = res.isError() ? nullptr : res.takeValue();
  }

  if (!attr)
    return {};

  auto typedAttr = dyn_cast<TypedAttr>(attr);
  if (!typedAttr) {
    // wrap non-typed attribute with #kgen.deferred attribute
    typedAttr = DeferredAttr::get(attr);
  }
  return PValue(typedAttr);
}

/// Given an __mlir_type[a,b,c] or __mlir_attr[a,b,c] usage, stringize the
/// subscript operands and return the result.  On error, emit an error and
/// return an empty string.
static std::string substituteMLIRMagic(const SubscriptNode &node,
                                       ExprEmitter &emitter) {
  std::string result;
  llvm::raw_string_ostream os(result);

  SMLoc loc = node.getLoc();
  for (const Operand &operand : node.operands) {
    ExprNode *expr = operand.expr;
    if (!operand.isPositional()) {
      emitter.emitError(loc, "only positional operands allowed in mlir magic")
          << expr->getRange();
      return {};
    }

    // If the index is an identifier, and if it is a backtick identifier, we
    // treat it as an interpolated literal string.  Otherwise we look it up as
    // an expression.  Rationale: this allows using strings attributes, which
    // could be useful someday, and keeps __mlir_attr.`thing` more consistent
    // with __mlir_attr[`thing`].
    if (auto *dre = dyn_cast<DeclRefNode>(expr))
      if (dre->spelling.data()[dre->spelling.size()] == '`') {
        os << dre->spelling;
        continue;
      }

    // As a very special hack, we treat a unary plus as a marker that the type
    // should not be printed when the attribute is stringized.
    bool elideType = false;
    if (expr->kind == ExprNode::kPos) {
      elideType = true;
      expr = cast<UnaryOpNode>(expr)->subExpr;
    }

    auto indexVal = emitter.emitExprPValue(expr, EC_MLIRMagic);
    if (!indexVal)
      return "";

    // If this is a wrapper for a type, print it as such.
    if (isa<TraitType>(indexVal.getType())) {
      // values of trait type are printed in a kgen compatible way, e.g.
      // "":!lit.trait<@stdlib::@builtin::@stubs::@AnyType> someParamValue"
      if (!elideType)
        os << ":" << ASTType(indexVal.getType()).mlirType << " ";
      os << ASTType(indexVal).mlirType;
    } else if (isa<TypeType, StructMetaType, AnyTraitType>(indexVal.getType()))
      os << ASTType(indexVal).mlirType;
    else // Otherwise print it as an attribute.
      indexVal.get().print(os, elideType);
  }

  if (result.empty())
    emitter.emitError(loc, "mlir magic expanded to an empty string");
  return result;
}

static AttrCtorDeferredAttr
buildAttrCtorDeferredAttrFromMLIRAttr(const SubscriptNode &node,
                                      ExprEmitter &emitter) {
  SmallVector<TypedAttr> strings;
  SMLoc loc = node.getLoc();
  for (const Operand &operand : node.operands) {
    ExprNode *expr = operand.expr;
    if (!operand.isPositional()) {
      emitter.emitError(loc, "only positional operands allowed in mlir magic")
          << expr->getRange();
      return {};
    }

    // If the index is an identifier, and if it is a backtick identifier, we
    // treat it as an interpolated literal string.  Otherwise we look it up as
    // an expression.  Rationale: this allows using strings attributes, which
    // could be useful someday, and keeps __mlir_attr.`thing` more consistent
    // with __mlir_attr[`thing`].
    if (auto *dre = dyn_cast<DeclRefNode>(expr))
      if (dre->spelling.data()[dre->spelling.size()] == '`') {
        strings.push_back(StringAttr::get(emitter.getContext(), dre->spelling));
        continue;
      }

    bool elideType = false;
    if (expr->kind == ExprNode::kPos) {
      elideType = true;
      expr = cast<UnaryOpNode>(expr)->subExpr;
    }

    auto indexVal = emitter.emitExprPValue(expr, EC_MLIRMagic);
    if (!indexVal)
      return {};

    strings.push_back(ToStringDeferredAttr::get(indexVal.get(), elideType));
  }

  if (strings.empty())
    emitter.emitError(loc, "mlir magic expanded to an empty string");

  return AttrCtorDeferredAttr::get(strings);
}

#ifndef NDEBUG
/// Return the string representation of the \p attr by concatenating strings it
/// holds.
/// Pass extra argument \p node to properly query how type information needs to
/// be printed.
static std::string getStringRepresentation(AttrCtorDeferredAttr attr) {
  std::string result;
  llvm::raw_string_ostream os(result);
  for (auto str : attr.getStrings()) {
    if (auto strAttr = dyn_cast<StringAttr>(str)) {
      os << strAttr.str();
    } else if (auto toStrAttr = dyn_cast<ToStringDeferredAttr>(str)) {
      auto val = cast<TypedAttr>(toStrAttr.getAttr());
      bool elideType = toStrAttr.getNeedElideType() != nullptr;

      if (isa<TraitType>(val.getType())) {
        // values of trait type are printed in a kgen compatible way, e.g.
        // "":!lit.trait<@stdlib::@builtin::@stubs::@AnyType> someParamValue"
        if (!elideType)
          os << ":" << ASTType(val.getType()).mlirType << " ";
        os << ASTType(val).mlirType;
      } else if (isa<TypeType, StructMetaType, AnyTraitType>(val.getType()))
        os << ASTType(val).mlirType;
      else // Otherwise print it as an attribute.
        val.print(os, elideType);
    } else {
      llvm_unreachable("unexpected attribute type");
    }
  }

  return result;
}
#endif // NDEBUG

/// Report a warning to let user know that they should use __mlir_attr instead.
static void emitMLIRDeferedAttrToMLIRAttrWarning(SMLoc loc,
                                                 ExprEmitter &emitter) {
  emitter.emitWarning(loc)
      << "trivially constructable attribute. Use `__mlir_attr` "
         "instead.";
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

/// Given an expression, try to resolve it into an Attribute that we can install
/// on this operation.
static Attribute getAttrFromExpr(StringRef name, ExprNode *node,
                                 ExprEmitter &emitter) {
  // Special case handling of __mlir_attr.`xxx` directly in this parser,
  // because we want to be able to install arbitrary attributes into an
  // operation's attribute list, and emitPValue only supports TypedAttrs.
  if (auto attrRef = dyn_cast<AttributeRefNode>(node)) {
    auto mlirAttr = dyn_cast<DeclRefNode>(attrRef->base);
    if (mlirAttr && mlirAttr->spelling == "__mlir_attr") {
      if (attrRef->spelling.empty())
        return {};
      return parseMLIRAttrFromStringWithError(
          attrRef->spelling, attrRef->getLoc(), emitter.shared);
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
      return parseMLIRAttrFromStringWithError(result, subscript->getLoc(),
                                              emitter.shared);
    }
  }

  // Otherwise emit the value as an PValue.
  return emitter.emitExprPValue(node, EC_MLIRMagic);
}

/// Calculate the result of an __mlir_op.`thing`[attributes], applying the
/// attributes list to the operation specification.
static PValue
bindAttributesToMLIROperatorCall(const SubscriptNode &subscript,
                                 UnboundMLIROperationAttr unboundOp,
                                 ExprEmitter &emitter) {
  SMLoc loc = subscript.getLoc();
  MLIRContext *context = emitter.getContext();

  // Only allow applying attributes to something without them.
  if (!unboundOp.getAttrs().empty()) {
    emitter.emitError(loc, "operation already has attributes")
        << subscript.getRange();
    return {};
  }

  // Each element of the subscript must have a name identifier and a value as an
  // PValue.
  SmallVector<NamedAttribute> attrValues;
  for (const Operand &operand : subscript.operands) {
    ExprNode *valueExpr = operand.expr;
    if (!operand.isKeyword()) {
      InflightDiag diag =
          emitter.emitError(loc, "attribute spec requires a keyword parameter");

      // Jump through some hoops to emit a hint about using the old syntax.
      if (auto *slice = dyn_cast<SliceNode>(valueExpr);
          slice && slice->upper && !slice->colon2Loc.isValid()) {
        if (auto *kwRef = dyn_cast_or_null<DeclRefNode>(slice->lower))
          diag << "; did you mean '" << kwRef->spelling << "=...'?"
               << FixIt::replaceToken(slice->colon1Loc, "=");
      }

      diag << valueExpr->getRange();
      return {};
    }

    auto value = getAttrFromExpr(operand.name, valueExpr, emitter);
    if (!value)
      return {};
    attrValues.push_back({operand.name, value});
  }

  // Return it.
  auto attrs = DictionaryAttr::get(context, attrValues);
  return UnboundMLIROperationAttr::get(unboundOp.getName(), attrs);
}

/// Given a reference to an alias (either a direct reference from a DeclRefNode
/// "x" or an AttributeRefNode "x.y"), return the PValue for the result.
///
/// 'decl' is the alias declaration to resolve.
PValue M::KGEN::LIT::resolveAliasReference(AliasDeclOp decl, StringRef declName,
                                           ArrayRef<TypedAttr> paramValues,
                                           SMLoc errLoc, SharedState &shared) {

  // If the param is declared in a function, then just directly use it.
  Operation *parent = decl->getParentOp();
  while (parent && !isa<FileModuleOp, FnOp>(parent)) {

    // If this reference is within a trait then keep it symbolic since the
    // conforming type will ultimately provide the value to use.
    // TODO: What does it mean to write "SomeTrait.alias"?  This handles a
    // reference from within the body of the trait, but wouldn't a reference
    // from outside of it be an error?
    if (isa<TraitDeclOp>(parent))
      return ParamDeclRefAttr::get(decl.getName(), decl.getType());

    // If this is in a struct, then the value may refer to parameters declared
    // on the struct, whose values come through 'bindings'.  Remap.
    if (auto structDecl = dyn_cast<StructDeclOp>(parent)) {
      assert(decl.getValueAttr() && "Struct's alias should have value");
      ArrayRef<ParamDeclAttr> paramDecls = structDecl.getParams();
      assert(paramDecls.size() == paramValues.size() &&
             "incorrect number of type parameters for struct");

      // If the reference is to a member of the struct that has bindings, remap
      // them.  This allows things like `SomeType[a,b].someAlias` to substitute
      // the a/b values into the body of `someAlias`.  Note that the type may
      // be only partially bound: e.g. you might use `SomeType.someAlias` or
      // `SomeType[b=42].someAlias`. We want to support this so long as the
      // alias doesn't refer to an unbound parameter:
      //    struct X[a: Int, b: Int]:
      //        alias a1 = 42
      //        alias a2 = a+1
      //    fn test():
      //        use(X.a1) # Ok
      //        use(X[1].a2) # Ok
      //        use(X.a2) # Error: 'a' needs to be bound
      //
      // TODO: Should this return a parametric alias instead?
      ParserParameterEvaluator evaluator(shared, paramDecls, paramValues);
      TypedAttr result = evaluator.getReboundAttribute(decl.getValueAttr());
      // Check to make sure that no unbound parameters were used.
      if (!result
               .walk([&](UnboundAttr) -> WalkResult {
                 return WalkResult::interrupt();
               })
               .wasInterrupted()) {
        // If everything is ok, then we're done.
        return result;
      }

      // FIXME: We currently have a weird version of "parameterized aliases"
      // that accidentally works out, including things like:
      //
      //   struct _lit_mut_cast[
      //       is_mutable: Bool, //,
      //       result_mutable: Bool,
      //       operand: Origin[is_mutable],
      //   ]: ...
      //   struct Origin[is_mutable: Bool]:
      //       alias cast_from = _lit_mut_cast[result_mutable=is_mutable]
      //
      // The access to SomeOrigin.cast_from returns an alias that has unbound
      // parameters from `_lit_mut_cast`, instead of returning a parameterized
      // alias.  We want this to continue to work until it gets built the right
      // way, so perpetuate a hack that retains it.
      if (!llvm::count_if(paramValues, [](TypedAttr attr) {
            return isa<UnboundAttr>(attr);
          }))
        return result;

      // Otherwise, we need to diagnose an error.  We don't know which
      // UnboundAttr is the problem, so work harder to get a good error message:
      // We replace the UnboundAttrs with something that has a name we can
      // diagnose.
      SmallVector<TypedAttr> paramsToBind;
      for (auto [paramDecl, paramValue] : llvm::zip(paramDecls, paramValues)) {
        if (!isa<UnboundAttr>(paramValue))
          paramsToBind.push_back(paramValue);
        else {
          // The type may have been remapped due to earlier parameters, so use
          // the type of paramValue, not paramDecl.
          paramsToBind.push_back(
              ParamDeclRefAttr::get(paramDecl.getName(), paramValue.getType()));
        }
      }
      ParserParameterEvaluator evaluatorForError(shared, paramDecls,
                                                 paramsToBind);
      result = evaluatorForError.getReboundAttribute(decl.getValueAttr());

      // Check to make sure that no unbound parameters were used.
      result.walk([&](ParamDeclRefAttr attr) -> WalkResult {
        shared.emitError(errLoc, "cannot access alias '")
            << declName << "' with unbound parameter '" << structDecl.getName()
            << "." << attr.getName().str() << "'";
        result = TypedAttr();
        return WalkResult::interrupt();
      });
      assert(!result && "didn't find the problematic parameter");
      return {};
    }

    // Ignore 'if' and other control flow things.
    parent = parent->getParentOp();
  }

  // If this is at file or function scope, inline the value of the alias.
  return decl.getValueAttr();
}

//===----------------------------------------------------------------------===//
// ExprNode Implementation
//===----------------------------------------------------------------------===//

ExprNode::~ExprNode() = default;

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

AnyValue BoolLiteralNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // Create the SIMDAttr to represent the constant.
  auto boolAttr = BoolAttr::get(emitter.getContext(), value);

  // Convert this to an instance of Bool. Bool must be in scope since it is
  // auto-imported.
  return emitter.emitBool({boolAttr, this}, dest);
}

AnyValue SimpleLiteralNode::emitIR(ValueDest &dest,
                                   ExprEmitter &emitter) const {
  if (kind == kNoneLiteral)
    return emitter.emitResult(emitter.shared.getNoneAttr(), this, dest);

  if (kind == kDiscardLiteral) {
    ASTType initializerType = dest.getIfLValueInitializerType();
    // The discard pattern can only be used in case where we have an inferred
    // type for the lvalue.
    if (!initializerType) {
      emitter.emitError(getLoc(), "cannot read from discard pattern '_'");
      return {};
    }
    DLValue result(RCRef<DiscardDLValue>::create(initializerType, this));
    return emitter.emitResult(result, this, dest);
  }

  assert(kind == kSelfLiteral && "Unknown simple literal kind");
  // Self resolves to the type of the enclosing structure type.
  ASTDecl *astDecl =
      emitter.declScope.getNearestDeclOfType<StructDeclOp, TraitDeclOp>();
  if (!astDecl) {
    emitter.emitError(getLoc(),
                      "'Self' type may only be used inside a struct or trait");
    return {};
  }

  // Notify the listener that the Self is a reference of the parent
  // struct.
  emitter.shared.notifyListenerOnRef(astDecl, "Self", getRange());

  // Once we have the type in question we can just return its Self type as an
  // PValue.  This already includes bound parameters etc.
  assert(astDecl->resolvedness >= DeclResolvedness::signature);
  return emitter.emitResult(astDecl->getTypeDeclSelf(), this, dest);
}

// Handle generation of a constructor call to one of::
//     IntLiteral[value: __mlir_type.`!pop.int_literal`]
//     FloatLiteral[value: __mlir_type.`!pop.float_literal`]
//     StringLiteral[value: __mlir_type.`!kgen.string`]
static CValue handleIntFPStringLiteral(TypedAttr value, ASTType type,
                                       const ExprNode *expr, ValueDest &dest,
                                       ExprEmitter &emitter) {
  if (isa<TypeCheckErrorType>(type))
    return {}; // Sanity check the returned declaration.
  ASTDecl *decl = type.getDecl(emitter.shared);

  auto litStruct = dyn_cast_if_present<StructDeclOp>(decl);
  if (!litStruct || litStruct.getParams().size() != 1 ||
      !isa<POP::IntLiteralType, POP::FloatLiteralType, StringType>(
          litStruct.getParams()[0].getType())) {
    emitter.emitError(expr->getLoc(), "malformed Literal type");
    return {};
  }

  // Bind the IntLiteral[value] parameter.
  type = litStruct.bindReference({value});

  // Create an instance of IntLiteral[val]()
  return emitter.emitConstructorCall(type, CallOperands(), expr,
                                     CallSyntax::kTypeCall, dest);
}

AnyValue IntLiteralNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  APInt value = Lexer::getIntegerLiteralValue(spelling);
  // Values produced are sometimes produced unsigned, so we must add an extra
  // sign bit.
  if (value.slt(APInt::getZero(value.getBitWidth())))
    value = value.zext(value.getBitWidth() + 1);
  auto attr = POP::IntLiteralAttr::get(emitter.getContext(), IPInt(value));

  // Look up the IntLiteral type.
  ASTType type =
      emitter.shared.getBuiltinIntLiteralType(emitter.declScope, getLoc());

  return handleIntFPStringLiteral(attr, type, this, dest, emitter);
}

AnyValue FloatLiteralNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  IPRational value = Lexer::getFloatLiteralValue(spelling);
  auto attr = POP::FloatLiteralAttr::get(emitter.getContext(), value);
  ASTType type =
      emitter.shared.getBuiltinFloatLiteralType(emitter.declScope, getLoc());
  return handleIntFPStringLiteral(attr, type, this, dest, emitter);
}

/// The value of a string is the concatenated value with escapes and quotes
/// removed.
std::string StringLiteralNode::getValue() const {
  std::string result;
  for (auto spelling : spellings)
    result += Lexer::getStringLiteralValue(spelling);
  return result;
}

/// Emit a constructor call for a string literal with the specified data, that
/// does not include enclosing quotes.  The specified expression specifies the
/// location but need not by a StringLiteral.
CValue StringLiteralNode::emitCtorCall(StringRef bytes, const ExprNode *expr,
                                       ValueDest &dest, ExprEmitter &emitter) {
  auto attr = StringAttr::get(bytes, StringType::get(emitter.getContext()));
  ASTType type = emitter.shared.getBuiltinStringLiteralType(emitter.declScope,
                                                            expr->getLoc());
  return handleIntFPStringLiteral(attr, type, expr, dest, emitter);
}

AnyValue StringLiteralNode::emitIR(ValueDest &dest,
                                   ExprEmitter &emitter) const {
  // Flatten multiple concat'd strings, remove """'s and "'s etc.
  std::string value = getValue();
  return emitCtorCall(value, this, dest, emitter);
}

bool Operand::isPositionalStringLiteral(StringRef str) const {
  if (isPositional())
    if (auto *strExpr = dyn_cast<StringLiteralNode>(expr))
      return strExpr->getValue() == str;
  return false;
}

AnyValue SyntheticNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  assert(irValue && "emitIR is undefined for synthetic nodes.");
  return emitter.emitResult(irValue, this, dest);
}

/// When analyzing a DeclRefNode lookup result in a context that allows implicit
/// variable definitions, check to see if the lookup set contains immutable
/// symbols found through global lookup. If so, return true.
static bool isImmutableValuesInOtherScope(const LookupResult &lookup,
                                          ExprEmitter &emitter) {
  for (ASTDecl *decl : lookup.getIfSuccess()) {
    // If this contains anything mutable, return false.
    if (isa<VarDeclOp, GlobalVarDeclOp>(*decl) ||
        decl->getIfIRValue().getIfLValue())
      return false;

    // If this is an immutable thing in the current scope, then return false.
    if (decl->getParentDecl() == &emitter.declScope)
      return false;
  }

  return true;
}

/// Emit IR for an unqualified declaration reference "x" looked up in current
/// context.
AnyValue DeclRefNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  ASTDecl &container = emitter.declScope;

  // Notify the listener of a normal decl reference lookup.
  emitter.shared.notifyListenerOnMemberLookup(container, getLoc(),
                                              /*searchParentScopes=*/true);

  // Perform a lookup of the specified decl in the current container.
  LookupResult lookup = emitter.shared.lookupAndResolveDecl(
      spelling, getLoc(), container, /*searchParentScopes=*/true);

  // If we're in a function and have a contextual type, then this may be an
  // implicit declaration of a variable.  However, name lookup could find
  // global symbols (e.g. the "slice" function in `slice = foo()`) which are
  // obviously not mutable.  Handle this by filtering out the overload set if
  // it is obviously not mutable, but we know we're in an lvalue context with
  // inferred type.
  if (emitter.varDeclCursor && dest.getIfLValueInitializerType() &&
      !lookup.isFailure() && isImmutableValuesInOtherScope(lookup, emitter))
    lookup = LookupResult::getFailure({});

  // If that lookup failed, but we can synthesize a variable declaration in this
  // scope, do that.  We can only do this if there is a varDeclCursor,
  // indicating that we're in a `def` node, and if we have a contextual type
  // (which tells us we need to emit an LValue).
  if (lookup.isFailure() && emitter.varDeclCursor &&
      dest.getIfLValueInitializerType()) {
    auto contextualType = dest.getIfLValueInitializerType();
    assert(contextualType && "must have contextual type");

    // Use this builder to place any VarDeclOps. In Python there is only one
    // scope for the whole def and all variables belong to that scope.
    OpBuilder varDeclBuilder(
        emitter.varDeclCursor->getInsertionBlock(),
        std::next(emitter.varDeclCursor->getInsertionPoint()));
    ExprEmitter varDeclEmitter(emitter.declScope, varDeclBuilder);

    // Add implicitly declared variable to the name table OF THE FUNCTION, so
    // subsequent uses find this one.  We don't want implicit declarations in
    // different subscopes to get different implicit declarations.
    ASTDecl *scopeToInsert = &container;
    while (!isa<FnOp>(*scopeToInsert)) {
      scopeToInsert = scopeToInsert->getParentDecl();
      assert(scopeToInsert && "not in a def?");
    }

    // Get the raw FileLineColLoc, and fuse with the debug scope of the
    // container if it exists.
    Location varDeclLoc = emitter.shared.diags.translateLocation(getLoc());
    if (DebugInfo::DISubprogramAttr varDeclSubprogram = DebugInfo::extractScope(
            cast<mlir::FunctionOpInterface>(scopeToInsert))) {
      varDeclLoc = mlir::FusedLoc::get(emitter.getContext(), {varDeclLoc},
                                       varDeclSubprogram);
    }
    VarDeclOp varDecl =
        varDeclEmitter.emitVarDecl(spelling, contextualType, varDeclLoc,
                                   // Marked Implicit to disable warnings.
                                   VarDeclKind::Implicit);

    ASTDecl &varASTDecl = emitter.getDeclResolver().addFullyResolvedDecl(
        DeclIRValue(varDecl), varDecl.getNameAttr(), getLoc(), scopeToInsert);
    emitter.shared.notifyListenerOnVariableDecl(varASTDecl, getLoc());

    return emitter.emitResult(MLValue(varDecl), this, dest);
  }

  ArrayRef<ASTDecl *> decls = lookup.getIfSuccess();
  if (decls.empty()) {
    if (lookup.isErroneous())
      return {}; // Error already diagnosed.
    ArrayRef<ASTDecl *> failureDecls = lookup.getIfFailure();
    if (!failureDecls.empty()) {
      // Reject unqualified struct field references.
      if (auto fieldOp = dyn_cast<StructFieldOp>(failureDecls[0])) {
        emitter.emitError(getLoc(), "cannot access instance field '")
            << spelling << "' directly; did you mean 'self.'?" << getRange()
            << FixIt::insertBeforeToken(getLoc(), "self.");
        return {};
        // Rejected unqualified struct method references.
      } else if (isa<StructDeclOp>(*failureDecls[0]->getParentDecl())) {
        const char *replacement = "self.";
        // References to static methods can always use capital Self.
        if (auto firstCandidate = dyn_cast<FnOp>(failureDecls[0]))
          if (firstCandidate.getIsStatic())
            replacement = "Self.";

        // References /from/ static methods can only use capital Self.
        if (auto curFn = dyn_cast<FnOp>(container))
          if (curFn.getIsStatic())
            replacement = "Self.";

        emitter.emitError(getLoc(), "cannot access method '")
            << spelling << "' directly; did you mean '" << replacement << "'?"
            << getRange() << FixIt::insertBeforeToken(getLoc(), replacement);
        return {};
      }
    }

    auto diag = emitter.emitError(getLoc()) << getRange();
    if (auto structDecl = dyn_cast<StructDeclOp>(container))
      diag << structDecl.getNameAttr() << " has no '" << spelling << "' member";
    else
      diag << "use of unknown declaration '" << spelling << "'";
    return {};
  }

  emitter.shared.notifyListenerOnRef(decls, spelling, this);

  // Functions form an address, and may be overloaded.
  if (auto firstCandidate = dyn_cast<FnOp>(decls[0])) {
    // Form an overload set value with all the candidates.
    auto result = OverloadSetUValue::create(
        spelling, decls, ParamBindings(emitter.getDeclScope()), this,
        CallSyntax::kDirectCall);
    return emitter.emitResult(result, this, dest);
  }

  assert(decls.size() == 1 && "Only functions may be overloaded");
  ASTDecl &decl = *decls[0];

  // If the referenced decl is deprecated, emit a deprecation warning.
  // Overloaded declarations like functions can't be handled here. They are
  // handled when overload sets are resolved to a deprecated entry.
  if (auto declItf = dyn_cast<ASTDeclInterface>(decl)) {
    if (StringAttr warning = declItf.getDeprecationWarningAttr()) {
      auto diag = emitter.emitWarning(getLoc(), warning.getValue())
                  << getRange();
      diag.attachNote(decl.getLoc())
          << "'" << declItf.getDeclName().getValue() << "' declared here";
    }
  }

  // Aliases form a PValue.
  if (auto param = dyn_cast<AliasDeclOp>(decl)) {
    PValue result = resolveAliasReference(param, spelling, /*bindings=*/{},
                                          getLoc(), emitter.shared);
    return emitter.emitResult(result.get(), this, dest);
  }

  // If this is a type declaration, return it as a type.
  if (auto structOp = dyn_cast<StructDeclOp>(decl))
    return emitter.emitResult(structOp.bindReference(), this, dest);
  if (auto traitOp = dyn_cast<TraitDeclOp>(decl))
    return emitter.emitResult(traitOp.bindReference(), this, dest);

  // If this is a module or package declaration, form a module reference.
  if (isa<FileModuleOp, PackageOp>(decl)) {
    PValue result(ModuleAttr::get(StructMetaType::get(LIT::StructType::get(
        decl.getSymbolRef(), TypeSignatureType::get(emitter.getContext())))));
    return emitter.emitResult(result, this, dest);
  }

  if (auto pvalue = decl.getIfIRValue().getIfPValue())
    return emitter.emitResult(pvalue, this, dest);

  // Narrow the decl to a CValue.
  CValue value;
  if (auto var = dyn_cast<VarDeclOp>(decl)) {
    value = MLValue(var); // Var decls are always mutable.
  } else if (auto globalOp = dyn_cast<GlobalVarDeclOp>(decl)) {
    // If this is a parameter context then we cannot return a dynamic field.
    if (!emitter.builder)
      return emitter.emitErrorForDynamicValueInParameter(this);
    // Return a mutable value only if the global variable is mutable.
    auto ref =
        emitter.builder->create<GlobalVarRefOp>(getLocation(emitter), globalOp);
    value = MLValue(ref);
  } else if (auto cv = decl.getIfIRValue()) {
    value = cv;
  } else {
    emitter.emitError(getLoc(), "use of declaration '")
        << spelling << "' as a value isn't supported yet" << getRange();
    return {};
  }

  // Now that we're referencing a potentially dynamic value, see if it is from
  // an outer function.  If so, record it as a capture in this nested function.
  ASTDecl *declRef = nullptr;
  if (!isa<FnOp>(*decls[0])) {
    assert(decls.size() == 1 && "Only functions may be overloaded");
    declRef = decls[0];
  }

  // Find the nearest escaping closure, if there is one.
  ASTDecl *nearestEscapingFnOrNone =
      declRef ? container.getNearestDeclOfType<FnOp>() : nullptr;
  if (nearestEscapingFnOrNone) {
    assert(declRef && "can only reach here if single decl known");
    auto needsCapture = [&]() -> bool {
      for (ASTDecl *decl = declRef->getParentDecl(); decl;
           decl = decl->getParentDecl()) {
        if (decl == nearestEscapingFnOrNone)
          return false;
      }
      return true;
    };

    // If this is a reference to a value from an outer function scope, record
    // the capture.
    if (needsCapture()) {
      emitter.shared.addCaptureToScope(*nearestEscapingFnOrNone, declRef,
                                       Capture(value, Capture::kRef));
    }
  }

  return emitter.emitResult(value, this, dest);
}

/// This uses the MLIR parser to turn the specified MLIR type name into an MLIR
/// type.
static ASTType parseMLIRType(StringRef name, const ExprNode *node,
                             SharedState &shared) {
  Type result;
  std::vector<Diagnostic> typeDiagnostics;
  {
    // Capture errors thrown by parseType.
    auto diagHandler = [&](Diagnostic &diag) {
      typeDiagnostics.push_back(std::move(diag));
    };
    mlir::ScopedDiagnosticHandler handler(shared.getContext(),
                                          std::move(diagHandler));
    result = mlir::parseType(name, shared.getContext());
  }
  if (!result) {
    InflightDiag diagnostic =
        shared.emitError(node->getLoc(), "invalid MLIR type: ")
        << name << node->getRange();
    for (Diagnostic &diag : typeDiagnostics) {
      std::string str;
      llvm::raw_string_ostream os(str);
      diag.print(os);
      diagnostic.attachNote(node->getLoc()) << "MLIR error: " << str;
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

  // Keep things in the parameter expression domain if we can.
  if (PValue baseMV = base.ir.getIfPValue()) {
    auto extractVal = LIT::StructExtractAttr::get(baseMV.get(), fieldOp);
    return emitter.emitCResult(PValue(extractVal), expr, dest);
  }

  // Okay, handle dynamic field references.
  if (!emitter.builder) {
    emitter.emitErrorForDynamicValueInParameter(expr);
    return {};
  }

  // If we got a trivial SSA value, extract the field and return as SBValue.
  if (base.ir.isSValue() &&
      base.ir.getRValueType().isTrivial(expr->getLoc(), emitter.shared)) {
    auto extractVal = emitter.builder->create<StructExtractOp>(
        mlirLoc, base.ir.getSValueRegister(), fieldOp);
    return emitter.emitCResult(SBValue(extractVal), expr, dest);
  }

  // Otherwise emit this to memory, and work with that.
  MBValue baseBVal = emitter.emitMBValue(base, dest.getContext());
  if (!baseBVal)
    return {};

  auto fieldRef =
      emitter.builder->create<RefStructGEROp>(mlirLoc, baseBVal, fieldOp);
  // Result kind depends on the input kind.
  CValue result;
  if (base.ir.getIfMLValue())
    result = MLValue(fieldRef);
  else if (base.ir.getIfMBPValue())
    result = MBPValue(fieldRef);
  else
    result = MBValue(fieldRef);
  return emitter.emitCResult(result, expr, dest);
}

namespace {
class InProgressBindings {
public:
  InProgressBindings(ArrayRef<Type> paramTypes, PogListAttr paramList,
                     unsigned numPosBindings = 0)
      : paramTypes(paramTypes), paramList(paramList), posIdx(numPosBindings),
        seenKeyword(false), numInferred(countNumInferredKinds(paramList)) {}

  /// Given the current set of bindings and the next operand to bind, determine
  /// the next expected parameter type according to the current in-progress
  /// bindings.
  ASTType getNextParamType(const Operand &operand,
                           const ParamBindings &bindings, ExprEmitter &emitter);

private:
  /// The types of the parameters in the list.
  ArrayRef<Type> paramTypes;
  /// The passing kinds of the parameters in the list.
  PogListAttr paramList;
  /// The number of positionally-bound parameters processed so far.
  unsigned posIdx;
  /// If any non-inferred keyword parameters have been processed so far.
  bool seenKeyword;
  /// The number of inferred parameters in the list.
  const unsigned numInferred;
};
} // namespace

ASTType InProgressBindings::getNextParamType(const Operand &operand,
                                             const ParamBindings &paramBindings,
                                             ExprEmitter &emitter) {
  // Unpacked operands don't have expected types.
  if (operand.isUnpacked())
    return {};

  // Identify which parameter type corresponds to this operand.
  // FIXME: This function doesn't emit any diagnostics, and simply fails to
  // determine the next type if verification fails. If the user passes a context
  // sensitive expression to an invalid parameter, they get an ambiguity error
  // instead of an incorrect parameter error.
  ASTType nextType;
  unsigned paramIdx;
  if (operand.isPositional()) {
    if (seenKeyword) {
      emitter.emitError(operand.getLoc(),
                        "positional parameter follows keyword parameter");
      return {};
    }

    // This parameter is lexically passed positionally. Skip over the inferred
    // parameters to find its type.
    paramIdx = numInferred + posIdx;
    if (paramIdx >= paramTypes.size())
      return {}; // out-of-bounds
    nextType = paramTypes[paramIdx];

    // Remember that another parameter was passed positionally, if the current
    // parameter isn't variadic.
    posIdx += !paramList.isPosVarArg(paramIdx);
  } else {
    // This parameter is passed with a keyword. Find a matching keyword
    // parameter on the parameter list.
    assert(operand.isKeyword() && "expected a keyword operand");
    for (auto [i, type] : llvm::enumerate(paramTypes)) {
      if (paramList.getName(i) == operand.name) {
        paramIdx = i;
        nextType = type;
        break;
      }
    }
    // Failed to find a matching keyword parameter.
    if (!nextType)
      return {};
    // Remember that we've seen a non-inferred keyword operand for error
    // checking.
    if (paramList.getPassingKind(paramIdx) != PassingKind::Inferred)
      seenKeyword = true;
  }

  // Attempt to apply the current binding set to the parameter list.
  ParameterExprArrayAttr bindings =
      paramBindings.verifyBindings(paramTypes, paramList, /*partial=*/true);
  // If that failed, return
  if (!bindings)
    return {};

  // Use the bindings determined so far to specialize the type.
  ParserParameterEvaluator evaluator(emitter.shared, bindings);
  nextType = evaluator.getReboundType(nextType);

  // Unwrap the variadic element type.
  if (paramList.isPosVarArg(paramIdx))
    nextType = nextType.getVariadicElementType();

  // HACK: The type could depend on an infer-only parameter for which a value
  // has not yet been determined. If that's the case, the parameter is left
  // unbound and parameter binding cannot resolve the co-dependency between the
  // infer-only parameter and the incoming expression.
  if (paramList.hasInferredParams() || operand.isKeyword()) {
    mlir::AttrTypeWalker walker;
    walker.addWalk([](UnboundAttr) { return WalkResult::interrupt(); });
    if (walker.walk(nextType).wasInterrupted())
      return {};
  }

  return nextType;
}

/// Emit the PValue result for a single parameter binding.
static TypedAttr emitSingleParamBinding(const Operand &operand,
                                        ExprEmitter &emitter,
                                        ExprContext context,
                                        ASTType expectedType) {
  MLIRContext *ctx = emitter.getContext();

  // _, *_, and **_ in parameter expressions are magically treated as special
  // syntax for unbound values, which get a special representation in a
  // parameter list. They are not general expressions, so don't emit them as
  // such.

  // Handle `**_` or `_` syntax, based on the operand passing kind.
  if (operand.expr->kind == ExprNode::kDiscardLiteral) {
    TypedAttr value = UnboundAttr::get(UnresolvedType::get(ctx));
    if (operand.isUnpackedKeyword())
      value = UnpackedAttr::get(value, /*kwOnly=*/true, value.getType());
    return value;
  }

  // Handle unpacked variadics in the form of `*x`.
  if (operand.expr->kind == ExprNode::kUnpack) {
    auto *unpackExpr = cast<UnaryOpNode>(operand.expr);

    // Handle the *_ syntax, which is parsed as an Unpack(DiscardLiteral)
    // specially.
    if (unpackExpr->subExpr->kind == ExprNode::kDiscardLiteral) {
      auto unresolved = UnresolvedType::get(ctx);
      return UnpackedAttr::get(UnboundAttr::get(unresolved),
                               /*kwOnly=*/false, unresolved);
    }

    // Only variadic-typed parameters can be unpacked. Emit the subexpression.
    // `expectedType` isn't needed here, because a variadic value would have
    // already had its element type disambiguated.
    PValue value = emitter.emitExprPValue(unpackExpr->subExpr, context);
    if (!value)
      return {};
    if (!isa<VariadicType>(value.getType())) {
      emitter.emitError(unpackExpr->getLoc(), "only variadics can be unpacked")
          << unpackExpr->getRange();
      return {};
    }
    return UnpackedAttr::get(value, /*kwOnly=*/false,
                             value.getType().getVariadicElementType());
  }

  // Emit the expression as a parameter, coercing it to the expected type, if
  // one is provided.
  return emitter.emitExprPValue(operand.expr, context, expectedType);
}

/// Given PValue operands and a list of in-progress bindings to parameter lists,
/// emit the operands and add them to a binding set. If any operand fails be
/// emitted as a PValue, the function fails.
static LogicalResult
bindParameterOperands(MutableArrayRef<InProgressBindings> bindables,
                      ArrayRef<Operand> operands, ParamBindings &paramBindings,
                      ExprEmitter &emitter, ExprContext context) {
  assert(!bindables.empty() && "expected something to bind");

  // Process each operand.
  for (const Operand &operand : operands) {
    // Query each bindable for the type of the next parameter. If there are
    // multiple bindables, they all have to agree.
    SmallVector<ASTType> expectedTypes;
    for (InProgressBindings &bindable : bindables) {
      expectedTypes.push_back(
          bindable.getNextParamType(operand, paramBindings, emitter));
    }
    ASTType expectedType;
    if (!expectedTypes.empty()) {
      expectedType = expectedTypes.front();
      // If any of the other types disagree, give up.
      if (llvm::any_of(llvm::drop_begin(expectedTypes), [&](ASTType type) {
            return !type.isEqualCanon(expectedType);
          }))
        expectedType = ASTType();
    }

    TypedAttr value =
        emitSingleParamBinding(operand, emitter, context, expectedType);
    if (!value)
      return failure(); // error already emitted
    if (operand.isKeywordOrUnpackedKeyword())
      paramBindings.add(operand.expr, value, operand.name);
    else
      paramBindings.add(operand.expr, value);
  }

  return success();
}

/// Return a ParamBindings set for a list of PValue operands. If any operand
/// fails be emitted as a PValue, the function returns null.
static std::optional<ParamBindings> getBindingsForParameterOperands(
    ArrayRef<Operand> operands, ArrayRef<Type> paramTypes,
    PogListAttr paramList, ExprEmitter &emitter, ExprContext context) {
  InProgressBindings bindable(paramTypes, paramList);
  // Start with an empty binding set.
  ParamBindings paramBindings(emitter.getDeclScope());
  if (failed(bindParameterOperands(bindable, operands, paramBindings, emitter,
                                   context)))
    return std::nullopt;
  return std::move(paramBindings);
}

/// Given a value of type type, substitute parameters into the type, producing
/// a more concrete type.  This syntax is `SomeType[1, 4, Int]`.
static PValue substituteParametersIntoUserDefinedType(
    PValue typeValue, ArrayRef<Operand> operands, SMLoc loc, SMLoc lhsLoc,
    SMLoc rhsLoc, ExprEmitter &emitter) {
  auto metaType = cast<StructMetaType>(typeValue.getType());
  ASTDecl *typeDecl = ASTType(typeValue).getDecl(emitter.shared);
  auto structOp = dyn_cast_or_null<StructDeclOp>(typeDecl);
  if (!structOp) {
    auto diag = emitter.emitError(loc);
    if (isa<FileModuleOp, PackageOp>(typeDecl)) {
      // Emit helpful error message when user tried to subscript a module.
      emitModuleCallSubscriptDiag(diag, metaType, "subscript", loc,
                                  emitter.shared);
    } else {
      diag << "unknown parameterized type " << ASTType(typeValue)
           << SourceRange{loc, rhsLoc};
    }
    return {};
  }

  // Notify the listener on the parameter binding.
  emitter.shared.notifyListenerOnParameterBinding(typeDecl, rhsLoc, operands);

  // Build up a ParamBindings set to validate and check the bindings.
  TypeSignatureType sig = metaType.getSignature();
  std::optional<ParamBindings> paramBindings = getBindingsForParameterOperands(
      operands, sig.getParamTypes(), sig.getParamListAttrs(), emitter,
      EC_TypeParamValue);
  if (!paramBindings)
    return {};

  // Check the bindings.
  // FIXME: The error messages are bad for partial binding, because the
  // diagnostic emitter points to the original struct definition.
  ParameterExprArrayAttr bindingValuesAttr =
      paramBindings->verifyBindings(structOp, sig, loc, /*partial=*/true);
  if (!bindingValuesAttr)
    return {};

  // Ok, we succeeded at reparameterizing the type.
  LIT::StructType boundType =
      metaType.getType().bindUnbound(bindingValuesAttr.getValue());
  return TypeParamAttr::get(boundType, StructMetaType::get(boundType));
}

/// Bind parameter operands to a callable parameter.
static PValue bindToIndirectCall(PValue callable, FnTypeGeneratorType sig,
                                 ArrayRef<Operand> operands,
                                 ExprEmitter &emitter,
                                 const SourceRange &range) {
  // Build up a ParamBindings set to validate and check the bindings.
  std::optional<ParamBindings> paramBindings = getBindingsForParameterOperands(
      operands, sig.getInputParamTypes(), sig.getParamListAttrs(), emitter,
      EC_CallParamValue);
  if (!paramBindings)
    return {};

  ParameterExprArrayAttr newBindings = paramBindings->verifyBindings(
      sig, "parametric callable", range.getStart());
  if (!newBindings)
    return {};

  return BindParamsAttr::get(callable.get(), newBindings);
}

/// When subscripting a callable with a bound symbol (i.e. a direct method call
/// or call to a method), apply parameter bindings to it.
// TODO(MOCO-1106): Removed `static` on this so variadic thunks could use it,
// let's refactor emitGetterSetterAccess so it can be used by thunks instead.
LogicalResult bindParamValuesToDirectCall(OverloadSet &overloadSet,
                                          ArrayRef<Operand> operands,
                                          ExprEmitter &emitter) {
  // Build a list of all the things we can bind. Remember the number of
  // positional parmaeters that have already been bound.
  SmallVector<InProgressBindings> bindables;
  unsigned numPosBindings =
      overloadSet.paramBindings.getParameters().getNumPositional();
  for (ASTDecl *fnDecl : overloadSet.fnDecls) {
    FnTypeGeneratorType sig = cast<FnOp>(fnDecl).getFullSignature();
    bindables.emplace_back(sig.getInputParamTypes(), sig.getParamListAttrs(),
                           numPosBindings);
  }

  // Now attempt to bind the new operands.
  return bindParameterOperands(bindables, operands, overloadSet.paramBindings,
                               emitter, EC_CallParamValue);
}

/// Given a base value, emit access to a base value element using either a
/// reference-producing-method or getter-and-setter-methods using the provided
/// operands.
///
/// This prefers the reference method if present.  If not, and if a getter is
/// present on the base type but a setter is not, this method immediately emits
/// a getter call.
///
/// Otherwise, it returns a SubscriptDLValue for later materializing calls to
/// the getter or setter as appropriate. When doing this it  takes ownership of
/// the operands because it might move them to a SubscriptDLValue, if emitted.
AnyValue emitGetterSetterAccess(const ExprNode *node, ASTExprAnd<CValue> base,
                                ArrayRef<Operand> exprOperands, ValueDest &dest,
                                ExprEmitter &emitter) {
  ASTType baseType = base.ir.getRValueType();

  // This is either a SubscriptNode for x[i,j] or a AttributeRefNode for x.name.
  bool isSubscript = isa<SubscriptNode>(node);
  CallSyntax syntax =
      isSubscript ? CallSyntax::kSubscript : CallSyntax::kAttribute;

  auto lookupError = [&] {
    auto diagType = baseType;
    // Complain about "SomeType" in 'SomeType.foo' not 'AnyStruct[SomeType]'.
    if (auto anyStruct = dyn_cast<StructMetaType>(diagType))
      diagType = anyStruct.getType();
    else if (auto anyTrait = dyn_cast<AnyTraitType>(diagType))
      diagType = anyTrait.getTraitType();

    auto diag = emitter.emitError(node->getLoc())
                << diagType << base.expr->getRange();

    if (isSubscript)
      diag << " is not subscriptable, it does not implement the "
              "`__getitem__`/`__setitem__` methods";
    else
      diag << " value has no attribute '"
           << cast<StringLiteralNode>(exprOperands[0].expr)->getValue() << "'";
  };

  // Look up the getter and setter candidate list on the self type.
  StringRef getterName = isSubscript ? "__getitem__" : "__getattr__";
  OverloadSet getterSet = OverloadSet::lookup(emitter.getDeclScope(), baseType,
                                              getterName, node, syntax);
  if (getterSet.isErroneous())
    return {}; // Ignore already emitted errors.

  StringRef setterName = isSubscript ? "__setitem__" : "__setattr__";
  OverloadSet setterSet = OverloadSet::lookup(emitter.getDeclScope(), baseType,
                                              setterName, node, syntax);
  if (setterSet.isErroneous())
    return {}; // Ignore already emitted errors.

  // If there is no getter or setter, then we need to fail.
  if (!setterSet && !getterSet) {
    lookupError();
    return {};
  }

  // Otherwise we'll be calling the getter and/or setter.  Let's emit any index
  // operands and determine whether they are arguments or parameters (e.g. for
  // indexing into Tuple with parameter for the index).
  CallOperands operands;
  operands.addSelf(base);

  // We look at one of the sets so we can detect whether we're emitting the
  // operands as parameters or dynamic values.
  bool isGetterPresent = (bool)getterSet;
  OverloadSet *nonemptySet = isGetterPresent ? &getterSet : &setterSet;
  assert(*nonemptySet && "at least one of the two should be nonempty");

  // The exprOperands provided may be binding either to parameters or to
  // arguments, and may even be mixed in the theoretical future.  For now, we
  // keep things simple and just decide to bind all of the expressions to
  // parameters if no candidates have an argument (other than the set value if
  // this is a setter list).
  bool shouldBindParameters = true;
  for (ASTDecl *elt : nonemptySet->fnDecls) {
    // TODO: This is really naive: it doesn't account for default arguments,
    // variadic, byref_result, etc etc etc.
    if (cast<FnOp>(*elt).getFuncTypeGenerator().getArguments().size() !=
        size_t(/*newValue*/ !isGetterPresent) + /*self*/ 1) {
      shouldBindParameters = false;
      break;
    }
  }

  // If we're binding these indices to parameters, do so and leave the
  // arguments lists empty.
  if (shouldBindParameters) {
    // Start the parameter set with the parameters from the base type of the
    // method we're invoking, so we set additional parameters.
    //
    // FIXME: This is incorrect!  The overload set can contain members of
    // types that baseType implicitly converts to, and they will take
    // different parameters that should be inferred from the actual arguments
    // passed to the call.  We need ParameterizedType() to represent unbound
    // self parameters but bound function parameters.
    //
    // FIXME2: What about the other set?  This seems like it only handles
    // getitem.
    nonemptySet->paramBindings = ParamBindings::getForDeclaredType(
        emitter.getDeclScope(), baseType, node);
    if (failed(
            bindParamValuesToDirectCall(*nonemptySet, exprOperands, emitter)))
      return {};
  } else {
    // Otherwise we're passing these exprOperands as normal dynamic arguments.
    for (const Operand &operand : exprOperands) {
      ExprNode *expr = operand.expr;
      AnyValue exprVal = emitter.emitExpr(expr, EC_Subscript);
      if (!exprVal)
        return {};
      if (operand.isKeywordOrUnpackedKeyword()) {
        operands.add(operand.name, ASTExprAnd<AnyValue>{exprVal, expr});
      } else {
        operands.add({exprVal, expr});
      }
    }
  }

  // If we /just/ have a getter, emit this as a call to the getter, allowing
  // us to get nice tuned diagnostics.
  if (!setterSet)
    return getterSet.emitCall(std::move(operands), dest, emitter);

  // Okay, we definitely have a setter, and we might have a getter.  The problem
  // is that we don't know in which context this expression will be used - it
  // could be loaded from, stored to, or both (with a mut argument), and it
  // might even have computed contextual parameters.

  // If we have a getter, resolve it and get the element type from it.
  ASTType elementType;
  PValue getter;
  if (getterSet) {
    getter =
        getterSet.filterOverloadSet(operands,
                                    /*emitDiagnosticOnFailure*/ true, emitter);
    if (!getter) // Error already emitted.
      return {};
    // ElementType is the result of the getter, processing by-ref results and
    // ignoring the variant for raising functions.
    elementType = getter.getType().getSignatureUserResultType();

    // Also look through ref results.
    if (cast<FnTypeGeneratorType>(getter.getType()).isRefResult())
      elementType = cast<RefType>(elementType).getElementType();
  }

  // We need to figure out which setter to use, but can't just filter the set
  // unless we know the element type from the getter.  If not, do something
  // grotty to figure it out.
  if (!elementType) {
    // Cannot support overloaded setter with no getters.
    if (setterSet.fnDecls.size() != 1) {
      auto diag = emitter.emitError(node->getLoc())
                  << baseType << " has overloaded " << setterName
                  << " implementations, which isn't supported"
                  << node->getRange();
      for (auto candidate : setterSet.fnDecls)
        diag.attachNote(candidate->getLoc()) << "candidate declared here";
      return {};
    }

    // TODO: This won't handle parameterized setters right, inferring the
    // parameter types.  We should use something like
    // `filterOverloadSetForValueType` or use a dummy value to filter the
    // overload set.
    // Hard code the parameter bindings for 'self' since we aren't using type
    // inference properly.
    setterSet.paramBindings = ParamBindings::getForDeclaredType(
        emitter.getDeclScope(), baseType, node);
    auto directSymbolAttr = setterSet.getBoundConstantAttr();
    if (!directSymbolAttr) {
      lookupError();
      return {}; // Getter invalid.
    }
    auto sigType = cast<FnTypeGeneratorType>(directSymbolAttr.getType());
    // Check basic sanity.
    size_t setValueIdx = operands.getNumPositional();
    if (sigType.getNumArguments() <= setValueIdx) {
      auto diag = emitter.emitError(node->getLoc())
                  << setterName << " has too few arguments";
      diag.attachNote(setterSet.fnDecls[0]->getLoc())
          << setterName << " declared here";
      return {};
    }
    elementType = sigType.getArguments()[setValueIdx];
    auto setValueConvention = sigType.getArgConvention(setValueIdx);
    if (setValueConvention != ArgConvention::ReadReg)
      elementType = elementType.getReferenceElementType();
  }

  // Ok, now that we know the elementType, we can look up any setter that we
  // need to use.

  // If the accessors are defined with the new value as a keyword-only
  // argument (eg because the indices are variadic), then we need to pass as a
  // keyword, otherwise we can pass as a positional argument.  This is a bit
  // awkward for overload resolution because we don't know what name each
  // overload might use.  It seems reasonable to require that all overloads of
  // __setitem__ use the same name for their value argument, so we just sniff
  // at the first entry of the set to see what it uses and assume the rest use
  // the same name.
  auto firstFnSig =
      cast<FnOp>(*setterSet.fnDecls.front()).getFuncTypeGenerator();

  // Find the last user declared argument.
  auto argNo = firstFnSig.getNumArguments();
  do {
    --argNo;
    // Can't use the self argument as the new value.
    if (argNo == 0) {
      auto diag = emitter.emitError(setterSet.fnDecls.front()->getLoc())
                  << setterName
                  << " must take at least one argument for the value to set"
                  << node->getRange();
      diag.attachNote(node->getLoc())
          << "used in an expression here" << node->getRange();
      return {};
    }
    // Ignore the byref return and error arguments.
  } while (isResultSlot(firstFnSig.getArgConvention(argNo)));

  StringAttr setterValueName = firstFnSig.getArgName(argNo);
  if (operands.findKwArg(setterValueName)) {
    auto diag = emitter.emitError(node->getLoc())
                << "keyword argument " << setterValueName
                << " may not be specified in the index list, it is needed "
                   "for the new value"
                << node->getRange();
    return {};
  }

  // Otherwise, this expression may be used as an LValue so form it.
  DLValue result(RCRef<SubscriptDLValue>::create(
      getter, setterValueName, std::move(operands), elementType, node));
  return emitter.emitResult(result, node, dest);
}

/// Emit a qualified attribute reference to MLIR.  On error, emit an error and
/// return a null value.
AnyValue AttributeRefNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  auto &shared = emitter.shared;

  // In-order to allow parameter expressions which technically include a runtime
  // reference, i.e `x.static_field` we allow some values which would otherwise
  // produce a value in a parameter context to still propagate up.
  CValue baseVal = emitter.emitExprCValue(base, EC_AttributeRefBase);
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
  ASTDecl *typeDecl = baseRVType.getDecl(shared);
  if (!typeDecl) {
    // If the attribute spelling is empty, we couldn't find a name to look up.
    // This was already diagnosed during initial parsing, so we can just bail
    // here.
    if (spelling.empty())
      return {};

    // If there is no decl, the type is an MLIR type.
    Type baseMLIRType = baseRVType.mlirType;

    // Handle __mlir_op.`xxx` references, lazily synthesizing values when
    // they are referenced.
    if (isa<MagicMLIRAttrType>(baseMLIRType)) {
      PValue result = synthesizeMLIRAttrFromString(spelling, getLoc(), shared);
      return emitter.emitResult(result, this, dest);
    }
    if (isa<MagicMLIRAttrType, MagicMLIRDeferredAttrType>(baseMLIRType)) {
      /// `__mlir_deferred_attr` always behaves like `__mlir_attr` when used
      /// with backticks. Don't want to strictly enforce that, but user should
      /// be aware that use of `__mlir_attr` is preferred
      emitMLIRDeferedAttrToMLIRAttrWarning(getLoc(), emitter);
      PValue result = synthesizeMLIRAttrFromString(spelling, getLoc(), shared);
      return emitter.emitResult(result, this, dest);
    }
    if (isa<MagicMLIROpType>(baseMLIRType)) {
      PValue result = synthesizeMLIROpFromString(spelling, emitter);
      return emitter.emitResult(result, this, dest);
    }
    if (isa<MagicMLIRTypeType>(baseMLIRType)) {
      ASTType result = parseMLIRType(spelling, this, shared);
      return emitter.emitResult(result, this, dest);
    }

    emitter.emitError(getLoc(), "MLIR type ")
        << baseRVType << " has no attributes" << base->getRange();
    return {};
  }

  // Notify the listener of a member lookup.
  shared.notifyListenerOnMemberLookup(*typeDecl, getIdentifierLoc());

  // If the attribute spelling is empty, we couldn't find a name to look up.
  // This was already diagnosed during initial parsing, so we can just bail
  // here.
  if (spelling.empty())
    return {};

  // Handle module or package references.
  if (isa<PackageOp, FileModuleOp>(*typeDecl)) {
    // Look up the unqualified identifier in the right scope.
    //
    // declRef is allocated persistently because when it refers to a function
    // it gets captured in the overloadSet returned by declRef->emitIR(), and,
    // thus, cannot be on the stack.
    DeclRefNode *declRef = shared.allocPersistent<DeclRefNode>(spelling);
    if (emitter.builder) {
      ExprEmitter moduleEmitter(*typeDecl, *emitter.builder,
                                emitter.varDeclCursor);

      return declRef->emitIR(dest, moduleEmitter);
    }

    ExprEmitter moduleEmitter(*typeDecl, emitter.paramContext);
    return declRef->emitIR(dest, moduleEmitter);
  }

  if (!isa<StructDeclOp, TraitDeclOp>(*typeDecl) &&
      !isa<TraitType>(typeDecl->getIfTypeValue())) {
    emitter.emitError(getLoc(), "cannot access attribute in type ")
        << baseVal.getType() << base->getRange();
    return {};
  }

  // Find the member being accessed.
  LookupResult lookup =
      shared.lookupAndResolveDecl(spelling, getLoc(), *typeDecl,
                                  /*searchParentScopes=*/false);
  ArrayRef<ASTDecl *> memberDecls = lookup.getIfSuccess();

  // If the struct has no static member of the required name, try to look for
  // dynamic lookup attribute methods (__getattr__ etc) on the type.
  if (memberDecls.empty()) {
    // Convert the attribute name to a StringLiteralNode since that's how the
    // operand will be emitted.  We need to allocate it persistently to enable
    // DLValues to capture it.
    std::string quotedName = '"' + spelling.str() + '"';
    StringRef quotedNameRef = shared.getPersistentCopy(StringRef(quotedName));
    // StringLiteral wants ArrayRef<StringRef>.
    auto strRefs = shared.getPersistentCopy(ArrayRef(quotedNameRef));
    auto *strLiteral = shared.allocPersistent<StringLiteralNode>(strRefs);
    return emitGetterSetterAccess(
        this, {baseVal, base},
        Operand(strLiteral, getLoc(), Operand::kPositional), dest, emitter);
  }
  shared.notifyListenerOnRef(memberDecls, spelling, this);

  // Handle method references, which might be overloaded.
  if (isa<FnOp>(memberDecls[0])) {
    // Build an overload set of all matching function declarations.

    // TODO(ParameterizedType): This representation is subtly wrong.  We should
    // be inferring Self parameters from the expression later rather than
    // installing "getForDeclaredType", because this won't work correctly with
    // non-materializable types that need an implicit conversion.
    //
    // We currently need to bind the Self parameters here so that subsequent
    // parameters are bound correctly.  Consider something like:
    //     foo.dyn_cast[Int]()
    // If typeof(foo) has parameters A and B, we need to form a parameter list
    // of `[A, B, Int]`.  If we had ParameterizedType then we could model this
    // correctly as have an unspecified first set of bindings for the type,
    // and the Int binding could go in a subsequent parameter list.
    auto bindings = ParamBindings::getForDeclaredType(emitter.getDeclScope(),
                                                      baseRVType, this);
    auto result =
        OverloadSetUValue::create(spelling, memberDecls, std::move(bindings),
                                  this, CallSyntax::kDirectCall);

    // If the callee is a static method, we can directly reference it
    // without binding a self parameter.  If this is an instance method, we
    // bind the base value and the symbol together into a callable.
    if (!hasTypeBase) {
      result->baseValue = {baseVal, base};
      result->syntax = CallSyntax::kMethodCall;
    }
    return emitter.emitResult(result, this, dest);
  }

  assert(memberDecls.size() == 1 && "only methods may be overloaded");
  ASTDecl &memberDecl = *memberDecls[0];

  // Parameters form a meta-value.
  if (auto aliasDeclOpParam = dyn_cast<AliasDeclOp>(memberDecl)) {
    if (isa<StructDeclOp>(*typeDecl)) {
      PValue result = resolveAliasReference(aliasDeclOpParam, spelling,
                                            baseRVType.getParamBindings(),
                                            getLoc(), shared);
      return emitter.emitResult(result.get(), this, dest);
    }

    // If we get here, we're accessing an alias in a trait.
    assert((isa<TraitDeclOp>(*typeDecl) ||
            isa<TraitType>(typeDecl->getIfTypeValue())) &&
           "Alias's parent should be struct or trait");
    PValue basePValue = baseVal.getIfPValue();
    if (!basePValue) {
      emitter.emitError(getLoc(), "can't access '")
          << spelling << "' in non-parameter " << baseRVType << getRange();
      return {};
    }
    // Make a get_vtable_entry call to extract the value out of the trait
    // value's vtable.
    // See
    // https://www.notion.so/modularai/verifyConformance-Arcana-13e1044d37bb80e88cb5c285a232784e?pvs=4#13e1044d37bb80bf8b42f3953af880f8
    // for why and where else we do this.
    auto vtableEntryName =
        StringAttr::get(spelling, StringType::get(emitter.getContext()));
    auto vtableEntryResult = ParamOperatorAttr::get(
        POC::GetVTableEntry, {basePValue, vtableEntryName},
        aliasDeclOpParam.getType());

    return emitter.emitResult(vtableEntryResult, this, dest);
  }

  // If the field is a variable, emit a reference to it.
  if (auto fieldOp = dyn_cast<StructFieldOp>(memberDecl)) {
    if (hasTypeBase || isa<StructMetaType>(baseRVType)) {
      emitter.emitError(getLoc(), "cannot access instance field '")
          << spelling << "' without an instance of " << baseRVType
          << getRange();
      return {};
    }

    // We know that baseVal is a CValue, so handle all the cases.

    // If the base is a DLValue, we need to emit this as a projected DLValue.
    // This allows to emit a get and/or set as needed.
    if (DLValue baseLV = baseVal.getIfDLValue()) {
      // The base is a known StructType because we got the ASTDecl from it.
      ASTType elementType =
          fieldOp.getReboundType(cast<LIT::StructType>(baseRVType.mlirType));
      DLValue result(RCRef<StoredAttributeRefDLValue>::create(
          ASTExprAnd<DLValue>{baseLV, base}, fieldOp, elementType, this));
      return emitter.emitResult(result, this, dest);
    }

    // Otherwise, emit the stored field reference.
    return emitStoredFieldRef({baseVal, base}, fieldOp, this, dest, emitter);
  }

  // This parameter will refer to the generic parameter on the base type decl,
  // e.g the base struct. We need to substitute it for the "real" parameter used
  // to construct this specific type, not the shared type on the struct.
  if (auto parameter = memberDecl.getIfIRValue().getIfPValue()) {
    auto paramRef = cast<ParamDeclRefAttr>(parameter.get());
    if (auto baseDecl = dyn_cast<StructMetaType>(baseRVType.getMetaType())) {
      for (auto [name, value] :
           llvm::zip(cast<StructDeclOp>(typeDecl).getParams(),
                     baseDecl.getParamValues())) {
        // If this binding is for this parameter propagate the bound
        // parameter.
        if (name.getName() == paramRef.getName())
          return emitter.emitResult(value, this, dest);
      }
    }
  }

  // Reference to some non-function/struct member of the type.
  emitter.emitError(getLoc(), "reference to unknown member '")
      << spelling << "'" << getRange();
  return {};
}

/// Given a call to an UnboundMLIROperator, generate an MLIR operation with
/// the operands as SSA values.
static AnyValue emitMLIROperatorCall(const CallNode &call,
                                     UnboundMLIROperationAttr unboundOp,
                                     ValueDest &dest, ExprEmitter &emitter) {
  auto *context = emitter.getContext();
  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(&call);

  // Emit all the arguments so we can encode them as SSA values.
  SmallVector<Value> opOperands;
  for (const Operand &argument : call.operands) {
    if (!argument.isPositional()) {
      emitter.emitError(argument.getLoc(),
                        "MLIR operators only support positional arguments");
      return {};
    }
    Value value = emitter.emitExprSRValue(argument.expr, EC_MLIRMagic);
    if (!value)
      return {};
    opOperands.push_back(value);
  }

  StringAttr opName = unboundOp.getName();
  bool isDeferredOp = false;
  NamedAttrList attrs;
  if (llvm::any_of(unboundOp.getAttrs(), [](NamedAttribute attr) {
        if (auto typedAttr = dyn_cast<TypedAttr>(attr.getValue()))
          return isa<DeferredType>(typedAttr.getType());
        return false;
      })) {
    opName = StringAttr::get(context, DeferredOp::getOperationName());
    isDeferredOp = true;
  }

  // Set up the OperationState for the thing we're building.
  OperationState state(call.getLocation(emitter), opName);
  state.addOperands(opOperands);

  // Process the attributes and figure out the result type if specified.
  bool hadTypeSpec = false;
  std::optional<Attribute> propsAttr = std::nullopt;
  for (auto &attr : unboundOp.getAttrs()) {
    if (attr.getName() == "_type") {
      // We expect either a single type, `None`, or a `Tuple` of types.
      if (isa<NoneAttr>(attr.getValue())) {
        hadTypeSpec = true;
        continue;
      }

      auto value = dyn_cast<TypedAttr>(attr.getValue());
      if (!value) {
        emitter.emitError(call.getLoc(), "unknown _type value");
        return {};
      }

      auto pushTypeToState = [&](TypedAttr type,
                                 const Twine &message) -> LogicalResult {
        if (!LIT::isTypeExpr(type)) {
          emitter.emitError(call.getLoc(), message);
          return failure();
        }
        state.types.push_back(ASTType(type));
        return success();
      };
      if (auto valueMetaType = dyn_cast<StructMetaType>(value.getType())) {
        ASTType tupleType = emitter.shared.getBuiltinTupleType(
            emitter.declScope, call.getLoc());
        // If the _type field is a Tuple of types, then the operation
        // returns multiple results, with types specified in the list.  We
        // need to take apart the Tuple value to get the types from inside it.
        if (valueMetaType.getSymbol() ==
            cast<StructMetaType>(tupleType.getMetaType()).getSymbol()) {
          // Dig out the types from the tuple.  Tuple literals must always
          // have this particular shape.
          auto tca = cast<TypeParamAttr>(value);
          auto drt = cast<LIT::StructType>(tca.getMlirType());
          ArrayRef<TypedAttr> paramValues = drt.getParamValues();
          assert(paramValues.size() == 1 &&
                 "_types tuple ParamValues must be size 1");
          auto variadic = cast<VariadicAttr>(paramValues[0]);
          for (TypedAttr type : variadic.getValues()) {
            if (pushTypeToState(type, "value in _type tuple is not a type")
                    .failed())
              return {};
          }
          hadTypeSpec = true;
          continue;
        }
      }
      if (pushTypeToState(value, "_type value is not a type").failed())
        return {};
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
    if (attr.getName() == "_properties") {
      propsAttr = attr.getValue();
      continue;
    }
    if (isDeferredOp) {
      // For deferred operation, fill the dictionary with all attributes
      // regardless if attribute itself is deferred or not. That allows to have
      // any number of any attributes for deferred operation.
      attrs.set(attr.getName(), attr.getValue());
      continue;
    }
    state.addAttributes(attr);
  }

  if (isDeferredOp) {
    auto deferredName =
        mlir::OperationName(DeferredOp::getOperationName(), context);
    state.addAttribute(DeferredOp::getOpNameAttrName(deferredName),
                       unboundOp.getName());
    state.addAttribute(DeferredOp::getOpAttrsAttrName(deferredName),
                       attrs.getDictionary(context));
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

    SmallVector<char> propStorage(opNameInfo->getOpPropertyByteSize());
    auto properties = mlir::OpaqueProperties(propStorage.data());
    auto attributes = state.attributes.getDictionary(context);

    // Initialize the properties storage
    if (!propStorage.empty() && !state.attributes.empty()) {
      auto emitError = [&]() {
        return mlir::emitError(state.location)
               << " failed properties conversion while building "
               << state.name.getStringRef() << " with `" << attributes << "`: ";
      };
      if (failed(opNameInfo->setOpPropertiesFromAttribute(
              state.name, properties, attributes, emitError)))
        return failure();
    }

    if (failed(inferTypesItf->inferReturnTypes(
            context, state.location, state.operands, attributes, properties,
            state.regions, state.types)))
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
      emitter.emitError(
          call.getLoc(),
          "cannot use operations with multiple inferred results (yet) ")
          << unboundOp.getName() << call.getRange();
      return {};
    }
  }

  for (Type type : state.types) {
    if (!ASTType(type).isRegisterPassable(call.getLoc(), emitter.shared)) {
      emitter.emitError(call.getLoc())
          << ASTType(type)
          << " cannot be returned directly from __mlir_op as it is not a "
             "'@register_passable' types";
      return {};
    }
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

  // Check if the attributes specified are all inherent attributes.
  DenseSet<StringAttr> inherentAttrs;
  inherentAttrs.insert_range(resultOp->getName().getAttributeNames());
  for (NamedAttribute &attr : state.attributes) {
    if (!inherentAttrs.contains(attr.getName())) {
      emitter.emitError(call.getLoc(), "attribute ")
          << attr.getName() << " is not an inherent attribute of "
          << unboundOp.getName();
      return {};
    }
  }

  // Set the properties if needed. We do this here, because errors result in a
  // crash in the op builder if we simply set state.propertiesAttr.
  if (propsAttr) {
    if (failed(resultOp->setPropertiesFromAttribute(
            *propsAttr, [&]() { return resultOp->emitError(); }))) {
      emitter.emitError(call.getLoc(), "cannot set property");
      return {};
    }
  }

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
    emitter.emitError(call.getLoc(), "MLIR verification error: ")
        << errorMessage;
    return {};
  }

  // Helper to emit an SValue or PValue to the destination.
  auto emitResult = [&](CValue value) -> AnyValue {
    return emitter.emitResult(value, &call, dest);
  };

  // If we succeeded and have no types, then install a None value.
  if (resultOp->getNumResults() == 0)
    return emitResult(PValue(emitter.shared.getNoneAttr()));

  if (resultOp->getNumResults() == 1) {
    OpResult res = resultOp->getResult(0);
    ASTType resType = res.getType();

    // Check to see if we can fold this operation.  This enables use of
    // __mlir_op to produce meta-values without forcing them into the dynamic
    // value domain.
    SmallVector<Attribute, 4> constOperands(resultOp->getNumOperands());
    for (unsigned i = 0, e = constOperands.size(); i != e; ++i)
      matchPattern(resultOp->getOperand(i),
                   mlir::m_Constant(&constOperands[i]));
    SmallVector<OpFoldResult, 4> foldResults;
    if (succeeded(resultOp->fold(constOperands, foldResults)) &&
        foldResults.size() == 1) {
      auto folded = PointerUnion<Attribute, Value>(foldResults[0]);
      CValue result;
      // If the result was some other value that already exists, use it.
      if (auto val = dyn_cast<Value>(folded)) {
        result = SRValue(val);
      } else {
        // If it is a constant, make an PValue result.
        if (auto attr = dyn_cast<TypedAttr>(cast<Attribute>(folded)))
          result = PValue(attr);
        // If it isn't a TypedAttr, then we cannot fold to it.
      }

      if (result) {
        if (result.getRValueType().isEqualCanon(resType)) {
          resultOp->erase();
          return emitResult(result);
        }
        emitter.emitError(call.getLoc())
            << unboundOp.getName() << " operation folded to result type "
            << result.getRValueType() << " but we expected it to be " << resType
            << call.getRange();
        return {};
      }
    }

    // If folding failed, return the operation's result normally.
    return emitResult(SRValue(res));
  }

  // Pack results into a tuple and return it.
  auto tupleType =
      emitter.shared.getBuiltinTupleType(emitter.declScope, call.getLoc());
  if (tupleType.isTypeCheckErrorType())
    return {};

  // Construct the Tuple type without parameters so we infer them.
  tupleType = tupleType.getWithoutParameters(emitter.shared);

  CallOperands operands;
  for (OpResult opResult : resultOp->getResults())
    operands.add({SRValue(opResult), &call});
  return emitter.emitConstructorCall(tupleType, std::move(operands), &call,
                                     CallSyntax::kTypeCall, dest);
}

AnyValue CallNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  AnyValue calleeVal = emitter.emitExpr(callee, EC_CallCalleeValue);
  if (!calleeVal)
    return {};

  // If this is the invocation of an unbound MLIR operator, bind it into an
  // actual operator!
  if (auto mValue = calleeVal.getIfPValue()) {
    if (auto unboundOp = dyn_cast<UnboundMLIROperationAttr>(mValue.get()))
      return emitMLIROperatorCall(*this, unboundOp, dest, emitter);
  }

  /// Emit all the operands that we'll need.
  CallOperands operandsList;
  for (const Operand &operand : operands) {
    if (operand.isUnpacked()) {
      auto diag = emitter.emitError(operand.getLoc());
      ExprNode *packedExpr = dyn_cast<UnaryOpNode>(operand.expr)->subExpr;
      if (packedExpr && packedExpr->kind == ExprNode::kDiscardLiteral)
        diag << "unbound packs not supported yet in runtime arguments";
      else
        diag << "unpacked arguments are not supported yet";
      return {};
    }

    ASTExprAnd<AnyValue> exprAndVal = {
        emitter.emitExpr(operand.expr, EC_CallArgValue), operand.expr};
    if (!exprAndVal)
      return {};
    if (operand.isPositional()) {
      operandsList.add(std::move(exprAndVal));
    } else {
      assert(operand.isKeyword());
      operandsList.add(operand.name, std::move(exprAndVal));
    }
  }

  // If the callee is a type value (as in `T()` or `T[123]()`), then this is an
  // invocation of the initializer for the type, or a call to a custom op.
  if (ASTType calledType = calleeVal.getIfTypeValue()) {
    auto *calleeDecl = calledType.getDecl(emitter.shared);
    if (!calleeDecl) {
      emitter.emitError(getLoc(), "cannot use initializer syntax on MLIR type ")
          << calledType << callee->getRange();
      return {};
    }

    // Check to see if we can invoke an __init__ method to convert it.
    return emitter.emitConstructorCall(calledType, std::move(operandsList),
                                       this, CallSyntax::kTypeCall, dest);
  }

  // If this is an overloaded operand, resolve it and call the result.
  if (auto overloads = calleeVal.getIfOverloadSet()) {
    emitter.shared.notifyListenerOnCall(overloads->fnDecls, rparenLoc,
                                        overloads->syntax, operandsList);
    overloads->expr = this;
    return overloads->emitCall(std::move(operandsList), dest, emitter);
  }

  // Otherwise, we must have a concrete RValue, emit an indirect call.
  if (auto crVal = calleeVal.getIfCValue())
    return emitter.emitIndirectCall(crVal, std::move(operandsList), dest,
                                    CallSyntax::kIndirectCall, this);

  emitter.emitError(getLoc(), "cannot call this unresolved expression");
  return {};
}

AnyValue SliceNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  auto getOperand = [&](const ExprNode *expr) -> ASTExprAnd<AnyValue> {
    if (expr)
      return {emitter.emitExpr(expr, ExprContext::EC_SliceIndex), expr};

    // Missing expressions resolve into None.
    return {PValue(NoneAttr::get(emitter.getContext())), this};
  };

  // TODO: Generalize to more than 3 operands.  We might also want to turn this
  // into a well-known static method instead of overloading onto constructor.
  CallOperands operands;
  operands.add(getOperand(lower));
  if (!operands.values.back().ir)
    return {};
  operands.add(getOperand(upper));
  if (!operands.values.back().ir)
    return {};
  operands.add(getOperand(stride));
  if (!operands.values.back().ir)
    return {};
  return emitter.emitResult(InitializerUValue::create(std::move(operands)),
                            this, dest);
}

AnyValue SubscriptNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // Subscripting a generic function binds the parameter expressions.
  auto baseAnyValue = emitter.emitExpr(base, EC_SubscriptBase);
  if (!baseAnyValue)
    return {};

  // If the baseAnyValue has a bound callable symbol, then this is applying
  // (more?) parameter expressions to bind its parameters.
  if (auto overloads = baseAnyValue.getIfOverloadSet()) {
    emitter.shared.notifyListenerOnParameterBinding(overloads->fnDecls,
                                                    rsquareLoc, operands);
    // Mutate the overloadset directly.  This is a bit gross, but we know we're
    // the only user of it.
    if (failed(bindParamValuesToDirectCall(*overloads, operands, emitter)))
      return {};
    return emitter.emitResult(overloads, this, dest);
  }

  // Otherwise, this must be a concrete value to be able to subscript it.
  CValue baseValue = emitter.emitCValue({baseAnyValue, base}, EC_SubscriptBase);
  if (!baseValue)
    return {};
  ASTType baseType = baseValue.getRValueType();

  if (auto value = baseValue.getIfPValue()) {
    // Check for attribute bindings to an MLIR operation.
    if (auto unboundOperator =
            dyn_cast<UnboundMLIROperationAttr>(value.get())) {
      PValue result =
          bindAttributesToMLIROperatorCall(*this, unboundOperator, emitter);
      return emitter.emitResult(result, this, dest);
    }

    // If this is a signature-type PValue callable, this is binding parameter
    // values to a call.
    if (auto sig = dyn_cast<FnTypeGeneratorType>(baseType)) {
      PValue result =
          bindToIndirectCall(value, sig, operands, emitter, getIndexRange());
      if (!result)
        return {};
      return emitter.emitResult(result, this, dest);
    }
  }

  // If the sub-value is an unbound Type, try binding parameters to it!
  if (Type typeValue = baseValue.getIfTypeValue()) {
    // Handle user-defined types and custom MLIR types.
    if (auto structValue = dyn_cast<StructMetaType>(baseType)) {
      PValue result = substituteParametersIntoUserDefinedType(
          baseValue.getIfPValue(), operands, getLoc(), lsquareLoc, rsquareLoc,
          emitter);
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
    if (isa<MagicMLIRAttrType, MagicMLIRDeferredAttrType>(typeValue)) {
      std::string result = substituteMLIRMagic(*this, emitter);
      if (result.empty())
        return {};
      const bool fallbackToDeferredAttr =
          isa<MagicMLIRDeferredAttrType>(typeValue);
      // When we are not allowed to fallback to deferred attrobute, report an
      // error if attribute cannot be constructed.
      PValue attr = synthesizeMLIRAttrFromString(
          result, getLoc(), emitter.shared,
          /* reportError = */ !fallbackToDeferredAttr);
#ifndef NDEBUG
      {
        // Ideally we want to build deferred attribute first and stringize it.
        // This will help us to keep everything in sync. The downside of this is
        // it will add unnecessary overhead as many attributes don't need to be
        // deferred.
        // Therefore, to make sure deferred attributes are built correctly, for
        // non-production build do extra check.
        auto deferredAttr =
            buildAttrCtorDeferredAttrFromMLIRAttr(*this, emitter);
        assert(result == getStringRepresentation(deferredAttr) &&
               "string representations of an attribute don't match");
      }
#endif // NDEBUG
      if (fallbackToDeferredAttr) {
        if (!attr) {
          auto deferredAttr =
              buildAttrCtorDeferredAttrFromMLIRAttr(*this, emitter);
          // Wrap this attribute with `#kgen.deferred` to let elaborator
          // concretize attributes that are not concrete now.
          attr = PValue(deferredAttr);
        } else {
          // If attribute can be constructed at this point, but user used
          // `__mlir_deferred_attr`, let user know that regular `__mlir_attr`
          // should be used instead.
          emitMLIRDeferedAttrToMLIRAttrWarning(getLoc(), emitter);
        }
      }
      return emitter.emitResult(attr, this, dest);
    }
  }

  // Support subscripting a !kgen.variadic value, which are used in parameter
  // lists.  This enables us to work with parameter backs in a more natural way,
  // e.g. fn thing[*Ts: Copyable & Movable]():
  //      type = Ts[123]
  // We should really remove this when going to a better parameter pack rep.
  //
  // FIXME(#13015): We shouldn't need this code. Variadic arguments should emit
  // a standard library type that implements `__getitem__` and `__setitem__`.
  if (auto variadic = dyn_cast<VariadicType>(baseType)) {
    // Attempt to convert the index.
    if (operands.size() != 1 || operands[0].isKeywordOrUnpackedKeyword()) {
      emitter.emitError(getLoc()) << "variadic can only be subscripted with a "
                                     "single positional operand";
      return {};
    }

    // We're going to emit the index as a PValue even if in a dynamic context.
    auto paramEmitter = emitter.getParamEmitter(EC_Subscript);
    CValue index = paramEmitter.emitIndex(operands[0].expr, EC_Subscript);
    if (!index)
      return {};
    // Inside a parameter context, emit a parameter operator.
    if (auto indexPV = index.getIfPValue())
      if (auto basePV = baseValue.getIfPValue()) {
        auto res = ParamOperatorAttr::get(POC::VariadicGet, {basePV, indexPV});
        return emitter.emitResult(PValue(res), this, dest);
      }
    emitter.emitError(getLoc())
        << "can only subscript variadics in parameter expressions";
    return {};
  }

  // Otherwise, if there is no symbol, it is just an LValue or RValue being
  // subscript, invoking a dynamic subscript with __getitem__ and __setitem__.
  return emitGetterSetterAccess(this, {baseValue, base}, operands, dest,
                                emitter);
}

AnyValue ParenNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  return emitter.emitExpr(subExpr, dest);
}

/// Both tuple literals and list literals are emitted as heterogenous sequences,
/// with each element type encoded in a variadic type parameter.
static AnyValue emitHeterogenousSequence(ValueDest &dest, ExprEmitter &emitter,
                                         ASTType type, const ExprNode *node,
                                         ArrayRef<ExprNode *> exprs) {
  // If we failed to look up the tuple/list type, fail.
  if (!type || type.isTypeCheckErrorType()) {
    dest.resetForError();
    return {};
  }

  // Emit each of the tuple elements.
  CallOperands operands;
  for (ExprNode *expr : exprs) {
    auto exprVal = emitter.emitExpr(expr, EC_TupleElement);
    if (!exprVal) {
      dest.resetForError();
      return {};
    }
    operands.add({std::move(exprVal), expr});
  }

  // The ASTType will carry around parameters bound, we want to unbind them so
  // they can be inferred from the elements.
  type = type.getWithoutParameters(emitter.shared);

  // Emit a call to the builtin type constructor as an implicit conversion.
  // The type parameters are inferred from the element types.
  return emitter.emitConstructorCall(type, std::move(operands), node,
                                     CallSyntax::kTypeCall, dest);
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

AnyValue DictSubscriptNode::emitIR(ValueDest &dest,
                                   ExprEmitter &emitter) const {
  // TODO: T{...} is a Mojo extension for typed dictionary literals.  This is
  // speculative.  If it isn't the right thing, then remove the parsing and
  // ExprNode entirely.
  ASTType typeValue = emitter.emitExprType(base);
  if (!typeValue)
    return {};

  emitter.emitError(getLoc(), "TODO: cannot emit dictionary literals yet")
      << getRange();
  return {};
}

/// Given an operator, return the SpecialFunctionInfo that implements it.
static SpecialFunctionInfo getOpSpecialFunctions(ExprNode::Kind kind,
                                                 bool isReversed) {

  // Use an if chain to find the right match.  We can't use switch here because
  // multiple special functions may implement the same kind, e.g. __add__ and
  // __radd__ special methods both implement kAdd.
#define SF(ENUM, NAME, MINOPERANDS, MAXOPERANDS, EXPRNODE, FLAGS)              \
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

  // If this is a 'not in' emit the 'in' expression and then invert the result.
  //  We use this style to make sure that a direct emission emits into
  // the ValueDest directly.
  if (kind == ExprNode::Kind::kCmpNotIn) {
    ValueDest inDest(EC_OperatorOperandValue);
    auto inResult = emitBinOpCall(lhs, rhs, ExprNode::Kind::kCmpIn, inDest,
                                  callExpr, emitter);
    return UnaryOpNode::emitArith(ExprNode::Kind::kBoolNot, callExpr,
                                  {inResult, callExpr}, dest, emitter);
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnInfo = getOpSpecialFunctions(kind, /*isReversed=*/false);
  if (specialFnInfo.kind == SpecialFunctionKind::kNormal) {
    // This means that the operator is not defined in SpecialFunctions.def.
    emitter.shared.emitError(callExpr->getLoc(), "operator not yet supported");
    return {};
  }

  // Use one 'operands' set for the arg values even though we switch them back
  // and forth.  Resolving a set can mutate the argument list (e.g. emitting
  // PValues to dynamic values) even if the lookup fails and we don't want to
  // materialize them multiple times
  CallOperands operands({lhs, rhs});

  // `a in b` => `b.__contains__(a)` and there is no reversed form.
  if (kind == ExprNode::Kind::kCmpIn)
    std::swap(operands[0], operands[1]);

  // Check to see if we have a forward version of this function on the primary
  // receiver.
  if (auto lhsCV = lhs.ir.getIfCValue()) {
    if (PValue callee = OverloadSet::lookupAndResolve(
            lhsCV.getRValueType(), specialFnInfo.name, operands, callExpr,
            CallSyntax::kOperator, emitter))
      return emitter.emitIndirectCall(callee, std::move(operands), dest,
                                      CallSyntax::kOperator, callExpr);
  }

  // Check to see if we have the reverse version of this operator.
  auto reversedFnInfo = getOpSpecialFunctions(kind, /*isReversed=*/true);
  if (reversedFnInfo.kind != SpecialFunctionKind::kNormal) {
    // Swap the operand order.
    std::swap(operands[0], operands[1]);
    if (auto rhsCV = rhs.ir.getIfCValue()) {
      if (PValue callee = OverloadSet::lookupAndResolve(
              rhsCV.getRValueType(), reversedFnInfo.name, operands, callExpr,
              CallSyntax::kReversedOperator, emitter))
        return emitter.emitIndirectCall(callee, std::move(operands), dest,
                                        CallSyntax::kOperator, callExpr);
    }

    // Swap these back so we emit the right error.
    std::swap(operands[0], operands[1]);
  }

  // Emit an error complaining about the forward version of the operator.
  return emitter.emitNamedMethodCall(specialFnInfo.name, std::move(operands),
                                     dest, CallSyntax::kOperator, callExpr);
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
  auto resultValue = emitter.emitExpr(rhs, assignDest);
  if (!resultValue)
    return {};

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

  // Emit trait composition if both operands have AnyTrait types.
  if (kind == kAnd) {
    auto lhsTrait =
        dyn_cast_or_null<AnyTraitType>(lhsRV.getRValueTypeIfResolvable());
    auto rhsTrait =
        dyn_cast_or_null<AnyTraitType>(rhsRV.getRValueTypeIfResolvable());
    if (lhsTrait && rhsTrait) {
      SmallVector<SymbolRefAttr> symbols(lhsTrait.getTraitType().getSymbols());
      llvm::append_range(symbols, rhsTrait.getTraitType().getSymbols());
      return emitter.emitResult(TraitType::get(emitter.getContext(), symbols),
                                this, dest);
    }
  }

  return emitBinOpCall({lhsRV, lhs}, {rhsRV, rhs}, kind, dest, this, emitter);
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

  // Emit the LHS value as a bool/i1 value.
  CValue lhsV = emitter.emitExprCValue(lhs, EC_OperatorOperandValue);
  RValue lhsI1Val = emitter.emitI1({lhsV, lhs}, EC_OperatorOperandValue);
  PValue lhsI1PVal = lhsI1Val.getIfPValue();

  if (!emitter.builder) {
    lhsI1PVal = emitter.emitPValue({lhsI1Val, lhs}, EC_BoolCondition);
    if (!lhsI1PVal)
      return {};
    CValue rhsV = emitter.emitExprCValue(rhs, EC_BoolCondition);

    // Coerce the true/false values into a compatible type if they disagree.
    if (emitter.coerceTypesToEachOther(getLoc(), lhsV, lhs, rhsV, rhs, {}))
      return {};

    PValue lhsPV = emitter.emitPValue({lhsV, lhs}, EC_OperatorOperandValue);
    if (!lhsPV)
      return {};
    PValue rhsPV = emitter.emitPValue({rhsV, rhs}, EC_OperatorOperandValue);
    if (!rhsPV)
      return {};

    if (kind == kBoolOr) // and/or swap true/false operands
      std::swap(lhsPV, rhsPV);

    auto value = ParamOperatorAttr::get(POC::Cond, {lhsI1PVal, rhsPV, lhsPV});
    return emitter.emitResult(value, this, dest);
  }

  SRValue lhsI1SRValue =
      emitter.emitSRValue({AnyValue(lhsI1Val), lhs}, EC_BoolCondition);
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
  auto configEmitter = [&](bool isLHS) {
    emitter.builder = isLHS ? falseBuilder : trueBuilder;
  };
  // Try to find compatibility between the raw values.  Pass in a null SMLoc so
  // that an error isn't diagnosed with an error message.
  if (emitter.coerceTypesToEachOther(SMLoc(), lhsV, lhs, rhsV, rhs,
                                     configEmitter)) {
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
      RValue rhsI1Value = emitter.emitI1({rhsV, rhs}, EC_OperatorOperandValue);
      emitter.builder = trueBuilder;
      rhsV = emitter.emitCValue({rhsI1Value, rhs}, EC_OperatorOperandValue,
                                boolType);
    }

    // Similarly, if the LHS was already a Bool then use it, otherwise convert
    // the i1 we already have back to Bool with a ctor.
    if (!lhsV.getRValueType().isEqualCanon(boolType)) {
      emitter.builder = falseBuilder;
      lhsV = emitter.emitCValue({lhsI1SRValue, lhs}, EC_OperatorOperandValue,
                                boolType);
    }

    if (!lhsV || !rhsV)
      return {};
  }

  // Detect unreachable code and warn about it.
  auto deadCodeCheck = [&]() {
    if (lhsI1PVal) {
      IntegerAttr asIntAttr = dyn_cast<IntegerAttr>(lhsI1PVal.get());
      if (!asIntAttr)
        return;
      bool isZero = asIntAttr.getValue().isZero();
      bool deadElse = false;
      if (kind == kBoolOr && !isZero) {
        deadElse = true;
        emitter.emitWarning(this->getLoc())
            << "unreachable code on right side of 'True or ...'";
      } else if (kind == kBoolAnd && isZero) {
        deadElse = true;
        emitter.emitWarning(this->getLoc())
            << "unreachable code on right side of 'False and ...'";
      } else {
        // This has no dead code, but let's still warn about a constant branch
        // condition.
        emitter.emitWarning(this->getLoc())
            << "constant value on left side of '" << (isZero ? "False" : "True")
            << " " << (kind == kBoolOr ? "or" : "and") << " ...'";
      }
      // Eliminate the dead code.
      if (deadElse)
        markRegionUnreachable(&ifOp.getElseRegion(), ifOp.getLoc());
    }
  };

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
    deadCodeCheck();
    return emitter.emitResult(SRValue(ifOp.getResult(0)), this, dest);
  }

  // If we have a memory only type, we have to handle the various issues with
  // the ValueDest.  It may specify an MLValue to emit into, it may be
  // ambiguous (like a call argument) or it may even be something like a
  // DLValue.  We handle this by projecting the ValueDest to an MLValue if we
  // can, but otherwise using a scratch buffer if not.
  emitter.builder->setInsertionPoint(ifOp);
  MLValue destBuffer = dest.getMLValueForResult(getLoc(), resultType, emitter);

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
  deadCodeCheck();
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
  ifOp->erase();

  return emitter.emitCResult(MRValue(destBuffer), this, dest);
}

/// Emit the x^ expression.
AnyValue UnaryOpNode::emitTransfer(AnyValue argValue, ValueDest &dest,
                                   ExprEmitter &emitter) const {
  auto loc = getLoc();
  // We don't track ownership right in parameter expressions, e.g. we don't have
  // a PBValue to correspond to PRValue (which is what PValue really is).  As
  // such, transfering in a parameter context isn't useful, just disallow it.
  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(
        loc, "cannot transfer a value in a parameter context");

  // If the input is already an owned RValue, then there is no need to
  // transfer from the temporary.
  if (argValue.getIfRValue()) {
    if (argValue.getIfPValue()) {
      emitter.emitError(loc, "cannot transfer from a parameter expression; did "
                             "you want to introduce a local 'var'?");
      return {};
    }

    emitter.emitWarning(loc)
        << "transfer from an owned value has no effect and can be removed"
        << FixIt::remove(loc);
    return emitter.emitResult(argValue, this, dest);
  }

  // We don't support transferring from trivial values, since this won't end the
  // origin. CheckLifetimes doesn't and can't track these things because they
  // don't have consume operators, move operators, etc.
  if (CValue argCValue = argValue.getIfCValue();
      argCValue && argCValue.getRValueType().isTrivial(loc, emitter.shared)) {
    emitter.emitWarning(loc)
        << "transfer from a value of trivial register type "
        << argCValue.getRValueType() << " has no effect and can be removed"
        << FixIt::remove(loc);
    return emitter.emitResult(argValue, this, dest);
  }

  // The operand value must be in memory to have a origin.
  if (!argValue.isMValue()) {
    if (argValue.getIfSBValue()) {
      emitter.emitError(loc, "expression is a register value, "
                             "transfer requires ownership");
      return {};
    }
    // Note: a DLValue like a[i] could be copied into a local value, but that
    // would almost always return an RValue which need not be transferred
    // anyway.
    emitter.emitError(loc, "expression does not live in a memory location, so "
                           "it need not be transferred");
    return {};
  }

  // The transfer expression expects the result to be a ownable value that it
  // can launder into an RValue.
  Value value = argValue.getMValueReference();

  // Origin checking needs to understand this value or field.
  Value trackableValue;
  if (value)
    trackableValue = OriginTrackable::findUnderlyingValueFromField(value);
  if (!trackableValue) {
    emitter.emitError(loc,
                      "expression does not designate a value with an origin");
    return {};
  }

  // If the memory type isn't mutable, then we can't transfer out of it.
  if (!cast<RefType>(value.getType()).isMutableKnown(true)) {
    emitter.emitError(loc, "cannot transfer out of immutable reference");
    return {};
  }

  // Make sure the origin of the value is extended to at least here.  This
  // is a use, and the `_ = x^` pattern to extend the origin of something is
  // very common.
  emitter.builder->create<OwnershipUseOp>(getLocation(emitter), value);

  // For memory values, we can just treat the value as an MRValue, and whoever
  // consumes this can consume it directly.
  return emitter.emitResult(MRValue(value), this, dest);
}

AnyValue UnaryOpNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  auto exprRep = emitter.emitExpr(subExpr, EC_OperatorOperandValue);
  if (!exprRep)
    return {};

  if (kind == kTransfer)
    return emitTransfer(exprRep, dest, emitter);

  if (kind == kUnpack) {
    // Unpacks are only allowed inside parameter or argument lists.
    emitter.emitError(getLoc(), "can't use starred expression here")
        << getRange();
    return {};
  }

  return emitArith(kind, this, {exprRep, subExpr}, dest, emitter);
}

/// Emit a unary arithmetic operation as a dynamic expression.
AnyValue UnaryOpNode::emitArith(Kind kind, const ExprNode *expr,
                                ASTExprAnd<AnyValue> argValue, ValueDest &dest,
                                ExprEmitter &emitter) {
  if (!argValue.ir)
    return {};

  if (kind == kBoolNot) {
    // Turn this into a call to __bool__.
    ValueDest subDest(EC_OperatorOperandValue);
    argValue.ir =
        emitter.emitNamedMethodCall("__bool__", CallOperands(argValue), subDest,
                                    CallSyntax::kMethodCall, expr);
    if (!argValue.ir)
      return {};
    // Now that we know we bool-ized the expression, invert it with ~.
    return emitArith(kInvert, expr, argValue, dest, emitter);
  }

  // If this operator maps onto a special function, attempt to lower it.
  auto specialFnInfo = getOpSpecialFunctions(kind, /*isReversed=*/false);
  assert(specialFnInfo.kind != SpecialFunctionKind::kNormal &&
         "Unary operators are implemented via special methods");

  return emitter.emitNamedMethodCall(specialFnInfo.name, CallOperands(argValue),
                                     dest, CallSyntax::kOperator, expr);
}

AnyValue IfElseOpNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  RValue condRVal = emitter.emitExprI1(condExpr, EC_BoolCondition);

  // Inside a parameter context, emit conditional expression.
  if (!emitter.builder) {
    PValue condPVal =
        emitter.emitPValue({condRVal, condExpr}, EC_BoolCondition);
    if (!condPVal)
      return {};

    CValue trueVal = emitter.emitExprCValue(trueExpr, EC_CondExpr);
    CValue falseVal = emitter.emitExprCValue(falseExpr, EC_CondExpr);

    if (emitter.coerceTypesToEachOther(getLoc(), trueVal, trueExpr, falseVal,
                                       falseExpr, {}))
      return {};

    PValue truePVal = emitter.emitPValue({trueVal, trueExpr}, EC_CondExpr);
    if (!truePVal)
      return {};
    PValue falsePVal = emitter.emitPValue({falseVal, falseExpr}, EC_CondExpr);
    if (!falsePVal)
      return {};

    auto value =
        ParamOperatorAttr::get(POC::Cond, {condPVal, truePVal, falsePVal});
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

  auto deadCodeCheck = [&]() {
    if (PValue condPVal = condRVal.getIfPValue()) {
      // Warn about dead code and remove it.
      IntegerAttr asIntAttr = dyn_cast<IntegerAttr>(condPVal.get());
      if (!asIntAttr)
        return;
      Region *deadRegion = &ifOp.getElseRegion();
      if (asIntAttr.getValue().isZero()) {
        deadRegion = &ifOp.getThenRegion();
        emitter.emitWarning(this->getLoc())
            << "left hand side expression of 'if False' is dead";
      } else {
        emitter.emitWarning(this->getLoc())
            << "right hand side expression of 'if True' is dead";
      }
      markRegionUnreachable(deadRegion, ifOp.getLoc());
    }
  };

  // If both results were M values and both sides agree with the result type,
  // then we can propagate the result as an MValue that has a merged lifetime.
  auto handleTwoMValues = [&]() -> AnyValue {
    if (!falseVal.isMValue() || !trueVal.isMValue())
      return {};

    Value falseMVal = falseVal.getMValueReference();
    Value trueMVal = trueVal.getMValueReference();

    // See if the true and false values directly union together.  This
    // requires the rvalue types to be the same but allows the reference
    // types to be different.
    RefType commonRefType = emitter.getCommonRefType(
        cast<RefType>(falseMVal.getType()), cast<RefType>(trueMVal.getType()));
    if (!commonRefType)
      return {};

    // Check to see if the two values dominate the 'if'.  We don't want to form
    // an union'ed origin that includes an origin for something in the else
    // block, like an RValue temporary.  Such things will require a copy.
    //
    // The ideal thing to do would be to have use-def chains on the origin
    // itself, which would allow us to handle ref results from functions, but
    // we don't have that.  Instead, find the underlying values and see if we
    // can reason about them from the IR tree.
    auto isAcceptableSource = [&](Value origin) -> bool {
      // Strip off GERs and Rebinds and RefImmutOp, and the get the block that
      // defines the operation or block argument.
      origin = OriginTrackable::findUnderlyingValueFromField(origin);
      if (!origin)
        return false;
      Block *originBlock = origin.getParentBlock();
      Operation *curOp = ifOp;
      // Scan up the region tree.
      do {
        if (curOp->getBlock() == originBlock)
          return true; // Found a dominating block containing the origin.
        curOp = curOp->getParentOp();
      } while (curOp);
      return false;
    };

    if (!isAcceptableSource(trueMVal) || !isAcceptableSource(falseMVal))
      return {};

    // If this succeeded, we can emit a conversion to the common type in each
    // branch and produce the result as the right MValue type.
    auto handleBlock = [&](Block &block, const ExprNode *expr,
                           Value value) -> LogicalResult {
      emitter.builder->setInsertionPointToEnd(&block);
      auto conv =
          emitter.emitZeroCostConvert({SRValue(value), expr}, commonRefType);
      if (!conv)
        return failure();
      auto convVal = conv.getIfSRValue();
      assert(convVal && "zero cost convert changed value type");
      emitter.builder->create<HLCF::YieldOp>(ifLoc, convVal);
      return success();
    };
    if (failed(handleBlock(ifOp.getThenBlock(), trueExpr, trueMVal)) ||
        failed(handleBlock(ifOp.getElseBlock(), falseExpr, falseMVal)))
      return {};
    emitter.builder->setInsertionPointAfter(ifOp);

    // Ensure the correct type is used.
    ifOp->getResult(0).setType(commonRefType);
    deadCodeCheck();

    // Compute the right IRValue type based on what we were given, we know the
    // inputs are some kind of MValue.
    AnyValue result;
    // TODO: CheckLifetimes cannot handle consumption of indirect RValues.
    // if (falseVal.getIfMRValue() && trueVal.getIfMRValue())
    //   result = MRValue(ifOp.getResult(0));
    if (falseVal.getIfMLValue() && trueVal.getIfMLValue())
      result = MLValue(ifOp.getResult(0));
    else if (falseVal.getIfMBPValue() && trueVal.getIfMBPValue())
      result = MBPValue(ifOp.getResult(0));
    else
      result = MBValue(ifOp.getResult(0));
    return emitter.emitResult(result, this, dest);
  };

  if (AnyValue result = handleTwoMValues())
    return result;

  /// If the types disagree, then we need to emit a conversion to a common
  /// type. See if one is convertible to the other, and if so, emit a
  /// conversion to get to a common type.
  auto configEmitter = [&](bool isLHS) {
    Block &b = isLHS ? ifOp.getThenBlock() : ifOp.getElseBlock();
    emitter.builder->setInsertionPointToEnd(&b);
  };
  if (emitter.coerceTypesToEachOther(getLoc(), trueVal, trueExpr, falseVal,
                                     falseExpr, configEmitter)) {
    dest.resetForError();
    return {};
  }

  auto resultType = trueVal.getRValueType();

  // Ok, we now know if the types were register_passable or not, so finish up
  // the logic.  register_passable values get merged together as SSA registers
  // in the 'if' result.
  if (resultType.isRegisterPassable(trueExpr->getLoc(), emitter.shared)) {
    // Finish false.
    emitter.builder->setInsertionPointToEnd(&ifOp.getElseBlock());
    auto falseSR = emitter.emitSRValue({falseVal, falseExpr}, EC_CondExpr);
    if (!falseSR)
      return {};
    emitter.builder->create<HLCF::YieldOp>(ifLoc, falseSR);
    // Finish true.
    emitter.builder->setInsertionPointToEnd(&ifOp.getThenBlock());
    auto trueSR = emitter.emitSRValue({trueVal, trueExpr}, EC_CondExpr);
    if (!trueSR)
      return {};
    emitter.builder->create<HLCF::YieldOp>(ifLoc, trueSR);
    emitter.builder->setInsertionPointAfter(ifOp);
    // Ensure the correct type is used.
    ifOp->getResult(0).setType(trueSR.getType());
    deadCodeCheck();
    return emitter.emitResult(SRValue(ifOp.getResult(0)), this, dest);
  }

  // If we have a memory only type, we have to handle the various issues with
  // the ValueDest.  It may specify an MLValue to emit into, it may be
  // ambiguous (like a call argument) or it may even be something like a
  // DLValue.  We handle this by projecting the ValueDest to an MLValue if we
  // can, but otherwise using a scratch buffer if not.
  emitter.builder->setInsertionPoint(ifOp);
  MLValue destBuffer = dest.getMLValueForResult(getLoc(), resultType, emitter);

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
  deadCodeCheck();
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
RValue ChainedCmpOpNode::emitNextCmp(ExprEmitter &emitter, size_t opIdx,
                                     RValue prevCmpVal, RValue prevRHS,
                                     bool hasPrevIfOp, ValueDest &dest) const {
  ExprContext context = dest.getContext();
  bool isLastOne = opIdx + 1 == ops.size();
  SMLoc ifLoc = exprs[opIdx - 1]->getLoc();
  Location ifLocation = emitter.translateLocation(ifLoc);
  std::optional<OpBuilder> lastBuilder = {};
  if (emitter.builder)
    lastBuilder = emitter.builder.value();
  RValue prevCmpI1Value = emitter.emitI1({prevCmpVal, this}, EC_BoolCondition);
  if (!prevCmpI1Value)
    return {};
  SRValue prevCmpI1SRValue;
  HLCF::IfOp ifOp;
  if (emitter.builder) {
    prevCmpI1SRValue =
        emitter.emitSRValue({prevCmpI1Value, this}, EC_BoolCondition);
    if (!prevCmpI1SRValue)
      return {};
    // In the dynamic case we need to build the RHS evaluation in the Then
    // region of an IfOp.  But if we end up having all parameters, it will not
    // have been necessary.
    ifOp = emitter.builder->create<HLCF::IfOp>(
        ifLocation, prevCmpVal.getType().mlirType, prevCmpI1SRValue);
    emitter.builder->createBlock(&ifOp.getThenRegion());
  }
  RValue newRHS =
      emitter.emitExprRValue(exprs[opIdx + 1], EC_OperatorOperandValue);
  if (!newRHS)
    return {};
  ValueDest newCmpDest(context);
  AnyValue newCmp =
      emitBinOpCall({prevRHS, exprs[opIdx]}, {newRHS, exprs[opIdx + 1]},
                    ops[opIdx], newCmpDest, this, emitter);
  RValue newCmpCRV = emitter.emitRValue({newCmp, exprs[opIdx]}, context);
  if (!newCmpCRV)
    return {};

  if (prevCmpVal.getIfPValue() && prevCmpI1Value.getIfPValue() &&
      newCmpCRV.getIfPValue()) {
    // Since we have PValues, we didn't actually need that ifOp after all. Let's
    // clean up before returning a PValue directly or recurring.
    if (emitter.builder) {
      ifOp.erase();
      emitter.builder = lastBuilder;
    }
    if (!prevCmpVal.getRValueType().isEqualCanon(newCmpCRV.getRValueType())) {
      emitter.emitError(
          ifLocation,
          "comparison result types of chained comparison must match");
      return {};
    }
    auto chainedBool = ParamOperatorAttr::get(
        POC::Cond,
        {prevCmpI1Value.getIfPValue(), /*trueVal=*/newCmpCRV.getIfPValue(),
         /*falseVal=*/prevCmpVal.getIfPValue()});
    RValue ret = isLastOne ? chainedBool
                           : emitNextCmp(emitter, opIdx + 1, chainedBool,
                                         newRHS, false, dest);
    if (hasPrevIfOp) {
      emitter.builder->create<HLCF::YieldOp>(
          ifLocation, emitter.emitSRValue({ret, exprs[opIdx]}, context));
    }
    return ret;
  }

  // We need to return the result of the IfOp as a RValue.
  // More concretely, it will be an SRValue or, for exotic memory-only bool
  // equivalents, one of the pointer type RValues.
  // But for simplicity, let's only support return values that can fit in an
  // SRValue.
  // TODO - make this more general.
  // To refuse memory types right now, check what (other) comparison results
  // are.
  if (!newCmpCRV.getRValueType().isRegisterPassable(ifLoc, emitter.shared)) {
    emitError(ifLocation,
              "chained comparison operator does not currently support "
              "memory-only return types");
    return {};
  }

  RValue newOrNextResult;
  if (isLastOne) {
    newOrNextResult = newCmpCRV;
    auto newCmpSRV = emitter.emitSRValue({newCmpCRV, exprs[opIdx]}, context);
    if (!newCmpSRV)
      return {};
    emitter.builder->create<HLCF::YieldOp>(ifLocation, newCmpSRV);
  } else {
    newOrNextResult =
        emitNextCmp(emitter, opIdx + 1, newCmpCRV, newRHS, true, dest);
    if (!newOrNextResult)
      return {};
  }

  if (!newOrNextResult.getRValueType().isEqualCanon(prevCmpVal.getType())) {
    emitter.emitError(
        ifLocation, "comparison result types of chained comparison must match");
  }
  emitter.builder->createBlock(&ifOp.getElseRegion());
  ifOp->getResult(0).setType(prevCmpVal.getType());

  auto newCmpSRV = emitter.emitSRValue({prevCmpVal, exprs[opIdx - 1]}, context);
  if (!newCmpSRV)
    return {};
  emitter.builder->create<HLCF::YieldOp>(ifLocation, newCmpSRV);
  if (lastBuilder)
    emitter.builder = lastBuilder;
  auto r0 = ifOp->getResult(0);
  if (hasPrevIfOp)
    emitter.builder->create<HLCF::YieldOp>(ifLocation, r0);

  return SRValue(r0);
}

AnyValue ChainedCmpOpNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  AnyValue e0Rep = emitter.emitExpr(exprs[0], EC_OperatorOperandValue);
  AnyValue e1Rep = emitter.emitExpr(exprs[1], EC_OperatorOperandValue);
  if (!e0Rep || !e1Rep)
    return {};

  ValueDest cmpDest(dest.getContext());
  AnyValue cmpe0e1RV =
      emitBinOpCall({e0Rep, exprs[0]}, {e1Rep, exprs[1]}, ops[0],
                    exprs.size() == 2 ? dest : cmpDest, this, emitter);
  if (exprs.size() == 2)
    return cmpe0e1RV;

  RValue lastCmpExpr =
      emitter.emitRValue({cmpe0e1RV, exprs[1]}, EC_BoolCondition);
  RValue e1RV = emitter.emitRValue({e1Rep, exprs[1]}, EC_OperatorOperandValue);
  if (!lastCmpExpr || !e1RV)
    return {};
  return emitter.emitResult(
      emitNextCmp(emitter, 1, lastCmpExpr, e1RV, false, dest), this, dest);
}

AnyValue FunctionTypeNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  // Parameters declared within the function type must be visible. Create a
  // dummy declaration.
  ASTDecl &dummyScope = emitter.getDeclResolver().addFullyResolvedDecl(
      nullptr, StringAttr(), getLoc(), &emitter.declScope);

  // Type check any parameters we have.
  TypeCheckedParamList paramList(parsedParams, dummyScope);

  // Reinflate a ParsedArgumentList.
  ParsedArgumentList argList;
  argList.parsedArgs = llvm::to_vector(parsedArgs);
  argList.resultArg = resultArg;
  argList.effects = effects;

  SpecialFunctionInfo fnInfo; // Not a named function.
  TypeCheckedFnSignature tcSignature(paramList, argList, originExpr,
                                     /*fnDecl=*/nullptr, fnInfo);

  // Compute the signature of the function.
  FnTypeGeneratorType signature = tcSignature.getFnTypeGeneratorType();
  if (!signature)
    return {}; // Error already emitted.

  // Check to see if any of the arguments were erroneous. If so, we don't want
  // to produce a function type with a nested type check error, just fail.  This
  // is unlike function defs which want to handle arguments that are invalid.
  for (auto &arg : tcSignature.argList.parsedArgs) {
    if (arg.isErroneous)
      return {};
  }

  // Set the value of the dummy scope to the generated signature so that we can
  // still resolve information about it in tools.
  dummyScope.setIRValue(PValue(signature));

  // The parsed FuncTypeGeneratorType is set to the pretty type that includes
  // implicit origins, we strip off the named origin decl references and replace
  // them with indices.
  signature = signature.replaceImplicitOriginsWithIndexes(
      tcSignature.implicitOriginDecls);

  if (argList.effects.isEscaping()) {
    // Create a self contained signature type that represents the closure.
    auto [capturedRefs, wrapperSig] =
        DeclResolver::createSelfContainedSignature(signature);
    ASTDecl *decl = emitter.declScope.getNearestDeclOfType<FileModuleOp>();
    StructDeclOp structOp =
        emitter.shared.getOrCreateClosureWrapper(getLoc(), wrapperSig, decl);

    // Closure creation failed. Error emitted in ClosureEmitter.
    if (!structOp)
      return {};

    // Build the return type by binding the parent parameter values to the
    // struct parameters.
    // TODO: Handle partial binding.
    StructType selfType = structOp.bindReference(llvm::map_to_vector(
        capturedRefs, [](ParamDeclRefAttr ref) -> TypedAttr { return ref; }));
    return emitter.emitResult(ASTType(selfType), this, dest);
  }
  return emitter.emitResult(ASTType(signature), this, dest);
}

AnyValue MagicFunctionNode::emitIR(ValueDest &dest,
                                   ExprEmitter &emitter) const {
  if (kind == kOriginOf)
    return emitOriginOf(dest, emitter);

  // __get_nearest_error_slot returns an MLValue.
  if (kind == kGetNearestErrorSlot) {
    if (!subExprs.empty()) {
      emitter.emitError(getLoc())
          << "unexpected argument in call to '__get_nearest_error_slot'"
          << getRange();
      return {};
    }
    MLValue err = emitter.findNearestErrorSlot();
    if (!err) {
      emitter.emitError(getLoc())
          << "cannot use '__get_nearest_error_slot' in non-raising context"
          << getRange();
      return {};
    }
    return emitter.emitResult(err, this, dest);
  }

  // All other magic function types take exactly one argument.
  if (subExprs.size() != 1) {
    emitter.emitError(getLoc(), "expected a single argument") << getRange();
    return {};
  }

  if (kind == kTypeOf)
    return emitTypeOf(dest, emitter);

  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(this);

  // Emit the subexpression.
  ExprNode *subExpr = subExprs.front();
  CValue subExprValue = emitter.emitExprCValue(subExpr, dest.getContext());
  if (!subExprValue)
    return {};

  // __get_mvalue_as_litref(someMValue) returns the !lit.ref.
  if (kind == kGetMValueAsLitRef) {
    if (!subExprValue.isMValue()) {
      emitter.emitError(getLoc(), "cannot use non-memory value") << getRange();
      return {};
    }

    // Return the MValue as an SRValue since the ref itself is the result.
    Value refValue = subExprValue.getMValueReference();
    return emitter.emitResult(SRValue(refValue), this, dest);
  }

  // __get_litref_as_mvalue(someLITRef) returns an MValue.
  if (kind == kGetLitRefAsMValue) {
    Value exprVal =
        emitter.emitSRValue({subExprValue, subExpr}, dest.getContext());
    if (!exprVal)
      return {};
    if (!isa<RefType>(exprVal.getType())) {
      emitter.emitError(getLoc(), "operand isn't a '!lit.ref' type ")
          << ASTType(exprVal.getType()) << getRange();
      return {};
    }

    return emitter.emitResult(CValue::getMValueForRef(exprVal), this, dest);
  }

  // __get_address_as_uninit_lvalue and __get_address_as_owned_value take a
  // !kgen.pointer.
  RValue exprRVal =
      emitter.emitRValue({subExprValue, subExpr}, dest.getContext());
  if (!exprRVal)
    return {};

  if (!isa<PointerType>(exprRVal.getRValueType())) {
    emitter.emitError(getLoc(),
                      "operand must have '!kgen.pointer<T>' type, not ")
        << exprRVal.getRValueType() << getRange();
    return {};
  }

  Value exprVal = emitter.emitSRValue({exprRVal, subExpr}, dest.getContext());
  if (!exprVal)
    return {};

  // TODO(references): if we keep these functions, they should take a origin.
  auto immortal = emitter.builder->getAttr<AnyOriginAttr>(/*isMut=*/true);
  bool startsUninit = kind == ExprNode::kGetAddressAsUninitLValue;
  bool endsUninit = kind == ExprNode::kGetAddressAsOwned;
  exprVal = emitter.builder->create<RefFromPointerOp>(
      getLocation(emitter), exprVal, immortal, startsUninit, endsUninit);

  /// __get_address_as_owned_value(ptr) # returns RValue
  if (kind == ExprNode::kGetAddressAsOwned)
    return emitter.emitResult(MRValue(exprVal), this, dest);

  // __get_address_as_uninit_lvalue(ptr) returns an MLValue
  assert(kind == kGetAddressAsUninitLValue);
  return emitter.emitResult(MLValue(exprVal), this, dest);
}

AnyValue MagicFunctionNode::emitOriginOf(ValueDest &dest,
                                         ExprEmitter &emitter) const {
  // Gather the origins of each subexpression value. If any of the origins
  // are immutable, then we mutcast the rest to immutable.
  SmallVector<TypedAttr> origins;
  for (ExprNode *subExpr : subExprs) {
    emitter.emitExpressionWithOutEvaluatingIt(
        subExpr, EC_Origin, [&](CValue result) {
          if (auto origin = emitter.extractOriginOf(subExpr, result))
            origins.push_back(origin);
        });
  }

  auto result = OriginUnionAttr::get(emitter.getContext(), origins);
  return emitter.emitResult(PValue(result), this, dest);
}

AnyValue MagicFunctionNode::emitTypeOf(ValueDest &dest,
                                       ExprEmitter &emitter) const {
  // TypeOf can reference dynamic values even when in a parameter context.
  ASTType resultType;
  emitter.emitExpressionWithOutEvaluatingIt(
      subExprs.front(), EC_Origin,
      [&](CValue result) { resultType = result.getRValueType(); });

  if (!resultType)
    return {};

  return emitter.emitResult(PValue(resultType), this, dest);
}

// There are two options. We are either emitting a type or an instance of Tuple.
// That is, either
//      (T1,T2) is sugar for Tuple[T1,T2]
// and we want to reuse 'substituteParametersIntoUserDefinedType' so
// that bindings are verified and parameter evaluation is used or
//      (exp, exp) is sugar for Tuple[typeof(expr), typeof(expr)](exp, exp)
// and we want to emit a constructor call and infer the parameter types
// of Tuple.
AnyValue TupleNode::emitIR(ValueDest &dest, ExprEmitter &emitter) const {
  ASTType tupleType =
      emitter.shared.getBuiltinTupleType(emitter.declScope, getLoc());

  // If the tuple has an inferred type from the RHS, as in `(a, b)=foo()`,
  // propagate the element types into the subexpressions if possible to enable
  // implicit var definition.
  SmallVector<ASTType> eltTypes;
  if (auto destLVType = dest.getIfLValueInitializerType()) {
    // Special case the element type of Tuple.  We could be more general than
    // this if there was a reason to, e.g. looking up a __getitem__
    // implementation.
    if (tupleType.isEqualCanon(
            destLVType.getWithoutParameters(emitter.shared))) {
      assert(destLVType.getParamBindings().size() == 1 &&
             "Tuple has one variadic parameter");
      if (auto variadicAttr =
              dyn_cast<VariadicAttr>(destLVType.getParamBindings()[0])) {
        if (variadicAttr.getValues().size() == exprs.size()) {
          for (auto typeElt : variadicAttr.getValues())
            eltTypes.push_back(ASTType(typeElt));
        }
      }
    }
  }

  bool allEltsLValue = true;
  bool allEltsTypes = true;
  SmallVector<ASTExprAnd<AnyValue>> elements;
  for (auto [i, expr] : llvm::enumerate(exprs)) {
    // Use an inferred element type if we have one.
    ValueDest eltDest(EC_TupleElement);
    if (!eltTypes.empty())
      eltDest = ValueDest(LValueInitializerType{eltTypes[i]}, EC_TupleElement);

    auto exprVal = emitter.emitExpr(expr, eltDest);
    if (!exprVal)
      return {};
    allEltsLValue &= !exprVal.getIfLValue().isNull();
    allEltsTypes &= !exprVal.getIfTypeValue().isNull();
    elements.push_back({std::move(exprVal), expr});
  }
  assert(!(allEltsTypes && allEltsLValue && !elements.empty()));

  // HACK: Tuple emission should not be context dependent.
  if (allEltsTypes && !elements.empty()) {
    SmallVector<Operand> operands;
    for (ExprNode *exprNode : exprs)
      operands.push_back(Operand(exprNode, exprNode->getLoc(),
                                 Operand::PassKind::kPositional));

    if (tupleType.isTypeCheckErrorType())
      return {};
    PValue concretizedTupleType = substituteParametersIntoUserDefinedType(
        PValue(tupleType), operands, getLoc(), exprs.front()->getRangeStart(),
        exprs.back()->getRangeEnd(), emitter);

    return emitter.emitResult(concretizedTupleType, this, dest);
  }
  // If this is a tuple with all LValue elements, return a DLValue since we
  // can assign into this expression.
  // TODO: Add support for list LValues as well.
  if (allEltsLValue) {
    SmallVector<Type> typeElts;
    for (ASTExprAnd<AnyValue> elt : elements)
      typeElts.push_back(elt.ir.getIfLValue().getRValueType());
    ASTType concretizedTupleType =
        emitter.getBuiltinTupleInstantiation(getLoc(), typeElts);
    if (!concretizedTupleType || concretizedTupleType.isTypeCheckErrorType())
      return {};
    DLValue result(
        RCRef<TupleDLValue>::create(elements, concretizedTupleType, this));
    return emitter.emitResult(result, this, dest);
  }

  // The ASTType will carry around parameters bound, we want to unbind them so
  // they can be inferred from the elements.
  tupleType = tupleType.getWithoutParameters(emitter.shared);

  // Emit a call to the builtin type constructor as an implicit conversion.
  // The type parameters are inferred from the element types.
  return emitter.emitConstructorCall(tupleType, CallOperands(elements), this,
                                     CallSyntax::kTypeCall, dest);
}
