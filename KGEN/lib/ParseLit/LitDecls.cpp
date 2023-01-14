//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Declaration parsing and name binding logic.
//
//===----------------------------------------------------------------------===//

#include "LitDecls.h"
#include "ASTDecl.h"
#include "IRValues.h"
#include "LitExprEmitter.h"
#include "LitExprNodes.h"
#include "LitLexer.h"
#include "LitParserBase.h"

#include "KGEN/CompilationOptions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LitSharedState.h"
#include "SpecialFunctions.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

namespace llvm {
// Allow (dynamic)casting ASTDecl to ParamDeclRefAttr
template <>
struct CastInfo<ParamDeclRefAttr, LIT::ASTDecl>
    : public NullableValueCastFailed<ParamDeclRefAttr>,
      public DefaultDoCastIfPossible<ParamDeclRefAttr, ASTDecl &,
                                     CastInfo<ParamDeclRefAttr, ASTDecl>> {
  static bool isPossible(ASTDecl &decl) {
    auto mValue = dyn_cast<MValue>(decl.getIRValue());
    return mValue && isa<ParamDeclRefAttr>(mValue.get());
  }
  static ParamDeclRefAttr doCast(ASTDecl &decl) {
    return cast<ParamDeclRefAttr>(cast<MValue>(decl.getIRValue()).get());
  }
};
} // namespace llvm

/// Parse an expression and immediately resolve it to a type.  This returns
/// failure on parse error.
static ParseResult parseType(LitParserBase &p, ASTType &result,
                             ASTDecl &declScope,
                             std::optional<size_t> stmtIndent) {
  ExprNode *expr = nullptr;
  if (p.parseExpression(expr, stmtIndent))
    return failure();

  ExprEmitter emitter(p.shared, declScope, std::nullopt, nullptr);
  result = emitter.emitExprType(expr);
  if (!result)
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ASTDecl
//===----------------------------------------------------------------------===//

MLIRContext *ASTDecl::getContext() const {
  if (auto *op = getIfOperation())
    return op->getContext();
  if (auto mv = dyn_cast<MValue>(getIRValue()))
    return mv.get().getContext();
  if (auto dr = dyn_cast<DRValue>(getIRValue()))
    return dr.getContext();

  return cast<LValue>(getIRValue()).getContext();
}

/// If this is an RValue, return it otherwise return null.
RValue ASTDecl::getIfRValue() const {
  // Meta value.
  if (auto attr = dyn_cast_or_null<MValue>(irValue))
    return attr;
  // DRValue.
  if (auto value = dyn_cast_or_null<DRValue>(irValue))
    return value;
  return {};
}

/// Return the SymbolRefAttr for a declaration, including all scoping that may
/// be needed, making it unique for every declaration.  This returns null for
/// named values that do not have a declaration.
SymbolRefAttr ASTDecl::getSymbolRef() const {
  auto op = dyn_cast_if_present<mlir::SymbolOpInterface>(getIfOperation());
  if (!op)
    return {};
  assert((!isa<LIT::FuncOp>(op) ||
          resolvedness >= DeclResolvedness::signatureResolved) &&
         "Functions don't have a symbol until their signatures are resolved");
  return getFullyResolvedSymbolRef(op);
}

/// Given an MLIR op for a struct declaration, return the self type.
ASTType ASTDecl::computeSelfTypeForStruct(LitSharedState &state) {
  auto structOp = cast<StructDeclOp>(*this);

  SmallVector<ParamBindAttr> parameters;
  for (auto decl : structOp.getInputParamDecls()) {
    // We're using the parameter from the type declaration scope in the
    // parameter binding list.
    TypedAttr ref = ParamDeclRefAttr::get(decl.getName(), decl.getType());
    parameters.push_back(ParamBindAttr::get(decl, ref));
  }

  // Methods on structs (but not classes) take the struct implicitly by
  // pointer so they can use and mutate it.
  return DeclRefType::get(getSymbolRef(), parameters);
}

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

// Declarations (e.g. module, class, function) are parsed in multiple phases
// to increase laziness of the parse as well as make circular references
// possible.
//
// This ensures that the forward references between peer declarations are
// handled correctly as well as circular references, for example in mutually
// recursive functions and code like this:
//
//   def foo():
//     def bar():
//       print(x)
//     x = 42
//     bar()
//   foo()

DeclResolver::DeclResolver(LitSharedState &state) : LitSharedStateUser(state) {}

DeclResolver::~DeclResolver() {
  // Run the destructors on all the ASTDecl objects to make sure any
  // transitively allocated data is released.
  for (ASTDecl *decl : parsedDeclList)
    decl->~ASTDecl();
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(DeclIRValue irValue, SMLoc loc, StringAttr name,
                               ASTDecl *parentDecl, LitLexerCursor cursor,
                               LitLexerCursor endCursor, ssize_t indentation) {
  ASTDecl *decl = shared.allocPersistent<ASTDecl>(
      irValue, loc, parentDecl, cursor, endCursor, indentation);
  parsedDeclList.push_back(decl);

  // If this is a declaration which has a TypeCheckErrorType, then all
  // references to it are invalid.
  if (auto rv = decl->getIfRValue()) {
    if (isa<TypeCheckErrorType>(rv.getType()))
      decl->hasReferenceError = true;
  } else if (auto lv = decl->getIfLValue()) {
    if (isa<TypeCheckErrorType>(lv.getRValueType().mlirType))
      decl->hasReferenceError = true;
  }

  // If this has a parent and a name, insert it into the parents name table so
  // name lookup will resolve it.  If it does, then we're done.
  if (!name)
    return *decl;

  // Remember the named decl in the symbol table so it can be looked up.
  TinyPtrVector<ASTDecl *> &entries = parentDecl->declsInScope[name];
  if (entries.empty()) {
    entries.push_back(decl);

    // If the decl is a type or alias that has a symbol, remember it.  This
    // allows us to look up decls by symbol when referenced as types.
    if (auto structDecl = dyn_cast<StructDeclOp>(*decl)) {
      // Make sure there are no name conflicts with the MLIR symbol.  If there
      // are, then addDecl will have rejected it with an error.
      shared.setResolvedDeclSymbol(structDecl);

      SymbolRefAttr symbol = decl->getSymbolRef();
      assert(!declForTypeSymbol.count(symbol) &&
             "Symbol redefinition/collision");
      declForTypeSymbol[symbol] = decl;
    }
    return *decl;
  }

  // Function support method overloading on input arguments.  Variables and
  // types cannot be overloaded because they have no inputs.  Well, we could
  // actually allow type overloading on parameters theoretically to support
  // T[4] and T[1,7] as different things, but let's no proactively add
  // complexity.
  if (isa<FuncOp>(*decl)) {
    // Verify that all previous entries are also functions.  Note that we can't
    // check the overload set is compatible with each other because the
    // signatures aren't all resolved.
    for (ASTDecl *previous : entries) {
      if (!isa<FuncOp>(*previous)) {
        auto diag = emitError(decl->getLoc(), "invalid redefinition of ")
                    << name;
        diag.attachNote(translateLocation(previous->getLoc()))
            << "cannot overload with this non-function definition";
        decl->hasReferenceError = true;
        previous->hasReferenceError = true;
        return *decl;
      }
    }

    // Otherwise, we're good, charge forwards.
    entries.push_back(decl);
    return *decl;
  }

  ASTDecl *existing = entries.back();
  auto diag = emitError(decl->getLoc(), "invalid redefinition of ") << name;
  diag.attachNote(translateLocation(existing->getLoc()))
      << "previous definition here";

  // Mark the existing decl and this one as erroneous so uses of either
  // don't create confusing errors.
  decl->hasReferenceError = true;
  for (ASTDecl *previous : entries)
    previous->hasReferenceError = true;
  return *decl;
}

void DeclResolver::aliasDecls(const TinyPtrVector<ASTDecl *> &decls,
                              StringAttr name, llvm::SMLoc aliasLoc,
                              ASTDecl &context) {
  auto [it, inserted] = context.declsInScope.try_emplace(name, decls);
  if (inserted)
    return;

  // Rejecting overlap is conservative and not what python does, but we can
  // relax this in the future when we know what the right policy should be.
  ASTDecl *existing = it->second.back();
  auto diag = emitError(aliasLoc, "invalid redefinition of ") << name;
  diag.attachNote(translateLocation(existing->getLoc()))
      << "previous definition here";

  for (ASTDecl *previous : it->second)
    previous->hasReferenceError = true;
}

void DeclResolver::importDeclsFromModule(
    ASTDecl &module, ASTDecl &context,
    ArrayRef<std::tuple<StringRef, StringRef, llvm::SMLoc>> importList) {
  if (importList.empty())
    return;

  // Make sure the body of the module has been resolved.
  if (failed(resolve(module, DeclResolvedness::fullyResolved,
                     std::get<2>(importList[0]))))
    return;

  // Process the import list.
  for (auto [sourceName, destName, loc] : importList) {
    StringAttr sourceNameAttr = StringAttr::get(getContext(), sourceName);

    // Check to see if the module has the construct we are importing.
    const TinyPtrVector<ASTDecl *> *importDecls =
        module.lookupInCurrentScope(sourceNameAttr);
    if (importDecls) {
      StringAttr destNameAttr = StringAttr::get(getContext(), destName);
      aliasDecls(*importDecls, destNameAttr, loc, context);
      continue;
    }

    // Emit an error with the module name without the leading `$` mangle.
    StringRef moduleName =
        cast<FileModuleOp>(module.getIfOperation()).getName();
    assert(moduleName.startswith("$") && "unexpected module name mangling");
    emitError(loc, "module '" + moduleName.drop_front() +
                       "' does not contain '" + sourceName + "'");

    // If we can't find the decl, recover by adding dummy decl with the dest
    // name.
    addErroneousDecl(destName, loc, &context);
  }
}

void DeclResolver::importWildCardDeclsFromModule(ASTDecl &module,
                                                 ASTDecl &context,
                                                 llvm::SMLoc loc) {
  // Make sure the body of the module has been resolved.
  if (failed(resolve(module, DeclResolvedness::fullyResolved, loc)))
    return;

  // Wildcard imports don't import decls with a leading '_'.
  for (const auto &[name, decls] : module.declsInScope)
    if (name.getValue()[0] != '_')
      aliasDecls(decls, name, loc, context);
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(Operation *op, SMLoc loc, StringAttr name,
                               ASTDecl *parentDecl, LitLexerCursor cursor,
                               LitLexerCursor endCursor, ssize_t indentation) {
  return addDecl(DeclIRValue(op), loc, name, parentDecl, cursor, endCursor,
                 indentation);
}

ASTDecl &DeclResolver::addFullyResolvedDecl(Operation *op, SMLoc loc,
                                            StringAttr name,
                                            ASTDecl *parentDecl) {
  auto &decl =
      addDecl(op, loc, name, parentDecl, LitLexerCursor(), LitLexerCursor(), 0);
  decl.resolvedness = DeclResolvedness::fullyResolved;
  return decl;
}

/// Add a declaration that is already fully resolved.
ASTDecl &DeclResolver::addFullyResolvedDecl(DeclIRValue declVal,
                                            StringAttr name, SMLoc loc,
                                            ASTDecl *parentDecl) {
  auto &decl = addDecl(declVal, loc, name, parentDecl, LitLexerCursor(),
                       LitLexerCursor(), 0);
  decl.resolvedness = DeclResolvedness::fullyResolved;
  return decl;
}

ASTDecl &DeclResolver::addFullyResolvedDecl(DeclIRValue declVal, StringRef name,
                                            llvm::SMLoc loc,
                                            ASTDecl *parentDecl) {
  return addFullyResolvedDecl(declVal, StringAttr::get(getContext(), name), loc,
                              parentDecl);
}

ASTDecl &DeclResolver::addErroneousDecl(StringRef baseName, llvm::SMLoc loc,
                                        ASTDecl *parentDecl) {
  // Use a dummy attribute representation for the error.
  BoolAttr dummyAttr = BoolAttr::get(parentDecl->getContext(), true);
  ASTDecl &errDecl =
      addFullyResolvedDecl(MValue(dummyAttr), baseName, loc, parentDecl);
  errDecl.hasReferenceError = true;
  return errDecl;
}

/// Resolve all of the declarations that are visible.
void DeclResolver::resolveAll() {
  // We can do this in any order, but choose to use the order they are
  // discovered so diagnostics are mostly top-down.  Resolving declarations
  // may cause more entries to be added to this list.
  for (size_t i = 0; i != parsedDeclList.size(); ++i) {
    (void)resolve(*parsedDeclList[i], DeclResolvedness::fullyResolved,
                  parsedDeclList[i]->getLoc());
  }
}

/// Resolve the specified declaration to at least the specified level of
/// resolution, performing incremental type checking as appropriate.
LogicalResult DeclResolver::resolve(ASTDecl &decl, DeclResolvedness howResolved,
                                    SMLoc loc) {
  // If decl is already resolved enough, we're done.
  if (decl.resolvedness >= howResolved) {
    // If decl is busted, then return failure.
    return success(!decl.hasReferenceError);
  }

  auto emitError = [&](SMLoc loc, const Twine &message) -> LitDiagnostic {
    return this->emitError(loc, message);
  };

  // If we are currently name binding this operation, we found a cycle, reject
  // it with an error.
  if (!declsCurrentlyProcessing.insert({&decl, loc}).second) {
    emitError(loc, "recursive reference to declaration")
            .attachNote(translateLocation(declsCurrentlyProcessing[&decl]))
        << "previously used here";
    decl.hasReferenceError = true;
    return failure();
  }

  // If the signature hasn't been parsed, do so.
  if (decl.resolvedness < DeclResolvedness::signatureResolved) {
    // Handle each operation that can be name bound.  We handle this by
    // restoring the lexer to the position where parsing can continue, calling
    // the `resolveSignature` method for the op, and re-saving the new cursor
    // for the next stage of resolution.
    TypeSwitch<ASTDecl &>(decl)
        .Case<LIT::FuncOp, StructDeclOp, StructFieldOp, LetDeclOp, VarDeclOp,
              ParamDeclareOp, ParamDeclRefAttr>([&](auto op) {
          LitLexer lexer(shared, decl.getCursor());

          // Resolve the signature: on a parse error, we note that the decl
          // is malformed and should not be referenced to silence downstream
          // errors.
          if (failed(resolveSignature(op, lexer, decl)))
            decl.hasReferenceError = true;
          decl.getCursor() = lexer.getCursor();
        })
        .Case<LIT::FileModuleOp, ModuleOp>([&](auto op) { /*Nothing*/ })
        .Default([&](auto &attr) {
          emitError(decl.getLoc(),
                    "do not know how to resolve the signature of this decl!");
          decl.hasReferenceError = true;
        });
    decl.resolvedness = DeclResolvedness::signatureResolved;
  }

  // If the declaration hasn't been fully parsed and we need to, do so.
  if (decl.resolvedness < DeclResolvedness::fullyResolved &&
      howResolved == DeclResolvedness::fullyResolved) {
    auto checkEndOfBodyCursor = [&](LitLexer &lexer) {
      // If the final parse of the declaration didn't match the initial
      // parse, report an error about unrecognized tokens at end of
      // declaration.
      if (!decl.isMatchingEndCursor(lexer.getCursor()) &&
          !decl.hasReferenceError) {
        if (lexer.getToken().isAny(LitToken::kw_def, LitToken::kw_struct,
                                   LitToken::kw_class, LitToken::kw_var))
          lexer.emitTokenError(
              "definition isn't on its own line at the correct "
              "indentation");
        else
          lexer.emitTokenError("unknown tokens at the end of a declaration");
      }
    };

    // Handle each operation that can be name bound.
    TypeSwitch<ASTDecl &>(decl)
        .Case<FileModuleOp, LIT::FuncOp, StructDeclOp, StructFieldOp, LetDeclOp,
              VarDeclOp, ParamDeclareOp, ParamDeclRefAttr, AliasForwardDeclOp>(
            [&](auto op) {
              // Parse the body of the declaration from the correct point.
              LitLexer lexer(shared, decl.getCursor());
              if (resolveBody(op, lexer, decl))
                return;

              checkEndOfBodyCursor(lexer);
            })
        .Case<ModuleOp>([&](auto op) { /*Nothing*/ })
        .Default([&](auto &attr) {
          emitError(decl.getLoc(),
                    "do not know how to resolve the body of this decl!");
        });
    decl.resolvedness = DeclResolvedness::fullyResolved;
  }

  declsCurrentlyProcessing.erase(&decl);
  // If decl is busted, then return failure.
  return success(!decl.hasReferenceError);
}

//===----------------------------------------------------------------------===//
// Meta signature implementation
//===----------------------------------------------------------------------===//

namespace {
/// identifier_opt_type  ::= identifier [":" expression]
/// meta_signature    ::= "[" [meta_param_list] ("->" meta_result_types)? "]"
/// meta_signature    ::= "[" "(" ")" ("->" meta_result_types)? "]"
/// meta_param_list   ::= identifier_opt_type ("," identifier_opt_type)
/// meta_result_types ::= expression ("," expression)*
struct ParsedMetaSignature {
  /// This is the function or struct that we're parsing the meta signature for.
  ASTDecl &decl;
  /// These are the parsed input parameters.
  SmallVector<ASTDecl *> parsedInputs;
  SmallVector<ExprNode *> resultTypes;

  ParsedMetaSignature(ASTDecl &decl) : decl(decl) {}

  /// If this declaration has a parameter signature, parse it and install the
  /// prototypes into the
  ParseResult parseOptionalMetaSignature(LitParserBase &p) {
    if (!p.consumeIf(LitToken::l_square) || p.consumeIf(LitToken::r_square))
      return success();

    auto &declResolver = p.getDeclResolver();

    auto parseMetaParameter = [&]() -> ParseResult {
      auto loc = p.getToken().getLoc();
      StringAttr name;
      LitLexerCursor typeStartCursor, typeEndCursor;
      ExprNode *typeExpr; // Unused, because we reparse this.
      if (p.parseIdentifier(name, "expected parameter name") ||
          p.parseToken(LitToken::colon,
                       "meta parameters always require a type") ||
          p.getCursor(typeStartCursor) ||
          p.parseExpression(typeExpr, std::nullopt) ||
          p.getCursor(typeEndCursor))
        return failure();

      // Even though we parsed the type expression, we cannot just bind it.  It
      // could have forward references to other parameters, and the declaration
      // we're parsing into isn't fully resolved yet.  Instead, add the decls
      // with unresolved values.
      auto tmpDecl =
          ParamDeclRefAttr::get(name, UnresolvedType::get(p.getContext()));
      ASTDecl &paramDecl = declResolver.addDecl(
          MValue(tmpDecl), loc, name, &decl, typeStartCursor, typeEndCursor, 0);
      parsedInputs.push_back(&paramDecl);
      return success();
    };

    // Parse the meta parameters.  We either have () or a parameter list.
    if (p.consumeIf(LitToken::l_paren)) {
      if (p.parseToken(LitToken::r_paren,
                       "expected ')' in empty parameter list; try dropping the "
                       "'(' if you have parameters"))
        return failure();
    } else {
      // Parse an actual parameter list.
      if (p.parseCommaSeparatedList(
              parseMetaParameter,
              {LitToken::r_square, LitToken::minus_greater}))
        return failure();
    }

    // Parse the meta results if present.
    if (p.consumeIf(LitToken::minus_greater)) {
      auto parseResultType = [&]() -> ParseResult {
        return p.parseExpression(resultTypes.emplace_back(nullptr),
                                 std::nullopt);
      };
      if (p.parseCommaSeparatedList(parseResultType, LitToken::r_square))
        return failure();
    }
    return p.parseToken(LitToken::r_square, "expected ']' for parameter list");
  }

  /// Given a parsed parameter signature, resolve the types of each of them,
  /// which can of course be recursively referenced.
  SmallVector<ParamDeclAttr>
  getResolvedInputParamDecls(DeclResolver &resolver) {
    SmallVector<ParamDeclAttr> result;
    // Force resolve all of the declarations, which could be recursive w.r.t.
    // each other.

    // Mark the decl container as 'fully resolved' temporarily to facilitate
    // this, so it doesn't attempt to get resolved again.
    // FIXME(5975): This is a hack and shouldn't be needed.  The problem is that
    // parameters should be accessible before the body is, and we have no way to
    // express this currently.
    assert(decl.resolvedness == DeclResolvedness::unparsed);
    llvm::SaveAndRestore X(decl.resolvedness, DeclResolvedness::fullyResolved);

    for (ASTDecl *paramDecl : parsedInputs) {
      (void)resolver.resolve(*paramDecl, DeclResolvedness::fullyResolved,
                             paramDecl->getLoc());
      auto resolvedParam =
          cast<ParamDeclRefAttr>(cast<MValue>(paramDecl->getIRValue()).get());
      result.push_back(
          ParamDeclAttr::get(resolvedParam.getName(), resolvedParam.getType()));
    }

    return result;
  }

  SmallVector<Type> getResolvedResultTypes(ExprEmitter &emitter) const {
    SmallVector<Type> results;
    for (ExprNode *expr : resultTypes) {
      auto type = emitter.emitExprType(expr);
      if (!type)
        type = emitter.shared.getTypeCheckErrorType();
      results.push_back(type);
    }
    return results;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Meta Parameter Decl implementation
//===----------------------------------------------------------------------===//

LogicalResult DeclResolver::resolveSignature(ParamDeclRefAttr paramDeclRef,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);

  ASTType type;
  if (parseType(p, type, *decl.getParentDecl(), std::nullopt))
    return failure(); // Should never happen, we already checked this.

  // Update the value to the newly resolved type.
  decl.irValue = MValue(ParamDeclRefAttr::get(paramDeclRef.getName(), type));
  return success(!isa<TypeCheckErrorType>(type.mlirType));
}

ParseResult DeclResolver::resolveBody(ParamDeclRefAttr op, LitLexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

//===----------------------------------------------------------------------===//
// Decorator support logic
//===----------------------------------------------------------------------===//

static SmallVector<ExprNode *> parseDecorators(ASTDecl &decl,
                                               LitParserBase &p) {
  SmallVector<ExprNode *> result;
  while (p.consumeIf(LitToken::at)) {
    ExprNode *decoratorExpr;
    if (p.parseExpression(decoratorExpr,
                          decl.getParentDecl()->getIndentation()))
      break;
    result.push_back(decoratorExpr);
  }
  return result;
}

static void rejectDecorators(ArrayRef<ExprNode *> decoratorExprs, ASTDecl &decl,
                             LitSharedState &shared) {
  if (!decoratorExprs.empty())
    shared.emitError(decoratorExprs[0]->getLoc(),
                     "decorators not supported on this statement")
        << LitSourceRange(decoratorExprs.front()->getRangeStart(),
                          decoratorExprs.back()->getRangeEnd());
}

//===----------------------------------------------------------------------===//
// Function Decl implementation
//===----------------------------------------------------------------------===//

namespace {
/// Parsing support for a function argument:
///
/// value_param_list   ::= value_param ("," value_param)*
/// value_param        ::= value_parammarker identifier_opt_type
///                        value_parampostfix ["=" expression]
/// value_parammarker  ::= "/" | "*" | "**"
/// value_parampostfix ::= "&"
///
struct ParsedArgument {
  SMLoc loc;
  // Specify argument passing convention, e.g. byval/byref etc.
  ValueInputConvention convention = ValueInputConvention::ByVal;
  StringAttr name;
  ExprNode *typeExpr;
  ExprNode *initValue = nullptr;

  ParseResult parse(LitParserBase &p, ASTDecl &declScope) {
    // TODO: Implement support for variadic parameter markers:
    // Python's parameter grammar embeds checking for `/` and `*` and `**` into
    // the grammar, we can just check for it using ad-hoc logic for simplicity,
    // according to the following rules:
    //   1) Only one /, *, and ** parameter may exist in the parameter list.
    //   2) They are specified in that order.
    //   3) These do not permit default arguments.
    loc = p.getToken().getLoc();

    if (p.parseIdentifier(name, "expected parameter name"))
      // TODO: Scan ahead for better recovery.
      return failure();

    // Handle & for by-ref arguments.
    if (p.consumeIf(LitToken::amp))
      convention = ValueInputConvention::ByRef;

    if (p.consumeIf(LitToken::colon)) {
      if (p.parseExpression(typeExpr, std::nullopt))
        return failure();
    }
    if (p.consumeIf(LitToken::equal)) {
      if (p.parseExpression(initValue, std::nullopt))
        return failure();
    }
    return success();
  };
};
} // namespace

/// If this is a special function like __init__ return the enum that
/// identifies it, otherwise return kNormal.
SpecialFunctionKind SpecialFunctionInfo::getKind(StringRef name) {
  if (name.size() < 5 || !name.startswith("__") || !name.endswith("__"))
    return SpecialFunctionKind::kNormal;

#define SF(ENUM, NAME, NUMOPERANDS, EXPRNODE, FLAGS)                           \
  if (name == NAME)                                                            \
    return SpecialFunctionKind::ENUM;
#include "SpecialFunctions.def"

  // Otherwise, this declaration isn't known.
  return SpecialFunctionKind::kNormal;
}

/// If this is a special function like __init__ return the enum that
/// identifies it, otherwise return kNormal.
const SpecialFunctionInfo &SpecialFunctionInfo::get(SpecialFunctionKind kind) {
  static const SpecialFunctionInfo infos[] = {
      {nullptr, SpecialFunctionKind::kNormal, /*numOperands=*/-1, /*flags=*/0},
#define SF(ENUM, NAME, NUMOPERANDS, EXPRNODE, FLAGS)                           \
  {NAME, SpecialFunctionKind::ENUM, (NUMOPERANDS), (FLAGS)},
#include "SpecialFunctions.def"
  };

  assert(unsigned(kind) < sizeof(infos) / sizeof(infos[0]));
  return infos[unsigned(kind)];
}

/// Now that all the structural properties are determined, perform any
/// name-binding specific checks over the declaration.  This happens after
/// decorator processing because that is how defs work in Python.  This also
/// fills in any implicitly declared types, performs name mangling, and sets up
/// the signature correctly.
///
/// This allows magic behavior (like __new__ being static, checking of method
/// self requirements and enforcement of other invariants.
///
/// This returns failure (after emitting an error) when a type checking problem
/// is detected.
static void verifyFunctionNameBinding(ASTDecl &decl, LIT::FuncOp funcOp,
                                      StringAttr &name,
                                      SmallVector<ParsedArgument> &args,
                                      MutableArrayRef<Type> argTypes,
                                      ASTType &resultType,
                                      LitSharedState &shared) {
  SpecialFunctionInfo fnInfo = SpecialFunctionInfo::get(name);

  // On any semantic error we mark the declaration erroneous - so references to
  // it don't type check, and we clear our special function information.  This
  // reduces cascade errors.
  auto emitErrorLoc = [&](SMLoc loc, const Twine &message) {
    fnInfo = SpecialFunctionInfo();
    decl.hasReferenceError = true;
    return shared.emitError(loc, message);
  };
  auto emitError = [&](const Twine &message) {
    fnInfo = SpecialFunctionInfo();
    decl.hasReferenceError = true;
    return shared.emitError(funcOp.getLoc(), message);
  };

  // Fill in any missing arguments or diagnose missing ones in fn's.
  bool seenInitValue = false;
  for (auto [arg, type] : llvm::zip(args, argTypes)) {
    if (!type) {
      if (funcOp.getIsDef()) {
        // If we are in a 'def', we infer object type for Python compatibility.
        type = shared.lookupObjectType(arg.loc, *decl.getParentDecl());
        if (!type)
          type = shared.getTypeCheckErrorType();
      } else {
        // In an 'fn' we report an error.
        emitErrorLoc(arg.loc, "'fn' parameter type must be specified")
            << LitSourceRange(arg.loc, arg.loc);
        type = shared.getTypeCheckErrorType();
      }
    }
    if (arg.initValue) {
      seenInitValue = true;
    } else if (seenInitValue) {
      shared.emitError(arg.loc, "non-default argument follows default argument")
          << arg.typeExpr->getRange();
    }
  }

  // If this definition is a struct/class member, compute the self type.
  ASTType selfType;
  if (auto *parentDecl = decl.getParentDecl())
    if (isa<StructDeclOp>(*parentDecl)) {
      //  The parent decl must be fully resolved in order to resolve any members
      //  of it.
      assert(parentDecl->resolvedness == DeclResolvedness::fullyResolved);
      selfType = parentDecl->getSelfType();
    }

  // Check any special function information.

  // __new__ and similar methods are implicitly static.
  if (fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod)
    funcOp.setIsStatic(true);

  // Check that the 'self' argument of a method was specified correctly.
  if (selfType && !funcOp.getIsStatic()) {
    if (argTypes.empty()) {
      // TODO: We can/should relax this for 'def' declarations in the future,
      // they should be able to implicit ignore arguments like Python does.
      emitError("self argument must be present in instance method");
    } else if (!ASTType(argTypes[0]).isEqualCanon(selfType)) {
      auto diag = emitErrorLoc(args[0].loc, "'self' argument must have type ")
                  << selfType << " but actually has type "
                  << ASTType(argTypes[0]);
      if (args[0].typeExpr)
        diag << args[0].typeExpr->getRange();
    }
  }

  if (funcOp.getIsStatic() && !selfType) {
    emitError("only methods on structs may be declared static");
    funcOp.setIsStatic(false);
  }

  // Verify the operand count lines up.
  if (fnInfo.numOperands != -1 && size_t(fnInfo.numOperands) != args.size()) {
    size_t numOperands = fnInfo.numOperands;
    emitError("special function must have ")
        << numOperands << " operand" << plural(numOperands);
  }

  // Check other invariants based on method flags.
  if (fnInfo.flags & SpecialFunctionInfo::kInstMethod) {
    auto convent = fnInfo.isByRefSelfInstMethod() ? ValueInputConvention::ByRef
                                                  : ValueInputConvention::ByVal;
    if (!selfType)
      emitError("special function must be a method");
    else if (funcOp.getIsStatic())
      emitError("special method may not be a static method");
    else if (convent != args[0].convention)
      emitErrorLoc(args[0].loc, "self argument must ")
          << (convent == ValueInputConvention::ByRef ? "" : "not ")
          << "be passed by reference";
  }

  switch (fnInfo.kind) {
  default:
    // Ignore methods without special handling.
    break;
  case SpecialFunctionKind::kInit:
    if (isa<StructDeclOp>(*decl.getParentDecl())) {
      emitError("__init__ is not allowed on structs, use __new__ instead");
      // __init__ on classes must return NoneType.
    } else if (!resultType.isEqualCanon(shared.getNoneType())) {
      emitError("__init__ result type must be elided (or None)");
    }
    break;

  case SpecialFunctionKind::kNew:
    // __new__ must return containing type.
    // TODO: We could allow omitting result type and default it.
    if (!resultType.isEqualCanon(selfType))
      emitError("result type must be ") << selfType;
    break;
  }

  // Mangle 'name', ensuring that overloaded methods get unique symbol names.
  SmallString<64> mangledName(name.getValue().begin(), name.getValue().end());
  mangledName += '(';
  llvm::interleave(
      llvm::zip(args, argTypes),
      [&](auto argAndArgType) {
        auto [arg, argType] = argAndArgType;
        mangledName += ASTType(argType).getAsString();
        if (arg.convention == ValueInputConvention::ByRef)
          mangledName += '&';
      },
      [&]() { mangledName += ","; });
  mangledName += ')';

  name = StringAttr::get(funcOp.getContext(), mangledName);

  // Finally, after all semantic checks are done, update the types to reflect
  // ABI information form the calling convention.

  // Now that all the types and signature information have been resolved,
  // compute the final MLIR types, mixing in conventions etc.
  for (auto [arg, argType] : llvm::zip(args, argTypes)) {
    if (arg.convention == ValueInputConvention::ByRef)
      argType = POP::PointerType::get(argType);
  }

  // If the method can raise an exception, wrap the result type in a variant
  // with the error type.
  if (funcOp.getRaises()) {
    auto errorOr = shared.lookupErrorOrType(resultType, decl.getLoc(),
                                            *decl.getParentDecl());
    // If we couldn't find an Error type then recover by pretending we didn't
    // raise.
    if (!errorOr) {
      funcOp.setRaises(false);
      decl.hasReferenceError = true;
    } else {
      resultType = errorOr;
    }
  }
}

namespace {
struct FnDecorators : public LitSharedStateUser {
  FnDecorators(ASTDecl &decl, LitSharedState &shared)
      : LitSharedStateUser(shared), decl(decl), funcOp(cast<LIT::FuncOp>(decl)),
        isMethod(isa<StructDeclOp>(*decl.getParentDecl())) {}

  void apply(SmallVector<ExprNode *> &decoratorExprs);
  void applyLate(SymbolRefAttr symbolName,
                 SmallVector<ExprNode *> &decoratorExprs);

private:
  void applyInterface(const DeclRefNode &node);
  void applyRaises(const DeclRefNode &node);
  void applyImplements(const CallNode &callNode);
  void applyEvaluator(const CallNode &callNode);
  void applyLateExport(SymbolRefAttr symbolName);

  ASTDecl &decl;
  LIT::FuncOp funcOp;
  const bool isMethod;
};
} // namespace

void FnDecorators::applyInterface(const DeclRefNode &node) {
  if (isMethod) {
    emitError(node.getLoc(), "interfaces cannot be nested inside a struct")
        << node.getRange();
    return;
  }

  if (funcOp.getImplementsAttr())
    emitError(node.getLoc(), "interfaces cannot implement other interfaces")
        << node.getRange();

  funcOp.setIsInterface(true);
}

void FnDecorators::applyRaises(const DeclRefNode &node) {
  if (funcOp.getIsDef()) {
    emitError(node.getLoc(), "methods defined with 'def' always raise")
        << node.getRange();
    return;
  }

  funcOp.setRaises(true);
}

// @implements interface.
void FnDecorators::applyImplements(const CallNode &node) {
  if (funcOp.getImplementsAttr()) {
    emitError(node.getLoc(), "only one @implements decorator is allowed")
        << node.getRange();
    return;
  }

  if (node.args.size() != 1 || !isa<DeclRefNode>(node.args.front())) {
    emitError(node.getLoc(),
              "@implements decorator must specify one interface by name")
        << node.getParenRange();
    return;
  }

  // Perform a name lookup to find the right symbol.
  const DeclRefNode &nameNode = *cast<DeclRefNode>(node.args.front());
  StringRef interfaceName = nameNode.spelling;
  auto result = shared.lookupAndResolveDecl(interfaceName, node.getLoc(), decl,
                                            /*searchParentScopes=*/true);

  // Reject the code if the interface wasn't found.
  ArrayRef<ASTDecl *> resultDecls = result.getIfSuccess();
  if (resultDecls.empty()) {
    if (result.isFailure())
      emitError(node.getLoc(), "unable to resolve interface named '")
          << interfaceName << "'" << nameNode.getRange();
    return;
  }

  // Reject implementation of overloaded interface.
  // TODO: Use signature matching to pick the right overload.
  if (resultDecls.size() > 1) {
    auto diag =
        emitError(node.getLoc(),
                  "TODO: cannot (yet!) implement overloaded interface '")
        << interfaceName << "'" << nameNode.getRange();
    return;
  }
  auto interfaceDecl = resultDecls[0];

  // Okay, if we found an interface we're implementing, check that it makes
  // sense.
  auto funcInterface =
      dyn_cast_or_null<LIT::FuncOp>(interfaceDecl->getIfOperation());
  if (!funcInterface || !funcInterface.getIsInterface()) {
    auto diag = emitError(node.getLoc(), "'")
                << interfaceName << "' is not a kgen interface"
                << nameNode.getRange();
    diag.attachNote(translateLocation(interfaceDecl->getLoc()))
        << "'" << interfaceName << "' declared here";
    return;
  }

  // FIXME: This needs to type check the signature here, not defer to
  // lowering.  This also needs to resolve the interface.
  funcOp.setImplementsAttr(interfaceDecl->getSymbolRef());
}

// @evaluator interface.
void FnDecorators::applyEvaluator(const CallNode &node) {
  if (funcOp.getEvaluatorAttr()) {
    emitError(node.getLoc(), "only one @evaluator decorator is allowed")
        << node.getRange();
    return;
  }

  if (node.args.size() != 1 || !isa<DeclRefNode>(node.args.front())) {
    emitError(node.getLoc(),
              "@evaluator decorator must specify one function by name")
        << node.getRange();
    return;
  }

  // Perform a name lookup to find the right symbol.
  DeclRefNode &nameNode = *cast<DeclRefNode>(node.args.front());
  StringRef evaluatorName = nameNode.spelling;
  auto result = shared.lookupAndResolveDecl(evaluatorName, node.getLoc(), decl,
                                            /*searchParentScopes=*/true);

  // Reject the code if no function was found.
  ArrayRef<ASTDecl *> resultDecls = result.getIfSuccess();
  if (resultDecls.empty()) {
    if (result.isFailure())
      emitError(node.getLoc(), "unable to resolve function named '")
          << evaluatorName << "'" << nameNode.getRange();
    return;
  }

  // Reject implementation of overloaded function.
  // TODO: Use signature matching to pick the right overload.
  if (resultDecls.size() > 1) {
    emitError(node.getLoc(), "cannot (yet!) implement overloaded functions '")
        << evaluatorName << "'" << nameNode.getRange();
    return;
  }
  auto funcDecl = resultDecls[0];

  auto evaluatorFuncOp =
      dyn_cast_or_null<LIT::FuncOp>(funcDecl->getIfOperation());
  if (!evaluatorFuncOp) {
    auto diag = emitError(node.getLoc(), "'")
                << evaluatorName << "' is not a valid function"
                << nameNode.getRange();
    diag.attachNote(translateLocation(funcDecl->getLoc()))
        << '\'' << evaluatorName << "' declared here";
    return;
  }

  if (!funcOp.getIsInterface())
    emitError(node.getLoc(), "only interfaces can have an evaluator");
  SignatureType signature = evaluatorFuncOp.getSignature();
  auto evaluatorAttr =
      SymbolConstantAttr::get(funcDecl->getSymbolRef(), signature);
  funcOp.setEvaluatorAttr(evaluatorAttr);
}

// Apply all signature decorators.
void FnDecorators::apply(SmallVector<ExprNode *> &decoratorExprs) {
  SmallVector<ExprNode *> unprocessed;
  for (ExprNode *decorator : decoratorExprs) {
    bool processedIt = false;

    // Process all the decorators we know about.
    if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
      processedIt = true;
      if (declRef->spelling == "staticmethod")
        funcOp.setIsStatic(true);
      else if (declRef->spelling == "interface")
        applyInterface(*declRef);
      else if (declRef->spelling == "raises")
        applyRaises(*declRef);
      else if (declRef->spelling == "always_inline")
        funcOp.setAlwaysInline(true);
      else if (declRef->spelling == "nodebug_inline")
        funcOp.setNoDebugInline(true);
      else
        processedIt = false;
    }

    // `x()` forms.
    if (auto callNode = dyn_cast<CallNode>(decorator)) {
      if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
        processedIt = true;
        if (declRef->spelling == "implements")
          applyImplements(*callNode);
        else if (declRef->spelling == "evaluator")
          applyEvaluator(*callNode);
        else
          processedIt = false;
      }
    }

    if (!processedIt)
      unprocessed.push_back(decorator);
  }
  decoratorExprs = unprocessed;
}

void FnDecorators::applyLateExport(SymbolRefAttr symbolName) {
  if (isMethod) {
    emitError(funcOp.getLoc(), "methods cannot be exported");
    return;
  }

  ASTDecl *containingDecl = decl.getParentDecl();
  auto builder = containingDecl->getDeclEndBuilder();
  builder.create<LIT::ExportOp>(funcOp.getLoc(),
                                builder.getArrayAttr(symbolName));
}

void FnDecorators::applyLate(SymbolRefAttr symbolName,
                             SmallVector<ExprNode *> &decoratorExprs) {
  // Scan through and process decorator expressions that are in the late pass.
  for (ExprNode *decorator : decoratorExprs) {
    // Process all the decorators we know about.
    if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
      if (declRef->spelling == "export") {
        applyLateExport(symbolName);
        continue;
      }

      emitError(decorator->getLoc(), "unsupported decorator: ")
          << declRef->spelling << declRef->getRange();
      continue;
    }
    emitError(decorator->getLoc(), "unsupported decorator")
        << decorator->getRange();
  }
}

/// Given a FuncOp that had its decorator processed, compute the FnEffects.
static FnEffects computeFnEffects(LIT::FuncOp funcOp) {
  FnEffects effects = FnEffects::None;
  if (funcOp.getAlwaysInline())
    effects = effects | FnEffects::ForceInline;
  if (funcOp.getRaises())
    effects = effects | FnEffects::Throws;
  return effects;
}

/// funcdef ::=  [decorators] "def" identifier [meta_signature]
///              "(" [value_param_list] ")" ["->" expression] ":" suite
///
LogicalResult DeclResolver::resolveSignature(LIT::FuncOp funcOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);
  SmallVector<ExprNode *> decoratorExprs = parseDecorators(decl, p);
  assert(p.getToken().isAny(LitToken::kw_def, LitToken::kw_fn) &&
         "not a function definition?");
  p.consumeToken();

  StringAttr baseName;
  if (p.parseIdentifier(baseName, "expected function name"))
    return failure();

  // Add meta parameters from an enclosing declaration to the symbol table.
  // These are /in/ our current scope because we do not want name conflicts with
  // them and they are instance (not type-level) values.
  // TODO: Generalize this to support nested structs and functions.
  if (auto structDecl = dyn_cast<StructDeclOp>(*decl.getParentDecl())) {
    auto parentLoc = decl.getParentDecl()->getLoc();
    for (auto param : structDecl.getInputParamDecls()) {
      auto paramRef = ParamDeclRefAttr::get(param.getName(), param.getType());
      addFullyResolvedDecl(MValue(paramRef), param.getName(), parentLoc, &decl);
    }
  }

  // Parse declared meta parameters and add them to the current scope.
  ParsedMetaSignature metaSignature(decl);
  SmallVector<ParsedArgument> args;

  if (metaSignature.parseOptionalMetaSignature(p) ||
      p.parseToken(LitToken::l_paren, "expected '(' for parameter list"))
    return failure();

  // Add the meta parameters to the symbol table, and resolve their types.  We
  // add all of these after generic signature parsing so types used in the
  // signature list resolve to enclosing scopes, and we add them before the
  // value signature list so the types and parameters can resolve to the bound
  // values.
  if (!p.consumeIf(LitToken::r_paren)) {
    if (p.parseCommaSeparatedList(
            [&]() {
              return args.emplace_back(ParsedArgument()).parse(p, decl);
            },
            LitToken::r_paren) ||
        p.parseToken(LitToken::r_paren, "expected ')' for parameter list"))
      return failure();
  }

  // Parse the result type if present.
  ExprNode *resultTypeExpr = nullptr;
  if (p.consumeIf(LitToken::minus_greater)) {
    if (p.parseExpression(resultTypeExpr, std::nullopt))
      return failure();
  }
  if (p.parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  // Now that the full signature has been parsed, resolve the meta signature,
  // arguments types, and result type.
  SmallVector<ParamDeclAttr> inputParamDecls =
      metaSignature.getResolvedInputParamDecls(*this);

  // Resolve the result parameter types now that the arguments are in scope.
  ExprEmitter typeEmitter(shared, decl, std::nullopt, nullptr);
  SmallVector<Type> resultParamTypes =
      metaSignature.getResolvedResultTypes(typeEmitter);

  // Resolve the result type and any argument types that are present, leaving
  // any unspecified types null.
  SmallVector<Type> argTypes;
  for (auto &arg : args) {
    // This returns a TypeCheckErrorType on error, no extra check is needed.
    ASTType type;
    if (arg.typeExpr) {
      type = typeEmitter.emitExprType(arg.typeExpr);

      // If the type couldn't be emitted, mark this function erroneous and put
      // in a placeholder type so we can continue type checking.
      if (!type) {
        decl.hasReferenceError = true;
        type = shared.getTypeCheckErrorType();
      }

    } else if (arg.name == "self" && isa<StructDeclOp>(*decl.getParentDecl())) {
      // If this is a 'self' argument in a fn that is a method, default to a
      // self type.  TODO: Should we do this, or default to object in a 'def'?
      assert(decl.getParentDecl()->resolvedness ==
             DeclResolvedness::fullyResolved);
      type = decl.getParentDecl()->getSelfType();
    }
    argTypes.push_back(type);
  }

  ASTType resultType;
  if (!resultTypeExpr) {
    // TODO: We shouldn't default this to none for 'def's.  This should default
    // to object type.  Our return checker is currently a lame duck.
    resultType = shared.getNoneType();
  } else {
    resultType = typeEmitter.emitExprType(resultTypeExpr);
    // On error, a diagnostic will be emitted, but we don't want to kill the
    // entire function definition.  We won't be able to correctly type check any
    // calls to this function though.
    if (!resultType) {
      resultType = shared.getTypeCheckErrorType();
      decl.hasReferenceError = true;
    }
  }

  // Now that we have figured out the lexical structure, allow decorators to
  // take a crack at the signature.
  // Okay, apply them now.
  FnDecorators(decl, shared).apply(decoratorExprs);

  // Now that all the structural properties are determined, perform any
  // name-binding specific checks over the declaration.  This happens after
  // decorator processing because that is how defs work in Python.  This also
  // fills in any implicitly declared types.
  StringAttr name = baseName;
  verifyFunctionNameBinding(decl, funcOp, name, args, argTypes, resultType,
                            shared);

  // Finally now that the full signature has been resolved, build our IR.

  // Set the symbol to the mangled name and check for redefinition.
  funcOp.setName(name);

  // Remove the temporary "sym_namex" attribute set up in FuncOp::build, see
  // that method for an explanation.
  funcOp->removeAttr("sym_namex");

  if (Operation *existing = shared.setResolvedDeclSymbol(funcOp)) {
    // On redefinition this is an overload of the same name and same signature.
    auto diag = p.emitError(funcOp.getLoc(), "redefinition of function ")
                << name << " with identical signature";
    diag.attachNote(existing->getLoc()) << "previous definition here";
    decl.hasReferenceError = true;
  }

  // TODO: Handle the export attribute somehow else.  It should be a 'body
  // decorator' that is handled after the decl is fully resolved.
  SymbolRefAttr symbolName = getFullyResolvedSymbolRef(funcOp);
  FnDecorators(decl, shared).applyLate(symbolName, decoratorExprs);

  // Generate a debug subprogram for this function.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (auto &diBuilder = shared.diBuilder) {
    FileLineColLoc fileLineCol =
        funcOp.getLoc()->findInstanceOf<FileLineColLoc>();

    // Compute the subprogram flags.
    /// If we have any optimizations, mark the subprogram as optimized.
    DebugInfo::SubprogramFlags spFlags =
        shared.options.optimizationLevel ? DebugInfo::SubprogramFlags::Optimized
                                         : DebugInfo::SubprogramFlags::None;
    /// If the function has a body, treat it as a definition.
    if (!funcOp.isExternal())
      spFlags = spFlags | DebugInfo::SubprogramFlags::Definition;

    // Use unresolved types now for simplicity, these will get resolved during
    // compilation.
    auto mapUnresolvedType = [](Type type) -> DebugInfo::DIType {
      return DebugInfo::DIUnresolvedMLIRType::get(type);
    };
    auto type = DebugInfo::DISubroutineType::get(
        getContext(),
        llvm::to_vector(llvm::map_range(argTypes, mapUnresolvedType)),
        mapUnresolvedType(resultType.mlirType));
    diScopeGuard = diBuilder->pushSubprogram(
        baseName, name, diBuilder->createFile(fileLineCol),
        fileLineCol.getLine(), fileLineCol.getLine(), spFlags, type);
    funcOp->setLoc(diBuilder->createScopedLoc(fileLineCol));
  }

  // Handle function effects.
  SmallVector<Location> argLocs;
  SmallVector<StringAttr> argNames;
  SmallVector<ValueInputConvention> inputConventions;
  for (const ParsedArgument &arg : args) {
    argLocs.push_back(p.translateLocation(arg.loc));
    argNames.push_back(arg.name);
    inputConventions.push_back(arg.convention);
  }

  OpBuilder builder = decl.getDeclEndBuilder();
  funcOp.setValueParamNamesAttr(builder.getAttr<StringArrayAttr>(argNames));
  funcOp.setSignature(SignatureType::get(
      builder.getAttr<ParamDeclArrayAttr>(inputParamDecls),
      builder.getAttr<TypeArrayAttr>(resultParamTypes),
      builder.getFunctionType(argTypes, {resultType.mlirType}),
      builder.getAttr<ConventionsAttr>(inputConventions,
                                       computeFnEffects(funcOp))));
  funcOp.getBody()->addArguments(argTypes, argLocs);

  // Interfaces don't have anything else to do.
  if (funcOp.getIsInterface())
    return success();

  // Functor used to build the debug info for an argument.
  auto buildArgDIInfo = [&](Value argVal, StringRef name, unsigned argIdx) {
    auto &diBuilder = shared.diBuilder;
    if (!diBuilder ||
        shared.options.debugLevel != CompilationOptions::kFullDebugInfo)
      return;
    auto bbArgLoc = argVal.getLoc()->findInstanceOf<FileLineColLoc>();

    auto varAttr = diBuilder->createLocalVariable(
        name, diBuilder->createFile(bbArgLoc), bbArgLoc.getLine(), argIdx + 1,
        /*alignInBits=*/0,
        DebugInfo::DIUnresolvedMLIRType::get(argVal.getType()));
    builder.create<DebugInfo::ValueOp>(argVal.getLoc(), argVal, varAttr);
  };

  // Set up the body of the def, creating declarations for the value
  // parameters and adding them to the symbol table.
  for (auto [bbArg, parsedArg] :
       llvm::zip(funcOp.getBody()->getArguments(), args)) {
    // Arguments passed by-reference can be directly used.
    if (parsedArg.convention == ValueInputConvention::ByRef) {
      buildArgDIInfo(bbArg, parsedArg.name, bbArg.getArgNumber());
      addFullyResolvedDecl(LValue(bbArg), parsedArg.name, parsedArg.loc, &decl);
      continue;
    }
    assert(parsedArg.convention == ValueInputConvention::ByVal &&
           "Unknown convention");

    // If this was passed by-value, then it becomes an rvalue in a `fn`.
    if (!funcOp.getIsDef()) {
      buildArgDIInfo(bbArg, parsedArg.name, bbArg.getArgNumber());
      addFullyResolvedDecl(DRValue(bbArg), parsedArg.name, parsedArg.loc,
                           &decl);
      continue;
    }

    // In a `def`, we create a mutable var.decl lvalue to allow reassignment.
    auto type = POP::PointerType::get(bbArg.getType());
    auto varDecl =
        builder.create<VarDeclOp>(bbArg.getLoc(), type, parsedArg.name);
    addFullyResolvedDecl(varDecl, parsedArg.loc, parsedArg.name, &decl);
    builder.create<POP::StoreOp>(bbArg.getLoc(), bbArg, varDecl,
                                 /*alignment=*/std::nullopt);
  }
  return success();
}

/// Return true if the result type is nominally a none type.
static bool isNoneResultType(LIT::FuncOp defOp) {
  Type type = defOp.getResultType();
  if (defOp.getConventions().getFnEffects() == FnEffects::Throws)
    type = cast<POP::VariantType>(type).getType(1);
  return isa<LIT::NoneType>(type);
}

/// Once a @nodebug_inline has been fully parsed and the body is complete, we
/// check it to see if it is simple enough for inlining.  We intentionally limit
// it to try to keep this to purely functional stuff that will fold when
// inlined.
static void verifyNoDebugInline(LIT::FuncOp funcOp, LitSharedState &shared) {
  size_t numOps = 0;

  auto rejectFunc = [&](const Twine &badThing) -> LitDiagnostic {
    funcOp.setNoDebugInline(false);
    return shared.emitError(funcOp.getLoc(),
                            "@nodebug_inline does not allow " + badThing);
  };

  // We don't allow anything other than by-value arguments right now.
  if (!funcOp.getConventions().isDefault()) {
    rejectFunc("byref arguments or effects");
    return;
  }

  // We don't allow parameters.  TODO: Relax this.
  if (!funcOp.getInputParamDecls().empty() ||
      !funcOp.getResultParamTypes().empty()) {
    rejectFunc("input or result parameters");
    return;
  }

  for (Operation &op : *funcOp.getBody()) {
    auto reject = [&](const Twine &badThing) {
      rejectFunc(badThing).attachNote(op.getLoc()) << "operation defined here";
    };

    // Let decls are folded/dropped during inlining so they are free, these
    // other ops are glue that don't compute anything and generally get folded,
    // so we treat them as free so abstraction doesn't get in the way of
    // inlining.
    if (isa<LetDeclOp, DebugInfo::ValueOp>(op) ||
        isa<ReturnOp, StructExtractOp, StructCreateOp>(op) ||
        // Constants aren't computation and can often be dropped as well.
        (op.getNumOperands() == 0 && op.getNumResults() == 1 &&
         op.hasTrait<OpTrait::ConstantLike>()))
      continue;

    // Disallow large function bodies.  We only want this to be used for small
    // constructs.
    if (++numOps == 4)
      return reject("large function body with " +
                    Twine(std::distance(funcOp.getBody()->begin(),
                                        funcOp.getBody()->end())) +
                    " ops");

    // We have a disallow-list for specific things we don't want to support.
    // The goal here is to allow simple leaf functions that fold when inlined.
    if (isa<VarDeclOp>(op))
      return reject("var declarations");
    if (auto callOp = dyn_cast<CallOp>(op))
      return reject("function call to '" +
                    Twine(callOp.getCalleeSymbol().getLeafReference()) + "'");
    if (isa<CallParamOp, POP::CallIndirectOp>(op))
      return reject("indirect function calls");
    if (isa<TryRaiseOp>(op))
      return reject("control flow");
    if (!KGEN::getParamDecls(&op).empty())
      return reject("parameter declarations");
    if (op.getNumRegions())
      return reject("operations with regions");
  }
}

ParseResult DeclResolver::resolveBody(LIT::FuncOp funcOp, LitLexer &lexer,
                                      ASTDecl &decl) {
  // Push the debug scope for this function if necessary so that nested
  // operations have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (auto spAttr = DebugInfo::extractScope(funcOp))
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  // Resolve the body of the decl.
  if (LitParserBase::parseSuite(decl, lexer))
    return failure();

  // Check to see if we have a kgen.return at the end of function.  If not,
  // complain or add one implicitly if we have no results.
  Block *bodyBlock = funcOp.getBody();
  bool isInterface = funcOp.getIsInterface();

  if (isInterface) {
    if (!bodyBlock->empty())
      emitError(funcOp.getLoc(), "interfaces must have no body");
    // Drop the body block so the function becomes external.
    bodyBlock->erase();

    if (funcOp.getNoDebugInline()) {
      emitError(funcOp.getLoc(), "interfaces may not be @nodebug_inline");
      funcOp.setNoDebugInline(false);
    }
    return success();
  }

  // Check for a return op at the end of the function.
  // TODO: This should really be moved to a dataflow pass after the parser.
  if (bodyBlock->empty() || !isa<ReturnOp>(bodyBlock->back())) {
    auto loc = funcOp.getLoc();
    if (isNoneResultType(funcOp) && funcOp.getResultParamTypes().empty()) {
      auto b = OpBuilder::atBlockEnd(bodyBlock);
      Value noneVal =
          b.create<ParamConstantOp>(loc, NoneAttr::get(getContext()));
      if (funcOp.getConventions().getFnEffects() == FnEffects::Throws)
        noneVal = b.create<POP::VariantCreateOp>(loc, funcOp.getResultType(),
                                                 noneVal);
      b.create<ReturnOp>(loc, ArrayRef<TypedAttr>(), noneVal);
    } else if (!shared.diags.isErrorEmitted()) {
      Location endLoc = bodyBlock->empty() ? loc : bodyBlock->back().getLoc();
      emitError(endLoc, "return expected at end of 'def' with results");
    }
  }

  // Check that any alias forward declarations have been completed.
  if (!shared.diags.isErrorEmitted()) {
    bodyBlock->walk([&](AliasForwardDeclOp aliasFwdDeclOp) {
      // If the location for the resultParam was never set then this forward
      // declaration was never defined.
      if (!aliasFwdDeclOp.getResultParamLoc().has_value()) {
        emitError(aliasFwdDeclOp.getLoc(), "alias ")
            << aliasFwdDeclOp.getNameAttr()
            << " was never defined by a result parameter";
      }
    });
  }

  // If this is a nodebug_inline function, verify its invariants.
  if (funcOp.getNoDebugInline())
    verifyNoDebugInline(funcOp, shared);

  return success();
}

//===----------------------------------------------------------------------===//
// Module Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(LIT::FileModuleOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  // Push a scope for the file of this module.
  DebugInfo::DIBuilder::ScopeGuard fileGuard;
  if (shared.diBuilder) {
    auto &sourceMgr = lexer.getSourceMgr();
    int fileId = sourceMgr.FindBufferContainingLoc(lexer.getToken().getLoc());
    if (fileId) {
      StringRef filename =
          sourceMgr.getMemoryBuffer(fileId)->getBufferIdentifier();
      fileGuard = shared.diBuilder->pushFile(filename, "/");
    }
  }

  return LitParserBase::parseSuite(decl, lexer);
}

//===----------------------------------------------------------------------===//
// LetDecl / VarDecl implementation
//===----------------------------------------------------------------------===//

namespace {
struct ParsedLetVarDecl {
  SmallVector<ExprNode *> decorators;
  ASTType type;
  ExprNode *initValue = nullptr;

  ParseResult parse(LitLexer &lexer, ASTDecl &decl);
  std::pair<DRValue, OpBuilder> emitInitValue(Operation *declOp, ASTDecl &decl,
                                              LitSharedState &shared);
};
} // namespace

/// Parse the structure of a let/var declaration.
ParseResult ParsedLetVarDecl::parse(LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);
  decorators = parseDecorators(decl, p);

  p.consumeToken(); // eat the let/var.
  if (p.parseToken(LitToken::identifier,
                   "internal error: checked by stmt parser"))
    return failure();

  //  Parse the type if present.
  if (p.consumeIf(LitToken::colon)) {
    if (parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
      return failure();
  }

  // Parse and emit the initializer if present.
  if (p.consumeIf(LitToken::equal)) {
    if (p.parseExpression(initValue, decl.getIndentation()))
      return failure();
  }
  return success();
}

/// Emit the initializer at hte specified point and convert it to the declared
/// type if known.
std::pair<DRValue, OpBuilder>
ParsedLetVarDecl::emitInitValue(Operation *declOp, ASTDecl &decl,
                                LitSharedState &shared) {
  // We insert after var decl, but before let decl.
  auto iterator = Block::iterator(declOp);
  if (isa<VarDeclOp>(declOp))
    ++iterator;
  OpBuilder builder(declOp->getBlock(), iterator);
  ExprEmitter emitter(shared, *decl.getParentDecl(), builder,
                      /*varDeclCursor*/ nullptr);

  auto value = emitter.emitExprDRValue(initValue);
  if (!value)
    return {value, builder};

  // If we had a declared type, coerce the expression value to it.
  if (type) {
    const char *kind = isa<LetDeclOp>(declOp) ? "let" : "var";
    value = emitter.emitDRValue(
        {emitter.getAsExpectedType(value, initValue, type,
                                   " in " + Twine(kind) + " declaration"),
         initValue});
  } else {
    // Infer the type if we lack a declared type (`var x = 42`).
    // TODO(literal autopromotion).
    type = value.getType();
  }
  return {value, builder};
}

/// let_decl_stmt ::= "let" identifier ":" expression ["=" expression]
///                 | "let" identifier "=" expression
LogicalResult DeclResolver::resolveSignature(LetDeclOp letOp, LitLexer &lexer,
                                             ASTDecl &decl) {
  ParsedLetVarDecl parsed;
  if (parsed.parse(lexer, decl))
    return failure();

  // Handle the initializer if present.
  if (parsed.initValue) {
    auto [initVal, _] = parsed.emitInitValue(letOp, decl, shared);
    if (!initVal)
      return failure();

    letOp->setOperands(initVal);
  } else {
    // Reject let's without an initializer.
    // TODO: Use definitive initialization to allow late initialization, e.g.:
    //   let x : Int
    //   if cond:
    //     x = foo()
    //   else
    //     y = bar()
    // If there was neither a type or initializer, reject the var.
    emitError(letOp.getLoc(), "'let' declaration must have an initializer");
    return failure();
  }

  letOp.getResult().setType(parsed.type);
  rejectDecorators(parsed.decorators, decl, shared);
  return success();
}

ParseResult DeclResolver::resolveBody(LetDeclOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

/// var_decl_stmt ::= "var" identifier ":" expression ["=" expression]
///                 | "var" identifier "=" expression
LogicalResult DeclResolver::resolveSignature(VarDeclOp varOp, LitLexer &lexer,
                                             ASTDecl &decl) {
  ParsedLetVarDecl parsed;
  if (parsed.parse(lexer, decl))
    return failure();

  // Handle the initializer if present.
  if (parsed.initValue) {
    auto [initVal, builder] = parsed.emitInitValue(varOp, decl, shared);
    if (!initVal)
      return failure();

    // Store the initializer value into the VarDecl.
    auto loc = translateLocation(parsed.initValue->getLoc());
    builder.create<POP::StoreOp>(loc, initVal, varOp,
                                 /*alignment=*/std::nullopt);
  }

  if (parsed.type)
    varOp.getResult().setType(POP::PointerType::get(parsed.type));
  else {
    // If there was neither a type or initializer, reject the var.
    emitError(varOp.getLoc(),
              "declaration must have either a type or an initializer");
    return failure();
  }

  rejectDecorators(parsed.decorators, decl, shared);
  return success();
}

ParseResult DeclResolver::resolveBody(VarDeclOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

//===----------------------------------------------------------------------===//
// Alias Decl implementation
//===----------------------------------------------------------------------===//

/// alias_decl_stmt ::= "alias" identifier ":" expression ["=" expression]
///                   | "alias" identifier "=" expression
///
LogicalResult DeclResolver::resolveSignature(ParamDeclareOp paramDeclOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);
  SmallVector<ExprNode *> decoratorExprs = parseDecorators(decl, p);

  // Parse the type if present.
  if (p.parseToken(LitToken::kw_alias,
                   "internal error: checked by stmt parser") ||
      p.parseToken(LitToken::identifier,
                   "internal error: checked by stmt parser"))
    return failure();

  ASTType type;
  if (p.consumeIf(LitToken::colon)) {
    if (parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
      return failure();
  }

  // Handle the case where there is no initializer.
  if (!p.consumeIf(LitToken::equal)) {
    // If there was neither a type or initializer, reject the var.
    if (!type) {
      p.emitError(paramDeclOp.getLoc(),
                  "declaration must have either a type or an initializer");
      return failure();
    }

    // `alias x: Int` is a forward declaration of a return parameter from a
    // function call, so it must occur in a function.
    if (!isa<LIT::FuncOp>(paramDeclOp->getParentOp())) {
      p.emitError(paramDeclOp.getLoc(),
                  "parameter results may only be declared in a function");
      return failure();
    }

    // Ok, things seem set up right, replace the ParamDeclOp with the right
    // operation that will allow us to track things.
    OpBuilder builder(paramDeclOp);
    Operation *forwardDecl = builder.create<AliasForwardDeclOp>(
        paramDeclOp.getLoc(), paramDeclOp.getName(), TypeAttr::get(type),
        mlir::LocationAttr());
    decl.setIRValue(forwardDecl);

    // Remove the paramDeclOp from the IR, since we ended up changing our mind
    // about how to represent this.
    paramDeclOp->erase();

    // The check that the alias was specified is handled when the function body
    // has been fully resolved.
    rejectDecorators(decoratorExprs, decl, shared);
    return success();
  }

  // Otherwise this is a normal `alias` declaration with an initializer.
  ExprNode *initValue = nullptr;
  if (p.parseExpression(initValue, decl.getIndentation()))
    return failure();

  ASTDecl &parentDecl = *decl.getParentDecl();
  ExprEmitter emitter(shared, parentDecl, /*builder*/ {},
                      /*varDeclCursor*/ nullptr);

  // Emit the value and convert to the expected type if we know it.
  auto rhsValue =
      emitter.emitExprMValue(initValue, type, " in alias declaration");
  if (!rhsValue)
    return failure();

  // If we had no declared type (`alias x = 42`), infer the type from the
  // initializer.
  if (!type)
    type = rhsValue.getType();

  // Remember the value, and update the type from UnresolvedType.
  paramDeclOp.setValueAttr(rhsValue.get());
  paramDeclOp.setParamDecl(ParamDeclAttr::get(paramDeclOp.getName(), type));
  rejectDecorators(decoratorExprs, decl, shared);
  return success();
}

ParseResult DeclResolver::resolveBody(ParamDeclareOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

ParseResult DeclResolver::resolveBody(AliasForwardDeclOp aliasFwdDeclOp,
                                      LitLexer &lexer, ASTDecl &decl) {
  return success();
}

//===----------------------------------------------------------------------===//
// Struct Decl implementation
//===----------------------------------------------------------------------===//

/// structdef ::=
///   [decorators] "struct" identifier [meta_signature] ":" suite
///
LogicalResult DeclResolver::resolveSignature(StructDeclOp structOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);
  SmallVector<ExprNode *> decoratorExprs = parseDecorators(decl, p);

  ParsedMetaSignature metaSignature(decl);
  if (p.parseToken(LitToken::kw_struct,
                   "internal error: checked by stmt parser") ||
      p.parseToken(LitToken::identifier,
                   "internal error: checked by stmt parser") ||
      metaSignature.parseOptionalMetaSignature(p) ||
      p.parseToken(LitToken::colon, "expected ':' in struct definition"))
    return failure();

  // Resolve the meta parameters and get their decls.
  SmallVector<ParamDeclAttr> inputParamDecls =
      metaSignature.getResolvedInputParamDecls(*this);
  structOp.setInputParamDecls(inputParamDecls);

  // Reject result parameters.
  if (!metaSignature.resultTypes.empty())
    emitError(metaSignature.resultTypes[0]->getLoc(),
              "struct declarations do not support result parameters")
        << metaSignature.resultTypes[0]->getRange();

  // This is a struct, so we can use 'computeSelfTypeForStruct' to figure out
  // the self type.
  decl.setSelfType(decl.computeSelfTypeForStruct(shared));
  rejectDecorators(decoratorExprs, decl, shared);
  return success();
}

ParseResult DeclResolver::resolveBody(StructDeclOp structOp, LitLexer &lexer,
                                      ASTDecl &decl) {
  return LitParserBase::parseSuite(decl, lexer);
}

//===----------------------------------------------------------------------===//
// StructFieldDecl implementation
//===----------------------------------------------------------------------===//

/// struct_field_decl_stmt ::= "var" identifier ":" expression
/// TODO: Support default values?
///
LogicalResult DeclResolver::resolveSignature(StructFieldOp fieldOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);
  SmallVector<ExprNode *> decoratorExprs = parseDecorators(decl, p);

  ASTType type;
  // Parse the type if present.
  if (p.parseToken(LitToken::kw_var,
                   "internal error: checked by stmt parser") ||
      p.parseToken(LitToken::identifier,
                   "internal error: checked by stmt parser") ||
      p.parseToken(LitToken::colon,
                   "struct field declaration must have a type") ||
      parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
    return failure();

  fieldOp.setType(type);
  rejectDecorators(decoratorExprs, decl, shared);
  return success();
}

ParseResult DeclResolver::resolveBody(StructFieldOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  return success();
}
