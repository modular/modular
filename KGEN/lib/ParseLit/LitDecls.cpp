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
                             ASTDecl &declScope, Optional<size_t> stmtIndent) {
  ExprNode *expr = nullptr;
  if (p.parseExpression(expr, stmtIndent))
    return failure();
  result =
      ExprEmitter(p.getSharedState(), declScope, None, nullptr).emitType(expr);
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
  auto *op = getIfOperation();
  if (!op)
    return {};

  if (auto structOp = dyn_cast<StructDeclOp>(op)) {
    // TODO: Support nested/local structs.
    return FlatSymbolRefAttr::get(structOp.getNameAttr());
  }

  if (auto fnOp = dyn_cast<LIT::FuncOp>(op)) {
    // TODO: Support multiple levels of nesting.  This should be recursive, and
    // SymbolRefAttr should support a get(FlatSymbol, SymbolRef) helper that
    // forms a properly flattened reference by unwinding the RHS if it isn't
    // flat.
    SymbolRefAttr symbolRef = FlatSymbolRefAttr::get(fnOp.getNameAttr());
    if (auto parentStruct = dyn_cast<StructDeclOp>(*getParentDecl()))
      symbolRef = SymbolRefAttr::get(parentStruct.getNameAttr(),
                                     cast<FlatSymbolRefAttr>(symbolRef));
    return symbolRef;
  }

  return {};
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
DeclResolver::DeclResolver(LitSharedState &state) : sharedState(state) {}
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
  ASTDecl *decl = sharedState.allocPersistent<ASTDecl>(
      irValue, loc, parentDecl, cursor, endCursor, indentation);
  parsedDeclList.push_back(decl);

  // If this has a parent and a name, insert it into the parents name table so
  // name lookup will resolve it.
  if (!parentDecl || !name) {
    assert(!decl->getSymbolRef() && "Can't have symbol without a name");
    return *decl;
  }

  // Remember the named decl in the symbol table so it can be looked up.
  auto [it, inserted] = parentDecl->declsInScope.insert({name, decl});
  if (inserted) {
    // If the decl is a type or alias that has a symbol, remember it.  This
    // allows us to look up decls by symbol when referenced as types.
    if (SymbolRefAttr symbol = decl->getSymbolRef()) {
      if (isa<StructDeclOp>(*decl) || isa<ParamDeclareOp>(*decl)) {
        assert(!declForTypeSymbol.count(symbol) &&
               "Symbol redefinition/collision");
        declForTypeSymbol[symbol] = decl;
      }
    }
  } else {
    ASTDecl *existing = it->second;
    auto diag =
        sharedState.emitError(decl->getLoc(), "invalid redefinition of ")
        << name;
    diag.attachNote(sharedState.translateLocation(existing->getLoc()))
        << "previous definition here";

    // Mark the existing decl and this one as erroneous so uses of either
    // don't create confusing errors.
    decl->hasReferenceError = true;
    existing->hasReferenceError = true;
  }

  return *decl;
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

  auto emitError = [&](SMLoc loc, const Twine &message) -> InFlightDiagnostic {
    return mlir::emitError(sharedState.translateLocation(loc), message);
  };

  // If we are currently name binding this operation, we found a cycle, reject
  // it with an error.
  if (!declsCurrentlyProcessing.insert({&decl, loc}).second) {
    emitError(loc, "recursive reference to declaration")
            .attachNote(
                sharedState.translateLocation(declsCurrentlyProcessing[&decl]))
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
        .Case<LIT::FuncOp, StructDeclOp, StructFieldOp, VarDeclOp,
              ParamDeclareOp, ParamDeclRefAttr>([&](auto op) {
          LitLexer lexer(sharedState, decl.getCursor());

          // Resolve the signature: on a parse error, we note that the decl
          // is malformed and should not be referenced to silence downstream
          // errors.
          if (failed(resolveSignature(op, lexer, decl)))
            decl.hasReferenceError = true;
          decl.getCursor() = lexer.getCursor();
        })
        .Case([&](ModuleOp op) { /*Nothing*/ })
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
    // Handle each operation that can be name bound.
    TypeSwitch<ASTDecl &>(decl)
        .Case<LIT::FuncOp, StructDeclOp, StructFieldOp, VarDeclOp,
              ParamDeclareOp, ParamDeclRefAttr>([&](auto op) {
          // Parse the body of the declaration from the correct point.
          LitLexer lexer(sharedState, decl.getCursor());
          if (resolveBody(op, lexer, decl))
            return;

          // If the final parse of the declaration didn't match the initial
          // parse, report an error about unrecognized tokens at end of
          // declaration.
          if (!decl.isMatchingEndCursor(lexer.getCursor()) &&
              !decl.hasReferenceError) {
            if (lexer.getToken().isAny(LitToken::kw_def, LitToken::kw_struct,
                                       LitToken::kw_class, LitToken::kw_var))
              lexer.emitError("definition isn't on its own line at the correct "
                              "indentation");
            else
              lexer.emitError("unknown tokens at the end of a declaration");
          }
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
/// meta_signature    ::= "[" [meta_param_list] "]"
/// meta_param_list   ::= identifier_opt_type ("," identifier_opt_type)
struct ParsedMetaSignature {
  /// This is the function or struct that we're parsing the meta signature for.
  ASTDecl &decl;
  /// These are the parsed input parameters.
  SmallVector<ASTDecl *> parsedInputs;

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
          p.getCursor(typeStartCursor) || p.parseExpression(typeExpr, None) ||
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

    if (p.parseCommaSeparatedList(parseMetaParameter, LitToken::r_square) ||
        p.parseToken(LitToken::r_square, "expected ']' for parameter list"))
      return failure();
    return success();
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
};
} // namespace

//===----------------------------------------------------------------------===//
// Meta Parameter Decl implementation
//===----------------------------------------------------------------------===//

LogicalResult DeclResolver::resolveSignature(ParamDeclRefAttr paramDeclRef,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);
  ExprNode *typeExpr;
  if (p.parseExpression(typeExpr, None))
    return failure(); // Should never happen, we already checked this.

  // Emit the type.
  ExprEmitter emitter(sharedState, *decl.getParentDecl(), /*builder*/ {},
                      /*varDeclCursor*/ nullptr);
  // This always succeeds, reporting an error and returning erroneous on
  // failure.
  Type type = emitter.emitType(typeExpr);

  // Update the value to the newly resolved type.
  decl.irValue = MValue(ParamDeclRefAttr::get(paramDeclRef.getName(), type));
  return success(!isa<TypeCheckErrorType>(type));
}

ParseResult DeclResolver::resolveBody(ParamDeclRefAttr paramDeclRef,
                                      LitLexer &lexer, ASTDecl &decl) {
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
                             LitParserBase &p) {
  if (!decoratorExprs.empty())
    p.emitError(decoratorExprs[0]->getLoc(),
                "decorators not supported on this statement");
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
      if (p.parseExpression(typeExpr, None))
        return failure();
    }
    if (p.consumeIf(LitToken::equal)) {
      if (p.parseExpression(initValue, None))
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
  for (auto [arg, type] : llvm::zip(args, argTypes)) {
    if (!type) {
      if (funcOp.getIsDef()) {
        // If we are in a 'def', we infer object type for Python compatibility.
        type = shared.getObjectType();
      } else {
        // In an 'fn' we report an error.
        emitErrorLoc(arg.loc, "'fn' parameter type must be specified");
        type = shared.getTypeCheckErrorType();
      }
    }
    // TODO: add support for default parameter expressions.
    if (arg.initValue)
      shared.emitError(arg.loc, "TODO: No default values yet");
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
  if (fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod) {
    // Verify that implicitly static methods are declared as methods.
    if (!selfType) {
      emitError("special function must be a method");

    } else {
      funcOp.setIsStaticAttr(mlir::UnitAttr::get(shared.getContext()));
    }
  }

  // Check that the 'self' argument of a method was specified correctly.
  if (selfType && !funcOp.getIsStatic()) {
    if (argTypes.empty()) {
      // TODO: We can/should relax this for 'def' declarations in the future,
      // they should be able to implicit ignore arguments like Python does.
      emitError("self argument must be present in instance method");
    } else if (!ASTType(argTypes[0]).isEqualCanon(selfType)) {
      emitErrorLoc(args[0].loc, "'self' argument must have type ")
          << selfType << " but actually has type " << ASTType(argTypes[0]);
    }
  }

  // Verify the operand count lines up.
  if (fnInfo.numOperands != -1 && size_t(fnInfo.numOperands) != args.size()) {
    auto numOperands = fnInfo.numOperands;
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

  // TODO: Mangle 'name'.

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
  if (funcOp.getRaises())
    resultType = shared.getErrorOrType(resultType);
}

namespace {
struct FnDecorators {
  /// This is set to true by @staticmethod.
  bool isStatic = false;
  /// This is set to true by @interface.
  bool isInterface = false;
  /// This is set to true by @raises.
  bool raises = false;
  /// This is set by @implementedInterface(x).
  FlatSymbolRefAttr implementedInterface;
  // This is set to true by @export.
  bool isExported = false;

  void processDecorator(ExprNode *decorator, LitParserBase &parser);
  void applyEarly(ASTDecl &decl, LitSharedState &shared);
  void applyLate(ASTDecl &decl, LitSharedState &shared);
};
} // namespace

// Process a function decorator.
void FnDecorators::processDecorator(ExprNode *decorator,
                                    LitParserBase &parser) {
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "staticmethod")
      isStatic = true;
    else if (declRef->spelling == "interface")
      isInterface = true;
    else if (declRef->spelling == "export")
      isExported = true;
    else if (declRef->spelling == "raises")
      raises = true;
    else
      parser.emitError(decorator->getLoc(), "unsupported decorator: ")
          << declRef->spelling;
    return;
  }

  // `x()` forms.
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    auto declRef = dyn_cast<DeclRefNode>(callNode->callee);
    if (!declRef || declRef->spelling != "implements") {
      parser.emitError(decorator->getLoc(), "unsupported decorator");
      return;
    }
    if (callNode->args.size() != 1 ||
        !isa<DeclRefNode>(callNode->args.front())) {
      parser.emitError(
          decorator->getLoc(),
          "@implements decorator must specify one interface by name");
      return;
    }
    if (implementedInterface) {
      parser.emitError(decorator->getLoc(),
                       "only one @implements decorator is allowed");
      return;
    }

    // FIXME: This is incorrect. This should do name lookup on the specified
    // name and use getSymbolRef() on the returned ASTDecl, rather than forming
    // it directly.  There is no reason to force interfaces to be top-level.
    StringRef interfaceName =
        cast<DeclRefNode>(callNode->args.front())->spelling;
    implementedInterface =
        FlatSymbolRefAttr::get(parser.getContext(), interfaceName);
    return;
  }

  parser.emitError(decorator->getLoc(), "unsupported decorator");
}

void FnDecorators::applyEarly(ASTDecl &decl, LitSharedState &shared) {
  LIT::FuncOp funcOp = cast<LIT::FuncOp>(decl);
  Location loc = funcOp.getLoc();

  // Is this a method?
  bool isMethod = isa<StructDeclOp>(*decl.getParentDecl());
  if (isStatic && !isMethod) {
    shared.emitError(loc, "only methods on structs may be declared static");
    isStatic = false;
  }

  if (isInterface && implementedInterface)
    shared.emitError(loc, "interfaces cannot implement other interfaces");

  if (isInterface && isMethod)
    shared.emitError(loc, "interfaces cannot be nested inside a struct");

  if (isInterface)
    funcOp.setIsInterface(true);
  if (implementedInterface)
    funcOp.setImplementsAttr(implementedInterface);
  if (isStatic)
    funcOp.setIsStatic(true);

  // Remember if this was declared as a 'def' or 'fn' because this affects
  // certain downstream behavior.
  if (funcOp.getIsDef() && raises)
    shared.emitError(loc, "methods defined with 'def' always raise");

  if (funcOp.getIsDef() || raises)
    funcOp.setRaises(true);
}

/// This method is used to apply decorators to the function after it is
/// resolved.
void FnDecorators::applyLate(ASTDecl &decl, LitSharedState &shared) {
  LIT::FuncOp funcOp = cast<LIT::FuncOp>(decl);
  Location loc = funcOp.getLoc();
  bool isMethod = isa<StructDeclOp>(*decl.getParentDecl());

  if (isExported) {
    if (isMethod)
      shared.emitError(loc, "methods cannot be exported");
    else {
      ASTDecl *containingDecl = decl.getParentDecl();
      containingDecl->getDeclEndBuilder().create<ExportOp>(
          loc, ArrayAttr::get(containingDecl->getContext(),
                              {FlatSymbolRefAttr::get(funcOp.getNameAttr())}));
    }
  }
}

/// funcdef ::=  [decorators] "def" identifier [meta_signature]
///              "(" [value_param_list] ")" ["->" expression] ":" suite
///
LogicalResult DeclResolver::resolveSignature(LIT::FuncOp funcOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);
  SmallVector<ExprNode *> decoratorExprs = parseDecorators(decl, p);

  // Okay, apply them now.
  FnDecorators decorators;
  for (auto *expr : decoratorExprs)
    decorators.processDecorator(expr, p);

  assert(p.getToken().isAny(LitToken::kw_def, LitToken::kw_fn) &&
         "not a function definition?");
  p.consumeToken();

  StringAttr name;
  if (p.parseIdentifier(name, "expected function name"))
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
    if (p.parseExpression(resultTypeExpr, None))
      return failure();
  }
  if (p.parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  // Now that the full signature has been parsed, resolve the meta signature,
  // arguments types, and result type.
  SmallVector<ParamDeclAttr> inputParamDecls =
      metaSignature.getResolvedInputParamDecls(*this);

  // Resolve the result type and any argument types that are present, leaving
  // any unspecified types null.
  SmallVector<Type> argTypes;
  ExprEmitter typeEmitter(sharedState, decl, None, nullptr);
  for (auto &arg : args) {
    // This returns a TypeCheckErrorType on error, no extra check is needed.
    ASTType type =
        arg.typeExpr ? typeEmitter.emitType(arg.typeExpr) : ASTType();

    // If this is a 'self' argument in a fn that is a method, default to a self
    // type.  TODO: Should we do this, or default to object in a 'def'?
    if (!type && arg.name == "self" &&
        isa<StructDeclOp>(*decl.getParentDecl())) {
      assert(decl.getParentDecl()->resolvedness ==
             DeclResolvedness::fullyResolved);
      type = decl.getParentDecl()->getSelfType();
    }
    argTypes.push_back(type);
  }

  ASTType resultType =
      resultTypeExpr ? typeEmitter.emitType(resultTypeExpr) : ASTType();
  if (!resultType) {
    // TODO: We shouldn't default this to none for 'def's.  This should default
    // to object type.  Our return checker is currently a lame duck.
    resultType = sharedState.getNoneType();
  }

  // Now that we have figured out the lexical structure, allow decorators to
  // take a crack at the signature.
  decorators.applyEarly(decl, sharedState);

  // Now that all the structural properties are determined, perform any
  // name-binding specific checks over the declaration.  This happens after
  // decorator processing because that is how defs work in Python.  This also
  // fills in any implicitly declared types.
  verifyFunctionNameBinding(decl, funcOp, name, args, argTypes, resultType,
                            sharedState);

  // Finally now that the full signature has been resolved, build our IR.

  // Set the symbol to the mangled name.
  funcOp.setName(name);

  // TODO: Handle the export attribute somehow else.
  decorators.applyLate(decl, sharedState);

  // Generate a debug subprogram for this function.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (auto &diBuilder = sharedState.diBuilder) {
    FileLineColLoc fileLineCol =
        funcOp.getLoc()->findInstanceOf<FileLineColLoc>();

    // Compute the subprogram flags.
    /// If we have any optimizations, mark the subprogram as optimized.
    DebugInfo::SubprogramFlags spFlags =
        sharedState.options.optimizationLevel
            ? DebugInfo::SubprogramFlags::Optimized
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

    // TODO: When we have mangled names, we'll want to the base name here.
    StringRef name = funcOp.getName();
    diScopeGuard = diBuilder->pushSubprogram(
        name, name, diBuilder->createFile(fileLineCol), fileLineCol.getLine(),
        fileLineCol.getLine(), spFlags, type);
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
      builder.getAttr<TypeArrayAttr>(
          /*TODO: result params*/ ArrayRef<Type>()),
      builder.getFunctionType(argTypes, {resultType.mlirType}),
      builder.getAttr<ConventionsAttr>(inputConventions,
                                       funcOp.getRaises() ? FnEffects::Throws
                                                          : FnEffects::None)));
  funcOp.getBody()->addArguments(argTypes, argLocs);

  if (FlatSymbolRefAttr implementsAttr = funcOp.getImplementsAttr()) {
    StringRef interfaceName = implementsAttr.getAttr().getValue();
    if (ASTDecl *interfaceDecl = decl.lookup(implementsAttr.getAttr())) {
      if (auto funcInterface =
              dyn_cast_or_null<LIT::FuncOp>(interfaceDecl->getIfOperation());
          !funcInterface || !funcInterface.getIsInterface())
        p.emitError(funcOp->getLoc(), "not an interface: ") << interfaceName;

      // FIXME: This needs to type check the signature here, not defer to
      // lowering.  This also needs to resolve the interface.
    } else {
      p.emitError(funcOp->getLoc(),
                  "this function implements an unknown interface: ")
          << interfaceName;
      funcOp.setImplements(llvm::None);
    }
  }

  // Interfaces don't have anything else to do.
  if (funcOp.getIsInterface())
    return success();

  // Functor used to build the debug info for an argument.
  auto buildArgDIInfo = [&](Value argVal, StringRef name, unsigned argIdx) {
    auto &diBuilder = sharedState.diBuilder;
    if (!diBuilder ||
        sharedState.options.debugLevel != CompilationOptions::kFullDebugInfo)
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
                                 /*alignment*/ None);
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

ParseResult DeclResolver::resolveBody(LIT::FuncOp defOp, LitLexer &lexer,
                                      ASTDecl &decl) {
  // Push the debug scope for this function if necessary so that nested
  // operations have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (auto spAttr = DebugInfo::extractScope(defOp))
    diScopeGuard = sharedState.diBuilder->pushScopeGuard(spAttr);

  // Resolve the body of the decl.
  if (LitParserBase::parseSuite(decl, lexer))
    return failure();

  // Check to see if we have a kgen.return at the end of function.  If not,
  // complain or add one implicitly if we have no results.
  Block *bodyBlock = defOp.getBody();
  bool isInterface = defOp.getIsInterface();

  if (isInterface) {
    if (!bodyBlock->empty())
      emitError(defOp.getLoc(), "interfaces must have no body");
    // Drop the body block so the function becomes external.
    bodyBlock->erase();
    return success();
  }

  if (bodyBlock->empty() || !isa<ReturnOp>(bodyBlock->back())) {
    auto loc = defOp.getLoc();
    if (isNoneResultType(defOp) && defOp.getResultParamTypes().empty()) {
      auto b = OpBuilder::atBlockEnd(bodyBlock);

      Value noneVal =
          b.create<ParamConstantOp>(loc, NoneAttr::get(getContext()));
      if (defOp.getConventions().getFnEffects() == FnEffects::Throws)
        noneVal =
            b.create<POP::VariantCreateOp>(loc, defOp.getResultType(), noneVal);
      b.create<ReturnOp>(loc, ArrayRef<TypedAttr>(), noneVal);
    } else if (!sharedState.errorOccurred) {
      Location endLoc = bodyBlock->empty() ? loc : bodyBlock->back().getLoc();
      emitError(endLoc, "return expected at end of 'def' with results");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Variable Decl implementation
//===----------------------------------------------------------------------===//

/// var_decl_stmt ::= "var" identifier ":" expression ["=" expression]
///                 | "var" identifier "=" expression
///
LogicalResult DeclResolver::resolveSignature(VarDeclOp varOp, LitLexer &lexer,
                                             ASTDecl &decl) {
  LitParserBase p(lexer);
  SmallVector<ExprNode *> decoratorExprs = parseDecorators(decl, p);

  ASTType type;
  // Parse the type if present.
  if (p.parseToken(LitToken::kw_var,
                   "internal error: checked by stmt parser") ||
      p.parseToken(LitToken::identifier,
                   "internal error: checked by stmt parser") ||
      p.consumeIf(LitToken::colon)) {
    if (parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
      return failure();
    varOp.getResult().setType(POP::PointerType::get(type));
  }

  // Parse the initializer if present.
  ExprNode *initValue = nullptr;
  if (p.consumeIf(LitToken::equal)) {
    if (p.parseExpression(initValue, decl.getIndentation()))
      return failure();

    ASTDecl &parentDecl = *decl.getParentDecl();
    OpBuilder builder(varOp->getBlock(), ++Block::iterator(varOp));
    ExprEmitter emitter(sharedState, parentDecl, builder,
                        /*varDeclCursor*/ nullptr);

    // TODO: If the initializer is a parameter value/constant, we can install it
    // directly on the var decl instead of doing a store.  This will work better
    // in structs etc.
    auto rhsValue = emitter.emitDRValue(initValue);
    if (!rhsValue)
      return failure();

    // If we had a declared type, coerce the expression value to it.
    // TODO(implicit conversions etc).
    if (type && !type.isEqualCanon(rhsValue.getType())) {
      p.emitError(initValue->getLoc(), "initializer has type ")
          << ASTType(rhsValue.getType()) << " but declared type is " << type;
      return failure(); // Not sure which type is right.
    }

    // Infer the type if we lack a declared type (`var x = 42`)
    if (!type) {
      type = rhsValue.getType();
      varOp.getResult().setType(POP::PointerType::get(type));
    }

    // The types line up, do a store.
    auto loc = sharedState.translateLocation(initValue->getLoc());
    builder.create<POP::StoreOp>(loc, rhsValue, varOp,
                                 /*alignment*/ None);
  }

  // If there was neither a type or initializer, reject the var.
  if (!type) {
    p.emitError(varOp.getLoc(),
                "declaration must have either a type or an initializer");
    return failure();
  }

  rejectDecorators(decoratorExprs, decl, p);
  return success();
}

ParseResult DeclResolver::resolveBody(VarDeclOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  // Nothing to do for a var decl, we parse everything as part of its
  // signature. We could move to parsing an initializer expression lazily when
  // a type is present if there were a reason to do that (e.g. more laziness
  // desired) in the future.
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

  ASTType type;
  // Parse the type if present.
  if (p.parseToken(LitToken::kw_alias,
                   "internal error: checked by stmt parser") ||
      p.parseToken(LitToken::identifier,
                   "internal error: checked by stmt parser"))
    return failure();

  if (p.consumeIf(LitToken::colon)) {
    if (parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
      return failure();
  }

  // Parse the initializer if present.
  ExprNode *initValue = nullptr;
  if (p.consumeIf(LitToken::equal)) {
    if (p.parseExpression(initValue, decl.getIndentation()))
      return failure();

    ASTDecl &parentDecl = *decl.getParentDecl();
    ExprEmitter emitter(sharedState, parentDecl, /*builder*/ {},
                        /*varDeclCursor*/ nullptr);

    auto rhsValue =
        emitter.emitMValue(initValue, "expected meta parameter value");
    if (!rhsValue)
      return failure();

    // If we had a declared type, coerce the expression value to it.
    // TODO(implicit conversions etc).
    if (!type) {
      // Infer the type since we lack a declared type (`var x = 42`)
      type = rhsValue.getType();
    } else if (!type.isEqualCanon(rhsValue.getType())) {
      p.emitError(initValue->getLoc(), "initializer has type ")
          << ASTType(rhsValue.getType()) << " but declared type is " << type;
      return failure();
    }

    // Remember the value.
    paramDeclOp.setValueAttr(rhsValue.get());
  } else {
    // If there was neither a type or initializer, reject the var.
    if (!type) {
      p.emitError(paramDeclOp.getLoc(),
                  "declaration must have either a type or an initializer");
      return failure();
    }
  }

  // Regardless of whether we have a type of value initializer, update the type.
  paramDeclOp.setParamDecl(ParamDeclAttr::get(paramDeclOp.getName(), type));

  rejectDecorators(decoratorExprs, decl, p);
  return success();
}

ParseResult DeclResolver::resolveBody(ParamDeclareOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  // Nothing to do for a var decl, we parse everything as part of its
  // signature. We could move to parsing an initializer expression lazily when
  // a type is present if there were a reason to do that (e.g. more laziness
  // desired) in the future.
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

  // This is a struct, so we can use 'computeSelfTypeForStruct' to figure out
  // the self type.
  decl.setSelfType(decl.computeSelfTypeForStruct(sharedState));
  rejectDecorators(decoratorExprs, decl, p);
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
  rejectDecorators(decoratorExprs, decl, p);
  return success();
}

ParseResult DeclResolver::resolveBody(StructFieldOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  // Nothing to do for a var decl, we parse everything as part of its
  // signature. We could move to parsing an initializer expression lazily when
  // a type is present if there were a reason to do that (e.g. more laziness
  // desired) in the future.
  return success();
}
