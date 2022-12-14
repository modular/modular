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
#include "LitLexer.h"
#include "LitParserBase.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LitSharedState.h"
#include "SpecialFunctions.h"
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

/// Given the symbol for a lit declaration, return the ASTDecl that
/// corresponds to it.  This doesn't allow null symbols, so it always
/// succeeds.
ASTDecl &LitSharedState::getDeclForSymbol(SymbolRefAttr symbol) const {
  return declResolver->getDeclForSymbol(symbol);
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
    // If the decl also has a symbol, remember it, so we can look up decls by
    // symbol.
    if (SymbolRefAttr symbol = decl->getSymbolRef()) {
      assert(!declForSymbol.count(symbol) && "Symbol redefinition/collision");
      declForSymbol[symbol] = decl;
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
        .Default([&](auto attr) {
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
        .Default([&](auto attr) {
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

#define SF(ENUM, NAME, NUMOPERANDS, FLAGS)                                     \
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
#define SF(ENUM, NAME, NUMOPERANDS, FLAGS)                                     \
  {NAME, SpecialFunctionKind::ENUM, (NUMOPERANDS), (FLAGS)},
#include "SpecialFunctions.def"
  };

  assert(unsigned(kind) < sizeof(infos) / sizeof(infos[0]));
  return infos[unsigned(kind)];
}

/// Perform type checking for a function signature that has just been parsed
/// but that has not been installed into the specified decl.  This allows
/// magic behavior (like __new__ being static, self getting implicitly
/// declared), checking of method self requirements, inference of default object
/// argument types and enforcement of other invariants.
///
/// This fills in argTypes with the resolved types of arguments, on either
/// success or error.
///
/// This returns failure (after emitting an error) when a type checking problem
/// is detected.
static ParseResult resolveFunctionSignature(ASTDecl &decl, LIT::FuncOp op,
                                            SmallVector<ParsedArgument> &args,
                                            SmallVectorImpl<Type> &argTypes,
                                            ASTType &resultType,
                                            LitSharedState &shared) {
  SpecialFunctionInfo fnInfo = SpecialFunctionInfo::get(op.getName());

  // __new__ and similar methods are implicitly static.
  if (fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod)
    op.setIsStaticAttr(mlir::UnitAttr::get(shared.getContext()));

  // If this definition is a struct/class member, compute the self type.
  ASTType selfType;
  if (auto *parentDecl = decl.getParentDecl())
    if (isa<StructDeclOp>(*parentDecl)) {
      //  The parent decl must be fully resolved in order to resolve any members
      //  of it.
      assert(parentDecl->resolvedness == DeclResolvedness::fullyResolved);
      selfType = parentDecl->getSelfType();
    }

  // Type check all arguments, figuring out a type to use for them (incl
  // possibly an error type).
  ExprEmitter argEmitter(shared, decl, None, nullptr);
  for (auto &arg : args) {
    ASTType type;
    if (arg.typeExpr) {
      // This returns a TypeCheckErrorType on error, no extra check is needed.
      type = argEmitter.emitType(arg.typeExpr);
    } else if (&arg == &args[0] && selfType && !op.getIsStatic()) {
      // We can always default 'self' in a method.
      type = selfType;
    } else if (op.getIsDef()) {
      // If we are in a 'def', we infer object type for Python compatibility.
      type = shared.getObjectType();
    } else {
      // In an 'fn' we report an error.
      op->emitError("'fn' parameter type must be specified");
      type = shared.getTypeCheckErrorType();
    }

    // Apply any conventions requested.
    if (arg.convention == ValueInputConvention::ByRef)
      type = POP::PointerType::get(type);

    // TODO: add support for default parameter expressions.
    if (arg.initValue)
      argEmitter.emitError(arg.loc, "TODO: No default values yet");

    argTypes.push_back(type);
  }

  // Check that the 'self' argument of a method was specified correctly.
  if (selfType && !op.getIsStatic() && !argTypes.empty()) {
    auto adjustedSelf = selfType;
    if (args[0].convention == ValueInputConvention::ByRef)
      adjustedSelf = POP::PointerType::get(adjustedSelf);

    if (!ASTType(argTypes[0]).isEqualCanon(adjustedSelf))
      op->emitError("'self' argument must have type ") << selfType;
  }

  // Verify that implicitly static methods are declared as methods.
  if ((fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod) &&
      !selfType)
    return op->emitError("special function must be a method");

  // Verify the operand count lines up.
  if (fnInfo.numOperands != -1 && size_t(fnInfo.numOperands) != args.size())
    return op->emitError("special function must have ")
           << fnInfo.numOperands << " operand" << plural(fnInfo.numOperands);

  // Check other invariants based on method flags.
  if (fnInfo.flags & SpecialFunctionInfo::kInstMethod) {
    auto convent = fnInfo.isByRefSelfInstMethod() ? ValueInputConvention::ByRef
                                                  : ValueInputConvention::ByVal;
    if (!selfType)
      return op->emitError("special function must be a method");
    if (op.getIsStatic())
      return op->emitError("special method may not be a static method");
    if (args.empty())
      return op->emitError("self argument must be present in instance method");

    // TODO: Instead of enforcing byref is specified correctly, we
    // could reject invalid explicit settings and default it correctly.
    if (convent != args[0].convention)
      return op->emitError("self argument must ")
             << (convent == ValueInputConvention::ByRef ? "" : "not ")
             << "be passed by reference";
  }

  switch (fnInfo.kind) {
  default:
    // Ignore methods without special handling.
    break;
  case SpecialFunctionKind::kInit:
    if (isa<StructDeclOp>(*decl.getParentDecl()))
      return op->emitError(
          "__init__ is not allowed on structs, use __new__ instead");
    // __init__ on classes must return NoneType.
    if (!resultType.isEqualCanon(shared.getNoneType()))
      return op->emitError("__init__ result type must be elided (or None)");
    break;

  case SpecialFunctionKind::kNew:
    // __new__ must return containing type.
    // TODO: We could allow omitting result type and default it.
    if (!resultType.isEqualCanon(selfType))
      return op->emitError("result type must be ") << selfType;
    break;
  }

  return success();
}

/// funcdef ::=  [decorators] "def" identifier [meta_signature]
///              "(" [value_param_list] ")" ["->" expression] ":" suite
///
LogicalResult DeclResolver::resolveSignature(LIT::FuncOp funcOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);

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

  ASTType resultType;
  if (!resultTypeExpr) {
    // TODO: This will be one difference between a def and fn: no result type on
    // a def should default to returning a (default initialized) Object, whereas
    // a fn can return void.
    resultType = sharedState.getNoneType();
  } else {
    resultType =
        ExprEmitter(sharedState, decl, None, nullptr).emitType(resultTypeExpr);
  }

  // If the method can raise an exception, wrap the result type in a variant
  // with the error type.
  if (funcOp.getRaises())
    resultType = sharedState.getErrorOrType(resultType);

  // Verify that methods and functions like __add__ have the right signature,
  // and adjust them if there are implicit declarations.
  SmallVector<Type> argTypes;
  if (resolveFunctionSignature(decl, funcOp, args, argTypes, resultType,
                               sharedState)) {
    // If the function wasn't type checked correctly, uses of it may be
    // broken.
    decl.hasReferenceError = true;
  }

  // Finally now that the full signature has been resolved, build our IR.

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

  if (funcOp.getIsInterface())
    return success();

  // Set up the body of the def, creating declarations for the value
  // parameters and adding them to the symbol table.
  for (auto [bbArg, parsedArg] :
       llvm::zip(funcOp.getBody()->getArguments(), args)) {
    // Arguments passed by-reference can be directly used.
    if (parsedArg.convention == ValueInputConvention::ByRef) {
      addFullyResolvedDecl(LValue(bbArg), parsedArg.name, parsedArg.loc, &decl);
      continue;
    }
    assert(parsedArg.convention == ValueInputConvention::ByVal &&
           "Unknown convention");

    // If this was passed by-value, then it becomes an rvalue in a `fn`.
    if (!funcOp.getIsDef()) {
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
  ASTType type;
  // Parse the type if present.
  if (p.consumeIf(LitToken::colon)) {
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
  ASTType type;
  // Parse the type if present.
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

  ParsedMetaSignature metaSignature(decl);
  if (metaSignature.parseOptionalMetaSignature(p) ||
      p.parseToken(LitToken::colon, "expected ':' in struct definition"))
    return failure();

  // Resolve the meta parameters and get their decls.
  SmallVector<ParamDeclAttr> inputParamDecls =
      metaSignature.getResolvedInputParamDecls(*this);

  structOp.setInputParamDecls(inputParamDecls);

  // This is a struct, so we can use 'computeSelfTypeForStruct' to figure out
  // the self type.
  decl.setSelfType(decl.computeSelfTypeForStruct(sharedState));
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
  ASTType type;
  // Parse the type if present.
  if (p.consumeIf(LitToken::colon)) {
    if (parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
      return failure();
    fieldOp.setType(type);
  } else {
    p.emitError(fieldOp.getLoc(), "struct field declaration must have a type");
    return failure();
  }

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
