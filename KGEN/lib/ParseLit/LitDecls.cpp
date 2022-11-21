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

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// ASTDecl
//===----------------------------------------------------------------------===//

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

  if (auto structOp = dyn_cast<LITStructDeclOp>(op)) {
    // TODO: Support nested/local structs.
    return FlatSymbolRefAttr::get(structOp.getNameAttr());
  }

  if (auto fnOp = dyn_cast<LITFuncOp>(op)) {
    // TODO: Support multiple levels of nesting.  This should be recursive, and
    // SymbolRefAttr should support a get(FlatSymbol, SymbolRef) helper that
    // forms a properly flattened reference by unwinding the RHS if it isn't
    // flat.
    SymbolRefAttr symbolRef = FlatSymbolRefAttr::get(fnOp.getNameAttr());
    if (auto parentStruct = dyn_cast<LITStructDeclOp>(*getParentDecl()))
      symbolRef = SymbolRefAttr::get(parentStruct.getNameAttr(),
                                     cast<FlatSymbolRefAttr>(symbolRef));
    return symbolRef;
  }

  return {};
}

/// Given an MLIR op for a struct declaration, return the self type.
ASTType ASTDecl::computeSelfTypeForStruct(LitSharedState &state) {
  auto structOp = cast<LITStructDeclOp>(*this);

  SmallVector<ParamBindAttr> parameters;
  for (auto decl : structOp.getParamDecls()) {
    // We're using the parameter from the type declaration scope in the
    // parameter binding list.
    TypedAttr ref = ParamDeclRefAttr::get(decl.getName(), decl.getType());
    parameters.push_back(ParamBindAttr::get(decl, ref));
  }

  // Methods on structs (but not classes) take the struct implicitly by
  // pointer so they can use and mutate it.
  return LITDeclRefType::get(getSymbolRef(), parameters);
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
ASTDecl &LitSharedState::getDeclForSymbol(SymbolRefAttr symbol) {
  return declResolver->getDeclForSymbol(symbol);
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(DeclIRValue irValue, Location loc,
                               StringAttr name, ASTDecl *parentDecl,
                               LitLexerCursor cursor, LitLexerCursor endCursor,
                               ssize_t indentation) {
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
    diag.attachNote(existing->getLoc()) << "previous definition here";

    // Mark the existing decl and this one as erroneous so uses of either
    // don't create confusing errors.
    decl->hasReferenceError = true;
    existing->hasReferenceError = true;
  }

  return *decl;
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(Operation *op, StringAttr name,
                               ASTDecl *parentDecl, LitLexerCursor cursor,
                               LitLexerCursor endCursor, ssize_t indentation) {
  return addDecl(op, op->getLoc(), name, parentDecl, cursor, endCursor,
                 indentation);
}

ASTDecl &DeclResolver::addFullyResolvedDecl(Operation *op, StringAttr name,
                                            ASTType type, ASTDecl *parentDecl) {
  auto &decl =
      addDecl(op, name, parentDecl, LitLexerCursor(), LitLexerCursor(), 0);
  decl.resolvedness = DeclResolvedness::fullyResolved;
  decl.setResolvedType(type);
  return decl;
}

/// Add a declaration that is already fully resolved.
ASTDecl &DeclResolver::addFullyResolvedDecl(DeclIRValue declVal,
                                            StringAttr name, Location loc,
                                            ASTType type, ASTDecl *parentDecl) {
  auto &decl = addDecl(declVal, loc, name, parentDecl, LitLexerCursor(),
                       LitLexerCursor(), 0);
  decl.resolvedness = DeclResolvedness::fullyResolved;
  decl.setResolvedType(type);
  return decl;
}

/// Resolve all of the declarations that are visible.
void DeclResolver::resolveAll(SMLoc loc) {
  // We can do this in any order, but choose to use the order they are
  // discovered so diagnostics are mostly top-down.  Resolving declarations
  // may cause more entries to be added to this list.
  for (size_t i = 0; i != parsedDeclList.size(); ++i)
    (void)resolve(*parsedDeclList[i], DeclResolvedness::fullyResolved, loc);
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

  // If we are currently name binding this operation, we found a cycle, reject
  // it with an error.
  if (!declsCurrentlyProcessing.insert(&decl).second) {
    emitError(sharedState.translateLocation(loc),
              "recursive reference to declaration");
    return failure();
  }

  // If the signature hasn't been parsed, do so.
  if (decl.resolvedness < DeclResolvedness::signatureResolved) {
    // Handle each operation that can be name bound.  We handle this by
    // restoring the lexer to the position where parsing can continue, calling
    // the `resolveSignature` method for the op, and re-saving the new cursor
    // for the next stage of resolution.
    TypeSwitch<ASTDecl &>(decl)
        .Case<LITFuncOp, LITStructDeclOp, VarDeclOp>([&](auto op) {
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
        });
    decl.resolvedness = DeclResolvedness::signatureResolved;
  }

  // If the declaration hasn't been fully parsed and we need to, do so.
  if (decl.resolvedness < DeclResolvedness::fullyResolved &&
      howResolved == DeclResolvedness::fullyResolved) {
    // Handle each operation that can be name bound.
    TypeSwitch<ASTDecl &>(decl)
        .Case<LITFuncOp, LITStructDeclOp, VarDeclOp>([&](auto op) {
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
// Function Decl implementation
//===----------------------------------------------------------------------===//

namespace {
/// identifier_opt_type  ::= identifier [":" expression]
/// meta_signature    ::= "[" [meta_param_list] "]"
/// meta_param_list   ::= identifier_opt_type ("," identifier_opt_type)
struct ParsedMetaSignature {
  SmallVector<ParamDeclAttr> inputDecls;
  SmallVector<ASTType> inputASTTypes;
  std::vector<Location> inputLocs;

  ParseResult parseOptionalMetaSignature(LitParserBase &p, ASTDecl &decl) {
    if (!p.consumeIf(LitToken::l_square) || p.consumeIf(LitToken::r_square))
      return success();

    auto parseMetaParameter = [&]() -> ParseResult {
      inputLocs.push_back(p.getTokenLocation());

      StringAttr name;
      if (p.parseIdentifier(name, "expected parameter name")) {
        // TODO: Scan ahead for better recovery.
        return failure();
      }

      ASTType paramType;
      if (p.parseToken(LitToken::colon,
                       "meta parameters always require a type") ||
          p.parseType(paramType, decl, None))
        return failure();
      inputDecls.push_back(ParamDeclAttr::get(name, paramType.getMLIRType()));
      inputASTTypes.push_back(paramType);
      return success();
    };

    if (p.parseCommaSeparatedList(parseMetaParameter, LitToken::r_square) ||
        p.parseToken(LitToken::r_square, "expected ']' for parameter list"))
      return failure();
    return success();
  }

  /// Add ASTDecl objects for each declared parameter and insert them into the
  /// specified scope for the declaration, so name lookup will find them.
  void addToScope(LitSharedState &sharedState, ASTDecl &decl) {
    auto &declResolver = *sharedState.declResolver;
    // TODO: Use inputTypes.
    for (auto [paramDecl, type, loc] :
         llvm::zip(inputDecls, inputASTTypes, inputLocs)) {
      auto paramRef =
          ParamDeclRefAttr::get(paramDecl.getName(), paramDecl.getType());
      declResolver.addFullyResolvedDecl(MValue(paramRef), paramDecl.getName(),
                                        loc, type, &decl);
    }
  }
};
} // namespace

namespace {
/// Parsing support for a function value (not meta) parameter:
///
/// value_param_list  ::= value_param ("," value_param)*
/// value_param       ::= value_parammarker identifier_opt_type ["=" expression]
/// value_parammarker ::= "/" | "*" | "**" | "&"
///
struct ParsedParam {
  SMLoc loc;
  // True if this is a `&` byref argument. TODO: Expand this to being an enum so
  // we can do move semantics etc.
  bool isByRef = false;
  StringAttr name;
  ASTType type;
  ExprNode *initValue = nullptr;

  ParseResult parse(LitParserBase &p, ASTDecl &declScope) {
    loc = p.getToken().getLoc();

    // TODO: Implement support for variadic parameter markers:
    // Python's parameter grammar embeds checking for `/` and `*` and `**` into
    // the grammar, we can just check for it using ad-hoc logic for simplicity,
    // according to the following rules:
    //   1) Only one /, *, and ** parameter may exist in the parameter list.
    //   2) They are specified in that order.
    //   3) These do not permit default arguments.

    // Handle & for by-ref arguments.
    if (p.consumeIf(LitToken::amp))
      isByRef = true;

    if (p.parseIdentifier(name, "expected parameter name"))
      // TODO: Scan ahead for better recovery.
      return failure();

    if (p.consumeIf(LitToken::colon)) {
      if (p.parseType(type, declScope, None))
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
/// This returns failure after emitting an error when a type checking problem
/// is detected.
static ParseResult checkFunctionSignature(ASTDecl &decl, Operation *op,
                                          ParsedMetaSignature &metaSignature,
                                          SmallVector<ParsedParam> &params,
                                          ASTType &resultType,
                                          LitSharedState &shared) {
  SpecialFunctionInfo fnInfo;
  bool isStatic = false;

  // We either have a function or interface.  Functions are more general and
  // therefore have more checking to perform.
  auto funcOp = dyn_cast<LITFuncOp>(op);
  if (funcOp) {
    fnInfo = SpecialFunctionInfo::get(funcOp.getName());
    isStatic = funcOp.getIsStatic();
  }

  // If this definition is a struct/class member, return the self type
  // otherwise return a null type.
  ASTType selfType;
  if (auto *parentDecl = decl.getParentDecl())
    if (isa<LITStructDeclOp>(*parentDecl)) {
      // If this is a method, the signature for the enclosing type must be
      // resolved.
      (void)shared.declResolver->resolve(*parentDecl,
                                         DeclResolvedness::signatureResolved,
                                         decl.getCursor().getToken().getLoc());
      // The self type is stored as the resolved type.
      selfType = parentDecl->getResolvedType();
    }

  // __new__ and similar methods are implicitly static.
  if (fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod) {
    if (!selfType)
      return op->emitError("special function must be a method");

    assert(funcOp && "Cannot have special function generators");
    funcOp.setIsStaticAttr(mlir::UnitAttr::get(shared.getContext()));
    isStatic = true;
  }

  // If this is an instance method, enforce that self is declared correctly.
  if (selfType && !isStatic) {
    // If there are no parameters, install an implicit by-val self parameter.
    if (params.empty()) {
      params.push_back({decl.getCursor().getToken().getLoc(),
                        /*isByRef=*/false,
                        StringAttr::get(shared.getContext(), "self"), selfType,
                        /*initPtr*/ nullptr});
    }

    // Check that the first method has the right type.
    if (!params[0].type)
      params[0].type = selfType;
    else if (!params[0].type.isEqualCanon(selfType))
      return op->emitError("'self' argument must have type ") << selfType;
  }

  // Check other invariants based on method flags.
  if (fnInfo.flags & SpecialFunctionInfo::kInstMethod) {
    bool isByRef = fnInfo.isByRefSelfInstMethod();
    if (!selfType)
      return op->emitError("special function must be a method");
    if (isStatic)
      return op->emitError("special method may not be a static method");

    // TODO: Instead of enforcing byref is specified correctly, we could reject
    // invalid explicit settings and default it correctly.
    if (isByRef != params[0].isByRef)
      return op->emitError("self parameter must ")
             << (isByRef ? "" : "not ") << "be passed by reference";
  }

  // Verify the operand count lines up.
  if (fnInfo.numOperands != -1 && size_t(fnInfo.numOperands) != params.size())
    return op->emitError("special function must have ")
           << fnInfo.numOperands << " operand" << plural(fnInfo.numOperands);

  switch (fnInfo.kind) {
  default:
    // Ignore methods without special handling.
    break;
  case SpecialFunctionKind::kInit:
    if (isa<LITStructDeclOp>(*decl.getParentDecl()))
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

  // Handle any parameters that are still missing a type.
  // TODO(default args): Get the type from the default arg when present.
  for (auto &param : params) {
    if (!param.type) {
      // If we are in a 'def', we infer object type for Python compatibility, in
      // an 'fn' we report an error.
      if (decl.isDef) {
        param.type = shared.getObjectType();
      } else {
        op->emitError("'fn' parameter type must be specified");
        param.type = shared.getTypeCheckErrorType();
      }
    }
  }

  return success();
}

/// funcdef ::=  [decorators] "def" identifier [meta_signature]
///              "(" [value_param_list] ")" ["->" expression] ":" suite
///
LogicalResult DeclResolver::resolveSignature(LITFuncOp funcOp, LitLexer &lexer,
                                             ASTDecl &decl) {
  LitParserBase p(lexer);

  ParsedMetaSignature metaSignature;
  SmallVector<ParsedParam> params;
  if (metaSignature.parseOptionalMetaSignature(p, decl) ||
      p.parseToken(LitToken::l_paren, "expected '(' for parameter list"))
    return failure();

  // Add the meta parameters to the symbol table.  We add all of these after
  // generic signature parsing so types used in the signature list resolve to
  // enclosing scopes, and we add them before the value signature list so the
  // types and parameters can resolve to the bound values.
  metaSignature.addToScope(sharedState, decl);

  if (!p.consumeIf(LitToken::r_paren)) {
    if (p.parseCommaSeparatedList(
            [&]() { return params.emplace_back(ParsedParam()).parse(p, decl); },
            LitToken::r_paren) ||
        p.parseToken(LitToken::r_paren, "expected ')' for parameter list"))
      return failure();
  }

  // Parse the result type if present.

  // TODO: This will be one difference between a def and fn: no result type on
  // a def should default to returning a (default initialized) Object, whereas
  // a fn can return void.  We can provide a guaranteed optimization to remove
  // it though.
  ASTType resultType;
  if (p.consumeIf(LitToken::minus_greater)) {
    if (p.parseType(resultType, decl, None))
      return failure();
  } else {
    resultType = sharedState.getNoneType();
  }

  if (p.parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  // Verify that methods and functions like __add__ have the right signature,
  // and adjust them if there are implicit declarations.
  if (checkFunctionSignature(decl, funcOp, metaSignature, params, resultType,
                             sharedState)) {
    // If the function wasn't type checked correctly, uses of it may be
    // broken.
    decl.hasReferenceError = true;

    // Set any unspecifies argument types to error type.
    for (auto &param : params) {
      if (!param.type)
        param.type = sharedState.getTypeCheckErrorType();
    }
  }

  // The resolvedType for a function is the return type of the function.
  decl.setResolvedType(resultType);

  SmallVector<Location> paramLocs;
  SmallVector<StringAttr> paramNames;
  SmallVector<Type> paramTypes;
  for (auto &param : params) {
    paramLocs.push_back(p.translateLocation(param.loc));
    paramNames.push_back(param.name);

    // Install the correct MLIR Type, which is the MLIR projection of the AST
    // type with any convention changes applied.
    auto mlirType = param.type.getMLIRType();
    if (param.isByRef)
      mlirType = POP::PointerType::get(mlirType);
    paramTypes.push_back(mlirType);

    // TODO: add support for default parameter expressions.
    if (param.initValue)
      p.emitError(param.loc, "TODO: No default values yet");
  }

  auto builder = decl.getDeclEndBuilder();
  funcOp.setValueParamNamesAttr(builder.getAttr<StringArrayAttr>(paramNames));
  funcOp.setType(builder.getFunctionType(paramTypes, resultType.getMLIRType()));
  funcOp.setParamDeclsAttr(
      builder.getAttr<ParamDeclArrayAttr>(metaSignature.inputDecls));
  funcOp.getBody()->addArguments(paramTypes, paramLocs);

  if (FlatSymbolRefAttr implementsAttr = funcOp.getImplementsAttr()) {
    StringRef interfaceName = implementsAttr.getAttr().getValue();
    if (ASTDecl *interfaceDecl = decl.lookup(implementsAttr.getAttr())) {
      if (auto funcInterface =
              dyn_cast_or_null<LITFuncOp>(interfaceDecl->getIfOperation());
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
  for (auto [arg, param] :
       llvm::zip(funcOp.getBody()->getArguments(), params)) {
    // We need to know what parameter convention that argument is passed
    // with, e.g. by-value, by-ref, by-transfer, etc.  By-reference arguments
    // are modeled with the IR-level POP::PointerType.
    if (isa<POP::PointerType>(arg.getType())) {
      // Arguments passed by-reference can be directly used.
      addFullyResolvedDecl(LValue(arg), param.name, arg.getLoc(), param.type,
                           &decl);
      continue;
    }

    // If this was passed by-value, then it becomes an rvalue in a `fn`.
    if (!decl.isDef) {
      addFullyResolvedDecl(DRValue(arg), param.name, arg.getLoc(), param.type,
                           &decl);
      continue;
    }

    // In a `def`, we create a mutable var.decl lvalue to allow reassignment.
    auto type = POP::PointerType::get(arg.getType());
    auto varDecl = builder.create<VarDeclOp>(arg.getLoc(), type, param.name);
    addFullyResolvedDecl(varDecl, param.name, param.type, &decl);
    builder.create<POP::StoreOp>(arg.getLoc(), arg, varDecl,
                                 /*alignment*/ None);
  }

  return success();
}

ParseResult DeclResolver::resolveBody(LITFuncOp defOp, LitLexer &lexer,
                                      ASTDecl &decl) {
  if (LitParserBase::parseSuite(decl, lexer))
    return failure();

  // Check to see if we have a kgen.return at the end of function.  If not,
  // complain or add one implicitly if we have no results.
  Block *bodyBlock = defOp.getBody();
  auto loc = decl.getLoc();
  bool isInterface = defOp.getIsInterface();

  if (isInterface) {
    if (!bodyBlock->empty())
      emitError(loc, "interfaces must have no body");
    return success();
  }
  if (bodyBlock->empty() || !isa<ReturnOp>(bodyBlock->back())) {
    auto loc = decl.getLoc();
    if (isa<KGEN::NoneType>(defOp.getResultType()) &&
        defOp.getResultParamTypes().empty()) {
      auto b = OpBuilder::atBlockEnd(bodyBlock);

      auto noneAttr = b.getType<NoneAttr>(defOp.getResultType());
      Value noneVal = b.create<ParamConstantOp>(loc, noneAttr);
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
    if (p.parseType(type, decl, decl.getIndentation()))
      return failure();
    varOp.getResult().setType(POP::PointerType::get(type.getMLIRType()));
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
    if (type && !type.isEqualCanon(rhsValue.type)) {
      p.emitError(initValue->getLoc(), "initializer has type ")
          << rhsValue.type << " but declared type is " << type;
      return failure(); // Not sure which type is right.
    }

    // Infer the type if we lack a declared type (`var x = 42`)
    if (!type) {
      type = rhsValue.type;
      varOp.getResult().setType(POP::PointerType::get(type.getMLIRType()));
    }

    // The types line up, do a store.
    auto loc = sharedState.translateLocation(initValue->getLoc());
    builder.create<POP::StoreOp>(loc, rhsValue.ir, varOp,
                                 /*alignment*/ None);
  }

  // If there was neither a type or initializer, reject the var.
  if (!type) {
    p.emitError(varOp.getLoc(),
                "declaration must have either a type or an initializer");
    return failure();
  }

  // The resolvedType of a variable declaration is the type of the decl.
  decl.setResolvedType(type);
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
// Struct Decl implementation
//===----------------------------------------------------------------------===//

/// structdef ::=
///   [decorators] "struct" identifier [meta_signature] ":" suite
///
LogicalResult DeclResolver::resolveSignature(LITStructDeclOp structOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);

  ParsedMetaSignature metaSignature;
  if (metaSignature.parseOptionalMetaSignature(p, decl) ||
      p.parseToken(LitToken::colon, "expected ':' in struct definition"))
    return failure();

  structOp.setParamDeclsAttr(
      ParamDeclArrayAttr::get(getContext(), metaSignature.inputDecls));

  // Add the meta parameters to the struct's symbol table.
  metaSignature.addToScope(sharedState, decl);

  // This is a struct, so we can use 'computeSelfTypeForStruct' to figure out
  // the self type.
  decl.setResolvedType(decl.computeSelfTypeForStruct(sharedState));
  return success();
}

ParseResult DeclResolver::resolveBody(LITStructDeclOp structOp, LitLexer &lexer,
                                      ASTDecl &decl) {
  return LitParserBase::parseSuite(decl, lexer);
}
