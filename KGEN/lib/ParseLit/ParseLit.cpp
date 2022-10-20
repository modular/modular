//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the main entrypoints for the lit parser.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ParseLit.h"

#include "LitDecls.h"
#include "LitExprNodes.h"
#include "LitParserBase.h"
#include "LitScope.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LitSharedState.h"
#include "Support/IndexDialect/IndexDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// Definitions
//===----------------------------------------------------------------------===//

namespace {
/// identifier_opt_type  ::= identifier [":" expression]
/// meta_signature    ::= "[" [meta_param_list] "]"
/// meta_param_list   ::= identifier_opt_type ("," identifier_opt_type)
struct ParsedMetaSignature {
  SmallVector<ParamDeclAttr> inputDecls;
  std::vector<Location> inputLocs;

  ParseResult parseOptionalMetaSignature(LitParserBase &p, Scope &scope) {
    if (!p.consumeIf(LitToken::l_square) || p.consumeIf(LitToken::r_square))
      return success();

    auto parseMetaParameter = [&]() -> ParseResult {
      inputLocs.push_back(p.getTokenLocation());

      StringAttr name;
      if (p.parseIdentifier(name, "expected parameter name")) {
        // TODO: Scan ahead for better recovery.
        return failure();
      }

      Type paramType;
      if (p.parseToken(LitToken::colon,
                       "meta parameters always require a type") ||
          p.parseType(paramType, scope))
        return failure();
      inputDecls.push_back(ParamDeclAttr::get(name, paramType));
      return success();
    };

    if (p.parseCommaSeparatedList(parseMetaParameter) ||
        p.parseToken(LitToken::r_square, "expected ']' for parameter list"))
      return failure();
    return success();
  };
};
} // namespace

namespace {
struct ParsedParam {
  SMLoc loc;
  StringAttr name;
  Type type;
  ExprNode *initValue = nullptr;

  // TODO: Implement support for variadic parameter markers:
  // Python's parameter grammar embeds checking for `/` and `*` and `**` into
  // the grammar, we can just check for it using ad-hoc logic for simplicity,
  // according to the following rules:
  //   1) Only one /, *, and ** parameter may exist in the parameter list.
  //   2) They are specified in that order.
  //   3) These do not permit default arguments.
  ParseResult parse(LitParserBase &p, Scope &scope) {
    loc = p.getToken().getLoc();

    if (p.parseIdentifier(name, "expected parameter name"))
      // TODO: Scan ahead for better recovery.
      return failure();

    if (p.consumeIf(LitToken::colon)) {
      if (p.parseType(type, scope))
        return failure();
    }
    if (p.consumeIf(LitToken::equal)) {
      if (p.parseExpression(initValue))
        return failure();
    }
    return success();
  };
};
} // namespace

/// funcdef ::=  [decorators] "def" identifier [meta_signature]
///              "(" [value_param_list] ")" ["->" expression] ":" suite
///
/// value_param_list  ::= value_parameter ("," value_parameter)*
/// value_parameter   ::= value_parammarker identifier_opt_type ["=" expression]
/// value_parammarker ::= "/" | "*" | "**"
///
LogicalResult DeclResolver::resolveSignature(LITFuncOp defDecl, LitLexer &lexer,
                                             Scope &scope) {
  LitParserBase p(lexer);

  ParsedMetaSignature metaSignature;
  SmallVector<ParsedParam> params;
  Type resultType;

  if (metaSignature.parseOptionalMetaSignature(p, scope) ||
      p.parseToken(LitToken::l_paren, "expected '(' for parameter list"))
    return failure();

  if (!p.consumeIf(LitToken::r_paren)) {
    if (p.parseCommaSeparatedList([&]() {
          return params.emplace_back(ParsedParam()).parse(p, scope);
        }) ||
        p.parseToken(LitToken::r_paren, "expected ')' for parameter list"))
      return failure();
  }

  // Parse the result type if present.

  // TODO: This will be one difference between a def and fn: no result type on
  // a def should default to returning a (default initialized) Object, whereas
  // a fn can return void.  We can provide a guaranteed optimization to remove
  // it though.
  if (p.consumeIf(LitToken::minus_greater)) {
    if (p.parseType(resultType, scope))
      return failure();
  }

  if (p.parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  auto builder = scope.getDeclEndBuilder();

  // We have parsed the signature but skipped over the actual types, we use
  // unresolved types for now.
  SmallVector<Location> paramLocs;
  SmallVector<StringAttr> paramNames;
  SmallVector<Type> paramTypes;
  for (auto &param : params) {
    paramLocs.push_back(p.translateLocation(param.loc));
    paramNames.push_back(param.name);

    // If the parameter is missing a type, infer object type.
    // TODO(fn): /require/ types on parameters instead of defaulting to object.
    // TODO: I think there are some other special cases to evaluate, e.g. "self"
    // arguments should be containing type in methods?
    // TODO(default args): Get the type from the default arg when present.
    if (!param.type)
      param.type = builder.getType<ObjectType>();
    paramTypes.push_back(param.type);

    // TODO: add support for default parameter expressions.
    if (param.initValue)
      p.emitError(param.initValue->getLoc(), "TODO: No default values yet");
  }

  SmallVector<Type> resultTypes;
  if (resultType)
    resultTypes.push_back(resultType);

  defDecl.setValueParamNamesAttr(
      StringArrayAttr::get(getContext(), paramNames));
  defDecl.setType(builder.getFunctionType(paramTypes, resultTypes));
  defDecl.setParamDeclsAttr(
      ParamDeclArrayAttr::get(getContext(), metaSignature.inputDecls));
  defDecl.getBody()->addArguments(paramTypes, paramLocs);

  // Add the meta parameters to the symbol table.
  for (auto [param, loc] :
       llvm::zip(defDecl.getParamDecls(), metaSignature.inputLocs)) {
    auto value = ParamDeclRefAttr::get(param.getName(), param.getType());
    scope.addToScope(param.getName(), Scope::MetaParameterValue{value, loc},
                     sharedState);
  }

  // Set up the body of the def, creating declarations for the value parameters
  // and adding them to the symbol table.
  for (auto [arg, name] : llvm::zip(defDecl.getBody()->getArguments(),
                                    defDecl.getValueParamNames())) {
    // Create a mutable var.decl that references to the name can load from.
    // TODO: This is the wrong default, reconsider this for 'fn's when we have
    // a notion of immutability.
    auto type = POP::PointerType::get(arg.getType());
    auto varDecl = builder.create<VarDeclOp>(arg.getLoc(), type, name);
    addFullyResolvedDecl(varDecl, &scope);
    builder.create<POP::StoreOp>(arg.getLoc(), arg, varDecl,
                                 /*alignment*/ None);
  }
  return success();
}

void DeclResolver::resolveBody(LITFuncOp defDecl, LitLexer &lexer,
                               Scope &scope) {
  (void)LitParserBase::parseSuite(scope, lexer);

  // Check to see if we have a kgen.return at the end of function.  If not,
  // complain or add one implicitly if we have no results.
  Block *bodyBlock = defDecl.getBody();
  if (bodyBlock->empty() || !isa<ReturnOp>(bodyBlock->back())) {
    if (defDecl.getResultTypes().empty() &&
        defDecl.getResultParamTypes().empty()) {
      // TODO: Generalize lit.func.
      OpBuilder::atBlockEnd(bodyBlock).create<ReturnOp>(
          defDecl->getLoc(), ArrayRef<TypedAttr>(), ArrayRef<Value>());
    } else if (!sharedState.errorOccurred) {
      Location endLoc =
          bodyBlock->empty() ? defDecl.getLoc() : bodyBlock->back().getLoc();
      emitError(endLoc, "return expected at end of 'def' with results");
    }
  }

  // TODO: Do more type checking: verify that functions like __add__ have the
  // right signature.
}

/// var_decl_stmt ::= "var" identifier ":" expression ["=" expression]
///                 | "var" identifier "=" expression [TODO]
///
LogicalResult DeclResolver::resolveSignature(VarDeclOp varDecl, LitLexer &lexer,
                                             Scope &scope) {
  LitParserBase p(lexer);
  Type type;
  ExprNode *initValue = nullptr;
  // Parse the type if present.
  // TODO: Make type optional.
  if (p.parseToken(LitToken::colon, "var declaration requires a type") ||
      p.parseType(type, scope))
    return failure();

  varDecl.getResult().setType(POP::PointerType::get(type));

  if (p.consumeIf(LitToken::equal)) {
    p.emitError("var initializers not supported yet");
    if (p.parseExpression(initValue))
      return failure();
  }
  return success();
}

void DeclResolver::resolveBody(VarDeclOp op, LitLexer &lexer, Scope &scope) {
  // Nothing to do for a var decl, we parse everything as part of its signature.
  // We could move to parsing an initializer expression lazily when a type is
  // present if there were a reason to do that (e.g. more laziness desired) in
  // the future.
}

/// structdef ::=
///   [decorators] "struct" identifier [meta_signature] ":" suite
///
LogicalResult DeclResolver::resolveSignature(LITStructDeclOp structDecl,
                                             LitLexer &lexer, Scope &scope) {
  LitParserBase p(lexer);

  ParsedMetaSignature metaSignature;
  if (metaSignature.parseOptionalMetaSignature(p, scope) ||
      p.parseToken(LitToken::colon, "expected ':' in struct definition"))
    return failure();

  structDecl.setParamDeclsAttr(
      ParamDeclArrayAttr::get(getContext(), metaSignature.inputDecls));

  // Add the meta parameters to the struct's symbol table.
  for (auto [param, loc] :
       llvm::zip(structDecl.getParamDecls(), metaSignature.inputLocs)) {
    auto value = ParamDeclRefAttr::get(param.getName(), param.getType());
    scope.addToScope(param.getName(), Scope::MetaParameterValue{value, loc},
                     sharedState);
  }
  return success();
}

void DeclResolver::resolveBody(LITStructDeclOp op, LitLexer &lexer,
                               Scope &scope) {
  (void)LitParserBase::parseSuite(scope, lexer);
}

//===----------------------------------------------------------------------===//
// LitSharedState
//===----------------------------------------------------------------------===//

/// Get the name of the main buffer so we can rapidly build Location objects
/// on demand.
static StringAttr getMainBufferNameIdentifier(const SourceMgr &sourceMgr,
                                              MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

LitSharedState::LitSharedState(llvm::SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr), context(context),
      declResolver(std::make_unique<DeclResolver>(*this)),
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)) {}

LitSharedState::~LitSharedState() { declResolver.reset(); }

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.
Location LitSharedState::translateLocation(SMLoc loc) {
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  return FileLineColLoc::get(bufferNameIdentifier, lineAndColumn.first,
                             lineAndColumn.second);
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Parse the specified .lit file into the specified MLIR context.
OwningOpRef<mlir::ModuleOp> M::importLitFile(SourceMgr &sourceMgr,
                                             MLIRContext *context,
                                             mlir::TimingScope &ts) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  context->loadDialect<POP::POPDialect, LITDialect, index::IndexDialect,
                       KGENDialect, mlir::scf::SCFDialect>();

  // This is the result module we are parsing into.
  auto fileLoc =
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0);
  mlir::OwningOpRef<ModuleOp> module(ModuleOp::create(fileLoc));

  LitSharedState sharedState(sourceMgr, context);
  LitLexer lexer(sharedState);
  auto startSMLoc = lexer.getToken().getLoc();

  // The outermost scope contains the __builtins__ function definitions.
  // TODO: Add these:
  // https://docs.python.org/3/library/functions.html#built-in-funcs
  // https://docs.python.org/3/reference/executionmodel.html#naming-and-binding
  Scope &builtinsScope = sharedState.declResolver->addDecl(
      *module, nullptr, lexer.getCursor(), -1);

  // Create the module scope which will contain all things we parse.  These
  // shadow the builtins module during name lookup.
  Scope &fileScope = sharedState.declResolver->addDecl(*module, &builtinsScope,
                                                       lexer.getCursor(), -1);

  // Parse the file.
  /// file ::= statements
  if (LitParserBase::parseSuite(fileScope, lexer))
    return nullptr;

  // With the top-level of the file parsed, we can now go ahead and resolve all
  // of the deferred declarations.
  sharedState.declResolver->resolveAll(startSMLoc);

  // We fail either if we have a non-recoverable parse error, or if we emitted
  // an error and then recovered.  In either case, the IR will not be valid and
  // the caller should not verify it.
  if (sharedState.errorOccurred)
    return nullptr;
  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  auto verificationTimer = ts.nest("Verify module");
  if (failed(verify(*module)))
    return {};
  return module;
}
