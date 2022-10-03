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
#include "LitExprNodes.h"
#include "LitLexer.h"
#include "LitScope.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/IndexDialect/IndexDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"

using namespace M;
using namespace M::LLCL;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// SharedParserState
//===----------------------------------------------------------------------===//

namespace {
/// This is state shared across multiple different instances of LitParserBase
/// which are always shared across them.
struct SharedParserState {
  MLIRContext *const context;
  llvm::BumpPtrAllocator exprAllocator;
  bool hasExprParser = false;
  bool errorOccurred = false;

  SharedParserState(MLIRContext *context) : context(context) {}
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LitParserBase
//===----------------------------------------------------------------------===//

namespace {
/// This class implements logic that is common to many parts of the parser, but
/// which is independent of the concrete grammar.
struct LitParserBase {
  LitParserBase(LitLexer &lexer, SharedParserState *sharedParserState)
      : lexer(lexer), sharedParserState(sharedParserState) {}

  MLIRContext *getContext() const { return sharedParserState->context; }
  LitLexer &getLexer() { return lexer; }

  /// Return the current token the parser is inspecting.
  const LitToken &getToken() const { return lexer.getToken(); }
  StringRef getTokenSpelling() const { return getToken().getSpelling(); }

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and notice that so we don't verify the IR at the end of
  /// compilation.
  InFlightDiagnostic emitError(Location loc, const Twine &message = {}) {
    sharedParserState->errorOccurred = true;
    return mlir::emitError(loc, message);
  }

  /// Emit an error at the current token.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(getToken().getLoc(), message);
  }
  /// Emit an error at a specific lexer location.
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  /// Return true if we encountered an error during compilation.
  bool hadError() const { return sharedParserState->errorOccurred; }

  //===--------------------------------------------------------------------===//
  // Location Handling
  //===--------------------------------------------------------------------===//

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location translateLocation(llvm::SMLoc loc) {
    return lexer.translateLocation(loc);
  }

  /// Return the location for the current token.
  Location getTokenLocation() { return translateLocation(getToken().getLoc()); }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.  If 'tokLoc' is non-null, it is filled in with the
  /// location of the consumed token (on success).
  bool consumeIf(LitToken::Kind kind, SMLoc *tokLoc = nullptr) {
    if (getToken().isNot(kind))
      return false;
    if (tokLoc)
      *tokLoc = getToken().getLoc();
    consumeToken(kind);
    return true;
  }

  /// Advance the current lexer onto the next token.
  ///
  /// This returns the consumed token.
  LitToken consumeToken() {
    LitToken consumedToken = getToken();
    assert(consumedToken.isNot(LitToken::eof) && "shouldn't advance past EOF");
    lexer.lexToken();
    return consumedToken;
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  ///
  /// This returns the consumed token.
  LitToken consumeToken(LitToken::Kind kind) {
    LitToken consumedToken = getToken();
    assert(consumedToken.is(kind) && "consumed an unexpected token");
    consumeToken();
    return consumedToken;
  }

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure.
  ParseResult parseToken(LitToken::Kind expectedToken, const Twine &message);

  /// Parse a list of elements, terminated with an arbitrary token.  This does
  /// not consume the stop token.
  ///
  /// list ::= (element)* STOPTOKEN
  ///
  ParseResult parseListUntil(LitToken::Kind stopToken,
                             const std::function<ParseResult()> &parseElement);

  /// Parse a list of elements continued with a separator token, like a comma.
  ///
  /// separated_list ::= (element (SEPARATOR element)*
  ///
  ParseResult
  parseSeparatedList(LitToken::Kind separator,
                     const std::function<ParseResult()> &parseElement);
  ParseResult
  parseCommaSeparatedList(const std::function<ParseResult()> &parseElement) {
    return parseSeparatedList(LitToken::comma, parseElement);
  }

  /// Consume tokens until we get to the end of the current line, used for error
  /// recovery.
  /// TODO: we should know the indentation of the current statement so we can
  /// eat trailing components that lack a \.
  void eatToEndOfLine() {
    while (!getToken().getIndentation().has_value() &&
           getToken().isNot(LitToken::eof))
      consumeToken();
  }

  /// Consume tokens until one of the specified set of token, leaving the
  /// stopToken in the stream.  This produces an error if EOF is encountered.
  ///
  /// NOTE: This shouldn't be used in a real parser, this is just for phasing
  /// things in.
  ParseResult eatUntil(ArrayRef<LitToken::Kind> stopTokens) {
    while (getToken().isNot(LitToken::eof)) {
      // If we found our stop token, we succeeded!
      if (getToken().isAny(stopTokens))
        return success();
      consumeToken();
    }

    return emitError("expected end token");
  }

public:
  LitLexer &lexer;
  SharedParserState *const sharedParserState;

  LitParserBase(const LitParserBase &) = delete;
  void operator=(const LitParserBase &) = delete;
};

} // end anonymous namespace

InFlightDiagnostic LitParserBase::emitError(SMLoc loc, const Twine &message) {
  auto diag = emitError(translateLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(LitToken::error))
    diag.abandon();
  return diag;
}

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult LitParserBase::parseToken(LitToken::Kind expectedToken,
                                      const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitError(message);
}

/// Parse a list of elements, terminated with an arbitrary token.  This does
/// not consume the stop token.
///
/// list ::= (element)* STOPTOKEN
///
ParseResult LitParserBase::parseListUntil(
    LitToken::Kind rightToken,
    const std::function<ParseResult()> &parseElement) {

  while (!consumeIf(rightToken)) {
    if (parseElement())
      return failure();
  }
  return success();
}

/// Parse a list of elements continued with a separator token, like a comma.
///
/// separated_list ::= (element (SEPARATOR element)*
///
ParseResult LitParserBase::parseSeparatedList(
    LitToken::Kind separator,
    const std::function<ParseResult()> &parseElement) {
  if (parseElement())
    return failure();
  while (consumeIf(separator)) {
    if (parseElement())
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Expression Parsing
//===----------------------------------------------------------------------===//

/// Expression parsing in Lightning is done in with a 2-phase approach where we
/// parse one or more expressions into an AST-like representation in a first
/// pass, then type check and generate IR for it in a second pass.  This enables
/// a number of features:
///
///   1) Non-lexical variable references: `[x.strip().upper() for x in flags]`
///   2) Weird order of evaluations: `foo() if cond() else bar()`
///   3) Parser ambiguity of the LHS of an assignment, which we don't know if it
///      is a target until we see the equals: `x[foo()] = bar()`
///   4) Contextually sensitive type checking, e.g. x = 42 where x is known to
///      be Int8 instead of Int.
///
/// We handle this by having an expression parser distinct from the main parser
/// that builds this tree and manages the lifetime of the nodes.  Only one
/// expression parser may be active at a time, which allows us to bump pointer
/// allocate the notes we create for the expression tree.
///
class ExprParser : public LitParserBase {
public:
  ExprParser(LitParserBase &existing)
      : LitParserBase(existing.lexer, existing.sharedParserState) {
    // Only a single expression parser can be active at a time, because we clear
    // the bump pointer allocator when done.
    assert(!sharedParserState->hasExprParser &&
           "Cannot create multiple expr parsers at once");
    sharedParserState->hasExprParser = true;
  }

  ~ExprParser() {
    assert(sharedParserState->hasExprParser);
    sharedParserState->hasExprParser = false;
    /// Free all the expression nodes.
    sharedParserState->exprAllocator.Reset();
  }

  // Expressions.  These methods always return a non-null ExprNode, but it may
  // be (or include) an Error node if parsing failed.
  void parseExpressionList(SmallVectorImpl<ExprNode *> &results);
  ExprNode *parseExpression();

  /// Emit the specified expression tree to MLIR in the current context.
  MLIRValueRep emit(ExprNode *node, Scope *scope);

private:
  /// Allocate an expression node into the expression bump pointer allocator.
  template <typename T, typename... Args>
  T *alloc(Args &&...args) {
    void *node = sharedParserState->exprAllocator.Allocate(
        sizeof(T), llvm::Align::Of<T>());
    return new (node) T(std::forward<Args>(args)...);
  }

  /// memcpy the specified ArrayRef into the expression allocator and return a
  /// pointer to the new data.  This cannot be used with things that have
  /// non-trivial copyctors/dtors because the expression allocator does run
  /// destructors.
  template <typename T>
  ArrayRef<T> copyArrayRef(ArrayRef<T> elements) {
    if (elements.empty())
      return elements;

    size_t dataSize = sizeof(T) * elements.size();
    T *result = static_cast<T *>(sharedParserState->exprAllocator.Allocate(
        dataSize, llvm::Align::Of<T>()));
    memcpy(result, elements.data(), dataSize);
    return ArrayRef<T>(result, elements.size());
  }
};

/// Emit the specified expression tree to MLIR in the current context.
MLIRValueRep ExprParser::emit(ExprNode *node, Scope *scope) {
  // TODO: Need a notion of a current builder that isn't just end of decl.
  auto builder = scope->getBuilder();
  EmitterState state{
      builder, scope,
      [&](SMLoc loc) -> Location { return translateLocation(loc); },
      [&](SMLoc loc, const Twine &twine) -> InFlightDiagnostic {
        return emitError(loc, twine);
      }};
  return node->emit(state);
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

/// expression_list ::= expression ("," expression)* [","]
void ExprParser::parseExpressionList(SmallVectorImpl<ExprNode *> &results) {
  // TODO: Support trailing comma for singleton tuple.
  (void)parseCommaSeparatedList([&]() -> ParseResult {
    results.push_back(parseExpression());
    return success();
  });
}

/// expression ::= atom | call
///
/// atom    ::= identifier | literal | enclosure [TODO]
/// call    ::=  primary "(" [argument_list [","] | comprehension] ")"
///
/// literal ::= [TODO]
///     stringliteral | bytesliteral | integer | floatnumber | imagnumber
///
ExprNode *ExprParser::parseExpression() {
  auto getErrorAtToken = [&]() -> ExprNode * {
    return alloc<ErrorNode>(getToken().getLoc());
  };

  // TODO: Handle precedence.
  ExprNode *result;
  switch (getToken().getKind()) {
  case LitToken::identifier: // expression -> atom -> identifier
    result = alloc<DeclRefNode>(getToken().getSpelling());
    consumeToken(LitToken::identifier);
    break;
  case LitToken::integer: // expression -> literal -> integer
    result = alloc<IntLiteralNode>(getToken().getSpelling());
    consumeToken(LitToken::integer);
    break;
  default:
    emitError("unexpected token in expression");
    result = getErrorAtToken();

    // TODO: Probably shouldn't consume this token in all cases, this could be
    // the introducer of another statement etc.  We should check to see what it
    // looks like and be smarter about this.
    consumeToken();
    break;
  }

  // Parse postfix productions.
  while (1) {
    // Handle calls.
    auto loc = getToken().getLoc();
    if (consumeIf(LitToken::l_paren)) {
      SmallVector<ExprNode *> argExprs;
      // TODO: Handle comprehension arguments.
      if (!consumeIf(LitToken::r_paren)) {
        parseExpressionList(argExprs);
        if (parseToken(LitToken::r_paren, "expected ')' in call argument list"))
          return getErrorAtToken();
      }

      result = alloc<CallNode>(result, loc, copyArrayRef<ExprNode *>(argExprs));
      continue;
    }
    break;
  }

  return result;
}

//===----------------------------------------------------------------------===//
// LitParser
//===----------------------------------------------------------------------===//

/// Declaration bodies are parsed after all the signatures at the current level
/// of the file are parsed.  This keeps track of
struct DeferredDeclBodyToParse {
  /// This is the scope for the declaration, which also contains the declaration
  /// itself.
  RCRef<Scope> declScope;

  /// This is where to start lexing the body from.
  LitLexerCursor lexerCursor;

  /// This is the indentation level of the decl.
  size_t indentLevel;
};

/// This class provides the implementation details of the concrete Lit Grammar.
struct LitParser : public LitParserBase {
  LitParser(LitLexer &lexer, SharedParserState *sharedParserState,
            ModuleOp module)
      : LitParserBase(lexer, sharedParserState), module(module) {}

  ParseResult parseFile();
  void finalizeScopeDecl();

private:
  // Statements.
  ParseResult parseSuite(size_t curIndent);
  ParseResult parseStmts(size_t minIndent);
  ParseResult parseStmt(size_t curIndent);
  ParseResult parseSimpleStmt();

  // Compound statements.
  ParseResult parseDefStmt(size_t curIndent);
  void parseDefBody(size_t defIndent, HLGeneratorOp defDecl);

  // Simple statements.
  ParseResult parseReturnStmt();
  ParseResult parseAssignmentStmt(ExprParser &exprParser, ExprNode *lhs,
                                  SMLoc equalsLoc);

private:
  const ModuleOp module;

  /// This is the current context that we're parsing into.
  RCRef<Scope> currentScope;

  /// These are deferred declarations that need parsing, which are processed
  /// after other things in a scope have been resolved.
  std::vector<DeferredDeclBodyToParse> deferredDecls;
};

/// file ::= statements
ParseResult LitParser::parseFile() {
  // The outermost scope contains the __builtins__ function definitions.
  // TODO: Add these:
  // https://docs.python.org/3/library/functions.html#built-in-funcs
  // https://docs.python.org/3/reference/executionmodel.html#naming-and-binding
  auto builtinsScope = RCRef<Scope>::create(module, RCRef<Scope>());

  // Create the module scope which will contain all things we parse.  These
  // shadow the builtins module during name lookup.
  currentScope = RCRef<Scope>::create(module, std::move(builtinsScope));

  // We fail either if we have a non-recoverable parse error, or if we emitted
  // an error and then recovered.  In either case, the IR will not be valid and
  // the caller should not verify it.
  if (parseStmts(/*indent=*/0))
    return failure();

  // Finalize the current scope, parsing any deferred declarations in it.
  finalizeScopeDecl();

  if (hadError())
    return failure();

  return success();
}

/// Finalize parsing of a scoped declaration (e.g. module, class, function).
///
/// Once its body is fully parsed, we loop back around to parse the bodies of
/// any nested scopes (e.g. nested functions) that are encountered while parsing
/// this scope.  This ensures that the forward references between peer
/// declarations are handled correctly, for example in mutually recursive
/// functions and code like this:
///
///   def foo():
///     def bar():
///       print(x)
///     x = 42
///     bar()
///   foo()
void LitParser::finalizeScopeDecl() {
  // We're done with the current scope and the declaration we're parsing into.
  currentScope.reset();
  if (deferredDecls.empty())
    return;

  // If we have deferred declarations, process each of them.
  std::vector<DeferredDeclBodyToParse> decls;
  std::swap(deferredDecls, decls);

  for (DeferredDeclBodyToParse &decl : decls) {
    currentScope = std::move(decl.declScope);
    decl.lexerCursor.restore(lexer);

    // Only support def's right now.
    parseDefBody(decl.indentLevel,
                 cast<HLGeneratorOp>(currentScope->getDecl()));
  }
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

/// Parse a suite, which is either a series of comma separated simple_stmt's on
/// one line, or an indented block of statements. curIndent is the containing
/// statement's indentation level.
///
/// suite     ::=  stmt_list NEWLINE | NEWLINE INDENT statement+ DEDENT
/// stmt_list ::=  simple_stmt (";" simple_stmt)* [";"]
ParseResult LitParser::parseSuite(size_t curIndent) {
  auto indent = getToken().getIndentation();
  // If there is a newline, then parse a list of statements.
  if (indent.has_value()) {
    if (indent.value() <= curIndent)
      emitError("body should be indented more than containing statement");
    return parseStmts(indent.value());
  }

  // Otherwise, parse a stmt_list.
  do {
    if (parseSimpleStmt())
      return failure();
    // Stop if we see a semicolon at the end of line or a missing semicolon.
  } while (consumeIf(LitToken::semi) &&
           !getToken().getIndentation().has_value());

  return success();
}

/// statements ::= statement+
///
/// This parses statements at the current indentation level or greater, it
/// refuses to parse things at lower indentation level.
ParseResult LitParser::parseStmts(size_t minIndent) {
  while (getToken().isNot(LitToken::eof)) {
    auto indent = getToken().getIndentation();
    if (!indent.has_value())
      return emitError("statements must start at the beginning of a line");
    if (indent.value() < minIndent)
      break;

    if (parseStmt(indent.value()))
      return failure();
  }
  return success();
}

/// statement ::= compound_stmt | simple_stmt
///
/// compound_stmt ::= if_stmt [TODO]
///                 | while_stmt [TODO]
///                 | for_stmt [TODO]
///                 | try_stmt [TODO]
///                 | with_stmt [TODO]
///                 | match_stmt [TODO]
///                 | funcdef
///                 | classdef [TODO]
///                 | async_with_stmt [TODO]
///                 | async_for_stmt [TODO]
///                 | async_funcdef [TODO]
///
ParseResult LitParser::parseStmt(size_t curIndent) {
  // Handle compound stmts here and chain to simple statements to handle the
  // whole "statement" production.
  switch (getToken().getKind()) {
  case LitToken::kw_def:
    return parseDefStmt(curIndent);
  default:
    // Otherwise must be a simple statement.
    return parseSimpleStmt();
  }
}

/// simple_stmt ::= expression_stmt
///               | assert_stmt [TODO]
///               | assignment_stmt
///               | augmented_assignment_stmt [TODO]
///               | annotated_assignment_stmt [TODO]
///               | pass_stmt
///               | del_stmt [TODO]
///               | return_stmt
///               | yield_stmt [TODO]
///               | raise_stmt [TODO]
///               | break_stmt [TODO]
///               | continue_stmt [TODO]
///               | import_stmt [TODO]
///               | future_stmt [TODO]
///               | global_stmt [TODO]
///               | nonlocal_stmtParseResult [TODO]
ParseResult LitParser::parseSimpleStmt() {
  switch (getToken().getKind()) {
  case LitToken::kw_pass:
    // pass_stmt ::= "pass"
    consumeToken(LitToken::kw_pass);
    return success();
  case LitToken::kw_return:
    return parseReturnStmt();
  default:
    break;
  }

  // Otherwise, we must have a statement that starts with the expression
  // grammar.

  // expression_stmt ::= starred_expression
  // assignment_stmt ::=
  //                 (target_list "=")+ (starred_expression | yield_expression)
  ExprParser exprParser(*this);
  ExprNode *expr = exprParser.parseExpression();

  // If the expression was followed by a `=` then we have an assignment.  If not
  // then we have an expression_stmt.
  SMLoc equalsLoc;
  if (consumeIf(LitToken::equal, &equalsLoc))
    return parseAssignmentStmt(exprParser, expr, equalsLoc);

  // Materialize the expression statement in our current scope but discard the
  // result on the floor.
  (void)exprParser.emit(expr, currentScope.getPointer());
  return success();
}

/// Parse an assignment_stmt after having parsed a leading expression (which
/// we need to resolve into a target_list) and an `=` sign.
///
/// assignment_stmt ::=
///                 (target_list "=")+ (starred_expression | yield_expression)
/// target_list     ::=  target ("," target)* [","]
/// target ::= identifier
///          | "(" [target_list] ")" | "[" [target_list] "]"
///          | attributeref | subscription | slicing | "*" target
///
ParseResult LitParser::parseAssignmentStmt(ExprParser &exprParser,
                                           ExprNode *lhs, SMLoc equalsLoc) {
  // Resolve the parse expression on the LHS into an lvalue that we can store
  // into.
  // TODO: implement support for generalized lvalues / target_list.
  auto dre = dyn_cast<DeclRefNode>(lhs);
  if (!dre) {
    if (!lhs->containsError())
      emitError(lhs->getLoc(), "cannot assign to expression");
    eatToEndOfLine();
    return success();
  }

  auto builder = currentScope->getBuilder();

  // Look up the name being assigned to if it already exists.
  Value lvalue;
  if (Operation *decl = currentScope->lookupInCurrentScope(dre->spelling)) {
    // Don't allow reassigning to functions and other declarations.
    // TODO: We actually just need type consistency.  If there were a reason to
    // need this, we could support reassignment.
    if (auto varDecl = dyn_cast<VarDeclOp>(decl))
      lvalue = varDecl;
    else
      emitError(lhs->getLoc(), "this declaration isn't reassignable");
  } else {
    // Otherwise, introduce a new lit.var.decl node.

    // TODO: Add types instead of hard coding to index type!
    auto declType = POP::PointerType::get(builder.getIndexType());

    // TODO: This will emit it on first use, which can be in weird places (e.g.
    // inside of the branch of an if statement).  We want to use dataflow
    // analysis to do definitive analysis of the accesses to the declaration. We
    // could just emit all these in the entry to the enclosing function/module
    // to maintain SSA.
    auto varDecl =
        builder.create<VarDeclOp>(translateLocation(lhs->getLoc()), declType,
                                  builder.getStringAttr(dre->spelling));
    (void)currentScope->addToScope(dre->spelling, varDecl);
    lvalue = varDecl;
  }

  ExprNode *rhs = exprParser.parseExpression();

  // Materialize the expression statement in our current scope.
  // TODO: Should pass in contextual type if known from previous declaration.
  auto rhsValue = exprParser.emit(rhs, currentScope.getPointer());

  // If everything worked out, store the resultant value into the lvalue for the
  // destination.  If things didn't work, just drop this on the floor.
  if (lvalue && rhsValue) {
    if (Value rhsValueValue = dyn_cast<Value>(rhsValue)) {
      if (!rhsValueValue.getType().isIndex())
        emitError(rhs->getLoc(), "TODO: don't support non-index types yet");
      else
        builder.create<POP::StoreOp>(translateLocation(equalsLoc),
                                     rhsValueValue, lvalue, /*alignment*/ None);
    } else {
      emitError(rhs->getLoc(),
                "TODO: don't support referring to parameters yet");
    }
  }

  return success();
}

/// funcdef ::=  [decorators] "def" funcname generic_signature?
///              "(" [parameter_list] ")"
///              ["->" expression] ":" suite
ParseResult LitParser::parseDefStmt(size_t curIndent) {
  auto loc = getTokenLocation();

  // TODO: Add support for decorators.
  consumeToken(LitToken::kw_def);

  auto parseParameter = [&]() -> ParseResult {
    // TODO: implement this correctly.
    return eatUntil({LitToken::comma, LitToken::r_paren});
  };

  auto parseGenericSignature = [&]() -> ParseResult {
    // TODO: implement this correctly.
    return eatUntil(LitToken::l_paren);
  };

  StringRef funcName = getToken().getSpelling();
  if (parseToken(LitToken::identifier, "expected function name") ||
      parseGenericSignature() ||
      parseToken(LitToken::l_paren, "expected '(' for parameter list"))
    return failure();

  if (!consumeIf(LitToken::r_paren)) {
    if (parseCommaSeparatedList(parseParameter) ||
        parseToken(LitToken::r_paren, "expected ')' for parameter list"))
      return failure();
  }

  if (consumeIf(LitToken::minus_greater)) {
    // TODO: Parse return type.
    if (eatUntil(LitToken::colon))
      return failure();
  }
  if (parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  auto builder = currentScope->getBuilder();
  auto nameAttr = builder.getStringAttr(funcName);
  auto functionType = builder.getFunctionType({}, {});
  auto symVisibility = builder.getStringAttr("public");
  // TODO: Should have nicer builder.
  auto newFunc = builder.create<HLGeneratorOp>(
      loc, nameAttr, symVisibility, TypeAttr::get(functionType),
      ParamDeclArrayAttr::get(getContext(), {}),
      TypeArrayAttr::get(getContext(), {}),
      ConstraintArrayAttr::get(getContext(), {}), FlatSymbolRefAttr());
  newFunc.getRegion().push_back(new Block());

  auto prevDecl = currentScope->addToScope(funcName, newFunc);
  if (prevDecl) {
    auto diag = emitError(loc, "invalid redefinition of function ") << nameAttr;
    diag.attachNote(prevDecl->getLoc()) << "previous definition here";
    // Keep parsing even though we failed to add to the scope.  Note that this
    // can cause type errors downstream.
    // TODO: We should mark both declarations erroneous in the symbol table so
    // reference to them get squashed as errors during name lookup, avoiding
    // cascading errors.
  }

  // We cannot parse the current body without having parsed other declarations
  // at the current level, so we defer parsing it.  Remember that we need to
  // do so.
  deferredDecls.push_back({RCRef<Scope>::create(newFunc, currentScope.copy()),
                           lexer.getCursor(), curIndent});

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the function definition.
  while (getToken().isNot(LitToken::eof)) {
    auto indent = getToken().getIndentation();
    if (indent.has_value() && indent.value() <= curIndent)
      break;
    consumeToken();
  }

  return success();
}

/// Parse a deferred 'def' body.
void LitParser::parseDefBody(size_t defIndent, HLGeneratorOp defDecl) {
  (void)parseSuite(defIndent);

  // Add kgen.return so the IR verifies.
  // TODO: Generalize lit.generator.
  auto returnParams = ArrayAttr::get(getContext(), {});
  OpBuilder::atBlockEnd(defDecl.getBody())
      .create<ReturnOp>(defDecl->getLoc(), returnParams, ArrayRef<Value>());

  finalizeScopeDecl();
}

/// return_stmt ::= "return" [expression_list]
ParseResult LitParser::parseReturnStmt() {
  consumeToken(LitToken::kw_return);

  SmallVector<ExprNode *> operands;

  // If there is an expression list present, parse it.
  if (!getToken().getIndentation().has_value()) {
    ExprParser exprParser(*this);
    exprParser.parseExpressionList(operands);
    // TODO: Resolve expressions.
  }

  return success();
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
                       KGENDialect>();

  // This is the result module we are parsing into.
  mlir::OwningOpRef<ModuleOp> module(ModuleOp::create(
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0)));

  SharedParserState sharedState(context);

  // Parse the file.
  LitLexer lexer(sourceMgr, context);
  if (LitParser(lexer, &sharedState, *module).parseFile())
    return nullptr;

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  auto verificationTimer = ts.nest("Verify module");
  if (failed(verify(*module)))
    return {};
  return module;
}
