//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements basic statement parsing.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/Lexer.h"
#include "KGEN/MojoParser/ParserBase.h"
#include "MojoUtils.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ToolCommon/CompilationOptions.h"

#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"
#include <filesystem>
#include <limits>

using namespace M::KGEN::LIT;
using namespace M::KGEN;
using namespace M;

//===----------------------------------------------------------------------===//
// Doc String support logic
//===----------------------------------------------------------------------===//

void ParserBase::parseDocString(ASTDecl &decl) {
  // The doc string is simply a follow-on string literal.
  Token docToken = getToken();
  if (!consumeIf(Token::string))
    return;
  if (auto astDeclOp = dyn_cast<ASTDeclInterface>(decl)) {
    StringRef docSpelling = docToken.getSpelling();
    Location loc = shared.diags.translateLocation(
        lexer.getStringLiteralStartLoc(docSpelling));

    astDeclOp.setDocStringAttr(DocStringAttr::get(
        StringAttr::get(getContext(), lexer.getStringLiteralValue(docSpelling)),
        dyn_cast<FileLineColLoc>(loc)));
  }
}

//===----------------------------------------------------------------------===//
// Decorator support logic
//===----------------------------------------------------------------------===//

/// Return true if this token is the start of a statement that should not exist
/// on the same line as a @decorator specification. This is used to improve
/// error recovery.
static bool isStatementThatMightHaveDecorators(Token::Kind tokenKind) {
  switch (tokenKind) {
  case Token::kw_if:
  case Token::kw_for:
  case Token::kw_while:
  case Token::kw_try:
  case Token::kw_with:
  case Token::kw_async:
  case Token::kw_def:
  case Token::kw_fn:
  case Token::kw_struct:
  case Token::kw_class:
  case Token::kw_from:
  case Token::kw_import:
  case Token::kw_pass:
  case Token::kw_var:
  case Token::kw_alias:
  case Token::kw___mlir_region:
  case Token::kw_return:
  case Token::kw_raise:
  case Token::kw_continue:
  case Token::kw_break:
    return true;
  default:
    return false;
  }
}

SmallVector<std::pair<ExprNode *, LexerCursor>>
ParserBase::parseDecorators(ASTDecl &decl) {
  return parseDecorators(decl.getIndentation());
}

/// Parse any decorators that may be present for a statement at the specified
/// indentation level.  Note that this must be kept in sync with the logic in
/// parseStmt which skips over things until the right indentation level.
SmallVector<std::pair<ExprNode *, LexerCursor>>
ParserBase::parseDecorators(ssize_t indentation) {
  SmallVector<std::pair<ExprNode *, LexerCursor>> result;

  auto stopOnStatement = [&]() -> bool {
    return isStatementThatMightHaveDecorators(getToken().getKind());
  };

  llvm::SMLoc atLoc;
  while (consumeIf(Token::at, &atLoc)) {
    if (getToken().isStartOfLine()) {
      emitError(atLoc, "missing decorator expression after '@'");
      skipUntilIndentation(indentation, /*stopOnSemicolon=*/false,
                           stopOnStatement);
      continue;
    }

    ExprNode *decoratorExpr;
    LexerCursor cursor = lexer.getCursor();
    if (parseExpression(decoratorExpr, indentation)) {
      skipUntilIndentation(indentation, /*stopOnSemicolon=*/false,
                           stopOnStatement);
      break;
    }
    result.push_back({decoratorExpr, cursor});

    if (!getToken().isStartOfLine() ||
        ssize_t(getToken().getIndentation().value()) > indentation) {
      emitTokenError("unexpected tokens after decorator, each need to be on "
                     "their own line");
      skipUntilIndentation(indentation, /*stopOnSemicolon=*/false,
                           stopOnStatement);
    }
  }
  // Decorators are applied to a decl starting from the one closest to it, so
  // reverse the vector.
  std::reverse(result.begin(), result.end());
  return result;
}

//===----------------------------------------------------------------------===//
// StmtParser
//===----------------------------------------------------------------------===//

/// This class provides the implementation details of the concrete Lightning
/// grammar.
namespace {
struct StmtParser : public ParserBase {
  StmtParser(SharedState &shared, Lexer &lexer, ASTDecl &curDeclScope)
      : ParserBase(shared, lexer), parentDecl(curDeclScope),
        curDeclScope(&curDeclScope), builder(curDeclScope.getDeclEndBuilder()) {

    // If we are parsing into a function, then we need a position to synthesize
    // variable definitions at the top of the function.
    // TODO: If we're parsing into top level code, we don't know how to do this.
    if (auto funcOp = dyn_cast<LIT::FuncOp>(getParentDecl())) {
      // The operation builder inserts before its insertion point, but for a
      // stable insertion point, keep the previous iterator position.
      varDeclCursor = OpBuilder(builder.getInsertionBlock(),
                                std::prev(builder.getInsertionPoint()));
    }
  }

  TypeCheckScopeInfo getScopeInfo() const { return {*curDeclScope, shared}; }

  ASTDecl &getParentDecl() { return parentDecl; }
  OpBuilder &getBuilder() { return builder; }
  UnresolvedType getUnresolvedType() {
    return UnresolvedType::get(getContext());
  }

  /// Push a debug info lexical block to represent a local variable scope.
  void pushLocalScope(DebugInfo::DIBuilder::ScopeGuard &scopeGuard);

  /// A local decl put in a scope when entering a new scope.
  /// The astDeclCallback field is called after constructing an ASTDecl.
  struct ScopeDecl {
    DeclIRValue value;
    SMLoc loc;
    StringRef name;
    function_ref<void(ASTDecl &)> astDeclCallback;
    ScopeDecl(DeclIRValue value, SMLoc loc, StringRef name)
        : value(value), loc(loc), name(name), astDeclCallback(nullptr) {}
    ScopeDecl(DeclIRValue value, SMLoc loc, StringRef name,
              function_ref<void(ASTDecl &)> astDeclCallback)
        : value(value), loc(loc), name(name), astDeclCallback(astDeclCallback) {
    }
  };

  // Expression emission.

  ExprEmitter getEmitter() {
    return ExprEmitter(shared, *curDeclScope, builder, varDeclCursor);
  }

  /// Get an expression emitter for a parameter expression.
  ExprEmitter getParamEmitter(ExprContext context) {
    return ExprEmitter(shared, *curDeclScope, context);
  }

  ParseResult parseSuite(ssize_t curIndent);
  void pushChildScope(DebugInfo::DIBuilder::ScopeGuard &scopeGuard,
                      llvm::SaveAndRestore<ASTDecl *> &keepDecl);
  ParseResult parseLocalScopeSuite(ssize_t curIndent,
                                   ArrayRef<ScopeDecl> decls = {});
  ParseResult parseStmt(bool onlySimpleStmt, bool &parsedCompound,
                        size_t stmtIndent);

  // Compound statements.
  ParseResult parseIfStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseElif(Location ifLoc, LexerCursor startCursor,
                        size_t curIndent);
  ParseResult parseParamIf(Location ifLoc, LexerCursor startCursor,
                           size_t curIndent);
  ParseResult parseWhileStmt(size_t curIndent);
  ParseResult parseForStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseForElse(size_t curIndent, ExprNode *seqExpr,
                           StringAttr target, SMLoc smLoc, SMLoc targetLoc);
  ParseResult parseParamFor(size_t curIndent, ExprNode *seqExpr,
                            StringAttr target, SMLoc smLoc, SMLoc targetLoc);
  ParseResult parseTryStmt(size_t curIndent);
  ParseResult parseWithStmt(size_t curIndent);
  ParseResult parseSingleWithStmt(size_t curIndent, SMLoc smLoc, Location loc);
  ParseResult
  handleRaisingFinallyRegion(TryOp tryOp, ASTType errorType, SMLoc loc,
                             function_ref<ParseResult()> populateFinallyBody);

  // Simple statements.
  ParseResult parseReturnStmt(size_t returnIndent);
  ParseResult parseRaiseStmt(size_t raiseIndent);
  ParseResult parseBreakOrContinueStmt(Token::Kind kind, StringRef name,
                                       StringRef opName);

  // Declarations.
  ParseResult parseFromImportStmt();
  ParseResult parseImportStmt();
  ParseResult
  parseImportModuleName(StringAttr &parsedName,
                        StringRef *nonIdentLeafModuleName = nullptr);
  ParseResult parseDefFnStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseStructStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseTraitStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseClassStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseVarStmt(LexerCursor startCursor, size_t stmtIndent);
  ParseResult parseAliasDeclStmt(LexerCursor startCursor, size_t stmtIndent);
  ParseResult parseMLIRRegionStmt(LexerCursor startCursor, size_t curIndent);

private:
  /// This is parent declaration / scope that we're parsing into.
  ASTDecl &parentDecl;
  /// This is the current declaration / scope.
  ASTDecl *curDeclScope;

  /// This is the builder that we are constructing IR into.
  OpBuilder builder;

  /// This is the insertion point we should install VarDecl's after if we are
  /// parsing into a 'def'.  This ensures they are emitted ahead of anything
  /// else in the region for the decl, and in decls with multiple regions (e.g.
  /// function bodies with if statements) it ensures the decl dominates the
  /// whole body.
  std::optional<OpBuilder> varDeclCursor;
};
} // namespace

void StmtParser::pushLocalScope(DebugInfo::DIBuilder::ScopeGuard &scopeGuard) {
  SMLoc curLoc = getToken().getLoc();
  auto &sourceMgr = getSourceMgr();
  unsigned bufferID = sourceMgr.FindBufferContainingLoc(curLoc);
  auto [line, column] = sourceMgr.getLineAndColumn(curLoc, bufferID);

  scopeGuard = shared.diBuilder->pushNestedLexicalBlock(
      shared.diBuilder->createFile(
          sourceMgr.getMemoryBuffer(bufferID)->getBufferIdentifier(), "/"),
      line, column);
}

/// Parse a suite, which is either a series of comma separated simple_stmt's on
/// one line, or an indented block of statements. curIndent is the containing
/// statement's indentation level.
///
/// suite     ::=  [stmt_list NEWLINE] | NEWLINE INDENT statement+ DEDENT
/// statement ::=  stmt_list NEWLINE | compound_stmt
/// stmt_list ::=  simple_stmt (";" simple_stmt)* [";"]
ParseResult StmtParser::parseSuite(ssize_t curIndent) {
  // Ignore empty body at end of file: a `pass` is not required.
  if (getToken().is(Token::eof))
    return success();

  /// This function parses a stmt_list, and if simpleStmtOnly is false, it
  /// also allows a compound statement.
  auto parseStmtListOrCompound = [&](bool stmtListOnly,
                                     size_t stmtIndent) -> ParseResult {
    do {
      bool parsedCompound = false;
      if (parseStmt(/*onlySimpleStmt=*/stmtListOnly, parsedCompound,
                    stmtIndent))
        return failure();

      // If we parsed a compound statement, then we don't allow trailing
      // semicolons after it.
      if (parsedCompound)
        return success();

      // Otherwise, we parsed a simple statement, which means no more compound
      // statements are allowed.
      stmtListOnly = true;

      // Continue if we see a semicolon that isn't at the end of the line.
    } while (consumeIf(Token::semi) && !getToken().isStartOfLine());
    return success();
  };

  // If this suite is on the same line as the enclosing entity, just parse a
  // single stmt_list.
  if (!getToken().isStartOfLine())
    return parseStmtListOrCompound(
        /*stmtListOnly=*/true,
        /*stmtIndent=*/std::numeric_limits<size_t>::max());

  ssize_t indent = getToken().getIndentation().value();

  // If there is a newline, then parse a list of statements which can be either
  // a statement list or a compound_stmt.  Parse all the statements that are
  // more nested than this suite, and reject it if there are none.
  if (indent <= curIndent) {
    emitError(getTokenLocOrEndOfPreviousLineIfOnNewLine())
        << "expected body statements; use 'pass' if none is required";
    return success();
  }

  // The first statement sets the expected indentation level for the whole body.
  auto bodyIndent = indent;
  SMLoc bodyIndentLoc = getToken().getLoc();
  while (getToken().isNot(Token::eof)) {
    if (!getToken().isStartOfLine())
      return emitTokenError("statements must start at the beginning of a line");

    indent = getToken().getIndentation().value();

    // If the indentation is less than we expect, then the suite is done.
    if (indent < bodyIndent)
      break;

    // Diagnose cases where the indentation is too great.
    if (indent > bodyIndent) {
      auto diag = emitError(getToken().getLoc())
                  << "statement has excess indentation";
      diag.attachNote(bodyIndentLoc)
          << "indentation should match previous statement";
    } else {
      bodyIndentLoc = getToken().getLoc();
    }

    if (parseStmtListOrCompound(/*stmtListOnly=*/false, indent))
      return failure();
  }
  return success();
}

void StmtParser::pushChildScope(DebugInfo::DIBuilder::ScopeGuard &scopeGuard,
                                llvm::SaveAndRestore<ASTDecl *> &keepDecl) {
  // If we are generating debug info, push a local scope
  if (shared.diBuilder)
    pushLocalScope(scopeGuard);

  // Push a new local variable scope.
  SMLoc loc;
  (void)getLocation(loc);
  curDeclScope = &getDeclResolver().addFullyResolvedDecl(nullptr, StringAttr(),
                                                         loc, curDeclScope);
}

ParseResult StmtParser::parseLocalScopeSuite(ssize_t curIndent,
                                             ArrayRef<ScopeDecl> decls) {
  DebugInfo::DIBuilder::ScopeGuard scopeGuard;
  llvm::SaveAndRestore<ASTDecl *> keepDecl(curDeclScope);
  // Push a new local variable scope for the subsequent suite.
  pushChildScope(scopeGuard, keepDecl);

  // Add the scope variables.
  for (const ScopeDecl &decl : decls) {
    ASTDecl &astDecl = getDeclResolver().addFullyResolvedDecl(
        decl.value, decl.name, decl.loc, curDeclScope);
    if (decl.astDeclCallback)
      decl.astDeclCallback(astDecl);
  }

  // Forward to the normal suite parse method.
  return parseSuite(curIndent);
}

/// Emit a warning when an expression is emitted at statement context, and it
/// returns a result.
static void diagnoseIgnoredResult(const ExprNode *expr, CValue value,
                                  SharedState &shared) {
  ASTType valueType = value.getRValueType();

  // Return true if the specified type can be implicitly ignored.
  // TODO(MOCO-32):
  //  Should have a better way to say that it is safe to
  //  implicitly ignore a value of a type (e.g. a type decorator)
  auto isImplicitlyIgnorableType = [&](ASTType type) -> bool {
    if (type.isNoneType() || type.isEqualCanon(shared.getTypeCheckErrorType()))
      return true;

    // Allow object/PythonObject to be ignored.  This should really be
    // implemented with a decorator on the type, not hard coded here.
    auto declRef = dyn_cast<LIT::StructType>(type.mlirType);
    if (!declRef || !declRef.getParamValues().empty())
      return false;

    // object is implicitly returned by 'def's, PythonObject is pervasive in
    // interop.
    StringRef name = declRef.getName().getValue();

    return name == "object" || name == "PythonObject" || name == "NoneType";
  };

  if (isImplicitlyIgnorableType(valueType) ||
      // The `x = y` operation returns a borrowed version of its operand but its
      // result can be ignored.
      expr->kind == ExprNode::kAssign)
    return;

  // If this type is a function with no formal arguments and an ignorable type,
  // we emit a warning with a fix it hint suggesting that it get called.
  // TODO: This is incorrect for default arguments and varargs.
  if (auto sig = dyn_cast<SignatureType>(valueType)) {
    // Get the result type without any error handling in the way.
    Type resultType = ASTType(sig).getSignatureUserResultType();
    if ((sig.getNumArguments() ==
         ((unsigned)sig.hasMemoryOnlyResult() + (unsigned)sig.isThrows())) &&
        isImplicitlyIgnorableType(resultType)) {
      shared.emitWarning(expr->getLoc())
          << "function pointer was formed but not called, did you forget '()'s?"
          << expr->getRange()
          << FixIt::insertAfterToken(expr->getRange().getEnd(), "()",
                                     shared.diags);
      return;
    }
  }

  // If the expression returned an unawaited value, then the expression should
  // be awaited. Check for an '__await__' function.
  if (shared.typeHasMember(valueType, "__await__", expr->getLoc())) {
    shared.emitWarning(expr->getLoc())
        << "awaitable " << valueType << " value was never awaited"
        << expr->getRange()
        << FixIt::insertBeforeToken(expr->getRangeStart(), "await ");
    return;
  }

  // Otherwise emit a warning, and suggest assigning to _.
  auto startLoc = expr->getRange().getStart();
  shared.emitWarning(expr->getLoc())
      << valueType << " value is unused" << expr->getRange()
      << FixIt::insertBeforeToken(startLoc, "_ = ");
}

/// When `onlySimpleStmt` is true, this parses the simple_stmt production,
/// otherwise it parses the broader `statement` production that includes
/// compound statements.  This sets `parsedCompound` to true if
/// `onlySimpleStmt` was false and we parsed a compound stmt.
///
/// statement ::= compound_stmt | simple_stmt
///
/// compound_stmt ::= if_stmt
///                 | while_stmt
///                 | for_stmt
///                 | try_stmt
///                 | with_stmt
///                 | match_stmt [TODO]
///                 | funcdef
///                 | structdef
///                 | classdef [TODO]
///                 | async_with_stmt [TODO]
///                 | async_for_stmt [TODO]
///                 | async_funcdef [TODO]
///
/// simple_stmt ::= expression_stmt
///               | assert_stmt [TODO]
///               | alias_decl_stmt
///               | var_decl_stmt
///               | assignment_stmt
///               | augmented_assignment_stmt
///               | annotated_assignment_stmt [TODO]
///               | pass_stmt
///               | del_stmt [TODO]
///               | return_stmt
///               | yield_stmt [TODO]
///               | raise_stmt [TODO]
///               | break_stmt [TODO]
///               | continue_stmt [TODO]
///               | import_stmt
///               | future_stmt [TODO]
///               | global_stmt [TODO]
///               | nonlocal_stmtParseResult [TODO]
///
ParseResult StmtParser::parseStmt(bool onlySimpleStmt, bool &parsedCompound,
                                  size_t stmtIndent) {
  // Generate pretty stack traces if a crash happens in this scope.
  LexerCrashReporter crashReporter(getLexer(), "parsing statement");

  // This is the cursor for the start of the declaration, that will be used in
  // the signature resolution phase.
  LexerCursor startCursor = getLexer().getCursor();

  // This emits an error message if we parsed a decorator, because this
  // statement doesn't support them.
  auto rejectDecorator = [&]() {
    if (startCursor == getLexer().getCursor())
      return;
    emitTokenError() << "'" << getToken().getSpelling()
                     << "' statement does not allow decorators";
  };

  // This lambda is used to generate an error when a compound statement is used
  // in a scenario that expects simple statements.
  auto rejectSimpleStmt = [&]() {
    parsedCompound = true; // Tell the caller that we parsed a compound stmt.
    if (!onlySimpleStmt)
      return;
    emitTokenError() << "'" << getToken().getSpelling()
                     << "' statement must be on its own line";
  };

  auto emitTopLevelViolationError = [&]() {
    Operation *parent = parentDecl.getIfOperation();
    if (!parent)
      return;
    if (isa<LIT::FileModuleOp>(parent))
      emitTokenError() << "'" << getToken().getSpelling()
                       << "' must be contained in a function but is contained "
                          "in a file scope.";
  };

  // Skip over any decorators that are present.  These will be reparsed during
  // signature resolution phase of a declaration.
  while (consumeIf(Token::at)) {
    auto stopOnStatement = [&]() -> bool {
      return isStatementThatMightHaveDecorators(getToken().getKind());
    };

    skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/false,
                         stopOnStatement);

    // If the next token isn't indented, but is the start of a statement, then
    // these decorators are incorrectly on the same line as the statement.
    // Reject with a specific error message and ignore the whole thing.
    if (!getToken().isStartOfLine() && stopOnStatement()) {
      emitError(startCursor.getToken().getLoc())
          << "decorators must be on their own line, not ahead of a statement";
      // Skip the body of the statement entirely.
      skipUntilIndentation(stmtIndent);
      return success();
    }

    // If the next token is for a less indented declaration, then this is a
    // floating decorator not necessarily attached to it.  Ignore the
    // decorators and let the outer level of the parser keep finding stuff.
    // This leads to better error recovery.
    if (getToken().isStartOfLine() &&
        getToken().getIndentation().value() < stmtIndent) {
      emitError(startCursor.getToken().getLoc())
          << "orphaned decorator not associated with a declaration or "
             "statement";
      return success();
    }
  }

  switch (getToken().getKind()) {
    //===------------------------------------------------------------------===//
    // Compound statements.
    //===------------------------------------------------------------------===//
  case Token::kw_if:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseIfStmt(startCursor, stmtIndent);
  case Token::kw_for:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseForStmt(startCursor, stmtIndent);
  case Token::kw_while:
    rejectDecorator();  // Decorators not allowed.
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseWhileStmt(stmtIndent);
  case Token::kw_try:
    rejectDecorator(); // Decorators not allowed.
    rejectSimpleStmt();
    emitTopLevelViolationError();
    return parseTryStmt(stmtIndent);
  case Token::kw_with:
    rejectDecorator(); // Decorators not allowed.
    rejectSimpleStmt();
    return parseWithStmt(stmtIndent);
  case Token::kw_async:
  case Token::kw_def:
  case Token::kw_fn:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseDefFnStmt(startCursor, stmtIndent);
  case Token::kw_struct:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseStructStmt(startCursor, stmtIndent);
  case Token::kw_trait:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseTraitStmt(startCursor, stmtIndent);
  case Token::kw_class:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseClassStmt(startCursor, stmtIndent);

    //===------------------------------------------------------------------===//
    // Simple statements.
    //===------------------------------------------------------------------===//
  case Token::kw_from:
    rejectDecorator(); // Decorators not allowed.
    return parseFromImportStmt();
  case Token::kw_import:
    rejectDecorator(); // Decorators not allowed.
    return parseImportStmt();

  case Token::kw_pass:
  case Token::dot_dot_dot:
  case Token::string:
    // pass_stmt ::= "pass"
    // doc string
    consumeToken();
    return success();
  case Token::kw_var:
    return parseVarStmt(startCursor, stmtIndent);
  case Token::kw_alias:
    return parseAliasDeclStmt(startCursor, stmtIndent);
  case Token::kw___mlir_region:
    rejectDecorator();
    rejectSimpleStmt();
    return parseMLIRRegionStmt(startCursor, stmtIndent);
  case Token::kw_return:
    rejectDecorator(); // Decorators not allowed.
    return parseReturnStmt(stmtIndent);
  case Token::kw_raise:
    rejectDecorator(); // Decorators not allowed.
    return parseRaiseStmt(stmtIndent);
  case Token::kw_continue:
    rejectDecorator(); // Decorators not allowed.
    return parseBreakOrContinueStmt(Token::kw_continue, "continue",
                                    LIT::ContinueOp::getOperationName());
  case Token::kw_break:
    rejectDecorator(); // Decorators not allowed.
    return parseBreakOrContinueStmt(Token::kw_break, "break",
                                    LIT::BreakOp::getOperationName());
  default:
    break;
  }

  // Parse a single expression, an assignment stmt, or augmented assignment
  // statement.
  ExprNode *expr = nullptr;
  Operation *parent = parentDecl.getIfOperation();
  // TODO: Top level expressions will be supported in the future.
  if (parent && isa<LIT::FileModuleOp>(parent)) {
    emitTokenError()
        << "TODO: expressions are not yet supported at the file scope level";
  }
  if (parseSimpleStmtExprs(expr, stmtIndent))
    return failure();

  // Emit the expression and ignore the results.  If it is an assignment
  // statement, it will return None.  Other expressions can return whatever they
  // will naturally return.
  auto emitter = getEmitter();
  // Result is ignored, so we don't care where it goes.
  CValue result = emitter.emitExprCValue(expr, EC_TopLevelStmt);
  if (!result)
    return success();

  // Emit a warning if the result is a value we should warn when unused.
  diagnoseIgnoredResult(expr, result, shared);
  return success();
}

//===----------------------------------------------------------------------===//
// Simple statements.
//===----------------------------------------------------------------------===//

/// return_stmt ::= "return" [expression_list]
ParseResult StmtParser::parseReturnStmt(size_t returnIndent) {
  auto func = dyn_cast<LIT::FuncOp>(getParentDecl());
  auto loc = consumeToken(Token::kw_return).getLoc();

  // If there is an expression list present, parse it.
  ExprNode *operandExpr = nullptr;
  SimpleLiteralNode noneExpr(ExprNode::kNoneLiteral, loc);
  if (!isTokenInCurrentStatement(returnIndent))
    operandExpr = &noneExpr;
  else if (parseExpressionList(operandExpr, returnIndent))
    return failure();

  // Ok, now that we parsed all the tokens for this statement, do semantic
  // analysis.
  if (!func) {
    emitError(loc, "cannot return from this context");
    return success();
  }

  auto emitter = getEmitter();

  // Materialize the expression values into IR.
  AnyValue resultValue;
  SignatureType declSig = func.getSignature();
  ASTType userResultType = func.getUserResultType();

  // If the result is memory-only, return into the result slot, otherwise we
  // just need a value of the right type. If the function has a named result
  // slot, we allow it to omit the expression.
  ValueDest resultDest(userResultType, EC_ReturnValue);
  if (func.getNamedResultAttr() && operandExpr == &noneExpr)
    resultDest = ValueDest(shared.getNoneType(), EC_ReturnValue);
  else if (declSig.hasMemoryOnlyResult())
    resultDest = ValueDest(MLValue(func.getArguments().back()), EC_ReturnValue);

  if (!declSig.isRefResult()) {
    // Convert the returned value to the returned type of the function.
    resultValue = emitter.emitExpr(operandExpr, resultDest);
    if (!resultValue) {
      resultDest.resetForError();
      return {};
    }
  } else {
    // When returning a reference, emit it as an MValue then coerce.
    auto resultCValue = emitter.emitExprCValue(
        operandExpr, EC_ReturnValue, userResultType.getReferenceElementType());
    if (!resultCValue) {
      resultDest.resetForError();
      return {};
    }

    Value refValue =
        emitter.emitRefValue({resultCValue, operandExpr}, EC_ReturnValue);
    if (!refValue) {
      resultDest.resetForError();
      return {};
    }
    RefType argType = cast<RefType>(refValue.getType());

    // If the lifetime is an InvalidRefLifetimeAttr then this value is
    // derived from an argument which might be bound (after elaboration)
    // to a register value that has no lifetime.  Emit an error because
    // you can't return a reference to it.
    if (isa<InvalidRefLifetimeAttr>(argType.getLifetime())) {
      emitError(operandExpr->getLoc())
          << "cannot return a reference to an argument that "
             "might instantiate to @register_passable type "
          << ASTType(argType.getElementType()) << operandExpr->getRange();
      resultDest.resetForError();
      return {};
    }

    // We already checked the element type, check the lifetime and address
    // space.
    if (!userResultType.isEqualCanon(argType)) {
      if (!canConvertWithRebind(argType, userResultType, shared)) {
        auto expectedRefType = cast<RefType>(userResultType);
        auto diag = emitter.emitError(operandExpr->getLoc())
                    << "cannot return reference with incompatible ";
        if (argType.getLifetime() != expectedRefType.getLifetime())
          diag << "lifetime: " << argType.getLifetime() << " vs "
               << expectedRefType.getLifetime();
        else {
          assert(argType.getAddressSpace() !=
                     expectedRefType.getAddressSpace() &&
                 "Only lifetime and address space can disagree given the "
                 "element types agree");
          diag << "address space: " << argType.getAddressSpace() << " vs "
               << expectedRefType.getAddressSpace();
        }
        resultDest.resetForError();
        return {};
      }
      // Rebind to make the reference compatible, e.g. converting to a more
      // general lifetime union.
      refValue =
          emitter.rebindValue({SRValue(refValue), operandExpr}, userResultType)
              .getIfSRValue();
    }

    // We're returning the reference itself, so switch to SRValue.
    resultValue = SRValue(refValue);
    // ... and emit to the ValueDest
    resultValue = emitter.emitRValue({resultValue, operandExpr}, resultDest);
    if (!resultValue) {
      resultDest.resetForError();
      return success();
    }

    if (declSig.hasMemoryOnlyResult())
      resultValue = {};
  }

  // If the result is a memory-only result, then handle the scalar result.
  if (declSig.hasMemoryOnlyResult()) {
    // The register result is a None value, or false if it throws.
    if (func.isThrows())
      resultValue = PValue(BoolAttr::get(getContext(), false));
    else
      resultValue = PValue(shared.getNoneAttr());
  }

  if (!resultValue)
    return {};

  // The normal return type in a raising initializer is an i1.
  if (declSig.hasInitSelfArg() && declSig.isThrows())
    resultValue = PValue(BoolAttr::get(getContext(), false));

  auto resultVal = emitter.emitSRValue(
      {resultValue, operandExpr}, EC_ReturnValue, func.getMLIRResultType());
  if (!resultVal)
    return {};
  ImplicitLocOpBuilder b(translateLocation(loc), builder);
  ExprEmitter::emitNormalReturn(b, resultVal, getParentDecl());
  return success();
}

/// Given an insertion point in a block, scan up the parent hierarchy to see if
/// this block is nested under a try.  If so, return that operation and whether
/// this block is nested within the 'except' part of the operation.  If
/// we are currently in the 'else' part of a try, we keep scanning since the try
/// isn't relevant.
static std::pair<TryOp, bool> findParentTry(Block *currentBlock) {
  while (Operation *parentOp = currentBlock->getParentOp()) {
    // If we hit the top of the function we aren't nested.
    if (isa<LIT::FuncOp>(parentOp))
      break;

    // If this is a try, determine which region we're in.

    TryOp tryOp = dyn_cast<TryOp>(parentOp);
    if (tryOp) {
      if (&tryOp.getTryRegion().front() == currentBlock)
        return {tryOp, false};
      if (&tryOp.getExceptRegion().front() == currentBlock)
        return {tryOp, true};

      // Must be in the else, which doesn't stop propagation.
      assert(&tryOp.getElseRegion().front() == currentBlock);
    }

    // If this is not a try op, keep scanning.
    currentBlock = parentOp->getBlock();
  }

  // Didn't find a try.
  return {TryOp(), false};
}

/// Inject a call to a special method that the debugger stops at when
/// supporting exception/error breakpoints.
static LogicalResult injectDebuggerRaiseHookCall(SharedState &shared,
                                                 ExprEmitter &&emitter,
                                                 ASTDecl &declContext,
                                                 llvm::SMLoc loc,
                                                 ExprNode *node) {
  ArrayRef<ASTDecl *> raiseHookFns = shared.getBuiltinFunction(
      declContext, "builtin.error", "__mojo_debugger_raise_hook", loc);
  if (raiseHookFns.empty())
    return failure();

  ParamBindings bindings(TypeCheckScopeInfo{declContext, shared});
  OverloadSet call("__mojo_debugger_raise_hook", raiseHookFns,
                   std::move(bindings), node, CallSyntax::kDirectCall);
  ValueDest raseHookDest(EC_RaiseValue);
  call.emitCall({}, raseHookDest, emitter);
  return success();
}

ParseResult StmtParser::parseRaiseStmt(size_t raiseIndent) {
  llvm::SMRange loc = consumeToken(Token::kw_raise).getLocRange();

  ExprNode *errorExpr = nullptr;
  // If there is an error expression, parse it.
  if (isTokenInCurrentStatement(raiseIndent) &&
      parseExpression(errorExpr, raiseIndent))
    return failure();

  // TODO: Support "from" exception chaining.

  // Ok, we are syntactically sound.  Check to see if we're in a try block, and
  // (if so) whether we are in.  Python's notion of a current exception is fully
  // dynamic, which we don't support yet.  For now, we only support 'raise' with
  // no expression in the 'except' block of a 'try'.
  //
  //    def foo(): raise   # Rethrow any currently-being-handled exception
  //    try:
  //      print(1/0)
  //    except Exception as exc:
  //      print("hello")
  //      foo()   # rethrows the caught exception
  //

  // Find the nearest error slot if the parser is in a context that can raise.
  MLValue errSlot = getEmitter().findNearestErrorSlot();
  if (!errSlot) {
    InflightDiag diag =
        emitError(loc.Start, "cannot raise error in this context") << loc;
    diag.attachNote(loc.Start) << "try surrounding 'raise' in a 'try' block";
    if (auto func =
            getBlockParentOfType<LIT::FuncOp>(builder.getInsertionBlock()))
      diag.attachNote(func.getLoc())
          << "or mark surrounding function as 'raises'";
    return success();
  }
  ValueDest dest(errSlot, EC_RaiseValue);
  if (errorExpr) {
    // If we had an error, emit it.
    getEmitter().emitExpr(errorExpr, dest);
  } else {
    // Figure it if we're in a try, and if so, which subregion.
    auto [tryOp, inExceptRegion] = findParentTry(builder.getInsertionBlock());

    // Otherwise, we must be in the 'except' part of the try block and are
    // rethrowing the current error.  This isn't correct Python semantics, see
    // the caveat above.
    if (!inExceptRegion) {
      InflightDiag diag = emitError(loc.Start, "no contextual error to reraise")
                          << loc;
      diag.attachNote(loc.Start) << "provide an error to raise or place "
                                    "'raise' statement inside an except region";
      dest.resetForError();
      return success();
    }

    // Re-raise the contextual exception.
    getEmitter().emitResult(MRValue(tryOp.getErr()), SyntheticNode(loc.Start),
                            dest);
  }

  // If we are in a debug build, we inject a call to a stop hook for the
  // debugger right before a RaiseOp.
  if (shared.options.debugLevel !=
      CompilationOptions::DebugInfoLevel::kNoDebug) {
    if (failed(injectDebuggerRaiseHookCall(
            shared, getEmitter(), getParentDecl(), loc.Start, errorExpr)))
      return failure();
  }

  builder.create<LIT::RaiseOp>(translateLocation(loc.Start));
  return success();
}

/// break_stmt ::= "break"
/// continue_stmt ::= "continue"
ParseResult StmtParser::parseBreakOrContinueStmt(Token::Kind kind,
                                                 StringRef name,
                                                 StringRef opName) {
  llvm::SMLoc loc = consumeToken(kind).getLoc();

  // We diagnose break/continue that are not in a loop in LowerSemanticCF.

  // Split the block at the insertion point. Any subsequent statements are dead
  // code. Let region DCE handle it.
  OperationState state(translateLocation(loc), opName);
  builder.create(state);
  return success();
}

//===----------------------------------------------------------------------===//
// Compound statements.
//===----------------------------------------------------------------------===//

/// while_stmt ::=  "while" assignment_expression ":" suite
///                 ["else" ":" suite]
ParseResult StmtParser::parseWhileStmt(size_t curIndent) {
  Location whileLoc = translateLocation(consumeToken(Token::kw_while).getLoc());

  ExprNode *condExp = nullptr;
  if (parseAssignExpression(condExp, curIndent))
    return failure();

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  // Create the LoopOp
  auto loopOp = builder.create<LIT::LoopOp>(whileLoc);
  Block *condBlock = builder.createBlock(&loopOp.getCondRegion());
  Block *bodyBlock = builder.createBlock(&loopOp.getBodyRegion());
  Block *elseBlock = builder.createBlock(&loopOp.getElseRegion());

  // Create the condition region.
  builder = OpBuilder::atBlockEnd(condBlock);
  RValue condRVal = getEmitter().emitExprI1(condExp, EC_BoolCondition);
  Value condVal =
      getEmitter().emitSRValue({AnyValue(condRVal), condExp}, EC_BoolCondition);

  // After the condition is evaluated, validate the end of the statement.
  if (parseToken(Token::colon, "expected ':' after expression"))
    return failure();
  if (!condVal)
    return success(); // IRGen error already emitted; parse succeeded!
  builder.create<LIT::LoopConditionOp>(whileLoc, condVal);

  // Create the body region.
  builder.setInsertionPointToStart(bodyBlock);

  if (failed(parseLocalScopeSuite(curIndent)))
    return failure();
  builder.create<LIT::LoopContinueOp>(whileLoc);

  // Create the else region.
  builder.setInsertionPointToStart(elseBlock);
  // The 'else' block is executed only when the condition check fails.
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    if (parseToken(Token::colon, "expected ':' after else") ||
        parseLocalScopeSuite(curIndent))
      return failure();
  }
  builder.create<LIT::LoopYieldOp>(whileLoc);

  return success();
}

/// for_stmt ::=  "for" target_list "in" starred_list ":" suite
///              ["else" ":" suite]
ParseResult StmtParser::parseForStmt(LexerCursor startCursor,
                                     size_t curIndent) {
  // This is enabled with the @parameter decorator.
  bool isParamFor = false;

  // We parse the decorators for the 'for' if they exist.
  if (startCursor != getLexer().getCursor()) {
    startCursor.restore(getLexer());
    for (auto [decorator, cursor] : parseDecorators(curIndent)) {
      // Handle recognized decorators.
      if (auto *dre = dyn_cast<DeclRefNode>(decorator)) {
        if (dre->spelling == "parameter") {
          isParamFor = true;
          continue;
        }
      }

      emitError(decorator->getLoc(), "unsupported decorator on 'for' statement")
          << decorator->getRange();
    }
  }

  SMLoc smLoc = consumeToken(Token::kw_for).getLoc();

  // parse [target_list] in [starred_list]
  // for now, we expect target_list to be an identifier
  // the [starred_list] needs to be a sequence with a __iter__ method that
  // returns a type that defines __len__ and __next__
  StringAttr target = StringAttr::get(getContext(), getToken().getSpelling());

  // FIXME: This needs to parse this as a target expression and then handle it
  // like a destructuring pattern.
  SMLoc targetLoc;
  if (!consumeIf(Token::kw__, &targetLoc))
    if (parseIdentifier("expected identifier for target in 'for'", &targetLoc))
      return failure();
  if (parseToken(Token::kw_in, "expected 'in' after target identifier. Note "
                               "that target lists are not yet supported."))
    return failure();

  ExprNode *seqExpr = nullptr;
  if (parseExpression(seqExpr) ||
      parseToken(Token::colon, "expected ':' after expression"))
    return failure();

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  if (isParamFor)
    return parseParamFor(curIndent, seqExpr, target, smLoc, targetLoc);
  return parseForElse(curIndent, seqExpr, target, smLoc, targetLoc);
}

ParseResult StmtParser::parseParamFor(size_t curIndent, ExprNode *seqExpr,
                                      StringAttr target, SMLoc smLoc,
                                      SMLoc targetLoc) {
  Location forLoc = translateLocation(targetLoc);
  ASTDecl &scope = getParentDecl();

  // Emit the sequence as a PValue parameter.
  PValue seqPValue = getEmitter().emitExprPValue(seqExpr, EC_ForParamSeq);
  if (!seqPValue)
    return failure();

  // Bind the sequence initial value to the parameter for iterator generator.
  // Start by looking up the builtin generator.
  ArrayRef<ASTDecl *> paramForImpl = shared.getBuiltinFunction(
      scope, "builtin._stubs", "parameter_for_generator", smLoc);
  if (paramForImpl.empty())
    return failure();

  // Resolve the overload with the sequence's type. This succeeds if the
  // iterator type is currently supported.
  ParamBindings bindings(TypeCheckScopeInfo{scope, shared});
  bindings.add(seqExpr, PValue(seqPValue.getType()));
  OverloadSet call("parameter_for_generator", paramForImpl, std::move(bindings),
                   seqExpr, CallSyntax::kDirectCall);
  PValue iterate = call.getDirectSymbol(/*expectedType=*/{});
  if (!iterate)
    return failure();

  // Sniff the type of the induction variable and create its declaration.
  // TODO: Expand the induction variable to support other types.
  Type intType = shared.lookupNamedType("Int", scope, smLoc);
  if (!intType)
    return failure();
  auto indVarDecl =
      ParamDeclAttr::get(scope.mangleParamName(target.getValue()), intType);

  // Create the loop and parse the body into it.
  auto paramFor =
      builder.create<ParamForOp>(forLoc, seqPValue, iterate, indVarDecl);
  builder.createBlock(&paramFor.getBody());
  if (parseLocalScopeSuite(curIndent,
                           ScopeDecl{PValue(ParamDeclRefAttr::get(indVarDecl)),
                                     targetLoc, target}))
    return failure();
  builder.create<ParamForContinueOp>(forLoc);

  // Parse the else region if present.
  builder.createBlock(&paramFor.getElseRegion());
  // The 'else' block is executed only when the condition check fails.
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    if (parseToken(Token::colon, "expected ':' after else") ||
        parseLocalScopeSuite(curIndent))
      return failure();
  }
  builder.create<ParamYieldOp>(forLoc);

  // Advance the insertion point.
  builder.setInsertionPointAfter(paramFor);
  return success();
}

ParseResult StmtParser::parseForElse(size_t curIndent, ExprNode *seqExpr,
                                     StringAttr target, SMLoc smLoc,
                                     SMLoc targetLoc) {
  Location forLoc = translateLocation(targetLoc);

  // Create a VarDeclOp for the induction variable.  We infer its type from the
  // call to __next__ down below.
  VarDeclOp indvarDeclOp = getEmitter().emitVarDecl(
      target, getUnresolvedType(), forLoc, VarDeclKind::Implicit);

  bool isInvalid = false;
  auto notifyVarDecl = [&](ASTDecl &decl) {
    getEmitter().shared.notifyListenerOnVariableDecl(decl, targetLoc);
    if (isInvalid)
      decl.setErroneous();
  };

  auto indvarScopeDecl =
      ScopeDecl{&*indvarDeclOp, targetLoc, target, notifyVarDecl};

  // If there is a failure before we parse the for loop body, we still want to
  // call the parser on it so that it builds an ASTDecl node and adds the for
  // loop VarDecl to the lookup path.  Otherwise, we will get spurious “use of
  // unknown declaration” errors on it besides whatever error is raised while
  // processing the loop header.
  auto avoidDroppingDeclOnFail = llvm::make_scope_exit([&]() {
    isInvalid = true;
    (void)parseLocalScopeSuite(curIndent, indvarScopeDecl);
  });

  // retrieve the iterator object from the sequence expression
  ASTExprAnd<AnyValue> loadedSeq = {
      getEmitter().emitExpr(seqExpr, EC_ForIterator), seqExpr};
  if (!loadedSeq.ir)
    return {};

  // Emit a call to __iter__ into a var with an inferred type.
  VarDeclOp rangeRef = getEmitter().emitVarDecl(
      "$RANGE", getUnresolvedType(), forLoc, VarDeclKind::Synthesized);
  ValueDest rangeDest(rangeRef, EC_ForIterator);
  if (!getEmitter().emitNamedMethodCall("__iter__", {loadedSeq}, rangeDest,
                                        CallSyntax::kImplicitConvert,
                                        seqExpr)) {
    auto newRefType =
        indvarDeclOp.getType().getWithElement(shared.getTypeCheckErrorType());
    indvarDeclOp.getResult().setType(newRefType);
    return {};
  }

  // Create the LoopOp
  auto loopOp = builder.create<LIT::LoopOp>(forLoc);
  Block *condBlock = builder.createBlock(&loopOp.getCondRegion());
  Block *bodyBlock = builder.createBlock(&loopOp.getBodyRegion());
  Block *elseBlock = builder.createBlock(&loopOp.getElseRegion());

  // Create the condition region.
  builder = OpBuilder::atBlockEnd(condBlock);

  // For Loop condition: if the length of the range is greater than zero,
  // continue. Otherwise break
  ValueDest lengthDest(EC_ForIterator);
  AnyValue currentLength = getEmitter().emitNamedMethodCall(
      "__len__", CallOperands({{MLValue(rangeRef), seqExpr}}), lengthDest,
      CallSyntax::kImplicitConvert, seqExpr);
  CValue lengthIndex =
      getEmitter().emitMLIRIndex({currentLength, seqExpr}, EC_ForIterator);
  if (!lengthIndex)
    return {};
  SRValue length =
      getEmitter().emitSRValue({lengthIndex, seqExpr}, EC_ForIterator);
  if (!length)
    return {};
  Value shouldContinue = builder.create<mlir::index::CmpOp>(
      forLoc, mlir::index::IndexCmpPredicate::SGT, length,
      builder.create<mlir::index::ConstantOp>(forLoc, 0));
  builder.create<LIT::LoopConditionOp>(forLoc, shouldContinue);

  // Create the body. Add Target element to the continue block by calling next
  // method. Emit the result into an implicitly declared variable at the current
  // scope.
  builder.setInsertionPointToStart(bodyBlock);
  ValueDest indvarDest(indvarDeclOp, EC_ForIterator);
  if (!getEmitter().emitNamedMethodCall(
          "__next__", CallOperands({{MLValue(rangeRef), seqExpr}}), indvarDest,
          CallSyntax::kImplicitConvert, seqExpr))
    return {};

  avoidDroppingDeclOnFail.release();

  // Parse the body of the for loop into a new scope, pushing the iterator
  // variable into that new scope.
  if (failed(parseLocalScopeSuite(curIndent, indvarScopeDecl)))
    return failure();
  builder.create<LIT::LoopContinueOp>(forLoc);

  // Create the else region.
  builder.setInsertionPointToStart(elseBlock);
  // The 'else' block is executed only when the condition check fails.
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    if (parseToken(Token::colon, "expected ':' after else") ||
        parseLocalScopeSuite(curIndent))
      return failure();
  }
  builder.create<LIT::LoopYieldOp>(forLoc);
  return success();
}

/// The finally block executes whenever control flow leaves any of the other
/// regions of a `try`, whether through a yield, return, or raise. Its overall
/// control flow effect takes precedence over how control flow left the other
/// `try` regions originally.
///
/// This is conceptually implemented by branching before said exits to the
/// finally region, and when yielding from the finally region, branch back to
/// where control was before.
///
/// However, this means that the finally region can conditionally overwrite an
/// error slot and then choose not to raise, causing this issue:
///
/// ```
/// def raising_finally():
///     try:
///         raise Error() # initializes %__error__, then branch to 'finally'
///     finally:
///         # conservatively destroy %__error__ before the raising call
///         might_raise()
///         # if the call didn't raise, branch back to where we were in 'try',
///         # but now we have an error return with an uninitialized %__error__!
/// ```
///
/// Thus, the error slot in the finally region cannot alias the one used in the
/// other regions, because it might conditionally overwrite it while it still
/// needs to be used. Fix this by rewriting the above into:
///
/// ```
/// def raising_finally():
///     try:
///         raise Error()
///     finally:
///         try:
///             might_raise()
///         except:
///             raise
/// ```
ParseResult StmtParser::handleRaisingFinallyRegion(
    TryOp tryOp, ASTType errorType, SMLoc loc,
    function_ref<ParseResult()> populateFinallyBody) {
  if (tryOp.hasTrivialFinally())
    return success();

  MLValue errSlot = getEmitter().findNearestErrorSlot();
  if (!errSlot)
    return populateFinallyBody();

  VarDeclOp errDecl = getEmitter().emitVarDecl(
      "__finally_error__", errorType, tryOp.getLoc(), VarDeclKind::Synthesized);
  auto nestedTry =
      builder.create<TryOp>(tryOp.getLoc(), errDecl, /*suppressWarnings=*/true);

  // Stub out the else and finally regions of this try.
  builder.createBlock(&nestedTry.getElseRegion());
  builder.create<TryYieldOp>(tryOp.getLoc());
  builder.createBlock(&nestedTry.getFinallyRegion());
  builder.create<TryYieldOp>(tryOp.getLoc());

  // Move the error into the overall error slot.
  builder.createBlock(&nestedTry.getExceptRegion());
  ValueDest moveDest(errSlot, EC_RaiseValue);
  getEmitter().emitResult(MRValue(errDecl), SyntheticNode(loc), moveDest);
  builder.create<RaiseOp>(tryOp.getLoc());
  builder.create<TryYieldOp>(tryOp.getLoc());

  Block *tryBlock = builder.createBlock(&nestedTry.getTryRegion());
  if (populateFinallyBody())
    return failure();
  builder.setInsertionPointToEnd(tryBlock);
  builder.create<TryYieldOp>(tryOp.getLoc());
  builder.setInsertionPointAfter(nestedTry);
  return success();
}

/// try_stmt ::= "try" ":" suite "except" [identifier] ":" suite
///              ["else" suite]
ParseResult StmtParser::parseTryStmt(size_t curIndent) {
  SMLoc smLoc = consumeToken(Token::kw_try).getLoc();
  Location loc = translateLocation(smLoc);

  if (parseToken(Token::colon, "expected ':' after 'try'"))
    return failure();

  // If we see a 'try' block in a context that cannot raise, we need to check if
  // the user explicitly provided an 'except' region, otherwise this is a
  // try-finally block where the try block cannot raise.
  bool inExceptRegion = !!getEmitter().findNearestErrorSlot();
  if (!inExceptRegion) {
    Lexer subLexer(shared.diags, lexer.getCursor());
    ParserBase subParser(shared, subLexer);
    subParser.skipUntilIndentation(curIndent);
    inExceptRegion = subParser.consumeIf(Token::kw_except);
  }

  // Restore the builder to its current insertion point after parsing.
  llvm::SaveAndRestore builderSaver(builder);
  ASTType errorType = shared.getBuiltinErrorType(getParentDecl(), smLoc);
  VarDeclOp errDecl = getEmitter().emitVarDecl("__try_error__", errorType, loc,
                                               VarDeclKind::Synthesized);
  auto tryOp = builder.create<TryOp>(loc, errDecl);
  if (!inExceptRegion) {
    builder.createBlock(&tryOp.getExceptRegion());
    builder.create<UnreachableOp>(loc);
  }

  // Parse the try suite.
  builder.createBlock(&tryOp.getTryRegion());
  if (parseLocalScopeSuite(curIndent))
    return failure();
  builder.create<TryYieldOp>(translateLocation(getToken().getLoc()));

  SMLoc errValLoc = getToken().getLoc();
  bool hasFinally = false;
  if (getToken().is(Token::kw_except)) {
    errValLoc = consumeToken().getLoc();

    // Parse an optional identifier to bind the error.
    StringAttr errName;
    if (getToken().isIdentifier())
      (void)parseIdentifier(errName, "<this can't fail>", &errValLoc);

    if (parseToken(Token::colon, "expected ':' after 'except'"))
      return failure();

    builder.createBlock(&tryOp.getExceptRegion());

    // If an identifier was declared for the error value, add a declaration that
    // references it.
    SmallVector<ScopeDecl> decls;
    if (errName) {
      // If the user bound the error to a name, adjust the vardecl and add the
      // declaration.
      errDecl.setName(errName);
      errDecl.setKind(VarDeclKind::Var);
      errDecl->setLoc(translateLocation(errValLoc));
      decls.push_back(ScopeDecl{DeclIRValue(errDecl), errValLoc, errName});
    }

    // Parse the except suite.
    if (parseLocalScopeSuite(curIndent, decls))
      return failure();
    builder.create<TryYieldOp>(translateLocation(getToken().getLoc()));

    // Parse the else suite if present. Otherwise, leave it as empty.
    builder.createBlock(&tryOp.getElseRegion());
    if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
        consumeIf(Token::kw_else)) {
      if (parseToken(Token::colon, "expected ':' after 'else'") ||
          parseLocalScopeSuite(curIndent))
        return failure();
    }
    builder.create<TryYieldOp>(translateLocation(getToken().getLoc()));

    hasFinally = consumeIf(Token::kw_finally);
  } else {
    hasFinally = consumeIf(Token::kw_finally);
    if (!hasFinally)
      return emitTokenError("expected 'except' or 'finally' block");
    // In a raising context, the default 'except' block just forwards the error.
    if (inExceptRegion) {
      builder.createBlock(&tryOp.getExceptRegion());
      MLValue errSlot = getEmitter().findNearestErrorSlot();
      ValueDest dest(errSlot, EC_RaiseValue);
      getEmitter().emitResult(MRValue(errDecl), SyntheticNode(smLoc), dest);
      builder.create<LIT::RaiseOp>(loc);
      builder.create<TryYieldOp>(loc);
    }

    // Stub the 'else' region.
    builder.createBlock(&tryOp.getElseRegion());
    builder.create<TryYieldOp>(loc);
  }
  builder.createBlock(&tryOp.getFinallyRegion());
  if (hasFinally) {
    if (handleRaisingFinallyRegion(tryOp, errorType, smLoc, [&] {
          if (parseToken(Token::colon, "expected ':' after 'finally'") ||
              parseLocalScopeSuite(curIndent))
            return failure();
          return mlir::success();
        }))
      return failure();
  }
  builder.create<TryYieldOp>(loc);

  return success();
}

/// with_stmt ::=
///    "with" ( "(" with_stmt_contents ","? ")" | with_stmt_contents ) ":" suite
/// with_stmt_contents ::=  with_item ("," with_item)*
/// with_item          ::=  expression ["as" target]
ParseResult StmtParser::parseWithStmt(size_t curIndent) {
  SMLoc smLoc = consumeToken(Token::kw_with).getLoc();
  Location loc = shared.translateLocation(smLoc);

  return parseSingleWithStmt(curIndent, smLoc, loc);
}

/// Parses a single clause in the `with` statement, and possibly the body as
/// well.
/// This could recurse if there are multiple clauses in the `with` statement,
/// like:
///     with MyClass() as a, MyClass() as b:
///         ...
/// In that case, it interprets it as multiple nested "single" with statements,
/// like:
///     with MyClass() as a:
///         with MyClass() as b:
///             ...
/// This function handles just the `MyClass() as a`, then for everything
/// afterward it either recurses (for other clauses) or calls out to
/// `parseLocalScopeSuite` (for the body).
ParseResult StmtParser::parseSingleWithStmt(size_t curIndent, SMLoc smLoc,
                                            Location loc) {
  // With statements are just sugar for other constructs.  We desugar this:
  //     with EXPRESSION as TARGET:
  //       SUITE
  // Into:
  //     contextMgr = EXPRESSION
  //     TARGET = contextMgr.__enter__()
  //     try {
  //       SUITE
  //     } except(errorVal : Error) {
  //       hlcf.if (contextMgr.__exit__(errorVal)) {
  //         hlcf.yield
  //       } else {
  //         raise errorVal
  //       }
  //       try.yield
  //     } else {
  //       contextMgr.__exit__()
  //     }
  // We elide the try and except logic when in a context that doesn't support
  // raising an error (like a non-raising fn).

  // Parse and emit the context mgr.
  ExprNode *contextExp = nullptr;
  if (parseExpression(contextExp))
    return failure();

  // Emit the context manager expression into a var with an inferred type.
  VarDeclOp contextMgrDecl = getEmitter().emitVarDecl(
      "$CONTEXTMGR", getUnresolvedType(),
      shared.translateLocation(contextExp->getLoc()), VarDeclKind::Synthesized);
  ValueDest contextMgrDest(contextMgrDecl, EC_WithContextMgr);
  if (!getEmitter().emitExpr(contextExp, contextMgrDest))
    return failure();

  // Determine if the context manager has an __exit__ method.  If not, that is
  // fine, we silently just don't call it.  This mode of supporting context
  // managers with just an __enter__ method is useful for strong Mojo types
  // working with context managers even if they don't need them, e.g. we want
  // file descriptors to support both of these patterns:
  //
  //    with open("foo.txt", "r") as f:
  //        print(f.read())
  //
  // and:
  //    let f = open("foo.txt", "r")
  //    print(f.read())
  //
  // The later works because of Mojo's strong early-destruction guarantees and
  // lack of frame-objects-capturing-variables problems, but the former is more
  // familiar to Pythonistas.

  // In erroneous code, ASTDecl may be missing, e.g. a 'with' on an MLIR type.
  ASTType contextRVType = MLValue(contextMgrDecl).getRValueType();
  bool hasExitMethod =
      shared.typeHasMember(contextRVType, "__exit__", contextExp->getLoc());

  // Determine whether we're in a region that is allowed to raise.  If so,
  // generate logic to deal with it.
  MLValue errSlot = getEmitter().findNearestErrorSlot();
  bool inExceptRegion = !!errSlot;

  // If this has a 'as TARGET' specifier, parse the name into targetName,
  // otherwise targetName will be null.
  std::optional<DeclRefNode> targetNode;
  if (consumeIf(Token::kw_as)) {
    // FIXME: This needs to parse this as a target expression and then handle it
    // like a destructuring pattern.
    targetNode.emplace(DeclRefNode(getToken().getSpelling(),
                                   getToken().is(Token::escaped_identifier)));
    if (parseIdentifier("expected identifier for target in 'with'"))
      return failure();
  }

  // We are about to generate the call to __enter__ but need to decide how to
  // pass the context expression, either as an LValue referring to the bound
  // variable, or as a transferred RValue if it takes it owned (enabling some
  // advanced use cases with unique context managers).
  AnyValue contextVal = MLValue(contextMgrDecl);

  // Interrogate the caller to see what convention the first argument to the
  // __enter__ method is.  Be careful about invalid cases - the errors will get
  // diagnosed when emitting the method call.
  CallOperands enterOperands;
  enterOperands.addSelf({contextVal, contextExp});
  auto enterEmitter = getEmitter();
  if (PValue enterMethod = OverloadSet::lookupAndResolve(
          contextRVType, "__enter__", enterOperands, contextExp,
          CallSyntax::kMethodCall, enterEmitter)) {
    // If there is no exit method, we can pass the argument as an RValue so the
    // enter method can consume the value... unless __enter__ takes self inout.
    if (auto signature = dyn_cast<SignatureType>(enterMethod.getType());
        signature && !signature.getArgConventions().empty()) {
      auto firstArgConvention = signature.getArgConventions()[0];
      if (firstArgConvention != ArgConvention::InOut && !hasExitMethod)
        contextVal = MRValue(contextMgrDecl);

      // One error that people hit is defining a context manager with both an
      // owned enter method and an exit method.  This will generate a terrible
      // error message in check lifetimes, so cut that off here.
      if ((firstArgConvention == ArgConvention::OwnedInReg ||
           firstArgConvention == ArgConvention::OwnedInMem) &&
          hasExitMethod) {
        auto diag =
            emitError(contextExp->getLoc(), "context manager of type ")
            << contextRVType
            << " defines a consuming __enter__ method as well as an __exit__ "
               "method; either remove 'owned' from its '__enter__' method or "
               "remove the '__exit__' method"
            << contextExp->getRange();
        if (ASTDecl *contextDecl = contextRVType.getDecl(shared))
          diag.attachNote(contextDecl->getLoc())
              << contextRVType << " declared here";

        // Make the emission work even if the type isn't copyable.
        contextVal = MRValue(contextMgrDecl);
      }
      enterOperands[0].ir = contextVal;
    }
  }

  // If we are in a def, we need to use function scoping.  If we are in a fn,
  // we need to use lexical scope.  When we support `with` at the top level, we
  // should decide whether it is lexical or global scope.  This largely depends
  // on our view of what `python superset` or `python++` means.
  bool useLexicalScope = true;
  if (auto funcDecl = curDeclScope->getNearestDeclOfType<LIT::FuncOp>())
    useLexicalScope = !cast<LIT::FuncOp>(*funcDecl).isDef();

  // If there is an explicit target specified, use it.
  ValueDest enterDest(EC_WithContextMgr);
  VarDeclOp targetDecl;
  if (targetNode.has_value()) {
    if (useLexicalScope) {
      auto name = StringAttr::get(getContext(), targetNode->spelling);
      auto emitter = getEmitter();
      targetDecl = emitter.emitVarDecl(name, getUnresolvedType(),
                                       targetNode->getLocation(emitter),
                                       VarDeclKind::Implicit);
      enterDest = ValueDest(targetDecl, EC_WithContextMgr);
    } else {
      // If we're in a 'def' just use the DeclRefNode as the destination. This
      // ensures that we reuse and/or implicitly declare variables at the top
      // level of the function, just like "x = foo()" does for "x".
      enterDest = ValueDest(&*targetNode, EC_WithContextMgr);
    }
  }

  // Emit the call to __enter__ and (if 'as TARGET' was specified), bind to
  // result to a named TARGET vardecl, inferring its type.
  CValue enterResult = getEmitter().emitNamedMethodCall(
      "__enter__", std::move(enterOperands), enterDest, CallSyntax::kMethodCall,
      contextExp);

  DebugInfo::DIBuilder::ScopeGuard scopeGuard;
  llvm::SaveAndRestore<ASTDecl *> keepDecl(curDeclScope);
  if (useLexicalScope)
    pushChildScope(scopeGuard, keepDecl);

  // Inject the target into our scope if asked for.
  if (targetDecl) {
    auto &targetDeclResolved = getDeclResolver().addFullyResolvedDecl(
        targetDecl.getOperation(), targetDecl.getNameAttr(),
        targetNode->getLoc(), curDeclScope);
    if (!enterResult)
      targetDeclResolved.setErroneous();
    shared.notifyListenerOnVariableDecl(targetDeclResolved,
                                        targetNode->getIdentifierLoc());
  }

  // Lookup the error type and emit a vardecl for the error.
  ASTType errorType = shared.getBuiltinErrorType(getParentDecl(), smLoc);
  VarDeclOp errDecl = getEmitter().emitVarDecl("__with_error__", errorType, loc,
                                               VarDeclKind::Synthesized);

  // Restore the builder to its current insertion point after parsing.
  llvm::SaveAndRestore builderSaver(builder);
  auto tryOp = builder.create<TryOp>(loc, errDecl, /*suppressWarnings=*/true);
  // Stub the 'except' and 'else' regions.
  builder.createBlock(&tryOp.getExceptRegion());

  // If the body of this try can throw, then the "except" block in it needs to
  // catch the current exception and then re-raise it.
  if (inExceptRegion) {
    ValueDest dest(errSlot, EC_RaiseValue);
    getEmitter().emitResult(MRValue(errDecl), contextExp, dest);
    builder.create<LIT::RaiseOp>(loc);
    builder.create<TryYieldOp>(loc);
  } else {
    // Otherwise it will be unreachable.
    builder.create<UnreachableOp>(loc);
  }
  builder.createBlock(&tryOp.getElseRegion());
  builder.create<TryYieldOp>(loc);
  builder.createBlock(&tryOp.getTryRegion());

  // Check if the context manager provides an `__exit__` overload that accepts
  // an error. If it doesn't, then we know the exit is unconditional.
  CallOperands exitCallOperands;
  exitCallOperands.addSelf({contextVal, contextExp});
  exitCallOperands.add({PValue(UnknownAttr::get(errorType)), contextExp});
  PValue conditionalExit;
  if (inExceptRegion && hasExitMethod) {
    ExprEmitter exitEmitter = getEmitter();
    conditionalExit = OverloadSet::lookupAndResolve(
        contextRVType, "__exit__", exitCallOperands, contextExp,
        CallSyntax::kMethodCall, exitEmitter);
  }

  // Otherwise, we have to emit a conditional finally. PEP343 states that the
  // general 'with' statement corresponds to:
  //   contextMgr = EXPRESSION
  //   TARGET = contextMgr.__enter__()
  //   exc = True
  //   try:
  //     try:
  //       SUITE
  //     except e:
  //       exc = False
  //       if not contextMgr.__exit__(e):
  //         raise e
  //   finally:
  //     if exc:
  //       contextMgr.__exit__()
  Value excVar;
  TryOp nestedTryOp;
  VarDeclOp nestedErrDecl;
  if (conditionalExit) {
    // Insert the flag and initialize it to 'True'.
    OpBuilder::InsertPoint ip = builder.saveInsertionPoint();
    builder.setInsertionPoint(tryOp);
    excVar = getEmitter().emitVarDecl("__with_exc__", builder.getI1Type(), loc,
                                      VarDeclKind::Synthesized);
    builder.create<RefStoreOp>(
        loc, builder.create<mlir::index::BoolConstantOp>(loc, true), excVar);
    builder.restoreInsertionPoint(ip);

    // Generate the nested try. Stub the 'else' and 'finally' regions.
    nestedErrDecl = getEmitter().emitVarDecl("__inner_error__", errorType, loc,
                                             VarDeclKind::Synthesized);
    nestedTryOp =
        builder.create<TryOp>(loc, nestedErrDecl, /*suppressWarnings=*/true);
    builder.create<TryYieldOp>(loc);
    builder.createBlock(&nestedTryOp.getElseRegion());
    builder.create<TryYieldOp>(loc);
    builder.createBlock(&nestedTryOp.getFinallyRegion());
    builder.create<TryYieldOp>(loc);

    // Parse the body into the try region.
    builder.createBlock(&nestedTryOp.getTryRegion());
  }

  if (consumeIf(Token::comma)) {
    // We get here if the `with` statement had multiple clauses, like:
    //     with MyClass() as a, MyClass() as b:
    //         ...
    // so recurse to handle them and interpret it as:
    //     with MyClass() as a:
    //         with MyClass() as b:
    //             ...
    // The base case of this recursion call will also handle parsing the body
    // suite for us, so our current call doesn't have to worry about that.
    if (parseSingleWithStmt(curIndent, smLoc, loc))
      return success();
    builder.create<TryYieldOp>(loc);
  } else if (consumeIf(Token::colon)) {
    if (parseLocalScopeSuite(curIndent))
      return failure();
    builder.create<TryYieldOp>(loc);
  } else {
    // DO NOT SUBMIT Should we be calling a helper method here?
    auto message = "expected ':' or ',' after 'with' expression";
    auto diagLoc = getTokenLocOrEndOfPreviousLineIfOnNewLine();
    // Report the error.
    auto diag = emitError(diagLoc, message);
    return failure();
  }

  // This emits the call to the 'contextMgr.__exit__()' methods on the
  // context managers in the normal path.  If the type has no __exit__ method,
  // then we extend the result of the __enter__ method with this pattern:
  //
  //   TARGET = contextMgr.__enter__()
  //   try:
  //     SUITE
  //   finally:
  //     lit.ownership.use(TARGET)
  auto emitNormalExitLogic = [&]() {
    if (hasExitMethod) {
      ValueDest exitDest(EC_WithExitResult);
      (void)getEmitter().emitNamedMethodCall(
          "__exit__", CallOperands({{MLValue(contextMgrDecl), contextExp}}),
          exitDest, CallSyntax::kMethodCall, contextExp);
    } else if (auto targetBV = getEmitter().emitBValue(
                   {enterResult, contextExp}, ExprContext::EC_WithContextMgr)) {
      // If the target value has no __exit__ method, we need it to be
      // live all the way across the suite, so add an extra use so it isn't
      // destroyed early.
      // We don't care about extending PValues if one ever happened.
      if (Value ptrOrScalar = enterResult.getMlirValue())
        builder.create<OwnershipUseOp>(loc, ptrOrScalar);
    }
  };

  // If we're in a non-raising region (or have no __exit__ method), then we have
  // a simple pattern to emit:
  //   contextMgr = EXPRESSION
  //   TARGET = contextMgr.__enter__()
  //   try:
  //     SUITE
  //   finally:
  //     contextMgr.__exit__()
  if (!inExceptRegion || !hasExitMethod) {
    builder.createBlock(&tryOp.getFinallyRegion());
    emitNormalExitLogic();
    builder.create<TryYieldOp>(loc);
    return success();
  }

  if (conditionalExit) {
    // Set up the except region.  Pseudo code:
    //  except(%val : Error) {
    //    hlcf.if (

    builder.createBlock(&nestedTryOp.getExceptRegion());

    // Set the flag to 'False'.
    builder.create<RefStoreOp>(
        loc, builder.create<mlir::index::BoolConstantOp>(loc, false), excVar);

    // Pass the error value to the __exit__ method.
    // TODO: this isn't using the same convention that Python does.  We support
    // overloading though and this is going to be way better for anything real
    // that wants to implement this. We can support both styles when we need to.
    ValueDest exitResultDest(EC_WithExitResult);
    CallOperands exitOperandList({{MLValue(contextMgrDecl), contextExp},
                                  {MBValue(nestedErrDecl), contextExp}});
    CValue exitResult = getEmitter().emitIndirectCall(
        conditionalExit, std::move(exitOperandList), exitResultDest,
        CallSyntax::kMethodCall, contextExp);
    RValue exitI1RVal =
        getEmitter().emitI1({exitResult, contextExp}, EC_WithExitResult);
    SRValue exitI1Val =
        getEmitter().emitSRValue({exitI1RVal, contextExp}, EC_WithExitResult);
    if (!exitI1Val)
      // Fail, but non-fatal so return success to keep parsing.
      return success();
    // If __exit__ returns false, then re-raise the error.
    auto ifOp = builder.create<HLCF::IfOp>(loc, exitI1Val);
    builder.create<TryYieldOp>(loc);

    builder.createBlock(&ifOp.getThenRegion());
    // On true, nothing is to be done.
    builder.create<HLCF::YieldOp>(loc);

    // On false, we re-raise the error.
    builder.createBlock(&ifOp.getElseRegion());
    ValueDest dest(MLValue(tryOp.getErr()), EC_RaiseValue);
    getEmitter().emitResult(MRValue(nestedErrDecl), contextExp, dest);
    builder.create<LIT::RaiseOp>(loc);
    builder.create<HLCF::YieldOp>(loc);
  }

  // Emit the conditional call to __exit__.
  builder.createBlock(&tryOp.getFinallyRegion());
  (void)handleRaisingFinallyRegion(tryOp, errorType, smLoc, [&] {
    HLCF::IfOp excIf;
    if (conditionalExit) {
      excIf = builder.create<HLCF::IfOp>(
          loc, builder.create<RefLoadOp>(loc, excVar));
      builder.createBlock(&excIf.getThenRegion());
    }
    emitNormalExitLogic();
    if (conditionalExit) {
      builder.create<HLCF::YieldOp>(loc);
      // Stub the 'else' region.
      builder.createBlock(&excIf.getElseRegion());
      builder.create<HLCF::YieldOp>(loc);
    }
    return success();
  });

  builder.create<TryYieldOp>(loc);
  return success();
}

ParseResult StmtParser::parseParamIf(Location ifLoc, LexerCursor startCursor,
                                     size_t curIndent) {
  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);
  ExprNode *condExp = nullptr;
  if (parseAssignExpression(condExp, curIndent))
    return failure();

  // Each if/elif conditions could be dynamic or static, use some helpers to
  // generate the right structure.
  ParamIfOp paramIfOp;
  auto parseCondAndTerminateElifCondition = [&](Location loc) -> ParseResult {
    // For a @parameter if we emit the condition as an PValue
    // without a builder.
    RValue condRVal = getParamEmitter(EC_BoolParamCondition)
                          .emitExprI1(condExp, EC_BoolParamCondition);
    if (!condRVal)
      return failure();
    PValue condPVal = condRVal.getIfPValue();
    if (!condPVal)
      return emitError(condExp->getLoc(), "@parameter 'if' requires a "
                                          "parameter expression as a condition")
             << condExp->getRange();

    paramIfOp = builder.create<ParamIfOp>(loc, condPVal.get());
    return success();
  };

  if (parseCondAndTerminateElifCondition(ifLoc) ||
      parseToken(Token::colon, "expected ':' after 'if' expression"))
    return failure();
  builder.createBlock(&paramIfOp.getThenRegion());
  if (failed(parseLocalScopeSuite(curIndent)))
    return failure();
  builder.create<ParamYieldOp>(ifLoc);

  while (getToken().is(Token::kw_elif) &&
         isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true)) {
    Location elifLoc = translateLocation(consumeToken(Token::kw_elif).getLoc());
    if (parseAssignExpression(condExp, std::nullopt))
      return failure();

    // Moves emission into "Condition" block if elif.
    builder.createBlock(&cast<ParamIfOp>(paramIfOp).getElseRegion());

    if (parseCondAndTerminateElifCondition(elifLoc) ||
        parseToken(Token::colon, "expected ':' after 'elif' expression"))
      return failure();

    builder.create<ParamYieldOp>(elifLoc);
    builder.createBlock(&paramIfOp.getThenRegion());
    if (failed(parseLocalScopeSuite(curIndent)))
      return failure();
    builder.create<ParamYieldOp>(elifLoc);
  }

  builder.createBlock(&cast<ParamIfOp>(paramIfOp).getElseRegion());
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    if (parseToken(Token::colon, "expected ':' after else"))
      return failure();
    if (failed(parseLocalScopeSuite(curIndent)))
      return failure();
  }
  builder.create<ParamYieldOp>(ifLoc);
  return success();
}

struct DeadCodeInfo {
  /// The value of the constant condition.
  bool conditionValue;

  /// The location of the constant condition block.
  Location location;

  /// The index of the condition region within the ElifOp.
  unsigned index;
};

ParseResult StmtParser::parseElif(Location ifLoc, LexerCursor startCursor,
                                  size_t curIndent) {
  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  // Create a new elifOp state and initialize it with 2 blocks.
  HLCF::ElifOp elifOp = builder.create<HLCF::ElifOp>(ifLoc, TypeRange(), 2);
  elifOp.getElifRegions()[0].emplaceBlock();
  elifOp.getElifRegions()[1].emplaceBlock();

  auto parseCondition =
      [&](Location loc) -> std::pair<ParseResult, std::optional<DeadCodeInfo>> {
    unsigned indexOfCondition = elifOp.getElifRegions().size() - 2;
    Block &conditionBlock = elifOp.getElifRegions()[indexOfCondition].front();
    builder.setInsertionPointToStart(&conditionBlock);
    auto emitter = getEmitter();

    ExprNode *condExp = nullptr;
    if (parseAssignExpression(condExp, curIndent))
      return {failure(), {}};

    // Create the 'elif' and parse the body into its "then" region.
    RValue condI1RVal = emitter.emitExprI1(condExp, EC_BoolCondition);
    if (!condI1RVal)
      return {failure(), {}};
    std::optional<bool> knownConditionForWarning = {};
    if (PValue condI1PVal = condI1RVal.getIfPValue();
        IntegerAttr asIntAttr =
            dyn_cast_or_null<IntegerAttr>(condI1PVal.get())) {
      knownConditionForWarning = !asIntAttr.getValue().isZero();
    }
    SRValue condRVal =
        emitter.emitSRValue({condI1RVal, condExp}, EC_BoolCondition);
    if (!condRVal)
      return {failure(), {}};

    // Terminate the condition region of the current ElifOp.
    builder.create<HLCF::ElifYieldOp>(loc, condRVal);

    std::optional<DeadCodeInfo> deadCodeInfo = {};
    if (knownConditionForWarning.has_value()) {
      deadCodeInfo = {knownConditionForWarning.value(),
                      condExp->getLocation(emitter), indexOfCondition};
    }
    return {success(), deadCodeInfo};
  };

  auto appendElifRegionPair = [&]() {
    // We need to add two regions.
    builder.setInsertionPoint(elifOp);
    mlir::IRRewriter rewriter{builder};
    HLCF::ElifOp replacement =
        builder.create<HLCF::ElifOp>(elifOp.getLoc(), elifOp->getResultTypes(),
                                     elifOp.getElifRegions().size() + 2);

    // Take previously parsed regions from old op.
    for (auto [index, source] : llvm::enumerate(elifOp.getElifRegions()))
      replacement.getElifRegions()[index].takeBody(source);

    // Add another (Condition, Then) pair.
    Region &lastConditionRegion =
        replacement.getElifRegions()[replacement.getElifRegions().size() - 2];
    Region &lastThenRegion = replacement.getElifRegions().back();
    lastConditionRegion.emplaceBlock();
    lastThenRegion.emplaceBlock();

    // Replace the original elif with the expanded elif.
    rewriter.replaceOp(elifOp, replacement);
    elifOp = replacement;
  };

  // Vector of unreachable code metadata.  After emitting code, these need to
  // raise warnings and be marked as dead.
  SmallVector<DeadCodeInfo> ifOpsWithDeadCode;
  auto [ifParseResult, maybeDeadCodeInfo] = parseCondition(ifLoc);
  if (maybeDeadCodeInfo.has_value())
    ifOpsWithDeadCode.push_back(maybeDeadCodeInfo.value());
  if (ifParseResult ||
      parseToken(Token::colon, "expected ':' after 'if' expression"))
    return failure();
  // Parse Then region.
  builder.setInsertionPointToStart(&elifOp.getElifRegions().back().front());
  if (failed(parseLocalScopeSuite(curIndent)))
    return failure();
  builder.create<HLCF::YieldOp>(ifLoc);

  // Parse Elif chain if it exists.
  while (getToken().is(Token::kw_elif) &&
         isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true)) {
    Location elifLoc = translateLocation(consumeToken(Token::kw_elif).getLoc());
    appendElifRegionPair();

    // Parse Condition region.
    auto [ifParseResult, maybeDeadCodeInfo] = parseCondition(elifLoc);
    if (ifParseResult ||
        parseToken(Token::colon, "expected ':' after 'elif' expression"))
      return failure();
    if (maybeDeadCodeInfo.has_value())
      ifOpsWithDeadCode.push_back(maybeDeadCodeInfo.value());

    // Parse Then region.
    builder.setInsertionPointToStart(&elifOp.getElifRegions().back().front());
    if (failed(parseLocalScopeSuite(curIndent)))
      return failure();
    builder.create<HLCF::YieldOp>(elifLoc);
  }

  builder.setInsertionPointToStart(&elifOp.getElseRegion().emplaceBlock());
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    if (parseToken(Token::colon, "expected ':' after else"))
      return failure();
    if (failed(parseLocalScopeSuite(curIndent)))
      return failure();
  }
  builder.create<HLCF::YieldOp>(ifLoc);

  // Process dead code.  Go backward to avoid needing to erase an already erased
  // IfOp.
  if (!ifOpsWithDeadCode.empty()) {
    for (auto [condition, condExprLoc, index] :
         llvm::reverse(ifOpsWithDeadCode)) {
      shared.emitWarning(condExprLoc)
          << "if statement with constant condition 'if "
          << (condition ? "True" : "False") << "'";
      if (condition) {
        // Condition is true which means all subsequent regions, including else
        // region, are unreachable.
        markRegionUnreachable(&elifOp.getElseRegion(), ifLoc);
        for (auto &region : elifOp.getElifRegions().slice(index + 2))
          markRegionUnreachable(&region, ifLoc);
      } else {
        // Condition is false. Only the first Then region is unreachable.
        markRegionUnreachable(&elifOp.getElifRegions()[index + 1], ifLoc);
      }
    }
  }

  return success();
}

/// if_stmt ::=  "if" assignment_expression ":" suite
///             ("elif" assignment_expression ":" suite)*
///             ["else" ":" suite]
ParseResult StmtParser::parseIfStmt(LexerCursor startCursor, size_t curIndent) {
  // This is enabled with the @parameter decorator.
  bool isParamIf = false;

  // We parse the decorators for the 'if' if they exist.
  if (startCursor != getLexer().getCursor()) {
    startCursor.restore(getLexer());
    for (auto [decorator, cursor] : parseDecorators(curIndent)) {
      // Handle recognized decorators.
      if (auto *dre = dyn_cast<DeclRefNode>(decorator)) {
        if (dre->spelling == "parameter") {
          isParamIf = true;
          continue;
        }
      }

      emitError(decorator->getLoc(), "unsupported decorator on 'if' statement")
          << decorator->getRange();
    }
  }

  Location ifLoc = translateLocation(getToken().getLoc());
  if (parseToken(Token::kw_if, "expected 'if' token after decorators"))
    return failure();
  if (!isParamIf)
    return parseElif(ifLoc, startCursor, curIndent);
  return parseParamIf(ifLoc, startCursor, curIndent);
}

/// import_stmt     ::=  "from" relative_module "import" identifier
///                        ["as" identifier] ("," identifier ["as" identifier])*
///                      | "from" relative_module "import" "(" identifier
///                        ["as" identifier] ("," identifier ["as" identifier])*
///                        [","] ")"
///                      | "from" relative_module "import" "*"
/// module          ::=  (identifier ".")* identifier
/// relative_module ::=  "."* module | "."+
ParseResult StmtParser::parseFromImportStmt() {
  consumeToken(Token::kw_from);

  SMLoc importLoc = getToken().getLoc();
  StringAttr moduleAttr;
  if (parseImportModuleName(moduleAttr) ||
      parseToken(Token::kw_import, "expected 'import' after module name"))
    return failure();

  // Check for a wildcard import.
  if (consumeIf(Token::star)) {
    builder.create<LIT::UnresolvedWildcardImportOp>(
        translateLocation(importLoc), moduleAttr);
    getParentDecl().addUnresolvedWildCardImport(
        moduleAttr, /*isFullImport=*/false, importLoc);
    return success();
  }

  // A functor used to signal to any parser listener that we're importing a decl
  // from the module. If we do emit any notifications, keep track of the
  // currently resolved parent module/package so that the listener can have
  // context for the import.
  ASTDecl *currentResolvedModule = nullptr;
  auto notifyListenerOfImport = [&]() {
    if (!shared.parserListener)
      return;
    SMLoc loc = getToken().getLoc();
    shared.notifyListenerOnMemberLookup(loc, [&]() -> ASTDecl & {
      // Resolve the module if we haven't yet.
      if (!currentResolvedModule) {
        ASTDecl *curModuleDecl = curDeclScope;
        while (curModuleDecl && !isa<FileModuleOp>(curModuleDecl))
          curModuleDecl = curModuleDecl->getParentDecl();

        currentResolvedModule = &shared.importModule(
            moduleAttr,
            curModuleDecl
                ? curModuleDecl->getIfOperation()->getParentOfType<PackageOp>()
                : PackageOp(),
            importLoc);
      }
      return *currentResolvedModule;
    });
  };

  // Parse the set of constructs to import.
  bool isTupleImport = consumeIf(Token::l_paren);
  do {
    // Parse the next construct to import.
    SMLoc importSourceNameLoc = getToken().getLoc();
    StringRef importSourceName = getTokenSpelling();
    bool missingIdentifier =
        failed(parseIdentifier("expected construct name to import"));
    notifyListenerOfImport();

    // If there was no identifier, then we're done.
    if (missingIdentifier)
      return failure();
    StringRef importDestName = importSourceName;
    SMLoc importDestLoc = importSourceNameLoc;
    if (consumeIf(Token::kw_as)) {
      importDestName = getTokenSpelling();
      importDestLoc = getToken().getLoc();
      if (parseIdentifier("expected name to import '" + importSourceName +
                          "' as"))
        return failure();
    }

    // Create an unresolved decl for this import.
    StringAttr importDestNameAttr = builder.getStringAttr(importDestName);
    auto importDecl = builder.create<LIT::UnresolvedImportOp>(
        translateLocation(importLoc), moduleAttr, importDestNameAttr,
        builder.getStringAttr(importSourceName),
        translateLocation(importDestLoc),
        translateLocation(importSourceNameLoc));
    getDeclResolver().addDecl(importDecl, importLoc, importDestNameAttr,
                              curDeclScope, getLexer().getCursor(),
                              getLexer().getCursor(), /*indentation=*/-1);

    // Check for more elements to import.
    if (!consumeIf(Token::comma))
      break;
    // For tuple imports, there may optionally be a trailing comma at the end of
    // the list.
    if (isTupleImport && getToken().is(Token::r_paren))
      break;
  } while (true);

  // Check for the end of the tuple import.
  if (isTupleImport &&
      parseToken(Token::r_paren, "expected ')' after import list"))
    return failure();
  return success();
}

/// import_stmt ::=  "import" module ["as" identifier]
///                  ("," module ["as" identifier])*
/// module      ::=  (identifier ".")* identifier
ParseResult StmtParser::parseImportStmt() {
  consumeToken(Token::kw_import);

  // Parse the next module to import.
  do {
    SMLoc importLoc = getToken().getLoc();
    StringAttr moduleAttr;
    StringRef boundModuleName;
    if (parseImportModuleName(moduleAttr, &boundModuleName))
      return failure();

    // Check for a name binding.
    LocationAttr boundModuleLocAttr;
    if (consumeIf(Token::kw_as)) {
      boundModuleName = getTokenSpelling();
      boundModuleLocAttr = translateLocation(getToken().getLoc());
      if (parseIdentifier("expected name to bind import"))
        return failure();
    }

    // Create an unresolved decl for the import.
    StringAttr importDestNameAttr = builder.getStringAttr(boundModuleName);
    auto importDecl = builder.create<LIT::UnresolvedImportOp>(
        translateLocation(importLoc), moduleAttr, importDestNameAttr,
        /*declName=*/StringAttr(), boundModuleLocAttr,
        /*declNameLoc=*/LocationAttr());
    getDeclResolver().addDecl(importDecl, importLoc, importDestNameAttr,
                              curDeclScope, getLexer().getCursor(),
                              getLexer().getCursor(), /*indentation=*/-1);
  } while (consumeIf(Token::comma));
  return success();
}

/// Parse a module name for use in an import statement.
/// module          ::=  (identifier ".")* identifier
/// relative_module ::=  "."* module | "."+
///
/// If `leafModuleName` is provided, the name is required to be have a `module`
/// component.
ParseResult StmtParser::parseImportModuleName(StringAttr &parsedName,
                                              StringRef *leafModuleName) {
  // The individual name components making up a module.
  SmallVector<StringRef> moduleNames;

  // A functor used to signal to any parser listener that we're importing a
  // module.
  auto notifyListenerOfImport = [&]() {
    if (!shared.parserListener)
      return;
    SMLoc loc = getToken().getLoc();

    // If there isn't a module name, this is a top-level import.
    if (moduleNames.empty())
      return shared.notifyListenerOnImport(loc);

    // Otherwise, this is importing from within a package.
    shared.notifyListenerOnImport(loc, [&]() -> ASTDecl & {
      std::string parentModuleName = llvm::join(moduleNames, ".");
      auto curOp = curDeclScope->getIfOperation()->getParentOfType<PackageOp>();

      // Handle single parent lookups.
      if (parentModuleName.empty())
        parentModuleName = ".";
      return shared.importModule(parentModuleName, curOp, loc);
    });
  };

  // Parse the relative '.' indicators that resolve to a parent package. These
  // push "" to the set to indicate relative resolution.
  while (true) {
    if (consumeIf(Token::dot))
      moduleNames.push_back("");
    else if (consumeIf(Token::dot_dot_dot))
      llvm::append_range(moduleNames, ArrayRef<StringRef>{"", "", ""});
    else
      break;
  }

  // If we have a non-relative module name, or we require one, try to parse it.
  if (leafModuleName || moduleNames.empty() || getToken().isIdentifier()) {
    // Parse the first module name.
    StringRef rootModuleName = getTokenSpelling();
    bool missingIdentifier = failed(parseIdentifier("expected module name"));
    notifyListenerOfImport();

    // If there was no identifier, then we're done.
    if (missingIdentifier)
      return failure();
    moduleNames.push_back(rootModuleName);

    // Parse nested module names.
    while (consumeIf(Token::dot)) {
      notifyListenerOfImport();

      moduleNames.push_back(getTokenSpelling());
      if (parseIdentifier("expected module name"))
        return failure();
    }

    if (leafModuleName)
      *leafModuleName = moduleNames.back();
  } else {
    notifyListenerOfImport();
    moduleNames.push_back("");
  }

  parsedName = builder.getStringAttr(llvm::join(moduleNames, "."));
  return success();
}

//===----------------------------------------------------------------------===//
// Definition statements
//===----------------------------------------------------------------------===//

ParseResult StmtParser::parseDefFnStmt(LexerCursor startCursor,
                                       size_t curIndent) {
  consumeIf(Token::kw_async);
  // isDef is true when introduced by the 'def' keywords instead of 'fn'.
  bool isDef = getToken().is(Token::kw_def);
  consumeToken(); // Consume either 'def' or 'fn'.

  SMLoc loc;
  StringAttr baseName;
  if (parseIdentifier(baseName, "expected function name", &loc))
    return failure();

  auto funcOp = builder.create<LIT::FuncOp>(translateLocation(loc));

  // If marked as 'def', remember this on the function decl.
  if (isDef)
    funcOp.setIsDef(true);

  // Skip the body of this definition: go to a token at the start of the next
  // line at the same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);
  ASTDecl &funcDecl =
      getDeclResolver().addDecl(funcOp, loc, baseName, curDeclScope,
                                startCursor, getLexer().getCursor(), curIndent);
  // If this is a nested function, parse its body right now so captures can be
  // resolved correctly.
  if (curDeclScope->getNearestDeclOfType<LIT::FuncOp>())
    (void)getDeclResolver().resolveFully(funcDecl, loc);
  return success();
}

/// var_decl_stmt ::= "var" identifier ":" expression ["=" expression]
///                 | "var" identifier "=" expression
ParseResult StmtParser::parseVarStmt(LexerCursor startCursor,
                                     size_t stmtIndent) {
  // Global var decls are allowed to have decorators, but nothing else.
  bool hasDecorators = startCursor != getLexer().getCursor();
  auto rejectDecorator = [&, declTok = getToken()]() {
    if (!hasDecorators)
      return;
    emitError(declTok.getLoc()) << "'" << declTok.getSpelling()
                                << "' statement does not allow decorators";
  };

  auto smLoc = consumeToken().getLoc();
  auto loc = translateLocation(smLoc);
  SMLoc identifierLoc;
  StringAttr name;
  if (parseIdentifier(name, "expected name for 'var' declaration",
                      &identifierLoc))
    return failure();

  auto unresolvedType = getUnresolvedType();
  bool delayAddingName = false;
  // If we're in a struct, then this is a field declaration.
  Operation *declOp;
  if (isa<StructDeclOp>(getParentDecl())) {
    rejectDecorator();
    declOp = builder.create<StructFieldOp>(loc, name, unresolvedType);

    // Skip the body of this definition: go to a token the starts a line at the
    // same indent level (or less) as the current definition.
    skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true);
  } else if (isa<TraitDeclOp>(getParentDecl())) {
    rejectDecorator();
    emitError(loc, "TODO: fields in traits are not supported yet");
    skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true);
    return success();
  } else if (isa<LIT::FuncOp>(getParentDecl())) {
    rejectDecorator();
    // This is a local let/var declaration.

    // Emit the vardecl at the current insertion point.  Unlike implicitly
    // declared variables, let/var declarations are always correctly scoped.
    declOp =
        getEmitter().emitVarDecl(name, unresolvedType, loc, VarDeclKind::Var);
    delayAddingName = true;
  } else {
    // Otherwise this is a global let/var declaration.
    declOp = builder.create<GlobalVarDeclOp>(loc, name, unresolvedType);
    skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true);
  }

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  // If the declaration is in a function body, we delay adding it until after
  // resolving the RHS, so that the RHS can reference any identifiers that the
  // decl is shadowing.
  ASTDecl &decl = getDeclResolver().createUnlistedDecl(
      declOp, smLoc, curDeclScope, startCursor, getLexer().getCursor(),
      stmtIndent);
  if (!delayAddingName)
    getDeclResolver().attachDeclToParentNameTable(&decl, name);
  auto temporaryNameReplace = llvm::make_scope_exit([&]() {
    if (delayAddingName)
      getDeclResolver().attachDeclToParentNameTable(&decl, name);
  });

  auto varOp = dyn_cast<VarDeclOp>(decl);
  if (!varOp) {
    // Parse docstrings for struct fields here.
    parseDocString(decl);
    return success();
  }

  // Local variable declarations inside functions are lexically resolved, so
  // fully resolve the decl now. If an error occurs, skip the declaration and
  // keep parsing to emit as many diagnostics as possible.
  auto declParseError = [&] {
    decl.setErroneous();
    skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true);
    return success();
  };

  // Parse the type if present.
  ASTType parsedType;
  ExprEmitter emitter = getEmitter();
  if (consumeIf(Token::colon)) {
    ExprNode *typeExpr = nullptr;
    if (parseExpression(typeExpr, stmtIndent))
      return declParseError();
    parsedType = emitter.emitExprType(typeExpr);
    if (!parsedType)
      return declParseError();
  }

  // Parse the initializer if present.
  ExprNode *initExpr = nullptr;
  if (consumeIf(Token::equal)) {
    if (parseVarInitExpression(initExpr, stmtIndent))
      return declParseError();
  }

  // Now that parsing succeeded, we do IR emission and semantic processing.

  // Handle the initializer if present.
  if (initExpr) {
    // If we have a type, then emit directly into the LValue.  Otherwise emit
    // into the varOp to infer its type.
    ValueDest dest(EC_VarInit);
    ExprContext exprContext = EC_VarInit;
    if (parsedType) {
      varOp.getResult().setType(
          RefType::get(parsedType, varOp.getType().getLifetime()));
      dest = ValueDest(MLValue(varOp), exprContext);
    } else {
      // If we don't, we emit into the varOp itself, because this will infer the
      // type of the varOp from the initializer expression.
      dest = ValueDest(varOp, exprContext);
    }

    if (!emitter.emitExpr(initExpr, dest))
      return declParseError();

    assert(!isa<UnresolvedType>(varOp.getType().getElementType()) &&
           "RValue emission should have inferred var type");

  } else if (parsedType) {
    varOp.getResult().setType(
        RefType::get(parsedType, varOp.getType().getLifetime()));
  } else {
    // If there was neither a type or initializer, reject the var.
    emitError(varOp.getLoc(),
              "declaration must have either a type or an initializer");
    return declParseError();
  }

  // Now mark the decl as fully resolved.
  decl.resolvedness = DeclResolvedness::fully;

  shared.notifyListenerOnVariableDecl(decl, identifierLoc);
  return success();
}

ParseResult StmtParser::parseAliasDeclStmt(LexerCursor startCursor,
                                           size_t stmtIndent) {
  SMLoc smLoc = consumeToken(Token::kw_alias).getLoc();
  Location loc = translateLocation(smLoc);
  StringAttr name;
  if (parseIdentifier(name, "expected name for 'alias' declaration"))
    return failure();

  // Before parsing the rest of the alias, the type is unresolved and value is
  // UnresolvedAliasValueAttr.
  auto type = getUnresolvedType();
  auto value = UnresolvedAliasValueAttr::get(type);

  // TODO(fixme): currently, we cannot rely on looking up name collisions of
  // aliases because of things like this:
  // fn foo():
  //     fn bar():
  //         alias z = __mlir_attr.`0: index`
  //     alias z = __mlir_attr.`1: index`
  // So we treat them as implicitly declared to force a mangling. We could
  // probably fix this when parameters stop being non-lexical.
  StringAttr mangledName = parentDecl.mangleParamName(name.strref());
  auto decl = ParamDeclAttr::get(mangledName, type);
  auto declOp = builder.create<AliasDeclOp>(loc, decl, value);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true);

  // Skip the trailing docstring if it has one. We treat these as part of the
  // alias decl.
  (void)consumeIf(Token::string);

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(declOp, smLoc, name, curDeclScope, startCursor,
                            getLexer().getCursor(), stmtIndent);
  return success();
}

ParseResult StmtParser::parseStructStmt(LexerCursor startCursor,
                                        size_t curIndent) {
  // We don't support non-top level structs (yet?).
  bool nestFailure = false;
  if (isa<StructDeclOp>(getParentDecl())) {
    emitTokenError("nested struct not supported here");
    nestFailure = true;
  } else if (isa<TraitDeclOp>(getParentDecl())) {
    emitTokenError("nested struct in a trait not supported here");
    nestFailure = true;
  } else if (isa<LIT::FuncOp>(getParentDecl())) {
    emitTokenError("struct inside a function not supported here");
    nestFailure = true;
  }
  consumeToken(Token::kw_struct);

  SMLoc smLoc;
  StringAttr nameAttr;
  if (parseIdentifier(nameAttr, "expected struct name", &smLoc))
    return failure();
  auto loc = translateLocation(smLoc);

  auto newStruct = builder.create<StructDeclOp>(loc, nameAttr);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);

  if (nestFailure) {
    getDeclResolver().addErroneousDecl(nameAttr.getValue(), smLoc,
                                       curDeclScope);
  } else {
    // Remember that we parsed this declaration so we can finish type checking
    // it when it gets referenced.
    getDeclResolver().addDecl(newStruct, smLoc, nameAttr, curDeclScope,
                              startCursor, getLexer().getCursor(), curIndent);
  }
  return success();
}

ParseResult StmtParser::parseTraitStmt(LexerCursor startCursor,
                                       size_t curIndent) {
  // We don't support non-top level traits (yet?).
  if (!isa<FileModuleOp>(getParentDecl()))
    emitTokenError("nested trait not supported here");

  consumeToken(Token::kw_trait);

  SMLoc smLoc;
  StringAttr nameAttr;
  if (parseIdentifier(nameAttr, "expected trait name", &smLoc))
    return failure();
  auto loc = translateLocation(smLoc);

  auto newTrait = builder.create<TraitDeclOp>(loc, nameAttr);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(newTrait, smLoc, nameAttr, curDeclScope,
                            startCursor, getLexer().getCursor(), curIndent);
  return success();
}

ParseResult StmtParser::parseClassStmt(LexerCursor startCursor,
                                       size_t curIndent) {
  emitTokenError("classes are not supported yet");
  consumeToken(Token::kw_class).getLoc();

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);
  return success();
}

/// An MLIR region declaration defines a single block region body as a suite
/// with the declaration arguments corresponding to the region arguments. It is
/// used to define regions for MLIR operations.
///
/// region_stmt ::= "__mlir_region" identifier "(" [argument_list] ")" ":" suite
ParseResult StmtParser::parseMLIRRegionStmt(LexerCursor startCursor,
                                            size_t curIndent) {
  SMLoc loc = consumeToken(Token::kw___mlir_region).getLoc();

  // We will be moving the builder into the contained region, so save it here.
  llvm::SaveAndRestore builderSaver(builder);

  // Resolve the signature and the body immediately.
  StringAttr identifier;
  if (parseIdentifier(identifier, "expected a region name") ||
      parseToken(Token::l_paren, "expected '(' for parameter list"))
    return failure();

  // Create the decl corresponding to the region declaration.
  auto op = builder.create<UnboundRegionOp>(translateLocation(loc));
  ASTDecl &decl =
      getDeclResolver().addDecl(op, loc, identifier, curDeclScope, startCursor,
                                getLexer().getCursor(), curIndent);
  decl.resolvedness = DeclResolvedness::fully;

  // Parse the argument list if present.
  struct RegionArgument {
    StringAttr name;
    SMLoc loc;
  };
  SmallVector<RegionArgument> args;
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  if (!consumeIf(Token::r_paren)) {
    // Parse simple argument: MLIR operations don't have input conventions.
    auto parseArg = [&]() -> ParseResult {
      RegionArgument &arg = args.emplace_back();
      ExprNode *typeExpr;
      if (getLocation(arg.loc) ||
          parseIdentifier(arg.name, "expected an identifier") ||
          parseToken(Token::colon, "expected ':' after region argument") ||
          parseExpression(typeExpr))
        return failure();
      ASTType type = getEmitter().emitExprType(typeExpr);
      if (!type)
        return failure();
      argTypes.push_back(type);
      argLocs.push_back(translateLocation(arg.loc));
      return success();
    };
    if (parseCommaSeparatedList(parseArg, Token::r_paren))
      return failure();
    consumeToken(Token::r_paren);
  }

  builder.createBlock(&op.getRegion());
  for (auto [regionArg, parsedArg] :
       llvm::zip(op.getRegion().addArguments(argTypes, argLocs), args)) {
    // Add the declaration for the argument within the region declaration.
    getDeclResolver().addFullyResolvedDecl(SBValue(regionArg), parsedArg.name,
                                           parsedArg.loc, &decl);
  }

  if (parseToken(Token::colon, "expected ':' after region argument list"))
    return failure();
  StmtParser parser(shared, lexer, decl);
  return parser.parseLocalScopeSuite(curIndent);
}

//===----------------------------------------------------------------------===//
// Entry point to this file
//===----------------------------------------------------------------------===//

/// Parse a 'suite' production into the declaration specified by `ASTDecl`.
/// This is the main entrypoint to this file.
ParseResult ParserBase::parseSuite(ASTDecl &containingDecl) {
  StmtParser parser(shared, lexer, containingDecl);

  // Parse the docstring if present.
  parser.parseDocString(containingDecl);

  // Parse the remaining body of the declaration.
  return parser.parseSuite(containingDecl.getIndentation());
}
