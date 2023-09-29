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
// StmtParser
//===----------------------------------------------------------------------===//

/// This class provides the implementation details of the concrete Lightning
/// grammar.
namespace {
struct StmtParser : public ParserBase {
  StmtParser(SharedState &shared, Lexer &lexer, ASTDecl &containingDecl)
      : ParserBase(shared, lexer), parentDecl(containingDecl),
        curDeclScope(&containingDecl),
        builder(containingDecl.getDeclEndBuilder()) {

    // If we are parsing into a 'def', then we need a position to synthesize
    // variable definitions at the top of the function.
    if (auto funcOp = dyn_cast<LIT::FuncOp>(getParentDecl())) {
      if (funcOp.getIsDef()) {
        // The operation builder inserts before its insertion point, but for a
        // stable insertion point, keep the previous iterator position.
        varDeclCursor = OpBuilder(builder.getInsertionBlock(),
                                  std::prev(builder.getInsertionPoint()));
      }
    }
  }

  ASTDecl &getParentDecl() { return parentDecl; }
  OpBuilder &getBuilder() { return builder; }

  /// Push a debug info lexical block to represent a local variable scope.
  void pushLocalScope(DebugInfo::DIBuilder::ScopeGuard &scopeGuard);

  /// A local decl put in a scope when entering a new scope.
  /// The astDeclCallback field is called after constructing an ASTDecl.
  struct ScopeDecl {
    DeclIRValue value;
    SMLoc loc;
    StringRef name;
    void (*astDeclCallback)(ASTDecl &decl);
    ScopeDecl(DeclIRValue value, SMLoc loc, StringRef name)
        : value(value), loc(loc), name(name), astDeclCallback(nullptr) {}
    ScopeDecl(DeclIRValue value, SMLoc loc, StringRef name,
              void (*astDeclCallback)(ASTDecl &decl))
        : value(value), loc(loc), name(name), astDeclCallback(astDeclCallback) {
    }
  };

  // Expression emission.

  ExprEmitter getEmitter(bool allowImplicitVarDecl = false) {
    return ExprEmitter(shared, *curDeclScope, builder,
                       allowImplicitVarDecl ? varDeclCursor : std::nullopt);
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
  ParseResult parseWhileStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseForStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseTryStmt(size_t curIndent);
  ParseResult parseWithStmt(size_t curIndent);

  // Simple statements.
  ParseResult parseReturnStmt(size_t returnIndent);
  ParseResult parseParamReturnStmt(size_t returnIndent);
  ParseResult parseRaiseStmt(size_t raiseIndent);
  ParseResult parseBreakOrContinueStmt(Token::Kind kind, StringRef name,
                                       StringRef opName);

  // Declarations.
  ParseResult parseFromImportStmt();
  ParseResult parseImportStmt();
  ParseResult parseImportModuleName(bool allowRelativeModules,
                                    StringAttr &parsedName,
                                    StringRef *leafModuleName = nullptr);
  ParseResult parseDefFnStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseStructStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseTraitStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseClassStmt(LexerCursor startCursor, size_t curIndent);
  ParseResult parseLetVarStmt(LexerCursor startCursor, size_t stmtIndent);
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

  scopeGuard = shared.diBuilder->pushLexicalBlock(
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
    } while (consumeIf(Token::semi) &&
             !getToken().getIndentation().has_value());
    return success();
  };

  // If this suite is on the same line as the enclosing entity, just parse a
  // single stmt_list.
  auto indent = getToken().getIndentation();
  if (!indent.has_value())
    return parseStmtListOrCompound(
        /*stmtListOnly=*/true,
        /*stmtIndent=*/std::numeric_limits<size_t>::max());

  // If there is a newline, then parse a list of statements which can be either
  // a statement list or a compound_stmt.  Parse all the statements that are
  // more nested than this suite, and reject it if there are none.
  if (ssize_t(*indent) <= curIndent) {
    emitError(getTokenLocOrEndOfPreviousLineIfOnNewLine())
        << "expected body statements; use 'pass' if none is required";
    return success();
  }

  while (getToken().isNot(Token::eof)) {
    auto indent = getToken().getIndentation();
    if (!indent.has_value())
      return emitTokenError("statements must start at the beginning of a line");
    if (ssize_t(*indent) <= curIndent)
      break;

    if (parseStmtListOrCompound(/*stmtListOnly=*/false, *indent))
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
  // TODO: Should have a better way to say that it is safe to implicitly ignore
  // a value of a type (e.g. a type decorator)
  auto isImplicitlyIgnorableType = [&](ASTType type) -> bool {
    // TODO: This is incorrect for throwing functions that return None.
    return type.isNoneType() ||
           type.isEqualCanon(shared.getTypeCheckErrorType());
  };

  if (isImplicitlyIgnorableType(valueType) ||
      // The `x = y` operation returns a borrowed version of its operand but its
      // result can be ignored.
      expr->kind == ExprNode::kAssign)
    return;

  // If this type is a function with no arguments and an ignorable type, we
  // emit a warning with a fix it hint suggesting that it get called.
  if (auto sig = dyn_cast<SignatureType>(valueType.mlirType)) {
    // TODO: This is incorrect for default arguments and varargs.
    assert(sig.getNumResults() == 1);

    // Get the result type without any error handling in the way.
    Type resultType = sig.getValueResults()[0];
    if (sig.isThrows())
      resultType = cast<POP::VariantType>(resultType).getType(1);

    if (sig.getValueInputs().empty() && isImplicitlyIgnorableType(resultType)) {
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
  if (valueType.getDecl(shared)) {
    OverloadSet await(valueType, "__await__", expr, CallSyntax::kMethodCall,
                      shared, /*no error on failure*/ {});
    if (!await.isNull()) {
      auto loc = expr->getRange().getStart();
      shared.emitWarning(expr->getLoc())
          << "awaitable " << valueType << " value was never awaited"
          << expr->getRange() << FixIt::insertBeforeToken(loc, "await ");
      return;
    }
  }

  // Otherwise emit a warning, and suggest assigning to _.
  auto startLoc = expr->getRange().getStart();
  shared.emitWarning(expr->getLoc())
      << valueType << " value is unused" << expr->getRange()
      << FixIt::insertBeforeToken(startLoc, "_ = ");
}

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
  case Token::kw_let:
  case Token::kw_var:
  case Token::kw_alias:
  case Token::kw___mlir_region:
  case Token::kw_return:
  case Token::kw_param_return:
  case Token::kw_raise:
  case Token::kw_continue:
  case Token::kw_break:
    return true;
  default:
    return false;
  }
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
    if (!getToken().getIndentation() && stopOnStatement()) {
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
    if (getToken().getIndentation() &&
        *getToken().getIndentation() < stmtIndent) {
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
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseWhileStmt(startCursor, stmtIndent);
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
  case Token::kw_let:
  case Token::kw_var:
    return parseLetVarStmt(startCursor, stmtIndent);
  case Token::kw_alias:
    return parseAliasDeclStmt(startCursor, stmtIndent);
  case Token::kw___mlir_region:
    rejectDecorator();
    rejectSimpleStmt();
    return parseMLIRRegionStmt(startCursor, stmtIndent);
  case Token::kw_return:
    rejectDecorator(); // Decorators not allowed.
    return parseReturnStmt(stmtIndent);
  case Token::kw_param_return:
    rejectDecorator(); // Decorators not allowed.
    return parseParamReturnStmt(stmtIndent);
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
  if (parent && isa<LIT::FileModuleOp>(parent))
    emitTokenError()
        << "TODO: expressions are not yet supported at the file scope level";
  if (parseSimpleStmtExprs(expr, stmtIndent))
    return failure();

  // Emit the expression and ignore the results.  If it is an assignment
  // statement, it will return None.  Other expressions can return whatever they
  // will naturally return.
  auto emitter = getEmitter(/*allowImplicitVarDecl=*/true);
  // Result is ignored, so we don't care where it goes.
  CValue result = emitter.emitExprCValue(expr, EC_TopLevelStmt);
  if (!result)
    return success();

  // Emit a warning if the result is a value we should warn when unused.
  auto funcOp = dyn_cast<LIT::FuncOp>(parentDecl);
  if (!funcOp || !funcOp.getIsDef())
    diagnoseIgnoredResult(expr, result, shared);
  return success();
}

//===----------------------------------------------------------------------===//
// Simple statements.
//===----------------------------------------------------------------------===//

/// return_stmt ::= "return" [expression_list]
ParseResult StmtParser::parseReturnStmt(size_t returnIndent) {
  auto decl = dyn_cast<LIT::FuncOp>(getParentDecl());
  auto loc = consumeToken(Token::kw_return).getLoc();

  // If there is an expression list present, parse it.
  ExprNode *operandExpr = nullptr;
  if (!isTokenInCurrentStatement(returnIndent))
    operandExpr = getNoneExpr(loc);
  else if (parseExpressionList(operandExpr, returnIndent))
    return failure();

  // Ok, now that we parsed all the tokens for this statement, do semantic
  // analysis.
  if (!decl) {
    emitError(loc, "cannot return from this context");
    return success();
  }

  auto emitter = getEmitter();

  // Materialize the expression values into IR.
  Value resultValue;
  SignatureType declSig = decl.getSignature();
  if (declSig.hasMemoryOnlyResult()) {
    // If the result is memory-only, return into the result slot.
    ValueDest resultDest(MLValue(decl.getArgument(0)), EC_ReturnValue);
    if (!operandExpr->emitIR(resultDest, emitter)) {
      resultDest.resetForError();
      return success();
    }
    resultValue = emitter.emitSRValue(
        {PValue(shared.getNoneAttr()), operandExpr}, EC_ReturnValue);

  } else {
    // Convert the returned value to the returned type of the function.
    resultValue = emitter.emitExprSRValue(operandExpr, EC_ReturnValue,
                                          decl.getUserResultType());
  }

  if (!resultValue)
    return {};

  auto mlirLoc = translateLocation(loc);

  // If this function throws, we silently wrap the result value in the returned
  // variant type.
  ImplicitLocOpBuilder b(mlirLoc, builder);
  if (decl.isThrows())
    resultValue =
        b.create<POP::VariantCreateOp>(decl.getMLIRResultType(), resultValue);

  ExprEmitter::emitNormalReturn(b, resultValue, getParentDecl());
  return success();
}

/// param_return_stmt ::= "param_return" "[" expression_list "]"
ParseResult StmtParser::parseParamReturnStmt(size_t returnIndent) {
  SMLoc loc = consumeToken(Token::kw_param_return).getLoc();
  auto decl = dyn_cast<LIT::FuncOp>(getParentDecl());
  if (!decl) {
    emitError(loc, "invalid context for parameter return");
    return success();
  }

  // Parse the result parameter list.
  SMLoc startLoc, endLoc;
  if (parseToken(Token::l_square, "expected '[' to begin parameter list",
                 &startLoc))
    return success();

  ExprNode *expr = nullptr;
  if (parseExpressionList(expr, returnIndent) ||
      parseToken(Token::r_square, "expected ']' at end of parameter list",
                 &endLoc))
    return failure();

  // We treat each comma separated value as a separate parameter, not the whole
  // thing as a tuple value.
  ArrayRef<ExprNode *> exprs = expr;
  if (auto *tuple = dyn_cast<TupleNode>(expr))
    exprs = tuple->exprs;

  // Check the number of result parameters.
  size_t numResultParams = decl.getResultParams().size();
  if (exprs.size() != numResultParams) {
    emitError(startLoc, "expected ")
        << numResultParams << " result parameter" << plural(numResultParams)
        << SourceRange(startLoc, endLoc);
    return success();
  }

  // Emit the result parameters into PValues.
  SmallVector<TypedAttr> paramValues;
  for (auto [paramExpr, param] : llvm::zip(exprs, decl.getResultParams())) {
    auto result = getEmitter().emitExprPValue(
        paramExpr, EC_ReturnResultParamList, param.getType());
    if (!result)
      return success();
    paramValues.push_back(result);
  }

  builder.create<LIT::ParamReturnOp>(translateLocation(loc), paramValues);
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

  // If we had an error, emit it.
  Value errorVal;
  if (errorExpr) {
    ASTType errorType = shared.getBuiltinErrorType(getParentDecl(), loc.Start);

    // TODO: Support memory-only error values.
    errorVal =
        getEmitter().emitExprSRValue(errorExpr, EC_RaiseValue, errorType);
    if (!errorVal)
      return success();
  } else {
    // Figure it if we're in a try, and if so, which subregion.
    auto [tryOp, inExceptRegion] = findParentTry(builder.getInsertionBlock());

    // Otherwise, we must be in the 'except' part of the try block and are
    // rethrowing the current error.  This isn't correct Python semantics, see
    // the caveat above.
    if (!inExceptRegion) {
      InflightDiag diag = emitError(loc.Start, "no contextual error to reraise")
                          << loc;
      diag.attachNote(loc.Start) << "provide an error to raise or place 'raise'"
                                    "statement inside an except region";
      return success();
    }
    errorVal = tryOp.getExceptRegion().getArgument(0);
  }

  // Emit the logic to raise the error.
  if (failed(getEmitter().emitRaise(errorVal, translateLocation(loc.Start)))) {
    InflightDiag diag =
        emitError(loc.Start, "cannot raise error in this context") << loc;
    diag.attachNote(loc.Start) << "try surrounding 'raise' in a 'try' block";
    if (auto func = getBlockParentOfType<LIT::FuncOp>(
            getEmitter().builder->getInsertionBlock()))
      diag.attachNote(func.getLoc())
          << "or mark surrounding function as 'raises'";
  }

  return success();
}

/// break_stmt ::= "break"
/// continue_stmt ::= "continue"
ParseResult StmtParser::parseBreakOrContinueStmt(Token::Kind kind,
                                                 StringRef name,
                                                 StringRef opName) {
  llvm::SMLoc loc = consumeToken(kind).getLoc();

  // Ensure the break statement is being parsed within a loop context.
  if (!getBlockParentOfType<HLCF::LoopOp>(builder.getInsertionBlock())) {
    emitError(loc, "'" + name + "' not inside a loop");
    return success();
  }

  // Split the block at the insertion point. Any subsequent statements are dead
  // code. Let region DCE handle it.
  OperationState state(translateLocation(loc), opName);
  builder.create(state);
  return success();
}

static ParseResult parseLoopDecorators(ParserBase &parser,
                                       LexerCursor startCursor,
                                       size_t curIndent, Token::Kind kind,
                                       HLCF::UnrollLevel &level) {
  StringRef kindName = parser.getToken().getSpelling();

  if (startCursor != parser.getLexer().getCursor()) {
    startCursor.restore(parser.getLexer());
    for (auto [decorator, cursor] : parser.parseDecorators(curIndent)) {
      // Handle recognized decorators.
      if (auto *dre = dyn_cast<DeclRefNode>(decorator)) {
        if (dre->spelling == "unroll") {
          level = HLCF::UnrollLevel::full();
          continue;
        }
      } else if (auto *callNode = dyn_cast<CallNode>(decorator)) {
        if (auto dre = dyn_cast<DeclRefNode>(callNode->callee)) {
          int32_t factor;
          if (dre->spelling == "unroll" && callNode->operands.size() == 1 &&
              callNode->operands[0].isPositionalIntLiteral(factor)) {
            level = HLCF::UnrollLevel(factor);
            continue;
          }
        }
      }

      // TODO: Parse unroll with a integer number or a parameter expression
      return parser.emitError(decorator->getLoc(),
                              "unsupported decorator on '" + kindName +
                                  "' statement")
             << decorator->getRange();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Compound statements.
//===----------------------------------------------------------------------===//

/// while_stmt ::=  "while" assignment_expression ":" suite
///                 ["else" ":" suite]
ParseResult StmtParser::parseWhileStmt(LexerCursor startCursor,
                                       size_t curIndent) {
  // We parse the decorators for the 'while' if they exist.
  HLCF::UnrollLevel unrollLevel = HLCF::UnrollLevel::none();
  if (parseLoopDecorators(*this, startCursor, curIndent, Token::kw_while,
                          unrollLevel))
    return success();

  Location whileLoc = translateLocation(consumeToken(Token::kw_while).getLoc());

  ExprNode *condExp = nullptr;
  if (parseAssignExpression(condExp, curIndent) ||
      parseToken(Token::colon, "expected ':' after expression"))
    return failure();

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  auto loopOp = builder.create<HLCF::LoopOp>(whileLoc, unrollLevel);
  Block *body = builder.createBlock(&loopOp.getBody());
  builder = OpBuilder::atBlockEnd(body);

  RValue condRVal = getEmitter().emitExprI1(condExp, EC_BoolCondition);
  Value condVal =
      getEmitter().emitSRValue({AnyValue(condRVal), condExp}, EC_BoolCondition);
  if (!condVal)
    return success(); // IRGen error already emitted; parse succeeded!

  // Generate the while condition check.
  auto condOp = builder.create<HLCF::IfOp>(whileLoc, condVal);
  builder.createBlock(&condOp.getThenRegion());
  builder.create<HLCF::YieldOp>(whileLoc);
  Block *exit = builder.createBlock(&condOp.getElseRegion());
  builder.create<HLCF::BreakOp>(whileLoc);

  // Create the body.
  builder.setInsertionPointAfter(condOp);
  if (failed(parseLocalScopeSuite(curIndent)))
    return failure();
  builder.create<HLCF::ContinueOp>(whileLoc);

  // The 'else' block is executed only when the condition check fails.
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    builder.setInsertionPointToStart(exit);
    if (parseToken(Token::colon, "expected ':' after else") ||
        parseLocalScopeSuite(curIndent))
      return failure();
  }
  return success();
}

/// for_stmt ::=  "for" target_list "in" starred_list ":" suite
///              ["else" ":" suite]
ParseResult StmtParser::parseForStmt(LexerCursor startCursor,
                                     size_t curIndent) {
  // We parse the decorators for the 'for' if they exist.
  HLCF::UnrollLevel unrollLevel = HLCF::UnrollLevel::none();
  if (parseLoopDecorators(*this, startCursor, curIndent, Token::kw_for,
                          unrollLevel))
    return success();

  Location forLoc = translateLocation(consumeToken(Token::kw_for).getLoc());

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

  auto funcOp = dyn_cast<LIT::FuncOp>(parentDecl);
  StringAttr targetLifenameName = curDeclScope->getAnonymousLifetimeFor(target);
  auto varDeclOp = builder.create<VarLetDeclOp>(
      forLoc, UnresolvedType::get(getContext()), target, targetLifenameName,
      /*isVar=*/funcOp && funcOp.getIsDef(), /*isSynth=*/true);

  // If there is a failure before we parse the for loop body, we still
  // want to call the parser on it so that it builds an ASTDecl node
  // and adds the for loop VarLetDecl to the lookup path.  Otherwise,
  // we will get spurious “use of unknown declaration” errors on it
  // besides whatever error is raised while processing the loop
  // header.
  auto avoidDroppingDeclOnFail = llvm::make_scope_exit([&]() {
    std::ignore = parseLocalScopeSuite(
        curIndent, ScopeDecl{&*varDeclOp, targetLoc, target,
                             [](ASTDecl &d) { d.hasReferenceError = true; }});
  });

  // retrieve the iterator object from the sequence expression
  auto tmpEmitter = getEmitter();
  ASTExprAnd<AnyValue> loadedSeq = {
      seqExpr->emitIR(ValueDest::none(), tmpEmitter), seqExpr};
  if (!loadedSeq.ir)
    return {};

  // Emit a call to __iter__ into a var with an inferred type.
  StringAttr rangeName = StringAttr::get(builder.getContext(), "$RANGE");
  StringAttr rangeLifenameName =
      curDeclScope->getAnonymousLifetimeFor(rangeName);
  auto rangeRef = builder.create<VarLetDeclOp>(
      forLoc, UnresolvedType::get(getContext()), rangeName, rangeLifenameName,
      /*isVar*/ true, /*isSynth=*/true);
  ValueDest rangeDest(rangeRef, EC_ForIterator);
  if (!getEmitter().emitNamedMethodCall("__iter__", {loadedSeq}, rangeDest,
                                        CallSyntax::kImplicitConvert,
                                        seqExpr)) {
    rangeDest.resetForError();
    varDeclOp.getResult().setType(
        RefType::get(/*isMut*/ true, shared.getTypeCheckErrorType(),
                     varDeclOp.getType().getLifetime()));
    return {};
  }

  HLCF::LoopOp loopOp = builder.create<HLCF::LoopOp>(forLoc, unrollLevel);
  Block *body = builder.createBlock(&loopOp.getBody());
  builder = OpBuilder::atBlockEnd(body);

  // For Loop condition: if the length of the range is greater than zero,
  // continue. Otherwise break
  AnyValue currentLength = getEmitter().emitNamedMethodCall(
      "__len__", CallOperands({{XLValue(rangeRef), seqExpr}}),
      ValueDest::none(), CallSyntax::kImplicitConvert, seqExpr);
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

  // Generate the for condition check.
  auto condOp = builder.create<HLCF::IfOp>(forLoc, shouldContinue);
  builder.createBlock(&condOp.getThenRegion());
  builder.create<HLCF::YieldOp>(forLoc);
  Block *exit = builder.createBlock(&condOp.getElseRegion());
  builder.create<HLCF::BreakOp>(forLoc);

  // Create the body. Add Target element to the continue block by calling next
  // method. Emit the result into an implicitly declared variable at the current
  // scope.
  builder.setInsertionPointAfter(condOp);
  ValueDest ivarDest(varDeclOp, EC_ForIterator);
  if (!getEmitter().emitNamedMethodCall(
          "__next__", CallOperands({{XLValue(rangeRef), seqExpr}}), ivarDest,
          CallSyntax::kImplicitConvert, seqExpr)) {
    ivarDest.resetForError();
    return {};
  }

  avoidDroppingDeclOnFail.release();
  if (failed(parseLocalScopeSuite(curIndent,
                                  ScopeDecl{&*varDeclOp, targetLoc, target})))
    return failure();
  builder.create<HLCF::ContinueOp>(forLoc);

  // The 'else' block is executed only when the condition check fails.
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    builder.setInsertionPointToStart(exit);
    if (parseToken(Token::colon, "expected ':' after else") ||
        parseLocalScopeSuite(curIndent))
      return failure();
  }
  return success();
}

/// try_stmt ::= "try" ":" suite "except" [identifier] ":" suite
///              ["else" suite]
ParseResult StmtParser::parseTryStmt(size_t curIndent) {
  Location loc = translateLocation(consumeToken(Token::kw_try).getLoc());

  // Restore the builder to its current insertion point after parsing.
  llvm::SaveAndRestore builderSaver(builder);
  auto tryOp = builder.create<TryOp>(loc);
  if (parseToken(Token::colon, "expected ':' after 'try'"))
    return failure();

  // Parse the try suite.
  builder.createBlock(&tryOp.getTryRegion());
  if (parseLocalScopeSuite(curIndent))
    return failure();
  builder.create<TryYieldOp>(translateLocation(getToken().getLoc()));

  SMLoc errValLoc = getToken().getLoc();
  ASTType errorType = shared.getBuiltinErrorType(getParentDecl(), errValLoc);
  if (!errorType.isRegisterPassable(errValLoc, shared)) {
    emitError(errValLoc) << errorType << " is not a @register_passable type";
    return failure();
  }

  bool hasFinally = false;
  if (getToken().is(Token::kw_except)) {
    errValLoc = consumeToken().getLoc();

    // Parse an optional identifier to bind the error.
    StringAttr errName;
    if (getToken().isIdentifier())
      (void)parseIdentifier(errName, "<this can't fail>", &errValLoc);

    if (parseToken(Token::colon, "expected ':' after 'except'"))
      return failure();

    Block *exceptBlock = builder.createBlock(&tryOp.getExceptRegion());
    Value errVal =
        exceptBlock->addArgument(errorType, translateLocation(errValLoc));

    // If an identifier was declared for the error value, add a declaration that
    // references it.
    SmallVector<ScopeDecl> decls;
    if (errName) {
      auto func = dyn_cast<LIT::FuncOp>(parentDecl);
      if (func && func.getIsDef()) {
        // If we are parsing inside a 'def', create a mutable LValue to allow
        // reassignment.
        StringAttr errLifetimeName =
            curDeclScope->getAnonymousLifetimeFor(errName);
        auto varDecl = builder.create<VarLetDeclOp>(
            errVal.getLoc(), errVal.getType(), errName, errLifetimeName,
            /*isVar=*/true, /*isSynth=*/true);
        decls.push_back(ScopeDecl{DeclIRValue(varDecl), errValLoc, errName});
        builder.create<RefStoreOp>(errVal.getLoc(), errVal, varDecl);
      } else {
        // If we are parsing inside an 'fn', the error declaration is an BValue,
        // because any reference to it needs to copy/move out.
        decls.push_back(ScopeDecl{SBValue(errVal), errValLoc, errName});
      }
    }

    // Parse the except suite.
    if (parseLocalScopeSuite(curIndent, decls))
      return failure();
    builder.create<TryYieldOp>(translateLocation(getToken().getLoc()));

    // Parse the else suite if present. Otherwise, leave it as empty.
    builder.createBlock(&tryOp.getElseRegion());
    if (consumeIf(Token::kw_else)) {
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
    // Stub out the 'except' and 'else' regions.
    Block *exceptBlock = builder.createBlock(&tryOp.getExceptRegion());
    Value errVal =
        exceptBlock->addArgument(errorType, translateLocation(errValLoc));
    // Propagate the error if it is possible in this context.
    (void)getEmitter().emitRaise(SRValue(errVal), loc);
    builder.create<TryYieldOp>(loc);

    builder.createBlock(&tryOp.getElseRegion());
    builder.create<TryYieldOp>(loc);
  }
  builder.createBlock(&tryOp.getFinallyRegion());
  if (hasFinally) {
    if (parseToken(Token::colon, "expected ':' after 'finally'") ||
        parseLocalScopeSuite(curIndent))
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
  // TODO: Generalize to multiple of them.
  ExprNode *contextExp = nullptr;
  if (parseExpression(contextExp))
    return failure();

  // If we are in a def, we need to use function scoping.  If we are in a fn,
  // we need to use lexical scope.  When we support `with` at the top level, we
  // should decide whether it is lexical or global scope.  This largely depends
  // on our view of what `python superset` or `python++` means.
  bool useLexicalScope = true;
  if (auto funcOp = dyn_cast<LIT::FuncOp>(getParentDecl())) {
    if (funcOp.getIsDef())
      useLexicalScope = false;
  }

  // FIXME: This needs to parse this as a target expression and then handle it
  // like a destructuring pattern.
  VarLetDeclOp targetDecl;
  bool addDecl = false;
  SMLoc targetLoc;
  SMLoc asLoc;
  ValueDest enterDest(EC_WithContextMgr);
  if (consumeIf(Token::kw_as, &asLoc)) {
    StringAttr name = StringAttr::get(getContext(), getToken().getSpelling());
    if (parseIdentifier("expected identifier for target in 'with'", &targetLoc))
      return failure();
    ArrayRef<ASTDecl *> decls = curDeclScope->lookupInCurrentScope(name);
    if (!useLexicalScope && !decls.empty()) {
      SMLoc declLoc = decls[0]->getLoc();
      AnyValue emitted = getEmitter().emitDeclReference(name.getValue(), decls);
      if (auto slval = emitted.getIfMLValue()) {
        enterDest = ValueDest(slval, EC_WithContextMgr);
      } else if (auto ref = emitted.getIfXLValue()) {
        enterDest = ValueDest(ref, EC_WithContextMgr);
      } else {
        (emitError(targetLoc)
         << name
         << " is not a valid mutable variable for `with ... as` to target")
                .attachNote(declLoc)
            << name << " declared here";
        return failure();
      }
    } else {
      StringAttr lifetimeName = curDeclScope->getAnonymousLifetimeFor(name);
      targetDecl = builder.create<VarLetDeclOp>(
          shared.translateLocation(targetLoc),
          UnresolvedType::get(getContext()), name, lifetimeName,
          /*isVar=*/!useLexicalScope, /*isSynth=*/true);
      enterDest = ValueDest(targetDecl, EC_WithContextMgr);
      addDecl = true;
    }
  }

  if (parseToken(Token::colon, "expected ':' after 'with' expression")) {
    enterDest.resetForError();
    return failure();
  }

  CValue contextRV = getEmitter().emitExprCValue(contextExp, EC_WithContextMgr);

  // Emit the call to __enter__ and (if 'as TARGET' was specified), bind to
  // result to a named TARGET vardecl, inferring its type.
  CValue enterResult = getEmitter().emitNamedMethodCall(
      "__enter__", CallOperands({{contextRV, contextExp}}), enterDest,
      CallSyntax::kMethodCall, contextExp);
  if (!enterResult)
    enterDest.resetForError();

  DebugInfo::DIBuilder::ScopeGuard scopeGuard;
  llvm::SaveAndRestore<ASTDecl *> keepDecl(curDeclScope);
  if (useLexicalScope)
    pushChildScope(scopeGuard, keepDecl);

  // Inject the target into our scope if asked for.
  if (addDecl) {
    auto &targetDeclResolved = getDeclResolver().addFullyResolvedDecl(
        targetDecl.getOperation(), targetDecl.getNameAttr(), targetLoc,
        curDeclScope);
    if (!enterResult)
      targetDeclResolved.hasReferenceError = true;
  }

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
  // This will already have been diagnosed when the __enter__ was emitted.
  ASTDecl *contextDecl = contextRV.getRValueType().getDecl(shared);
  bool hasExitMethod =
      contextDecl && !contextDecl->lookupInCurrentScope("__exit__").empty();

  // If the context manager has no __exit__ method, then we emit this pattern
  // with no try operation at all:
  //   contextMgr = EXPRESSION
  //   TARGET = contextMgr.__enter__()
  //   SUITE
  //   lit.ownership.use(TARGET)
  // This is useful for compatibility, and does scope "TARGET" (when
  // useLexicalScope is true) and the context manager expression.
  if (!hasExitMethod) {
    // Parse the body suite.
    if (parseLocalScopeSuite(curIndent))
      return failure();

    // The target value may not have a __exit__ method, but we need it to be
    // live all the way across the suite, so add an extra use so it isn't
    // destroyed early.
    if (enterResult) {
      if (auto targetBV = getEmitter().emitBValue(
              {enterResult, contextExp}, ExprContext::EC_WithContextMgr)) {
        Value ptrOrScalar;
        // We don't care about extending PValues if one ever happened.
        if (auto scalar = targetBV.getIfSBValue())
          ptrOrScalar = scalar;
        if (auto scalar = targetBV.getIfXBValue())
          ptrOrScalar = scalar;
        if (auto scalar = targetBV.getIfMBValue())
          ptrOrScalar = scalar;
        if (ptrOrScalar)
          builder.create<OwnershipUseOp>(loc, ptrOrScalar);
      }
    }
    return success();
  }

  // If we're in a non-raising region, then we have a simple pattern to emit:
  //   contextMgr = EXPRESSION
  //   TARGET = contextMgr.__enter__()
  //   try:
  //     SUITE
  //   finally:
  //     contextMgr.__exit__()
  auto [_, inExceptRegion] = findParentTry(builder.getInsertionBlock());

  if (!inExceptRegion) {
    auto funcOp =
        getBlockParentOfType<LIT::FuncOp>(builder.getInsertionBlock());
    inExceptRegion = funcOp.isThrows();
  }

  // Restore the builder to its current insertion point after parsing.
  llvm::SaveAndRestore builderSaver(builder);
  auto tryOp = builder.create<TryOp>(loc, /*suppressWarnings=*/true);
  // Stub the 'except' and 'else' regions.
  Block *parentExceptBlock = builder.createBlock(&tryOp.getExceptRegion());
  builder.create<TryYieldOp>(loc);
  builder.createBlock(&tryOp.getElseRegion());
  builder.create<TryYieldOp>(loc);
  builder.createBlock(&tryOp.getTryRegion());

  // This emits the call to the 'contextMgr.__exit__()' methods on the
  // context managers in the normal path.
  auto emitNormalExitLogic = [&]() {
    (void)getEmitter().emitNamedMethodCall(
        "__exit__", CallOperands({{contextRV, contextExp}}), ValueDest::none(),
        CallSyntax::kMethodCall, contextExp);
  };

  if (!inExceptRegion) {
    // Parse the body suite.
    if (parseLocalScopeSuite(curIndent))
      return failure();
    builder.create<TryYieldOp>(loc);

    builder.createBlock(&tryOp.getFinallyRegion());
    emitNormalExitLogic();
    builder.create<TryYieldOp>(loc);
    // Stub out the except argument.
    parentExceptBlock->addArgument(builder.getI1Type(), loc);
    return success();
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
  //       contextMGr.__exit__()

  Value excVar;
  {
    // Insert the flag and initialize it to 'True'.
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(tryOp);

    StringAttr name = StringAttr::get(builder.getContext(), "__with_exc__");
    StringAttr lifetimeName = curDeclScope->getAnonymousLifetimeFor(name);
    excVar = builder.create<VarLetDeclOp>(loc, builder.getI1Type(), name,
                                          lifetimeName,
                                          /*isVar=*/true, /*isSynth=*/true);
    builder.create<RefStoreOp>(
        loc, builder.create<mlir::index::BoolConstantOp>(loc, true), excVar);
  }

  // Lookup the error type.
  ASTType errorType = shared.getBuiltinErrorType(getParentDecl(), smLoc);
  if (!errorType.isRegisterPassable(smLoc, shared)) {
    emitError(loc) << errorType << " is not a @register_passable type";
    return failure();
  }

  // Re-raise any exceptions thrown in the nested try.
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(parentExceptBlock);
    Value errVal = parentExceptBlock->addArgument(errorType, loc);
    [[maybe_unused]] LogicalResult result =
        getEmitter().emitRaise(SRValue(errVal), loc);
    assert(succeeded(result) && "expected to be in except context");
  }

  // Generate the nested try. Stub the 'else' and 'finally' regions.
  auto nestedTryOp = builder.create<TryOp>(loc, /*suppressWarnings=*/true);
  builder.create<TryYieldOp>(loc);
  builder.createBlock(&nestedTryOp.getElseRegion());
  builder.create<TryYieldOp>(loc);
  builder.createBlock(&nestedTryOp.getFinallyRegion());
  builder.create<TryYieldOp>(loc);

  // Parse the body into the try region.
  builder.createBlock(&nestedTryOp.getTryRegion());
  if (parseLocalScopeSuite(curIndent))
    return failure();
  builder.create<TryYieldOp>(loc);

  // Set up the except region.  Pseudo code:
  //  except(%val : Error) {
  //    hlcf.if (

  Block *exceptBlock = builder.createBlock(&nestedTryOp.getExceptRegion());
  SRValue errorVal = exceptBlock->addArgument(errorType, loc);

  // Set the flag to 'False'.
  builder.create<RefStoreOp>(
      loc, builder.create<mlir::index::BoolConstantOp>(loc, false), excVar);

  // Pass the error value to the __exit__ method.
  // TODO: this isn't using the same convention that Python does.  We support
  // overloading though and this is going to be way better for anything real
  // that wants to implement this. We can support both styles when we need to.
  CValue exitResult = getEmitter().emitNamedMethodCall(
      "__exit__",
      CallOperands({{contextRV, contextExp}, {errorVal, contextExp}}),
      ValueDest::none(), CallSyntax::kMethodCall, contextExp);
  RValue exitI1RVal = getEmitter().emitI1({exitResult, contextExp});
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
  [[maybe_unused]] LogicalResult result = getEmitter().emitRaise(errorVal, loc);
  assert(succeeded(result) && "expected to be in except context");
  builder.create<HLCF::YieldOp>(loc);

  // Emit the conditional call to __exit__.
  builder.createBlock(&tryOp.getFinallyRegion());
  auto excIf =
      builder.create<HLCF::IfOp>(loc, builder.create<RefLoadOp>(loc, excVar));
  builder.create<TryYieldOp>(loc);
  builder.createBlock(&excIf.getThenRegion());
  emitNormalExitLogic();
  builder.create<HLCF::YieldOp>(loc);
  // Stub the 'else' region.
  builder.createBlock(&excIf.getElseRegion());
  builder.create<HLCF::YieldOp>(loc);
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

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  ExprNode *condExp = nullptr;
  if (parseAssignExpression(condExp, curIndent) ||
      parseToken(Token::colon, "expected ':' after 'if' expression"))
    return failure();

  // Each if/elif conditions could be dynamic or static, use some helpers to
  // generate the right structure.
  SmartVariant<HLCF::IfOp, ParamIfOp> ifOp;
  auto parseCondAndCreateIf = [&](Location loc) -> ParseResult {
    auto emitter = getEmitter();
    // If this is a normal if statement, emit the condition as a SRValue.
    if (!isParamIf) {
      // Create the 'if' and parse the body into its "then" region.
      SRValue condRVal = emitter.emitSRValue(
          {emitter.emitExprI1(condExp, EC_BoolCondition), condExp},
          EC_BoolCondition);
      if (!condRVal)
        return failure();
      ifOp = builder.create<HLCF::IfOp>(loc, condRVal);
      return success();
    }

    // Otherwise, for a @parameter if, we emit the condition as an PValue
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

    ifOp = builder.create<ParamIfOp>(loc, condPVal.get());
    return success();
  };

  auto createThenBlock = [&]() {
    if (auto hifOp = dyn_cast<HLCF::IfOp>(ifOp))
      builder.createBlock(&hifOp.getThenRegion());
    else
      builder.createBlock(&cast<ParamIfOp>(ifOp).getThenRegion());
  };

  auto createElseBlock = [&]() {
    if (auto hifOp = dyn_cast<HLCF::IfOp>(ifOp))
      builder.createBlock(&hifOp.getElseRegion());
    else
      builder.createBlock(&cast<ParamIfOp>(ifOp).getElseRegion());
  };

  auto createYield = [&](Location loc) {
    if (isa<HLCF::IfOp>(builder.getBlock()->getParentOp()))
      builder.create<HLCF::YieldOp>(loc);
    else
      builder.create<ParamYieldOp>(loc);
  };

  if (parseCondAndCreateIf(ifLoc))
    return failure();
  createThenBlock();
  if (failed(parseLocalScopeSuite(curIndent)))
    return failure();
  createYield(ifLoc);

  while (getToken().is(Token::kw_elif) &&
         isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true)) {
    Location elifLoc = translateLocation(consumeToken(Token::kw_elif).getLoc());
    if (parseAssignExpression(condExp, std::nullopt) ||
        parseToken(Token::colon, "expected ':' after 'elif' expression"))
      return failure();

    createElseBlock();
    if (parseCondAndCreateIf(elifLoc))
      return failure();
    createYield(elifLoc);

    createThenBlock();
    if (failed(parseLocalScopeSuite(curIndent)))
      return failure();
    createYield(elifLoc);
  }

  createElseBlock();
  if (isTokenInCurrentStatement(curIndent, /*allowSameIndent=*/true) &&
      consumeIf(Token::kw_else)) {
    if (parseToken(Token::colon, "expected ':' after else"))
      return failure();
    if (failed(parseLocalScopeSuite(curIndent)))
      return failure();
  }
  createYield(ifLoc);
  return success();
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
  if (parseImportModuleName(/*allowRelativeModules=*/true, moduleAttr) ||
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
        currentResolvedModule = &shared.importModule(
            moduleAttr,
            curDeclScope->getIfOperation()->getParentOfType<PackageOp>(),
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
    if (parseImportModuleName(/*allowRelativeModules=*/true, moduleAttr,
                              &boundModuleName))
      return failure();

    // Check for a name binding.
    mlir::LocationAttr boundModuleLocAttr;
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
        /*declNameLoc=*/mlir::LocationAttr());
    getDeclResolver().addDecl(importDecl, importLoc, importDestNameAttr,
                              curDeclScope, getLexer().getCursor(),
                              getLexer().getCursor(), /*indentation=*/-1);
  } while (consumeIf(Token::comma));
  return success();
}

/// Parse a module name for use in an import statement.
/// module          ::=  (identifier ".")* identifier
/// relative_module ::=  "."* module | "."+
ParseResult StmtParser::parseImportModuleName(bool allowRelativeModules,
                                              StringAttr &parsedName,
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
  if (allowRelativeModules) {
    while (true) {
      if (consumeIf(Token::dot))
        moduleNames.push_back("");
      else if (consumeIf(Token::dot_dot_dot))
        llvm::append_range(moduleNames, ArrayRef<StringRef>{"", "", ""});
      else
        break;
    }
  }

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

  parsedName = builder.getStringAttr(llvm::join(moduleNames, "."));
  if (leafModuleName)
    *leafModuleName = moduleNames.back();
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
  SMLoc loc = getToken().getLoc();
  consumeToken();

  StringAttr baseName;
  if (parseIdentifier(baseName, "expected function name"))
    return failure();

  auto funcDecl = builder.create<LIT::FuncOp>(translateLocation(loc));

  // If marked as 'def', remember this on the function decl.
  if (isDef)
    funcDecl.setIsDef(true);

  // Skip the body of this definition: go to a token at the start of the next
  // line at the same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);
  getDeclResolver().addDecl(funcDecl, loc, baseName, curDeclScope, startCursor,
                            getLexer().getCursor(), curIndent);
  return success();
}

/// var_decl_stmt ::= var_or_let identifier ":" expression ["=" expression]
///                 | var_or_let identifier "=" expression
/// var_or_let    ::= "var" | "let"
ParseResult StmtParser::parseLetVarStmt(LexerCursor startCursor,
                                        size_t stmtIndent) {
  // Global var decls are allowed to have decorators, but nothing else.
  bool hasDecorators = startCursor != getLexer().getCursor();
  auto rejectDecorator = [&, declTok = getToken()]() {
    if (!hasDecorators)
      return;
    emitError(declTok.getLoc()) << "'" << declTok.getSpelling()
                                << "' statement does not allow decorators";
  };

  bool isVar = getToken().is(Token::kw_var);
  auto smLoc = consumeToken().getLoc();
  auto loc = translateLocation(smLoc);
  SMLoc identifierLoc;
  StringAttr name;
  if (parseIdentifier(name,
                      isVar ? "expected name for 'var' declaration"
                            : "expected name for 'let' declaration",
                      &identifierLoc))
    return failure();

  auto unresolvedType = UnresolvedType::get(getContext());
  bool delayAddingName = false;
  // If we're in a struct, then this is a field declaration.
  Operation *declOp;
  if (isa<StructDeclOp>(getParentDecl())) {
    rejectDecorator();
    // TODO: implement support for constant struct fields when we have a
    // stronger init model with Definitive Initialization.
    if (!isVar)
      emitError(loc, "'let' fields in structs are not supported yet");
    declOp = builder.create<StructFieldOp>(loc, name, unresolvedType,
                                           DocStringAttr());

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
    auto lifetimeAttr = curDeclScope->getAnonymousLifetimeFor(name);
    declOp = builder.create<VarLetDeclOp>(
        loc, unresolvedType, name, lifetimeAttr, isVar, /*isSynth=*/false);
    delayAddingName = true;
  } else {
    // Otherwise this is a global let/var declaration.
    declOp = builder.create<GlobalVarDeclOp>(loc, name, unresolvedType, isVar);
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
  if (!delayAddingName) {
    getDeclResolver().attachDeclToParentNameTable(&decl, name);
  }
  auto temporaryNameReplace = llvm::make_scope_exit([&]() {
    if (delayAddingName) {
      getDeclResolver().attachDeclToParentNameTable(&decl, name);
    }
  });

  auto varOp = dyn_cast<VarLetDeclOp>(decl);
  if (!varOp) {
    // Parse docstrings for struct fields here.
    parseDocString(decl);
    return success();
  }

  // Local variable declarations inside functions are lexically resolved, so
  // fully resolve the decl now. If an error occurs, skip the declaration and
  // keep parsing to emit as many diagnostics as possible.
  auto declParseError = [&] {
    decl.hasReferenceError = true;
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
    if (parseVarLetInitExpression(initExpr, stmtIndent))
      return declParseError();
  }

  // Now that parsing succeeded, we do IR emission and semantic processing.

  // Handle the initializer if present.
  if (initExpr) {
    // If we have a type, then emit directly into the LValue.  Otherwise emit
    // into the varOp to infer its type.
    ValueDest dest;
    ExprContext exprContext = varOp.getIsVar() ? EC_VarInit : EC_LetInit;
    if (parsedType) {
      varOp.getResult().setType(RefType::get(/*isMutable=*/true, parsedType,
                                             varOp.getType().getLifetime()));
      dest = ValueDest(XLValue(varOp), exprContext);
    } else {
      // If we don't, we emit into the varOp itself, because this will infer the
      // type of the varOp from the initializer expression.
      dest = ValueDest(varOp, exprContext);
    }

    if (!initExpr->emitIR(dest, emitter)) {
      dest.resetForError();
      return declParseError();
    }

    assert(!isa<UnresolvedType>(varOp.getType().getElementAsType()) &&
           "RValue emission should have inferred var type");

  } else if (parsedType) {
    varOp.getResult().setType(RefType::get(/*isMutable=*/true, parsedType,
                                           varOp.getType().getLifetime()));
  } else {
    // If there was neither a type or initializer, reject the var.
    emitError(varOp.getLoc(),
              "declaration must have either a type or an initializer");
    return declParseError();
  }

  // Now that this has been fully checked, we can promote to a LetRegDeclOp
  // if this was a non-parameteric register-passable `let` declaration with
  // an initializer.  We don't care about the address being available and
  // this produces smaller IR.
  ASTType inferredRValueType = ASTType(varOp.getType().getElementAsType());
  if (initExpr && !varOp.getIsVar() &&
      // NOTE: This is assuming type parameters are valid register types.  We
      // will need to build out better support when we have traits, but this is
      // important for kernels in practice today.
      inferredRValueType.isRegisterPassable(initExpr->getLoc(), shared) &&
      // Variable could have multiple uses in invalid cases like "var x = x+1".
      varOp->hasOneUse()) {
    auto theStore = cast<RefStoreOp>(*varOp->user_begin());

    // Create new LetRegDeclOp and put it into the ASTDecl.
    OpBuilder builder(theStore);
    decl.setIRValue(&*builder.create<LetRegDeclOp>(
        varOp.getLoc(), varOp.getNameAttr(), theStore.getArg()));

    // Remove the store and the original VarLetDeclOp.
    theStore->erase();
    varOp->erase();
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
  auto type = UnresolvedType::get(getContext());
  auto value = UnresolvedAliasValueAttr::get(type);
  auto [line, col] = getSourceMgr().getLineAndColumn(smLoc);
  auto declOp = builder.create<AliasDeclOp>(
      loc, ParamDeclAttr::get(mangleParameter(name, line, col), type), value,
      DocStringAttr());

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
  if (isa<StructDeclOp>(getParentDecl()))
    emitTokenError("nested struct not supported here");
  else if (isa<TraitDeclOp>(getParentDecl()))
    emitTokenError("nested struct in a trait not supported here");
  else if (isa<LIT::FuncOp>(getParentDecl()))
    emitTokenError("struct inside a function not supported here");

  auto smLoc = consumeToken(Token::kw_struct).getLoc();
  auto loc = translateLocation(smLoc);

  StringAttr nameAttr;
  if (parseIdentifier(nameAttr, "expected struct name"))
    return failure();

  auto newStruct = builder.create<StructDeclOp>(loc, nameAttr);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(newStruct, smLoc, nameAttr, curDeclScope,
                            startCursor, getLexer().getCursor(), curIndent);
  return success();
}

ParseResult StmtParser::parseTraitStmt(LexerCursor startCursor,
                                       size_t curIndent) {
  // We don't support non-top level traits (yet?).
  if (!isa<FileModuleOp>(getParentDecl()))
    emitTokenError("nested trait not supported here");

  auto smLoc = consumeToken(Token::kw_trait).getLoc();
  auto loc = translateLocation(smLoc);

  StringAttr nameAttr;
  if (parseIdentifier(nameAttr, "expected trait name"))
    return failure();

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
    // Generate debug info for the region argument if requested.
    shared.buildArgDebugInfo(builder, regionArg, parsedArg.name);

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
