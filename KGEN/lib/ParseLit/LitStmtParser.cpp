//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements basic statement parsing.
//
//===----------------------------------------------------------------------===//

#include "ASTDecl.h"
#include "LitDecls.h"
#include "LitExprEmitter.h"
#include "LitExprNodes.h"
#include "LitLexer.h"
#include "LitParserBase.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/SaveAndRestore.h"
#include <filesystem>

using namespace M::KGEN::LIT;
using namespace M::KGEN;
using namespace M;

//===----------------------------------------------------------------------===//
// LitStmtParser
//===----------------------------------------------------------------------===//

/// This class provides the implementation details of the concrete Lightning
/// grammar.
namespace {
struct LitStmtParser : public LitParserBase {
  LitStmtParser(LitLexer &lexer, ASTDecl &containingDecl)
      : LitParserBase(lexer), containingDecl(containingDecl),
        builder(containingDecl.getDeclEndBuilder()) {

    // Create the varDeclCursor with an arbitrary op.  We delete it on
    // destruction of this statement parser.
    varDeclCursor = builder.create<mlir::index::ConstantOp>(
        mlir::UnknownLoc::get(getContext()), 1234567);
  }

  ~LitStmtParser() {
    // The varDeclCursor operation is no longer needed.
    varDeclCursor->erase();
  }

  const ASTDecl &getDecl() const { return containingDecl; }
  OpBuilder &getBuilder() { return builder; }

  // Expression emission.

  ExprEmitter getExprEmitter(bool allowImplicitVarDecl = false) {
    return ExprEmitter(getSharedState(), containingDecl, builder,
                       allowImplicitVarDecl ? varDeclCursor : nullptr);
  }

  ParseResult parseSuite(ssize_t curIndent);
  ParseResult parseLocalScopeSuite(ssize_t curIndent);
  ParseResult parseStmts(size_t minIndent);
  ParseResult parseStmt(bool isSimpleStmt, size_t curIndent);

  // Compound statements.
  ParseResult parseIfStmt(size_t curIndent);
  ParseResult parseWhileStmt(size_t curIndent);
  ParseResult parseTryStmt(size_t curIndent);

  // Simple statements.
  ParseResult parseReturnStmt(size_t returnIndent);
  ParseResult parseRaiseStmt(size_t raiseIndent);
  ParseResult parseBreakOrContinueStmt(LitToken::Kind kind, StringRef name,
                                       StringRef opName);

  // Declarations.
  ParseResult parseIncludeHack();
  ParseResult parseDefFnStmt(LitLexerCursor startCursor, size_t curIndent);
  ParseResult parseStructStmt(LitLexerCursor startCursor, size_t curIndent);
  ParseResult parseVarDeclStmt(LitLexerCursor startCursor, size_t stmtIndent);
  ParseResult parseAliasDeclStmt(LitLexerCursor startCursor, size_t stmtIndent);

private:
  /// This is declaration / scope that we're parsing into.
  ASTDecl &containingDecl;

  /// This is the builder that we are constructing IR into.
  OpBuilder builder;

  /// This is the operation we should install VarDecl's ahead of.  This ensures
  /// they are emitted ahead of anything else in the region for the decl, and
  /// in decls with multiple regions (e.g. function bodies with if statements)
  /// it ensures the decl dominates the whole body.
  Operation *varDeclCursor;
};
} // namespace

/// Parse a suite, which is either a series of comma separated simple_stmt's on
/// one line, or an indented block of statements. curIndent is the containing
/// statement's indentation level.
///
/// suite     ::=  [stmt_list NEWLINE] | NEWLINE INDENT statement+ DEDENT
/// stmt_list ::=  simple_stmt (";" simple_stmt)* [";"]
ParseResult LitStmtParser::parseSuite(ssize_t curIndent) {
  // Ignore empty body at end of file: a `pass` is not required.
  if (getToken().is(LitToken::eof))
    return success();

  // If there is a newline, then parse a list of statements.
  if (auto indent = getToken().getIndentation()) {
    // If the current token is less indented that the source of the suite,
    // then the body is empty.  We don't require a pass.
    if (ssize_t(*indent) <= curIndent)
      return success();
    return parseStmts(curIndent + 1);
  }

  // Otherwise, parse a stmt_list.
  do {
    if (parseStmt(/*isSimpleStmt=*/true, /*NoWrapping=*/~size_t(0)))
      return failure();
    // Stop if we see a semicolon at the end of line or a missing semicolon.
  } while (consumeIf(LitToken::semi) &&
           !getToken().getIndentation().has_value());

  return success();
}
ParseResult LitStmtParser::parseLocalScopeSuite(ssize_t curIndent) {
  // If we are generating debug info, push a local scope for the suite.
  DebugInfo::DIBuilder::ScopeGuard scopeGuard;
  if (getSharedState().diBuilder) {
    SMLoc curLoc = getToken().getLoc();
    auto &sourceMgr = getSharedState().sourceMgr;
    unsigned bufferID = sourceMgr.FindBufferContainingLoc(curLoc);
    auto [line, column] = sourceMgr.getLineAndColumn(curLoc, bufferID);

    scopeGuard = getSharedState().diBuilder->pushLexicalBlock(
        getSharedState().diBuilder->createFile(
            sourceMgr.getMemoryBuffer(bufferID)->getBufferIdentifier(), "/"),
        line, column);
  }

  // Forward to the normal suite parse method.
  return parseSuite(curIndent);
}

/// statements ::= statement+
///
/// This parses statements at the current indentation level or greater, it
/// refuses to parse things at lower indentation level.
ParseResult LitStmtParser::parseStmts(size_t minIndent) {
  while (getToken().isNot(LitToken::eof)) {
    auto indent = getToken().getIndentation();
    if (!indent.has_value())
      return emitError("statements must start at the beginning of a line");

    if (*indent < minIndent)
      break;

    if (parseStmt(/*isSimpleStmt=*/false, *indent))
      return failure();
  }
  return success();
}

/// When `isSimpleStmt` is true, this parses the simple_stmt production,
/// otherwise it parses the broader `statment` production that includes compound
/// statements.
///
/// statement ::= compound_stmt | simple_stmt
///
/// compound_stmt ::= if_stmt
///                 | while_stmt
///                 | for_stmt [TODO]
///                 | try_stmt [TODO]
///                 | with_stmt [TODO]
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
///               | import_stmt [TODO]
///               | future_stmt [TODO]
///               | global_stmt [TODO]
///               | nonlocal_stmtParseResult [TODO]
///
ParseResult LitStmtParser::parseStmt(bool isSimpleStmt, size_t stmtIndent) {
  // This is the cursor for the start of the declaration, that will be used in
  // the signature resolution phase.
  LitLexerCursor startCursor = getLexer().getCursor();

  // This emits an error message if we parsed a decorator, because this
  // statement doesn't support them.
  auto rejectDecorator = [&]() {
    if (startCursor.getState() == getLexer().getCursor().getState())
      return;
    emitError() << "'" << getToken().getSpelling()
                << "' statement does not allow decorators";
  };

  // This lambda is used to generate an error when a compound statement is used
  // in a scenario that expects simple statements.
  auto rejectSimpleStmt = [&]() {
    if (!isSimpleStmt)
      return;
    emitError() << "'" << getToken().getSpelling()
                << "' statement must be on its own line";
  };

  // Skip over any decorators that are present.  These will be reparsed during
  // signature resolution phase of a declaration.
  while (consumeIf(LitToken::at))
    skipUntilIndentation(stmtIndent);

  switch (getToken().getKind()) {
    //===------------------------------------------------------------------===//
    // Compound statements.
    //===------------------------------------------------------------------===//
  case LitToken::kw_if:
    rejectDecorator();  // Decorators not allowed.
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseIfStmt(stmtIndent);
  case LitToken::kw_while:
    rejectDecorator();  // Decorators not allowed.
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseWhileStmt(stmtIndent);
  case LitToken::kw_try:
    rejectDecorator(); // Decorators not allowed.
    rejectSimpleStmt();
    return parseTryStmt(stmtIndent);
  case LitToken::kw_def:
  case LitToken::kw_fn:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseDefFnStmt(startCursor, stmtIndent);
  case LitToken::kw_struct:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseStructStmt(startCursor, stmtIndent);

    //===------------------------------------------------------------------===//
    // Simple statements.
    //===------------------------------------------------------------------===//
  case LitToken::kw___include:
    rejectDecorator(); // Decorators not allowed.
    return parseIncludeHack();

  case LitToken::kw_pass:
  case LitToken::dot_dot_dot:
    // pass_stmt ::= "pass"
    consumeToken();
    return success();
  case LitToken::kw_var:
    return parseVarDeclStmt(startCursor, stmtIndent);
  case LitToken::kw_alias:
    return parseAliasDeclStmt(startCursor, stmtIndent);
  case LitToken::kw_return:
    rejectDecorator(); // Decorators not allowed.
    return parseReturnStmt(stmtIndent);
  case LitToken::kw_raise:
    rejectDecorator(); // Decorators not allowed.
    return parseRaiseStmt(stmtIndent);
  case LitToken::kw_continue:
    rejectDecorator(); // Decorators not allowed.
    return parseBreakOrContinueStmt(LitToken::kw_continue, "continue",
                                    HLCF::ContinueOp::getOperationName());
  case LitToken::kw_break:
    rejectDecorator(); // Decorators not allowed.
    return parseBreakOrContinueStmt(LitToken::kw_break, "break",
                                    HLCF::BreakOp::getOperationName());
  default:
    break;
  }

  // Otherwise, we must have a statement that starts with the expression
  // grammar.
  if (isa<StructDeclOp>(containingDecl)) {
    // TODO: Support type-level meta programs.
    emitError("invalid expression in this context");
  }

  // Parse a single expression, an assignment stmt, or augmented assignment
  // statement.
  ExprNode *expr = nullptr;
  if (parseExpressionOrAssignmentStmt(expr, stmtIndent))
    return failure();

  // If this wasn't an assignment statement, it is just a freestanding
  // expression.  Emit it and ignore the results.
  (void)getExprEmitter(/*allowImplicitVarDecl=*/true).emitDRValue(expr);
  return success();
}

//===----------------------------------------------------------------------===//
// Simple statements.
//===----------------------------------------------------------------------===//

/// Return the nearest parent operation of the block of the given kind.
template <typename OpT>
static OpT getBlockParentOfType(Block *block) {
  if (auto op = dyn_cast<OpT>(block->getParentOp()))
    return op;
  return block->getParentOp()->getParentOfType<OpT>();
}

/// return_stmt ::= "return" [expression_list]
ParseResult LitStmtParser::parseReturnStmt(size_t returnIndent) {
  auto loc = consumeToken(LitToken::kw_return).getLoc();

  // If there is an expression list present, parse it.
  SmallVector<ExprNode *> operandExprs;
  if (!getToken().getIndentation().has_value()) {
    // TODO use hadTrailingSep to return a singleton tuple ex. `return 1,`
    if (parseExpressionList(operandExprs, returnIndent,
                            /*hadTrailingSep=*/nullptr))
      return failure();
  } else {
    // If there was no returned value, then default to "return std::nullopt".
    // This allows type inference to uniformly support all the things that the
    // None literal coerces to (e.g. an Optional type).
    operandExprs.push_back(getNoneExpr(loc));
  }

  // Materialize the expression values into IR.
  SmallVector<Value> operandValues;
  for (auto expr : operandExprs) {
    auto value = getExprEmitter().emitDRValue(expr);
    if (!value)
      return failure();
    operandValues.push_back(value);
  }

  // We don't support formation of tuples / multiple result values yet.
  if (operandValues.size() > 1) {
    emitError(loc, "tuple return not supported yet");
    return success();
  }

  assert(operandValues.size() == 1 &&
         "Should have a single returned value now");
  Value resultValue = operandValues[0];

  // Check the result values match expected types.
  LIT::FuncOp decl = dyn_cast<LIT::FuncOp>(containingDecl);
  if (!decl) {
    emitError(loc, "cannot return from this context");
    return success();
  }

  // Convert the returned value to the returned type of the function.  If the
  // function is a 'raising' function we need to remove the extra variant type
  // to get the normal result type.
  resultValue = getExprEmitter().getAsExpectedType(resultValue, operandExprs[0],
                                                   decl.getNormalResultType());
  if (!resultValue)
    return {};

  // If the enclosing method raises, implicitly wrap the result in a variant.
  Location returnLoc = translateLocation(loc);
  if (decl.getRaises()) {
    auto errorOrType = getSharedState().lookupErrorOrType(resultValue.getType(),
                                                          loc, containingDecl);
    if (!errorOrType)
      return {};
    resultValue = builder.create<POP::VariantCreateOp>(
        returnLoc, Type(errorOrType), resultValue);
  }

  if (isa<LIT::FuncOp>(builder.getInsertionBlock()->getParentOp())) {
    // TODO: Support result parameters.
    builder.create<ReturnOp>(returnLoc, ArrayRef<TypedAttr>(), resultValue);
  } else {
    builder.create<HLCF::ReturnOp>(returnLoc, resultValue);
  }
  // Split the block here. Subsequent statements are dead code.
  builder.setInsertionPointToStart(
      builder.getInsertionBlock()->splitBlock(builder.getInsertionPoint()));
  return success();
}

ParseResult LitStmtParser::parseRaiseStmt(size_t raiseIndent) {
  auto loc = consumeToken(LitToken::kw_raise).getLoc();
  Block *block = builder.getInsertionBlock();
  auto tryOp = getBlockParentOfType<TryOp>(block);

  Value errorVal;
  bool inTry;
  if (!getToken().getIndentation().has_value()) {
    // If there is an error expression, parse and emit it.
    ExprNode *errorExpr;
    if (parseExpression(errorExpr, raiseIndent))
      return failure();
    errorVal = getExprEmitter().emitDRValue(errorExpr);

    // Determine whether we are raising an error inside a 'try'.
    inTry = tryOp && tryOp.getTryRegion().findAncestorBlockInRegion(*block);
  } else {
    // Otherwise, a plain 'raise' refers to the exception currently being
    // handled, only if nested inside an 'except'.
    if (!tryOp || tryOp.getExceptRegion().findAncestorBlockInRegion(*block)) {
      emitError(loc, "no contextual exception to reraise");
      return success();
    }
    errorVal = tryOp.getExceptRegion().getArgument(0);
    inTry = false;
  }

  // If we are raising inside a 'try', just emit a branch to the except region.
  Location raiseLoc = translateLocation(loc);
  if (inTry) {
    builder.create<TryRaiseOp>(raiseLoc, errorVal);
  } else {
    // Wrap the error and propagate it.
    auto func = getBlockParentOfType<LIT::FuncOp>(block);
    if (!func.getRaises()) {
      emitError(loc, "cannot raise error inside method that does not raise");
      return success();
    }
    Value wrappedErr = builder.create<POP::VariantCreateOp>(
        raiseLoc, func.getResultType(), errorVal);
    if (func == block->getParentOp())
      builder.create<ReturnOp>(raiseLoc, ArrayRef<TypedAttr>(), wrappedErr);
    else
      builder.create<HLCF::ReturnOp>(raiseLoc, wrappedErr);
  }
  // Split the block here. Subsequent statements are dead code.
  builder.setInsertionPointToStart(
      block->splitBlock(builder.getInsertionPoint()));
  return success();
}

/// break_stmt ::= "break"
/// continue_stmt ::= "continue"
ParseResult LitStmtParser::parseBreakOrContinueStmt(LitToken::Kind kind,
                                                    StringRef name,
                                                    StringRef opName) {
  llvm::SMLoc loc = consumeToken(kind).getLoc();
  Block *block = builder.getInsertionBlock();

  // Ensure the break statement is being parsed within a loop context.
  if (!getBlockParentOfType<HLCF::LoopOp>(block)) {
    emitError(loc, "'" + name + "' not inside a loop");
    return success();
  }

  // Split the block at the insertion point. Any subsequent statements are dead
  // code. Let region DCE handle it.
  Block *after = block->splitBlock(builder.getInsertionPoint());
  builder.setInsertionPointToEnd(block);
  OperationState state(translateLocation(loc), opName);
  builder.create(state);
  builder.setInsertionPointToStart(after);
  return success();
}

//===----------------------------------------------------------------------===//
// Compound statements.
//===----------------------------------------------------------------------===//

/// while_stmt ::=  "while" assignment_expression ":" suite
///                 ["else" ":" suite]
ParseResult LitStmtParser::parseWhileStmt(size_t curIndent) {
  Location whileLoc =
      translateLocation(consumeToken(LitToken::kw_while).getLoc());

  ExprNode *condExp = nullptr;
  if (parseExpression(condExp, std::nullopt) ||
      parseToken(LitToken::colon, "expected ':' after expression"))
    return failure();

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  auto loopOp = builder.create<HLCF::LoopOp>(whileLoc);
  Block *body = builder.createBlock(&loopOp.getBody());
  builder = OpBuilder::atBlockEnd(body);

  Value condVal = getExprEmitter().emitConditionValueAsI1(condExp);
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
  if (getToken().getIndentation().has_value() &&
      *getToken().getIndentation() >= curIndent &&
      consumeIf(LitToken::kw_else)) {
    builder.setInsertionPointToStart(exit);
    if (parseToken(LitToken::colon, "expected ':' after else") ||
        parseLocalScopeSuite(curIndent))
      return failure();
  }
  return success();
}

/// try_stmt ::= "try" ":" suite "except" [identifier] ":" suite
///              ["else" suite]
ParseResult LitStmtParser::parseTryStmt(size_t curIndent) {
  auto func = getBlockParentOfType<LIT::FuncOp>(builder.getInsertionBlock());
  SMLoc loc = consumeToken(LitToken::kw_try).getLoc();

  // Restore the builder to its current insertion point after parsing.
  llvm::SaveAndRestore builderSaver(builder);
  auto tryOp = builder.create<TryOp>(translateLocation(loc));
  if (parseToken(LitToken::colon, "expected ':' after 'try'"))
    return failure();

  // Parse the try suite.
  builder.createBlock(&tryOp.getTryRegion());
  if (parseLocalScopeSuite(curIndent))
    return failure();
  builder.create<TryYieldOp>(translateLocation(getToken().getLoc()));

  SMLoc errValLoc;
  if (parseToken(LitToken::kw_except, "expected 'except' after try block",
                 &errValLoc))
    return failure();

  // Parse an optional identifier to bind the error.
  StringAttr errName;
  if (getToken().is(LitToken::identifier)) {
    LitToken idTok = consumeToken(LitToken::identifier);
    errName = StringAttr::get(getContext(), idTok.getSpelling());
    errValLoc = idTok.getLoc();
  }

  if (parseToken(LitToken::colon, "expected ':' after 'except'"))
    return failure();

  auto errorType = getSharedState().lookupErrorType(errValLoc, containingDecl);
  if (!errorType)
    return failure();

  Block *exceptBlock = builder.createBlock(&tryOp.getExceptRegion());
  Value errVal =
      exceptBlock->addArgument(errorType, translateLocation(errValLoc));

  // If an identifier was declared for the error value, add a declaration that
  // references it.
  if (errName) {
    if (func.getIsDef()) {
      // If we are parsing inside a 'def', create a mutable LValue to allow
      // reassignment.
      auto varDecl = builder.create<VarDeclOp>(
          errVal.getLoc(), POP::PointerType::get(errVal.getType()), errName);
      getDeclResolver().addFullyResolvedDecl(varDecl, errValLoc, errName,
                                             &containingDecl);
      builder.create<POP::StoreOp>(errVal.getLoc(), errVal, varDecl,
                                   /*alignment=*/std::nullopt);
    } else {
      // If we are parsing inside an 'fn', the error declaration is an RValue.
      getDeclResolver().addFullyResolvedDecl(DRValue(errVal), errName,
                                             errValLoc, &containingDecl);
    }
  }

  // Parse the except suite.
  if (parseLocalScopeSuite(curIndent))
    return failure();
  builder.create<TryYieldOp>(translateLocation(getToken().getLoc()));

  // Parse the else suite if present. Otherwise, leave it as empty.
  builder.createBlock(&tryOp.getElseRegion());
  if (consumeIf(LitToken::kw_else)) {
    if (parseToken(LitToken::colon, "expected ':' after 'else'") ||
        parseLocalScopeSuite(curIndent))
      return failure();
  }
  builder.create<TryYieldOp>(translateLocation(getToken().getLoc()));

  return success();
}

/// if_stmt ::=  "if" assignment_expression ":" suite
///             ("elif" assignment_expression ":" suite)*
///             ["else" ":" suite]
ParseResult LitStmtParser::parseIfStmt(size_t curIndent) {
  Location ifLoc = translateLocation(consumeToken(LitToken::kw_if).getLoc());

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  ExprNode *condExp = nullptr;
  if (parseExpression(condExp, std::nullopt) ||
      parseToken(LitToken::colon, "expected ':' after 'if' expression"))
    return failure();

  Value cond = getExprEmitter().emitConditionValueAsI1(condExp);
  if (!cond)
    return success();

  // Create the 'if' and parse the body into its "then" region.
  auto ifOp = builder.create<HLCF::IfOp>(ifLoc, cond);
  builder.createBlock(&ifOp.getThenRegion());
  if (failed(parseLocalScopeSuite(curIndent)))
    return failure();
  builder.create<HLCF::YieldOp>(ifLoc);

  while (getToken().is(LitToken::kw_elif) &&
         getToken().getIndentation().has_value() &&
         *getToken().getIndentation() >= curIndent) {
    Location elifLoc =
        translateLocation(consumeToken(LitToken::kw_elif).getLoc());
    if (parseExpression(condExp, std::nullopt) ||
        parseToken(LitToken::colon, "expected ':' after 'elif' expression"))
      return failure();

    builder.createBlock(&ifOp.getElseRegion());
    cond = getExprEmitter().emitConditionValueAsI1(condExp);
    if (!cond)
      return success();
    ifOp = builder.create<HLCF::IfOp>(elifLoc, cond);
    builder.create<HLCF::YieldOp>(elifLoc);
    builder.createBlock(&ifOp.getThenRegion());
    if (failed(parseLocalScopeSuite(curIndent)))
      return failure();
    builder.create<HLCF::YieldOp>(ifLoc);
  }

  builder.createBlock(&ifOp.getElseRegion());
  if (getToken().getIndentation().has_value() &&
      *getToken().getIndentation() >= curIndent &&
      consumeIf(LitToken::kw_else)) {
    if (parseToken(LitToken::colon, "expected ':' after else"))
      return failure();
    if (failed(parseLocalScopeSuite(curIndent)))
      return failure();
  }
  builder.create<HLCF::YieldOp>(ifLoc);
  return success();
}

/// Parse the __include "somePath" directive.
ParseResult LitStmtParser::parseIncludeHack() {
  SMLoc includeLoc = consumeToken(LitToken::kw___include).getLoc();
  StringRef path = getTokenSpelling();
  if (parseToken(LitToken::string, "expected path in __include directive"))
    return failure();

  // Strip off the ""'s.
  path = path.drop_front().drop_back();

  llvm::SourceMgr &sourceMgr = getSharedState().sourceMgr;

  // Resolve the absolute filename of the target.
  std::string absolutePath;
  if (std::filesystem::path(path.str()).is_absolute()) {
    absolutePath = path.str();
  } else {
    // Resolve relative paths w.r.t. the including file.
    const llvm::MemoryBuffer *includerBuffer = sourceMgr.getMemoryBuffer(
        sourceMgr.FindBufferContainingLoc(includeLoc));
    assert(includerBuffer && "Must be in a source buffer");
    auto includerPath =
        std::filesystem::path(includerBuffer->getBufferIdentifier().str());
    absolutePath = includerPath.replace_filename(path.str()).string();
  }

  // Ask SourceMgr to open the file in question.
  std::string fullPath;
  unsigned fileID =
      sourceMgr.AddIncludeFile(absolutePath, includeLoc, fullPath);
  if (fileID == 0) {
    emitError(includeLoc, "could not find file '") << path << "'";
    return success(); // Parse success, semantic failure.
  }

  // Push a scope for this new file.
  DebugInfo::DIBuilder::ScopeGuard fileGuard;
  if (getSharedState().diBuilder)
    fileGuard = getSharedState().diBuilder->pushFile(fullPath, "/");

  // Now that we have a MemoryBuffer, we can lex it, and therefore parse it.
  // do so.
  const llvm::MemoryBuffer *includerBuffer = sourceMgr.getMemoryBuffer(fileID);
  LitLexer lexer(getSharedState(), includerBuffer);
  return LitParserBase::parseSuite(containingDecl, lexer);
}

//===----------------------------------------------------------------------===//
// Definition statements
//===----------------------------------------------------------------------===//

ParseResult LitStmtParser::parseDefFnStmt(LitLexerCursor startCursor,
                                          size_t curIndent) {
  // isDef is true when introduced by the 'def' keywords instead of 'fn'.
  bool isDef = getToken().is(LitToken::kw_def);
  SMLoc loc = getToken().getLoc();
  consumeToken();

  StringAttr baseName;
  if (parseIdentifier(baseName, "expected function name"))
    return failure();

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);

  auto funcDecl = builder.create<LIT::FuncOp>(translateLocation(loc));
  if (isDef) {
    funcDecl.setIsDef(true);
    funcDecl.setRaises(true);
  }
  getDeclResolver().addDecl(funcDecl, loc, baseName, &containingDecl,
                            startCursor, getLexer().getCursor(), curIndent);
  return success();
}

ParseResult LitStmtParser::parseVarDeclStmt(LitLexerCursor startCursor,
                                            size_t stmtIndent) {
  auto smLoc = consumeToken(LitToken::kw_var).getLoc();
  auto loc = translateLocation(smLoc);
  StringAttr name;
  if (parseIdentifier(name, "expected name for 'var' declaration"))
    return failure();

  // If we're in a struct, then this is a field declaration.
  Operation *declOp;
  if (isa<StructDeclOp>(containingDecl)) {
    declOp = builder.create<StructFieldOp>(loc, name,
                                           UnresolvedType::get(getContext()));
  } else { // Otherwise this is a local variable definition.

    // Emit the vardecl at the current insertion point.  Unlike implicitly
    // declared variables, let/var declarations are always correctly scoped.
    // TODO: Maintain scopes correctly so we don't have a conflict between
    // things like "if cond: var x = 1 else var x = 2"
    auto varType = POP::PointerType::get(UnresolvedType::get(getContext()));
    declOp = builder.create<VarDeclOp>(loc, varType, name);
  }

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true);

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(declOp, smLoc, name, &containingDecl, startCursor,
                            getLexer().getCursor(), stmtIndent);

  return success();
}

ParseResult LitStmtParser::parseAliasDeclStmt(LitLexerCursor startCursor,
                                              size_t stmtIndent) {
  auto smLoc = consumeToken(LitToken::kw_alias).getLoc();
  auto loc = translateLocation(smLoc);
  StringAttr name;
  if (parseIdentifier(name, "expected name for 'alias' declaration"))
    return failure();

  // Before parsing the rest of the alias, the is unresolved and value is
  // UnresolvedAliasValueAttr.
  auto type = UnresolvedType::get(getContext());
  auto value = UnresolvedAliasValueAttr::get(type);
  auto declOp = builder.create<ParamDeclareOp>(
      loc, ParamDeclAttr::get(name, type), value);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true);

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(declOp, smLoc, name, &containingDecl, startCursor,
                            getLexer().getCursor(), stmtIndent);
  return success();
}

ParseResult LitStmtParser::parseStructStmt(LitLexerCursor startCursor,
                                           size_t curIndent) {
  // We don't support structs in structs (yet?).
  if (isa<StructDeclOp>(containingDecl))
    emitError("nested struct not supported here");

  auto smLoc = consumeToken(LitToken::kw_struct).getLoc();
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
  getDeclResolver().addDecl(newStruct, smLoc, nameAttr, &containingDecl,
                            startCursor, getLexer().getCursor(), curIndent);

  return success();
}

//===----------------------------------------------------------------------===//
// Entry point to this file
//===----------------------------------------------------------------------===//

/// Parse a 'suite' production into the declaration specified by `ASTDecl`.
/// This is the main entrypoint to this file.
ParseResult LitParserBase::parseSuite(ASTDecl &containingDecl,
                                      LitLexer &lexer) {
  if (failed(LitStmtParser(lexer, containingDecl)
                 .parseSuite(containingDecl.getIndentation())))
    return failure();
  // Run region DCE to remove dead code.
  mlir::IRRewriter rewriter(containingDecl.getContext());
  containingDecl.getIfOperation()->walk([&](Operation *op) {
    (void)mlir::eraseUnreachableBlocks(rewriter, op->getRegions());
  });
  return success();
}
