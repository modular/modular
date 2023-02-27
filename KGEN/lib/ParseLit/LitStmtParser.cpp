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

#include "KGEN/CompilationOptions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPEnums.h.inc"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LitExprCalls.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
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

/// Return the nearest parent operation of the block of the given kind.
template <typename OpT>
static OpT getBlockParentOfType(Block *block) {
  if (auto op = dyn_cast<OpT>(block->getParentOp()))
    return op;
  return block->getParentOp()->getParentOfType<OpT>();
}

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

    // If we are parsing into a 'def', then we need a position to synthesize
    // variable definitions at the top of the function.
    if (auto funcOp = dyn_cast<LIT::FuncOp>(containingDecl)) {
      if (funcOp.getIsDef()) {
        // Create the varDeclCursor with an arbitrary op.  We delete it on
        // destruction of this statement parser.
        varDeclCursor = builder.create<mlir::index::ConstantOp>(
            mlir::UnknownLoc::get(getContext()), 1234567);
      }
    }
  }

  ~LitStmtParser() {
    // The varDeclCursor operation is no longer needed.
    if (varDeclCursor)
      varDeclCursor->erase();
  }

  const ASTDecl &getDecl() const { return containingDecl; }
  OpBuilder &getBuilder() { return builder; }

  /// Push a debug info lexical block to represent a local variable scope.
  void pushLocalScope(DebugInfo::DIBuilder::ScopeGuard &scopeGuard);

  // Expression emission.

  ExprEmitter getEmitter(bool allowImplicitVarDecl = false) {
    return ExprEmitter(shared, containingDecl, builder,
                       allowImplicitVarDecl ? varDeclCursor : nullptr);
  }

  /// Get an expression emitter for a parameter expression.
  ExprEmitter getParamEmitter() {
    return ExprEmitter(shared, containingDecl, {}, nullptr);
  }

  ParseResult parseSuite(ssize_t curIndent);
  ParseResult parseLocalScopeSuite(ssize_t curIndent);
  ParseResult parseStmt(bool onlySimpleStmt, bool &parsedCompound,
                        size_t curIndent);

  // Compound statements.
  ParseResult parseIfStmt(LitLexerCursor startCursor, size_t curIndent);
  ParseResult parseWhileStmt(size_t curIndent);
  ParseResult parseForStmt(size_t curIndent);
  ParseResult parseTryStmt(size_t curIndent);

  // Simple statements.
  ParseResult parseReturnStmt(size_t returnIndent);
  ParseResult parseRaiseStmt(size_t raiseIndent);
  ParseResult parseBreakOrContinueStmt(LitToken::Kind kind, StringRef name,
                                       StringRef opName);

  // Declarations.
  ParseResult parseFromImportStmt();
  ParseResult parseImportStmt();
  ParseResult parseDefFnStmt(LitLexerCursor startCursor, size_t curIndent);
  ParseResult parseStructStmt(LitLexerCursor startCursor, size_t curIndent);
  ParseResult parseLetVarStmt(LitLexerCursor startCursor, size_t stmtIndent);
  ParseResult parseAliasDeclStmt(LitLexerCursor startCursor, size_t stmtIndent);
  ParseResult parseMLIRRegionStmt(LitLexerCursor startCursor, size_t curIndent);

private:
  /// This is declaration / scope that we're parsing into.
  ASTDecl &containingDecl;

  /// This is the builder that we are constructing IR into.
  OpBuilder builder;

  /// This is the operation we should install VarDecl's ahead of if we are
  /// parsing into a 'def'.  This ensures they are emitted ahead of anything
  /// else in the region for the decl, and in decls with multiple regions (e.g.
  /// function bodies with if statements) it ensures the decl dominates the
  /// whole body.
  Operation *varDeclCursor = nullptr;
};
} // namespace

void LitStmtParser::pushLocalScope(
    DebugInfo::DIBuilder::ScopeGuard &scopeGuard) {
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
ParseResult LitStmtParser::parseSuite(ssize_t curIndent) {
  // Ignore empty body at end of file: a `pass` is not required.
  if (getToken().is(LitToken::eof))
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
    } while (consumeIf(LitToken::semi) &&
             !getToken().getIndentation().has_value());
    return success();
  };

  // If this suite is on the same line as the enclosing entity, just parse a
  // single stmt_list.
  auto indent = getToken().getIndentation();
  if (!indent.has_value())
    return parseStmtListOrCompound(/*stmtListOnly*/ true,
                                   /*NoWrapping=*/~size_t(0));

  // If there is a newline, then parse a list of statements which can be either
  // a statement list or a compount_stmt.  Parse all the statements that are
  // more nested than this suite.
  while (getToken().isNot(LitToken::eof)) {
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

ParseResult LitStmtParser::parseLocalScopeSuite(ssize_t curIndent) {
  // If we are generating debug info, push a local scope for the suite.
  DebugInfo::DIBuilder::ScopeGuard scopeGuard;
  if (shared.diBuilder)
    pushLocalScope(scopeGuard);

  // Forward to the normal suite parse method.
  return parseSuite(curIndent);
}

/// Emit a warning when an expression is emitted at statement context, and it
/// returns a result.
static void diagnoseIgnoredResult(const ExprNode *expr, AnyValue value,
                                  LitSharedState &shared) {
  ASTType valueType = value.getRValueType();

  // Return true if the specified type can be implicitly ignored.
  // TODO: Should have a better way to say that it is safe to implicily ignore a
  // value of a type (e.g. a type decorator)
  auto isImplicitlyIgnorableType = [&](ASTType type) -> bool {
    // TODO: This is incorrect for throwing functions that return None.
    return type.isEqualCanon(shared.getNoneType()) ||
           type.isEqualCanon(shared.getTypeCheckErrorType());
  };

  if (isImplicitlyIgnorableType(valueType))
    return;

  // If this type is a function with no arguments and an ignorable type, we
  // emit a warning with a fix it hint suggesting that it get called.
  if (auto sig = dyn_cast<SignatureType>(valueType.mlirType)) {
    // TODO: This is incorrect for default arguments and varargs.
    assert(sig.getValueResults().size() == 1);
    if (sig.getValueInputs().size() == 0 &&
        isImplicitlyIgnorableType(sig.getValueResults()[0])) {
      // Find end of token.  TODO: Gross.
      auto endLoc = expr->getRange().getEnd();
      size_t tokenSize = LitLexer::getTokenLength(shared, endLoc);
      endLoc = SMLoc::getFromPointer(endLoc.getPointer() + tokenSize);
      auto insertRange = LitSourceRange::getByteLevel(endLoc, endLoc);

      shared.emitWarning(expr->getLoc())
          << "function pointer was formed but not called, did you forget '()'s?"
          << expr->getRange() << LitFixIt(insertRange, "()");
      return;
    }
  }

  // If the expression returned an unevaluated coroutine, then the expression
  // should be awaited.
  if (isa<POP::CoroutineType>(valueType.mlirType)) {
    auto loc = expr->getRange().getStart();
    auto insertRange = LitSourceRange::getByteLevel(loc, loc);
    shared.emitWarning(expr->getLoc())
        << "coroutine was never awaited" << expr->getRange()
        << LitFixIt(insertRange, "await ");
    return;
  }

  // Otherwise emit a warning, and suggest assigning to _.
  auto startLoc = expr->getRange().getStart();
  shared.emitWarning(expr->getLoc())
      << valueType << " value is unused" << expr->getRange()
      << LitFixIt(LitSourceRange::getByteLevel(startLoc, startLoc), "_ = ");
}

/// When `onlySimpleStmt` is true, this parses the simple_stmt production,
/// otherwise it parses the broader `statment` production that includes compound
/// statements.  This sets `parsedCompound` to true if `onlySimpleStmt` was
/// false and we parsed a compound stmt.
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
///               | import_stmt
///               | future_stmt [TODO]
///               | global_stmt [TODO]
///               | nonlocal_stmtParseResult [TODO]
///
ParseResult LitStmtParser::parseStmt(bool onlySimpleStmt, bool &parsedCompound,
                                     size_t stmtIndent) {
  // This is the cursor for the start of the declaration, that will be used in
  // the signature resolution phase.
  LitLexerCursor startCursor = getLexer().getCursor();

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

  // Skip over any decorators that are present.  These will be reparsed during
  // signature resolution phase of a declaration.
  while (consumeIf(LitToken::at))
    skipUntilIndentation(stmtIndent);

  switch (getToken().getKind()) {
    //===------------------------------------------------------------------===//
    // Compound statements.
    //===------------------------------------------------------------------===//
  case LitToken::kw_if:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseIfStmt(startCursor, stmtIndent);
  case LitToken::kw_for:
    rejectDecorator();  // Decorators not allowed.
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseForStmt(stmtIndent);
  case LitToken::kw_while:
    rejectDecorator();  // Decorators not allowed.
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseWhileStmt(stmtIndent);
  case LitToken::kw_try:
    rejectDecorator(); // Decorators not allowed.
    rejectSimpleStmt();
    return parseTryStmt(stmtIndent);
  case LitToken::kw_async:
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
  case LitToken::kw_from:
    rejectDecorator(); // Decorators not allowed.
    return parseFromImportStmt();
  case LitToken::kw_import:
    rejectDecorator(); // Decorators not allowed.
    return parseImportStmt();

  case LitToken::kw_pass:
  case LitToken::dot_dot_dot:
  case LitToken::string:
    // doc string
    // pass_stmt ::= "pass"
    consumeToken();
    return success();
  case LitToken::kw_let:
  case LitToken::kw_var:
    return parseLetVarStmt(startCursor, stmtIndent);
  case LitToken::kw_alias:
    return parseAliasDeclStmt(startCursor, stmtIndent);
  case LitToken::kw___mlir_region:
    rejectDecorator();
    rejectSimpleStmt();
    return parseMLIRRegionStmt(startCursor, stmtIndent);
  case LitToken::kw_return:
    rejectDecorator(); // Decorators not allowed.
    return parseReturnStmt(stmtIndent);
  case LitToken::kw_raise:
    rejectDecorator(); // Decorators not allowed.
    return parseRaiseStmt(stmtIndent);
  case LitToken::kw_continue:
    rejectDecorator(); // Decorators not allowed.
    return parseBreakOrContinueStmt(LitToken::kw_continue, "continue",
                                    LIT::ContinueOp::getOperationName());
  case LitToken::kw_break:
    rejectDecorator(); // Decorators not allowed.
    return parseBreakOrContinueStmt(LitToken::kw_break, "break",
                                    LIT::BreakOp::getOperationName());
  default:
    break;
  }

  // TODO: Nail down a model for struct-level meta-programs.

  // Parse a single expression, an assignment stmt, or augmented assignment
  // statement.
  ExprNode *expr = nullptr;
  if (parseExpressionOrAssignmentStmt(expr, stmtIndent))
    return failure();

  // Emit the expression and ignore the results.  If it is an assignment
  // statement, it will return None.  Other expressions can return whatever they
  // will naturally return.
  auto emitter = getEmitter(/*allowImplicitVarDecl=*/true);
  auto result = expr->emitIR(emitter, /*No Contextual Type*/ {});
  if (!result)
    return success();

  // Emit a warning if the result is a value we should warn when unused.
  if (!getBlockParentOfType<LIT::FuncOp>(builder.getInsertionBlock())
           .getIsDef())
    diagnoseIgnoredResult(expr, result, shared);
  return success();
}

//===----------------------------------------------------------------------===//
// Simple statements.
//===----------------------------------------------------------------------===//

/// return_stmt ::= "return"(return_param_spec)? [expression_list]
/// return_param_spec ::= "[" expression ("," expression)* "]"
///
/// The return param spec is required if the enclosing function has return
/// parameters, otherwise it is absent.
ParseResult LitStmtParser::parseReturnStmt(size_t returnIndent) {
  LIT::FuncOp decl = dyn_cast<LIT::FuncOp>(containingDecl);

  auto loc = consumeToken(LitToken::kw_return).getLoc();

  // If this function declaration requires result parameters, parse them
  // specially.  'decl' may be missing in an invalid return.  This handles the
  // ambiguity where there may be result parameters /and/ a list-display in
  // square brackets may also be used as a normal result.
  ExprNode *resultParams = nullptr;
  if (decl && !decl.getResultParams().empty()) {
    auto numResultParams = decl.getResultParams().size();
    // Catch obvious missed parameter list.
    if (getToken().isNot(LitToken::l_square)) {
      emitError(loc, "expected '[' in function that returns ")
          << numResultParams << " result parameter" << plural(numResultParams);
    } else if (parseExpression(resultParams, returnIndent)) {
      return failure();
    }
  }

  // If there is an expression list present, parse it.
  SmallVector<ExprNode *> operandExprs;
  if (!getToken().getIndentation().has_value() ||
      *getToken().getIndentation() > returnIndent) {
    // TODO use hadTrailingSep to return a singleton tuple ex. `return 1,`
    if (parseExpressionList(operandExprs, returnIndent,
                            /*hadTrailingSep=*/nullptr))
      return failure();
  } else {
    // If there was no returned value, then default to "return std::nullopt".
    // This allows type inference to uniformly support all the things that the
    // None literal coerces to (e.g. an std::optional type).
    operandExprs.push_back(getNoneExpr(loc));
  }

  // We don't support formation of tuples / multiple result values yet.
  if (operandExprs.size() != 1) {
    emitError(loc, "tuple return not supported yet")
        << LitSourceRange(operandExprs.front()->getRangeStart(),
                          operandExprs.back()->getRangeEnd());
    return success();
  }

  // Materialize the expression values into IR.
  RValue resultValue = getEmitter().emitExprRValue(operandExprs[0]);

  // Ok, now that we parsed all the tokens for this statement, do semantic
  // analysis.
  if (!decl) {
    emitError(loc, "cannot return from this context");
    return success();
  }

  auto emitter = getEmitter();

  // Check the result parameters if present.
  SmallVector<TypedAttr> resultParamValues;
  if (resultParams) {
    auto resultParamList = dyn_cast<ListNode>(resultParams);
    size_t numResultParams = decl.getResultParams().size();
    if (!resultParamList || resultParamList->exprs.size() != numResultParams) {
      emitError(resultParamList->getLoc(), "expected ")
          << numResultParams << " result parameter" << plural(numResultParams)
          << resultParams->getRange();
      return success();
    }
    for (auto [paramExpr, param] :
         llvm::zip(resultParamList->exprs, decl.getResultParams())) {
      auto result = emitter.emitExprPRValue(paramExpr, param.getType(),
                                            " in result parameter list");
      if (!result)
        return success();
      resultParamValues.push_back(result);
    }
  }

  // Convert the returned value to the returned type of the function.  If the
  // function is a 'raising' function we need to remove the extra variant type
  // to get the normal result type.
  // TODO(memory_primary): Return slots.
  auto resultSRValue = emitter.emitSRValue(
      {emitter.getAsExpectedType({resultValue, operandExprs[0]},
                                 decl.getResultType(),
                                 // TODO(memory-primary): Return slots.
                                 ValueDest(), " in return"),
       operandExprs[0]});
  if (!resultSRValue)
    return {};

  builder.create<LIT::ReturnOp>(translateLocation(loc), resultSRValue,
                                resultParamValues);
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
    errorVal = getEmitter().emitExprSRValue(errorExpr);
    if (!errorVal)
      return success();

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

  if (!inTry && !getBlockParentOfType<LIT::FuncOp>(block).isThrows()) {
    emitError(loc, "cannot raise error in a context that cannot raise");
    return success();
  }
  Location raiseLoc = translateLocation(loc);
  builder.create<LIT::RaiseOp>(raiseLoc, errorVal);
  return success();
}

/// break_stmt ::= "break"
/// continue_stmt ::= "continue"
ParseResult LitStmtParser::parseBreakOrContinueStmt(LitToken::Kind kind,
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

  RValue condRVal = getEmitter().emitExprConditionValueAsI1(condExp);
  Value condVal = getEmitter().emitSRValue({AnyValue(condRVal), condExp});
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

/// for_stmt ::=  "for" target_list "in" starred_list ":" suite
///              ["else" ":" suite]
ParseResult LitStmtParser::parseForStmt(size_t curIndent) {
  Location forLoc = translateLocation(consumeToken(LitToken::kw_for).getLoc());

  // parse [target_list] in [starred_list]
  // for now, we expect target_list to be an identifier
  // the [starred_list] needs to be a sequence with a __iter__ method that
  // returns a type that defines __len__ and __next__
  StringAttr target = StringAttr::get(getContext(), getToken().getSpelling());
  SMLoc identifierLocation;
  if (parseToken(LitToken::identifier, "expected identifier for target in for",
                 &identifierLocation))
    return failure();
  if (parseToken(LitToken::kw_in, "expected 'in' after target identifier. Note "
                                  "that target lists are not yet supported."))
    return failure();

  ExprNode *seqExp = nullptr;
  if (parseExpression(seqExp, std::nullopt) ||
      parseToken(LitToken::colon, "expected ':' after expression"))
    return failure();

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  // retrieve the iterator object from the sequence expression
  ASTExprAnd<AnyValue> loadedSeq = {
      getEmitter().emitExprRValue(seqExp, ValueDest()), seqExp};
  if (!loadedSeq.ir)
    return {};

  AnyValue rangeValue =
      getEmitter().emitNamedMethodCall("__iter__", {loadedSeq}, ValueDest(),
                                       CallSyntax::kImplicitConvert, seqExp);
  if (!rangeValue)
    return {};
  LIT::VarDeclOp range_ref = builder.create<LIT::VarDeclOp>(
      forLoc, POP::PointerType::get(rangeValue.getType()), "$RANGE");
  builder.create<POP::StoreOp>(forLoc, rangeValue.getIfSRValue(), range_ref,
                               std::nullopt);

  HLCF::LoopOp loopOp = builder.create<HLCF::LoopOp>(forLoc);
  Block *body = builder.createBlock(&loopOp.getBody());
  builder = OpBuilder::atBlockEnd(body);

  // For Loop condition: if the length of the range is greater than zero,
  // continue. Otherwise break
  SRValue loaded_range = SRValue(builder.create<POP::LoadOp>(
      translateLocation(seqExp->getLoc()), range_ref, std::nullopt));
  AnyValue current_length = getEmitter().emitNamedMethodCall(
      "__len__", {{loaded_range, seqExp}}, ValueDest(),
      CallSyntax::kImplicitConvert, seqExp);
  if (!current_length)
    return {};
  SRValue pop_length = getEmitter().emitBoxedIntAsPopScalar(
      current_length.getIfSRValue(), seqExp);
  if (!pop_length)
    return {};
  Value pop_zero = builder.create<POP::CastFromBuiltinOp>(
      translateLocation(seqExp->getLoc()),
      POP::SIMDType::get(builder.getContext(), 1,
                         KGENDType(KGENDType::ExtraCases::index)),
      builder.create<mlir::index::ConstantOp>(forLoc, 0));
  POP::CmpOp cmpOp = builder.create<POP::CmpOp>(
      forLoc, KGEN::POP::CmpPredicate::GT, pop_length, pop_zero);
  POP::CastToBuiltinOp should_continue =
      builder.create<POP::CastToBuiltinOp>(forLoc, builder.getI1Type(), cmpOp);

  if (!should_continue)
    return success(); // IRGen error already emitted; parse succeeded!

  // Generate the for condition check.
  auto condOp = builder.create<HLCF::IfOp>(forLoc, should_continue);
  builder.createBlock(&condOp.getThenRegion());
  builder.create<HLCF::YieldOp>(forLoc);
  Block *exit = builder.createBlock(&condOp.getElseRegion());
  builder.create<HLCF::BreakOp>(forLoc);

  // Create the body. Add Target element to the continue block by calling next
  builder.setInsertionPointAfter(condOp);
  AnyValue nextCall = getEmitter().emitNamedMethodCall(
      "__next__", {{LValue(range_ref), seqExp}}, ValueDest(),
      CallSyntax::kImplicitConvert, seqExp);
  if (!nextCall) {
    return {};
  }
  getDeclResolver().addFullyResolvedDecl(nextCall.getIfSRValue(), target,
                                         identifierLocation, &containingDecl);
  if (failed(parseLocalScopeSuite(curIndent)))
    return failure();
  builder.create<HLCF::ContinueOp>(forLoc);

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

  auto errorType = shared.lookupErrorType(errValLoc, containingDecl);
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
      getDeclResolver().addFullyResolvedDecl(SRValue(errVal), errName,
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
ParseResult LitStmtParser::parseIfStmt(LitLexerCursor startCursor,
                                       size_t curIndent) {
  // This is enabled with the @parameter decorator.
  bool isParamIf = false;

  // We parse the decorators for the 'if' if they exist.
  if (startCursor != getLexer().getCursor()) {
    startCursor.restore(getLexer());
    for (auto *decorator : parseDecorators(curIndent)) {
      // Handle recognized decorators.
      if (auto *dre = dyn_cast<DeclRefNode>(decorator)) {
        if (dre->spelling == "parameter") {
          isParamIf = true;
          continue;
        }
      }

      emitError(decorator->getLoc(), "unsuported decorator on 'if' statement")
          << decorator->getRange();
    }
  }
  Location ifLoc = translateLocation(getToken().getLoc());
  if (parseToken(LitToken::kw_if, "expected 'if' token after decorators"))
    return failure();

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore builderSaver(builder);

  ExprNode *condExp = nullptr;
  if (parseExpression(condExp, std::nullopt) ||
      parseToken(LitToken::colon, "expected ':' after 'if' expression"))
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
          {AnyValue(emitter.emitExprConditionValueAsI1(condExp)), condExp});
      if (!condRVal)
        return failure();
      ifOp = builder.create<HLCF::IfOp>(loc, condRVal);
      return success();
    }

    // Otherwise, for a @parameter if, we emit the condition as an PRValue
    // without a builder.
    RValue condRVal = getParamEmitter().emitExprConditionValueAsI1(condExp);
    if (!condRVal)
      return failure();
    if (!condRVal.getIfPRValue())
      return emitError(condExp->getLoc(), "@parameter 'if' requires a "
                                          "parameter expression as a condition")
             << condExp->getRange();

    ifOp = builder.create<ParamIfOp>(loc, condRVal.getIfPRValue().get());
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

  while (getToken().is(LitToken::kw_elif) &&
         getToken().getIndentation().has_value() &&
         *getToken().getIndentation() >= curIndent) {
    Location elifLoc =
        translateLocation(consumeToken(LitToken::kw_elif).getLoc());
    if (parseExpression(condExp, std::nullopt) ||
        parseToken(LitToken::colon, "expected ':' after 'elif' expression"))
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
  if (getToken().getIndentation().has_value() &&
      *getToken().getIndentation() >= curIndent &&
      consumeIf(LitToken::kw_else)) {
    if (parseToken(LitToken::colon, "expected ':' after else"))
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
ParseResult LitStmtParser::parseFromImportStmt() {
  consumeToken(LitToken::kw_from);

  // TODO: Support packages, this currently just handles basic module importing.

  // Parse the relative module we are importing from.
  if (getToken().isAny(LitToken::dot, LitToken::dot_dot_dot))
    return emitTokenError(
        "TODO: relative package imports are not yet supported");
  SMLoc importLoc = getToken().getLoc();
  StringRef moduleName = getTokenSpelling();
  if (parseToken(LitToken::identifier, "expected module name") ||
      parseToken(LitToken::kw_import, "expected 'import' after module name"))
    return failure();

  // Check for a wildcard import.
  if (consumeIf(LitToken::star)) {
    containingDecl.addUnresolvedWildCardImport(
        builder.getStringAttr(moduleName), importLoc);
    return success();
  }

  // Parse the set of constructs to import.
  bool isTupleImport = consumeIf(LitToken::l_paren);
  do {
    // Parse the next construct to import.
    SMLoc importSourceNameLoc = getToken().getLoc();
    StringRef importSourceName = getTokenSpelling();
    if (parseToken(LitToken::identifier, "expected construct name to import"))
      return failure();
    StringRef importDestName = importSourceName;
    if (consumeIf(LitToken::kw_as)) {
      importDestName = getTokenSpelling();
      if (parseToken(LitToken::identifier,
                     "expected name to import '" + importSourceName + "' as"))
        return failure();
    }

    // Create an unresolved decl for this import.
    StringAttr importDestNameAttr = builder.getStringAttr(importDestName);
    auto importDecl = builder.create<LIT::UnresolvedImportOp>(
        translateLocation(importSourceNameLoc),
        builder.getStringAttr(moduleName), importDestNameAttr,
        builder.getStringAttr(importSourceName));
    getDeclResolver().addDecl(
        importDecl, importSourceNameLoc, importDestNameAttr, &containingDecl,
        getLexer().getCursor(), getLexer().getCursor(), /*indentation=*/-1);

    // Check for more elements to import.
    if (!consumeIf(LitToken::comma))
      break;
    // For tuple imports, there may optionally be a trailing comma at the end of
    // the list.
    if (isTupleImport && getToken().is(LitToken::r_paren))
      break;
  } while (true);

  // Check for the end of the tuple import.
  if (isTupleImport &&
      parseToken(LitToken::r_paren, "expected ')' after import list"))
    return failure();
  return success();
}

/// import_stmt ::=  "import" module ["as" identifier]
///                  ("," module ["as" identifier])*
/// module      ::=  (identifier ".")* identifier
ParseResult LitStmtParser::parseImportStmt() {
  consumeToken(LitToken::kw_import);

  // TODO: Support packages, this currently just handles basic module importing.

  // Parse the next module to import.
  do {
    SMLoc importLoc = getToken().getLoc();
    StringRef moduleName = getTokenSpelling();
    if (parseToken(LitToken::identifier, "expected module name"))
      return failure();

    // Check for a name binding.
    StringRef boundModuleName = moduleName;
    if (consumeIf(LitToken::kw_as)) {
      boundModuleName = getTokenSpelling();
      if (parseToken(LitToken::identifier, "expected name to bind import"))
        return failure();
    }

    // Create an unresolved decl for the import.
    StringAttr importDestNameAttr = builder.getStringAttr(boundModuleName);
    auto importDecl = builder.create<LIT::UnresolvedImportOp>(
        translateLocation(importLoc), builder.getStringAttr(moduleName),
        importDestNameAttr, /*declName=*/StringAttr());
    getDeclResolver().addDecl(importDecl, importLoc, importDestNameAttr,
                              &containingDecl, getLexer().getCursor(),
                              getLexer().getCursor(), /*indentation=*/-1);
  } while (consumeIf(LitToken::comma));
  return success();
}

//===----------------------------------------------------------------------===//
// Definition statements
//===----------------------------------------------------------------------===//

ParseResult LitStmtParser::parseDefFnStmt(LitLexerCursor startCursor,
                                          size_t curIndent) {
  bool isAsync = consumeIf(LitToken::kw_async);
  // isDef is true when introduced by the 'def' keywords instead of 'fn'.
  bool isDef = getToken().is(LitToken::kw_def);
  SMLoc loc = getToken().getLoc();
  consumeToken();

  StringAttr baseName;
  if (parseIdentifier(baseName, "expected function name"))
    return failure();

  auto funcDecl = builder.create<LIT::FuncOp>(translateLocation(loc));
  // Compute the correct function effects.
  auto effects = FnEffects::None;
  if (isDef) {
    funcDecl.setIsDef(true);
    effects = bitEnumSet(effects, FnEffects::Throws);
  }
  if (isAsync)
    effects = bitEnumSet(effects, FnEffects::Async);
  if (effects != FnEffects::None)
    funcDecl.setSignature(funcDecl.getSignature().setFnEffect(effects));

  // Skip the body of this definition: go to a token at the start of the next
  // line at the same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);
  getDeclResolver().addDecl(funcDecl, loc, baseName, &containingDecl,
                            startCursor, getLexer().getCursor(), curIndent);
  return success();
}

ParseResult LitStmtParser::parseLetVarStmt(LitLexerCursor startCursor,
                                           size_t stmtIndent) {
  bool isLet = getToken().is(LitToken::kw_let);
  auto smLoc = consumeToken().getLoc();
  auto loc = translateLocation(smLoc);
  StringAttr name;
  if (parseIdentifier(name, isLet ? "expected name for 'let' declaration"
                                  : "expected name for 'var' declaration"))
    return failure();

  auto unresolvedType = UnresolvedType::get(getContext());
  // If we're in a struct, then this is a field declaration.
  Operation *declOp;
  if (isa<StructDeclOp>(containingDecl)) {
    // TODO: implement support for constant struct fields when we have a
    // stronger init model with Definitive Initialization.
    if (isLet)
      emitError(loc, "'let' fields in structs are not supported yet");
    declOp = builder.create<StructFieldOp>(loc, name, unresolvedType);
  } else if (isLet) {
    declOp = builder.create<LetDeclOp>(loc, unresolvedType, name);

  } else {
    // Otherwise this is a local variable definition.

    // Emit the vardecl at the current insertion point.  Unlike implicitly
    // declared variables, let/var declarations are always correctly scoped.
    // TODO (Issue#5005): Maintain scopes correctly so we don't have a conflict
    // between things like "if cond: var x = 1 else var x = 2"
    auto varType = POP::PointerType::get(unresolvedType);
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
    emitTokenError("nested struct not supported here");

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

/// An MLIR region declaration defines a single block region body as a suite
/// with the declaration arguments corresponding to the region arguments. It is
/// used to define regions for MLIR operations.
///
/// region_stmt ::= "__mlir_region" identifier "(" [argument_list] ")" ":" suite
ParseResult LitStmtParser::parseMLIRRegionStmt(LitLexerCursor startCursor,
                                               size_t curIndent) {
  SMLoc loc = consumeToken(LitToken::kw___mlir_region).getLoc();

  // We will be moving the builder into the contained region, so save it here.
  llvm::SaveAndRestore builderSaver(builder);

  // Resolve the signature and the body immediately.
  StringAttr identifier;
  if (parseIdentifier(identifier, "expected a region name") ||
      parseToken(LitToken::l_paren, "expected '(' for parameter list"))
    return failure();

  // Create the decl corresponding to the region declaration.
  auto op = builder.create<UnboundRegionOp>(translateLocation(loc));
  ASTDecl &decl =
      getDeclResolver().addDecl(op, loc, identifier, &containingDecl,
                                startCursor, getLexer().getCursor(), curIndent);
  decl.resolvedness = DeclResolvedness::fully;

  // Parse the argument list if present.
  struct RegionArgument {
    StringAttr name;
    SMLoc loc;
  };
  SmallVector<RegionArgument> args;
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  if (!consumeIf(LitToken::r_paren)) {
    // Parse simple argument: MLIR operations don't have input conventions.
    auto parseArg = [&]() -> ParseResult {
      RegionArgument &arg = args.emplace_back();
      ExprNode *typeExpr;
      if (getLocation(arg.loc) ||
          parseIdentifier(arg.name, "expected an identifier") ||
          parseToken(LitToken::colon, "expected ':' after region argument") ||
          parseExpression(typeExpr, std::nullopt))
        return failure();
      ASTType type = getEmitter().emitExprType(typeExpr);
      if (!type)
        return failure();
      argTypes.push_back(type);
      argLocs.push_back(translateLocation(arg.loc));
      return success();
    };
    if (parseCommaSeparatedList(parseArg, LitToken::r_paren))
      return failure();
    consumeToken(LitToken::r_paren);
  }

  builder.createBlock(&op.getRegion());
  for (auto [regionArg, parsedArg] :
       llvm::zip(op.getRegion().addArguments(argTypes, argLocs), args)) {
    // Generate debug info for the region argument if requested.
    DebugInfo::DIBuilder *diBuilder = shared.diBuilder.get();
    if (diBuilder &&
        shared.options.debugLevel == CompilationOptions::kFullDebugInfo) {
      auto argLoc = regionArg.getLoc()->findInstanceOf<FileLineColLoc>();
      DebugInfo::DILocalVariableAttr var = diBuilder->createLocalVariable(
          parsedArg.name, diBuilder->createFile(argLoc), argLoc.getLine(),
          regionArg.getArgNumber() + 1, /*alignInBits=*/0,
          DebugInfo::DIUnresolvedMLIRType::get(regionArg.getType()));
      builder.create<DebugInfo::ValueOp>(regionArg.getLoc(), regionArg, var);
    }
    // Add the declaration for the argument within the region declaration.
    getDeclResolver().addFullyResolvedDecl(SRValue(regionArg), parsedArg.name,
                                           parsedArg.loc, &decl);
  }

  DebugInfo::DIBuilder::ScopeGuard scopeGuard;
  if (shared.diBuilder)
    pushLocalScope(scopeGuard);
  if (parseToken(LitToken::colon, "expected ':' after region argument list") ||
      LitParserBase::parseSuite(decl, lexer))
    return failure();

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
  return success();
}
