//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONPARSER_H
#define KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONPARSER_H

#include "JITExecutionUnit.h"
#include "MojoExpressionSourceCode.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN::LIT {
class FuncOp;
class StructDeclOp;
} // namespace M::KGEN::LIT

namespace M::KGEN::Mojo {
class MojoUserExpression;

class MojoExpressionParser {
public:
  MojoExpressionParser(lldb_private::ExecutionContextScope *exeScope,
                       MojoUserExpression &expr,
                       const lldb_private::EvaluateExpressionOptions &options);
  ~MojoExpressionParser();

  /// Attempt to find possible command line completions for the given
  /// expression.
  LogicalResult complete(lldb_private::CompletionRequest &request,
                         unsigned line, unsigned pos, unsigned typedPos) {
    return failure();
  }

  /// Rewrite the expression using the fix-its contained in the diagnostic
  /// manager.
  LogicalResult
  rewriteExpression(lldb_private::DiagnosticManager &diagnosticManager);

  /// Parse a single expression and convert it to IR.
  LogicalResult parse(lldb_private::DiagnosticManager &diagnosticManager);

  /// Ready an already-parsed expression for execution, possibly evaluating it
  /// statically.
  lldb_private::Status
  prepareForExecution(lldb::addr_t &funcAddr, lldb::addr_t &funcEnd,
                      std::shared_ptr<JITExecutionUnit> &executionUnit,
                      lldb_private::ExecutionContext &exeCtx,
                      lldb_private::ExecutionPolicy executionPolicy,
                      std::optional<MojoExpressionSourceCode> sourceCode,
                      bool keepResultInMemory);

  /// Get the name of the module where expressions are JITted.
  static StringRef getJITModuleName() { return "__lldb_module__"; }

private:
  //===--------------------------------------------------------------------===//
  // Persistent Variables
  //===--------------------------------------------------------------------===//

  /// Process the variables within the given function that should become
  /// persistent when the function is executed within a REPL. Persistent
  /// variables are added as fields to the given state struct, and references
  /// within the function are rewritten in place.
  void processPersistentReplVariables(LIT::FuncOp func,
                                      LIT::StructDeclOp stateStruct);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  struct Impl;

  std::unique_ptr<Impl> impl;
};
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONPARSER_H
