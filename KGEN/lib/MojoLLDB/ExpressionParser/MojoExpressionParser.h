//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONPARSER_H
#define KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONPARSER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "lldb/Expression/ExpressionParser.h"

namespace M::KGEN::Mojo {
class MojoExpressionParser : public lldb_private::ExpressionParser {
public:
  MojoExpressionParser(lldb_private::ExecutionContextScope *exeScope,
                       lldb_private::Expression &expr,
                       const lldb_private::EvaluateExpressionOptions &options);
  ~MojoExpressionParser() override;

  /// Attempt to find possible command line completions for the given
  /// expression.
  bool Complete(lldb_private::CompletionRequest &request, unsigned line,
                unsigned pos, unsigned typedPos) override {
    return false;
  }

  /// Rewrite the expression using the fix-its contained in the diagnostic
  /// manager. Returns true if any edits occurred, false if not.
  bool RewriteExpression(
      lldb_private::DiagnosticManager &diagnosticManager) override;

  /// Parse a single expression and convert it to IR.
  LogicalResult parse(lldb_private::DiagnosticManager &diagnosticManager);

  /// Ready an already-parsed expression for execution, possibly evaluating it
  /// statically.
  lldb_private::Status
  PrepareForExecution(lldb::addr_t &funcAddr, lldb::addr_t &funcEnd,
                      lldb::IRExecutionUnitSP &executionUnit,
                      lldb_private::ExecutionContext &exeCtx,
                      bool &canInterpret,
                      lldb_private::ExecutionPolicy executionPolicy) override;

private:
  struct Impl;

  std::unique_ptr<Impl> impl;
};
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONPARSER_H
