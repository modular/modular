//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONPARSER_H
#define KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONPARSER_H

#include "JITExecutionUnit.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN::LIT {
class FuncOp;
class StructDeclOp;
} // namespace M::KGEN::LIT

namespace M::KGEN::Mojo {
class MojoPersistentExpressionState;
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

  /// Parse a single expression and convert it to IR.
  LogicalResult parse(MojoPersistentExpressionState &state,
                      lldb_private::DiagnosticManager &diagnosticManager);

  /// Ready an already-parsed expression for execution, possibly evaluating it
  /// statically.
  lldb_private::Status
  prepareForExecution(lldb::addr_t &funcAddr, lldb::addr_t &funcEnd,
                      std::shared_ptr<JITExecutionUnit> &executionUnit,
                      lldb_private::ExecutionContext &exeCtx,
                      lldb_private::ExecutionPolicy executionPolicy,
                      bool keepResultInMemory);

private:
  struct Impl;

  std::unique_ptr<Impl> impl;
};
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONPARSER_H
