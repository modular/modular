//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOUSEREXPRESSION_H
#define KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOUSEREXPRESSION_H

#include "Support/LLVMForwardDecls.h"
#include "lldb/Expression/LLVMUserExpression.h"

namespace M::KGEN::Mojo {
class MojoTypeSystem;

//===----------------------------------------------------------------------===//
// MojoUserExpression
//===----------------------------------------------------------------------===//

/// MojoUserExpression encapsulates the objects needed to parse and interpret or
/// JIT an expression.
class MojoUserExpression : public lldb_private::LLVMUserExpression {
  static char ID;

public:
  MojoUserExpression(lldb_private::ExecutionContextScope &exeScope,
                     llvm::StringRef expr, llvm::StringRef prefix,
                     lldb::LanguageType language, ResultType desiredType,
                     const lldb_private::EvaluateExpressionOptions &options);
  ~MojoUserExpression() override;

  //===--------------------------------------------------------------------===//
  // Expression parsing and execution
  //===--------------------------------------------------------------------===//

  /// Return the function name that should be used for executing the expression.
  /// Text() should contain the definition of this function.
  const char *FunctionName() override { return "__lldb_expr__"; }

  /// Parse the expression.
  bool Parse(lldb_private::DiagnosticManager &diagnosticManager,
             lldb_private::ExecutionContext &exeCtx,
             lldb_private::ExecutionPolicy executionPolicy,
             bool keepResultInMemory, bool generateDebugInfo) override;

  /// Return the type system helper for this expression.
  lldb_private::ExpressionTypeSystemHelper *GetTypeSystemHelper() override;

  /// Return the result variable for this expression after dematerialization.
  lldb::ExpressionVariableSP GetResultAfterDematerialization(
      lldb_private::ExecutionContextScope *exeScope) override;

  //===--------------------------------------------------------------------===//
  // RTTI support
  //===--------------------------------------------------------------------===//

  bool isA(const void *classID) const override {
    return classID == &ID || lldb_private::LLVMUserExpression::isA(classID);
  }
  static bool classof(const Expression *obj) { return obj->isA(&ID); }

private:
  //===--------------------------------------------------------------------===//
  // Expression parsing and execution
  //===--------------------------------------------------------------------===//

  void ScanContext(lldb_private::ExecutionContext &exeCtx,
                   lldb_private::Status &err) override;

  /// Add the function arguments used when invoking the wrapper function for the
  /// generated expression.
  bool
  AddArguments(lldb_private::ExecutionContext &exeCtx,
               std::vector<lldb::addr_t> &args, lldb::addr_t structAddress,
               lldb_private::DiagnosticManager &diagnosticManager) override;

  /// Process and wrap the expression text, and then parse it.
  LogicalResult
  wrapTextAndParseExpression(lldb_private::DiagnosticManager &diagnosticManager,
                             lldb_private::ExecutionContext &exeCtx,
                             lldb_private::ExecutionContextScope *exeScope);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  struct Impl;

  std::unique_ptr<Impl> impl;
};

} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOUSEREXPRESSION_H
