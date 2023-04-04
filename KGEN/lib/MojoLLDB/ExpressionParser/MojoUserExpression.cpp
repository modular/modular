//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoUserExpression.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "MojoExpressionParser.h"
#include "MojoExpressionVariable.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "mlir/Support/IndentedOstream.h"

using namespace M;
using namespace M::KGEN::Mojo;
using namespace lldb_private;

namespace {
//===----------------------------------------------------------------------===//
// MojoUserExpressionHelper
//===----------------------------------------------------------------------===//

/// An expression helper for Mojo expressions.
class MojoUserExpressionHelper : public ExpressionTypeSystemHelper {
public:
  MojoUserExpressionHelper(Target &)
      : ExpressionTypeSystemHelper(eKindGoHelper) {}
  ~MojoUserExpressionHelper() = default;
};

//===----------------------------------------------------------------------===//
// ResultDelegate
//===----------------------------------------------------------------------===//

/// This class implements a variable delegate for the result of an expression.
class ResultDelegate : public Materializer::PersistentVariableDelegate {
public:
  ResultDelegate(lldb::TargetSP target) : target(std::move(target)) {}

  ConstString GetName() override {
    return persistentState->GetNextPersistentVariableName();
  }

  void DidDematerialize(lldb::ExpressionVariableSP &varArg) override {
    variable = varArg;
  }

  void RegisterPersistentState(PersistentExpressionState *persistentStateArg) {
    persistentState = persistentStateArg;
  }

  lldb::ExpressionVariableSP &GetVariable() { return variable; }

private:
  lldb::TargetSP target;
  PersistentExpressionState *persistentState;
  lldb::ExpressionVariableSP variable;
};

//===----------------------------------------------------------------------===//
// PersistentVariableDelegate
//===----------------------------------------------------------------------===//

/// This class implements a variable delegate for persistent variables.
class PersistentVariableDelegate
    : public Materializer::PersistentVariableDelegate {
public:
  PersistentVariableDelegate() = default;
  ConstString GetName() override { return ConstString(); }
  void DidDematerialize(lldb::ExpressionVariableSP &variable) override {}
};
} // namespace

//===----------------------------------------------------------------------===//
// MojoUserExpression::Impl
//===----------------------------------------------------------------------===//

struct MojoUserExpression::Impl {
  Impl(ExecutionContextScope &exeScope, Target &target)
      : typeSystemHelper(target), resultDelegate(target.shared_from_this()),
        persistentVariableDelegate() {}

  /// The type system helper.
  MojoUserExpressionHelper typeSystemHelper;

  /// The various expression delegates.
  ResultDelegate resultDelegate;
  PersistentVariableDelegate persistentVariableDelegate;

  /// The underlying expression parser.
  std::unique_ptr<MojoExpressionParser> parser;
};

//===----------------------------------------------------------------------===//
// MojoUserExpression
//===----------------------------------------------------------------------===//

MojoUserExpression::MojoUserExpression(ExecutionContextScope &exeScope,
                                       llvm::StringRef expr,
                                       llvm::StringRef prefix,
                                       lldb::LanguageType language,
                                       ResultType desiredType,
                                       const EvaluateExpressionOptions &options)
    : LLVMUserExpression(exeScope, expr, prefix, language, desiredType,
                         options),
      impl(std::make_unique<Impl>(exeScope, *m_target_wp.lock())) {}

MojoUserExpression::~MojoUserExpression() = default;
char MojoUserExpression::ID;

//===----------------------------------------------------------------------===//
// Expression parsing and execution
//===----------------------------------------------------------------------===//

/// Return the persistent Mojo expression state for the given target.
static MojoPersistentExpressionState *getPersistentState(Target *target) {
  return llvm::cast<MojoPersistentExpressionState>(
      target->GetPersistentExpressionStateForLanguage(eLanguageTypeMojo));
}

bool MojoUserExpression::Parse(DiagnosticManager &diagnosticManager,
                               ExecutionContext &exeCtx,
                               ExecutionPolicy executionPolicy,
                               bool keepResultInMemory,
                               bool generateDebugInfo) {
  // Setup the execution context.
  InstallContext(exeCtx);

  // Extract a target from the execution context.
  Target *target = exeCtx.GetTargetPtr();
  if (!target) {
    diagnosticManager.PutString(eDiagnosticSeverityError,
                                "couldn't start parsing (no target)");
    return false;
  }

  // Initialize the persistent state.
  auto *persistentState = getPersistentState(target);
  if (!persistentState) {
    diagnosticManager.PutString(eDiagnosticSeverityError,
                                "couldn't start parsing (no persistent data)");
    return false;
  }
  impl->resultDelegate.RegisterPersistentState(persistentState);

  // Scan the current execution context.
  Status error;
  ScanContext(exeCtx, error);
  if (!error.Success()) {
    diagnosticManager.Printf(eDiagnosticSeverityError, "warning: %s\n",
                             error.AsCString());
  }

  // Parse the expression text.
  Process *process = exeCtx.GetProcessPtr();
  auto *exeScope = process ? (ExecutionContextScope *)process : target;
  if (failed(wrapTextAndParseExpression(diagnosticManager, exeCtx, exeScope)))
    return false;

  // Prepare the output of the parser for execution, evaluating it statically if
  // possible.
  Status jitError = impl->parser->PrepareForExecution(
      m_jit_start_addr, m_jit_end_addr, m_execution_unit_sp, exeCtx,
      m_can_interpret, executionPolicy);

  // If a valid execution unit was produced and there is more than one external
  // function in the execution unit, it needs to keep living even if it's not
  // top level, because the result could refer to that function., register it if
  // necessary.
  if (m_execution_unit_sp &&
      (m_options.GetExecutionPolicy() == eExecutionPolicyTopLevel ||
       m_execution_unit_sp->GetJittedFunctions().size() > 1)) {
    persistentState->RegisterExecutionUnit(m_execution_unit_sp);
  }

  // Process any errors during code generation.
  if (!jitError.Success()) {
    const char *errorCStr = jitError.AsCString();
    if (errorCStr && errorCStr[0])
      diagnosticManager.PutString(eDiagnosticSeverityError, errorCStr);
    else
      diagnosticManager.PutString(eDiagnosticSeverityError,
                                  "expression can't be interpreted or run\n");
    return false;
  }

  if (process && m_jit_start_addr != LLDB_INVALID_ADDRESS)
    m_jit_process_wp = lldb::ProcessWP(process->shared_from_this());
  return true;
}

ExpressionTypeSystemHelper *MojoUserExpression::GetTypeSystemHelper() {
  return &impl->typeSystemHelper;
}

lldb::ExpressionVariableSP MojoUserExpression::GetResultAfterDematerialization(
    ExecutionContextScope *exeScope) {
  return impl->resultDelegate.GetVariable();
}

void MojoUserExpression::ScanContext(ExecutionContext &exeCtx, Status &err) {
  m_target = exeCtx.GetTargetPtr();
}

bool MojoUserExpression::AddArguments(ExecutionContext &exeCtx,
                                      std::vector<lldb::addr_t> &args,
                                      lldb::addr_t structAddress,
                                      DiagnosticManager &diagnosticManager) {
  args.push_back(structAddress);
  return true;
}

LogicalResult MojoUserExpression::wrapTextAndParseExpression(
    DiagnosticManager &diagnosticManager, ExecutionContext &exeCtx,
    ExecutionContextScope *exeScope) {
  Log *log = GetLog(LLDBLog::Expressions);

  // Wrap the expression text in a function so that we can execute it.
  // TODO: This currently doesn't support imports or any kind of persistent
  // state.
  llvm::raw_string_ostream exprRawOS(m_transformed_text);
  mlir::raw_indented_ostream exprOSIndented(exprRawOS);
  exprOSIndented << "from Pointer import Pointer\n"
                 << "@export\nfn __lldb_expr__(__lldb_arg: "
                    "Pointer[__mlir_type.`!pop.scalar<invalid>`]):\n";
  exprOSIndented.printReindented(m_expr_text, "  ");

  LLDB_LOG(log, "Parsing the following code:\n{0}", m_transformed_text.c_str());

  // Parse the expression.
  m_materializer_up = std::make_unique<Materializer>();
  impl->parser =
      std::make_unique<MojoExpressionParser>(exeScope, *this, m_options);
  return impl->parser->parse();
}
