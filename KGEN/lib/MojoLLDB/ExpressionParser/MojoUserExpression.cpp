//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoUserExpression.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "Logging.h"
#include "MojoExpressionParser.h"
#include "MojoExpressionVariable.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Signals.h"

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

  void RegisterPersistentState(PersistentExpressionState &persistentStateArg) {
    persistentState = &persistentStateArg;
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

static MojoTypeSystem &getMojoTypeSystem(Target &target) {
  if (auto typeSystemOr =
          target.GetScratchTypeSystemForLanguage(eLanguageTypeMojo))
    return *llvm::cast<MojoTypeSystem>(typeSystemOr.get().get());
  llvm::report_fatal_error(
      "The Mojo type system plug-in must have already been registered.");
}

static MojoPersistentExpressionState &
getMojoPersistentState(MojoTypeSystem &typeSystem) {
  return *llvm::cast<MojoPersistentExpressionState>(
      (typeSystem.GetPersistentExpressionState()));
}

struct MojoUserExpression::Impl {
  Impl(ExecutionContextScope &exeScope, Target &target)
      : target(target), typeSystemHelper(target),
        resultDelegate(target.shared_from_this()),
        typeSystem(getMojoTypeSystem(target)),
        persistentState(getMojoPersistentState(typeSystem)) {}

  Target &target;
  /// The type system helper.
  MojoUserExpressionHelper typeSystemHelper;

  /// The various expression delegates.
  ResultDelegate resultDelegate;
  PersistentVariableDelegate persistentVariableDelegate;

  /// The underlying expression parser.
  std::unique_ptr<MojoExpressionParser> parser;
  MojoTypeSystem &typeSystem;
  MojoPersistentExpressionState &persistentState;
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
    : JitUserExpression(exeScope, expr, prefix, language, desiredType, options),
      impl(std::make_unique<Impl>(exeScope, *m_target_wp.lock())) {}

MojoUserExpression::~MojoUserExpression() = default;
char MojoUserExpression::ID;

//===----------------------------------------------------------------------===//
// Expression parsing and execution
//===----------------------------------------------------------------------===//

bool MojoUserExpression::Parse(DiagnosticManager &diagnosticManager,
                               ExecutionContext &exeCtx,
                               ExecutionPolicy executionPolicy,
                               bool keepResultInMemory,
                               bool generateDebugInfo) {
  // Setup the execution context.
  InstallContext(exeCtx);

  // Initialize the persistent state.
  impl->resultDelegate.RegisterPersistentState(impl->persistentState);

  // Parse the expression text.
  Process *process = exeCtx.GetProcessPtr();
  auto *exeScope = process ? (ExecutionContextScope *)process : &impl->target;
  if (failed(wrapTextAndParseExpression(diagnosticManager, exeCtx, exeScope,
                                        impl->persistentState)))
    return false;

  // Prepare the output of the parser for execution, evaluating it statically if
  // possible.
  Status jitError = impl->parser->prepareForExecution(
      m_jit_start_addr, m_jit_end_addr, executionUnit, exeCtx, executionPolicy);

  // If a valid execution unit was produced and there is more than one external
  // function in the execution unit, it needs to keep living even if it's not
  // top level, because the result could refer to that function., register it if
  // necessary.
  if (executionUnit &&
      (m_options.GetExecutionPolicy() == eExecutionPolicyTopLevel ||
       executionUnit->getJittedFunctions().size() > 1)) {
    impl->persistentState.registerExecutionUnit(executionUnit);
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

bool MojoUserExpression::addArguments(ExecutionContext &exeCtx,
                                      std::vector<lldb::addr_t> &args,
                                      lldb::addr_t structAddress,
                                      DiagnosticManager &diagnosticManager) {
  args.push_back(structAddress);
  return true;
}

/// Signal handler that will dump the stack trace to the log.
static void dumpTraceOnSignal(void *) {
  std::string traceStr;
  llvm::raw_string_ostream trace(traceStr);
  llvm::sys::PrintStackTrace(trace);
  MOJO_EXPR_LOG("Backtrace:\n{0}", traceStr);
}

/// Register the trace dumping signal handler exactly once.
static void registerTraceDumpHandler() {
  static llvm::once_flag flag;
  llvm::call_once(
      flag, []() { llvm::sys::AddSignalHandler(dumpTraceOnSignal, nullptr); });
}

/// Collect the name and type of the current persistent variables within the
/// given state.
static void collectPersistentVariables(
    MojoPersistentExpressionState &state,
    SmallVectorImpl<std::pair<StringRef, mlir::Type>> &variables) {
  for (unsigned i = 0, e = state.GetSize(); i < e; ++i) {
    lldb::ExpressionVariableSP var = state.GetVariableAtIndex(i);
    assert(var && "expected valid variable in persistent state");

    mlir::Type varType = mlir::Type::getFromOpaquePointer(
        var->GetCompilerType().GetOpaqueQualType());
    variables.emplace_back(var->GetName().GetStringRef(), varType);
  }
}

void MojoUserExpression::notifyFixits(DiagnosticManager &diagnosticManager,
                                      StringRef fixedText) {
  std::string fixItsNotification;
  {
    llvm::raw_string_ostream os(fixItsNotification);
    os << diagnosticManager.GetString()
       << "Applied fix-its, evaluating new expression:\n"
       << fixedText << "\n";
  }

  impl->target.GetDebugger().GetAsyncErrorStream()->AsRawOstream()
      << fixItsNotification;

  // Besides writing the notification to stderr, we also notify it to the type
  // system broadcaster for external listeners.
  impl->typeSystem.broadcastUserMessage(fixItsNotification);
}

LogicalResult MojoUserExpression::wrapTextAndParseExpression(
    DiagnosticManager &diagnosticManager, ExecutionContext &exeCtx,
    ExecutionContextScope *exeScope, MojoPersistentExpressionState &state) {
  // Collect the current persistent variables.
  SmallVector<std::pair<StringRef, mlir::Type>> variables;
  collectPersistentVariables(state, variables);

  // Wrap the expression text in a function so that we can execute it.
  // TODO: This currently doesn't support imports or any kind of persistent
  // state.
  llvm::raw_string_ostream exprRawOS(transformedText);
  mlir::raw_indented_ostream exprOSIndented(exprRawOS);
  exprOSIndented << "from IO import print\n"
                 << "from Pointer import Pointer\n\n"
                 << "from PythonInterface import PythonInterface\n";

  // Build the input struct, which contains each of the persistent variables.
  exprOSIndented << "struct __lldb_context__:\n";
  for (auto &var : variables) {
    exprOSIndented << llvm::formatv(
        "  var {0}: Pointer[Pointer[__mlir_type.`{1}`]]\n", var.first,
        var.second);
  }
  if (variables.empty())
    exprOSIndented << "  pass\n";
  exprOSIndented << "\n";

  // Generate a wrapper function to handle the extracting function arguments.
  exprOSIndented << "@export\n"
                    "fn __lldb_expr__(__lldb_arg&: __lldb_context__):\n"
                    "  __lldb_expr_impl__(__lldb_arg";
  for (auto &var : variables) {
    exprOSIndented << formatv(
        ", __get_address_as_lvalue(__lldb_arg.{0}.load().address)", var.first);
  }
  exprOSIndented << ")\n\n";

  // Finally we can generate the actual expression function.
  exprOSIndented << "fn __lldb_expr_impl__(__lldb_arg&: __lldb_context__";
  for (auto &var : variables) {
    exprOSIndented << llvm::formatv(", {0}&: __mlir_type.`{1}`", var.first,
                                    var.second);
  }
  exprOSIndented << "):\n";

  size_t prefixSize = transformedText.size();
  exprOSIndented.printReindented(m_expr_text, "    ");

  MOJO_EXPR_LOG("Parsing the following code:\n{0}", transformedText.c_str());

  // Parse the expression.
  materializer = std::make_unique<Materializer>();
  impl->parser =
      std::make_unique<MojoExpressionParser>(exeScope, *this, m_options);

  LogicalResult result = failure();
  auto parseModule = [&]() {
    result = impl->parser->parse(diagnosticManager);
    if (succeeded(result))
      return;

    if (!diagnosticManager.HasFixIts())
      return;

    MOJO_EXPR_LOG("Attempting to rewrite the input expression");

    // If we can rewrite the expression, do so. If not, simply return.
    if (failed(impl->parser->rewriteExpression(diagnosticManager)))
      return;
    MOJO_EXPR_LOG("Rewrote the input, next parse will be the fixed code");
    llvm::raw_string_ostream fixedOS(m_fixed_text);
    mlir::raw_indented_ostream indentedFixedOS(fixedOS);

    // Drop the prefix and remove all indent.
    indentedFixedOS.printReindented(
        diagnosticManager.GetFixedExpression().substr(prefixSize), "");

    notifyFixits(diagnosticManager, m_fixed_text);
    // Clear the diagnostic manager so we don't re-fix something.
    diagnosticManager.Clear();
  };

  // Register the trace dump signal handler before we enable the
  // CrashRecoveryContext so it is picked up properly.
  registerTraceDumpHandler();
  llvm::CrashRecoveryContext::Enable();
  llvm::CrashRecoveryContext crc;
  // Signal handlers don't fire unless this flag is set.
  crc.DumpStackAndCleanupOnFailure = true;
  if (!crc.RunSafelyOnThread(parseModule)) {
    MOJO_EXPR_LOG("Crash recovered: CrashRecoveryContext::RetCode (on POSIX: "
                  "signal number + 128) = {0}",
                  crc.RetCode);
    diagnosticManager.PutString(eDiagnosticSeverityError, "crash detected");
    return failure();
  }

  return result;
}
