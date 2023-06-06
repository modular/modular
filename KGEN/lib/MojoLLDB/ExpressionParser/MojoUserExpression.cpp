//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoUserExpression.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "Logging.h"
#include "MojoDiagnostic.h"
#include "MojoExpressionParser.h"
#include "MojoExpressionVariable.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Sequence.h"
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
class MojoUserExpressionHelper
    : public llvm::RTTIExtends<MojoUserExpressionHelper,
                               ExpressionTypeSystemHelper> {
public:
  // LLVM RTTI support
  static char ID;

  MojoUserExpressionHelper(Target &) {}
};

char MojoUserExpressionHelper::ID;

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
          target.GetScratchTypeSystemForLanguage(lldb::eLanguageTypeMojo))
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

  /// The target associated with this expression.
  Target &target;

  /// The type system helper.
  MojoUserExpressionHelper typeSystemHelper;

  /// The various expression delegates.
  ResultDelegate resultDelegate;
  PersistentVariableDelegate persistentVariableDelegate;

  /// The name of the python module that wraps the expression, if the expression
  /// is a Python expression, nullopt otherwise.
  std::optional<std::string> pythonModuleName;

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

  // On exit, log all of the diagnostics that were collected.
  auto broadcastDiagnostics = llvm::make_scope_exit([&] {
    impl->typeSystem.broadcastDiagnostics(diagnosticManager);
    diagnosticManager.Clear();
  });

  // If the expression starts with `%%python`, the user wants to treat this as a
  // python expression. Otherwise, it should be treated as a Mojo expression.
  StringRef exprText(m_expr_text);
  if (!exprText.consume_front("%%python\n")) {
    if (failed(wrapTextAndParseExpression(diagnosticManager, exeCtx, exeScope,
                                          impl->persistentState)))
      return false;
  } else if (failed(wrapTextAndParsePythonExpression(
                 exprText, diagnosticManager, exeCtx, exeScope,
                 impl->persistentState))) {
    return false;
  }

  // Prepare the output of the parser for execution, evaluating it statically if
  // possible.
  Status jitError = impl->parser->prepareForExecution(
      m_jit_start_addr, m_jit_end_addr, executionUnit, exeCtx, executionPolicy,
      keepResultInMemory);
  if (!jitError.Success()) {
    m_jit_start_addr = m_jit_end_addr = LLDB_INVALID_ADDRESS;

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

/// Signal handler that will dump the stack trace to the log. If we can't pull
/// out the mojo type system, we simply return because the only purpose of this
/// handler is to print a stack trace to the mojo log.
static void dumpTraceOnSignal(void *cookie) {
  auto *debugger = (Debugger *)cookie;

  // Pull the type system out of the current target.
  lldb::TargetSP currentTarget = debugger->GetSelectedTarget();
  if (!currentTarget)
    return;
  auto typeSystemOr =
      currentTarget->GetScratchTypeSystemForLanguage(lldb::eLanguageTypeMojo);
  if (!typeSystemOr)
    return;

  // Make sure it's the Mojo type system, otherwise we can't necessarily
  // broadcast on its channel.
  std::shared_ptr<MojoTypeSystem> typeSystem =
      dyn_cast<MojoTypeSystem>(*typeSystemOr);
  if (!typeSystem)
    return;

  // Great - now we can broadcast to it.
  std::string traceStr;
  llvm::raw_string_ostream trace(traceStr);
  llvm::sys::PrintStackTrace(trace);
  // This will also flush the debug logs.
  typeSystem->crashLog("Backtrace:\n{0}", traceStr);
}

/// Register the trace dumping signal handler exactly once.
static void registerTraceDumpHandler(Debugger &debugger) {
  static llvm::once_flag flag;
  llvm::call_once(flag, [&]() {
    llvm::sys::AddSignalHandler(dumpTraceOnSignal, (void *)&debugger);
  });
}

LogicalResult MojoUserExpression::wrapTextAndParseExpression(
    DiagnosticManager &diagnosticManager, ExecutionContext &exeCtx,
    ExecutionContextScope *exeScope, MojoPersistentExpressionState &state) {
  // Parse the expression.
  materializer = std::make_unique<Materializer>();
  impl->parser =
      std::make_unique<MojoExpressionParser>(exeScope, *this, m_options);

  // Register the trace dump signal handler before we enable the
  // CrashRecoveryContext so it is picked up properly.
  registerTraceDumpHandler(exeCtx.GetTargetRef().GetDebugger());
  llvm::CrashRecoveryContext::Enable();

  // Disable the crash recovery context for the next time around.
  auto scopeExit =
      llvm::make_scope_exit([]() { llvm::CrashRecoveryContext::Disable(); });
  llvm::CrashRecoveryContext crc;

  // Signal handlers don't fire unless this flag is set.
  crc.DumpStackAndCleanupOnFailure = true;

  LogicalResult result = failure();
  if (!crc.RunSafelyOnThread(
          [&]() { result = impl->parser->parse(state, diagnosticManager); })) {
    impl->typeSystem.errorLog(
        "Crash recovered: CrashRecoveryContext::RetCode (on POSIX: "
        "signal number + 128) = {0}",
        crc.RetCode);
    diagnosticManager.PutString(
        eDiagnosticSeverityError,
        "The Mojo REPL has crashed and attempted recovery. If the REPL "
        "behaves inconsistently, please restart to ensure correct behavior.");
    return failure();
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Python expression parsing and execution

const std::optional<std::string> &MojoUserExpression::getPythonModuleName() {
  return impl->pythonModuleName;
}

static std::string createSymbolExtractorPythonCode(StringRef pythonExpr) {
  const char *rawSymbolsExtractor = R"(
def __lldb_python_extract_symbols():
  symbols = []

  import ast

  # The following class visits only top level constructs of the given python
  # expression. It doesn't traverse recursively the AST.
  class AssignmentVisitor(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
      symbols.append(['declaration', node.name])

    def visit_Assign(self, node):
      for target in node.targets:
        if isinstance(target, ast.Name):
          symbols.append(['declaration', target.id])

    def visit_Import(self, node):
      for alias in node.names:
        asname = alias.asname if alias.asname else alias.name
        symbols.append(['import', asname, alias.name])

    # We remove the default implementation of the following node visitors to
    # prevent finding assignments recursively.
    def visit_AsyncFunctionDef(self, node):
      pass

    def visit_ClassDef(self, node):
      pass


  __lldb_python_ast = ast.parse("{0}")
  AssignmentVisitor().visit(__lldb_python_ast)

  return symbols


# To simplify the parsing logic, we serialize the symbols as a multi-line
# string, where each line corresponds to a symbol, and each line is made of
# space-delimited tokens with the following format:
#
#   <symbol kind> <symbol name> [other tokens depending on the kind...]
def __lldb_python_serialize_symbols(symbols):
  serialized = []
  for symbol in symbols:
    serialized.append(' '.join(symbol))
  return '\n'.join(serialized)


__lldb_python_symbols = __lldb_python_serialize_symbols(__lldb_python_extract_symbols())
  )";

  std::string escapedPythonExpr;
  llvm::raw_string_ostream escapedPythonExprOS(escapedPythonExpr);
  escapedPythonExprOS.write_escaped(pythonExpr);

  return llvm::formatv(rawSymbolsExtractor, escapedPythonExpr).str();
}

/// Import the various top-level python symbols defined in the given python
/// expression into the current mojo context by emitting binding code to the
/// given stream.
static LogicalResult
importPythonSymbolsIntoMojo(Debugger &debugger, StringRef pythonExpr,
                            StringRef moduleName, raw_ostream &mojoExprOS,
                            DiagnosticManager &diagnosticManager) {
  // We extract the necessary python symbols using the python ast module, which
  // requires us to use the LLDB python interpreter.
  ScriptInterpreter *scriptInterpreter = debugger.GetScriptInterpreter(
      /*can_create=*/true, lldb::eScriptLanguagePython);
  if (!scriptInterpreter) {
    diagnosticManager.PutString(lldb_private::eDiagnosticSeverityWarning,
                                "persisting Python symbols requires LLDB to be "
                                "built with Python scripting support.");
    // We don't fail hard here because we can't expect all LLDB distributions to
    // have python integration.
    return success();
  }

  ExecuteScriptOptions excOptions =
      ExecuteScriptOptions().SetEnableIO(true).SetSetLLDBGlobals(false);

  // We first execute the code that extracts the symbols we need. Accessing the
  // result is done later.
  scriptInterpreter->ExecuteMultipleLines(
      createSymbolExtractorPythonCode(pythonExpr).c_str(), excOptions);

  // Here we access the result, which is a serialized description of each symbol
  // to extract.
  char *symbolsStr = nullptr;
  if (!scriptInterpreter->ExecuteOneLineWithReturn(
          "__lldb_python_symbols",
          ScriptInterpreter::eScriptReturnTypeCharStrOrNone, &symbolsStr,
          excOptions)) {
    diagnosticManager.PutString(lldb_private::eDiagnosticSeverityError,
                                "Unable to extract Python symbols into Mojo.");
    return failure();
  }

  StringRef symbols(symbolsStr);

  SmallVector<StringRef> symbolLines;
  symbols.split(symbolLines, '\n');

  // We process the symbols in reverse order so that we honor the last occurence
  // of a given symbol name.
  llvm::StringSet<> seenVariables;
  for (StringRef symbolLine : llvm::reverse(symbolLines)) {
    SmallVector<StringRef> items;
    symbolLine.split(items, ' ');

    StringRef kind = items[0];
    StringRef name = items[1];
    if (seenVariables.contains(name))
      continue;
    seenVariables.insert(name);

    if (kind == "declaration") {
      mojoExprOS << llvm::formatv("let {0} = {1}.{0}\n", name, moduleName);
    } else if (kind == "import") {
      StringRef module = items[2];
      // Private import aliases (starting with a leading underscore) should not
      // be exposed to mojo.
      if (!name.starts_with("_"))
        mojoExprOS << llvm::formatv("let {0} = Python.import_module(\"{1}\")\n",
                                    name, module);
    }
  }

  return success();
}

LogicalResult MojoUserExpression::wrapTextAndParsePythonExpression(
    StringRef pythonExpr, lldb_private::DiagnosticManager &diagnosticManager,
    lldb_private::ExecutionContext &exeCtx,
    lldb_private::ExecutionContextScope *exeScope,
    MojoPersistentExpressionState &state) {
  impl->typeSystem.debugLog("Parsing the following python code:\n{0}",
                            pythonExpr.data());

  // Generate a wrapper python expression that builds a new module from the
  // given source expression string.
  //   {0}: The escaped source expression string.
  //   {1}: The name of the module to create.
  const char *pythonWrapperExpr = R"(
import sys, types

code_string = "{0}"
expr_module = types.ModuleType('{1}')
exec(code_string, expr_module.__dict__)
sys.modules['{1}'] = expr_module
  )";

  // Generate an escaped version of the python expression to import, also taking
  // this time to add implicit imports for any previously defined modules.
  std::string escapedPythonExpr;
  llvm::raw_string_ostream escapedPythonExprOS(escapedPythonExpr);
  for (const auto &exprInst : state.getExpressionInstances()) {
    if (exprInst.pythonModuleName) {
      escapedPythonExprOS.write_escaped("try:\n");
      escapedPythonExprOS.write_escaped(
          llvm::formatv("  from {0} import *\n", *exprInst.pythonModuleName)
              .str());
      escapedPythonExprOS.write_escaped("except:\n  pass\n");
    }
  }
  escapedPythonExprOS.write_escaped(pythonExpr);

  std::string moduleName = state.getNextPythonExpressionModuleName();
  std::string wrappedPythonExpr =
      llvm::formatv(pythonWrapperExpr, escapedPythonExpr, moduleName).str();
  impl->typeSystem.debugLog("Wrapped python code:\n{0}",
                            wrappedPythonExpr.data());

  // Build the Mojo expression we'll use to process the python. This consists of
  // the wrapped python expression, and implicit imports for any of the
  // top-level entities we want extract from the python expression.
  std::string mojoExpr;
  llvm::raw_string_ostream mojoExprOS(mojoExpr);

  // Evaluate the wrapped python expression.
  mojoExprOS << "var __lldb_repl_python__ = Python()\n\n";
  mojoExprOS << "if not __lldb_repl_python__.eval(\"";
  mojoExprOS.write_escaped(wrappedPythonExpr);
  mojoExprOS
      << "\"):\n  raise Error('The Python expression raised an exception')\n";

  // If persistent results are enabled, we also import top-level symbols from
  // the python module into the mojo context.
  if (!m_options.GetSuppressPersistentResult()) {
    mojoExprOS << llvm::formatv("let {0} = Python.import_module(\"{0}\")\n\n",
                                moduleName);

    // Import the interesting top-level symbols from the python module into the
    // mojo context.
    if (failed(importPythonSymbolsIntoMojo(exeCtx.GetTargetRef().GetDebugger(),
                                           pythonExpr, moduleName, mojoExprOS,
                                           diagnosticManager)))
      return failure();
  }

  // Now that we've got a Mojo expression, parse it the way we would any other
  // expression.
  m_expr_text = mojoExprOS.str();
  impl->pythonModuleName = std::move(moduleName);

  return wrapTextAndParseExpression(diagnosticManager, exeCtx, exeScope, state);
}
