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
#include "llvm/ADT/ScopeExit.h"
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

/// Handle a 'special expression' - which starts with `!`. Returns an error if
/// we don't recognize the expression. On success, sets the text to an empty
/// string so we just don't execute anything.
static bool handleSpecialExpr(std::string &text,
                              DiagnosticManager &diagnosticManager,
                              MojoTypeSystem &typeSystem) {
  // Use rtrim to remove newlines/whitespace at the end. We want the exact
  // equality check here so we don't match on something like `!dump_logs_foo`.
  if (StringRef(text).rtrim() == "!dump_logs") {
    typeSystem.flushIRDumpAndDebugLog();
    text = "";
    return true;
  }

  diagnosticManager.Printf(eDiagnosticSeverityError,
                           "Unknown special expression: %s", text.c_str());
  return false;
}

bool MojoUserExpression::Parse(DiagnosticManager &diagnosticManager,
                               ExecutionContext &exeCtx,
                               ExecutionPolicy executionPolicy,
                               bool keepResultInMemory,
                               bool generateDebugInfo) {
  // Check to see if it's a special expression.
  if (StringRef(m_expr_text).starts_with("!"))
    if (!handleSpecialExpr(m_expr_text, diagnosticManager, impl->typeSystem))
      return false;

  // Setup the execution context.
  InstallContext(exeCtx);

  // Initialize the persistent state.
  impl->resultDelegate.RegisterPersistentState(impl->persistentState);

  // Parse the expression text.
  Process *process = exeCtx.GetProcessPtr();
  auto *exeScope = process ? (ExecutionContextScope *)process : &impl->target;

  // If the expression starts with `>python`, the user wants to treat this as a
  // python expression. Otherwise, it should be treated as a Mojo expression.
  StringRef exprText(m_expr_text);
  std::optional<MojoExpressionSourceCode> mojoSourceCode;
  if (!exprText.consume_front(">python\n")) {
    mojoSourceCode.emplace(exprText);
    if (failed(wrapTextAndParseExpression(*mojoSourceCode, diagnosticManager,
                                          exeCtx, exeScope,
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
      std::move(mojoSourceCode), keepResultInMemory);
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
      currentTarget->GetScratchTypeSystemForLanguage(eLanguageTypeMojo);
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
  typeSystem->errorLog("Backtrace:\n{0}", traceStr);
}

/// Register the trace dumping signal handler exactly once.
static void registerTraceDumpHandler(Debugger &debugger) {
  static llvm::once_flag flag;
  llvm::call_once(flag, [&]() {
    llvm::sys::AddSignalHandler(dumpTraceOnSignal, (void *)&debugger);
  });
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
  llvm::raw_string_ostream os(fixItsNotification);
  os << diagnosticManager.GetString()
     << "Applied fix-its, evaluating new expression:\n"
     << fixedText;

  // Besides writing the notification to stderr, we also notify it to the type
  // system broadcaster for external listeners.
  impl->typeSystem.broadcastUserMessage(fixItsNotification);
}

LogicalResult MojoUserExpression::wrapTextAndParseExpression(
    const MojoExpressionSourceCode &sourceCode,
    DiagnosticManager &diagnosticManager, ExecutionContext &exeCtx,
    ExecutionContextScope *exeScope, MojoPersistentExpressionState &state) {

  // We use the following comments to identify the chunks of code written by the
  // user. After we apply fix-its, we need to extract these two chunks and
  // combine them into the new source code to evaluate.
  constexpr StringLiteral kTopLevelBlockBegin =
      "#==__lldb_expr_top_level_code_begin\n";
  constexpr StringLiteral kTopLevelBlockEnd =
      "#==__lldb_expr_top_level_code_end\n";
  constexpr StringLiteral kMainBodyBlockBegin =
      "    #==__lldb_expr_main_body_code_begin\n";
  constexpr StringLiteral kMainBodyBlockEnd =
      "    #==__lldb_expr_main_body_code_end\n";

  // Collect the current persistent variables.
  SmallVector<std::pair<StringRef, mlir::Type>> variables;
  collectPersistentVariables(state, variables);

  // Wrap the expression text in a function so that we can execute it.
  // TODO: This currently doesn't support imports or any kind of persistent
  // state.
  llvm::raw_string_ostream exprRawOS(transformedText);
  mlir::raw_indented_ostream exprOSIndented(exprRawOS);

  exprOSIndented << "from IO import _printf, print\n"
                 << "from Pointer import Pointer\n\n"
                 << "from PythonInterface import PythonInterface\n";

  // We insert the previously executed top level code to ensure functions,
  // imports and classes are preserved. This also ensures that saved
  // variables can be inspected again because the user defined types are
  // included in these pieces of code.
  for (const auto &exprInst : impl->persistentState.getExpressionInstances()) {
    if (exprInst.sourceCode)
      exprOSIndented << exprInst.sourceCode->getTopLevelCode();
  }

  // The following is the first chunk of code written by the user.
  exprOSIndented << kTopLevelBlockBegin << sourceCode.getTopLevelCode()
                 << kTopLevelBlockEnd;

  // Build the input struct, which contains each of the persistent
  // variables.
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
                    "  try:\n"
                    "    __lldb_expr_impl__(__lldb_arg";
  for (auto &var : variables) {
    exprOSIndented << formatv(
        ", __get_address_as_lvalue(__lldb_arg.{0}.load().address)", var.first);
  }
  exprOSIndented << ")\n"
                    "  except error:\n"
                    "    _printf(\"Error: \")\n"
                    "    print(error.value)\n\n";

  // Finally we can generate the actual expression function.
  exprOSIndented << "def __lldb_expr_impl__(__lldb_arg&: __lldb_context__";
  for (auto &var : variables) {
    exprOSIndented << llvm::formatv(", {0}&: __mlir_type.`{1}`", var.first,
                                    var.second);
  }
  exprOSIndented << "):\n";

  // The following is the other chunk of code just written by the user.
  exprOSIndented << kMainBodyBlockBegin;
  exprOSIndented.printReindented(sourceCode.getMainBodyCode(), "    ");
  exprOSIndented << kMainBodyBlockEnd;

  impl->typeSystem.debugLog("Parsing the following code:\n{0}",
                            transformedText.c_str());

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

    impl->typeSystem.debugLog("Attempting to rewrite the input expression");

    // If we can rewrite the expression, do so. If not, simply return.
    if (failed(impl->parser->rewriteExpression(diagnosticManager)))
      return;
    impl->typeSystem.debugLog(
        "Rewrote the input, next parse will be the fixed code");
    llvm::raw_string_ostream fixedOS(m_fixed_text);
    mlir::raw_indented_ostream indentedFixedOS(fixedOS);
    StringRef fixedText = diagnosticManager.GetFixedExpression();
    // Get the modified top level code
    indentedFixedOS << fixedText.slice(fixedText.find(kTopLevelBlockBegin) +
                                           kTopLevelBlockBegin.size(),
                                       fixedText.find(kTopLevelBlockEnd));

    // Get the modified main body and remove all indent.
    indentedFixedOS.printReindented(
        fixedText.slice(fixedText.find(kMainBodyBlockBegin) +
                            kMainBodyBlockBegin.size(),
                        fixedText.find(kMainBodyBlockEnd)),
        "");

    impl->typeSystem.debugLog("The new unwrapped code will be:\n{0}",
                              m_fixed_text);
    notifyFixits(diagnosticManager, m_fixed_text);
    // Clear the diagnostic manager so we don't re-fix something.
    diagnosticManager.Clear();
  };

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
  if (!crc.RunSafelyOnThread(parseModule)) {
    impl->typeSystem.errorLog(
        "Crash recovered: CrashRecoveryContext::RetCode (on POSIX: "
        "signal number + 128) = {0}",
        crc.RetCode);
    diagnosticManager.PutString(eDiagnosticSeverityError, "crash detected");
    return failure();
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Python expression parsing and execution

const std::optional<std::string> &MojoUserExpression::getPythonModuleName() {
  return impl->pythonModuleName;
}

/// Import the various top-level python symbols defined in the given python
/// expression into the current mojo context by emitting binding code to the
/// given stream.
static void importPythonSymbolsIntoMojo(StringRef pythonExpr,
                                        StringRef moduleName,
                                        raw_ostream &mojoExprOS) {
  // FIXME: This is an extremely hacky and limited python ast extractor. We
  // should really be using pythons pre-existing ast utilities, but we currently
  // don't have that available. When LLDB is built with python enabled, we
  // should kill all of this logic and use that to ask python for the
  // interesting bits instead.
  SmallVector<StringRef> lines;
  pythonExpr.split(lines, "\n");

  llvm::Regex importRegex(R"(^import ([_0-9a-zA-Z]+)$)");
  llvm::Regex importAsRegex(R"(^import ([_0-9a-zA-Z]+) as ([_0-9a-zA-Z]+)$)");
  llvm::Regex defRegex(R"(^def ([_0-9a-zA-Z]+)\()");
  llvm::Regex valueRegex(R"(^([_0-9a-zA-Z]+) =)");
  for (StringRef line : lines) {
    SmallVector<StringRef, 2> matches;
    if (importRegex.match(line, &matches)) {
      mojoExprOS << llvm::formatv(
          "var {0} = __repl_python__.importModule(\"{0}\")\n", matches[1]);
    } else if (importAsRegex.match(line, &matches)) {
      mojoExprOS << llvm::formatv(
          "var {0} = __repl_python__.importModule(\"{1}\")\n", matches[2],
          matches[1]);
    } else if (defRegex.match(line, &matches) ||
               valueRegex.match(line, &matches)) {
      mojoExprOS << llvm::formatv("var {0} = {1}.{0}\n", matches[1],
                                  moduleName);
    }
  }
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

code_string = '{0}'
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
      escapedPythonExprOS.write_escaped(
          llvm::formatv("from {0} import *\n", *exprInst.pythonModuleName)
              .str());
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

  // If we haven't initialized python yet, do that as part of this expression.
  if (!state.hasInitializedPython())
    mojoExprOS << "var __repl_python__ = PythonInterface()\n\n";

  // Evaluate the wrapped python expression.
  mojoExprOS << "__repl_python__.eval(\"";
  mojoExprOS.write_escaped(wrappedPythonExpr);
  mojoExprOS << "\")\n\n"
             << llvm::formatv(
                    "var {0} = __repl_python__.importModule(\"{0}\")\n\n",
                    moduleName);

  // Import the interesting top-level symbols from the python module into the
  // mojo context.
  importPythonSymbolsIntoMojo(pythonExpr, moduleName, mojoExprOS);

  // Now that we've got a Mojo expression, parse it the way we would any other
  // expression.
  m_expr_text = mojoExprOS.str();
  impl->pythonModuleName = std::move(moduleName);

  MojoExpressionSourceCode sourceCode(m_expr_text);
  return wrapTextAndParseExpression(sourceCode, diagnosticManager, exeCtx,
                                    exeScope, state);
}
