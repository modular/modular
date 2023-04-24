//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoExpressionParser.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "JITExecutionUnit.h"
#include "KGEN/CompilerRT.h"
#include "KGEN/KGENCompiler.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/MojoParser.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Logging.h"
#include "MojoDiagnostic.h"
#include "MojoExpressionVariable.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Process.h"
#include "llvm/Target/TargetMachine.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;
using namespace lldb_private;

//===----------------------------------------------------------------------===//
// MojoExpressionParser::Impl
//===----------------------------------------------------------------------===//

struct MojoExpressionParser::Impl {
  Impl(ExecutionContextScope *exeScope, MojoUserExpression &expr,
       const EvaluateExpressionOptions &options);

  /// The expression being parsed.
  MojoUserExpression &expr;

  /// The type system associated with the evaluation of the current expression.
  MojoTypeSystem *typeSystem = nullptr;

  /// The compilation options to use when compiling.
  const KGEN::CompilationOptions *compilationOptions = nullptr;

  /// The pass manager to use when compiling.
  std::unique_ptr<mlir::PassManager> passManager;

  /// The llvm context to use when compiling.
  std::unique_ptr<llvm::LLVMContext> llvmContext;

  /// The compiler instance to use when parsing.
  std::unique_ptr<KGEN::ObjectCompiler> compiler;

  /// The compiled llvm module.
  std::unique_ptr<llvm::Module> llvmModule;

  /// The options to use when evaluating the expression.
  EvaluateExpressionOptions options;

  /// A set of new persistent variables to be added to the persistent expression
  /// state if compilation of the expression succeeds.
  SmallVector<std::pair<StringRef, mlir::Type>> newPersistentVariables;
};

MojoExpressionParser::Impl::Impl(ExecutionContextScope *exeScope,
                                 MojoUserExpression &expr,
                                 const EvaluateExpressionOptions &options)
    : expr(expr), options(options) {
  // Bail out if we don't have a valid execution context.
  lldb::TargetSP target = exeScope ? exeScope->CalculateTarget() : nullptr;
  if (!target)
    return;

  // Grab the type system from the target, bailing out if we can't.
  auto typeSystemOr =
      target->GetScratchTypeSystemForLanguage(eLanguageTypeMojo);
  if (!typeSystemOr) {
    llvm::consumeError(typeSystemOr.takeError());
    return;
  }
  typeSystem = llvm::cast<MojoTypeSystem>(typeSystemOr.get().get());
  compilationOptions = &typeSystem->getParserContext().getCompilationOptions();
  MLIRContext *ctx = typeSystem->getMLIRContext();

  // Compute the target info to use for compilation.
  ErrorOr<TargetInfoAttr> targetInfoOr = getTargetInfoFor(
      ctx, compilationOptions->targetTriple, compilationOptions->targetCpu,
      compilationOptions->targetFeatures);
  if (targetInfoOr.isError())
    return;
  auto targetInfo = targetInfoOr.takeValue();

  // Build the compilation pipeline.
  BuildInfoAttr buildInfo = BuildInfoAttr::getForCurrentBuild(ctx);
  passManager =
      std::make_unique<mlir::PassManager>(ctx, ModuleOp::getOperationName());
  populateElaborateModulePasses(*passManager, typeSystem->getRuntime(),
                                targetInfo, buildInfo, compilationOptions);

  // Create the compiler instance.
  auto compilerOr =
      ObjectCompiler::create(typeSystem->getRuntime(), *passManager,
                             ".kgen_cache", *compilationOptions);
  if (failed(compilerOr))
    return;
  compiler = std::make_unique<KGEN::ObjectCompiler>(std::move(*compilerOr));
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

namespace {
/// This class defines a simple raw ostream that can be used to emit colors when
/// processing diagnostic messages.
struct DiagnosticStream : public llvm::raw_string_ostream {
  DiagnosticStream(std::string &msg, bool supportsColors)
      : llvm::raw_string_ostream(msg) {
    enable_colors(supportsColors);
  }

  bool is_displayed() const override { return colors_enabled(); }
  bool has_colors() const override { return colors_enabled(); }
};
} // namespace

/// Format the given diagnostic into a string.
static std::string formatSMDiagnostic(const llvm::SMDiagnostic &diag,
                                      bool showColors) {
  std::string msg;
  DiagnosticStream msgOS(msg, showColors);

  // Set the default colors for the diagnostic printing. This ensures we use the
  // correct corresponding color for the diagnostic type.
  llvm::HighlightColor color;
  switch (diag.getKind()) {
  case llvm::SourceMgr::DK_Error:
    color = llvm::HighlightColor::Error;
    break;
  case llvm::SourceMgr::DK_Warning:
    color = llvm::HighlightColor::Warning;
    break;
  case llvm::SourceMgr::DK_Note:
    color = llvm::HighlightColor::Note;
    break;
  case llvm::SourceMgr::DK_Remark:
    color = llvm::HighlightColor::Remark;
    break;
  }
  llvm::WithColor colorOS(msgOS, color);

  diag.print("", msgOS, showColors, /*ShowKindLabel=*/false);
  return msg;
}

//===----------------------------------------------------------------------===//
// LLDBMojoREPLListener
//===----------------------------------------------------------------------===//

namespace {
/// This class implements a parser listener that communicates between the Mojo
/// parser and the repl.
class LLDBMojoREPLListener : public MojoParserREPLListener {
public:
  LLDBMojoREPLListener(
      StringRef currentModuleName, MojoTypeSystem &typeSystem,
      MojoUserExpression &expr, DiagnosticManager &diagnosticManager,
      const EvaluateExpressionOptions &options,
      SmallVectorImpl<std::pair<StringRef, mlir::Type>> &newPersistentVariables)
      : currentModuleName(currentModuleName), typeSystem(typeSystem),
        expr(expr), diagnosticManager(diagnosticManager), options(options),
        newPersistentVariables(newPersistentVariables) {}
  ~LLDBMojoREPLListener() override = default;

  //===--------------------------------------------------------------------===//
  // Notifications

  void notifyWrappedExpr(StringRef wrappedExpr) override {
    typeSystem.debugLog("Parsing the following code:\n{0}", wrappedExpr.data());
  }

  void notifyFixedExpr(StringRef fixedExpr) override {
    expr.setFixedText(fixedExpr);
  }

  void notifyDiagnostics(ArrayRef<llvm::SMDiagnostic> diagnostics) override {
    typeSystem.debugLog("Found {0} diagnostic{1}\n", diagnostics.size(),
                        diagnostics.size() == 1 ? "" : "s");

    for (const llvm::SMDiagnostic &diag : diagnostics) {
      typeSystem.debugLog("Diagnostic with fixits: {0}, message:\n{1}",
                          diag.getFixIts().size(), diag.getMessage());

      // If this is a warning or remark from a previous module, ignore it. This
      // removes problems with emitting multiple diagnostics for the same
      // expression.
      llvm::SourceMgr::DiagKind diagKind = diag.getKind();
      if (diagKind == llvm::SourceMgr::DK_Warning ||
          diagKind == llvm::SourceMgr::DK_Remark) {
        if (MojoPersistentExpressionState::isExpressionModuleName(
                diag.getFilename()) &&
            diag.getFilename() != currentModuleName) {
          lastDiagnosticIgnored = true;
          continue;
        }
      }

      // If this is a note and the previous diagnostic was ignored, ignore this
      // as well.
      if (diagKind == llvm::SourceMgr::DK_Note && lastDiagnosticIgnored)
        continue;
      lastDiagnosticIgnored = false;

      // Turn the diagnostic severity into LLDB's severity.
      DiagnosticSeverity severity;
      switch (diagKind) {
      case llvm::SourceMgr::DK_Error:
        severity = eDiagnosticSeverityError;
        break;
      case llvm::SourceMgr::DK_Warning:
        severity = eDiagnosticSeverityWarning;
        break;
      case llvm::SourceMgr::DK_Remark:
        LLVM_FALLTHROUGH;
      case llvm::SourceMgr::DK_Note:
        severity = eDiagnosticSeverityRemark;
        break;
      }

      std::string msg = formatSMDiagnostic(diag, options.GetColorizeErrors());
      diagnosticManager.AddDiagnostic(std::make_unique<MojoDiagnostic>(
          msg, severity, !diag.getFixIts().empty()));
    }
  }

  //===--------------------------------------------------------------------===//
  // Queries

  bool shouldPersistVariable(StringRef name, mlir::Type type) override {
    // Only consider variables that were written by users, not those generated
    // by LLDB, which start with __lldb.
    if (name.starts_with("__lldb"))
      return false;
    // TODO: For now, we only persist variables in REPL mode. We should define
    // a policy for non-REPL mode (e.g. clang/swift using leading $ for variable
    // names to indicate persistence).
    if (!options.GetREPLEnabled())
      return false;

    newPersistentVariables.emplace_back(name, type);
    return true;
  }

private:
  StringRef currentModuleName;
  MojoTypeSystem &typeSystem;
  MojoUserExpression &expr;
  DiagnosticManager &diagnosticManager;
  const EvaluateExpressionOptions &options;
  SmallVectorImpl<std::pair<StringRef, mlir::Type>> &newPersistentVariables;

  /// A flag indicating if that the last processed diagnostic was ignored.
  bool lastDiagnosticIgnored = false;
};
} // namespace

//===----------------------------------------------------------------------===//
// MojoExpressionParser
//===----------------------------------------------------------------------===//

MojoExpressionParser::MojoExpressionParser(
    ExecutionContextScope *exeScope, MojoUserExpression &expr,
    const EvaluateExpressionOptions &options)
    : impl(std::make_unique<Impl>(exeScope, expr, options)) {}

MojoExpressionParser::~MojoExpressionParser() = default;

/// Collect the name and type of the current persistent variables within the
/// given state.
static void collectPersistentVariables(
    MojoPersistentExpressionState &state,
    SmallVectorImpl<std::pair<StringRef, mlir::Type>> &variables) {
  DenseSet<ConstString> persistentVariableNames;
  for (int i : llvm::reverse(llvm::seq<int>(0, state.GetSize()))) {
    lldb::ExpressionVariableSP var = state.GetVariableAtIndex(i);
    assert(var && "expected valid variable in persistent state");
    if (!persistentVariableNames.insert(var->GetName()).second)
      continue;

    mlir::Type varType = mlir::Type::getFromOpaquePointer(
        var->GetCompilerType().GetOpaqueQualType());
    variables.emplace_back(var->GetName().GetStringRef(), varType);
  }
}

M::LogicalResult
MojoExpressionParser::parse(MojoPersistentExpressionState &state,
                            DiagnosticManager &diagnosticManager) {
  if (!impl->compiler) {
    impl->typeSystem->errorLog("No compiler");
    return failure();
  }

  MojoParserContext &parserContext = impl->typeSystem->getParserContext();
  MLIRContext *ctx = impl->typeSystem->getMLIRContext();
  llvm::SourceMgr &sourceMgr = parserContext.getSourceMgr();

  // Register the source manager diagnostic handler so we get all the MLIR
  // diagnostics through the handler we already have and so it's all forwarded
  // to the LLDB streams. If the handler can't use the source manager for an
  // error, it'll print to errStream, which we will flush if it's non-empty on
  // scope exit.
  std::string errs;
  llvm::raw_string_ostream errStream(errs);
  mlir::SourceMgrDiagnosticHandler handler(sourceMgr, ctx, errStream);

  // On scope exit, if we've printed any errors make sure to log them.
  auto printOnError = llvm::make_scope_exit([&]() {
    if (errs.empty())
      return;
    impl->typeSystem->errorLog("{0}", errs);
  });

  // Collect the current persistent variables.
  SmallVector<std::pair<StringRef, mlir::Type>> variables;
  collectPersistentVariables(state, variables);

  // Parse the expression.
  std::string expressionId = state.getNextExpressionModuleName();
  LLDBMojoREPLListener listener(expressionId, *impl->typeSystem, impl->expr,
                                diagnosticManager, impl->options,
                                impl->newPersistentVariables);
  StringRef exprFnName = impl->expr.FunctionName();
  MojoASTDeclRef exprFnDecl = parserContext.parseREPLExpresion(
      listener, expressionId, impl->expr.Text(), exprFnName, variables);

  // If the parser supplied a fixed expression, abort processing and use that
  // expression instead.
  if (!impl->expr.GetFixedText().empty() &&
      impl->options.GetAutoApplyFixIts()) {
    impl->typeSystem->debugLog(
        "Rewrote the input, next parse will be the fixed code:\n{0}",
        impl->expr.GetFixedText());

    // If we have a fixed expression string, we're going to fail here to let
    // LLDB retry execution with the fixed expression. Before then, we need to
    // emit all of the fixed diagnostics that were collected, given that these
    // won't be shown on the next parse.
    auto filterFn = [](MojoDiagnostic &diag) { return diag.hadFixits(); };
    impl->typeSystem->broadcastDiagnostics(diagnosticManager, filterFn);
    diagnosticManager.Clear();

    // If the parser was actually successful, make sure to reset it so that we
    // don't include the un-fixed module in the REPL history.
    if (exprFnDecl)
      parserContext.removeLastREPLExpression();
    return failure();
  }

  if (!exprFnDecl) {
    impl->typeSystem->errorLog("Failed to parse the module");
    return failure();
  }
  impl->typeSystem->debugLog("Parsed module successfully");

  // Setup a diagnostic handler to process diagnostics emitted during lowering.
  sourceMgr.setDiagHandler(
      [](const llvm::SMDiagnostic &diag, void *context) {
        static_cast<LLDBMojoREPLListener *>(context)->notifyDiagnostics(diag);
      },
      &listener);

  // Functor containing various cleanup performed in the case of an error.
  auto returnErrorCleanup = [&] {
    // If we encounter an error anywhere during compilation, make sure the
    // parser doesn't include this expression in the REPL history.
    parserContext.removeLastREPLExpression();
    return failure();
  };

  // Create a clone of the parser module so that we can compile it without
  // thrashing on the current parser state.
  OwningOpRef<ModuleOp> module = parserContext.getModule().clone();

  // Ensure the expression function gets exported.
  LIT::FuncOp exprFn = cast<LIT::FuncOp>(exprFnDecl.getIfOperation());
  OpBuilder exportBuilder = OpBuilder::atBlockEnd(module->getBody());
  exportBuilder.create<ExportOp>(exprFn.getLoc(),
                                 LIT::getFullyResolvedSymbolRef(exprFn),
                                 exprFnName, /*isCExport=*/true);

  // Log the pre-elaboration module.
  std::string preElaborationModule;
  llvm::raw_string_ostream preElaborationStream(preElaborationModule);
  impl->passManager->enableIRPrinting(
      [](Pass *pass, Operation *) {
        return pass->getName() == "ElaborateGenerators";
      },
      [](Pass *, Operation *) { return false; }, /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/false, /*printAfterOnlyOnFailure=*/false,
      /*out=*/preElaborationStream);

  // Run the elaboration pipeline.
  if (failed(impl->passManager->run(*module))) {
    impl->typeSystem->errorLog("Elaboration failed");
    return returnErrorCleanup();
  }

  impl->typeSystem->dumpIR("Pre-elaboration module:\n{0}",
                           preElaborationModule);
  impl->typeSystem->dumpIR("Elaborated module:\n{0}", *module);

  // Lower the module to LLVM IR.
  SymbolTable symtab(*module);
  ExportMap exportedSymbols = getExportedSymbols(*module);
  impl->llvmContext = std::make_unique<llvm::LLVMContext>();
  impl->llvmModule = impl->compiler->lowerAllFuncsToLLVM(
      symtab, exportedSymbols, *impl->llvmContext);
  if (!impl->llvmModule) {
    impl->typeSystem->errorLog("Lowering to LLVM failed");
    return returnErrorCleanup();
  }

  impl->typeSystem->dumpIR("Pre-optimization LLVM module:\n{0}",
                           *impl->llvmModule);

  // Create the target machine so we can run the optimizer.
  auto targetMachineOr =
      KGEN::createTargetMachine(impl->compilationOptions, /*isJIT=*/true);
  if (targetMachineOr.isError()) {
    impl->typeSystem->errorLog("Failed to create the target machine: {0}",
                               targetMachineOr.getError());
    return returnErrorCleanup();
  }

  if (failed(KGEN::runLLVMOptPasses(*impl->llvmModule, **targetMachineOr,
                                    impl->compilationOptions))) {
    impl->typeSystem->errorLog("LLVM optimization failed");
    return returnErrorCleanup();
  }
  return success();
}

Status MojoExpressionParser::prepareForExecution(
    lldb::addr_t &funcAddr, lldb::addr_t &funcEnd,
    std::shared_ptr<JITExecutionUnit> &executionUnit, ExecutionContext &exeCtx,
    ExecutionPolicy executionPolicy, bool keepResultInMemory) {
  // Grab the LLVM module built during the parse phase.
  std::unique_ptr<llvm::Module> module = std::move(impl->llvmModule);
  if (!module) {
    Status err;
    err.SetErrorString("Can't prepare a NULL module for execution");
    return err;
  }

  // Retrieve an appropriate symbol context.
  SymbolContext sc;
  if (const lldb::StackFrameSP &frame = exeCtx.GetFrameSP())
    sc = frame->GetSymbolContext(lldb::eSymbolContextEverything);
  else if (const lldb::TargetSP &target = exeCtx.GetTargetSP())
    sc.target_sp = target;

  // Extract the target features.
  SmallVector<StringRef> splitFeatures;
  StringRef(impl->compilationOptions->targetFeatures).split(splitFeatures, ",");
  std::vector<std::string> features(splitFeatures.begin(), splitFeatures.end());

  // Build the IR execution unit responsible for executing the generated IR.
  ConstString functionName(impl->expr.FunctionName());
  executionUnit = std::make_shared<JITExecutionUnit>(
      impl->llvmContext, module, functionName, exeCtx.GetTargetSP(), sc,
      features);

  // Extract the function information for the expression entry point.
  Status error = executionUnit->getRunnableInfo(funcAddr, funcEnd);
  if (error.Fail() || !keepResultInMemory)
    return error;

  // Compute the target info to use for the persistent variable state.
  lldb_private::Process *process = exeCtx.GetProcessPtr();
  lldb::ByteOrder byteOrder = process->GetByteOrder();
  size_t addressByteSize = process->GetAddressByteSize();

  // If we successfully compiled the expression, we can now comfortably register
  // the persistent state variables.
  auto *persistentState = static_cast<MojoPersistentExpressionState *>(
      impl->typeSystem->GetPersistentExpressionState());

  // Register the current persistent variables with the materializer.
  DenseSet<ConstString> persistentVariableNames;
  for (int i : llvm::reverse(llvm::seq<int>(0, persistentState->GetSize()))) {
    lldb::ExpressionVariableSP var = persistentState->GetVariableAtIndex(i);
    assert(var && "expected valid variable in persistent state");

    // Skip variables that got redefined.
    if (!persistentVariableNames.insert(var->GetName()).second)
      continue;

    // Try adding the variable to the expression materializer.
    impl->expr.GetMaterializer()->AddPersistentVariable(var, nullptr, error);
    if (error.Fail())
      return error;
  }

  // Register the newly created persistent variables.
  std::vector<lldb::ExpressionVariableSP> peristentVariables;
  for (auto [name, mlirType] : impl->newPersistentVariables) {
    CompilerType lldbType(impl->typeSystem->weak_from_this(),
                          const_cast<void *>(mlirType.getAsOpaquePointer()));
    lldb::ExpressionVariableSP var = persistentState->CreatePersistentVariable(
        exeCtx.GetBestExecutionContextScope(), ConstString(name), lldbType,
        byteOrder, addressByteSize);
    if (!var) {
      error.SetErrorString("failed to create persistent variable");
      return error;
    }

    // Mark the variable as persistent, and notify LLDB that it needs to be
    // allocated.
    var->m_frozen_sp->SetHasCompleteType();
    var->m_flags |= ExpressionVariable::EVKeepInTarget;
    var->m_flags |= ExpressionVariable::EVIsLLDBAllocated;
    var->m_flags |= ExpressionVariable::EVNeedsAllocation;

    // Adding the variable to the expression materializer.
    impl->expr.GetMaterializer()->AddPersistentVariable(var, nullptr, error);
    if (error.Fail())
      return error;
    peristentVariables.emplace_back(std::move(var));
  }

  // If a valid execution unit was produced and there is more than one external
  // function in the execution unit, it needs to keep living even if it's not
  // top level, because the result could refer to that function, register it if
  // necessary.
  std::shared_ptr<JITExecutionUnit> persistedExecutionUnit;
  if (executionUnit &&
      (impl->options.GetExecutionPolicy() == eExecutionPolicyTopLevel ||
       executionUnit->getJittedFunctions().size() > 1)) {
    persistedExecutionUnit = executionUnit;
  }

  // Register the persisted state for this execution.
  persistentState->registerExpressionInstance(std::move(persistedExecutionUnit),
                                              std::move(peristentVariables),
                                              impl->expr.getPythonModuleName());
  return error;
}
