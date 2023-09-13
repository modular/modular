//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoExpressionParser.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "JITExecutionUnit.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser.h"
#include "KGEN/MojoParser/ASTDeclRef.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Logging.h"
#include "MojoDiagnostic.h"
#include "MojoExpressionVariable.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlanCallFunction.h"
#include "lldb/Utility/LLDBLog.h"
#include "mlir/IR/IRMapping.h"
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

  /// Compile the list of functions to LLVM and return the module. This produces
  /// a 'standalone' module.
  ErrorOr<std::unique_ptr<llvm::Module>>
  compileFuncsToLLVM(const SymbolTable &symtab,
                     ArrayRef<KGEN::FuncOp> funcsToCompile);

  /// Given an evaluator, set of specializations, and a symbol table, construct
  /// a JITExecutionUnit. This function handles compiling the specializations to
  /// LLVM, adding the LLVM module to the MCJIT, and doing any interfacing with
  /// the JITExecutionUnit necessary.
  ErrorOr<std::shared_ptr<JITExecutionUnit>>
  produceExecutionUnit(ExecutionContext &exeCtx, KGEN::FuncOp evaluator,
                       const SymbolTable &symtab,
                       ArrayRef<KGEN::FuncOp> specializations);

  /// Callback that the elaborator can use to evaluate specializations and
  /// perform search using the LLDB JIT.
  ErrorOr<ElaboratorSearchFn>
  evaluateSpecializations(KGEN::FuncOp evaluator, const SymbolTable &symtab,
                          TargetInfoAttr target,
                          ArrayRef<KGEN::FuncOp> specializations);

  /// The expression being parsed.
  MojoUserExpression &expr;

  /// The execution context scope.
  ExecutionContextScope *exeScope;

  /// The type system associated with the evaluation of the current expression.
  MojoTypeSystem *typeSystem = nullptr;

  /// The compilation options to use when compiling.
  const KGEN::CompilationOptions *compilationOptions = nullptr;

  /// The pass manager to use (a) during elaboration, and (b) when compiling.
  /// They have to be different because the ObjectCompiler modifies the pass
  /// manager while it's compiling things and there's no good way to restore the
  /// state.
  std::unique_ptr<mlir::PassManager> duringElaborationPM, fullCompilationPM;

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
    : expr(expr), exeScope(exeScope), options(options) {
  // Bail out if we don't have a valid execution context.
  lldb::TargetSP target = exeScope ? exeScope->CalculateTarget() : nullptr;
  if (!target)
    return;

  // Grab the type system from the target, bailing out if we can't.
  auto typeSystemOr =
      target->GetScratchTypeSystemForLanguage(lldb::eLanguageTypeMojo);
  if (!typeSystemOr) {
    llvm::consumeError(typeSystemOr.takeError());
    return;
  }
  typeSystem = llvm::cast<MojoTypeSystem>(typeSystemOr.get().get());
  compilationOptions = &typeSystem->getParserContext().getCompilationOptions();
  MLIRContext *ctx = typeSystem->getMLIRContext();
  llvmContext = std::make_unique<llvm::LLVMContext>();

  // Get the target info to use for compilation.
  TargetInfoAttr targetInfo = typeSystem->GetTargetInfo();
  if (!targetInfo)
    return;

  // Build the compilation pipeline.
  BuildInfoAttr buildInfo = BuildInfoAttr::getForCurrentBuild(ctx);
  fullCompilationPM =
      std::make_unique<mlir::PassManager>(ctx, ModuleOp::getOperationName());
  buildGenerateLibraryPipeline(*fullCompilationPM, typeSystem->getRuntime(),
                               *compilationOptions);
  buildElaborateModulePipeline(
      *fullCompilationPM, typeSystem->getRuntime(), targetInfo, buildInfo,
      [&](KGEN::FuncOp evaluator, const SymbolTable &symtab,
          TargetInfoAttr target, ArrayRef<KGEN::FuncOp> specializations) {
        return evaluateSpecializations(evaluator, symtab, target,
                                       specializations);
      },
      compilationOptions);
  buildPostElaborationPipeline(*fullCompilationPM, typeSystem->getRuntime(),
                               compilationOptions);

  // Create the compiler instance.
  auto compilerOr = ObjectCompiler::create(typeSystem->getRuntime(),
                                           *fullCompilationPM, ".mojo_cache",
                                           *compilationOptions, /*isJIT=*/true);
  if (failed(compilerOr))
    return;
  compiler = std::make_unique<KGEN::ObjectCompiler>(std::move(*compilerOr));

  // Now create the pass manager we need during elaboration. They have
  // to be different so the pass managers don't clash.
  duringElaborationPM =
      std::make_unique<mlir::PassManager>(ctx, ModuleOp::getOperationName());
}

ErrorOr<std::unique_ptr<llvm::Module>>
MojoExpressionParser::Impl::compileFuncsToLLVM(
    const SymbolTable &symtab, ArrayRef<KGEN::FuncOp> funcsToCompile) {
  // Create the set of symbols to export.
  KGEN::ExportMap exports;
  for (auto e : funcsToCompile) {
    StringAttr symName = e.getSymNameAttr();
    typeSystem->debugLog("[evaluateSpecializations] Exporting {0}",
                         symName.getValue());
    exports.insert({symName, ExportedSymbol()});
  }

  // Create the target machine so we can run the optimizer.
  auto targetMachineOr =
      KGEN::createTargetMachine(compilationOptions, /*isJIT=*/true);
  if (targetMachineOr.isError()) {
    typeSystem->errorLog(
        "[evaluateSpecializations] Failed to create the target machine: {0}",
        targetMachineOr.getError());
    return M::Error("failed to create the target machine");
  }

  // Lower everything to LLVM and run the optimizer.
  compiler->setForSearch(true);
  auto module = compiler->lowerAllFuncsToLLVM(symtab, exports, *llvmContext);
  compiler->setForSearch(false);
  if (!module) {
    typeSystem->errorLog("[evaluateSpecializations] failed to lower to LLVM");
    return M::Error("failed to lower to LLVM");
  }

  if (failed(KGEN::runLLVMOptPasses(*module, **targetMachineOr,
                                    compilationOptions,
                                    typeSystem->getRuntime()))) {
    typeSystem->errorLog("[evaluateSpecializations] LLVM optimization failed");
    return M::Error("LLVM optimization failed");
  }

  return module;
}

ErrorOr<std::shared_ptr<JITExecutionUnit>>
MojoExpressionParser::Impl::produceExecutionUnit(
    ExecutionContext &exeCtx, KGEN::FuncOp evaluator, const SymbolTable &symtab,
    ArrayRef<KGEN::FuncOp> specializations) {
  // Grab the thread we want for execution.
  auto &threadList = exeCtx.GetProcessPtr()->GetThreadList();
  exeCtx.SetThreadSP(threadList.GetExpressionExecutionThread());

  SmallVector<FuncOp> funcsToCompile(specializations);
  funcsToCompile.push_back(evaluator);

  auto moduleOr = compileFuncsToLLVM(symtab, funcsToCompile);
  if (moduleOr.isError())
    return moduleOr.takeError();
  std::unique_ptr<llvm::Module> module = std::move(*moduleOr);

  // Extract the target features.
  SmallVector<StringRef> splitFeatures;
  StringRef(compilationOptions->targetFeatures).split(splitFeatures, ",");
  std::vector<std::string> features(splitFeatures.begin(), splitFeatures.end());

  // Pull together the symbol context.
  SymbolContext sc;
  if (const lldb::StackFrameSP &frame = exeCtx.GetFrameSP())
    sc = frame->GetSymbolContext(lldb::eSymbolContextEverything);
  else if (const lldb::TargetSP &target = exeCtx.GetTargetSP())
    sc.target_sp = target;

  // Now JIT the LLVM module.
  ConstString name(evaluator.getSymName());
  auto executionUnit = std::make_shared<JITExecutionUnit>(
      std::move(llvmContext), std::move(module), name, exeCtx.GetTargetSP(), sc,
      features);

  // Refresh the LLVM context in the impl.
  llvmContext = std::make_unique<llvm::LLVMContext>();
  return executionUnit;
}

namespace {
/// RAII wrapper for an allocation in the inferior (target) process.
struct InferiorProcessAllocation {
  /// Alloc a memory block on creation.
  InferiorProcessAllocation(lldb::ProcessSP p, size_t size,
                            lldb::Permissions permissions, Status &error)
      : process(std::move(p)), allocAddr(LLDB_INVALID_ADDRESS) {
    allocAddr = process->CallocateMemory(size, permissions, error);
  }

  /// These objects are non-copyable.
  InferiorProcessAllocation(const InferiorProcessAllocation &other) = delete;
  InferiorProcessAllocation &
  operator=(const InferiorProcessAllocation &other) = delete;

  /// These objects are move-able, but only one can own the allocAddr to avoid
  /// double-free.
  InferiorProcessAllocation(InferiorProcessAllocation &&other)
      : process(std::move(other.process)), allocAddr(LLDB_INVALID_ADDRESS) {
    std::swap(allocAddr, other.allocAddr);
  }

  /// Converts to lldb::addr_t so that we can use it without modification.
  /*implicit*/ operator lldb::addr_t() { return allocAddr; }

  /// Leak memory intentionally so we can inspect it after a failure.
  void leak() { allocAddr = LLDB_INVALID_ADDRESS; }

  /// Dealloc on destruction, assuming we have a valid address.
  ~InferiorProcessAllocation() {
    if (allocAddr != LLDB_INVALID_ADDRESS)
      process->DeallocateMemory(allocAddr);
  }

  lldb::ProcessSP process;
  lldb::addr_t allocAddr;
};
} // namespace

ErrorOr<ElaboratorSearchFn> MojoExpressionParser::Impl::evaluateSpecializations(
    KGEN::FuncOp evaluator, const SymbolTable &symtab, TargetInfoAttr target,
    ArrayRef<KGEN::FuncOp> specializations) {
  // Update the pass manager to be the one we use during elaboration. At scope
  // exit, reset it to the one we are using outside elaboration.
  compiler->updatePassManager(*duringElaborationPM);
  auto resetPM = llvm::make_scope_exit(
      [&]() { compiler->updatePassManager(*fullCompilationPM); });

  typeSystem->debugLog(
      "[evaluateSpecializations] Got {0} specializations to evaluate",
      specializations.size());

  // Get the execution context.
  ExecutionContext exeCtx;
  exeScope->CalculateExecutionContext(exeCtx);

  // Produce the execution unit - this adds objects to the JIT.
  auto executionUnitOr =
      produceExecutionUnit(exeCtx, evaluator, symtab, specializations);
  if (executionUnitOr.isError())
    return executionUnitOr.takeError();
  std::shared_ptr<JITExecutionUnit> executionUnit = std::move(*executionUnitOr);

  // Extract the function information for the expression entry point.
  lldb::addr_t funcAddr, funcEnd;
  Status error = executionUnit->getRunnableInfo(funcAddr, funcEnd);
  if (error.Fail())
    return toModularError(error.ToError());

  typeSystem->debugLog("[evaluateSpecializations] Evaluator at {0}",
                       (void *)funcAddr);

  // Compute the target info to use for the persistent variable state.
  lldb_private::Process *process = exeCtx.GetProcessPtr();
  size_t addressByteSize = process->GetAddressByteSize();

  // Pull together all the specialization addresses.
  SmallVector<lldb::addr_t> specializationAddrs;
  for (KGEN::FuncOp s : specializations) {
    auto fn = llvm::find_if(executionUnit->getJittedFunctions(),
                            [&](const JITExecutionUnit::JittedFunction &fn) {
                              return fn.name == ConstString(s.getName());
                            });
    if (fn == executionUnit->getJittedFunctions().end()) {
      typeSystem->errorLog(
          "[evaluateSpecializations] could not find specialization {0}",
          s.getName());
      return M::Error("could not find specialization " + s.getName());
    }

    assert(fn->remoteAddr != LLDB_INVALID_ADDRESS &&
           "remote addr must be resolved by now");
    specializationAddrs.push_back(fn->remoteAddr);
  }

  // Allocate space for the specializations. We have to alloc enough space for
  // `specializationAddrs.size()` pointers, and then write them to that spot.
  auto allocForSpecializations = InferiorProcessAllocation(
      exeCtx.GetProcessSP(), addressByteSize * specializationAddrs.size(),
      lldb::Permissions::ePermissionsReadable |
          lldb::Permissions::ePermissionsWritable,
      error);
  if (error.Fail())
    return toModularError(error.ToError());

  // Write the addresses to the allocation we made for the specializations.
  for (auto [index, sa] : llvm::enumerate(specializationAddrs)) {
    exeCtx.GetProcessRef().WritePointerToMemory(
        allocForSpecializations + index * addressByteSize, sa, error);
    if (error.Fail())
      return toModularError(error.ToError());
  }

  // Allocate space for the out-parameter pointer we use.
  auto allocForBest =
      InferiorProcessAllocation(exeCtx.GetProcessSP(), addressByteSize,
                                lldb::Permissions::ePermissionsReadable |
                                    lldb::Permissions::ePermissionsWritable,
                                error);
  if (error.Fail())
    return toModularError(error.ToError());

  // Find lldb_evaluate_specializations now.
  bool missingWeak;
  lldb::addr_t lldbEvaluateSpecializationsAddr = executionUnit->findSymbol(
      ConstString("lldb_evaluate_specializations"), missingWeak);
  typeSystem->debugLog(
      "[evaluateSpecializations] lldb_evaluate_specializations at {0}",
      (void *)lldbEvaluateSpecializationsAddr);

  EvaluateExpressionOptions opts;
  // FIXME(19136): Set an infinite timeout.
  opts.SetTimeout(Timeout<std::micro>(std::chrono::seconds(1000)));
  opts.SetOneThreadTimeout(Timeout<std::micro>(std::nullopt));

  // Create the thread plan to call `lldb_evaluate_specializations`.
  lldb::ThreadPlanSP callPlan(new ThreadPlanCallFunction(
      exeCtx.GetThreadRef(), lldbEvaluateSpecializationsAddr,
      typeSystem->GetBuiltinTypeByName(ConstString("void")),
      {funcAddr, allocForSpecializations, (lldb::addr_t)specializations.size(),
       allocForBest},
      opts));

  StreamString ss;
  if (!callPlan || !callPlan->ValidatePlan(&ss)) {
    typeSystem->errorLog(ss.GetString());
    return M::Error("could not set up the expression");
  }

  return [this, exeCtx = std::move(exeCtx), callPlan = std::move(callPlan),
          opts = std::move(opts),
          allocForSpecializations = std::move(allocForSpecializations),
          allocForBest = std::move(allocForBest),
          error = std::move(error)]() mutable -> ErrorOr<ssize_t> {
    typeSystem->debugLog(
        "-- [evaluateSpecializations] Execution of expression begins --");

    DiagnosticManager diagnosticManager;
    lldb::ExpressionResults executionResult =
        exeCtx.GetProcessRef().RunThreadPlan(exeCtx, callPlan, opts,
                                             diagnosticManager);

    typeSystem->debugLog("-- [evaluateSpecializations] Execution of expression "
                         "completed --");

    if (executionResult != lldb::eExpressionCompleted) {
      allocForSpecializations.leak();
      allocForBest.leak();
      typeSystem->errorLog(
          "[evaluateSpecializations] Couldn't execute function; result was {0}",
          Process::ExecutionResultAsCString(executionResult));
      return M::Error("couldn't execute the evaluator");
    }

    // Read the memory from the allocation for 'best'.
    uint64_t bestVar = exeCtx.GetProcessRef().ReadUnsignedIntegerFromMemory(
        allocForBest, sizeof(uint64_t), 0, error);
    if (error.Fail())
      return toModularError(error.ToError());

    typeSystem->debugLog("[evaluateSpecializations] Got best = {0}", bestVar);
    return bestVar;
  };
};

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
            !diag.getFilename().ends_with(currentModuleName)) {
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
    // Check if we were requested not to persist anything.
    if (options.GetSuppressPersistentResult())
      return false;
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
  state.collectPersistentVariables(variables);

  // Parse the expression.
  auto [expressionId, exprModuleName] = state.getNextExpressionModuleName();
  LLDBMojoREPLListener listener(exprModuleName, *impl->typeSystem, impl->expr,
                                diagnosticManager, impl->options,
                                impl->newPersistentVariables);
  // Create a function name for the expression. This string must be a valid Mojo
  // identifier.
  std::string exprFnName = ("__lldb_expr__" + Twine(expressionId)).str();
  int exprFileId = sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(impl->expr.Text(), exprModuleName),
      llvm::SMLoc());
  impl->expr.setFunctionName(exprFnName);
  MojoASTDeclRef exprFnDecl = parserContext.parseREPLExpresion(
      listener, exprFileId, exprFnName, variables);

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
  struct MLIRDiagnosticHandlerContext {
    LLDBMojoREPLListener &listener;
    MojoParserContext &parserContext;
  };
  MLIRDiagnosticHandlerContext handlerContext{listener, parserContext};
  sourceMgr.setDiagHandler(
      [](const llvm::SMDiagnostic &diag, void *context) {
        auto *ctx = static_cast<MLIRDiagnosticHandlerContext *>(context);
        ctx->listener.notifyDiagnostics(
            ctx->parserContext.getREPLLocMapper().mapDiagnostic(diag));
      },
      &handlerContext);

  // Functor containing various cleanup performed in the case of an error.
  auto returnErrorCleanup = [&] {
    // If we encounter an error anywhere during compilation, make sure the
    // parser doesn't include this expression in the REPL history.
    parserContext.removeLastREPLExpression();
    return failure();
  };

  // Create a clone of the parser module so that we can compile it without
  // thrashing on the current parser state.
  LIT::FuncOp exprFn = cast<LIT::FuncOp>(exprFnDecl.getIfOperation());
  exprFn.setLinkageName(exprFnName);
  mlir::IRMapping mapping;
  OwningOpRef<ModuleOp> module =
      cast<ModuleOp>(parserContext.getModule()->clone(mapping));

  // Ensure the expression function in the cloned module gets exported.
  auto clonedExprFn = cast<LIT::FuncOp>(mapping.lookup(&*exprFn));
  clonedExprFn.setCExported();

  // Log the pre-elaboration module.
  std::string preElaborationModule;
  llvm::raw_string_ostream preElaborationStream(preElaborationModule);
  impl->fullCompilationPM->enableIRPrinting(
      [](Pass *pass, Operation *) {
        return pass->getName() == "ElaborateGenerators";
      },
      [](Pass *, Operation *) { return false; }, /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/false, /*printAfterOnlyOnFailure=*/false,
      /*out=*/preElaborationStream);

  // Run the elaboration pipeline.
  if (failed(impl->fullCompilationPM->run(*module))) {
    impl->typeSystem->errorLog("Elaboration failed");
    return returnErrorCleanup();
  }

  impl->typeSystem->dumpIR("Pre-elaboration module:\n{0}",
                           preElaborationModule);
  impl->typeSystem->dumpIR("Elaborated module:\n{0}", *module);

  // Lower the module to LLVM IR.
  SymbolTable symtab(*module);
  ExportMap exportedSymbols = getExportedSymbols(*module);
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
                                    impl->compilationOptions,
                                    impl->typeSystem->getRuntime()))) {
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
  // NOTE: impl->llvmContext will be nullptr after this! Luckily, expressions
  // are generally destroyed shortly after this, so we don't have to be too
  // concerned - just something to be aware of.
  executionUnit = std::make_shared<JITExecutionUnit>(
      std::move(impl->llvmContext), std::move(module), functionName,
      exeCtx.GetTargetSP(), sc, features);

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
    // All persistent variables in the REPL are references, so wrap them in a
    // reference type.
    auto ptr = LIT::REPLResultRefType::get(mlirType);
    CompilerType lldbType(impl->typeSystem->weak_from_this(),
                          const_cast<void *>(ptr.getAsOpaquePointer()));
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
