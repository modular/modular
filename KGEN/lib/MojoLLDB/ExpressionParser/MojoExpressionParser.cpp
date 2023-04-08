//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoExpressionParser.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "KGEN/CompilerRT.h"
#include "KGEN/KGENCompiler.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/ParseLit.h"
#include "Logging.h"
#include "MojoDiagnostic.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Process.h"
#include "llvm/Target/TargetMachine.h"

using namespace M::KGEN::Mojo;
using namespace lldb_private;

//===----------------------------------------------------------------------===//
// MojoExpressionParser::Impl
//===----------------------------------------------------------------------===//

struct MojoExpressionParser::Impl {
  Impl(ExecutionContextScope *exeScope, Expression &expr,
       const EvaluateExpressionOptions &options);

  /// The compilation options to use when compiling.
  KGEN::CompilationOptions compilationOptions;

  /// A source manager to use when compiling.
  llvm::SourceMgr sourceManager;

  /// The main LLCL runtime.
  LLCL::Runtime *runtime = nullptr;

  /// The pass manager to use when compiling.
  std::unique_ptr<mlir::PassManager> passManager;

  /// The llvm context to use when compiling.
  std::unique_ptr<llvm::LLVMContext> llvmContext;

  /// The compiler instance to use when parsing.
  std::unique_ptr<KGEN::ObjectCompiler> compiler;

  /// The compiled llvm module.
  std::unique_ptr<llvm::Module> llvmModule;

  /// The expression to be parsed.
  Expression &exprToParse;

  /// The options to use when evaluating the expression.
  EvaluateExpressionOptions options;
};

MojoExpressionParser::Impl::Impl(ExecutionContextScope *exeScope,
                                 Expression &expr,
                                 const EvaluateExpressionOptions &options)
    : exprToParse(expr), options(options) {
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
  auto *typeSystem = llvm::cast<MojoTypeSystem>(typeSystemOr.get().get());
  MLIRContext *ctx = typeSystem->getMLIRContext();
  runtime = &typeSystem->getRuntime();

  // Compute the target information for the expression.
  // TODO: Populate cpu features properly here.
  ArchSpec targetArch = target->GetArchitecture();
  if (targetArch.IsValid())
    compilationOptions.targetTriple = targetArch.GetTriple().str();
  compilationOptions.targetCpu = targetArch.GetClangTargetCPU();

  // Compute the target info to use for compilation.
  ErrorOr<TargetInfoAttr> targetInfoOr = getTargetInfoFor(
      ctx, compilationOptions.targetTriple, compilationOptions.targetCpu,
      compilationOptions.targetFeatures);
  if (targetInfoOr.isError())
    return;
  auto targetInfo = targetInfoOr.takeValue();

  // Build the compilation pipeline.
  BuildInfoAttr buildInfo = BuildInfoAttr::getForCurrentBuild(ctx);
  passManager =
      std::make_unique<mlir::PassManager>(ctx, ModuleOp::getOperationName());
  populateElaborateModulePasses(*passManager, *runtime, targetInfo, buildInfo,
                                compilationOptions);

  // Create the compiler instance.
  auto compilerOr =
      ObjectCompiler::create(typeSystem->getRuntime(), *passManager,
                             ".kgen_cache", compilationOptions);
  if (failed(compilerOr))
    return;
  compiler = std::make_unique<KGEN::ObjectCompiler>(std::move(*compilerOr));
}

//===----------------------------------------------------------------------===//
// MojoDiagnostic
//===----------------------------------------------------------------------===//

/// Handle an llvm diagnostic by adding a MojoDiagnostic to the LLDB diagnostic
/// manager.
static void handleDiagnostic(const llvm::SMDiagnostic &diagnostic, void *ctx) {
  auto *manager = (DiagnosticManager *)ctx;
  // Turn the diagnostic severity into LLDB's severity.
  DiagnosticSeverity severity;
  switch (diagnostic.getKind()) {
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
  std::string msg;
  llvm::raw_string_ostream diagnosticStream(msg);
  diagnostic.print("mojo", diagnosticStream, /*ShowColors=*/false,
                   /*ShowKindLabel=*/false);

  manager->AddDiagnostic(
      std::make_unique<MojoDiagnostic>(msg, diagnostic.getFixIts(), severity));
}

//===----------------------------------------------------------------------===//
// MojoExpressionParser
//===----------------------------------------------------------------------===//

MojoExpressionParser::MojoExpressionParser(
    ExecutionContextScope *exeScope, Expression &expr,
    const EvaluateExpressionOptions &options)
    : ExpressionParser(exeScope, expr, options.GetGenerateDebugInfo()),
      impl(std::make_unique<Impl>(exeScope, expr, options)) {}
MojoExpressionParser::~MojoExpressionParser() = default;

/// Apply fixits in the diagnostic manager to `expr` and set the fixed
/// expression to the new text. Only if all diagnostics (a) have fix-its and (b)
/// they can all be applied, will this rewrite the text. It will otherwise
/// return false.
bool MojoExpressionParser::RewriteExpression(
    DiagnosticManager &diagnosticManager) {
  MOJO_EXPR_LOG("Found {0} diagnostic{1}",
                diagnosticManager.Diagnostics().size(),
                diagnosticManager.Diagnostics().size() == 1 ? "" : "s");
  StringRef originalText(impl->exprToParse.Text());
  // This takes advantage of the fact that fixits are ordered to apply multiple
  // fixits to a single expression.
  std::string newText;
  size_t prevEnd = 0;
  auto applyFixit = [&](const llvm::SMFixIt &fixit) -> bool {
    llvm::SMRange range = fixit.getRange();
    if (!range.isValid())
      return false;

    StringRef removedText(range.Start.getPointer(),
                          range.End.getPointer() - range.Start.getPointer());
    StringRef insertedText = fixit.getText();
    MOJO_EXPR_LOG("Change \"{0}\" to \"{1}\"", removedText, insertedText);

    // The current range starts at the previous end pointer.
    StringRef currentOriginalRange(originalText.begin() + prevEnd);

    // Add the substring from the start of the current original text range.
    if (range.Start.getPointer() < currentOriginalRange.end() &&
        range.Start.getPointer() >= currentOriginalRange.begin())
      newText += currentOriginalRange.substr(
          0, range.Start.getPointer() - currentOriginalRange.begin());

    MOJO_EXPR_LOG("New expr before fixit insertion:\n{0}", newText);

    // Add the text to insert.
    newText += insertedText;

    // Update prevEnd. At the *very* end, we will clean up by adding the
    // remaining substring. Subtract off the size of the inserted text because
    // the pointers are all indexed off the original text.
    prevEnd += range.End.getPointer() - currentOriginalRange.begin();
    return true;
  };

  bool allDiagsHandled = true;
  for (const auto &diag : diagnosticManager.Diagnostics()) {
    MOJO_EXPR_LOG("Diagnostic with fixits: {0}, message:\n{1}",
                  diag->HasFixIts(), diag->GetMessage());

    // If it's a mojo diagnostic, it might have fix-its. If it does, and if that
    // fails, return false. Otherwise, continue.
    if (const auto *mojoDiag = llvm::dyn_cast<MojoDiagnostic>(diag.get())) {
      for (const llvm::SMFixIt &fixit : mojoDiag->getFixIts()) {
        if (!applyFixit(fixit)) {
          allDiagsHandled = false;
          break;
        }
      }
      continue;
    }

    // Not a Mojo diagnostic, we didn't handle everything.
    allDiagsHandled = false;
  }

  // If we handled all the diagnostics, then we set the fixed expression.
  if (allDiagsHandled) {
    // Complete fixit handling by adding the substring from prevEnd to the end
    // of the buffer. We do this here because we only want to do it if/once
    // *all* diagnostics are handled.
    newText += originalText.substr(prevEnd);
    MOJO_EXPR_LOG("Fixits applied to expression: \n{0}", newText.c_str());
    diagnosticManager.SetFixedExpression(newText);
  } else {
    MOJO_EXPR_LOG("Unhandled diagnostics found!");
  }

  return allDiagsHandled;
}

M::LogicalResult
MojoExpressionParser::parse(DiagnosticManager &diagnosticManager) {
  if (!impl->compiler) {
    MOJO_EXPR_LOG("No compiler");
    return failure();
  }

  // Add the build folder as an include dir if we have the correct environment
  // variable. This is for the python configuration, which we use CMake to find.
  // TODO: This is kinda awful, and we should probably pull in the python
  //   location directly if we can.
  std::optional<std::string> pathOr =
      llvm::sys::Process::GetEnv("MODULAR_PATH");
  if (pathOr) {
    impl->sourceManager.setIncludeDirs(
        {std::filesystem::path(*pathOr) / ".derived" / "build" / "Kernels" /
         "mojo" / "Python"});
  }

  // TODO: We should print the expression to a file if we need debug information
  // attached.
  auto buffer = llvm::MemoryBuffer::getMemBuffer(impl->exprToParse.Text());
  impl->sourceManager.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  // Set the diagnostic handler to create MojoDiagnostics that we can use to
  // capture fix-its.
  impl->sourceManager.setDiagHandler(handleDiagnostic, &diagnosticManager);

  // Import the mojo module.
  mlir::TimingScope scope;
  mlir::MLIRContext *ctx = impl->passManager->getContext();
  MojoParserConfig config(ctx, *impl->runtime, impl->compilationOptions);
  OwningOpRef<ModuleOp> module =
      importMojoFile(impl->sourceManager, config, scope);
  if (!diagnosticManager.Diagnostics().empty()) {
    MOJO_EXPR_LOG("Emitted diagnostics");
    return failure();
  }

  if (!module) {
    MOJO_EXPR_LOG("Failed to parse the module");
    return failure();
  }

  MOJO_EXPR_LOG("Parsed module successfully");

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
    MOJO_EXPR_LOG("Elaboration failed");
    return failure();
  }

  MOJO_EXPR_LOG("Pre-elaboration module:\n{0}", preElaborationModule);
  MOJO_EXPR_LOG("Elaborated module:\n{0}", *module);

  // Lower the module to LLVM IR.
  SymbolTable symtab(*module);
  ExportMap exportedSymbols = getExportedSymbols(*module);
  impl->llvmContext = std::make_unique<llvm::LLVMContext>();
  impl->llvmModule = impl->compiler->lowerAllFuncsToLLVM(
      symtab, exportedSymbols, *impl->llvmContext);
  if (!impl->llvmModule) {
    MOJO_EXPR_LOG("Lowering to LLVM failed");
    return failure();
  }

  MOJO_EXPR_LOG("Pre-optimization LLVM module:\n{0}", *impl->llvmModule);

  // Create the target machine so we can run the optimizer.
  auto targetMachineOr =
      KGEN::createTargetMachine(impl->compilationOptions, /*isJIT=*/true);
  if (targetMachineOr.isError()) {
    MOJO_EXPR_LOG("Failed to create the target machine: {0}",
                  targetMachineOr.getError());
    return failure();
  }

  return KGEN::runLLVMOptPasses(*impl->llvmModule, **targetMachineOr);
}

Status MojoExpressionParser::PrepareForExecution(
    lldb::addr_t &funcAddr, lldb::addr_t &funcEnd,
    lldb::IRExecutionUnitSP &executionUnit, ExecutionContext &exeCtx,
    bool &canInterpret, ExecutionPolicy executionPolicy) {
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
  StringRef(impl->compilationOptions.targetFeatures).split(splitFeatures, ",");
  std::vector<std::string> features(splitFeatures.begin(), splitFeatures.end());

  // Build the IR execution unit responsible for executing the generated IR.
  ConstString functionName(m_expr.FunctionName());
  executionUnit =
      std::make_shared<IRExecutionUnit>(impl->llvmContext, module, functionName,
                                        exeCtx.GetTargetSP(), sc, features);

  // Extract the function information for the expression entry point.
  Status error;
  executionUnit->GetRunnableInfo(error, funcAddr, funcEnd);
  return error;
}
