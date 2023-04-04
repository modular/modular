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
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "mlir/Pass/PassManager.h"
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
// MojoExpressionParser
//===----------------------------------------------------------------------===//

MojoExpressionParser::MojoExpressionParser(
    ExecutionContextScope *exeScope, Expression &expr,
    const EvaluateExpressionOptions &options)
    : ExpressionParser(exeScope, expr, options.GetGenerateDebugInfo()),
      impl(std::make_unique<Impl>(exeScope, expr, options)) {}
MojoExpressionParser::~MojoExpressionParser() = default;

M::LogicalResult MojoExpressionParser::parse() {
  if (!impl->compiler)
    return failure();

  // TODO: We should print the expression to a file if we need debug information
  // attached.
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(impl->exprToParse.Text());
  impl->sourceManager.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  // Import the mojo module.
  // TODO: Capture fixits, and apply them to the expression text.
  mlir::TimingScope scope;
  mlir::MLIRContext *ctx = impl->passManager->getContext();
  OwningOpRef<ModuleOp> module =
      importMojoFile(impl->sourceManager, ctx, scope, impl->compilationOptions,
                     /*useMLIRDiagnostics=*/false, *impl->runtime);
  if (!module)
    return failure();

  // Run the elaboration pipeline.
  if (failed(impl->passManager->run(*module)))
    return failure();

  // Lower the module to LLVM IR.
  SymbolTable symtab(*module);
  ExportMap exportedSymbols = getExportedSymbols(*module);
  impl->llvmContext = std::make_unique<llvm::LLVMContext>();
  impl->llvmModule = impl->compiler->lowerAllFuncsToLLVM(
      symtab, exportedSymbols, *impl->llvmContext);
  if (!impl->llvmModule)
    return failure();

  // TODO: Apply optimizations to the generated module.
  return success();
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
