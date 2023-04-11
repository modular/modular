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
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ParseLit.h"
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
  Impl(ExecutionContextScope *exeScope, Expression &expr,
       const EvaluateExpressionOptions &options);

  /// The expression being parsed.
  Expression &expr;

  /// The type system associated with the evaluation of the current expression.
  MojoTypeSystem *typeSystem = nullptr;

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

  /// The options to use when evaluating the expression.
  EvaluateExpressionOptions options;

  /// A set of new persistent variables to be added to the persistent expression
  /// state if compilation of the expression succeeds.
  SmallVector<std::pair<StringRef, mlir::Type>> newPersistentVariables;
};

MojoExpressionParser::Impl::Impl(ExecutionContextScope *exeScope,
                                 Expression &expr,
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
    : impl(std::make_unique<Impl>(exeScope, expr, options)) {}
MojoExpressionParser::~MojoExpressionParser() = default;

/// Apply fixits in the diagnostic manager to `expr` and set the fixed
/// expression to the new text. Only if all diagnostics (a) have fix-its and (b)
/// they can all be applied, will this rewrite the text. It will otherwise
/// return false.
LogicalResult
MojoExpressionParser::rewriteExpression(DiagnosticManager &diagnosticManager) {
  impl->typeSystem->debugLog(
      "Found {0} diagnostic{1}\n", diagnosticManager.Diagnostics().size(),
      diagnosticManager.Diagnostics().size() == 1 ? "" : "s");
  // `originalText` is the wrapped code, not what the user wrote.
  StringRef originalText(impl->expr.Text());

  // This takes advantage of the fact that fixits are ordered to apply multiple
  // fixits to a single expression.
  std::string newText;
  size_t prevEnd = 0;
  auto applyFixit = [&](const llvm::SMFixIt &fixit) -> LogicalResult {
    llvm::SMRange range = fixit.getRange();
    if (!range.isValid())
      return failure();

    StringRef removedText(range.Start.getPointer(),
                          range.End.getPointer() - range.Start.getPointer());
    StringRef insertedText = fixit.getText();
    impl->typeSystem->debugLog("Change \"{0}\" to \"{1}\"", removedText,
                               insertedText);

    // The current range starts at the previous end pointer.
    StringRef currentOriginalRange(originalText.begin() + prevEnd);

    // Add the substring from the start of the current original text range.
    if (range.Start.getPointer() < currentOriginalRange.end() &&
        range.Start.getPointer() >= currentOriginalRange.begin())
      newText += currentOriginalRange.substr(
          0, range.Start.getPointer() - currentOriginalRange.begin());

    impl->typeSystem->debugLog("New expr before fixit insertion:\n{0}",
                               newText);

    // Add the text to insert.
    newText += insertedText;

    // Update prevEnd. At the *very* end, we will clean up by adding the
    // remaining substring. Subtract off the size of the inserted text because
    // the pointers are all indexed off the original text.
    prevEnd += range.End.getPointer() - currentOriginalRange.begin();
    return success();
  };

  bool allDiagsHandled = true;
  for (const auto &diag : diagnosticManager.Diagnostics()) {
    impl->typeSystem->debugLog("Diagnostic with fixits: {0}, message:\n{1}",
                               diag->HasFixIts(), diag->GetMessage());

    // If it's a mojo diagnostic, it might have fix-its. If it does, and if that
    // fails, return false. Otherwise, continue.
    if (const auto *mojoDiag = llvm::dyn_cast<MojoDiagnostic>(diag.get())) {
      for (const llvm::SMFixIt &fixit : mojoDiag->getFixIts()) {
        if (failed(applyFixit(fixit))) {
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
    impl->typeSystem->debugLog("Fixits applied to expression: \n{0}",
                               newText.c_str());
    diagnosticManager.SetFixedExpression(newText);
  } else {
    impl->typeSystem->debugLog("Unhandled diagnostics found!");
  }

  return success(allDiagsHandled);
}

M::LogicalResult
MojoExpressionParser::parse(DiagnosticManager &diagnosticManager) {
  if (!impl->compiler) {
    impl->typeSystem->errorLog("No compiler");
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
  StringRef moduleName = "__lldb_module__";
  auto buffer = llvm::MemoryBuffer::getMemBuffer(impl->expr.Text(), moduleName);
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
    impl->typeSystem->debugLog("Emitted diagnostics");
    return failure();
  }

  if (!module) {
    impl->typeSystem->errorLog("Failed to parse the module");
    return failure();
  }

  impl->typeSystem->debugLog("Parsed module successfully\n");

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

  // Extract the file module for the expression. File modules get mangled with
  // a leading `$`.
  auto fileModule =
      module->lookupSymbol<LIT::FileModuleOp>(("$" + moduleName).str());
  assert(fileModule && "expected to find the lldb file module");

  // Grab the struct containing the persistent expression state.
  auto persistentStateStruct =
      fileModule.lookupSymbol<LIT::StructDeclOp>("__lldb_context__");
  assert(persistentStateStruct && "expected to find persistent state struct");

  // Extract the internal expression function.
  auto functions = fileModule.getOps<LIT::FuncOp>();
  auto exprFnIt = llvm::find_if(functions, [](LIT::FuncOp func) {
    return func.getName().startswith("__lldb_expr_impl__(");
  });
  assert(exprFnIt != functions.end() && "expected lldb expression function");
  LIT::FuncOp exprFunc = *exprFnIt;

  // Process the variables within the expression.
  if (impl->options.GetREPLEnabled())
    processPersistentReplVariables(exprFunc, persistentStateStruct);

  // Run the elaboration pipeline.
  if (failed(impl->passManager->run(*module))) {
    impl->typeSystem->errorLog("Elaboration failed");
    return failure();
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
    return failure();
  }

  impl->typeSystem->dumpIR("Pre-optimization LLVM module:\n{0}",
                           *impl->llvmModule);

  // Create the target machine so we can run the optimizer.
  auto targetMachineOr =
      KGEN::createTargetMachine(impl->compilationOptions, /*isJIT=*/true);
  if (targetMachineOr.isError()) {
    impl->typeSystem->errorLog("Failed to create the target machine: {0}",
                               targetMachineOr.getError());
    return failure();
  }

  return KGEN::runLLVMOptPasses(*impl->llvmModule, **targetMachineOr);
}

Status MojoExpressionParser::prepareForExecution(
    lldb::addr_t &funcAddr, lldb::addr_t &funcEnd,
    std::shared_ptr<JITExecutionUnit> &executionUnit, ExecutionContext &exeCtx,
    ExecutionPolicy executionPolicy) {
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
  ConstString functionName(impl->expr.FunctionName());
  executionUnit = std::make_shared<JITExecutionUnit>(
      impl->llvmContext, module, functionName, exeCtx.GetTargetSP(), sc,
      features);

  // Extract the function information for the expression entry point.
  Status error = executionUnit->getRunnableInfo(funcAddr, funcEnd);
  if (error.Fail())
    return error;

  // Compute the target info to use for the persistent variable state.
  lldb_private::Process *process = exeCtx.GetProcessPtr();
  lldb::ByteOrder byteOrder = process->GetByteOrder();
  size_t addressByteSize = process->GetAddressByteSize();

  // If we successfully compiled the expression, we can now comfortably register
  // the persistent state variables.
  auto *persistentState = static_cast<MojoPersistentExpressionState *>(
      impl->typeSystem->GetPersistentExpressionState());

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
    peristentVariables.emplace_back(std::move(var));
  }

  // Register the persistent variables with the materializer.
  for (unsigned i = 0, e = persistentState->GetSize(); i < e; ++i) {
    lldb::ExpressionVariableSP var = persistentState->GetVariableAtIndex(i);
    assert(var && "expected valid variable in persistent state");

    // Try adding the variable to the expression materializer.
    impl->expr.GetMaterializer()->AddPersistentVariable(var, nullptr, error);
    if (error.Fail())
      return error;
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
                                              std::move(peristentVariables));
  return error;
}

//===----------------------------------------------------------------------===//
// Persistent Variables
//===----------------------------------------------------------------------===//

void MojoExpressionParser::processPersistentReplVariables(
    LIT::FuncOp func, LIT::StructDeclOp stateStruct) {
  OpBuilder structBuilder = OpBuilder::atBlockEnd(stateStruct.getBody());
  Value structValue = func.getArgument(0);
  Attribute targetAttr =
      KGEN::ParamOperatorAttr::get(POC::CurrentTarget, /*operands=*/{},
                                   structBuilder.getType<KGEN::TargetType>());

  // Utility functor to insert a new field into the persistent state struct.
  // Returns a value corresponding to the address of the field.
  auto insertField = [&](Operation *varOp, StringAttr name,
                         POP::PointerType type) {
    mlir::Type elementType = type.getResolvedElementType();
    impl->newPersistentVariables.emplace_back(name, elementType);

    structBuilder.create<LIT::StructFieldOp>(varOp->getLoc(), name,
                                             POP::PointerType::get(type));

    // Materialize a reference to the variable within the function.
    mlir::ImplicitLocOpBuilder builder(varOp->getLoc(), varOp);
    Value fieldGep = builder.create<LIT::StructGEPOp>(
        varOp->getLoc(), POP::PointerType::get(POP::PointerType::get(type)),
        name, structValue);
    Value fieldLoad = builder.create<POP::LoadOp>(varOp->getLoc(), fieldGep);

    // TODO: Whenever we have globals, we should be able to use a global
    // variable for the address and ensure it gets preserved. For now, we just
    // malloc the memory.
    mlir::Type indexType = structBuilder.getIndexType();
    Attribute sizeOfAttr = KGEN::ParamOperatorAttr::get(
        POC::GetSizeOf,
        {KGEN::ParameterizedTypeConstantAttr::get(elementType), targetAttr},
        indexType);
    Value sizeOf = builder.create<KGEN::ParamConstantOp>(indexType, sizeOfAttr);
    auto mallocCall = builder.create<POP::ExternalCallOp>(
        POP::PointerType::get(POP::SIMDType::get(
            1, builder.getAttr<KGEN::DTypeConstantAttr>(KGENDType::invalid))),
        "malloc", sizeOf, /*variadicType=*/TypeAttr());
    Value mallocResult = mallocCall.getResult(0);
    Value mallocCast =
        builder.create<POP::PointerBitcastOp>(type, mallocResult);
    builder.create<POP::StoreOp>(mallocCast, fieldLoad);

    // Return a pointer to the new address of the variable.
    return mallocCast;
  };

  // Functor used to consider if a variable should be persisted.
  auto shouldBePersisted = [&](StringRef name) {
    // Only consider variables that were written by users, not those
    // auto-generated by the compiler.
    return llvm::all_of(
        name, [&](char c) { return llvm::isAlnum(c) || c == '_' || c == '$'; });
  };

  // Walk the function body collecting all variables defined at the top scope.
  // In REPL mode, we persist any variables defined within the expression.
  for (Operation &op : llvm::make_early_inc_range(*func.getBody())) {
    // Handle register based let decls. These have an initializer, and never
    // expose the actual pointer.
    if (auto letOp = dyn_cast<LIT::LetRegDeclOp>(op)) {
      StringAttr name = letOp.getNameAttr();
      if (!shouldBePersisted(name.getValue()))
        continue;

      Value field =
          insertField(letOp, name, POP::PointerType::get(letOp.getType()));

      // Store the value in the persistent state struct.
      OpBuilder builder(letOp);
      builder.create<POP::StoreOp>(letOp.getLoc(), letOp.getValue(), field);

      // Replace all references of the original decl with the initializer.
      letOp.replaceAllUsesWith(letOp.getValue());
      letOp.erase();
      continue;
    }
    // Handle memory based let decls.
    if (auto letOp = dyn_cast<LIT::VarLetDeclOp>(op)) {
      StringAttr name = letOp.getNameAttr();
      if (!shouldBePersisted(name.getValue()))
        continue;

      Value field = insertField(letOp, name, letOp.getType());
      letOp.replaceAllUsesWith(field);
      letOp.erase();
      continue;
    }
  }
}
