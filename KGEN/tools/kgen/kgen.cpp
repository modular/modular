//===- kgen.cpp -----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CLOptions.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "Support/CommonCLOptions.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>

using namespace M;
using namespace mlir;

namespace {
class CLOptions : public CommonCLOptions {
public:
  using CommonCLOptions::CommonCLOptions;

  cl::opt<bool> ignoreFailures{
      "ignore-failure",
      cl::desc("Ignore execution failures. Any messages are still printed, but "
               "failures don't mean the tool fails to execute.")};

  cl::list<std::string> searchPaths{
      "I", cl::desc("Path to use to search for included files.")};

  cl::list<ExecutableKernel, bool, ExecutableKernelParser> exec{
      "execute", cl::desc("Specifies the kernels to execute. Defaults to an "
                          "empty list, which will not execute any kernel.")};

  cl::list<EmittableKernel, bool, EmittableKernelParser> emit{
      "emit",
      cl::desc("Specifies the kernels to emit. Defaults to an empty list, "
               "which will emit a file for each kernel in the input file.")};

  Optional<EmittableKernel>
  shouldEmitKernel(mlir::LLVM::LLVMFuncOp kernel) const {
    if (emit.empty())
      return EmittableKernel{kernel.getName().str(),
                             (kernel.getName() + ".o").str()};

    auto found = llvm::find_if(emit, [&](const EmittableKernel &ek) {
      return ek.name == kernel.getName();
    });
    if (found == emit.end())
      return None;
    return *found;
  }

  Optional<ExecutableKernel>
  shouldExecuteKernel(mlir::LLVM::LLVMFuncOp kernel) const {
    auto found = llvm::find_if(exec, [&](const ExecutableKernel &ek) {
      return ek.name == kernel.getName();
    });

    if (found == exec.end())
      return None;
    return *found;
  }
};
} // namespace

/// This function creates the elaborator pass and forwards the correct
/// arguments. If it fails, it fails with a fatal error.
static std::unique_ptr<Pass> createElaboratorPass(const CLOptions &clOptions) {
  auto elaborate = KGEN::createElaborateKernelsPass();
  std::string includes;
  llvm::raw_string_ostream includeStr(includes);
  for (StringRef include : clOptions.searchPaths)
    includeStr << "search-path=" << include << " ";

  if (failed(elaborate->initializeOptions(includeStr.str())))
    llvm::report_fatal_error("unable to initialize elaborator options");

  return elaborate;
}

namespace {
/// This provides a method by which we can emit a kernel's signature to an
/// llvm::formatv stream.
struct FormatKernel : public llvm::FormatAdapter<mlir::LLVM::LLVMFuncOp> {
  FormatKernel(mlir::LLVM::LLVMFuncOp func)
      : llvm::FormatAdapter<mlir::LLVM::LLVMFuncOp>(std::move(func)) {}

  void format(llvm::raw_ostream &os, StringRef style) override {
    // Construct a map of all structs defined in the kernel signature.
    llvm::MapVector<mlir::LLVM::LLVMStructType, std::string> structs;
    for (auto &t : llvm::enumerate(Item.getFunctionType().getParams())) {
      if (auto st = t.value().dyn_cast<mlir::LLVM::LLVMStructType>()) {
        std::string structName =
            st.isIdentified()
                ? st.getName().str()
                : ("__kgen_" + Item.getName() + "_struct_" + Twine(t.index()))
                      .str();
        structs.insert({st, std::move(structName)});
      }
    }

    // Insert the return type in the struct list if it's a struct.
    if (auto st = Item.getFunctionType()
                      .getReturnType()
                      .dyn_cast<mlir::LLVM::LLVMStructType>()) {
      std::string structName =
          st.isIdentified()
              ? st.getName().str()
              : ("__kgen_" + Item.getName() + "_struct_result").str();
      structs.insert({st, std::move(structName)});
    }

    // Helper to print a function as a C type.
    std::function<void(Type)> printTypeAsC = [&](Type t) {
      // If it's a pointer, recurse and add a '*'.
      if (auto pt = t.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
        printTypeAsC(pt.getElementType());
        os << "*";
        return;
      }

      // If it's a struct, refer to it by name. We've forced a name.
      if (auto st = t.dyn_cast<mlir::LLVM::LLVMStructType>())
        llvm::report_fatal_error("structs are supported through linearization");

      // Fixed vector types are easy.
      if (auto vt = t.dyn_cast<mlir::LLVM::LLVMFixedVectorType>()) {
        printTypeAsC(vt.getElementType());
        os << "__attribute__ ((vector_size(" << vt.getNumElements() << ")))";
        return;
      }

      // Scalable vector types are not, just pass in a pointer and trust that
      // we'll know what to do inside the kernel.
      if (auto vt = t.dyn_cast<mlir::LLVM::LLVMScalableVectorType>()) {
        printTypeAsC(vt.getElementType());
        os << "*";
        return;
      }

      assert(llvm::isPowerOf2_64(t.getIntOrFloatBitWidth()) &&
             "bitwidth must be a power of 2");

      // Elementary type, just print it.
      if (t.isa<IntegerType>())
        os << "int" << t.getIntOrFloatBitWidth() << "_t";
      else if (t.isa<IndexType>())
        os << "intptr_t";
      else if (t.isF16())
        llvm::report_fatal_error("no support for fp16 yet");
      else if (t.isF32())
        os << "float";
      else if (t.isF64())
        os << "double";
      else
        llvm::report_fatal_error("unknown type");
    };

    // First, iterate the structs and print them out.
    for (const auto &t : structs) {
      os << "struct " << t.second << " {\n";
      for (const auto &member : llvm::enumerate(t.first.getBody())) {
        printTypeAsC(member.value());
        os << " __kgen_member_" << member.index() << ";\n";
      }
      os << "};\n\n";
    }

    // Now print the function declaration.
    os << "extern ";
    printTypeAsC(Item.getFunctionType().getReturnType());
    os << " " << Item.getName() << "(";
    llvm::interleaveComma(Item.getFunctionType().getParams(), os, printTypeAsC);
    os << ");";
  }
};
} // namespace

/// This allows us to emit a header file for the given kernel so that we can
/// `#include` it and get nice autocompletion/etc. in users' IDEs.
static LogicalResult emitHeaderForKernel(StringRef filename,
                                         mlir::LLVM::LLVMFuncOp kernel) {
  std::string err;
  auto outFile = mlir::openOutputFile(filename, &err);
  if (!outFile)
    return mlir::emitError(kernel.getLoc(), err);

  llvm::StringLiteral fmtStr = R"literal(//===-{0}-===//
//
// This file is Modular Inc proprietary.
//
//==={1}===//
// THIS FILE IS AUTOGENERATED BY `kgen`, DO NOT EDIT!

#ifndef {2}
#define {2}

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

{3}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // {2}
)literal";

  outFile->os() << llvm::formatv(
      fmtStr.data(),
      llvm::fmt_align(" " + kernel.getName() + ".h ", llvm::AlignStyle::Left,
                      80 - 2 * strlen("//===-"), '-'),
      llvm::fmt_repeat('-', 80 - 2 * strlen("//===")),
      "__KGEN_" + kernel.getName().upper() + "_H", FormatKernel(kernel));

  outFile->keep();

  return mlir::success();
}

/// Runs the tool pipeline on the file fragment passed in. The pipeline does not
/// output to the specific ostream provided to it, rather it opens and writes to
/// files that are designated by the kernels it operates on.
static LogicalResult runToolPipeline(MLIRContext *ctx, llvm::SourceMgr &mgr,
                                     const CLOptions &clOptions) {
  DialectRegistry registry;

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registry.insert<mlir::arith::ArithmeticDialect, mlir::LLVM::LLVMDialect,
                  mlir::scf::SCFDialect>();

  mlir::registerLLVMDialectTranslation(registry);

  // Set up the dialects in the context.
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
  // Allow unregistered dialects, we will verify we know what to do with it
  // later.
  ctx->allowUnregisteredDialects();

  OwningOpRef<ModuleOp> theModule = parseSourceFile<ModuleOp>(mgr, ctx);
  if (!theModule)
    return failure(clOptions.reportError("could not parse the module"));

  // Set up the pass pipeline.
  mlir::PassManager pm(ctx);
  pm.addPass(KGEN::createLowerHLKGENPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Elaborate and canonicalize.
  pm.addPass(createElaboratorPass(clOptions));
  pm.addPass(mlir::createCanonicalizerPass());

  // Convert to LLVM.
  OpPassManager &kpm = pm.nest<KGEN::KernelOp>();
  kpm.addPass(KGEN::createConvertPOPToLLVMPass());

  pm.addPass(mlir::arith::createConvertArithmeticToLLVMPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::cf::createConvertControlFlowToLLVMPass());
  pm.addPass(KGEN::createConvertKGENToLLVMPass());

  // And finally canonicalize again before running through the JIT.
  pm.addPass(mlir::createCanonicalizerPass());

  // Now create the execution engine so we can JIT.
  auto engineOr = KGEN::ExecutionEngine::create();
  if (failed(engineOr))
    return failure(clOptions.reportError(engineOr.getError()));

  KGEN::ExecutionEngine engine = std::move(*engineOr);

  // Helper to emit the object for a kernel.
  auto emitObjectForKernel = [&](mlir::LLVM::LLVMFuncOp k,
                                 const Twine &filename) -> LogicalResult {
    // If the filename is not provided, then default to the current working
    // directory.
    std::filesystem::path objPath = filename.str();
    if (!objPath.is_absolute())
      objPath = std::filesystem::current_path() / filename.str();

    // Open the output file so we can emit to it.
    std::string err;
    auto outFile = mlir::openOutputFile(objPath.string(), &err);
    if (!outFile)
      return mlir::emitError(k.getLoc(), err);

    auto objOr = engine.getObject(k);
    if (failed(objOr))
      return mlir::emitError(k.getLoc(),
                             "could not get the object for the kernel '@" +
                                 k.getName() + "': " + objOr.getError());

    std::unique_ptr<llvm::MemoryBuffer> obj = std::move(*objOr);
    outFile->os().write(obj->getBufferStart(), obj->getBufferSize());
    outFile->keep();

    // Get a file path `.h` next to the object we're emitting. This will allow
    // us to emit a header for the kernel.
    std::filesystem::path headerPath = objPath;
    headerPath.replace_extension(".h");
    if (failed(emitHeaderForKernel(headerPath.string(), k)))
      return failure();

    return mlir::success();
  };

  // Run the pass manager. This will ensure that the module has been fully
  // lowered to LLVM.
  if (failed(pm.run(*theModule)))
    return failure(clOptions.reportError("compilation failed"));

  // Loop over the kernels and (1) add them to the engine and (2) maybe emit the
  // kernel as an object file.
  for (auto k : theModule->getOps<mlir::LLVM::LLVMFuncOp>()) {
    // First add the kernel to the engine.
    if (ErrorOrSuccess err = engine.add(k))
      return mlir::emitError(k.getLoc(), err.getError());

    // If we were asked to emit this kernel, do so.
    if (Optional<EmittableKernel> emittableKernel =
            clOptions.shouldEmitKernel(k))
      if (failed(emitObjectForKernel(k, emittableKernel->outputFilename)))
        return failure();
  }

  // Now, if we were asked to execute any kernels, do so.
  for (const auto &exec : clOptions.exec) {
    auto k = theModule->lookupSymbol<mlir::LLVM::LLVMFuncOp>(exec.name);
    if (!k) {
      mlir::emitError(theModule->getLoc())
          << "could not find kernel '@" << exec.name << "'";
      if (!clOptions.ignoreFailures)
        return failure();
      continue;
    }

    if (auto err = exec.verifyKernelSignature(k.getFunctionType())) {
      mlir::emitError(k.getLoc(), err.getError());
      if (!clOptions.ignoreFailures)
        return failure();
      continue;
    }

    if (auto err = exec.executeAndPrint(engine)) {
      mlir::emitError(k.getLoc(), err.getError());
      if (!clOptions.ignoreFailures)
        return failure();
    }
  }

  return mlir::success();
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);

  // Enable command line options for various MLIR internals.
  registerAsmPrinterCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file.
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      clOptions.openInputFileOrExit();

  return failed(clOptions.configureMLIRContextAndSourceMgrAndExecute(
      std::move(inputFile),
      [&](MLIRContext *ctx, llvm::SourceMgr &mgr) -> LogicalResult {
        return runToolPipeline(ctx, mgr, clOptions);
      }));
}
