//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Tool that computes the hash of an MLIR file. This is useful for debugging and
// testing determinism issues.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllKGENDialects(registry);

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  ctx.appendDialectRegistry(registry);

  OwningOpRef<ModuleOp> module = mlir::parseSourceFile<ModuleOp>(
      inputFilename.getValue(), mlir::ParserConfig(&ctx));
  if (!module)
    return -1;

  std::string bytecode;
  llvm::raw_string_ostream bytecodeOs(bytecode);
  if (failed(mlir::writeBytecodeToFile(*module, bytecodeOs)))
    return {};
  auto hash =
      llvm::BLAKE3::hash({(const uint8_t *)bytecode.data(), bytecode.size()});

  std::string message;
  auto outFile = mlir::openOutputFile(outputFilename.getValue(), &message);
  if (!outFile) {
    llvm::errs() << "failed to open output: " << message << "\n";
    return -1;
  }
  outFile->os() << llvm::toHex({(const char *)hash.data(), hash.size()})
                << "\n";
  outFile->keep();

  return 0;
}
