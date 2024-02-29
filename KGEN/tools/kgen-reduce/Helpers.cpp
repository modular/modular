//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Helpers.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ToolOutputFile.h"

#include <chrono>

using namespace M;

uint64_t M::getCurTimeMs() {
  using namespace std::chrono;
  auto ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  return ms.count();
}

std::string M::getTempFileName() {
  return ("kgen-reduce." + Twine(getCurTimeMs())).str();
}

ErrorOr<std::unique_ptr<llvm::ToolOutputFile>>
M::getTempFile(ModuleOp module, const Twine &fileName) {
  std::string err;
  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile((fileName + ".mlirbc").str());
  if (!output)
    return Error(err);
  if (failed(mlir::writeBytecodeToFile(module, output->os())))
    return Error("failed to write bytecode");
  return std::move(output);
}

ErrorOrSuccess M::stashFile(ModuleOp module, const Twine &fileName) {
  auto err = getTempFile(module, fileName);
  if (err.isError())
    return err.takeError();
  err.takeValue()->keep();
  return success();
}

void M::unkeepToolOutputFile(llvm::ToolOutputFile &file) {
  // HACK: Access private members of the type to reset the flag.
  struct DirtyHack {
    struct CleanupInstaller {
      std::string filename;
      bool keep;
    } installer;

    std::optional<llvm::raw_fd_ostream> osHolder;
    llvm::raw_fd_ostream *os;
  };
  ((DirtyHack *)&file)->installer.keep = false;
}
