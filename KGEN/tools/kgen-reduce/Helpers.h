//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"

#include <cstdint>
#include <string>

namespace llvm {
class ToolOutputFile;
} // namespace llvm

namespace M {
/// Get the current time in milliseconds.
uint64_t getCurTimeMs();

/// Get a filename for a snapshot file.
std::string getTempFileName();

/// Write the module to a temporary file that will by default be deleted on
/// exit.
ErrorOr<std::unique_ptr<llvm::ToolOutputFile>>
getTempFile(ModuleOp module, const Twine &fileName);

/// Store the module to a permanent file.
ErrorOrSuccess stashFile(ModuleOp module, const Twine &fileName);

/// Indicate that a file should be removed on exit from the process instead.
void unkeepToolOutputFile(llvm::ToolOutputFile &file);
} // namespace M
