//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CommonCLOptions.h"
#include "llvm/Support/ToolOutputFile.h"
#include <filesystem>

using namespace M;

std::unique_ptr<llvm::ToolOutputFile>
CommonCLOptions::getOutputFile(bool hasBinaryOutput) const {
  // We generally listen to the `-o filename` command, unless we're being
  // asked to emit a binary file format to the console.  In that case, we
  // default to emitting a variant of the input filename.
  std::string outFile = outputFilename.getValue();
  if (hasBinaryOutput && inputFilename != "-" &&
      outputFilename.getNumOccurrences() == 0) {
    outFile = inputFilename.getValue() + ".bef";
    llvm::outs() << "Emitting binary file to " << outFile << ".\n";
  }

  // Create the output directory if the directory doesn't exist and the output
  // file *is* a file.
  std::error_code ec;
  if (outFile != "-" && !std::filesystem::exists(outFile, ec) && !ec) {
    auto outFilePath = std::filesystem::path(outFile);
    if (outFilePath.has_parent_path())
      std::filesystem::create_directories(outFilePath.parent_path(), ec);
  }
  // If anything failed, report the failure.
  if (ec)
    exit(reportError("std::filesystem: " + ec.message() + ": " + outFile));

  std::error_code error;
  auto result = std::make_unique<llvm::ToolOutputFile>(outFile, error,
                                                       llvm::sys::fs::OF_None);
  if (error)
    exit(reportError("Cannot open output file: '" + outFile +
                     "': " + error.message()));

  return result;
}
