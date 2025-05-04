//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CommonCLOptions.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include <filesystem>

using namespace M;

std::unique_ptr<llvm::ToolOutputFile>
CommonOptions::getOutputFile(bool hasBinaryOutput,
                             StringRef fileExtension) const {
  // We generally listen to the `-o filename` command, unless we're being
  // asked to emit a binary file format to the console.  In that case, we
  // default to emitting a variant of the input filename.
  std::string outFile = outputFilename;
  if (hasBinaryOutput && inputFilename != "-" && outFile.empty()) {
    outFile = inputFilename + fileExtension.str();
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

std::unique_ptr<llvm::ToolOutputFile>
CommonOptions::getIntermediateFile(StringRef inputName, StringRef ext) const {
  if (!saveTemps)
    return nullptr;
  std::string outFile = (inputName + ext).str();

  // If a directory has been provided, use it.
  if (!tempsDir.empty()) {
    // Unconditionally create the dir, this simply returns a bool if the
    // directory already exists.
    std::error_code errorCode;
    std::filesystem::create_directories(tempsDir, errorCode);
    outFile = (std::filesystem::path(tempsDir) /
               std::filesystem::path(inputName.str()).filename())
                  .replace_extension(ext.str())
                  .string();
  }

  std::error_code errorCode;
  auto absoluteOutputFile = std::filesystem::absolute(outFile, errorCode);
  if (errorCode) {
    exit(reportError("Cannot get absolute path for output file: '" + outFile +
                     "': " + errorCode.message()));
  }

  // Get a unique filename by adding numerical suffix if file exists
  std::filesystem::path uniquePath = absoluteOutputFile;
  std::string stem = uniquePath.stem().string();
  std::string extension = uniquePath.extension().string();
  int suffix = 1;

  // Only try up to 999 suffixes before falling back to overwriting
  while (std::filesystem::exists(uniquePath) && suffix <= 999) {
    uniquePath = uniquePath.parent_path() /
                 (stem + "_" + std::to_string(suffix) + extension);
    suffix++;
  }

  llvm::outs() << "Emitting intermediate file to '" << uniquePath.string()
               << "'.\n";

  std::string errorMessage;
  auto result = mlir::openOutputFile(uniquePath.string(), &errorMessage);
  if (!result)
    exit(reportError(errorMessage));
  return result;
}

LogicalResult CommonOptions::emitArchive(StringRef object) const {
  std::unique_ptr<llvm::ToolOutputFile> outFile =
      getOutputFile(/*hasBinaryOutput=*/true);
  if (!outFile)
    return failure(reportError("failed to open the output file"));

  outFile->os().write(object.begin(), object.size());
  outFile->keep();

  return mlir::success();
}
