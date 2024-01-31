//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT/Registration.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "Support/Process.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace M;

//===----------------------------------------------------------------------===//
// KGEN_CompilerRT_Python_SetPythonPath
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT const char *
KGEN_CompilerRT_Python_SetPythonPath() {
  // `PYTHONPATH` isn't always set, but when it is, respect whatever it's been
  // set to, rather than overwriting or appending to it.
  if (llvm::sys::Process::GetEnv("PYTHONPATH"))
    return "";

  // If `PYTHONPATH` hasn't been set, then try to grab the host `python3` (or,
  // failing that, its `python`). If we can't find either, then we can't do
  // anything, so bail.

  llvm::ErrorOr<std::string> pyOrErr = llvm::sys::findProgramByName("python3");
  if (!pyOrErr)
    pyOrErr = llvm::sys::findProgramByName("python");
  if (!pyOrErr)
    return "could not find 'python3' or 'python' executables";
  std::string python = *pyOrErr;

  // Create a temporary file to capture the output of the `python` invocation.
  std::error_code ec;
  std::filesystem::path tmpDirPath = std::filesystem::temp_directory_path(ec);
  if (ec)
    return "could not find temporary directory for 'python' output";
  ErrorOr<TempFile> outOrErr =
      TempFile::create((tmpDirPath / "python-out-%%%%%%.txt").string());
  if (failed(outOrErr))
    return "could not create temporary file to capture 'python' output";
  std::string out = outOrErr->getPath().string();

  // Invoke `python`, directing its output to the file.
  const std::optional<StringRef> redirects[] = {
      /*stdin=*/"",
      /*stdout=*/out,
      /*stderr=*/"",
  };
  // The Python program prints the list of global site-package directories,
  // joined by the platform-specific PATH separator.
  const StringRef args[] = {
      python, "-c",
      "import os; import site; print(os.pathsep.join(site.getsitepackages()))"};
  if (llvm::sys::ExecuteAndWait(python, args, /*Env=*/std::nullopt,
                                redirects) != 0)
    return "could not execute 'python' to determine site-package directory";

  // Read the output from the Python program.
  auto bufferOrErr = llvm::MemoryBuffer::getFile(out);
  if (!bufferOrErr)
    return "could not read temporary file with 'python' output";

  // Set the `PYTHONPATH` environment variable to the site-package paths.
  std::string pythonPath = bufferOrErr.get()->getBuffer().trim().str();
  if (failed(setProcessEnv("PYTHONPATH", pythonPath)))
    return "an error occurred when attempting to set 'PYTHONPATH'";

  return "";
}

//===----------------------------------------------------------------------===//
// CompilerRT Registration
//===----------------------------------------------------------------------===//

void KGEN::registerPython(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_Python_SetPythonPath",
                   (void *)&KGEN_CompilerRT_Python_SetPythonPath});
}
