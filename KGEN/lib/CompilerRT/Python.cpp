//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT/Registration.h"
#include "Support/Process.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace M;

using llvm::sys::findProgramByName;

// Works across ubuntu 20.04, 22.04, macos, pyenv, conda, venv, virtual
const char *FIND_LIBPYTHON = R"PROG(
import os
import sys
from pathlib import Path
from sysconfig import get_config_var
ext = "dll" if os.name == "nt" else "dylib" if sys.platform == "darwin" else "so"
pyver = get_config_var("py_version_short")
binary = f"libpython{pyver}.{ext}"
for libpython in [Path(get_config_var(p)) / binary for p in ["LIBPL", "LIBDIR"]]:
    if libpython.exists():
        print(libpython.resolve())
        exit(0)
exit(1)
)PROG";

//===----------------------------------------------------------------------===//
// KGEN_CompilerRT_Python_SetPythonPath
//===----------------------------------------------------------------------===//

// TODO: add a subprocess module to Mojo so this can all be done natively
// Updates `libpython` with the path to an associated
static bool findLibPython(const std::string &pythonBin,
                          std::string &libpython) {
  std::string cmd = pythonBin + " -c '" + FIND_LIBPYTHON + "'";
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                pclose);
  if (!pipe) {
    return false;
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  result.erase(std::remove_if(result.begin(), result.end(), ::isspace),
               result.end());

  libpython = result;
  return true;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT const char *
KGEN_CompilerRT_Python_SetPythonPath() {
  // Find the Python on top of `PATH` and put it in `PYTHONEXECUTABLE` to enable
  // multiprocessing, and finding the correct virtual environment site-modules.
  llvm::ErrorOr<std::string> pythonBin = findProgramByName("python3");
  if (!pythonBin)
    pythonBin = llvm::sys::findProgramByName("python");
  if (!pythonBin)
    return "could not find any 'python3' or 'python' executables on `PATH`";
  if (failed(setProcessEnv("PYTHONEXECUTABLE", *pythonBin)))
    return "cannot set `PYTHONEXECUTABLE` to";

  // If `MOJO_PYTHON_LIBRARY` is not set, run a Python script to find it.
  auto libpythonOpt = llvm::sys::Process::GetEnv("MOJO_PYTHON_LIBRARY");
  if (!libpythonOpt || libpythonOpt->empty()) {
    auto libpython = std::string();
    auto foundLibPython = findLibPython(*pythonBin, libpython);
    if (!foundLibPython || libpython.empty())
      return "no python installation found on system";
    if (failed(setProcessEnv("MOJO_PYTHON_LIBRARY", libpython)))
      return "cannot set `MOJO_PYTHON_LIBRARY`";
  }
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