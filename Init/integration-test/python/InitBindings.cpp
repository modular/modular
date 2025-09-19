//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Init/DevelopmentSignalHandler.h"
#include "Init/Init.h"
#include "SDK/GraphAPI/python/TypeCasters.h"
#include "SDK/Support/PythonBacktrace.h"
#include "nanobind/nanobind.h"
#include <csignal>
#include <sstream>
#include <stdexcept>

namespace nb = nanobind;

void trigger_segfault_from_cpp() {
  // This will be called from Python, so we'll have Python stack frames
  // available when the signal handler runs
  std::raise(SIGSEGV);
}

void initialize_signal_handler(const std::string &program_name) {
  // Initialize the signal handler using the Init module
  auto contextOrError = M::Init::createContext(program_name);
  if (contextOrError.isError()) {
    // Convert Error to string using stringstream
    std::ostringstream oss;
    oss << contextOrError.takeError();
    std::string errorMsg = "Failed to create context: " + oss.str();
    throw std::runtime_error(errorMsg);
  }

  // Register Python stack trace callback
  M::Init::registerPythonStackTraceCallback(M::printPythonBacktrace);

  // No need to store the context for this test - it persists globally
}

NB_MODULE(init_bindings, m) {
  m.doc() = "Init module bindings for testing signal handler with Python stack "
            "traces";

  m.def("trigger_segfault_from_cpp", &trigger_segfault_from_cpp,
        "Trigger a segfault from C++ to test signal handler");

  m.def(
      "initialize_signal_handler", &initialize_signal_handler,
      "Initialize the development signal handler with the given program name");
}
