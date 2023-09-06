//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLDB.h"
#include "../Common/Telemetry.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLDB/MojoLLDB.h"
#include "Support/Driver/DriverSupport.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"
#include <filesystem>

using namespace M;

llvm::ErrorOr<std::string> M::getLLDB(const std::string &executable) {
  // Attempt to find an LLDB installed alongside the driver.
  std::string str = std::filesystem::path(executable).parent_path().string();
  return llvm::sys::findProgramByName("lldb",
                                      /*Paths=*/ArrayRef<StringRef>(str));
}

std::string M::getMojoLLDB(const std::string &executable) {
  // Attempt to find a MojoLLDB installed relative to the driver: if the driver
  // exists at "foo/bin/mojo", MojoLLDB should exist at "foo/bin/../lib/".
  std::filesystem::path lib =
      std::filesystem::path(executable).parent_path().parent_path() / "lib" /
      MOJO_LLDB;
  return lib.string();
}

int M::invokeLLDB(const State &state, llvm::opt::InputArgList &args,
                  std::initializer_list<StringRef> extraOptions) {
  // Initialize the LLCL runtime. We don't allow users to configure runtime
  // options, such as the allocator or the work queue threading model.
  LLCL::Runtime runtime(LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue());

  // Initialize telemetry.
  auto &telemetryCtx = runtime.emplaceContext<M::Telemetry::TelemetryContext>();
  initializeTelemetry(telemetryCtx, state, args);

  // Find the path to the LLDB executable and the MojoLLDB plugin library.
  std::string executable =
      llvm::sys::fs::getMainExecutable(state.programName, (void *)M::getLLDB);
  llvm::ErrorOr<std::string> lldb = M::getLLDB(executable);
  if (!lldb)
    return state.reportError("lldb must be installed alongside mojo");
  std::string mojoLLDB = M::getMojoLLDB(executable);

  // We forward all unparsed command line arguments to LLDB.
  SmallVector<StringRef> lldbArgs(state.arguments);
  std::string loadCommand = llvm::formatv("plugin load \"{0}\"", mojoLLDB);
  lldbArgs.insert(lldbArgs.begin(),
                  {lldb.get(), "-Q", "--one-line-before-file", loadCommand});
  lldbArgs.insert(lldbArgs.end(), extraOptions);
  return llvm::sys::ExecuteAndWait(lldb.get(), lldbArgs);
}
