//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef INIT_DEVELOPMENT_SIGNAL_HANDLER_H
#define INIT_DEVELOPMENT_SIGNAL_HANDLER_H

namespace llvm {
class StringRef;
}

namespace M::Init {

/// Register development signal handlers for crash reporting and debugging.
/// Only active in non-production builds. This function sets up comprehensive
/// signal handling for crash-like signals including SIGSEGV, SIGABRT, SIGFPE,
/// SIGILL, SIGBUS, SIGTRAP, and SIGSYS.
///
/// The signal handlers capture detailed signal information including signal
/// codes, fault addresses, and process information, then chain to LLVM's
/// signal handling infrastructure for stack traces and final cleanup.
void registerDevelopmentSignalHandler(llvm::StringRef programName);

/// Enable Python stack traces in signal handlers using async-safe faulthandler.
/// This configures the signal handler to use SIGUSR2 to trigger Python stack
/// traces without GIL deadlock risks.
void enablePythonStackTraceCallback();

} // namespace M::Init

#endif // INIT_DEVELOPMENT_SIGNAL_HANDLER_H
