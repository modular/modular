//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Facade in front of Crashpad for crash reporting.  This is not a part of the
// MSupport CMake target, instead you need to explicitly link against
// MCrashReporting.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CRASHREPORTING_H
#define SUPPORT_CRASHREPORTING_H

#include "Support/ForwardDecls.h"
#include <filesystem>

namespace M {

class Config;

/// Attempt to locate the Crashpad handler executable.
///
/// If specified in the configuration, that takes precedence.  Otherwise, we
/// look alongside the running executable, or failing that, anywhere on the
/// PATH.
ErrorOr<std::filesystem::path> getCrashpadHandlerPath(Config &config,
                                                      const char *argv0);

/// Pick a location to store crash data in.
///
/// Prefers a value from the "crash_reporting.database_path" configuration
/// option, but will fall back to a "crashdb" directory inside of the modular
/// home directory.
std::filesystem::path
getCrashDatabasePath(Config &config, const std::filesystem::path &modularHome);

/// Initialize crash reporting for currently running executable.
///
/// Note that this makes fairly invasive changes to the process environment
/// (removing existing signal handlers and adding new ones, spawning a
/// subprocess (potentially interfering with SIGCHLD handling), modifying the
/// process-global exception port on Darwin, etc) so it should only be called
/// from code that reasonably "owns" the process, not from a library where we
/// don't know what the rest of the code in the process is doing.
void initCrashpadForProgram(const char *argv0);

/// Generate a crash dump with the current state of the process, without
/// actually causing the current process to crash and terminate.
/// initCrashpadForProgram must have been previously called.
void generateNonFatalDump();

} // namespace M

#endif // SUPPORT_CRASHREPORTING_H
