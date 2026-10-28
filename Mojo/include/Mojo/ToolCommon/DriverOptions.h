//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file declares the options that the Mojo tools act on themselves, rather
// than pass to the parser or the compiler.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_DRIVEROPTIONS_H
#define KGEN_TOOLCOMMON_DRIVEROPTIONS_H

#include <string>
#include <vector>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace M {
struct State;

namespace KGEN {
struct FrontendOptions;
} // namespace KGEN

/// The options that a Mojo tool acts on itself, which its marshalled options
/// fill in. A tool leaves the options that its option table lacks at their
/// defaults.
struct DriverOptions {
  /// The directories to search for imported files, from `-I`.
  std::vector<std::string> includePaths;
  /// The names that `-D` defines, as `NAME` or `NAME=VALUE`.
  std::vector<std::string> defines;

  /// Whether to apply the parser's fix-its and rerun the command.
  bool applyFixIts = false;
  /// The YAML file to export the parser's fix-its to, if not empty.
  std::string exportFixItsPath;

  // The target queries, each of which prints information and exits.
  bool printEffectiveTarget = false;
  bool printSupportedTargets = false;
  bool printSupportedCpus = false;
  bool printSupportedAccelerators = false;

  /// The `lld` to link with, overriding the configured one, if not empty.
  std::string lldPath;

  // The pass timing reports.
  bool mlirTiming = false;
  std::string mlirTimingDisplay;
  bool llvmTiming = false;
  bool timingJSON = false;
  std::string timingFile;
};

/// Sets the include paths of `driverOptions` as the include directories of
/// `sourceMgr`, which reports its diagnostics in the format that `state`
/// parsed.
void configureSourceMgr(const State &state, const DriverOptions &driverOptions,
                        llvm::SourceMgr &sourceMgr);

/// Copies the warning policy that `state` parsed into `frontendOptions`.
void setWarningPolicy(const State &state,
                      KGEN::FrontendOptions &frontendOptions);

} // namespace M

#endif // KGEN_TOOLCOMMON_DRIVEROPTIONS_H
