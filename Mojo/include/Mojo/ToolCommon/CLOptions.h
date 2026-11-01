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

#ifndef KGEN_TOOLCOMMON_CLOPTIONS_H
#define KGEN_TOOLCOMMON_CLOPTIONS_H

#include "AsyncRT/Runtime/RuntimeCLOptions.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/ToolCommon/FrontendOptions.h"
#include "Support/CommonCLOptions.h"
#include "Support/ErrorOr.h"
#include "Support/Profiling/TimeProfiler.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include <filesystem>

namespace M {
namespace KGEN {
class CompiledFunc;
class ExecutionEngine;

//===----------------------------------------------------------------------===//
// CommandLineFunc
//===----------------------------------------------------------------------===//

/// This struct gives us a standard way to specify a func, its signature, and
/// its output filename on the command line. It also gives us a way to execute
/// this func. The format of this option is:
///
///  func ::= name `:` (signature)?
///  signature ::= return-type`(`arg-types...`)`
///
/// where name is just a string. Providing the signature is optional.
struct CommandLineFunc {

  std::string name;

  std::string signature;

  /// Parse a func from its command-line form.
  static CommandLineFunc parse(StringRef value);

  /// Verify that the signature of this func passed in on the command line
  /// matches the signature of the func as it exists in the IR.
  ErrorOrSuccess verifyFuncSignature(mlir::FunctionType funcType) const;
  /// Execute this func and print its result(s).
  ErrorOrSuccess executeAndPrint(CompiledFunc &compiledFunc) const;
};

//===----------------------------------------------------------------------===//
// Pass options
//===----------------------------------------------------------------------===//

class KGENPassCLOptions {
#ifndef MODULAR_PRODUCTION
#define DEFINE_CL_OPTION_GETTER(TYPE, VAR_NAME)                                \
  static std::optional<TYPE> VAR_NAME() {                                      \
    if (!passOptions->VAR_NAME.getNumOccurrences() &&                          \
        !passOptions->VAR_NAME.getDefault().hasValue())                        \
      return std::nullopt;                                                     \
    return passOptions->VAR_NAME;                                              \
  }

#define DEFINE_CL_OPTION_WITH_DEFAULT(TYPE, VAR_NAME, STR_NAME, DESC,          \
                                      DEFAULT_VALUE)                           \
  llvm::cl::opt<TYPE> VAR_NAME{STR_NAME, llvm::cl::desc(DESC),                 \
                               llvm::cl::init(DEFAULT_VALUE)};

#define DEFINE_CL_OPTION(TYPE, VAR_NAME, STR_NAME, DESC)                       \
  llvm::cl::opt<TYPE> VAR_NAME{STR_NAME, llvm::cl::desc((DESC))};

#else

#define DEFINE_CL_OPTION_GETTER(TYPE, VAR_NAME)                                \
  static std::optional<TYPE> VAR_NAME() { return passOptions->VAR_NAME; }

#define DEFINE_CL_OPTION(TYPE, VAR_NAME, STR_NAME, DESC)                       \
  std::optional<TYPE> VAR_NAME = std::nullopt;

#define DEFINE_CL_OPTION_WITH_DEFAULT(TYPE, VAR_NAME, STR_NAME, DESC,          \
                                      DEFAULT_VALUE)                           \
  TYPE VAR_NAME = DEFAULT_VALUE;
#endif // !MODULAR_PRODUCTION
public:
  /// Register all options
  static void registerOptions() { *passOptions; }

  DEFINE_CL_OPTION_GETTER(uint64_t, automaticInlineThreshold);
  DEFINE_CL_OPTION_GETTER(uint64_t, parametricInlineThreshold);
  DEFINE_CL_OPTION_GETTER(uint64_t, parametricInlineEstimatedLoopTripCount);
  DEFINE_CL_OPTION_GETTER(size_t, stackReusePromoteToGlobalThreshold);
  DEFINE_CL_OPTION_GETTER(size_t, kgenVerifierMaxErrors);

private:
  struct PassOptions {
    DEFINE_CL_OPTION(uint64_t, automaticInlineThreshold,
                     "kgen-automatic-inline-threshold",
                     "The threshold to automatically inline functions. "
                     "It has higher priority over pass's heuristic.")

    DEFINE_CL_OPTION(uint64_t, parametricInlineThreshold,
                     "kgen-parametric-inline-threshold",
                     "The threshold to inline parametric functions. "
                     "It has higher priority over pass's heuristic.")

    DEFINE_CL_OPTION_WITH_DEFAULT(uint64_t,
                                  parametricInlineEstimatedLoopTripCount,
                                  "kgen-parametric-inline-avg-loop-trip-count",
                                  "Estimated loop trip count that will be used "
                                  "in InlineParametric pass's heuristic.",
                                  4)

    DEFINE_CL_OPTION_WITH_DEFAULT(
        size_t, stackReusePromoteToGlobalThreshold,
        "kgen-stack-reuse-promote-to-global-threshold",
        "Threshold in byte above which a read-only stack "
        "allocations are promoted to global.",
        1024)

    DEFINE_CL_OPTION_WITH_DEFAULT(size_t, kgenVerifierMaxErrors,
                                  "kgen-verifier-max-errors",
                                  "Specify maximum number of errors "
                                  "KGENVerifier pass can emit at once.",
                                  10)
  };

  static llvm::ManagedStatic<PassOptions> passOptions;

#undef DEFINE_CL_OPTION
#undef DEFINE_CL_OPTION_WITH_DEFAULT
#undef DEFINE_CL_OPTION_GETTER
};

} // namespace KGEN
} // namespace M

#endif // KGEN_TOOLCOMMON_CLOPTIONS_H
