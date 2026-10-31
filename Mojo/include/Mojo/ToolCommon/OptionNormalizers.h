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
// This file declares the normalizers that the Mojo tools' shared compilation
// and target options name in their marshalling rows.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_OPTIONNORMALIZERS_H
#define KGEN_TOOLCOMMON_OPTIONNORMALIZERS_H

#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/Driver/OptionMarshalling.h"
#include "Support/ErrorOr.h"
#include "llvm/Option/ArgList.h"

#include <optional>

namespace M {

/// Check the constraints between compilation options that marshalling sets one
/// option at a time.
ErrorOrSuccess
validateCompilationOptions(const KGEN::CompilationOptions &options);

// Normalizers for the shared options whose values the generic integer and
// choice normalizers cannot parse. Each reports an invalid value to `errors`
// and leaves the option at its default.
std::optional<unsigned>
normalizeOptimizationLevel(llvm::opt::OptSpecifier opt, int,
                           const llvm::opt::ArgList &args,
                           MarshallingErrors &errors);
std::optional<KGEN::FpMode> normalizeFpMode(llvm::opt::OptSpecifier opt, int,
                                            const llvm::opt::ArgList &args,
                                            MarshallingErrors &errors);
std::optional<Sanitizers> normalizeSanitizer(llvm::opt::OptSpecifier opt, int,
                                             const llvm::opt::ArgList &args,
                                             MarshallingErrors &errors);

} // namespace M

#endif // KGEN_TOOLCOMMON_OPTIONNORMALIZERS_H
