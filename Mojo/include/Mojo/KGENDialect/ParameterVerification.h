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
// This file defines the default of the parameter verification that
// `--run-parameter-verifier` overrides.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_PARAMETERVERIFICATION_H
#define KGEN_PARAMETERVERIFICATION_H

#include "Support/SymbolExport.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace M::KGEN {

/// Whether the parameter verifier runs its checks when nothing asks
/// otherwise. A production build skips them, because they find compiler bugs
/// and not user errors.
#ifdef MODULAR_PRODUCTION
inline constexpr bool kVerifyParametersByDefault = false;
#else
inline constexpr bool kVerifyParametersByDefault = true;
#endif

/// Returns whether a parameter use-def graph built in `ctx` runs the checks of
/// the parameter verifier: the setting of the context's KGEN dialect, or
/// `kVerifyParametersByDefault` when the context has not loaded the dialect.
MODULAR_CXX_EXPORT bool runParamVerifier(mlir::MLIRContext *ctx);

} // namespace M::KGEN

#endif // KGEN_PARAMETERVERIFICATION_H
