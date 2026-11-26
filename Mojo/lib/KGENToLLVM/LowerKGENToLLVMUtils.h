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
// Helpers for LowerKGENToLLVM.cpp that do not depend on its patterns.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_KGENTOLLVM_LOWERKGENTOLLVMUTILS_H
#define KGEN_LIB_KGENTOLLVM_LOWERKGENTOLLVMUTILS_H

#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Attribute;
class NamedAttrList;
namespace LLVM {
class LLVMFuncOp;
} // namespace LLVM
} // namespace mlir

namespace M::KGEN {

/// Merges `llvm.denormal_fp_math` or `llvm.denormal_fp_math_f32` into the
/// function's `denormal_fpenv` attribute. LLVM reads the denormal mode only
/// from that attribute, and it ignores the `"denormal-fp-math"` strings of IR
/// it did not read from a file.
ErrorOrSuccess setDenormalFPEnv(mlir::LLVM::LLVMFuncOp func,
                                mlir::NamedAttrList &attrs,
                                llvm::StringRef suffix, mlir::Attribute value);

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_LOWERKGENTOLLVMUTILS_H
