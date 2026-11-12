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
// Rewrites abi("C") LLVM function definitions to use the platform C ABI
// (CABIInfo). Separated from LowerKGENToLLVM.cpp so that reviewers can read
// the definition-side logic in isolation from the call-site patterns.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_KGENTOLLVM_LOWERKGENTOLLVMREWRITECABIFNS_H
#define KGEN_LIB_KGENTOLLVM_LOWERKGENTOLLVMREWRITECABIFNS_H

#include "CABILowering.h"
#include "mlir/IR/Value.h"

namespace mlir::LLVM {
class LLVMFuncOp;
class ReturnOp;
} // namespace mlir::LLVM

namespace M::KGEN {

/// The C ABI form of the result of an abi("C") function definition. Its
/// returns use this to give the result in that form.
struct CABIDefinition {
  CoercionInfo ret;
  /// The sret pointer, or the memory through which a register result is
  /// coerced. Null if the result needs no coercion.
  mlir::Value retSlot;
  /// The type of a register result. Null for sret.
  mlir::Type coercedRetType;
};

/// Give an abi("C") LLVM function its C ABI signature and attributes. In a
/// definition, also make the C ABI entry block arguments and rebuild the
/// Mojo arguments from them at the top of the entry block. Mojo direct
/// callers are patched by ConvertKGENCall. `isResultSigned` selects signext
/// over zeroext for a narrow integer result.
CABIDefinition rewriteCABIFunctionEntry(mlir::LLVM::LLVMFuncOp func,
                                        CABIInfo &abiInfo, bool isResultSigned);

/// Make `ret` give its result in the C ABI form that `def` describes.
void rewriteCABIReturn(mlir::LLVM::ReturnOp ret, const CABIDefinition &def);

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_LOWERKGENTOLLVMREWRITECABIFNS_H
